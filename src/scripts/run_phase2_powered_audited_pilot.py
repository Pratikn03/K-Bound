"""Phase 2.C pilot — powered audited reproduction of A-POWERED-1 with archiving.

Drives a focused 30-seed rerun on the MVTec 3D-AD PatchCore
supervised-paired cell. Uses the existing runner's helpers
(_build_model, _train_model, _predict_static, _predict_craf, etc.)
but skips the adversarial / tau-sweep / k-of-D blocks; the pilot's
single job is to produce the **raw per-seed test prediction archive**
for the validation-frozen RGA+ head AND the validation-frozen
primary comparator.

Outputs:
  experiments/phase2/predictions/A-POWERED-1__.../<method>/<split>/seed_NN.parquet
  experiments/phase2/statistics/family_a_powered_seed_metrics.csv  (appended)
  experiments/phase2/statistics/family_a_selection_log.csv          (appended)

Validation-only selection:
  - RGA+ head per seed: argmax validation ROC-AUC over {router, boost}
    (tie-break boost). Selection logged before test metrics are read.
  - Primary comparator: validation-frozen — argmax seed-mean validation
    ROC-AUC over candidate pool, computed AT THE END across all archived
    val predictions (so comparator is per-cell, not per-seed).

Compute: ~2–4 min/seed on M-series MPS for this 3 226-sample cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the existing runner helpers we need.
from scripts.run_breakthrough_experiment import (  # noqa: E402
    _build_model,
    _load_data,
    _make_loaders,
    _make_reliability_estimator,
    _predict_static,
    _predict_craf,
    _split,
    _train_model,
    set_seed,
)
from uais.fusion.attention.baselines import run_baseline_suite  # noqa: E402
from uais.fusion.attention.meta_router import fit_rga_meta_router  # noqa: E402
from uais.fusion.attention.reliability_boosted_fusion import ReliabilityBoostedFusion  # noqa: E402

from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402


CANDIDATE_BASELINES = (
    "random_forest",
    "early_fusion_mlp",
    "late_fusion_ensemble",
    "confidence_weighted_mean",
    "tent_score_adapter",
    "ttt_pseudo_label_adapter",
    "eata_score_adapter",
    "sar_score_adapter",
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def run_one_seed(
    cfg: dict,
    seed: int,
    *,
    archive: PredictionArchive,
    experiment_id: str,
    benchmark: str,
    protocol: str,
    pairing_strength: str,
    cell_dir_slug: str,
) -> dict[str, Any]:
    """Train + evaluate one seed, archive predictions, return per-seed metrics."""
    device = _device()
    set_seed(int(seed))

    # Override the seed in the cfg copy before split.
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    (features, masks, labels, sample_ids, _domain_order, _, conf_idx,
     score_idx, sample_splits, _sample_cats) = _load_data(cfg_seed)

    train_idx, val_idx, test_idx = _split(
        labels,
        cfg_seed["training"],
        split_values=sample_splits,
    )
    train_loader, val_loader, _ = _make_loaders(
        features, masks, labels, train_idx, val_idx, test_idx,
        batch_size=int(cfg_seed["training"].get("batch_size", 64)),
    )
    model = _build_model(cfg_seed, features.shape[1], features.shape[2], conf_idx, device)
    _train_model(model, train_loader, val_loader, cfg_seed, device)
    model.eval()

    val_feat = features[val_idx]
    val_mask = masks[val_idx]
    val_labels = labels[val_idx]
    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]
    test_sids = [str(sample_ids[i]) for i in test_idx]
    val_sids = [str(sample_ids[i]) for i in val_idx]

    # Reliability estimator + train predictions for RGA path.
    rel_cfg = cfg_seed.get("reliability", {})
    estimator = _make_reliability_estimator(rel_cfg, list(_domain_order or []) or ["d0", "d1"], score_idx)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])

    # Static + RGA predictions (test + val)
    static_val_probs = _predict_static(model, val_feat, val_mask, device)
    static_probs = _predict_static(model, test_feat, test_mask, device)
    craf_val_probs = _predict_craf(model, estimator, val_feat, val_mask, device)
    craf_probs = _predict_craf(model, estimator, test_feat, test_mask, device)

    # RGA+ boosted fusion
    rga_boosted = ReliabilityBoostedFusion(
        score_index=score_idx,
        confidence_index=conf_idx,
        random_seed=int(seed),
        selection_metric=str(cfg_seed.get("rga_plus_selection_metric", "roc_auc")),
    ).fit(
        features[train_idx], masks[train_idx], labels[train_idx],
        val_feat, val_mask, val_labels,
        reliability_estimator=estimator,
    )
    boost_val_probs = rga_boosted.predict_proba(val_feat, val_mask)
    boost_probs = rga_boosted.predict_proba(test_feat, test_mask)

    # Baseline predictions (val + test)
    baseline_metrics, baseline_predictions = run_baseline_suite(
        features, masks, labels, train_idx, val_idx, test_idx,
        score_index=score_idx, device=device, random_seed=int(seed),
        decision_threshold_strategy=str(cfg_seed.get("clean_decision_threshold_strategy", "val_f1")),
        return_predictions=True,
    )

    # Meta-router selection over (RGA, every baseline) — validation-only.
    router_val_predictions = {
        "static_attention": static_val_probs,
        "craf_attention": craf_val_probs,
        "rga_boosted_fusion": boost_val_probs,
        **{name: pred["val_probs"] for name, pred in baseline_predictions.items()},
    }
    router_test_predictions = {
        "static_attention": static_probs,
        "craf_attention": craf_probs,
        "rga_boosted_fusion": boost_probs,
        **{name: pred["test_probs"] for name, pred in baseline_predictions.items()},
    }
    router = fit_rga_meta_router(
        val_predictions=router_val_predictions,
        val_labels=val_labels,
        random_seed=int(seed),
    )
    router_val_probs = router.predict_proba(router_val_predictions)
    router_test_probs = router.predict_proba(router_test_predictions)

    # Validation-frozen RGA+ head selection (router vs boost; tie-break boost).
    val_auc_router = _safe_auc(val_labels, router_val_probs)
    val_auc_boost = _safe_auc(val_labels, boost_val_probs)
    if abs(val_auc_router - val_auc_boost) < 1e-12 or val_auc_boost >= val_auc_router:
        chosen_head = "boost"
        chosen_val_probs = boost_val_probs
        chosen_test_probs = boost_probs
    else:
        chosen_head = "router"
        chosen_val_probs = router_val_probs
        chosen_test_probs = router_test_probs

    # Archive per-method per-split predictions.
    def _archive(method: str, scores_val: np.ndarray, scores_test: np.ndarray,
                 selected_status: str):
        for split_name, sids, lab, scores in [
            ("validation", val_sids, val_labels, scores_val),
            ("test", test_sids, test_labels, scores_test),
        ]:
            frame = archive.build_frame(
                sample_ids=sids,
                labels=np.asarray(lab, dtype=int),
                raw_scores=np.asarray(scores, dtype=float),
                method=method,
                method_variant=None,
                benchmark=benchmark,
                protocol=protocol,
                analysis_family="A",
                pairing_strength=pairing_strength,
                split=split_name,
                seed=int(seed),
                selection_rule="validation-only RGA+ head + validation-only primary comparator (Phase 2.B contract)",
                selection_used_test_metrics=False,
                selected_head_or_comparator_status=selected_status,
            )
            entry = archive.write(
                experiment_id=experiment_id,
                benchmark=benchmark,
                protocol=protocol,
                seed=int(seed),
                method=method,
                split=split_name,
                frame=frame,
                config=cfg_seed,
            )
            archive.append_index(entry)

    _archive("rga_meta_router", router_val_probs, router_test_probs, "candidate RGA+ head")
    _archive("rga_boosted_fusion", boost_val_probs, boost_probs, "candidate RGA+ head")
    _archive("static_attention", static_val_probs, static_probs, "static reference")
    _archive("craf_attention", craf_val_probs, craf_probs, "RGA reference")
    for name in CANDIDATE_BASELINES:
        if name not in baseline_predictions:
            continue
        _archive(
            name,
            baseline_predictions[name]["val_probs"],
            baseline_predictions[name]["test_probs"],
            "comparator candidate",
        )

    # Per-seed descriptive metrics.
    test_auc_router = _safe_auc(test_labels, router_test_probs)
    test_auc_boost = _safe_auc(test_labels, boost_probs)
    test_auc_static = _safe_auc(test_labels, static_probs)
    test_auc_craf = _safe_auc(test_labels, craf_probs)
    chosen_test_auc = test_auc_router if chosen_head == "router" else test_auc_boost
    return {
        "seed": int(seed),
        "n_test_samples": int(len(test_labels)),
        "val_auc_router": val_auc_router,
        "val_auc_boost": val_auc_boost,
        "chosen_head": chosen_head,
        "chosen_val_auc": (val_auc_router if chosen_head == "router" else val_auc_boost),
        "chosen_test_auc": chosen_test_auc,
        "static_test_auc": test_auc_static,
        "craf_test_auc": test_auc_craf,
        "router_test_auc": test_auc_router,
        "boost_test_auc": test_auc_boost,
        "baseline_test_aucs": {
            name: _safe_auc(test_labels, baseline_predictions[name]["test_probs"])
            for name in CANDIDATE_BASELINES if name in baseline_predictions
        },
        "baseline_val_aucs": {
            name: _safe_auc(val_labels, baseline_predictions[name]["val_probs"])
            for name in CANDIDATE_BASELINES if name in baseline_predictions
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/attention_mvtec3d_patchcore_supervised_paired.yaml"),
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("experiments/phase2/predictions"),
    )
    parser.add_argument(
        "--seed-metrics-out",
        type=Path,
        default=Path("experiments/phase2/statistics/family_a_powered_seed_metrics.csv"),
    )
    parser.add_argument(
        "--selection-log-out",
        type=Path,
        default=Path("experiments/phase2/statistics/family_a_selection_log.csv"),
    )
    parser.add_argument("--experiment-id", default="A-POWERED-1")
    parser.add_argument("--benchmark", default="MVTec 3D-AD")
    parser.add_argument("--protocol", default="PatchCore supervised-paired")
    parser.add_argument("--pairing-strength", default="independent_modalities")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    archive = PredictionArchive(root=args.archive_root)
    args.seed_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.selection_log_out.parent.mkdir(parents=True, exist_ok=True)

    # Force the configured seed pool to a contiguous range starting at seed_start.
    seeds_planned = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))

    # Append-mode for resumability.
    metrics_fields = [
        "experiment_id", "benchmark", "protocol", "seed", "n_test_samples",
        "val_auc_router", "val_auc_boost", "chosen_head",
        "chosen_val_auc", "chosen_test_auc",
        "static_test_auc", "craf_test_auc",
        "router_test_auc", "boost_test_auc",
    ]
    selection_fields = [
        "experiment_id", "benchmark", "protocol", "seed",
        "candidate", "val_auc", "test_auc", "selection_used_test_metrics",
    ]
    metrics_new = not args.seed_metrics_out.exists()
    selection_new = not args.selection_log_out.exists()
    metrics_f = args.seed_metrics_out.open("a", newline="")
    selection_f = args.selection_log_out.open("a", newline="")
    metrics_w = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
    selection_w = csv.DictWriter(selection_f, fieldnames=selection_fields)
    if metrics_new:
        metrics_w.writeheader()
    if selection_new:
        selection_w.writeheader()

    for s in seeds_planned:
        print(f"[pilot] seed={s} starting", flush=True)
        result = run_one_seed(
            cfg, s,
            archive=archive,
            experiment_id=args.experiment_id,
            benchmark=args.benchmark,
            protocol=args.protocol,
            pairing_strength=args.pairing_strength,
            cell_dir_slug="",
        )
        metrics_w.writerow({
            "experiment_id": args.experiment_id,
            "benchmark": args.benchmark,
            "protocol": args.protocol,
            "seed": result["seed"],
            "n_test_samples": result["n_test_samples"],
            "val_auc_router": result["val_auc_router"],
            "val_auc_boost": result["val_auc_boost"],
            "chosen_head": result["chosen_head"],
            "chosen_val_auc": result["chosen_val_auc"],
            "chosen_test_auc": result["chosen_test_auc"],
            "static_test_auc": result["static_test_auc"],
            "craf_test_auc": result["craf_test_auc"],
            "router_test_auc": result["router_test_auc"],
            "boost_test_auc": result["boost_test_auc"],
        })
        metrics_f.flush()
        # Selection log: for every candidate baseline + the two RGA+ heads.
        common = {"experiment_id": args.experiment_id, "benchmark": args.benchmark,
                  "protocol": args.protocol, "seed": result["seed"],
                  "selection_used_test_metrics": False}
        selection_w.writerow({**common, "candidate": "rga_meta_router",
                              "val_auc": result["val_auc_router"],
                              "test_auc": result["router_test_auc"]})
        selection_w.writerow({**common, "candidate": "rga_boosted_fusion",
                              "val_auc": result["val_auc_boost"],
                              "test_auc": result["boost_test_auc"]})
        for name, vauc in result["baseline_val_aucs"].items():
            selection_w.writerow({**common, "candidate": name,
                                  "val_auc": vauc,
                                  "test_auc": result["baseline_test_aucs"].get(name)})
        selection_f.flush()
        print(f"[pilot] seed={s} done  chosen={result['chosen_head']}  "
              f"chosen_test_auc={result['chosen_test_auc']:.4f}", flush=True)

    metrics_f.close()
    selection_f.close()
    print(f"[pilot] {len(seeds_planned)} seeds complete. Archive index updated.")


if __name__ == "__main__":
    main()
