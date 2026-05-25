"""Phase 2.2B.2 / Step 5 — add the clean k=0 arm to the B-MECH-1 archive.

Re-runs ELARA-Bench-LA × G0 mean-gate × τ=0.66 over the same 30 seeds
(42–71) as B-MECH-1, but with **no corruption injection** (k=0). This
gives the paired clean-fold predictions that are required to compute
the formal risk-dominance terms (q₀, q₁, Δ₀, Δ₁, π*) in B-CERT-1.

Output: experiments/phase2/mechanism/b_mech_1_prediction_archives/<existing cell dir>/
        {static_attention__clean_k0, rga_mean_gate_tau66__clean_k0}/test/seed_NN.parquet
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_breakthrough_experiment import (  # noqa: E402
    _build_model, _load_data, _make_loaders, _make_reliability_estimator,
    _predict_static, _predict_craf_with_stats, _split, _train_model, set_seed,
)
from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402

ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"
ARCHIVE_ROOT = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"
EID = "B-MECH-1"
BENCHMARK = "ELARA-Bench-LA"
PROTOCOL = "k-of-D corruption k=4 mean-gate at locked tau=0.66 (B1+B2 endpoints)"  # MUST match B-MECH-1 cell dir
TAU_MEAN = 0.66


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_one_seed(cfg, seed, archive):
    device = _device()
    set_seed(int(seed))
    cfg_seed = dict(cfg)
    cfg_seed["training"] = dict(cfg.get("training", {}))
    cfg_seed["training"]["seed"] = int(seed)

    (features, masks, labels, sample_ids, domain_order, _, conf_idx,
     score_idx, sample_splits, _) = _load_data(cfg_seed)
    train_idx, val_idx, test_idx = _split(labels, cfg_seed["training"],
                                          split_values=sample_splits)
    train_loader, val_loader, _ = _make_loaders(
        features, masks, labels, train_idx, val_idx, test_idx,
        batch_size=int(cfg_seed["training"].get("batch_size", 64)),
    )
    model = _build_model(cfg_seed, features.shape[1], features.shape[2], conf_idx, device)
    _train_model(model, train_loader, val_loader, cfg_seed, device)
    model.eval()

    rel_cfg = cfg_seed.get("reliability", {})
    estimator = _make_reliability_estimator(rel_cfg, list(domain_order) or ["d0", "d1"], score_idx)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])
    estimator.gate_mode = "mean"
    estimator.gate_threshold = TAU_MEAN

    test_feat = features[test_idx]   # uncorrupted = clean k=0
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]
    test_sids = [str(sample_ids[i]) for i in test_idx]

    static_probs = _predict_static(model, test_feat, test_mask, device)
    craf_probs, gate_stats = _predict_craf_with_stats(
        model, estimator, test_feat, test_mask, device,
        clean_gate_threshold=TAU_MEAN, per_sample_gating=False,
    )

    for method, scores in (("static_attention", static_probs),
                            ("rga_mean_gate_tau66", craf_probs)):
        frame = archive.build_frame(
            sample_ids=test_sids,
            labels=np.asarray(test_labels, dtype=int),
            raw_scores=np.asarray(scores, dtype=float),
            method=method,
            method_variant="clean__k0",
            benchmark=BENCHMARK, protocol=PROTOCOL,
            analysis_family="B", pairing_strength="label_aligned_stress_only",
            split="test", seed=int(seed),
            selection_rule="validation-only: tau=0.66 LOCKED; clean arm; no corruption",
            selection_used_test_metrics=False,
            selected_head_or_comparator_status=(
                "RGA G0 mean-gate clean arm" if method == "rga_mean_gate_tau66"
                else "static reference clean arm"
            ),
            gate_mode="mean",
            gate_fired=np.zeros(len(test_sids), dtype=bool),  # k=0 clean: no firing
            mean_reliability=np.full(len(test_sids), float(gate_stats.get("mean_reliability", 0.0))),
            min_reliability=np.full(len(test_sids), float(gate_stats.get("min_reliability", 0.0))),
            failure_type="none",
            failed_domain_count=0,
            fault_severity=0.0,
        )
        entry = archive.write(
            experiment_id=EID, benchmark=BENCHMARK, protocol=PROTOCOL,
            seed=int(seed), method=f"{method}__clean_k0",
            split="test", frame=frame, config=cfg_seed,
        )
        archive.append_index(entry)
    return {
        "seed": int(seed),
        "static_test_auc": _safe_auc(test_labels, static_probs),
        "rga_test_auc": _safe_auc(test_labels, craf_probs),
        "mean_reliability": float(gate_stats.get("mean_reliability", 0.0)),
    }


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=42)
    args = p.parse_args()
    cfg = yaml.safe_load(ELARA_BENCH_LA_CONFIG.read_text())
    archive = PredictionArchive(root=ARCHIVE_ROOT)
    for s in range(int(args.seed_start), int(args.seed_start) + int(args.seeds)):
        print(f"[b-mech-1 clean-arm seed={s} starting]", flush=True)
        r = run_one_seed(cfg, s, archive)
        print(f"[b-mech-1 clean-arm seed={s} done; "
              f"static_clean_auc={r['static_test_auc']:.4f}  "
              f"rga_clean_auc={r['rga_test_auc']:.4f}]", flush=True)
    print(f"[b-mech-1 clean-arm] {args.seeds} seeds complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
