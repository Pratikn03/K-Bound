"""ISSUE 2 benchmark: batch-level vs genuine per-sample reliability gating.

This is a forward / opt-in benchmark. It does NOT touch any locked Phase-2
artifact. It runs the end-to-end reliability-gated fusion experiment on the
synthetic smoke dataset twice -- once with the legacy batch-level estimator
(``estimator_type: batch``) and once with the per-sample estimator
(``estimator_type: per_sample`` + ``per_sample_gating: true``) -- and reports:

  * the within-batch dispersion of per-sample reliability (the construct-validity
    signal: batch-level collapses to ~0, per-sample is > 0);
  * clean RGA vs static mean ROC AUC and their delta for each mode.

Usage::

    PYTHONPATH=.:src python src/scripts/run_per_sample_gating_benchmark.py \
        --output output/issue2/per_sample_gating_benchmark.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.scripts.run_breakthrough_experiment import (
    _make_synthetic,
    _run_experiment_arrays,
)
from uais.fusion.attention.reliability_estimator import (
    PerSampleReliabilityEstimator,
    ReliabilityEstimator,
)


def _base_cfg() -> dict:
    return {
        "data": {
            "path": "/dev/null",
            "score_column": "score",
            "confidence_column": "confidence",
            "embedding_prefix": "embedding_",
            "id_column": "sample_id",
            "domain_column": "domain",
            "label_column": "label",
        },
        "model": {
            "num_domains": 3,
            "embed_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.1,
            "use_confidence": False,
            "use_input_confidence": False,
            "use_attention": True,
            "use_domain_embeddings": True,
            "use_positional_embeddings": True,
            "use_missing_embedding": True,
        },
        "training": {
            "seed": 42,
            "batch_size": 64,
            "epochs": 5,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "domain_dropout": 0.1,
            "test_size": 0.2,
            "val_size": 0.1,
            "early_stopping": 3,
            "lambda_reg": 0.01,
            "restore_best_weights": True,
        },
        "evaluation": {
            "seeds": [42, 43, 44],
            "cda_samples": 20,
            "n_bootstrap": 50,
            "domain_dropout_probs_extended": [0.0, 0.1, 0.3],
        },
        "reliability": {
            "ece_weight": 0.4,
            "ks_weight": 0.4,
            "sharpness_weight": 0.2,
            "n_calibration_bins": 5,
            "min_samples_for_ks": 10,
            "gate_threshold": 0.66,
        },
        "craf": {
            "drift_noise_levels": [0.0, 0.1, 0.3],
            "adversarial_attacks": ["zero_attack", "gaussian_noise"],
            "adversarial_sigma": 0.1,
        },
    }


def _reliability_dispersion(features, masks, labels, per_sample: bool) -> dict:
    """Fit on first half, measure per-sample reliability dispersion on second."""
    n = features.shape[0]
    cut = n // 2
    kwargs = dict(
        domain_order=[f"domain_{i}" for i in range(features.shape[1])],
        score_index=0,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    )
    cls = PerSampleReliabilityEstimator if per_sample else ReliabilityEstimator
    est = cls(**kwargs)
    est.fit(features[:cut], masks[:cut], labels[:cut])
    w = est.compute_reliability_weights(features[cut:], masks[cut:])
    present = ~masks[cut:]
    counts = present.sum(axis=1)
    r = np.where(present, w, 0.0).sum(axis=1) / np.maximum(counts, 1)
    return {
        "within_batch_reliability_std": float(np.std(r)),
        "mean_reliability": float(np.mean(r)),
    }


def _run_mode(per_sample: bool, features, masks, labels, sample_ids, domain_order) -> dict:
    cfg = _base_cfg()
    if per_sample:
        cfg["reliability"]["estimator_type"] = "per_sample"
        cfg["reliability"]["per_sample_gating"] = True
    else:
        cfg["reliability"]["estimator_type"] = "batch"
        cfg["reliability"]["per_sample_gating"] = False

    results = _run_experiment_arrays(
        copy.deepcopy(cfg),
        features,
        masks,
        labels,
        sample_ids,
        domain_order,
        confidence_index=None,
        score_index=0,
        device=torch.device("cpu"),
    )
    stat = results.get("statistical_summary", {})
    static_aucs = [a for a in stat.get("per_seed_static_auc", []) if a is not None]
    craf_aucs = [a for a in stat.get("per_seed_craf_auc", []) if a is not None]
    static_mean = float(np.mean(static_aucs)) if static_aucs else float("nan")
    craf_mean = float(np.mean(craf_aucs)) if craf_aucs else float("nan")
    dispersion = _reliability_dispersion(features, masks, labels, per_sample=per_sample)
    return {
        "mode": "per_sample" if per_sample else "batch",
        "static_auc_mean": static_mean,
        "rga_auc_mean": craf_mean,
        "delta_auc_mean": craf_mean - static_mean,
        "per_seed_static_auc": static_aucs,
        "per_seed_rga_auc": craf_aucs,
        "paired_ttest_p_rga_vs_static": stat.get("paired_ttest_p_craf_vs_static"),
        **dispersion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ISSUE 2 per-sample gating benchmark")
    parser.add_argument(
        "--output",
        default="output/issue2/per_sample_gating_benchmark.json",
        help="Where to write the JSON comparison.",
    )
    parser.add_argument("--n-samples", type=int, default=800)
    args = parser.parse_args()

    features, masks, labels, sample_ids, domain_order = _make_synthetic(n_samples=args.n_samples)

    batch = _run_mode(False, features, masks, labels, sample_ids, domain_order)
    per_sample = _run_mode(True, features, masks, labels, sample_ids, domain_order)

    report = {
        "benchmark": "issue2_batch_vs_per_sample_gating",
        "dataset": "synthetic_smoke",
        "note": "Forward/opt-in benchmark; not paper evidence and does not modify locked artifacts.",
        "batch": batch,
        "per_sample": per_sample,
        "delta_auc_difference_per_sample_minus_batch": per_sample["delta_auc_mean"] - batch["delta_auc_mean"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    def _fmt(x):
        return "nan" if x != x else f"{x:.4f}"

    print("\n=== ISSUE 2: batch-level vs per-sample reliability gating ===")
    print(f"{'metric':<34}{'batch':>12}{'per_sample':>14}")
    for key, label in [
        ("within_batch_reliability_std", "within-batch reliability std"),
        ("static_auc_mean", "static AUC (mean)"),
        ("rga_auc_mean", "RGA AUC (mean)"),
        ("delta_auc_mean", "delta AUC (RGA - static)"),
    ]:
        print(f"{label:<34}{_fmt(batch[key]):>12}{_fmt(per_sample[key]):>14}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
