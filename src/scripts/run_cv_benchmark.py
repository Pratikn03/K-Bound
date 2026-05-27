"""Unified 5-fold cross-validation benchmark for ALL baseline families.

Addresses the reviewer's full methodology checklist:
  - 5-fold StratifiedKFold (split BEFORE any preprocessing)
  - Identical dimensionality reduction applied across all baselines
  - Unsupervised models trained on normal-only training samples
  - All metrics reported: ROC-AUC, PR-AUC, F1, Precision, Recall, MCC,
    balanced_accuracy, confusion matrix, Brier, ECE, TPR@FPR=1%
  - Bootstrap 95% CIs for AUC + F1 on pooled-fold predictions
  - Pairwise DeLong tests between every pair of baselines
  - Leakage detection: train↔test contamination + test-set duplicate scan
  - Suspicious-metric flagging (AUC ≥ 0.99 etc.)
  - Full hyperparameter registry per baseline saved alongside results

Usage
-----
# Synthetic smoke run
python src/scripts/run_cv_benchmark.py --synthetic \\
    --reducer none --output experiments/cv_smoke.json

# Real fusion CSV
python src/scripts/run_cv_benchmark.py \\
    --config configs/attention_real_fusion.yaml \\
    --reducer pca --reducer-n-components 16 \\
    --n-splits 5 --bootstrap-resamples 1000 \\
    --output experiments/real_fusion/cv_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running directly without an installed package
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml  # noqa: E402

from uais.fusion.attention.baselines import (  # noqa: E402
    ConfidenceWeightedMean,
    EarlyFusionMLP,
    LateFusionEnsemble,
    RandomForestFusion,
)
from uais.fusion.attention.cv_evaluator import (  # noqa: E402
    BaselineSpec,
    CVConfig,
    cross_validate_baselines,
)
from uais.fusion.attention.unsupervised_baselines import (  # noqa: E402
    AEConfig,
    AutoencoderAnomalyDetector,
    BGMMAnomalyDetector,
    BGMMConfig,
    GMMAnomalyDetector,
    GMMConfig,
    IForestConfig,
    IsolationForestDetector,
    KMeansAnomalyDetector,
    KMeansConfig,
    LOFAnomalyDetector,
    LOFConfig,
    OCSVMConfig,
    OneClassSVMDetector,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _make_synthetic(
    n_samples: int = 600,
    n_domains: int = 3,
    n_features: int = 5,
    pos_rate: float = 0.25,
    missing: float = 0.1,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < pos_rate).astype(int)
    features = rng.random((n_samples, n_domains, n_features)).astype(np.float32)
    for d in range(n_domains):
        features[labels == 1, d, 0] = np.clip(features[labels == 1, d, 0] + 0.4, 0, 1)
    masks = rng.random((n_samples, n_domains)) < missing
    return features, masks.astype(bool), labels


def _load_real(cfg: dict):
    """Load fusion CSV via the canonical pipeline."""
    from uais.fusion.attention.attention_utils import (
        build_fusion_tensors,
        infer_feature_columns,
        load_fusion_dataframe,
    )

    dcfg = cfg["data"]
    df = load_fusion_dataframe(dcfg["path"])
    feat_cols = infer_feature_columns(
        df,
        score_column=dcfg.get("score_column", "score"),
        confidence_column=dcfg.get("confidence_column"),
        embedding_prefix=dcfg.get("embedding_prefix", "embedding_"),
        feature_columns=dcfg.get("feature_columns", []),
        include_confidence=dcfg.get("confidence_column") is not None,
    )
    features, masks, labels, sample_ids, domain_order = build_fusion_tensors(
        df=df,
        id_column=dcfg["id_column"],
        domain_column=dcfg["domain_column"],
        label_column=dcfg["label_column"],
        feature_columns=feat_cols,
        domain_order=dcfg.get("domain_order"),
    )
    return features, masks, labels


# ---------------------------------------------------------------------------
# Baseline specs (all baselines under the same protocol)
# ---------------------------------------------------------------------------


def build_specs(seed: int) -> list[BaselineSpec]:
    """Every spec uses the same random_state and identical DR upstream."""
    import torch

    device = torch.device("cpu")

    return [
        # Supervised multimodal models (operate on [N, D, F] + masks)
        BaselineSpec(
            name="early_fusion_mlp",
            make=lambda: EarlyFusionMLP(device=device, random_seed=seed),
            needs_3d=True,
            is_unsupervised=False,
        ),
        BaselineSpec(
            name="late_fusion_ensemble",
            make=lambda: LateFusionEnsemble(random_seed=seed),
            needs_3d=True,
            is_unsupervised=False,
        ),
        BaselineSpec(
            name="random_forest",
            make=lambda: RandomForestFusion(random_seed=seed),
            needs_3d=True,
            is_unsupervised=False,
        ),
        BaselineSpec(
            name="confidence_weighted_mean",
            make=lambda: ConfidenceWeightedMean(),
            needs_3d=True,
            is_unsupervised=False,
        ),
        # Unsupervised anomaly detectors (operate on [N, D, F] + masks too;
        # they filter labels==0 internally).
        BaselineSpec(
            name="bgmm",
            make=lambda: BGMMAnomalyDetector(BGMMConfig(random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="gmm",
            make=lambda: GMMAnomalyDetector(GMMConfig(random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="kmeans",
            make=lambda: KMeansAnomalyDetector(KMeansConfig(random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="isolation_forest",
            make=lambda: IsolationForestDetector(IForestConfig(random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="one_class_svm",
            make=lambda: OneClassSVMDetector(OCSVMConfig()),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="lof",
            make=lambda: LOFAnomalyDetector(LOFConfig()),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="autoencoder",
            make=lambda: AutoencoderAnomalyDetector(AEConfig(random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _to_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run unified 5-fold CV benchmark across all baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config (data section used to locate fusion CSV)")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no --config required)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--reducer", choices=["none", "pca", "autoencoder"], default="none")
    p.add_argument("--reducer-n-components", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("experiments/cv_benchmark.json"))
    p.add_argument("--leakage-warn-auc", type=float, default=0.99)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load data
    if args.synthetic or args.config is None:
        print("[info] Using synthetic data.")
        features, masks, labels = _make_synthetic(seed=args.seed)
    else:
        cfg = yaml.safe_load(args.config.read_text())
        features, masks, labels = _load_real(cfg)
        print(f"[info] Loaded real data: {features.shape}, " f"positive rate = {labels.mean():.3f}")

    # Build reducer kwargs
    if args.reducer == "pca":
        reducer_kwargs = {"n_components": args.reducer_n_components}
    elif args.reducer == "autoencoder":
        reducer_kwargs = {"latent_dim": args.reducer_n_components}
    else:
        reducer_kwargs = {}

    cv_cfg = CVConfig(
        n_splits=args.n_splits,
        random_state=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        reducer_name=args.reducer,
        reducer_kwargs=reducer_kwargs,
        leakage_warn_auc=args.leakage_warn_auc,
    )

    specs = build_specs(args.seed)
    print(f"[info] Running {len(specs)} baselines × {cv_cfg.n_splits} folds " f"with reducer={cv_cfg.reducer_name}")

    results = cross_validate_baselines(features, masks, labels, specs, cv_cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(_to_serializable(results), fh, indent=2)

    # Print a summary table
    print("\n=== Aggregate ROC-AUC across folds (mean ± std) ===")
    rows = []
    for name, data in results["per_baseline"].items():
        agg = data.get("aggregate", {}).get("roc_auc")
        if agg is None:
            rows.append((name, "N/A", "—"))
            continue
        rows.append(
            (
                name,
                f"{agg['mean']:.4f} ± {agg['std']:.4f}",
                f"[{agg['lo']:.4f}, {agg['hi']:.4f}]",
            )
        )
    for r in rows:
        print(f"  {r[0]:>28s}  {r[1]:>22s}  95% CI = {r[2]}")

    if results.get("pairwise_delong_pvalues"):
        print("\n=== DeLong pairwise p-values (sample) ===")
        for k, v in list(results["pairwise_delong_pvalues"].items())[:5]:
            print(f"  {k}: p = {v:.4f}")

    print(f"\n[info] Full results saved to {args.output}")
    contam = results.get("contamination_check", {})
    if contam.get("max_duplicates", 0) > 0:
        print(
            f"[WARN] Contamination detected: max {contam['max_duplicates']} "
            f"duplicate rows between train and test across folds!"
        )


if __name__ == "__main__":
    main()
