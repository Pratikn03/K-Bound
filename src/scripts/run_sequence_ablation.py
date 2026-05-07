"""CLI runner for the sequence model ablation study.

Generates synthetic behavior-like data when no dataset path is provided,
making the ablation runnable out-of-the-box for quick sanity checks.

Usage (with real data):
    python src/scripts/run_sequence_ablation.py \\
        --data data/processed/behavior_sequences.npz \\
        --output reports/sequence_ablation.csv \\
        --hidden-dims 32 64 128 \\
        --epochs 30

Usage (smoke test with synthetic data):
    python src/scripts/run_sequence_ablation.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uais.sequence.ablation import AblationConfig, run_sequence_ablation, summarise_ablation
from uais.sequence.build_sequences import pad_sequences
from uais.utils.logging_utils import setup_logging

logger = setup_logging("run_sequence_ablation")


def _make_synthetic(
    n_samples: int = 500,
    seq_len: int = 50,
    n_features: int = 8,
    anomaly_rate: float = 0.1,
    seed: int = 42,
):
    """Generate simple synthetic sequences for smoke-testing."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_samples) < anomaly_rate).astype(int)
    sequences = rng.standard_normal((n_samples, seq_len, n_features)).astype(np.float32)
    # Anomalies have slightly higher variance to make the task non-trivial
    sequences[labels == 1] *= 1.5
    mask = np.ones((n_samples, seq_len), dtype=np.float32)
    # Simulate variable-length sequences: randomly zero-pad the last 20% of rows
    for i in rng.choice(n_samples, size=n_samples // 5, replace=False):
        cut = rng.integers(seq_len // 2, seq_len)
        mask[i, cut:] = 0
        sequences[i, cut:] = 0
    return sequences, mask, labels


def _load_npz(path: Path):
    data = np.load(path)
    sequences = data["sequences"].astype(np.float32)
    labels = data["labels"].astype(int)
    mask = data["mask"].astype(np.float32) if "mask" in data else np.ones(
        (len(labels), sequences.shape[1]), dtype=np.float32
    )
    return sequences, mask, labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sequence model ablation study")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=Path, help="Path to .npz file with sequences/mask/labels")
    group.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke test")

    parser.add_argument("--output", type=Path, default=Path("reports/sequence_ablation.csv"))
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "transformer", "tcn"])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--num-layers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.synthetic:
        logger.info("Using synthetic data for ablation smoke test")
        sequences, mask, labels = _make_synthetic(seed=args.seed)
    else:
        logger.info("Loading data from %s", args.data)
        sequences, mask, labels = _load_npz(args.data)

    logger.info(
        "Dataset: %d sequences, shape %s, anomaly rate=%.2f%%",
        len(labels), sequences.shape, 100 * labels.mean(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = AblationConfig(
        model_types=args.models,
        hidden_dims=args.hidden_dims,
        num_layers_options=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        output_csv=str(args.output),
    )

    results = run_sequence_ablation(sequences, mask, labels, cfg=cfg)

    print("\n=== Ablation Results (sorted by AUROC) ===")
    display_cols = [
        "model_type", "hidden_dim", "num_layers",
        "roc_auc", "pr_auc", "f1", "precision", "recall",
        "epochs_run", "train_time_s",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    print(results[display_cols].to_string(index=False))

    print("\n=== Per-Architecture Summary (mean ± std) ===")
    print(summarise_ablation(results).to_string())

    print(f"\nFull results saved to: {args.output}")
    return results


if __name__ == "__main__":
    main()
