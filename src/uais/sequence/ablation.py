"""Sequence model ablation study.

Systematically compares LSTM, GRU, Transformer, and TCN across architectural
hyperparameters (hidden_dim, num_layers) on the same data split, so all
results are directly comparable.

Typical usage
-------------
from uais.sequence.ablation import AblationConfig, run_sequence_ablation
results_df = run_sequence_ablation(sequences, mask, labels)
print(results_df.sort_values("roc_auc", ascending=False).to_string())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from uais.utils.logging_utils import setup_logging

from .train_gru import GRUConfig, train_gru_classifier
from .train_lstm import LSTMConfig, train_lstm_classifier
from .transformer_tcn import SequenceModelConfig, train_sequence_model

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Defines the grid of hyperparameters to sweep."""

    # Architectures to compare
    model_types: list[str] = field(default_factory=lambda: ["lstm", "gru", "transformer", "tcn"])
    # Hidden dimensions to sweep — tests capacity scaling
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128])
    # Number of layers to sweep (for LSTM/GRU/Transformer)
    num_layers_options: list[int] = field(default_factory=lambda: [1, 2])

    # Training hyperparameters (fixed across all ablation runs for fair comparison)
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip: float = 1.0
    dropout: float = 0.1
    val_size: float = 0.15
    seed: int = 42

    # If True, save the results DataFrame to this path
    output_csv: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_one(
    model_type: str,
    hidden_dim: int,
    num_layers: int,
    sequences: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    cfg: AblationConfig,
) -> dict[str, object]:
    """Train a single model variant and return its metrics + timing."""
    t0 = time.time()
    logger.info(
        "Ablation: model=%s  hidden_dim=%d  num_layers=%d",
        model_type,
        hidden_dim,
        num_layers,
    )

    if model_type == "lstm":
        lstm_cfg = LSTMConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=cfg.dropout,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
        _, metrics = train_lstm_classifier(sequences, mask, labels, cfg=lstm_cfg)

    elif model_type == "gru":
        gru_cfg = GRUConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=cfg.dropout,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
        _, metrics = train_gru_classifier(sequences, mask, labels, cfg=gru_cfg)

    elif model_type in ("transformer", "tcn"):
        seq_cfg = SequenceModelConfig(
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=cfg.dropout,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
        _, metrics = train_sequence_model(sequences, labels, cfg=seq_cfg, mask=mask)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    elapsed = time.time() - t0
    history = metrics.pop("history", {})
    epochs_run = len(history.get("train_loss", []))

    return {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "roc_auc": metrics.get("roc_auc", float("nan")),
        "pr_auc": metrics.get("pr_auc", float("nan")),
        "f1": metrics.get("f1", float("nan")),
        "precision": metrics.get("precision", float("nan")),
        "recall": metrics.get("recall", float("nan")),
        "accuracy": metrics.get("accuracy", float("nan")),
        "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
        "brier": metrics.get("brier", float("nan")),
        "tpr_at_1pct_fpr": metrics.get("tpr_at_fpr", float("nan")),
        "epochs_run": epochs_run,
        "train_time_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sequence_ablation(
    sequences: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    cfg: AblationConfig | None = None,
) -> pd.DataFrame:
    """Run the full ablation grid and return results as a sorted DataFrame.

    Each row is one (model_type, hidden_dim, num_layers) combination.
    Rows are sorted by val AUROC descending so the best configuration is first.

    Parameters
    ----------
    sequences : (N, T, F) padded sequence array
    mask      : (N, T) binary mask — 1 for real timesteps, 0 for padding
    labels    : (N,) binary labels
    cfg       : AblationConfig controlling the sweep grid and training params
    """
    if cfg is None:
        cfg = AblationConfig()

    rows: list[dict] = []
    total = len(cfg.model_types) * len(cfg.hidden_dims) * len(cfg.num_layers_options)
    run = 0

    for model_type in cfg.model_types:
        for hidden_dim in cfg.hidden_dims:
            for num_layers in cfg.num_layers_options:
                run += 1
                logger.info("[%d/%d] Starting ablation run", run, total)
                try:
                    row = _train_one(
                        model_type,
                        hidden_dim,
                        num_layers,
                        sequences,
                        mask,
                        labels,
                        cfg,
                    )
                except Exception as exc:
                    logger.warning(
                        "Ablation run failed (model=%s, hidden=%d, layers=%d): %s",
                        model_type,
                        hidden_dim,
                        num_layers,
                        exc,
                    )
                    row = {
                        "model_type": model_type,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "roc_auc": float("nan"),
                        "error": str(exc),
                    }
                rows.append(row)

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    if cfg.output_csv:
        df.to_csv(cfg.output_csv, index=False)
        logger.info("Ablation results saved to %s", cfg.output_csv)

    return df


def summarise_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ablation results — mean ± std per model type for the report."""
    metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "balanced_accuracy"]
    available = [c for c in metric_cols if c in df.columns]
    summary = df.groupby("model_type")[available].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.sort_values("roc_auc_mean", ascending=False)


__all__ = ["AblationConfig", "run_sequence_ablation", "summarise_ablation"]
