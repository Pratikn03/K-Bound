"""Transformer and TCN sequence classifiers for behavior anomaly detection.

Fixes vs. the original:
- TransformerClassifier passes src_key_padding_mask so pad tokens are
  excluded from attention, and mean-pools only over real timesteps.
- TCNBlock gains a residual projection and dropout for regularisation.
- train_sequence_model now takes a val split, does early stopping with
  gradient clipping, and returns a full metrics dict instead of just loss.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from uais.utils.logging_utils import setup_logging
from uais.utils.metrics import compute_classification_metrics

logger = setup_logging(__name__)


@dataclass
class SequenceModelConfig:
    model_type: str = "transformer"  # "transformer" | "tcn"
    hidden_dim: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip: float = 1.0
    val_size: float = 0.15
    seed: int = 42


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if mask is not None:
            self.mask = torch.tensor(mask, dtype=torch.float32)
        else:
            self.mask = torch.ones(len(labels), sequences.shape[1])

    def __len__(self) -> int:  # pragma: no cover
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.mask[idx], self.labels[idx]


class TransformerClassifier(nn.Module):
    """Transformer encoder with mask-aware mean pooling."""

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # input_dim must be divisible by n_heads; project if needed
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)  # (batch, seq, hidden_dim)
        # src_key_padding_mask: True where the token should be ignored (i.e. padding)
        pad_mask = (mask == 0) if mask is not None else None
        enc = self.encoder(x, src_key_padding_mask=pad_mask)  # (batch, seq, hidden_dim)
        if mask is not None:
            # Mask-aware mean pool: sum over real timesteps, divide by real count
            real = mask.unsqueeze(-1)  # (batch, seq, 1)
            pooled = (enc * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = enc.mean(dim=1)
        return self.fc(self.drop(pooled)).squeeze(-1)


class TCNBlock(nn.Module):
    """Causal dilated conv block with residual projection and dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=causal_pad, dilation=dilation)
        self.trim = causal_pad  # amount to trim from right to enforce causality
        self.norm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # 1×1 projection for residual when channel dims differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.trim > 0:
            out = out[:, :, : -self.trim]
        out = self.drop(self.relu(self.norm(out)))
        return out + self.residual(x)


class TCNClassifier(nn.Module):
    """Three-layer TCN with exponentially growing dilation."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.block1 = TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.block3 = TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq, feat) → (batch, feat, seq) for Conv1d
        out = x.transpose(1, 2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled).squeeze(-1)


def _build_model(model_type: str, input_dim: int, cfg: SequenceModelConfig) -> nn.Module:
    if model_type == "transformer":
        return TransformerClassifier(input_dim, cfg.n_heads, cfg.num_layers, cfg.hidden_dim, cfg.dropout)
    if model_type == "tcn":
        return TCNClassifier(input_dim, cfg.hidden_dim, cfg.dropout)
    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'transformer' or 'tcn'.")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    training: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train() if training else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_x, batch_mask, batch_y in loader:
            batch_x, batch_mask, batch_y = (batch_x.to(device), batch_mask.to(device), batch_y.to(device))
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item() * len(batch_y)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    return (
        total_loss / max(len(loader.dataset), 1),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def train_sequence_model(
    sequences: np.ndarray,
    labels: np.ndarray,
    cfg: SequenceModelConfig | None = None,
    mask: np.ndarray | None = None,
    # Legacy dict-based config still accepted
    config: dict | None = None,
    model_type: str = "transformer",
) -> tuple[nn.Module, dict[str, float]]:
    """Train a Transformer or TCN sequence classifier with early stopping.

    Returns the best checkpoint (by val AUROC) and a full metrics dict.
    """
    if cfg is None:
        seq_cfg = (config or {}).get("sequence", {})
        cfg = SequenceModelConfig(
            model_type=model_type,
            hidden_dim=seq_cfg.get("hidden_dim", 64),
            batch_size=seq_cfg.get("batch_size", 64),
            epochs=seq_cfg.get("epochs", 30),
            lr=seq_cfg.get("lr", 1e-3),
        )

    if mask is None:
        mask = np.ones((len(labels), sequences.shape[1]), dtype=np.float32)

    torch.manual_seed(cfg.seed)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=cfg.val_size, stratify=labels, random_state=cfg.seed)

    train_ds = SequenceDataset(sequences[train_idx], labels[train_idx], mask[train_idx])
    val_ds = SequenceDataset(sequences[val_idx], labels[val_idx], mask[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = sequences.shape[-1]
    model = _build_model(cfg.model_type, input_dim, cfg).to(device)

    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auroc, best_state, patience_left = -1.0, None, cfg.patience
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(cfg.epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, cfg.grad_clip, training=True)
        val_loss, val_probs, val_labels_arr = _run_epoch(
            model, val_loader, criterion, None, device, cfg.grad_clip, training=False
        )
        val_metrics = compute_classification_metrics(val_labels_arr, val_probs)
        val_auroc = val_metrics["roc_auc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        scheduler.step(val_auroc)
        logger.info(
            "%s epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auroc=%.4f",
            cfg.model_type.upper(),
            epoch + 1,
            cfg.epochs,
            train_loss,
            val_loss,
            val_auroc,
        )

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_state)
    _, final_probs, final_labels_arr = _run_epoch(
        model, val_loader, criterion, None, device, cfg.grad_clip, training=False
    )
    final_metrics = compute_classification_metrics(final_labels_arr, final_probs)
    final_metrics["history"] = history
    return model, final_metrics


def predict_sequence_model(
    model: nn.Module,
    sequences: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if mask is None:
        mask = np.ones((len(sequences), sequences.shape[1]), dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(
            torch.tensor(sequences, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
        )
        return torch.sigmoid(logits).cpu().numpy()


__all__ = [
    "SequenceModelConfig",
    "SequenceDataset",
    "TransformerClassifier",
    "TCNClassifier",
    "TCNBlock",
    "train_sequence_model",
    "predict_sequence_model",
]
