"""LSTM classifier for sequence anomaly detection.

Key improvements over the original stub:
- Uses pack_padded_sequence so padding tokens never influence hidden state
- Bidirectional option doubles representational capacity
- Training loop has a val split, early stopping, gradient clipping,
  and returns full classification metrics (AUROC, F1, PR-AUC, etc.)
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
class LSTMConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip: float = 1.0
    val_size: float = 0.15
    seed: int = 42


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.mask[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # Concat forward and backward final hidden states
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.fc(self.drop(h_last)).squeeze(-1)


def _run_epoch(
    model: LSTMClassifier,
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


def train_lstm_classifier(
    sequences: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    cfg: LSTMConfig | None = None,
    # Legacy dict-based config still accepted for backward compat
    config: dict | None = None,
) -> tuple[LSTMClassifier, dict[str, float]]:
    """Train a bidirectional LSTM with early stopping.

    Returns the best checkpoint (by val AUROC) and a metrics dict containing
    roc_auc, pr_auc, f1, precision, recall, accuracy on the validation split.
    """
    if cfg is None:
        seq_cfg = (config or {}).get("sequence", {})
        cfg = LSTMConfig(
            hidden_dim=seq_cfg.get("hidden_dim", 64),
            batch_size=seq_cfg.get("batch_size", 64),
            epochs=seq_cfg.get("epochs", 30),
            lr=seq_cfg.get("lr", 1e-3),
        )

    torch.manual_seed(cfg.seed)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=cfg.val_size, stratify=labels, random_state=cfg.seed)

    train_ds = SequenceDataset(sequences[train_idx], mask[train_idx], labels[train_idx])
    val_ds = SequenceDataset(sequences[val_idx], mask[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = sequences.shape[-1]
    model = LSTMClassifier(input_dim, cfg.hidden_dim, cfg.num_layers, cfg.bidirectional, cfg.dropout).to(device)

    # Compensate for class imbalance
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auroc, best_state, patience_left = -1.0, None, cfg.patience
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(cfg.epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, cfg.grad_clip, training=True)
        val_loss, val_probs, val_labels = _run_epoch(
            model, val_loader, criterion, None, device, cfg.grad_clip, training=False
        )
        val_metrics = compute_classification_metrics(val_labels, val_probs)
        val_auroc = val_metrics["roc_auc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        scheduler.step(val_auroc)
        logger.info(
            "LSTM epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auroc=%.4f",
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
    _, final_probs, final_labels = _run_epoch(model, val_loader, criterion, None, device, cfg.grad_clip, training=False)
    final_metrics = compute_classification_metrics(final_labels, final_probs)
    final_metrics["history"] = history
    return model, final_metrics


def predict_lstm(model: LSTMClassifier, sequences: np.ndarray, mask: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(
            torch.tensor(sequences, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
        )
        return torch.sigmoid(logits).cpu().numpy()


__all__ = ["LSTMConfig", "LSTMClassifier", "SequenceDataset", "train_lstm_classifier", "predict_lstm"]
