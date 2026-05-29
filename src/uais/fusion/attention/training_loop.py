"""Unified attention-fusion training loop (early stopping + best-weight restore).

All fusion entry points should call :func:`train_attention_model` so early-stopping
criteria and checkpoint restoration stay consistent across:

- ``run_breakthrough_experiment.py``
- ``train_attention_fusion.py``
- ``evaluate_attention_harness.py``
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from uais.fusion.attention.attention_utils import apply_domain_dropout
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.fusion_training_utils import attention_fusion_loss, evaluate_model


@dataclass(frozen=True)
class AttentionTrainingResult:
    """Summary of a completed training run."""

    val_best_pr_auc: float
    val_best_loss: float
    epochs_run: int
    early_stopping_metric: str
    restored_best_weights: bool
    train_loss_last_epoch: float


def pseudo_targets_from_domain_scores(
    features: np.ndarray,
    masks: np.ndarray,
    score_index: int,
    *,
    aggregation: str = "max",
    quantile: float = 0.75,
    trim_frac: float = 0.2,
) -> np.ndarray:
    """Weak supervision from per-domain anomaly scores (one-class protocols)."""
    scores = features[:, :, score_index].astype(float).copy()
    scores[masks] = np.nan
    with np.errstate(invalid="ignore"):
        if aggregation == "mean":
            pseudo = np.nanmean(scores, axis=1)
        elif aggregation == "median":
            pseudo = np.nanmedian(scores, axis=1)
        elif aggregation == "quantile":
            pseudo = np.nanquantile(scores, float(quantile), axis=1)
        elif aggregation == "trimmed_mean":
            lo = np.nanquantile(scores, float(trim_frac), axis=1, keepdims=True)
            hi = np.nanquantile(scores, 1.0 - float(trim_frac), axis=1, keepdims=True)
            kept = np.where((scores >= lo) & (scores <= hi), scores, np.nan)
            pseudo = np.nanmean(kept, axis=1)
        else:
            pseudo = np.nanmax(scores, axis=1)
    return np.nan_to_num(pseudo, nan=0.5).astype(np.float32)


def dropout_score_input(
    feats: torch.Tensor,
    masks: torch.Tensor,
    score_index: int,
    p: float,
) -> torch.Tensor:
    """Training-only score-column dropout to break score-copy shortcuts (one-class)."""
    if p <= 0.0:
        return feats
    present = ~masks
    drop = (torch.rand(present.shape, device=feats.device) < float(p)) & present
    out = feats.clone()
    score_col = out[:, :, score_index]
    out[:, :, score_index] = torch.where(drop, torch.full_like(score_col, 0.5), score_col)
    return out


def _batch_loss(
    model: AttentionFusionModel,
    feats: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    *,
    lambda_reg: float,
) -> torch.Tensor:
    logits, attn_weights, confidences = model(feats, key_padding_mask=masks)
    loss, _ = attention_fusion_loss(
        logits.squeeze(-1),
        targets,
        attn_weights,
        confidences,
        lambda_reg=lambda_reg,
    )
    return loss


def _epoch_train_loss(
    model: AttentionFusionModel,
    train_loader: DataLoader,
    device: torch.device,
    train_cfg: dict,
    *,
    score_index: int | None,
    one_class_scores: bool,
) -> float:
    model.train()
    domain_dropout_p = float(train_cfg.get("domain_dropout", 0.1))
    lambda_reg = float(train_cfg.get("lambda_reg", 0.01))
    pseudo_agg = str(train_cfg.get("one_class_score_aggregation", "max"))
    pseudo_quantile = float(train_cfg.get("one_class_score_quantile", 0.75))
    pseudo_trim_frac = float(train_cfg.get("one_class_score_trim_frac", 0.2))
    score_input_dropout = float(train_cfg.get("one_class_score_input_dropout", 0.0)) if one_class_scores else 0.0

    losses: list[float] = []
    for batch in train_loader:
        feats, msks, lbls = [x.to(device) for x in batch]
        if domain_dropout_p > 0.0:
            msks = apply_domain_dropout(msks, drop_prob=domain_dropout_p)
        if one_class_scores and score_index is not None:
            pseudo = pseudo_targets_from_domain_scores(
                feats.detach().cpu().numpy(),
                msks.detach().cpu().numpy(),
                score_index,
                aggregation=pseudo_agg,
                quantile=pseudo_quantile,
                trim_frac=pseudo_trim_frac,
            )
            targets = torch.tensor(pseudo, dtype=torch.float32, device=device)
            model_feats = dropout_score_input(feats, msks, score_index, score_input_dropout)
        else:
            targets = lbls.float()
            model_feats = feats
        loss = _batch_loss(model, model_feats, msks, targets, lambda_reg=lambda_reg)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _epoch_val_metrics(
    model: AttentionFusionModel,
    val_loader: DataLoader,
    device: torch.device,
    train_cfg: dict,
    *,
    score_index: int | None,
    one_class_scores: bool,
) -> tuple[float, float]:
    """Return (mean_val_loss, val_pr_auc)."""
    model.eval()
    lambda_reg = 0.0
    pseudo_agg = str(train_cfg.get("one_class_score_aggregation", "max"))
    pseudo_quantile = float(train_cfg.get("one_class_score_quantile", 0.75))
    pseudo_trim_frac = float(train_cfg.get("one_class_score_trim_frac", 0.2))

    val_losses: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            feats, msks, lbls = [x.to(device) for x in batch]
            if one_class_scores and score_index is not None:
                pseudo = pseudo_targets_from_domain_scores(
                    feats.cpu().numpy(),
                    msks.cpu().numpy(),
                    score_index,
                    aggregation=pseudo_agg,
                    quantile=pseudo_quantile,
                    trim_frac=pseudo_trim_frac,
                )
                targets = torch.tensor(pseudo, dtype=torch.float32, device=device)
            else:
                targets = lbls.float()
            loss = _batch_loss(model, feats, msks, targets, lambda_reg=lambda_reg)
            val_losses.append(float(loss.item()))

    val_metrics = evaluate_model(model, val_loader, device)
    pr_auc = float(val_metrics.get("pr_auc", float("nan")))
    val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
    return val_loss, pr_auc


def train_attention_model(
    model: AttentionFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: dict,
    device: torch.device,
    *,
    score_index: int | None = None,
) -> AttentionTrainingResult:
    """Train with early stopping; restore best validation checkpoint in-memory.

    Parameters
    ----------
    train_cfg:
        Training section of the fusion YAML. Important keys:

        - ``early_stopping_metric``: ``pr_auc`` (default, audit standard) or ``val_loss``
        - ``restore_best_weights``: default ``True``; set ``False`` only for legacy repro
        - ``early_stopping``: patience (epochs without improvement)
        - ``epochs``, ``lr``, ``weight_decay``, ``domain_dropout``, ``lambda_reg``
        - ``one_class_score_supervision``: enable pseudo-label training
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    use_scheduler = bool(train_cfg.get("use_lr_scheduler", True))
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        if use_scheduler
        else None
    )

    patience = int(train_cfg.get("early_stopping", 5))
    max_epochs = int(train_cfg.get("epochs", 20))
    restore_best = bool(train_cfg.get("restore_best_weights", True))
    metric_name = str(train_cfg.get("early_stopping_metric", "pr_auc")).lower().strip()
    if metric_name not in {"pr_auc", "val_loss"}:
        raise ValueError(f"early_stopping_metric must be 'pr_auc' or 'val_loss', got {metric_name!r}")

    one_class_scores = bool(train_cfg.get("one_class_score_supervision", False)) and score_index is not None
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

    best_pr_auc = -1.0
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    no_improve = 0
    train_loss_last = float("nan")
    epochs_run = 0

    for _epoch in range(max_epochs):
        epochs_run = _epoch + 1
        model.train()
        domain_dropout_p = float(train_cfg.get("domain_dropout", 0.1))
        lambda_reg = float(train_cfg.get("lambda_reg", 0.01))
        pseudo_agg = str(train_cfg.get("one_class_score_aggregation", "max"))
        pseudo_quantile = float(train_cfg.get("one_class_score_quantile", 0.75))
        pseudo_trim_frac = float(train_cfg.get("one_class_score_trim_frac", 0.2))
        score_input_dropout = float(train_cfg.get("one_class_score_input_dropout", 0.0)) if one_class_scores else 0.0

        epoch_losses: list[float] = []
        for batch in train_loader:
            feats, msks, lbls = [x.to(device) for x in batch]
            if domain_dropout_p > 0.0:
                msks = apply_domain_dropout(msks, drop_prob=domain_dropout_p)
            optimizer.zero_grad()
            if one_class_scores and score_index is not None:
                pseudo = pseudo_targets_from_domain_scores(
                    feats.detach().cpu().numpy(),
                    msks.detach().cpu().numpy(),
                    score_index,
                    aggregation=pseudo_agg,
                    quantile=pseudo_quantile,
                    trim_frac=pseudo_trim_frac,
                )
                targets = torch.tensor(pseudo, dtype=torch.float32, device=device)
                model_feats = dropout_score_input(feats, msks, score_index, score_input_dropout)
            else:
                targets = lbls.float()
                model_feats = feats
            loss = _batch_loss(model, model_feats, msks, targets, lambda_reg=lambda_reg)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        train_loss_last = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_loss, pr_auc = _epoch_val_metrics(
            model, val_loader, device, train_cfg, score_index=score_index, one_class_scores=one_class_scores
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        improved = False
        if metric_name == "pr_auc":
            if np.isfinite(pr_auc) and pr_auc > best_pr_auc + 1e-5:
                best_pr_auc = pr_auc
                improved = True
        elif val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            improved = True

        if improved:
            no_improve = 0
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
            if np.isfinite(pr_auc):
                best_pr_auc = max(best_pr_auc, pr_auc)
            best_val_loss = min(best_val_loss, val_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    restored = False
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        restored = True

    if not np.isfinite(best_pr_auc) or best_pr_auc < 0:
        _, best_pr_auc = _epoch_val_metrics(
            model, val_loader, device, train_cfg, score_index=score_index, one_class_scores=one_class_scores
        )

    return AttentionTrainingResult(
        val_best_pr_auc=float(best_pr_auc),
        val_best_loss=float(best_val_loss),
        epochs_run=int(epochs_run),
        early_stopping_metric=metric_name,
        restored_best_weights=restored,
        train_loss_last_epoch=float(train_loss_last),
    )
