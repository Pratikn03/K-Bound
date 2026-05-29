"""Shared loss and evaluation helpers for attention-fusion training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.utils.metrics import classification_metrics


def attention_fusion_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_weights: torch.Tensor | None,
    confidences: torch.Tensor | None = None,
    lambda_reg: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    entropy = torch.tensor(0.0, device=logits.device)
    if attention_weights is not None:
        attn = attention_weights.mean(dim=1)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
    conf_reg = torch.tensor(0.0, device=logits.device)
    if confidences is not None:
        conf_reg = ((confidences - 1.0) ** 2).mean()
    total_loss = bce_loss - lambda_reg * entropy + lambda_reg * conf_reg
    return total_loss, {
        "bce": float(bce_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "conf_reg": float(conf_reg.detach().cpu()),
    }


def evaluate_model(model: AttentionFusionModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            features, masks, labels = batch
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            logits, _, _ = model(features, key_padding_mask=masks)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    return classification_metrics(y_true, y_prob, threshold=0.5)
