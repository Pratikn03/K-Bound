"""Coherence-certified gating + switching-certificate calibration for fusion inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.gate_decision_rule import GateDecision, decide_switch
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

__all__ = [
    "GateDecisionCalibration",
    "build_gate_decision_calibration",
    "decide_switch_batch",
    "predict_reliability_path_probs",
    "predict_static_probs",
    "binary_cross_entropy_loss",
]


def binary_cross_entropy_loss(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


@torch.no_grad()
def predict_static_probs(
    model: AttentionFusionModel,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    n = features.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_t = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(masks[start:end], dtype=torch.bool, device=device)
        logits, _, _ = model(feat_t, key_padding_mask=mask_t)
        chunks.append(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    return np.concatenate(chunks)


@torch.no_grad()
def predict_reliability_path_probs(
    model: AttentionFusionModel,
    reliability_weights: np.ndarray,
    features: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    feat_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
    craf_t = torch.as_tensor(reliability_weights, dtype=torch.float32, device=device).masked_fill(mask_t, 0.0)
    embeds = [enc(feat_t[:, i, :]) for i, enc in enumerate(model.domain_encoders)]
    domain_embeds = torch.stack(embeds, dim=1)
    logits, _ = model.fusion(domain_embeds, key_padding_mask=mask_t, confidence_weights=craf_t)
    return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()


class GateDecisionCalibration:
    """Validation-only arrays for bounded_switching_certificate."""

    def __init__(
        self,
        static_loss: np.ndarray,
        reliability_loss: np.ndarray,
        fire_decisions: np.ndarray,
        *,
        certificate: dict[str, Any] | None = None,
    ) -> None:
        self.static_loss = np.asarray(static_loss, dtype=float)
        self.reliability_loss = np.asarray(reliability_loss, dtype=float)
        self.fire_decisions = np.asarray(fire_decisions, dtype=bool)
        self.certificate = certificate

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": int(self.static_loss.shape[0]),
            "fire_rate": float(self.fire_decisions.mean()) if self.fire_decisions.size else 0.0,
            "certificate": self.certificate,
        }


def build_gate_decision_calibration(
    model: AttentionFusionModel,
    estimator: ReliabilityEstimator,
    val_features: np.ndarray,
    val_masks: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    *,
    tau: float,
    batch_size: int = 256,
    margin_epsilon: float = 0.0,
) -> GateDecisionCalibration:
    """Build per-validation-sample losses for the switching certificate."""
    n = val_features.shape[0]
    static_loss_all: list[np.ndarray] = []
    rel_loss_all: list[np.ndarray] = []
    fire_all: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat = val_features[start:end]
        mask = val_masks[start:end]
        labels = val_labels[start:end]
        rel_w = estimator.compute_reliability_weights(feat, mask)
        static_probs = predict_static_probs(model, feat, mask, device, batch_size=batch_size)
        rel_probs = predict_reliability_path_probs(model, rel_w, feat, mask, device)
        static_loss_all.append(binary_cross_entropy_loss(labels, static_probs))
        rel_loss_all.append(binary_cross_entropy_loss(labels, rel_probs))
        from uais.fusion.attention.gate_decision_rule import per_sample_mean_reliability

        r = per_sample_mean_reliability(rel_w, mask)
        fire = np.zeros(r.shape[0], dtype=bool)
        finite = np.isfinite(r)
        fire[finite] = r[finite] < float(tau)
        fire_all.append(fire)

    static_loss = np.concatenate(static_loss_all)
    reliability_loss = np.concatenate(rel_loss_all)
    fire_decisions = np.concatenate(fire_all)
    from uais.utils.metrics import bounded_switching_certificate

    cert = bounded_switching_certificate(
        static_loss,
        reliability_loss,
        fire_decisions,
        margin_epsilon=margin_epsilon,
    )
    return GateDecisionCalibration(static_loss, reliability_loss, fire_decisions, certificate=cert)


def decide_switch_batch(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
    tau: float,
    calibration: GateDecisionCalibration | None,
    *,
    coherence_min: float = 0.5,
    margin_epsilon: float = 0.0,
) -> GateDecision:
    kwargs: dict[str, Any] = {
        "coherence_min": coherence_min,
        "margin_epsilon": margin_epsilon,
    }
    if calibration is not None:
        kwargs.update(
            {
                "calibration_static_loss": calibration.static_loss,
                "calibration_reliability_loss": calibration.reliability_loss,
                "calibration_fire_decisions": calibration.fire_decisions,
            }
        )
    return decide_switch(reliability_weights, masks, tau, **kwargs)
