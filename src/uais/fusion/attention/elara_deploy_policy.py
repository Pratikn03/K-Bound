"""ELARA deploy policy: coherence-certified gating with SAR fallback.

When the gate decision rule blocks switching (incoherent batch or switching
certificate not certified), predictions fall back to the frozen-validation
SAR score adapter instead of the static attention path. When switching is
allowed, per-sample fires route through the reliability (CRAF) path; other
samples use SAR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from uais.fusion.attention.baselines import SARScoreAdapter
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.fusion_inference import (
    GateDecisionCalibration,
    decide_switch_batch,
    predict_reliability_path_probs,
)
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

__all__ = [
    "DeployPolicySpec",
    "ElaraDeployArtifacts",
    "load_deploy_policy_spec",
    "predict_elara_deploy",
    "deploy_policy_from_cfg",
]


@dataclass
class DeployPolicySpec:
    """Locked deploy routing specification (ELARA_DEPLOY_v1)."""

    policy_id: str = "ELARA_DEPLOY_v1"
    routing_mode: str = "gdr_sar_fallback"
    fallback_method: str = "sar_score_adapter"
    gate_decision_rule: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "coherence_min": 0.5,
            "tau": 0.66,
            "margin_epsilon": 0.0,
        }
    )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DeployPolicySpec:
        routing = raw.get("routing") or {}
        gdr = routing.get("gate_decision_rule") or raw.get("gate_decision_rule") or {}
        return cls(
            policy_id=str(raw.get("policy_id", "ELARA_DEPLOY_v1")),
            routing_mode=str(routing.get("mode", raw.get("routing_mode", "gdr_sar_fallback"))),
            fallback_method=str(routing.get("fallback_method", "sar_score_adapter")),
            gate_decision_rule={
                "enabled": bool(gdr.get("enabled", True)),
                "coherence_min": float(gdr.get("coherence_min", 0.5)),
                "tau": float(gdr.get("tau", 0.66)),
                "margin_epsilon": float(gdr.get("margin_epsilon", 0.0)),
            },
        )


def load_deploy_policy_spec(path: str | Path) -> DeployPolicySpec:
    with Path(path).open(encoding="utf-8") as handle:
        return DeployPolicySpec.from_mapping(yaml.safe_load(handle) or {})


def deploy_policy_from_cfg(cfg: dict[str, Any]) -> DeployPolicySpec:
    block = cfg.get("elara_deploy") or {}
    if block.get("policy_path"):
        return load_deploy_policy_spec(block["policy_path"])
    if block:
        return DeployPolicySpec.from_mapping(block)
    lock = cfg.get("deploy_policy_lock")
    if lock:
        return load_deploy_policy_spec(lock)
    return DeployPolicySpec()


@dataclass
class ElaraDeployArtifacts:
    model: AttentionFusionModel
    estimator: ReliabilityEstimator
    sar: SARScoreAdapter
    device: torch.device
    gate_decision_calibration: GateDecisionCalibration | None = None
    gate_decision_rule_cfg: dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "coherence_min": 0.5, "tau": 0.66, "margin_epsilon": 0.0}
    )
    clean_gate_threshold: float = 0.66
    batch_size: int = 256
    rga_probs: np.ndarray | None = None


def select_validation_fallback(
    val_labels: np.ndarray,
    sar_val_probs: np.ndarray,
    rga_val_probs: np.ndarray,
) -> tuple[str, float, float]:
    """Pick SAR vs RGA+ fallback using validation ROC-AUC only."""
    from sklearn.metrics import roc_auc_score

    def _auc(labels: np.ndarray, probs: np.ndarray) -> float:
        if len(np.unique(labels)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(labels, probs))
        except ValueError:
            return 0.5

    sar_auc = _auc(val_labels, sar_val_probs)
    rga_auc = _auc(val_labels, rga_val_probs)
    if rga_auc >= sar_auc:
        return "rga_boosted_fusion", rga_auc, sar_auc
    return "sar_score_adapter", sar_auc, rga_auc


def predict_elara_deploy(
    artifacts: ElaraDeployArtifacts,
    features: np.ndarray,
    masks: np.ndarray,
    *,
    policy: DeployPolicySpec | None = None,
    fallback_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Apply ELARA deploy routing on a feature matrix."""
    policy = policy or DeployPolicySpec()
    gdr = artifacts.gate_decision_rule_cfg if artifacts.gate_decision_rule_cfg.get("enabled") else policy.gate_decision_rule
    if not gdr.get("enabled"):
        raise ValueError("ELARA deploy policy requires gate_decision_rule.enabled")

    model = artifacts.model
    estimator = artifacts.estimator
    device = artifacts.device
    batch_size = int(artifacts.batch_size)
    n = features.shape[0]
    probs_chunks: list[np.ndarray] = []
    if fallback_probs is not None:
        fallback_probs_full = np.asarray(fallback_probs, dtype=np.float32)
        if fallback_probs_full.shape[0] != n:
            raise ValueError("fallback_probs length must match features batch dimension.")
    else:
        fallback_probs_full = artifacts.sar.predict_proba(features, masks)
    adapted_samples = 0
    sar_fallback_samples = 0
    switch_allowed_batches = 0
    coherence_numer = 0.0
    coherence_denom = 0
    tau = float(gdr.get("tau", artifacts.clean_gate_threshold))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feat_np = features[start:end]
        mask_np = masks[start:end]
        fallback_batch = fallback_probs_full[start:end]
        rel_w = estimator.compute_reliability_weights(feat_np, mask_np)
        decision = decide_switch_batch(
            rel_w,
            mask_np,
            tau,
            artifacts.gate_decision_calibration,
            coherence_min=float(gdr.get("coherence_min", 0.5)),
            margin_epsilon=float(gdr.get("margin_epsilon", 0.0)),
        )
        batch_n = end - start
        if np.isfinite(decision.coherence):
            coherence_numer += float(decision.coherence) * batch_n
            coherence_denom += batch_n

        if not decision.switch_allowed:
            probs_chunks.append(fallback_batch.astype(np.float32))
            sar_fallback_samples += batch_n
            continue

        switch_allowed_batches += 1
        gate_per_sample = decision.decisions
        if not gate_per_sample.any():
            probs_chunks.append(fallback_batch.astype(np.float32))
            sar_fallback_samples += batch_n
            continue

        with torch.no_grad():
            rel_probs = predict_reliability_path_probs(model, rel_w, feat_np, mask_np, device)
        gate = gate_per_sample.astype(bool)
        batch_probs = np.where(gate, rel_probs, fallback_batch)
        probs_chunks.append(batch_probs.astype(np.float32))
        adapted_samples += int(gate.sum())
        sar_fallback_samples += int((~gate).sum())

    out = np.concatenate(probs_chunks) if probs_chunks else np.zeros(0, dtype=np.float32)
    stats = {
        "policy_id": policy.policy_id,
        "routing_mode": policy.routing_mode,
        "adaptation_rate": float(adapted_samples / n) if n else 0.0,
        "fallback_rate": float(sar_fallback_samples / n) if n else 0.0,
        "sar_fallback_rate": float(sar_fallback_samples / n) if n else 0.0,
        "fallback_method": str(policy.fallback_method),
        "switch_allowed_batches": int(switch_allowed_batches),
        "n_batches": int((n + batch_size - 1) // batch_size) if batch_size > 0 else 0,
        "mean_batch_coherence": float(coherence_numer / coherence_denom) if coherence_denom else float("nan"),
        "gate_decision_rule": True,
    }
    return out, stats
