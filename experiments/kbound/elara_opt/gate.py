"""gate.py — g_phi: map label-free reliability features -> nonnegative objective
weights over {entropy, filtered_entropy, aug_consistency}.

Three modes:
  * uniform : constant [1/3, 1/3, 1/3].
  * rule    : a deterministic function of the reliability scalar + aug-disagreement,
              fully specified by constants in config.py / the lock file.
  * meta    : a tiny MLP trained ONLY on source/dev shift tasks (see
              train_meta_gate.py); deterministic forward.

All inputs are label-free.  Weights are a softmax (nonnegative, sum to 1).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .reliability import RELIABILITY_NAMES, FEATURE_DIM, to_vector, reliability_score


def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class MetaGate(nn.Module):
    """Small gate network: reliability features -> 3 objective logits."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def uniform_weights() -> np.ndarray:
    return np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3.0


def rule_weights(feats: Dict[str, float], cfg: Dict) -> np.ndarray:
    """Deterministic reliability rule. Fully specified by cfg['rule'] + coeffs."""
    s = reliability_score(feats, cfg["reliability_coeffs"])  # in (0,1)
    r = cfg["rule"]
    logit_entropy = r["a_entropy"] + r["k_reliable"] * s
    logit_filtered = r["a_filtered"]
    logit_aug = r["a_aug"] + r["k_unreliable"] * (1.0 - s) + r["k_disagree"] * feats["aug_disagreement"]
    logits = np.array([logit_entropy, logit_filtered, logit_aug], dtype=np.float64)
    return _softmax(logits, cfg.get("gate_temperature", 1.0))


def meta_weights(feats: Dict[str, float], cfg: Dict, model: MetaGate) -> np.ndarray:
    """Deterministic MLP gate forward on the canonical feature vector."""
    model.eval()
    with torch.no_grad():
        v = torch.tensor(to_vector(feats), dtype=torch.float32).unsqueeze(0)
        logits = model(v).squeeze(0).numpy().astype(np.float64)
    return _softmax(logits, cfg.get("gate_temperature", 1.0))


def compute_weights(mode: str, feats: Dict[str, float], cfg: Dict, meta_model: MetaGate | None = None) -> np.ndarray:
    """Dispatch to the requested gate mode. Returns nonneg weights summing to 1."""
    if mode == "elara_uniform":
        return uniform_weights()
    if mode == "elara_rule":
        return rule_weights(feats, cfg)
    if mode == "elara_meta":
        if meta_model is None:
            raise ValueError("elara_meta requires a trained MetaGate checkpoint")
        return meta_weights(feats, cfg, meta_model)
    raise ValueError(f"unknown ELARA-Opt mode: {mode}")
