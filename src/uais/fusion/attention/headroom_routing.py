"""Headroom-aware reliability routing for D15 natural-degradation evidence.

The functions here do not look at test labels. They transform per-modality
anomaly scores plus precomputed quality reliabilities into a routed score:
default to confidence-weighted mean in clean/no-headroom cases, and switch to
quality-reliability weighting only when a label-free stress score crosses a
validation-frozen threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "HeadroomThresholds",
    "confidence_weighted_mean",
    "headroom_routed_score",
    "headroom_subset_masks",
    "select_headroom_thresholds",
    "stress_score_from_reliability",
]


@dataclass(frozen=True)
class HeadroomThresholds:
    stress_threshold: float
    clean_threshold: float
    stress_quantile: float
    clean_quantile: float
    selection_split: str = "validation"
    used_test_labels: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stress_threshold": float(self.stress_threshold),
            "clean_threshold": float(self.clean_threshold),
            "stress_quantile": float(self.stress_quantile),
            "clean_quantile": float(self.clean_quantile),
            "selection_split": self.selection_split,
            "used_test_labels": bool(self.used_test_labels),
        }


def _finite_2d(x: np.ndarray | list[list[float]], *, fill: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expected a 2D [n_samples, n_modalities] array")
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def confidence_weighted_mean(
    scores: np.ndarray | list[list[float]],
    score_confidence: np.ndarray | list[list[float]] | None = None,
) -> np.ndarray:
    """Confidence-weighted mean baseline over modality anomaly scores."""
    s = _finite_2d(scores, fill=0.5)
    if score_confidence is None:
        return s.mean(axis=1)
    c = np.clip(_finite_2d(score_confidence, fill=1.0), 1e-6, None)
    return (s * c).sum(axis=1) / c.sum(axis=1)


def stress_score_from_reliability(
    quality_reliability: np.ndarray | list[list[float]],
) -> np.ndarray:
    """Label-free natural stress score from modality reliability.

    The score is high when one modality is much less reliable than another or
    when every modality is weak. It is bounded to [0, 1] for stable thresholds.
    """
    r = np.clip(_finite_2d(quality_reliability, fill=0.0), 0.0, 1.0)
    disagreement = r.max(axis=1) - r.min(axis=1)
    low_reliability = 1.0 - r.min(axis=1)
    return np.clip(np.maximum(disagreement, low_reliability), 0.0, 1.0)


def select_headroom_thresholds(
    validation_quality_reliability: np.ndarray | list[list[float]],
    *,
    stress_quantile: float = 0.70,
    clean_quantile: float = 0.30,
) -> HeadroomThresholds:
    """Freeze D15 stress/clean thresholds from validation reliability only."""
    if not 0.0 < clean_quantile < stress_quantile < 1.0:
        raise ValueError("expected 0 < clean_quantile < stress_quantile < 1")
    stress = stress_score_from_reliability(validation_quality_reliability)
    return HeadroomThresholds(
        stress_threshold=float(np.quantile(stress, stress_quantile)),
        clean_threshold=float(np.quantile(stress, clean_quantile)),
        stress_quantile=float(stress_quantile),
        clean_quantile=float(clean_quantile),
    )


def headroom_routed_score(
    scores: np.ndarray | list[list[float]],
    score_confidence: np.ndarray | list[list[float]] | None,
    quality_reliability: np.ndarray | list[list[float]],
    *,
    stress_threshold: float,
    reliability_floor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return routed score, gate-active mask, and label-free stress score."""
    s = _finite_2d(scores, fill=0.5)
    raw_reliability = np.clip(_finite_2d(quality_reliability, fill=0.0), 0.0, 1.0)
    r = np.clip(raw_reliability, reliability_floor, 1.0)
    cw = confidence_weighted_mean(s, score_confidence)
    stress = stress_score_from_reliability(raw_reliability)
    gate_active = stress >= float(stress_threshold)
    routed = cw.copy()
    quality_weighted = (s * r).sum(axis=1) / r.sum(axis=1)
    routed[gate_active] = quality_weighted[gate_active]
    return np.nan_to_num(routed, nan=0.5, posinf=1.0, neginf=0.0), gate_active, stress


def headroom_subset_masks(
    stress_scores: np.ndarray | list[float],
    *,
    stress_threshold: float,
    clean_threshold: float,
) -> dict[str, np.ndarray]:
    s = np.asarray(stress_scores, dtype=float)
    return {
        "stress": s >= float(stress_threshold),
        "clean": s <= float(clean_threshold),
        "all": np.ones_like(s, dtype=bool),
    }
