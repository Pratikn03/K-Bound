"""Shared calibration primitives for raw-data experiment runners.

The training runners produce cross-fitted or held-out residuals. This module
only computes the finite-sample rank radius from those residuals; the validity
claim still depends on how the caller constructed its calibration split.
"""

from __future__ import annotations

import math

import numpy as np


def exact_rank_radius(residuals: np.ndarray, alpha: float) -> float:
    """Return the finite-sample order statistic without interpolation."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one calibration residual is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration residuals must be finite")
    if np.any(values < 0.0):
        raise ValueError("calibration residuals must be non-negative")
    k = min(values.size, math.ceil((values.size + 1) * (1.0 - alpha)))
    return float(np.partition(values, k - 1)[k - 1])
