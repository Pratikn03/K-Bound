"""Small numerical primitives used by the multi-candidate diagnostics.

These functions estimate an off-diagonal rank-one agreement model. They are
diagnostic numerical routines, not machine-checked theorem implementations.
The sign convention assumes candidate zero is the declared positive anchor.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def rankone_fit_offdiag(
    matrix: np.ndarray,
    iterations: int = 90,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Fit a rank-one model to off-diagonal entries and return its residual."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    size = values.shape[0]
    work = values.copy()
    diagonal = np.sqrt(
        np.clip((values**2).sum(axis=1) / max(size - 1, 1), 1e-6, None)
    )
    previous = None
    estimate = np.zeros(size)
    for _ in range(iterations):
        np.fill_diagonal(work, diagonal**2)
        eigenvalues, eigenvectors = np.linalg.eigh(work)
        estimate = np.sqrt(max(float(eigenvalues[-1]), 0.0)) * eigenvectors[:, -1]
        if estimate[0] < 0.0:
            estimate = -estimate
        diagonal = np.abs(estimate)
        if previous is not None and np.linalg.norm(estimate - previous) < tolerance:
            break
        previous = estimate.copy()
    off_diagonal = ~np.eye(size, dtype=bool)
    residuals = values[off_diagonal] - np.outer(estimate, estimate)[off_diagonal]
    return estimate, float(np.linalg.norm(residuals))


def minor_estimator(matrix: np.ndarray) -> np.ndarray:
    """Estimate anchored advantages using the median of valid minor ratios."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    size = values.shape[0]
    squared = np.zeros(size)
    for index in range(size):
        others = [item for item in range(size) if item != index]
        ratios = []
        for left, right in combinations(others, 2):
            denominator = values[left, right]
            if abs(denominator) > 1e-9:
                ratios.append(values[index, left] * values[index, right] / denominator)
        squared[index] = np.median(ratios) if ratios else 0.0
    magnitudes = np.sqrt(np.clip(squared, 0.0, 1.0))
    signs = np.ones(size)
    for index in range(1, size):
        signs[index] = np.sign(values[0, index]) or 1.0
    return magnitudes * signs


def overdet_residual(matrix: np.ndarray) -> float:
    """Return the mean pairing spread over all four-candidate subsets."""
    values = np.asarray(matrix, dtype=float)
    size = values.shape[0]
    if values.ndim != 2 or values.shape[1] != size:
        raise ValueError("matrix must be square")
    if size < 4:
        return 0.0
    spreads = []
    for first, second, third, fourth in combinations(range(size), 4):
        products = (
            values[first, second] * values[third, fourth],
            values[first, third] * values[second, fourth],
            values[first, fourth] * values[second, third],
        )
        spreads.append(max(products) - min(products))
    return float(np.mean(spreads))


def w2_gaussian(mean_a: float, std_a: float, mean_b: float, std_b: float) -> float:
    """Return the Wasserstein-2 distance between one-dimensional Gaussians."""
    return float(np.hypot(mean_b - mean_a, std_b - std_a))
