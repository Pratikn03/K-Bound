"""Simple statistical utilities for CI and significance tests."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    midranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1
        midranks[sorted_idx[i:j]] = midrank
        i = j
    return midranks


def _fast_delong(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    predictions = np.asarray(predictions, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    finite = np.isfinite(predictions) & np.isfinite(labels)
    predictions = predictions[finite]
    labels = labels[finite]
    pos = predictions[labels == 1]
    neg = predictions[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    k = len(pos)
    m = len(neg)
    tx = _compute_midrank(np.concatenate([pos, neg]))[:k]
    ty = _compute_midrank(np.concatenate([neg, pos]))[:m]
    auc = (tx.sum() - k * (k + 1) / 2.0) / (k * m)
    v01 = (tx - np.arange(1, k + 1)) / m
    v10 = 1.0 - (ty - np.arange(1, m + 1)) / k
    sx = np.cov(v01, bias=True)
    sy = np.cov(v10, bias=True)
    delong_var = sx / k + sy / m
    return float(auc), float(delong_var)


def delong_roc_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> float:
    """Return p-value for DeLong test between two ROC AUCs."""
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    auc_a, var_a = _fast_delong(y_score_a, y_true)
    auc_b, var_b = _fast_delong(y_score_b, y_true)
    if np.isnan(auc_a) or np.isnan(auc_b):
        return float("nan")
    var = var_a + var_b
    if var <= 0:
        return float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p_value)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    """Return p-value of paired t-test between two score arrays."""
    _, p = stats.ttest_rel(a, b, nan_policy="omit")
    return float(p)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """Return p-value of Wilcoxon signed-rank test between two score arrays."""
    try:
        _, p = stats.wilcoxon(a, b)
    except ValueError:
        p = np.nan
    return float(p)


__all__ = ["bootstrap_ci", "paired_ttest", "wilcoxon_test", "delong_roc_test"]
