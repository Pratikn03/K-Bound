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


def holm_bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down multiple-comparison correction.

    Given an array of raw p-values from ``m`` hypothesis tests, returns the
    adjusted p-values and per-test reject decisions at family-wise error rate
    ``alpha``. Holm's method is uniformly more powerful than Bonferroni while
    keeping FWER ≤ α.

    Parameters
    ----------
    pvalues : array-like of floats in [0, 1] (may include NaN — those tests
              are treated as missing and never rejected).
    alpha   : family-wise α (default 0.05).

    Returns
    -------
    dict with keys:
      - "p_adjusted": [m] adjusted p-values, capped at 1.0, in input order
      - "reject":     [m] bool — True if H0_i is rejected at FWER ≤ α
      - "n_tests":    int — number of non-NaN tests in the family
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m_total = len(p)
    valid = np.isfinite(p)
    m = int(valid.sum())
    p_adj = np.full(m_total, np.nan, dtype=float)
    reject = np.zeros(m_total, dtype=bool)
    if m == 0:
        return {"p_adjusted": p_adj, "reject": reject, "n_tests": 0}

    valid_idxs = np.where(valid)[0]
    p_valid = p[valid_idxs]
    order = np.argsort(p_valid)
    sorted_p = p_valid[order]

    # Holm: adjusted_p_(k) = max_{j <= k} (m - j + 1) * p_(j), capped at 1.
    multipliers = np.arange(m, 0, -1, dtype=float)
    raw_scaled = sorted_p * multipliers
    # Enforce monotonic-non-decreasing (running max)
    monotone = np.maximum.accumulate(raw_scaled)
    monotone_capped = np.minimum(monotone, 1.0)

    # Map back to original positions
    sorted_p_adj = monotone_capped
    sorted_reject = sorted_p_adj <= alpha

    # Invert the sort to put values back in original ordering
    inverse_order = np.argsort(order)
    p_valid_adj = sorted_p_adj[inverse_order]
    p_valid_reject = sorted_reject[inverse_order]

    p_adj[valid_idxs] = p_valid_adj
    reject[valid_idxs] = p_valid_reject
    return {"p_adjusted": p_adj, "reject": reject, "n_tests": m}


__all__ = [
    "bootstrap_ci",
    "paired_ttest",
    "wilcoxon_test",
    "delong_roc_test",
    "holm_bonferroni",
]
