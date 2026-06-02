"""Simple statistical utilities for CI and significance tests."""

from __future__ import annotations

from typing import Callable

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


def _auc_structural_components(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    pos = predictions[labels == 1]
    neg = predictions[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float)
    comparisons = (pos[:, None] > neg[None, :]).astype(float)
    comparisons += 0.5 * (pos[:, None] == neg[None, :])
    auc = float(comparisons.mean())
    positive_components = comparisons.mean(axis=1)
    negative_components = comparisons.mean(axis=0)
    return auc, positive_components, negative_components


def _fast_delong(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predictions = np.asarray(predictions, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    finite = np.isfinite(predictions) & np.isfinite(labels)
    predictions = predictions[finite]
    labels = labels[finite]
    auc, v10, v01 = _auc_structural_components(predictions, labels)
    if np.isnan(auc):
        return float("nan"), float("nan")
    n_pos = len(v10)
    n_neg = len(v01)
    # Unbiased (ddof=1) sample variance of the placement values is the canonical
    # DeLong estimator; ddof=0 under-estimates variance and makes p-values
    # slightly anti-conservative at small n (audit M2).
    delong_var = float(np.var(v10, ddof=1) / n_pos + np.var(v01, ddof=1) / n_neg)
    return auc, delong_var


def delong_roc_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> float:
    """Return p-value for DeLong test between two ROC AUCs."""
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    finite = np.isfinite(y_true) & np.isfinite(y_score_a) & np.isfinite(y_score_b)
    y_true = y_true[finite]
    y_score_a = y_score_a[finite]
    y_score_b = y_score_b[finite]
    auc_a, v10_a, v01_a = _auc_structural_components(y_score_a.astype(float), y_true)
    auc_b, v10_b, v01_b = _auc_structural_components(y_score_b.astype(float), y_true)
    if np.isnan(auc_a) or np.isnan(auc_b):
        return float("nan")
    n_pos = len(v10_a)
    n_neg = len(v01_a)
    sx = np.cov(np.vstack([v10_a, v10_b]), bias=False)  # unbiased (audit M2)
    sy = np.cov(np.vstack([v01_a, v01_b]), bias=False)
    covariance = sx / n_pos + sy / n_neg
    var = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if var <= 1e-12:
        return 1.0 if abs(auc_a - auc_b) <= 1e-12 else 0.0
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
) -> tuple[float, float]:
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
