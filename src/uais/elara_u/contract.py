"""ELARA-U Universal Evidence Contract metrics (Gate U / D21).

Rank, regret-to-oracle, negative-transfer rate, calibration (ECE proxy), and a
paired bootstrap over tasks. Pure metric helpers; no I/O.
"""

from __future__ import annotations

import numpy as np


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error treating the [0,1] score as P(anomaly).
    Proxy on z-sigmoid scores (not a fitted probability); reported as such."""
    s = np.clip(np.asarray(scores, float), 0, 1)
    y = np.asarray(labels).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (s >= edges[i]) & (s < edges[i + 1] if i < bins - 1 else s <= edges[i + 1])
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e)


def ranks_and_regret(per_task_auc: dict[str, list[float]]) -> dict:
    """per_task_auc[method] = list of test AUROC over tasks (aligned).
    Returns mean_rank, worst_rank, mean_regret, mean_auc per method."""
    methods = list(per_task_auc)
    n = len(next(iter(per_task_auc.values())))
    ranks = {m: [] for m in methods}
    regret = {m: [] for m in methods}
    for i in range(n):
        col = {m: per_task_auc[m][i] for m in methods}
        order = sorted(methods, key=lambda m: col[m], reverse=True)
        oracle = col[order[0]]
        for m in methods:
            ranks[m].append(1 + order.index(m))
            regret[m].append(oracle - col[m])
    return {
        "mean_rank": {m: float(np.mean(ranks[m])) for m in methods},
        "worst_rank": {m: int(np.max(ranks[m])) for m in methods},
        "mean_regret": {m: float(np.mean(regret[m])) for m in methods},
        "mean_auc": {m: float(np.mean(per_task_auc[m])) for m in methods},
        "_ranks": ranks,
    }


def negative_transfer_rate(per_task_auc: dict, method: str, ref: str) -> float:
    """Fraction of tasks where `method` is worse than `ref` (the safe baseline)."""
    a, b = per_task_auc[method], per_task_auc[ref]
    return float(np.mean([1.0 if a[i] < b[i] - 1e-9 else 0.0 for i in range(len(a))]))


def bootstrap_delta(ranks_a: list[int], ranks_b: list[int], n_iter: int = 10000,
                    seed: int = 0) -> dict:
    """Paired bootstrap over tasks of mean(rank_a) - mean(rank_b). Negative = a better."""
    a, b = np.array(ranks_a, float), np.array(ranks_b, float)
    rng = np.random.default_rng(seed)
    d = [float(a[idx].mean() - b[idx].mean())
         for idx in (rng.integers(0, len(a), len(a)) for _ in range(n_iter))]
    return {"delta": float(a.mean() - b.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci_excludes_zero": bool(np.percentile(d, 97.5) < 0 or np.percentile(d, 2.5) > 0)}
