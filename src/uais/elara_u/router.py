"""ELARA-U reliability-aware meta-router: select / fuse / abstain / fall back.

Operates on a per-task score archive (validation + test detector scores in [0,1]
plus validation labels). The router uses VALIDATION-ONLY reliability features; it
never sees test labels. Policy thresholds (when to select vs fuse vs fall back)
are meant to be fit on meta-train tasks and applied to held-out tasks /
families (leave-family-out), so the router does not tune on its evaluation tasks.

Actions
-------
select   : trust the single most reliable expert (validation-AUROC argmax).
fuse     : reliability-weighted mean over non-degenerate experts (robust).
hybrid   : select when one expert is clearly+confidently best, else fuse.
fallback : if no expert is reliable (negative-transfer guard), use rank-mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def reliability_features(s_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Validation-only per-expert reliability summary for a task."""
    y = np.asarray(y_val).astype(int)
    M = s_val.shape[1]
    auc = np.array([roc_auc_score(y, s_val[:, m]) if len(np.unique(y)) > 1 else 0.5
                    for m in range(M)])
    sharp = np.abs(s_val - 0.5).mean(0)                 # confidence/sharpness
    # inter-expert disagreement: mean pairwise rank correlation distance
    order = np.argsort(np.argsort(s_val, axis=0), axis=0) / max(len(y) - 1, 1)
    disagree = float(np.mean([np.abs(order[:, i] - order[:, j]).mean()
                              for i in range(M) for j in range(i + 1, M)])) if M > 1 else 0.0
    return {"val_auc": auc, "sharpness": sharp, "disagreement": disagree,
            "best_auc": float(auc.max()), "gap": float(np.sort(auc)[-1] - np.sort(auc)[-2]) if M > 1 else 0.0}


def _reliab_weights(auc: np.ndarray, floor: float = 0.5) -> np.ndarray:
    w = np.clip(auc - floor, 0.0, None)
    return w / w.sum() if w.sum() > 1e-9 else np.ones_like(auc) / len(auc)


def _rank_mean(S: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)).mean(1)


def fuse(S: np.ndarray, auc: np.ndarray, floor: float = 0.55) -> np.ndarray:
    """Reliability-weighted fusion over non-degenerate experts."""
    keep = auc >= floor
    if not keep.any():
        return _rank_mean(S)                            # fallback: all weak
    w = _reliab_weights(auc[keep], floor=0.5)
    return S[:, keep] @ w


def select(S: np.ndarray, auc: np.ndarray) -> np.ndarray:
    return S[:, int(np.argmax(auc))]


@dataclass
class RouterPolicy:
    """Thresholds fit on meta-train tasks; applied unchanged to held-out tasks."""
    conf: float = 0.70      # min best-AUROC to trust a single expert
    gap: float = 0.05       # min lead of best over 2nd-best to select
    guard: float = 0.55     # below this best-AUROC -> negative-transfer fallback


def route(s_val, y_val, s_test, policy: RouterPolicy, action: str = "hybrid"):
    """Return (test_score, chosen_action) for one task under the given policy.

    action in {select, fuse, hybrid}. `hybrid` is the ELARA-U default: select only
    when one expert is clearly+confidently best, otherwise fuse; fall back to
    rank-mean when no expert clears the negative-transfer guard.
    """
    f = reliability_features(s_val, y_val)
    auc = f["val_auc"]
    if f["best_auc"] < policy.guard:
        return _rank_mean(s_test), "fallback"
    if action == "select":
        return select(s_test, auc), "select"
    if action == "fuse":
        return fuse(s_test, auc), "fuse"
    # hybrid
    if f["best_auc"] >= policy.conf and f["gap"] >= policy.gap:
        return select(s_test, auc), "select"
    return fuse(s_test, auc), "fuse"
