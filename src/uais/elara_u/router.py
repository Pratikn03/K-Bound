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
from typing import cast

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
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
    ranked = np.argsort(np.argsort(S, axis=0), axis=0) / max(S.shape[0] - 1, 1)
    return cast(np.ndarray, ranked.mean(1))


def fuse(S: np.ndarray, auc: np.ndarray, floor: float = 0.55) -> np.ndarray:
    """Reliability-weighted fusion over non-degenerate experts."""
    keep = auc >= floor
    if not keep.any():
        return _rank_mean(S)                            # fallback: all weak
    w = _reliab_weights(auc[keep], floor=0.5)
    return cast(np.ndarray, S[:, keep] @ w)


def select(S: np.ndarray, auc: np.ndarray) -> np.ndarray:
    return S[:, int(np.argmax(auc))]


@dataclass
class RouterPolicy:
    """Thresholds fit on meta-train tasks; applied unchanged to held-out tasks."""
    conf: float = 0.70      # min best-AUROC to trust a single expert
    gap: float = 0.05       # min lead of best over 2nd-best to select
    guard: float = 0.55     # below this best-AUROC -> negative-transfer fallback
    drift_threshold: float = 0.25 # KS statistic above this -> drop detector as drifted


def route(s_val, y_val, s_test, policy: RouterPolicy, action: str = "hybrid"):
    """Return (test_score, chosen_action) for one task under the given policy.

    action in {select, fuse, hybrid}. `hybrid` is the ELARA-U default: select only
    when one expert is clearly+confidently best, otherwise fuse; fall back to
    rank-mean when no expert clears the negative-transfer guard.
    """
    if action == "hybrid":
        return super_route(s_val, y_val, s_test, policy)
        
    f = reliability_features(s_val, y_val)
    auc = f["val_auc"]
    
    # Calculate drift per detector using Kolmogorov-Smirnov test against validation
    M = s_test.shape[1]
    drift = np.array([ks_2samp(s_val[:, m], s_test[:, m]).statistic for m in range(M)])
    
    # Apply degenerate-channel and drift guards
    # We drop any detector that is saturated (std < 1e-4), sign-inverted (auc < 0.5),
    # or drifted (drift > threshold).
    # Threshold uses the statistical KS critical value at alpha=0.05, with drift_threshold as a floor.
    n_val, n_test = len(s_val), len(s_test)
    critical_val = 1.36 * np.sqrt((n_val + n_test) / (n_val * n_test + 1e-9))
    threshold = max(policy.drift_threshold, critical_val)
    
    active = np.ones(M, dtype=bool)
    for m in range(M):
        std_val = np.std(s_val[:, m])
        if std_val < 1e-4 or auc[m] < 0.50 or drift[m] > threshold:
            active[m] = False

    # If all detectors are dead/drifted, trigger fallback action
    if not active.any():
        return _rank_mean(s_test), "fallback"
        
    # Filter AUC to active detectors
    active_auc = np.where(active, auc, -1.0)
    best_idx = int(np.argmax(active_auc))
    best_auc = active_auc[best_idx]
    
    if best_auc < policy.guard:
        return _rank_mean(s_test), "fallback"
        
    if action == "select":
        return s_test[:, best_idx], "select"
        
    # Calculate gap to the second-best active detector
    sorted_active_auc = np.sort(active_auc[active])
    gap = float(sorted_active_auc[-1] - sorted_active_auc[-2]) if len(sorted_active_auc) > 1 else 1.0
    
    if action == "fuse":
        # Fuse only active detectors
        w = _reliab_weights(auc[active], floor=0.5)
        fused = s_test[:, active] @ w
        return fused, "fuse"
        
    # Hybrid action: select if confidently leading, else fuse
    if best_auc >= policy.conf and gap >= policy.gap:
        return s_test[:, best_idx], "select"
        
    w = _reliab_weights(auc[active], floor=0.5)
    fused = s_test[:, active] @ w
    return fused, "fuse"


def super_route(s_val, y_val, s_test, policy: "RouterPolicy"):
    """ELARA-U super-selection: per-sample reliability-gated stacking.

    The path to beating auto-select. Auto-select picks one detector per *dataset*
    (near-oracle at that level); a per-*sample* combiner can exceed the best single
    detector by learning, from validation only, how to combine experts sample-by-
    sample. We drop saturated channels (no test labels), then fit a regularized
    logistic meta-learner on the relative ranks of active validation detector scores
    and apply it to the test ranks. Falls back to rank-mean when no channel survives
    or validation has a single class.
    """
    y = np.asarray(y_val).astype(int)
    M = s_test.shape[1]
    if len(np.unique(y)) < 2:
        return _rank_mean(s_test), "fallback"
    
    # Only drop saturated channels
    active = np.array([np.std(s_val[:, m]) >= 1e-4 for m in range(M)])
    if not active.any():
        return _rank_mean(s_test), "fallback"
    
    # Convert scores to relative ranks to be robust to score shift
    r_val = np.zeros_like(s_val)
    r_test = np.zeros_like(s_test)
    for m in range(M):
        r_val[:, m] = np.argsort(np.argsort(s_val[:, m])) / max(len(s_val) - 1, 1)
        r_test[:, m] = np.argsort(np.argsort(s_test[:, m])) / max(len(s_test) - 1, 1)
        
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(r_val[:, active], y)
    return clf.predict_proba(r_test[:, active])[:, 1], "stack"
