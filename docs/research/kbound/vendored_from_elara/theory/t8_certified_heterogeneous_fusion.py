"""T8: Certified heterogeneous fusion (CHF) under category / batch shift.

When per-sample reliability is heterogeneous (low drift coherence), a single
reliability-gated head is high-variance. CHF uses a validation-only certificate
to select among:
  (i) coherence-gated RGA+ path,
  (ii) frozen SAR test-time adaptation,
  (iii) a regularized stack of {SAR, RGA+, static} with optional reliability features.

Finite-sample selection is justified by extending the switching certificate:
the stack is fit only on validation; test uses the locked linear stack or route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Vendored inline from ELARA's uais.fusion.attention.gate_decision_rule so this
# certificate is self-contained (K-Bound must not import from the ELARA tree).
def per_sample_mean_reliability(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Mean reliability over each sample's *present* domains (nan if none)."""
    weights = np.asarray(reliability_weights, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    if weights.shape != masks.shape:
        raise ValueError("reliability_weights and masks must have the same shape.")
    if weights.ndim != 2:
        raise ValueError("reliability_weights must be a 2-D [N, D] array.")
    present = ~masks
    present_counts = present.sum(axis=1)
    masked_weights = np.where(present, weights, 0.0)
    sums = masked_weights.sum(axis=1)
    out = np.full(weights.shape[0], np.nan, dtype=float)
    nonempty = present_counts > 0
    out[nonempty] = sums[nonempty] / present_counts[nonempty]
    return out


def drift_coherence(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
) -> float:
    """Coherence of the drift signal across a batch, in [0, 1]."""
    r = per_sample_mean_reliability(reliability_weights, masks)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.clip(1.0 - 2.0 * float(np.std(r)), 0.0, 1.0))


@dataclass
class CHFCertificate:
    """Validation-only CHF selection record."""

    route: str
    coherence_threshold: float
    stack_weights: np.ndarray | None
    convex_weight_sar: float | None
    val_roc_auc: float
    certified: bool

    def as_dict(self) -> dict:
        return {
            "route": self.route,
            "coherence_threshold": float(self.coherence_threshold),
            "stack_weights": None if self.stack_weights is None else self.stack_weights.tolist(),
            "convex_weight_sar": None if self.convex_weight_sar is None else float(self.convex_weight_sar),
            "val_roc_auc": float(self.val_roc_auc),
            "certified": bool(self.certified),
        }


def batch_coherence_scores(
    reliability_weights: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-sample coherence proxy: batch coherence repeated per sample in batch."""
    n = reliability_weights.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        c = drift_coherence(reliability_weights[start:end], masks[start:end])
        out[start:end] = float(c) if np.isfinite(c) else 0.0
    return out


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.5


def select_chf_route_on_validation(
    labels: np.ndarray,
    *,
    sar_probs: np.ndarray,
    rga_probs: np.ndarray,
    static_probs: np.ndarray,
    coherence_per_sample: np.ndarray,
    reliability_mean: np.ndarray,
    switching_certified: bool,
    coherence_threshold: float = 0.5,
) -> CHFCertificate:
    """Pick CHF route using validation AUROC only."""
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(labels, dtype=int)
    candidates: list[tuple[str, np.ndarray, dict]] = []

    # Route 1: SAR only
    candidates.append(("sar_only", sar_probs, {}))

    # Route 2: RGA+ only
    candidates.append(("rga_plus_only", rga_probs, {}))

    # Route 3: coherence-gated blend (high coherence batch -> rga else sar per sample)
    coh = np.asarray(coherence_per_sample, dtype=float)
    rel = np.asarray(reliability_mean, dtype=float)
    gated = np.where(coh >= coherence_threshold, rga_probs, sar_probs)
    candidates.append(("coherence_gated", gated, {"coherence_threshold": coherence_threshold}))

    # Route 4: convex combination grid
    best_w = 0.5
    best_convex = sar_probs
    best_convex_auc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * sar_probs + (1.0 - w) * rga_probs
        auc = _safe_auc(labels, blend)
        if auc > best_convex_auc:
            best_convex_auc = auc
            best_w = float(w)
            best_convex = blend
    candidates.append(("convex_sar_rga", best_convex, {"convex_weight_sar": best_w}))

    # Route 5: regularized stack
    x_stack = np.column_stack(
        [
            sar_probs,
            rga_probs,
            static_probs,
            rel,
            coh,
        ]
    ).astype(np.float64)
    stack = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=0)
    try:
        stack.fit(x_stack, labels)
        stack_probs = stack.predict_proba(x_stack)[:, 1].astype(np.float32)
        candidates.append(("stack5", stack_probs, {"model": stack}))
    except ValueError:
        pass

    # When validation is saturated (many routes AUROC=1), prefer routes that generalize under shift.
    route_priority = ("stack5", "convex_sar_rga", "sar_only", "coherence_gated", "rga_plus_only")
    priority_rank = {name: i for i, name in enumerate(route_priority)}

    best_route = "sar_only"
    best_auc = -1.0
    best_meta: dict = {}
    for name, probs, meta in candidates:
        auc = _safe_auc(labels, probs)
        rank = priority_rank.get(name, 99)
        best_rank = priority_rank.get(best_route, 99)
        if auc > best_auc + 1e-6 or (
            math.isclose(auc, best_auc, rel_tol=0.0, abs_tol=1e-6) and rank < best_rank
        ):
            best_auc = auc
            best_route = name
            best_meta = meta

    stack_weights = None
    convex_w = None
    if best_route == "stack5" and "model" in best_meta:
        stack_weights = np.concatenate([best_meta["model"].coef_.reshape(-1), best_meta["model"].intercept_])
    if best_route == "convex_sar_rga":
        convex_w = float(best_meta.get("convex_weight_sar", 0.5))

    return CHFCertificate(
        route=best_route,
        coherence_threshold=float(coherence_threshold),
        stack_weights=stack_weights,
        convex_weight_sar=convex_w,
        val_roc_auc=float(best_auc),
        certified=bool(switching_certified),
    )


def predict_chf_shift_aware(
    certificate: CHFCertificate,
    *,
    sar_probs: np.ndarray,
    rga_probs: np.ndarray,
    static_probs: np.ndarray,
    coherence_per_sample: np.ndarray,
    reliability_mean: np.ndarray,
    stack_model=None,
    eval_categories: np.ndarray | None = None,
    train_categories: np.ndarray | None = None,
) -> np.ndarray:
    """On held-out categories (unknown at train), default to SAR."""
    if eval_categories is not None and train_categories is not None:
        known = set(np.asarray(train_categories).astype(str).tolist())
        cats = np.asarray(eval_categories).astype(str)
        if known and float(np.mean([c not in known for c in cats])) >= 0.99:
            return np.asarray(sar_probs, dtype=np.float32)
    return predict_chf(
        certificate,
        sar_probs=sar_probs,
        rga_probs=rga_probs,
        static_probs=static_probs,
        coherence_per_sample=coherence_per_sample,
        reliability_mean=reliability_mean,
        stack_model=stack_model,
    )


def predict_chf(
    certificate: CHFCertificate,
    *,
    sar_probs: np.ndarray,
    rga_probs: np.ndarray,
    static_probs: np.ndarray,
    coherence_per_sample: np.ndarray,
    reliability_mean: np.ndarray,
    stack_model=None,
) -> np.ndarray:
    """Apply a locked CHF certificate to test (or val) probabilities."""
    route = certificate.route
    if route == "sar_only":
        return np.asarray(sar_probs, dtype=np.float32)
    if route == "rga_plus_only":
        return np.asarray(rga_probs, dtype=np.float32)
    if route == "coherence_gated":
        coh = np.asarray(coherence_per_sample, dtype=float)
        return np.where(
            coh >= certificate.coherence_threshold,
            rga_probs,
            sar_probs,
        ).astype(np.float32)
    if route == "convex_sar_rga":
        w = float(certificate.convex_weight_sar if certificate.convex_weight_sar is not None else 0.5)
        return (w * sar_probs + (1.0 - w) * rga_probs).astype(np.float32)
    if route == "stack5" and stack_model is not None:
        x = np.column_stack(
            [
                sar_probs,
                rga_probs,
                static_probs,
                reliability_mean,
                coherence_per_sample,
            ]
        ).astype(np.float64)
        return stack_model.predict_proba(x)[:, 1].astype(np.float32)
    return np.asarray(sar_probs, dtype=np.float32)
