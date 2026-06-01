"""Cross-modal anomaly fusion methods (Gate-E research).

Motivation. On clean external transfer a parameter-free confidence-weighted
mean (CW) beats reliability-weighted RGA, because BOTH are convex combinations
of the two modality scores -- and a convex combination DILUTES an anomaly that
is strong in only one modality. To beat CW on clean data we need fusion rules
that capture *cross-modal interaction*: information in the joint (s_rgb, s_depth)
that no weighted average of the marginals can express.

This module implements several principled, low-capacity cross-modal rules and a
validation-only selector. None peeks at the test fold; the selector chooses the
single rule with the best VALIDATION metric, which is then reported once on test.

Rules (all operate on per-sample (s_rgb, s_depth) in [0,1]):
  - cw                 : confidence-weighted mean (the baseline to beat)
  - max                : noisy-OR-style max (an anomaly in either modality fires)
  - softor             : probabilistic OR  1-(1-a)(1-b)  (smooth max)
  - product            : geometric/AND emphasis (both must agree)
  - rank_fusion        : average of per-modality rank-normalised scores
  - logistic_xmodal    : logistic regression on [s_rgb, s_depth, s_rgb*s_depth,
                         |s_rgb-s_depth|]  -- the interaction terms are the point
  - copula_lite        : CW + a learned weight on the disagreement term
                         |s_rgb - s_depth| (cross-modal residual)

The logistic / copula rules are fit on VALIDATION ONLY (labels present), so they
are honest held-out-transfer methods: trained on the external set's own
validation split, evaluated on its test split, never the other way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

__all__ = ["FUSION_RULES", "select_and_eval", "FusionResult"]


def _rank_norm(x: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] by rank (robust to per-modality scale/shift)."""
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def cw(a, b, **_):
    """Confidence-weighted (here equal-confidence) mean -- the baseline."""
    return 0.5 * (a + b)


def fuse_max(a, b, **_):
    """Max fusion: an anomaly strong in EITHER modality survives (no dilution)."""
    return np.maximum(a, b)


def softor(a, b, **_):
    """Probabilistic OR (smooth max): 1-(1-a)(1-b)."""
    return 1.0 - (1.0 - a) * (1.0 - b)


def product(a, b, **_):
    """AND-emphasis: sqrt(a*b) fires only when both modalities agree."""
    return np.sqrt(np.clip(a, 0, 1) * np.clip(b, 0, 1))


def rank_fusion(a, b, **_):
    """Average of rank-normalised modality scores."""
    return 0.5 * (_rank_norm(a) + _rank_norm(b))


@dataclass
class _LogisticXModal:
    """Logistic regression over [a, b, a*b, |a-b|] -- the last two are the
    cross-modal interaction features a weighted average cannot represent."""
    model: LogisticRegression

    @staticmethod
    def features(a, b):
        return np.column_stack([a, b, a * b, np.abs(a - b)])

    def __call__(self, a, b, **_):
        return self.model.predict_proba(self.features(a, b))[:, 1]


@dataclass
class _CopulaLite:
    """CW plus a learned weight on the cross-modal disagreement |a-b|.
    score = clip( 0.5(a+b) + w * |a-b| ). w>0 boosts single-modality anomalies."""
    w: float

    def __call__(self, a, b, **_):
        return np.clip(0.5 * (a + b) + self.w * np.abs(a - b), 0.0, 1.0)


# parameter-free rules available without fitting
FUSION_RULES = {
    "cw": cw, "max": fuse_max, "softor": softor,
    "product": product, "rank_fusion": rank_fusion,
}


@dataclass
class FusionResult:
    selected_rule: str
    val_auroc_selected: float
    test_auroc_selected: float
    test_auroc_cw: float
    delta_vs_cw: float
    all_val_auroc: dict
    all_test_auroc: dict


def _fit_logistic(va, vb, vy, seed=0) -> _LogisticXModal:
    X = _LogisticXModal.features(va, vb)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    m.fit(X, vy)
    return _LogisticXModal(m)


def _fit_copula(va, vb, vy) -> _CopulaLite:
    """Select disagreement weight w on validation AUROC over a small grid."""
    best_w, best_au = 0.0, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        au = roc_auc_score(vy, _CopulaLite(w)(va, vb))
        if au > best_au:
            best_au, best_w = au, w
    return _CopulaLite(best_w)


def select_and_eval(
    val_rgb, val_depth, val_y,
    test_rgb, test_depth, test_y,
    seed: int = 0,
) -> FusionResult:
    """Fit fitted rules on validation, select the best-VALIDATION rule, and
    report it ONCE on test. CW test AUROC is reported as the baseline to beat."""
    va, vb, vy = map(np.asarray, (val_rgb, val_depth, val_y))
    ta, tb, ty = map(np.asarray, (test_rgb, test_depth, test_y))

    rules = dict(FUSION_RULES)
    rules["logistic_xmodal"] = _fit_logistic(va, vb, vy, seed=seed)
    rules["copula_lite"] = _fit_copula(va, vb, vy)

    val_au = {name: float(roc_auc_score(vy, fn(va, vb))) for name, fn in rules.items()}
    test_au = {name: float(roc_auc_score(ty, fn(ta, tb))) for name, fn in rules.items()}

    # selection is VALIDATION-ONLY (honest); ties broken by rule name order
    selected = max(val_au, key=lambda k: (val_au[k], -list(rules).index(k)))
    return FusionResult(
        selected_rule=selected,
        val_auroc_selected=val_au[selected],
        test_auroc_selected=test_au[selected],
        test_auroc_cw=test_au["cw"],
        delta_vs_cw=test_au[selected] - test_au["cw"],
        all_val_auroc=val_au,
        all_test_auroc=test_au,
    )
