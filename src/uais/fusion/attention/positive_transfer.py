"""Natural clean-transfer candidate rules and statistics.

This module is deliberately small: selection is validation-only, endpoint
checks are explicit, and opened development datasets cannot become official
evidence without the confirmatory runner marking them fresh/unopened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

__all__ = [
    "SelectionResult",
    "candidate_scores",
    "coendpoint_pass",
    "fit_candidate_on_validation",
    "paired_auc_bootstrap",
    "score_selected_candidate",
    "select_candidate_on_validation",
]


SIMPLE_RULE_ORDER = ("cw", "rank_cw", "product", "softor", "max")
RULE_PRIORITY = (
    "residual_stack",
    "sar_cw_stack",
    "rank_cw",
    "product",
    "cw",
    "softor",
    "max",
)


@dataclass(frozen=True)
class SelectionResult:
    selected_rule: str
    val_auroc_selected: float
    all_val_auroc: dict[str, float]
    selection_split: str = "validation"
    used_test_labels: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_rule": self.selected_rule,
            "val_auroc_selected": float(self.val_auroc_selected),
            "all_val_auroc": {k: float(v) for k, v in self.all_val_auroc.items()},
            "selection_split": self.selection_split,
            "used_test_labels": bool(self.used_test_labels),
            "params": self.params,
        }


def _as_float(x: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2 or not np.isfinite(score).all():
        return 0.5
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return 0.5


def _rank_norm(x: np.ndarray) -> np.ndarray:
    x = _as_float(x)
    if x.size <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(float) / float(max(len(x) - 1, 1))


def _cw(rgb: np.ndarray, depth: np.ndarray, rgb_conf: np.ndarray | None, depth_conf: np.ndarray | None) -> np.ndarray:
    if rgb_conf is None or depth_conf is None:
        return 0.5 * (rgb + depth)
    denom = rgb_conf + depth_conf + 1e-12
    return (rgb_conf * rgb + depth_conf * depth) / denom


def candidate_scores(
    rgb: np.ndarray | list[float],
    depth: np.ndarray | list[float],
    *,
    sar: np.ndarray | list[float] | None = None,
    rgb_confidence: np.ndarray | list[float] | None = None,
    depth_confidence: np.ndarray | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Return finite low-capacity natural-transfer candidate score vectors."""
    a = _as_float(rgb)
    b = _as_float(depth)
    rc = None if rgb_confidence is None else _as_float(rgb_confidence)
    dc = None if depth_confidence is None else _as_float(depth_confidence)
    cw = _cw(a, b, rc, dc)
    scores = {
        "cw": cw,
        "rank_cw": 0.5 * (_rank_norm(a) + _rank_norm(b)),
        "product": np.sqrt(np.clip(a, 0.0, 1.0) * np.clip(b, 0.0, 1.0)),
        "max": np.maximum(a, b),
        "softor": 1.0 - (1.0 - np.clip(a, 0.0, 1.0)) * (1.0 - np.clip(b, 0.0, 1.0)),
    }
    if sar is not None:
        s = _as_float(sar)
        scores["sar_cw_equal"] = 0.5 * (s + cw)
    return {k: np.nan_to_num(v, nan=0.5, posinf=1.0, neginf=0.0) for k, v in scores.items()}


def _base_features(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    sar: np.ndarray | None = None,
    rgb_confidence: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scores = candidate_scores(
        rgb,
        depth,
        sar=sar,
        rgb_confidence=rgb_confidence,
        depth_confidence=depth_confidence,
    )
    cols = [
        scores["cw"],
        scores["rank_cw"],
        scores["product"],
        np.abs(_as_float(rgb) - _as_float(depth)),
    ]
    if sar is not None:
        cols.append(_as_float(sar))
    if rgb_confidence is not None and depth_confidence is not None:
        rc = _as_float(rgb_confidence)
        dc = _as_float(depth_confidence)
        cols.extend([rc, dc, np.abs(rc - dc)])
    return np.column_stack(cols).astype(float), scores


def _fit_residual_stack(y: np.ndarray, x: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y)) < 2 or x.shape[0] < 8:
        return None
    model = LogisticRegression(C=0.2, class_weight="balanced", max_iter=1000, random_state=0)
    try:
        model.fit(x, y)
    except ValueError:
        return None
    return model


def _fit_sar_cw_weight(y: np.ndarray, sar: np.ndarray, cw: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    best_w = 0.5
    best_auc = -1.0
    for w in np.linspace(0.0, 1.0, 21):
        score = w * sar + (1.0 - w) * cw
        auc = _safe_auc(y, score)
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)
    return best_w


def select_candidate_on_validation(
    val_y: np.ndarray | list[int],
    val_rgb: np.ndarray | list[float],
    val_depth: np.ndarray | list[float],
    *,
    val_sar: np.ndarray | list[float] | None = None,
    val_rgb_confidence: np.ndarray | list[float] | None = None,
    val_depth_confidence: np.ndarray | list[float] | None = None,
) -> SelectionResult:
    """Select one candidate using validation labels only."""
    y = np.asarray(val_y, dtype=int)
    rgb = _as_float(val_rgb)
    depth = _as_float(val_depth)
    sar = None if val_sar is None else _as_float(val_sar)
    rc = None if val_rgb_confidence is None else _as_float(val_rgb_confidence)
    dc = None if val_depth_confidence is None else _as_float(val_depth_confidence)

    x, scores = _base_features(rgb, depth, sar=sar, rgb_confidence=rc, depth_confidence=dc)
    all_auc = {name: _safe_auc(y, score) for name, score in scores.items() if name in SIMPLE_RULE_ORDER}
    params: dict[str, Any] = {}

    if sar is not None:
        w = _fit_sar_cw_weight(y, sar, scores["cw"])
        if w is not None:
            score = w * sar + (1.0 - w) * scores["cw"]
            all_auc["sar_cw_stack"] = _safe_auc(y, score)
            params["sar_cw_stack"] = {"weight_sar": w}

    model = _fit_residual_stack(y, x)
    if model is not None:
        all_auc["residual_stack"] = _safe_auc(y, model.predict_proba(x)[:, 1])
        params["residual_stack"] = {
            "coef": model.coef_.reshape(-1).astype(float).tolist(),
            "intercept": model.intercept_.astype(float).tolist(),
        }

    priority = {name: i for i, name in enumerate(RULE_PRIORITY)}
    selected = max(all_auc, key=lambda k: (all_auc[k], -priority.get(k, 999)))
    return SelectionResult(
        selected_rule=selected,
        val_auroc_selected=all_auc[selected],
        all_val_auroc=all_auc,
        params=params,
    )


def fit_candidate_on_validation(
    val_y: np.ndarray | list[int],
    val_rgb: np.ndarray | list[float],
    val_depth: np.ndarray | list[float],
    **kwargs: Any,
) -> SelectionResult:
    return select_candidate_on_validation(val_y, val_rgb, val_depth, **kwargs)


def _linear_logistic_score(x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    coef = np.asarray(params["coef"], dtype=float)
    intercept = float(np.asarray(params["intercept"], dtype=float).reshape(-1)[0])
    z = x @ coef + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def score_selected_candidate(
    selection: SelectionResult | dict[str, Any],
    rgb: np.ndarray | list[float],
    depth: np.ndarray | list[float],
    *,
    sar: np.ndarray | list[float] | None = None,
    rgb_confidence: np.ndarray | list[float] | None = None,
    depth_confidence: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Apply a validation-selected candidate to a held-out split."""
    if isinstance(selection, SelectionResult):
        rule = selection.selected_rule
        params = selection.params
    else:
        rule = str(selection["selected_rule"])
        params = dict(selection.get("params") or {})
    sar_arr = None if sar is None else _as_float(sar)
    rc = None if rgb_confidence is None else _as_float(rgb_confidence)
    dc = None if depth_confidence is None else _as_float(depth_confidence)
    x, scores = _base_features(_as_float(rgb), _as_float(depth), sar=sar_arr, rgb_confidence=rc, depth_confidence=dc)
    if rule in scores:
        return scores[rule]
    if rule == "sar_cw_stack":
        if sar_arr is None:
            raise ValueError("sar_cw_stack requires sar scores")
        w = float(params["sar_cw_stack"]["weight_sar"])
        return w * sar_arr + (1.0 - w) * scores["cw"]
    if rule == "residual_stack":
        return _linear_logistic_score(x, params["residual_stack"])
    raise ValueError(f"unknown positive-transfer rule: {rule}")


def paired_auc_bootstrap(
    labels: np.ndarray | list[int],
    candidate: np.ndarray | list[float],
    comparator: np.ndarray | list[float],
    *,
    n_iter: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    a = _as_float(candidate)
    b = _as_float(comparator)
    if y.size != a.size or y.size != b.size:
        raise ValueError("labels, candidate, and comparator must have matching length")
    if y.size == 0 or len(np.unique(y)) < 2:
        return {"valid": False, "reason": "degenerate_labels", "delta": 0.0, "ci95": [0.0, 0.0]}

    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(int(n_iter)):
        idx = rng.integers(0, y.size, y.size)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])))
    if len(deltas) < 100:
        return {"valid": False, "reason": "insufficient_bootstrap_replicates", "delta": point, "ci95": [0.0, 0.0]}
    arr = np.asarray(deltas, dtype=float)
    return {
        "valid": True,
        "delta": point,
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "candidate_auc": float(roc_auc_score(y, a)),
        "comparator_auc": float(roc_auc_score(y, b)),
        "n": int(y.size),
        "bootstrap_iterations": int(n_iter),
    }


def coendpoint_pass(
    stat: dict[str, Any],
    *,
    minimum_practical_delta: float,
    ci_low_must_be_gt: float,
) -> bool:
    ci = stat.get("ci95") or [0.0, 0.0]
    return bool(
        stat.get("valid", True)
        and float(stat.get("delta", 0.0)) >= float(minimum_practical_delta)
        and float(ci[0]) > float(ci_low_must_be_gt)
    )

