"""Validation-trained reliability router over RGA and strong fusion baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1.0 - 1e-12)


def _prediction_matrix(predictions: dict[str, np.ndarray], method_names: list[str]) -> np.ndarray:
    columns = []
    n: int | None = None
    for name in method_names:
        if name not in predictions:
            raise KeyError(f"Missing prediction column for '{name}'.")
        values = _clip_probs(predictions[name])
        if n is None:
            n = len(values)
        elif len(values) != n:
            raise ValueError("All prediction arrays must have the same length.")
        columns.append(values)
    if not columns:
        raise ValueError("At least one prediction column is required.")
    return np.column_stack(columns)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def _safe_logloss(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(log_loss(labels, _clip_probs(probs), labels=[0, 1]))
    except ValueError:
        return float("inf")


def _safe_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(average_precision_score(labels, probs))
    except ValueError:
        return 0.0


def _best_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    best = 0.0
    for threshold in np.unique(probs):
        score = f1_score(labels, (probs >= threshold).astype(int), zero_division=0)
        best = max(best, float(score))
    return best


def _safe_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(brier_score_loss(labels, _clip_probs(probs)))
    except ValueError:
        return 1.0


def _validation_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(np.nan_to_num(_safe_auc(labels, probs), nan=0.5)),
        "pr_auc": _safe_pr_auc(labels, probs),
        "f1": _best_f1(labels, probs),
        "brier": _safe_brier(labels, probs),
        "log_loss": _safe_logloss(labels, probs),
    }


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    metric = (selection_metric or "roc_auc").strip().lower()
    if metric in {"roc_auc", "pr_auc", "f1"}:
        return float(metrics[metric])
    if metric in {"roc_pr_f1", "balanced", "multi_objective"}:
        return float((metrics["roc_auc"] + metrics["pr_auc"] + metrics["f1"]) / 3.0)
    if metric in {"calibrated", "roc_pr_f1_calibrated"}:
        return float((metrics["roc_auc"] + metrics["pr_auc"] + metrics["f1"] + (1.0 - metrics["brier"])) / 4.0)
    raise ValueError(f"Unknown RGA+ router selection metric: {selection_metric}")


@dataclass
class RGAMetaRouter:
    """Small validation-trained router used as an RGA+ modeling candidate.

    The router never sees test labels. It selects among base predictions,
    validation-trained logistic stacking, and simple top-k validation ensembles
    using only validation labels.
    """

    method_names: list[str]
    selected_candidate: str
    candidate_scores: dict[str, float]
    candidate_metric_scores: dict[str, dict[str, float]] | None = None
    logistic_model: LogisticRegression | None = None
    scaler: StandardScaler | None = None
    ensemble_methods: list[str] | None = None

    def predict_proba(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.selected_candidate.startswith("base:"):
            name = self.selected_candidate.split(":", 1)[1]
            return _clip_probs(predictions[name])
        if self.selected_candidate.startswith("mean:"):
            names = self.ensemble_methods or self.selected_candidate.split(":", 1)[1].split("+")
            stacked = np.column_stack([_clip_probs(predictions[name]) for name in names])
            return _clip_probs(stacked.mean(axis=1))
        if self.selected_candidate == "logistic_stack":
            if self.logistic_model is None or self.scaler is None:
                raise RuntimeError("Logistic stack was selected but is not fitted.")
            matrix = _prediction_matrix(predictions, self.method_names)
            return _clip_probs(self.logistic_model.predict_proba(self.scaler.transform(matrix))[:, 1])
        raise RuntimeError(f"Unknown router candidate: {self.selected_candidate}")


def _candidate_quality(
    labels: np.ndarray,
    probs: np.ndarray,
    selection_metric: str,
) -> tuple[float, float, dict[str, float]]:
    """Return sortable validation quality plus named validation metrics."""
    metrics = _validation_metrics(labels, probs)
    return _selection_score(metrics, selection_metric), -metrics["log_loss"], metrics


def _fit_logistic(
    matrix: np.ndarray, labels: np.ndarray, random_seed: int
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=random_seed,
    )
    model.fit(scaled, labels)
    return scaler, model


def _single_class_fallback(base_scores: dict[str, tuple[float, float]]) -> str:
    for preferred in ("base:craf_attention", "base:static_attention"):
        if preferred in base_scores:
            return preferred
    return max(base_scores, key=lambda name: base_scores[name])


def fit_rga_meta_router(
    val_predictions: dict[str, np.ndarray],
    val_labels: np.ndarray,
    random_seed: int = 42,
    selection_metric: str = "roc_auc",
) -> RGAMetaRouter:
    """Fit an honest validation-only RGA+ router over model probabilities."""
    labels = np.asarray(val_labels, dtype=np.int64)
    method_names = list(val_predictions)
    matrix = _prediction_matrix(val_predictions, method_names)
    if matrix.shape[0] != len(labels):
        raise ValueError("Prediction rows and validation labels must have the same length.")

    base_scores: dict[str, tuple[float, float, dict[str, float]]] = {
        f"base:{name}": _candidate_quality(labels, matrix[:, idx], selection_metric)
        for idx, name in enumerate(method_names)
    }
    candidate_scores: dict[str, float] = {name: score[0] for name, score in base_scores.items()}
    candidate_metric_scores: dict[str, dict[str, float]] = {name: score[2] for name, score in base_scores.items()}

    if len(np.unique(labels)) < 2 or len(labels) < 6:
        selected = _single_class_fallback(base_scores)
        return RGAMetaRouter(
            method_names=method_names,
            selected_candidate=selected,
            candidate_scores=candidate_scores,
            candidate_metric_scores=candidate_metric_scores,
        )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=random_seed)
    try:
        train_idx, select_idx = next(splitter.split(matrix, labels))
    except ValueError:
        selected = _single_class_fallback(base_scores)
        return RGAMetaRouter(method_names, selected, candidate_scores, candidate_metric_scores)

    select_labels = labels[select_idx]
    select_candidates: dict[str, tuple[np.ndarray, tuple[float, float, dict[str, float]]]] = {}
    for idx, name in enumerate(method_names):
        probs = matrix[select_idx, idx]
        select_candidates[f"base:{name}"] = (probs, _candidate_quality(select_labels, probs, selection_metric))

    if len(np.unique(labels[train_idx])) >= 2:
        scaler, model = _fit_logistic(matrix[train_idx], labels[train_idx], random_seed)
        logistic_probs = model.predict_proba(scaler.transform(matrix[select_idx]))[:, 1]
        select_candidates["logistic_stack"] = (
            logistic_probs,
            _candidate_quality(select_labels, logistic_probs, selection_metric),
        )

        train_scores = [
            (name, _safe_auc(labels[train_idx], matrix[train_idx, idx])) for idx, name in enumerate(method_names)
        ]
        ranked = [
            name
            for name, score in sorted(
                train_scores, key=lambda item: (np.nan_to_num(item[1], nan=0.5), item[0]), reverse=True
            )
        ]
        for k in (2, 3):
            top = ranked[: min(k, len(ranked))]
            if len(top) < 2:
                continue
            top_indices = [method_names.index(name) for name in top]
            probs = matrix[select_idx][:, top_indices].mean(axis=1)
            select_candidates[f"mean:{'+'.join(top)}"] = (
                probs,
                _candidate_quality(select_labels, probs, selection_metric),
            )

    selected = max(select_candidates, key=lambda name: select_candidates[name][1])
    candidate_scores.update({name: quality[0] for name, (_probs, quality) in select_candidates.items()})
    candidate_metric_scores.update({name: quality[2] for name, (_probs, quality) in select_candidates.items()})

    if selected == "logistic_stack":
        final_scaler, final_model = _fit_logistic(matrix, labels, random_seed)
        return RGAMetaRouter(
            method_names=method_names,
            selected_candidate=selected,
            candidate_scores=candidate_scores,
            candidate_metric_scores=candidate_metric_scores,
            logistic_model=final_model,
            scaler=final_scaler,
        )
    if selected.startswith("mean:"):
        return RGAMetaRouter(
            method_names=method_names,
            selected_candidate=selected,
            candidate_scores=candidate_scores,
            candidate_metric_scores=candidate_metric_scores,
            ensemble_methods=selected.split(":", 1)[1].split("+"),
        )
    return RGAMetaRouter(method_names, selected, candidate_scores, candidate_metric_scores)


__all__ = ["RGAMetaRouter", "fit_rga_meta_router"]
