"""Validation-selected reliability-boosted fusion head for ELARA/RGA+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _flatten_with_mask(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    n, d, f = features.shape
    flat = features.astype(np.float32, copy=True)
    for domain_idx in range(d):
        flat[:, domain_idx, :][masks[:, domain_idx]] = 0.5
    return np.concatenate([flat.reshape(n, d * f), masks.astype(np.float32)], axis=1)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.5


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
        return float(brier_score_loss(labels, np.clip(probs, 1e-12, 1.0 - 1e-12)))
    except ValueError:
        return 1.0


def _validation_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _safe_auc(labels, probs),
        "pr_auc": _safe_pr_auc(labels, probs),
        "f1": _best_f1(labels, probs),
        "brier": _safe_brier(labels, probs),
    }


def _selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    metric = (selection_metric or "roc_auc").strip().lower()
    if metric in {"roc_auc", "pr_auc", "f1"}:
        return float(metrics[metric])
    if metric in {"roc_pr_f1", "balanced", "multi_objective"}:
        return float((metrics["roc_auc"] + metrics["pr_auc"] + metrics["f1"]) / 3.0)
    if metric in {"calibrated", "roc_pr_f1_calibrated"}:
        return float((metrics["roc_auc"] + metrics["pr_auc"] + metrics["f1"] + (1.0 - metrics["brier"])) / 4.0)
    raise ValueError(f"Unknown RGA+ selection metric: {selection_metric}")


def _masked_mean(values: np.ndarray, present: np.ndarray, fill: float) -> np.ndarray:
    count = present.sum(axis=1)
    total = np.where(present, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), fill)


def _masked_extreme(values: np.ndarray, present: np.ndarray, fill: float, is_max: bool) -> np.ndarray:
    sentinel = -np.inf if is_max else np.inf
    reduced = np.where(present, values, sentinel)
    out = reduced.max(axis=1) if is_max else reduced.min(axis=1)
    return np.where(np.isfinite(out), out, fill)


def _masked_std(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    mean = _masked_mean(values, present, 0.0)
    count = present.sum(axis=1)
    centered = np.where(present, values - mean[:, None], 0.0)
    var = (centered * centered).sum(axis=1) / np.maximum(count, 1)
    return np.where(count > 0, np.sqrt(var), 0.0)


def _probability(model: ClassifierMixin, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(matrix)
        if proba.shape[1] == 1:
            return np.full(matrix.shape[0], float(model.classes_[0]), dtype=np.float32)
        return proba[:, 1].astype(np.float32)
    scores = model.decision_function(matrix)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)


@dataclass
class _Candidate:
    name: str
    factory: Callable[[], ClassifierMixin]


class ReliabilityBoostedFusion:
    """Small supervised RGA+ head selected strictly on validation metrics.

    The model uses the same train/validation split as the paper harness. It
    trains candidate heads on the training fold, chooses the candidate with a
    configured validation-only metric, and then evaluates that selected head on
    test.
    """

    def __init__(
        self,
        score_index: int = 0,
        confidence_index: int | None = None,
        random_seed: int = 42,
        selection_metric: str = "roc_auc",
    ) -> None:
        self.score_index = int(score_index)
        self.confidence_index = confidence_index
        self.random_seed = int(random_seed)
        self.selection_metric = selection_metric
        self.selected_candidate: str | None = None
        self.candidate_validation_auc: dict[str, float] = {}
        self.candidate_validation_metrics: dict[str, dict[str, float]] = {}
        self._model: ClassifierMixin | None = None
        self._constant_prediction: float | None = None
        self._reliability_estimator = None

    def _candidate_grid(self) -> list[_Candidate]:
        seed = self.random_seed
        candidates: list[_Candidate] = []
        for learning_rate, max_depth, min_samples_leaf in [
            (0.1, None, 20),
            (0.1, 4, 30),
            (0.1, 3, 20),
            (0.1, 3, 10),
            (0.1, 2, 20),
            (0.05, 3, 20),
            (0.05, 3, 10),
        ]:
            name = f"hgb_lr{learning_rate}_depth{max_depth}_leaf{min_samples_leaf}"
            candidates.append(
                _Candidate(
                    name=name,
                    factory=lambda lr=learning_rate, depth=max_depth, leaf=min_samples_leaf: HistGradientBoostingClassifier(
                        learning_rate=lr,
                        max_iter=300,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        l2_regularization=0.01,
                        random_state=seed,
                    ),
                )
            )
        for c_value in (0.1, 1.0, 10.0, 100.0):
            name = f"logistic_c{c_value}"
            candidates.append(
                _Candidate(
                    name=name,
                    factory=lambda c=c_value: make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=seed,
                        ),
                    ),
                )
            )
        return candidates

    def _features(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        present = ~masks
        score = features[:, :, self.score_index].astype(np.float32)
        if self.confidence_index is not None:
            confidence = features[:, :, self.confidence_index].astype(np.float32)
        else:
            confidence = np.ones_like(score, dtype=np.float32)

        score_max = _masked_extreme(score, present, 0.5, is_max=True)
        score_min = _masked_extreme(score, present, 0.5, is_max=False)
        confidence_max = _masked_extreme(confidence, present, 0.5, is_max=True)
        confidence_min = _masked_extreme(confidence, present, 0.5, is_max=False)

        extras = [
            _masked_mean(score, present, 0.5),
            score_max,
            score_min,
            score_max - score_min,
            _masked_std(score, present),
            _masked_mean(confidence, present, 0.5),
            confidence_max,
            confidence_min,
            confidence_max - confidence_min,
            _masked_std(confidence, present),
            present.sum(axis=1) / max(features.shape[1], 1),
        ]

        if features.shape[1] >= 2:
            first_score = np.where(masks[:, 0], 0.5, score[:, 0])
            second_score = np.where(masks[:, 1], 0.5, score[:, 1])
            first_conf = np.where(masks[:, 0], 0.5, confidence[:, 0])
            second_conf = np.where(masks[:, 1], 0.5, confidence[:, 1])
            extras.extend(
                [
                    np.abs(first_score - second_score),
                    first_conf - second_conf,
                ]
            )

        if self._reliability_estimator is not None:
            reliability = self._reliability_estimator.compute_reliability_weights(features, masks)
            rel_max = _masked_extreme(reliability, present, 0.0, is_max=True)
            rel_min = _masked_extreme(reliability, present, 0.0, is_max=False)
            rel_sum = np.where(present, reliability, 0.0).sum(axis=1)
            weighted_score = np.where(present, reliability * score, 0.0).sum(axis=1) / np.maximum(rel_sum, 1e-6)
            extras.extend(
                [
                    _masked_mean(reliability, present, 0.0),
                    rel_max,
                    rel_min,
                    rel_max - rel_min,
                    _masked_std(reliability, present),
                    np.where(rel_sum > 1e-6, weighted_score, 0.5),
                ]
            )

        return np.concatenate(
            [
                _flatten_with_mask(features, masks),
                np.column_stack(extras).astype(np.float32),
            ],
            axis=1,
        )

    def fit(
        self,
        train_features: np.ndarray,
        train_masks: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_masks: np.ndarray,
        val_labels: np.ndarray,
        reliability_estimator=None,
    ) -> ReliabilityBoostedFusion:
        train_labels = np.asarray(train_labels, dtype=np.int64)
        val_labels = np.asarray(val_labels, dtype=np.int64)
        self._reliability_estimator = reliability_estimator

        if len(np.unique(train_labels)) < 2:
            self._constant_prediction = float(np.mean(train_labels)) if len(train_labels) else 0.5
            self.selected_candidate = "constant"
            self.candidate_validation_auc = {"constant": 0.5}
            self.candidate_validation_metrics = {"constant": {"roc_auc": 0.5, "pr_auc": 0.0, "f1": 0.0, "brier": 0.0}}
            return self

        x_train = self._features(train_features, train_masks)
        x_val = self._features(val_features, val_masks)
        best_name = None
        best_score = -np.inf
        best_model = None
        self.candidate_validation_auc = {}
        self.candidate_validation_metrics = {}

        for candidate in self._candidate_grid():
            model = candidate.factory()
            model.fit(x_train, train_labels)
            val_probs = _probability(model, x_val)
            metrics = _validation_metrics(val_labels, val_probs)
            score = _selection_score(metrics, self.selection_metric)
            self.candidate_validation_auc[candidate.name] = metrics["roc_auc"]
            self.candidate_validation_metrics[candidate.name] = metrics
            if score > best_score or (np.isclose(score, best_score) and candidate.name < str(best_name)):
                best_score = score
                best_name = candidate.name
                best_model = model

        self._model = best_model
        self.selected_candidate = best_name
        self._constant_prediction = None
        return self

    def predict_proba(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if self._constant_prediction is not None:
            return np.full(features.shape[0], self._constant_prediction, dtype=np.float32)
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return _probability(self._model, self._features(features, masks))


__all__ = ["ReliabilityBoostedFusion"]
