"""Shared metric utilities for fraud / anomaly models."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += np.abs(bin_acc - bin_conf) * np.mean(mask)
    return float(ece)


def detection_rate_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    negatives = y_prob[y_true == 0]
    if negatives.size == 0:
        return {"tpr_at_fpr": float("nan"), "threshold_at_fpr": float("nan"), "fpr": float("nan")}
    threshold = float(np.quantile(negatives, 1 - target_fpr))
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tpr_at_fpr": float(tpr), "threshold_at_fpr": float(threshold), "fpr": float(fpr)}


def _compute_from_pred_and_prob(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    try:
        scores["roc_auc"] = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        scores["roc_auc"] = float("nan")
    try:
        scores["pr_auc"] = metrics.average_precision_score(y_true, y_prob)
    except ValueError:
        scores["pr_auc"] = float("nan")

    scores["f1"] = metrics.f1_score(y_true, y_pred, zero_division=0)
    scores["precision"] = metrics.precision_score(y_true, y_pred, zero_division=0)
    scores["recall"] = metrics.recall_score(y_true, y_pred, zero_division=0)
    scores["accuracy"] = metrics.accuracy_score(y_true, y_pred)
    scores["balanced_accuracy"] = metrics.balanced_accuracy_score(y_true, y_pred)
    scores["brier"] = brier_score(y_true, y_prob)
    scores["ece"] = expected_calibration_error(y_true, y_prob)
    scores.update(detection_rate_at_fpr(y_true, y_prob, target_fpr=0.01))
    return scores


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return _compute_from_pred_and_prob(y_true, y_pred, y_prob)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix (2x2 for binary classification)."""
    return metrics.confusion_matrix(y_true, y_pred)


def classification_metrics(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    y_prob: np.ndarray | float | None = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Backward-compatible wrapper for mixed call signatures."""
    y_true = np.asarray(y_true)
    if y_prob is None:
        return compute_classification_metrics(y_true, y_pred_or_prob, threshold=threshold)
    if np.isscalar(y_prob):
        return compute_classification_metrics(y_true, y_pred_or_prob, threshold=float(y_prob))
    y_pred = np.asarray(y_pred_or_prob)
    y_prob_arr = np.asarray(y_prob)
    return _compute_from_pred_and_prob(y_true, y_pred, y_prob_arr)


def anomaly_metrics(y_true, scores, threshold=None, contamination: float = 0.05):
    # Simple anomaly metric: threshold by quantile if none provided
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if threshold is None:
        threshold = np.quantile(scores, 1 - contamination)
    preds = (scores >= threshold).astype(int)
    out = classification_metrics(y_true, preds, threshold=0.5)
    out["threshold"] = float(threshold)
    return out


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold that maximizes F1 on given labels."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.unique(y_prob)
    if len(thresholds) == 0:
        return 0.5
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        f1 = metrics.f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


__all__ = [
    "brier_score",
    "expected_calibration_error",
    "detection_rate_at_fpr",
    "compute_classification_metrics",
    "classification_metrics",
    "compute_confusion_matrix",
    "anomaly_metrics",
    "best_f1_threshold",
]
