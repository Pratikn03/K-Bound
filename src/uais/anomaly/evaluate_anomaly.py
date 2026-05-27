"""Evaluation helpers for anomaly detectors."""

from collections.abc import Iterable

import numpy as np

from uais.utils.metrics import anomaly_metrics


def evaluate_anomaly_scores(y_true: Iterable[int], scores: Iterable[float], contamination: float) -> dict[str, float]:
    y_true = np.asarray(list(y_true))
    scores = np.asarray(list(scores))
    return anomaly_metrics(y_true, scores, contamination=contamination)


__all__ = ["evaluate_anomaly_scores"]
