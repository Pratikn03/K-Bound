"""Evaluation utilities for fusion models."""

import numpy as np

from uais.utils.metrics import classification_metrics


def evaluate_fusion(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return classification_metrics(y, y_pred, proba)


__all__ = ["evaluate_fusion"]
