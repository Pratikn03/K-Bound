from __future__ import annotations

import numpy as np

from uais.fusion.attention.baselines import run_baseline_suite


def test_baseline_suite_can_return_validation_and_test_predictions():
    rng = np.random.default_rng(0)
    features = rng.normal(0.5, 0.15, size=(18, 2, 3)).clip(0.0, 1.0).astype(np.float32)
    masks = np.zeros((18, 2), dtype=bool)
    labels = np.array([0, 1] * 9, dtype=np.int64)
    train_idx = np.arange(0, 10)
    val_idx = np.arange(10, 14)
    test_idx = np.arange(14, 18)

    metrics, predictions = run_baseline_suite(
        features,
        masks,
        labels,
        train_idx,
        val_idx,
        test_idx,
        baseline_epochs=1,
        tta_steps=1,
        return_predictions=True,
    )

    assert "random_forest" in metrics
    assert "random_forest" in predictions
    assert predictions["random_forest"]["val_probs"].shape == (4,)
    assert predictions["random_forest"]["test_probs"].shape == (4,)
