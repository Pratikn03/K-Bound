"""Smoke tests for flagship RGA+ variants."""

from __future__ import annotations

import numpy as np

from uais.fusion.attention.reliability_boosted_fusion_flagship import (
    ReliabilityBoostedFusionFlagship,
    _apply_entropy_tta,
)


def test_entropy_tta_preserves_shape():
    logits = np.linspace(-2, 2, 32).astype(np.float32)
    out = _apply_entropy_tta(logits, adaptation_steps=5)
    assert out.shape == (32,)


def test_flagship_fit_predict_smoke():
    n = 80
    d, f = 2, 4
    features = np.random.rand(n, d, f).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (features[:, 0, 0] > 0.5).astype(int)
    cats = np.array(["a"] * 40 + ["b"] * 40)
    model = ReliabilityBoostedFusionFlagship(
        use_category_features=True,
        tta_candidates=(0, 5),
        random_seed=0,
    )
    model.fit(
        features[:50],
        masks[:50],
        labels[:50],
        features[50:],
        masks[50:],
        labels[50:],
        train_categories=cats[:50],
        val_categories=cats[50:],
    )
    probs = model.predict_proba(features[50:], masks[50:], categories=cats[50:])
    assert probs.shape == (30,)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
