"""Tests for the four strong fusion baselines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uais.fusion.attention.baselines import (
    ConfidenceWeightedMean,
    EarlyFusionMLP,
    LateFusionEnsemble,
    RandomForestFusion,
    run_baseline_suite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, D, F = 160, 3, 5
SCORE_IDX = 0


def _make_data(seed: int = 0, missing_prob: float = 0.1):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N) < 0.2).astype(float)
    features = rng.random((N, D, F)).astype(np.float32)
    for d in range(D):
        features[labels == 1, d, SCORE_IDX] = np.clip(
            features[labels == 1, d, SCORE_IDX] + 0.4, 0.0, 1.0
        )
    masks = rng.random((N, D)) < missing_prob
    return features, masks.astype(bool), labels


def _splits():
    idx = np.arange(N)
    return idx[:110], idx[110:130], idx[130:]


# ---------------------------------------------------------------------------
# EarlyFusionMLP
# ---------------------------------------------------------------------------

def test_early_fusion_mlp_output_shape():
    features, masks, labels = _make_data()
    train_idx, val_idx, test_idx = _splits()
    mlp = EarlyFusionMLP(hidden_dims=[32, 16], epochs=3, patience=2, device=torch.device("cpu"))
    mlp.fit(features[train_idx], masks[train_idx], labels[train_idx],
            features[val_idx], masks[val_idx], labels[val_idx])
    probs = mlp.predict_proba(features[test_idx], masks[test_idx])
    assert probs.shape == (len(test_idx),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_early_fusion_mlp_unfitted_raises():
    mlp = EarlyFusionMLP()
    features, masks, _ = _make_data()
    with pytest.raises(RuntimeError):
        mlp.predict_proba(features, masks)


# ---------------------------------------------------------------------------
# LateFusionEnsemble
# ---------------------------------------------------------------------------

def test_late_fusion_ensemble_output_shape():
    features, masks, labels = _make_data()
    train_idx, val_idx, test_idx = _splits()
    lfe = LateFusionEnsemble(score_index=SCORE_IDX, min_samples_per_domain=5)
    lfe.fit(features[train_idx], masks[train_idx], labels[train_idx])
    probs = lfe.predict_proba(features[test_idx], masks[test_idx])
    assert probs.shape == (len(test_idx),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_late_fusion_ensemble_all_missing_domain():
    """A fully missing domain should not crash and produce valid probs."""
    features, masks, labels = _make_data()
    train_idx, val_idx, test_idx = _splits()
    lfe = LateFusionEnsemble(min_samples_per_domain=5)
    lfe.fit(features[train_idx], masks[train_idx], labels[train_idx])
    full_missing_masks = masks[test_idx].copy()
    full_missing_masks[:, 0] = True  # domain 0 fully masked at test
    probs = lfe.predict_proba(features[test_idx], full_missing_masks)
    assert probs.shape == (len(test_idx),)
    assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# RandomForestFusion
# ---------------------------------------------------------------------------

def test_random_forest_output_shape():
    features, masks, labels = _make_data()
    train_idx, _, test_idx = _splits()
    rf = RandomForestFusion(n_estimators=20)
    rf.fit(features[train_idx], masks[train_idx], labels[train_idx])
    probs = rf.predict_proba(features[test_idx], masks[test_idx])
    assert probs.shape == (len(test_idx),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# ConfidenceWeightedMean
# ---------------------------------------------------------------------------

def test_confidence_weighted_mean_bounds():
    features, masks, labels = _make_data()
    cwm = ConfidenceWeightedMean(score_index=SCORE_IDX)
    cwm.fit(features, masks, labels)
    probs = cwm.predict_proba(features, masks)
    assert probs.shape == (N,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_confidence_weighted_mean_all_missing_fallback():
    """All-missing input must return exactly 0.5."""
    features, _, labels = _make_data()
    cwm = ConfidenceWeightedMean(score_index=SCORE_IDX)
    cwm.fit(features, np.zeros((N, D), dtype=bool), labels)
    all_missing = np.ones((10, D), dtype=bool)
    probs = cwm.predict_proba(features[:10], all_missing)
    np.testing.assert_array_equal(probs, 0.5)


def test_confidence_weighted_mean_sharpness_ordering():
    """Domain with higher score deviation should get more weight."""
    rng = np.random.default_rng(42)
    # 2-domain case: domain 0 has high scores (far from 0.5), domain 1 near 0.5
    features = np.zeros((10, 2, 3), dtype=np.float32)
    features[:, 0, 0] = 0.9   # sharp, high anomaly signal
    features[:, 1, 0] = 0.52  # near 0.5, almost no signal
    masks = np.zeros((10, 2), dtype=bool)
    labels = np.ones(10)
    cwm = ConfidenceWeightedMean(score_index=0)
    cwm.fit(features, masks, labels)
    probs = cwm.predict_proba(features, masks)
    # Result should be pulled toward 0.9 (sharp domain dominates)
    assert probs.mean() > 0.75, f"Expected >0.75, got {probs.mean():.3f}"


# ---------------------------------------------------------------------------
# run_baseline_suite integration
# ---------------------------------------------------------------------------

def test_run_baseline_suite_keys():
    features, masks, labels = _make_data()
    train_idx, val_idx, test_idx = _splits()
    results = run_baseline_suite(
        features, masks, labels,
        train_idx, val_idx, test_idx,
        score_index=SCORE_IDX,
        device=torch.device("cpu"),
    )
    expected_keys = {
        "early_fusion_mlp", "late_fusion_ensemble", "random_forest",
        "confidence_weighted_mean", "tent_ttt", "pseudo_label_ttt",
    }
    assert set(results.keys()) == expected_keys, f"Missing keys: {expected_keys - set(results.keys())}"
    for name, metrics in results.items():
        assert "roc_auc" in metrics, f"roc_auc missing from {name}"
        auc = metrics["roc_auc"]
        if auc is not None:
            assert 0.0 <= auc <= 1.0, f"{name} AUC out of range: {auc}"
