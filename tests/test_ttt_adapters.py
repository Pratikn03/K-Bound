"""Tests for TentAdapter and PseudoLabelTTTAdapter (test-time adaptive baselines)."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.baselines import (
    EarlyFusionMLP,
    PseudoLabelTTTAdapter,
    TentAdapter,
    run_baseline_suite,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_TR, N_TE, D, F = 120, 40, 3, 5
SCORE_IDX = 0


def _make_data(seed: int = 0, missing_prob: float = 0.0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N_TR + N_TE) < 0.3).astype(float)
    features = rng.random((N_TR + N_TE, D, F)).astype(np.float32)
    # Give anomalies a higher score
    for d in range(D):
        features[labels == 1, d, SCORE_IDX] = np.clip(
            features[labels == 1, d, SCORE_IDX] + 0.4, 0.0, 1.0
        )
    masks = (rng.random((N_TR + N_TE, D)) < missing_prob).astype(bool)
    train_idx = np.arange(N_TR)
    test_idx = np.arange(N_TR, N_TR + N_TE)
    return features, masks, labels, train_idx, test_idx


@pytest.fixture
def fitted_tent():
    features, masks, labels, train_idx, test_idx = _make_data(seed=7)
    adapter = TentAdapter(n_steps=1, lr=1e-3)
    adapter.fit(features[train_idx], masks[train_idx], labels[train_idx])
    return adapter, features[test_idx], masks[test_idx], labels[test_idx]


@pytest.fixture
def fitted_plt():
    features, masks, labels, train_idx, test_idx = _make_data(seed=8)
    adapter = PseudoLabelTTTAdapter(confidence_threshold=0.8, n_steps=2, lr=5e-4)
    adapter.fit(features[train_idx], masks[train_idx], labels[train_idx])
    return adapter, features[test_idx], masks[test_idx], labels[test_idx]


# ---------------------------------------------------------------------------
# TentAdapter tests
# ---------------------------------------------------------------------------

def test_tent_predict_shape(fitted_tent):
    adapter, features, masks, _ = fitted_tent
    probs = adapter.predict_proba(features, masks)
    assert probs.shape == (N_TE,), f"Expected ({N_TE},), got {probs.shape}"


def test_tent_predictions_in_range(fitted_tent):
    adapter, features, masks, _ = fitted_tent
    probs = adapter.predict_proba(features, masks)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Tent predictions outside [0,1]"


def test_tent_unfitted_raises():
    adapter = TentAdapter()
    features = np.random.rand(10, D, F).astype(np.float32)
    masks = np.zeros((10, D), dtype=bool)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict_proba(features, masks)


def test_tent_stateless_across_calls(fitted_tent):
    """Two consecutive calls with the same input should return identical results.

    This verifies that the model is reset to its fitted state before each call.
    """
    adapter, features, masks, _ = fitted_tent
    p1 = adapter.predict_proba(features, masks)
    p2 = adapter.predict_proba(features, masks)
    np.testing.assert_allclose(p1, p2, rtol=1e-5,
                               err_msg="Tent predictions changed on second call (not stateless)")


def test_tent_finite_outputs(fitted_tent):
    adapter, features, masks, _ = fitted_tent
    probs = adapter.predict_proba(features, masks)
    assert np.all(np.isfinite(probs)), "Tent output contains NaN or Inf"


# ---------------------------------------------------------------------------
# PseudoLabelTTTAdapter tests
# ---------------------------------------------------------------------------

def test_plt_predict_shape(fitted_plt):
    adapter, features, masks, _ = fitted_plt
    probs = adapter.predict_proba(features, masks)
    assert probs.shape == (N_TE,), f"Expected ({N_TE},), got {probs.shape}"


def test_plt_predictions_in_range(fitted_plt):
    adapter, features, masks, _ = fitted_plt
    probs = adapter.predict_proba(features, masks)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "PLT predictions outside [0,1]"


def test_plt_unfitted_raises():
    adapter = PseudoLabelTTTAdapter()
    features = np.random.rand(10, D, F).astype(np.float32)
    masks = np.zeros((10, D), dtype=bool)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict_proba(features, masks)


def test_plt_stateless_across_calls(fitted_plt):
    """Two identical calls should return the same result."""
    adapter, features, masks, _ = fitted_plt
    p1 = adapter.predict_proba(features, masks)
    p2 = adapter.predict_proba(features, masks)
    np.testing.assert_allclose(p1, p2, rtol=1e-5,
                               err_msg="PLT predictions changed on second call (not stateless)")


def test_plt_high_confidence_threshold_falls_back(fitted_plt):
    """With threshold=1.0, no pseudo-labels are selected → identical to base model."""
    adapter, features, masks, _ = fitted_plt
    base_probs = adapter.base.predict_proba(features, masks)
    adapter_no_pseudo = PseudoLabelTTTAdapter(
        base=adapter.base, confidence_threshold=1.0, n_steps=5
    )
    adapter_no_pseudo._base_state = adapter._base_state
    probs = adapter_no_pseudo.predict_proba(features, masks)
    np.testing.assert_allclose(probs, base_probs, rtol=1e-4,
                               err_msg="With threshold=1.0 no adaptation should occur")


# ---------------------------------------------------------------------------
# run_baseline_suite integration — new TTT baselines included
# ---------------------------------------------------------------------------

def test_run_baseline_suite_includes_ttt():
    """run_baseline_suite should return metrics for tent_ttt and pseudo_label_ttt."""
    features, masks, labels, train_idx, test_idx = _make_data(seed=99, missing_prob=0.1)
    val_idx = train_idx[:20]
    train_idx_ = train_idx[20:]
    results = run_baseline_suite(
        features, masks, labels,
        train_idx_, val_idx, test_idx,
        score_index=SCORE_IDX,
    )
    assert "tent_ttt" in results, "tent_ttt missing from run_baseline_suite output"
    assert "pseudo_label_ttt" in results, "pseudo_label_ttt missing from run_baseline_suite output"

    for key in ("tent_ttt", "pseudo_label_ttt"):
        assert "roc_auc" in results[key], f"{key} is missing roc_auc"
        auc = results[key]["roc_auc"]
        assert auc is None or (0.0 <= auc <= 1.0), f"{key} roc_auc out of [0,1]: {auc}"
