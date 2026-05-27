"""Tests for consistent dimensionality reduction across baselines."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.dim_reduction import (
    AEReducerConfig,
    AutoencoderReducer,
    NoOpReducer,
    PCAReducer,
    PCAReducerConfig,
    make_reducer,
)

N, K = 200, 32


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, K)).astype(np.float32)


def test_noop_passes_through():
    X = _data()
    r = NoOpReducer().fit(X)
    Y = r.transform(X)
    np.testing.assert_array_equal(X, Y)


def test_pca_reduces_dimensionality():
    X = _data()
    r = PCAReducer(PCAReducerConfig(n_components=8)).fit(X)
    Y = r.transform(X)
    assert Y.shape == (N, 8)


def test_pca_explained_variance_monotonic():
    X = _data()
    r = PCAReducer(PCAReducerConfig(n_components=4)).fit(X)
    var = r.get_explained_variance_ratio()
    assert np.all(var[:-1] >= var[1:])


def test_pca_train_test_consistency():
    """PCA fit on train must produce a stable test transform (no refitting)."""
    X = _data(seed=0)
    r = PCAReducer(PCAReducerConfig(n_components=4)).fit(X[:100])
    Y_test_a = r.transform(X[100:])
    Y_test_b = r.transform(X[100:])
    np.testing.assert_allclose(Y_test_a, Y_test_b, rtol=1e-6)


def test_autoencoder_reducer_shape():
    X = _data()
    r = AutoencoderReducer(AEReducerConfig(latent_dim=4, epochs=3)).fit(X)
    Y = r.transform(X)
    assert Y.shape == (N, 4)


def test_autoencoder_reducer_finite():
    X = _data()
    r = AutoencoderReducer(AEReducerConfig(latent_dim=4, epochs=3)).fit(X)
    Y = r.transform(X)
    assert np.all(np.isfinite(Y))


def test_unfitted_transform_raises():
    X = _data()
    pca = PCAReducer()
    with pytest.raises(RuntimeError, match="fit"):
        pca.transform(X)


def test_factory_returns_correct_type():
    assert isinstance(make_reducer("none"), NoOpReducer)
    assert isinstance(make_reducer("pca", n_components=4), PCAReducer)
    assert isinstance(make_reducer("autoencoder", latent_dim=2, epochs=1), AutoencoderReducer)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown reducer"):
        make_reducer("nonexistent")
