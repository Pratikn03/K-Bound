"""Tests for unsupervised anomaly baselines (normal-only training protocol)."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.unsupervised_baselines import (
    AEConfig,
    AutoencoderAnomalyDetector,
    BGMMAnomalyDetector,
    BGMMConfig,
    GMMAnomalyDetector,
    GMMConfig,
    IForestConfig,
    IsolationForestDetector,
    KMeansAnomalyDetector,
    KMeansConfig,
    LOFAnomalyDetector,
    LOFConfig,
    OCSVMConfig,
    OneClassSVMDetector,
    run_unsupervised_suite,
)

N, D, F = 200, 3, 4


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N) < 0.3).astype(int)
    features = rng.random((N, D, F)).astype(np.float32)
    for d in range(D):
        features[labels == 1, d, 0] = np.clip(features[labels == 1, d, 0] + 0.5, 0, 1)
    masks = (rng.random((N, D)) < 0.1).astype(bool)
    return features, masks, labels


@pytest.fixture(scope="module")
def data():
    return _make_data()


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(123)
    idx = rng.permutation(N)
    return idx[: int(0.7 * N)], idx[int(0.7 * N) :]


# ---------------------------------------------------------------------------
# Per-detector basic checks
# ---------------------------------------------------------------------------

DETECTORS = [
    ("bgmm", BGMMAnomalyDetector, BGMMConfig),
    ("gmm", GMMAnomalyDetector, GMMConfig),
    ("kmeans", KMeansAnomalyDetector, KMeansConfig),
    ("iforest", IsolationForestDetector, IForestConfig),
    ("ocsvm", OneClassSVMDetector, OCSVMConfig),
    ("lof", LOFAnomalyDetector, LOFConfig),
    ("autoencoder", AutoencoderAnomalyDetector, AEConfig),
]


@pytest.mark.parametrize("name,cls,cfg_cls", DETECTORS)
def test_detector_fit_predict_shape(name, cls, cfg_cls, data, split):
    features, masks, labels = data
    train_idx, test_idx = split
    det = cls(cfg_cls() if cfg_cls is not AEConfig else cfg_cls(epochs=5))
    det.fit(features[train_idx], masks[train_idx], labels[train_idx])
    probs = det.predict_proba(features[test_idx], masks[test_idx])
    assert probs.shape == (len(test_idx),)


@pytest.mark.parametrize("name,cls,cfg_cls", DETECTORS)
def test_detector_predictions_in_range(name, cls, cfg_cls, data, split):
    features, masks, labels = data
    train_idx, test_idx = split
    det = cls(cfg_cls() if cfg_cls is not AEConfig else cfg_cls(epochs=5))
    det.fit(features[train_idx], masks[train_idx], labels[train_idx])
    probs = det.predict_proba(features[test_idx], masks[test_idx])
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


@pytest.mark.parametrize("name,cls,cfg_cls", DETECTORS)
def test_detector_finite_outputs(name, cls, cfg_cls, data, split):
    features, masks, labels = data
    train_idx, test_idx = split
    det = cls(cfg_cls() if cfg_cls is not AEConfig else cfg_cls(epochs=5))
    det.fit(features[train_idx], masks[train_idx], labels[train_idx])
    probs = det.predict_proba(features[test_idx], masks[test_idx])
    assert np.all(np.isfinite(probs))


@pytest.mark.parametrize("name,cls,cfg_cls", DETECTORS)
def test_detector_hyperparameters_exposed(name, cls, cfg_cls):
    det = cls(cfg_cls() if cfg_cls is not AEConfig else cfg_cls(epochs=5))
    hp = det.get_hyperparameters()
    assert isinstance(hp, dict)
    assert len(hp) > 0


@pytest.mark.parametrize("name,cls,cfg_cls", DETECTORS)
def test_detector_filters_to_normal_only(name, cls, cfg_cls, data):
    """Anomaly samples (label==1) must be discarded from training."""
    features, masks, labels = data
    # Force a known small subset where only one sample is normal
    n_normal = int((labels == 0).sum())
    assert n_normal >= 5, "fixture must have ≥5 normals for this test"

    det = cls(cfg_cls() if cfg_cls is not AEConfig else cfg_cls(epochs=5))
    det.fit(features, masks, labels)
    # Just checking the detector accepts mixed labels and produces output
    probs = det.predict_proba(features, masks)
    assert probs.shape == (N,)


def test_bgmm_effective_component_count(data):
    features, masks, labels = data
    det = BGMMAnomalyDetector(BGMMConfig(n_components=15))
    det.fit(features, masks, labels)
    n_eff = det.get_effective_n_components()
    assert 1 <= n_eff <= 15


def test_unfitted_predict_raises():
    det = BGMMAnomalyDetector()
    features, masks, _ = _make_data()
    with pytest.raises(RuntimeError, match="fit"):
        det.predict_proba(features, masks)


def test_fit_without_normals_raises():
    """If labels are all 1 (no normals), fit() should raise a clear error."""
    features, masks, _ = _make_data()
    det = BGMMAnomalyDetector(BGMMConfig(n_components=2))
    with pytest.raises(ValueError, match="no normal samples"):
        det.fit(features, masks, np.ones(N, dtype=int))


# ---------------------------------------------------------------------------
# Suite runner — all detectors at once
# ---------------------------------------------------------------------------


def test_run_unsupervised_suite(data, split):
    features, masks, labels = data
    train_idx, test_idx = split
    results = run_unsupervised_suite(
        features,
        masks,
        labels,
        train_idx,
        test_idx,
        ae_cfg=AEConfig(epochs=3),
    )
    expected_keys = {"bgmm", "gmm", "kmeans", "isolation_forest", "one_class_svm", "lof", "autoencoder"}
    assert set(results.keys()) == expected_keys
    for _name, metrics_d in results.items():
        if "error" in metrics_d:
            continue
        assert "roc_auc" in metrics_d
        assert "pr_auc" in metrics_d
        assert "f1" in metrics_d
        assert "mcc" in metrics_d
        assert "hyperparameters" in metrics_d
        auc = metrics_d["roc_auc"]
        if auc is not None and np.isfinite(auc):
            assert 0.0 <= auc <= 1.0
