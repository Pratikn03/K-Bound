"""Tests for ReliabilityEstimator (CRS component of CRAF)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N, D, F = 200, 3, 5
SCORE_INDEX = 0
DOMAIN_ORDER = ["fraud", "cyber", "behavior"]


def _make_data(seed: int = 0, missing_prob: float = 0.1):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N) < 0.2).astype(float)
    features = rng.random((N, D, F)).astype(np.float32)
    for d in range(D):
        features[labels == 1, d, SCORE_INDEX] = np.clip(
            features[labels == 1, d, SCORE_INDEX] + 0.4, 0.0, 1.0
        )
    masks = rng.random((N, D)) < missing_prob
    return features, masks.astype(bool), labels


@pytest.fixture
def fitted_estimator():
    features, masks, labels = _make_data(seed=1)
    est = ReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    )
    est.fit(features, masks, labels)
    return est, features, masks, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_produces_valid_ece(fitted_estimator):
    est, _, _, _ = fitted_estimator
    ece_dict = est.get_domain_ece()
    assert set(ece_dict.keys()) == set(DOMAIN_ORDER)
    for domain, ece_val in ece_dict.items():
        assert np.isfinite(ece_val), f"ECE for {domain} is not finite"
        assert 0.0 <= ece_val <= 1.0, f"ECE for {domain} out of [0,1]: {ece_val}"


def test_compute_reliability_weights_shape(fitted_estimator):
    est, features, masks, _ = fitted_estimator
    weights = est.compute_reliability_weights(features, masks)
    assert weights.shape == (N, D), f"Expected ({N}, {D}), got {weights.shape}"
    assert weights.dtype == np.float32
    assert np.all(weights >= 0.0), "Negative reliability weight found"
    assert np.all(weights <= 1.0), "Reliability weight > 1.0 found"


def test_reliability_zeros_for_missing_domains(fitted_estimator):
    est, features, masks, _ = fitted_estimator
    weights = est.compute_reliability_weights(features, masks)
    # Where domain is masked, weight must be exactly 0.0
    assert np.all(weights[masks] == 0.0), "Non-zero weight for missing domain"


def test_save_load_roundtrip(fitted_estimator):
    est, features, masks, _ = fitted_estimator
    original_ece = est.get_domain_ece()
    original_weights = est.compute_reliability_weights(features[:10], masks[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "estimator.joblib"
        est.save(path)
        loaded = ReliabilityEstimator.load(path)

    loaded_ece = loaded.get_domain_ece()
    assert loaded_ece == original_ece, "ECE dict changed after save/load"

    loaded_weights = loaded.compute_reliability_weights(features[:10], masks[:10])
    np.testing.assert_allclose(loaded_weights, original_weights, rtol=1e-5)


def test_unfitted_raises():
    est = ReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
    )
    features, masks, _ = _make_data()
    with pytest.raises(RuntimeError, match="fitted"):
        est.compute_reliability_weights(features, masks)

    with pytest.raises(RuntimeError, match="fit"):
        est.get_domain_ece()


def test_ks_reliability_decreases_with_drift():
    """A drifted domain should receive a lower reliability weight than the clean domain."""
    rng = np.random.default_rng(42)
    features, masks, labels = _make_data(seed=42, missing_prob=0.0)

    est = ReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
        ece_weight=0.0,     # disable ECE and sharpness; isolate KS component
        ks_weight=1.0,
        sharpness_weight=0.0,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    )
    est.fit(features, masks, labels)

    # Clean inference (same distribution as training)
    clean_weights = est.compute_reliability_weights(features, masks)

    # Drifted: shift domain 0 scores far from the training distribution
    drifted = features.copy()
    drifted[:, 0, SCORE_INDEX] = np.clip(drifted[:, 0, SCORE_INDEX] + 0.5, 0.0, 1.0)
    drifted_weights = est.compute_reliability_weights(drifted, masks)

    mean_clean_d0 = float(np.mean(clean_weights[:, 0]))
    mean_drifted_d0 = float(np.mean(drifted_weights[:, 0]))

    assert mean_drifted_d0 < mean_clean_d0, (
        f"Drifted domain 0 reliability ({mean_drifted_d0:.4f}) should be lower "
        f"than clean ({mean_clean_d0:.4f})"
    )

    # Domains 1 and 2 are unchanged; their weight should not differ between clean and drifted
    for d in [1, 2]:
        np.testing.assert_allclose(
            clean_weights[:, d], drifted_weights[:, d], rtol=1e-4,
            err_msg=f"Domain {d} weights changed despite no drift"
        )
