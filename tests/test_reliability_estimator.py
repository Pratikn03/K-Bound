"""Tests for ReliabilityEstimator (CRS component of CRAF)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from uais.fusion.attention.reliability_estimator import (
    CategoryAwareReliabilityEstimator,
    PerSampleReliabilityEstimator,
    ReliabilityEstimator,
)


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


def test_gate_decisions_shape_and_type(fitted_estimator):
    """gate_decisions() must return a bool array of shape [N]."""
    est, features, masks, _ = fitted_estimator
    weights = est.compute_reliability_weights(features, masks)
    gate = est.gate_decisions(weights, masks)
    assert gate.shape == (N,), f"Expected ({N},), got {gate.shape}"
    assert gate.dtype == bool, f"Expected bool, got {gate.dtype}"


def test_gate_decisions_high_reliability_stays_static():
    """Samples with perfect domain scores (sharpness=1) should stay on static path."""
    features, masks, labels = _make_data(seed=5, missing_prob=0.0)
    # Push all scores to 0.9 — very sharp, high reliability
    features[:, :, SCORE_INDEX] = 0.9
    est = ReliabilityEstimator(DOMAIN_ORDER, SCORE_INDEX, gate_threshold=0.66,
                               min_samples_for_ks=5)
    est.fit(features, masks, labels)
    weights = est.compute_reliability_weights(features, masks)
    gate = est.gate_decisions(weights, masks)
    # With high sharpness the mean reliability should exceed 0.66 → gate=False (static path)
    assert not gate.all(), "All-sharp domains should not all activate the reliability path"


def test_gate_decisions_low_reliability_activates():
    """Scores clustered at 0.5 give zero sharpness → low reliability → gate fires."""
    features, masks, labels = _make_data(seed=6, missing_prob=0.0)
    # Push all scores to exactly 0.5 — zero sharpness
    features[:, :, SCORE_INDEX] = 0.5
    est = ReliabilityEstimator(DOMAIN_ORDER, SCORE_INDEX,
                               ece_weight=0.0, ks_weight=0.0, sharpness_weight=1.0,
                               gate_threshold=0.66, min_samples_for_ks=5)
    est.fit(features, masks, labels)
    weights = est.compute_reliability_weights(features, masks)
    gate = est.gate_decisions(weights, masks)
    # Zero sharpness → reliability = 0.0 < 0.66 → all samples should activate reliability path
    assert gate.all(), "Zero-sharpness domains should always activate the reliability path"


def test_mean_gate_dilutes_single_domain_collapse():
    """With four domains and tau=0.66, one zero-reliability domain is hidden."""
    weights = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    masks = np.zeros_like(weights, dtype=bool)
    est = ReliabilityEstimator(["d0", "d1", "d2", "d3"], SCORE_INDEX, gate_threshold=0.66)

    gate = est.gate_decisions(weights, masks)

    assert gate.tolist() == [False]


def test_minimum_gate_catches_single_domain_collapse():
    """A minimum-reliability gate fires on one catastrophic domain failure."""
    weights = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    masks = np.zeros_like(weights, dtype=bool)
    est = ReliabilityEstimator(
        ["d0", "d1", "d2", "d3"],
        SCORE_INDEX,
        gate_threshold=0.66,
        gate_mode="minimum",
        min_gate_threshold=0.34,
    )

    gate = est.gate_decisions(weights, masks)

    assert gate.tolist() == [True]


def test_hybrid_gate_catches_isolated_and_broad_failures():
    """Hybrid RGA-v2 fires for either mean degradation or a low minimum domain."""
    masks = np.zeros((3, 4), dtype=bool)
    weights = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],      # isolated catastrophic failure
            [0.50, 0.55, 0.60, 0.65],  # broad mean degradation
            [0.80, 0.80, 0.85, 0.90],  # healthy
        ],
        dtype=np.float32,
    )
    est = ReliabilityEstimator(
        ["d0", "d1", "d2", "d3"],
        SCORE_INDEX,
        gate_threshold=0.66,
        gate_mode="hybrid",
        min_gate_threshold=0.34,
    )

    gate = est.gate_decisions(weights, masks)

    assert gate.tolist() == [True, True, False]


def test_gate_save_load_preserves_threshold(fitted_estimator):
    """gate_threshold must survive a save/load round-trip."""
    import tempfile
    est_custom = ReliabilityEstimator(DOMAIN_ORDER, SCORE_INDEX, gate_threshold=0.55,
                                      min_samples_for_ks=5)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/est.joblib"
        est_custom.save(path)
        loaded = ReliabilityEstimator.load(path)
    assert loaded.gate_threshold == 0.55, (
        f"Expected gate_threshold=0.55, got {loaded.gate_threshold}"
    )


def test_category_aware_ks_does_not_misfire_on_category_mix_shift():
    """Category-aware drift should not penalize legitimate category composition shifts."""
    rng = np.random.default_rng(123)
    n_per_category = 80
    domain_order = ["rgb"]
    masks = np.zeros((n_per_category * 2, 1), dtype=bool)
    labels = rng.integers(0, 2, size=n_per_category * 2).astype(float)
    features = np.zeros((n_per_category * 2, 1, 1), dtype=np.float32)
    categories = np.array(["bagel"] * n_per_category + ["cable_gland"] * n_per_category)
    features[:n_per_category, 0, 0] = rng.normal(0.2, 0.02, size=n_per_category)
    features[n_per_category:, 0, 0] = rng.normal(0.8, 0.02, size=n_per_category)
    features = np.clip(features, 0.0, 1.0).astype(np.float32)

    global_estimator = ReliabilityEstimator(
        domain_order=domain_order,
        score_index=0,
        ece_weight=0.0,
        ks_weight=1.0,
        sharpness_weight=0.0,
        min_samples_for_ks=20,
    ).fit(features, masks, labels)
    category_estimator = CategoryAwareReliabilityEstimator(
        domain_order=domain_order,
        score_index=0,
        ece_weight=0.0,
        ks_weight=1.0,
        sharpness_weight=0.0,
        min_samples_for_ks=20,
    ).fit(features, masks, labels, categories=categories)

    # Legitimate deployment batch contains only one known category, with the
    # same category-conditional distribution observed during validation.
    batch_features = features[:n_per_category].copy()
    batch_masks = masks[:n_per_category].copy()
    batch_categories = categories[:n_per_category].copy()

    global_weight = global_estimator.compute_reliability_weights(batch_features, batch_masks)
    category_weight = category_estimator.compute_reliability_weights(
        batch_features,
        batch_masks,
        categories=batch_categories,
    )

    assert float(category_weight.mean()) > 0.8
    assert float(category_weight.mean()) > float(global_weight.mean()) + 0.5


# ---------------------------------------------------------------------------
# Per-sample reliability estimator (C1)
# ---------------------------------------------------------------------------

def test_per_sample_reliability_varies_per_sample():
    """Per-sample estimator must produce non-trivial per-sample variance.

    The batch-level base class collapses to a constant per (domain, batch),
    which is the streaming-deployment objection that C1 closes.
    """
    features, masks, labels = _make_data(seed=2, missing_prob=0.0)
    base = ReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    ).fit(features, masks, labels)
    per_sample = PerSampleReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    ).fit(features, masks, labels)

    base_w = base.compute_reliability_weights(features, masks)
    sample_w = per_sample.compute_reliability_weights(features, masks)

    assert base_w.shape == sample_w.shape == (N, len(DOMAIN_ORDER))
    # Base estimator: per-domain column is constant within a batch.
    for d in range(len(DOMAIN_ORDER)):
        col = base_w[:, d]
        avail = col > 0
        if avail.sum() < 2:
            continue
        assert float(col[avail].std()) < 1e-6, (
            f"batch-level estimator should produce constant column for domain {d}"
        )
    # Per-sample estimator: every domain column must have non-trivial variance.
    for d in range(len(DOMAIN_ORDER)):
        col = sample_w[:, d]
        avail = col > 0
        assert avail.sum() > 10, f"too few available rows for domain {d}"
        assert float(col[avail].std()) > 0.01, (
            f"per-sample estimator should produce per-sample variance for domain {d}"
        )

    # Weights still bounded to [0, 1].
    assert float(sample_w.min()) >= 0.0
    assert float(sample_w.max()) <= 1.0


def test_per_sample_reliability_drops_at_tails():
    """A sample whose score sits in the validation distribution's tail must
    receive a lower reliability than one sitting near the median."""
    features, masks, labels = _make_data(seed=3, missing_prob=0.0)
    est = PerSampleReliabilityEstimator(
        domain_order=DOMAIN_ORDER,
        score_index=SCORE_INDEX,
        ece_weight=0.0,
        ks_weight=1.0,
        sharpness_weight=0.0,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    ).fit(features, masks, labels)

    # Craft a probe batch: one sample at validation median, one at extreme tail.
    probe = np.zeros((2, len(DOMAIN_ORDER), features.shape[2]), dtype=np.float32)
    probe[:, :, :] = 0.5
    median_score = float(np.median(features[~masks[:, 0], 0, SCORE_INDEX]))
    probe[0, 0, SCORE_INDEX] = median_score
    probe[1, 0, SCORE_INDEX] = 0.99  # tail
    probe_masks = np.zeros((2, len(DOMAIN_ORDER)), dtype=bool)

    w = est.compute_reliability_weights(probe, probe_masks)
    assert w[0, 0] > w[1, 0], (
        "sample near validation median should receive higher reliability"
        f" than a tail sample (got median={w[0, 0]}, tail={w[1, 0]})"
    )
