"""Tests for kbound.evidence.

Thm thm:disagree (K-Bound paper):
    The evidence vector Z captures label-free signals that correlate with
    the benefit Delta, including entropy drop and marginal-prediction shift.
"""

import math
import numpy as np
import pytest

from kbound.evidence import evidence_vector, EVIDENCE_NAMES


def make_probs(n_samples=64, n_classes=10, seed=0, concentration=1.0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(n_classes) * concentration, size=n_samples)
    return p


# ---------------------------------------------------------------------------
# Shape and content checks
# ---------------------------------------------------------------------------

class TestEvidenceShape:
    def test_shape_11(self):
        p0 = make_probs()
        pa = make_probs(seed=1)
        z = evidence_vector(p0, pa)
        assert z.shape == (11,), f"Expected shape (11,), got {z.shape}"

    def test_names_length(self):
        assert len(EVIDENCE_NAMES) == 11

    def test_output_dtype(self):
        p0 = make_probs()
        pa = make_probs(seed=1)
        z = evidence_vector(p0, pa)
        assert z.dtype == np.float64 or z.dtype == float

    def test_update_norm_stored(self):
        p0 = make_probs()
        pa = make_probs(seed=1)
        z = evidence_vector(p0, pa, upd_norm=0.42)
        assert z[-1] == pytest.approx(0.42), "Last component should be update_norm"

    def test_invalid_1d(self):
        with pytest.raises(ValueError):
            evidence_vector(np.ones(10), np.ones(10))

    def test_mismatched_shapes(self):
        p0 = make_probs(n_samples=32, n_classes=10)
        pa = make_probs(n_samples=64, n_classes=10, seed=1)
        with pytest.raises(ValueError):
            evidence_vector(p0, pa)

    def test_empty(self):
        with pytest.raises(ValueError):
            evidence_vector(np.zeros((0, 10)), np.zeros((0, 10)))


# ---------------------------------------------------------------------------
# Monotonicity / sanity checks (Thm thm:disagree)
# ---------------------------------------------------------------------------

class TestEvidenceMonotonicity:
    """Verify that the evidence vector responds correctly to known changes."""

    def test_entropy_drop_positive_when_adapted_more_confident(self):
        """If adapted model is much more confident, entropy_drop (index 7) > 0."""
        rng = np.random.default_rng(10)
        n, C = 64, 10
        # frozen: near-uniform (high entropy)
        p0 = rng.dirichlet(np.ones(C) * 0.5, size=n)
        # adapted: peaked (low entropy) -- simulate collapse
        pa_raw = np.zeros((n, C))
        pa_raw[np.arange(n), rng.integers(0, C, size=n)] = 1.0
        pa = pa_raw + 1e-6
        pa /= pa.sum(axis=1, keepdims=True)
        z = evidence_vector(p0, pa)
        entropy_drop = z[7]
        assert entropy_drop > 0, (
            f"entropy_drop should be positive when pa is more confident than p0; got {entropy_drop}"
        )

    def test_entropy_drop_negative_when_adapted_less_confident(self):
        """If adapted model is LESS confident, entropy_drop < 0."""
        rng = np.random.default_rng(11)
        n, C = 64, 10
        # frozen: peaked
        p0_raw = np.zeros((n, C))
        p0_raw[np.arange(n), rng.integers(0, C, size=n)] = 1.0
        p0 = p0_raw + 1e-6; p0 /= p0.sum(axis=1, keepdims=True)
        # adapted: near-uniform
        pa = rng.dirichlet(np.ones(C) * 0.5, size=n)
        z = evidence_vector(p0, pa)
        assert z[7] < 0, "entropy_drop should be negative when pa is less confident"

    def test_frac_highconf_near_1_on_collapse(self):
        """Collapse: pa assigns >0.9 prob to one class -> frac_highconf near 1."""
        n, C = 64, 10
        pa = np.full((n, C), 1e-4)
        pa[:, 0] = 1.0 - (C - 1) * 1e-4   # near-unit mass on class 0
        p0 = make_probs(n_samples=n, n_classes=C, seed=12)
        z = evidence_vector(p0, pa)
        frac_hi = z[8]
        assert frac_hi > 0.9, f"frac_highconf should be near 1 on collapse; got {frac_hi}"

    def test_frac_highconf_low_on_uniform(self):
        """Uniform distribution -> no high-confidence predictions."""
        n, C = 64, 10
        pa = np.full((n, C), 1.0 / C)
        p0 = make_probs(n_samples=n, n_classes=C, seed=13)
        z = evidence_vector(p0, pa)
        assert z[8] == 0.0, f"frac_highconf should be 0 for uniform pa; got {z[8]}"

    def test_marginal_kl_zero_when_same(self):
        """If p0 == pa, marginal KL should be ~0."""
        p0 = make_probs(seed=14)
        z = evidence_vector(p0, p0)
        assert abs(z[9]) < 1e-10, f"marginal_KL should be ~0 when p0==pa; got {z[9]}"

    def test_pre_entropy_positive(self):
        """Mean entropy of a distribution over 10 classes should be > 0."""
        p0 = make_probs(seed=15)
        pa = make_probs(seed=16)
        z = evidence_vector(p0, pa)
        assert z[0] > 0, "pre_entropy must be positive"

    def test_pre_conf_in_01(self):
        p0 = make_probs(seed=17)
        pa = make_probs(seed=18)
        z = evidence_vector(p0, pa)
        assert 0.0 <= z[1] <= 1.0, f"pre_conf out of [0,1]: {z[1]}"
        assert 0.0 <= z[4] <= 1.0, f"post_conf out of [0,1]: {z[4]}"

    def test_pbal_in_01(self):
        """Normalised marginal entropy is in [0,1]."""
        p0 = make_probs(seed=19)
        pa = make_probs(seed=20)
        z = evidence_vector(p0, pa)
        assert 0.0 <= z[2] <= 1.0 + 1e-9, f"pre_pbal out of [0,1]: {z[2]}"
        assert 0.0 <= z[5] <= 1.0 + 1e-9, f"post_pbal out of [0,1]: {z[5]}"

    def test_large_update_norm_stored(self):
        p0 = make_probs()
        pa = make_probs(seed=21)
        z = evidence_vector(p0, pa, upd_norm=99.9)
        assert z[10] == pytest.approx(99.9)
