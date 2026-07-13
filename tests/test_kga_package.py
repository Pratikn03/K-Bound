"""Behavioural contract for the top-level ``kga`` package.

Deterministic, numpy-only, fixed-seed tests (no torch, no external data files).
They exercise the public API and the K-Bound guarantees:

* identical distributions  -> ~0 drift evidence and ABSTAIN (no certifiable benefit);
* the certificate radius shrinks as the sample size n grows;
* the trichotomy boundaries (ADAPT / FREEZE / ABSTAIN) are hit exactly;
* a non-identifiability witness (identical Z, opposite truth) -> ABSTAIN;
* the empirical false-adapt rate stays <= alpha over >= 2000 synthetic trials
  (Theorem 3, ``thm:cert``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from kga import KGA, Certificate, Decision, Evidence, __version__
from kga.certificate import (
    conformal_split,
    empirical_bernstein,
    evalue_anytime,
    hoeffding,
)
from kga.evidence import compute_evidence
from kga.policy import decide


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------
def test_version_string():
    assert __version__ == "0.1.0"


def test_public_exports_are_usable():
    assert issubclass(Decision, object)
    assert {d.value for d in Decision} == {"ADAPT", "FREEZE", "ABSTAIN"}
    # Certificate / Evidence / KGA are constructible / referenceable.
    cert = Certificate(delta_hat=0.1, epsilon=0.05, method="ebern", alpha=0.1, n=10)
    assert cert.lower == pytest.approx(0.05)
    assert cert.upper == pytest.approx(0.15)
    assert KGA(alpha=0.1).alpha == 0.1
    assert Evidence is not None


# ---------------------------------------------------------------------------
# Evidence: identical distributions => ~0 drift
# ---------------------------------------------------------------------------
class TestEvidenceIdentical:
    def test_identical_distributions_zero_drift(self):
        """Same calib/test law -> KS drift, entropy shift, conf shift all ~ 0."""
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, size=(4000, 3))
        # Two independent draws from the SAME distribution.
        calib = base
        test = rng.normal(0.0, 1.0, size=(4000, 3))
        z = compute_evidence(calib, test)
        assert z.ks_mean < 0.05
        assert z.ks_max < 0.08
        assert abs(z.entropy_shift) < 0.2
        assert abs(z.conf_shift) < 0.1
        # No shift -> importance weights nearly uniform -> ESS ~ full sample.
        assert z.ess_frac > 0.8

    def test_strong_drift_detected(self):
        """A large location shift drives KS drift up and ESS fraction down."""
        rng = np.random.default_rng(1)
        calib = rng.normal(0.0, 1.0, size=(2000, 1))
        test = rng.normal(4.0, 1.0, size=(2000, 1))  # big covariate shift
        z = compute_evidence(calib, test)
        assert z.ks_mean > 0.5
        assert z.ess_frac < 0.5

    def test_single_detector_disagreement_zero(self):
        rng = np.random.default_rng(2)
        calib = rng.normal(size=100)
        test = rng.normal(size=100)
        z = compute_evidence(calib, test)
        assert z.disagree == 0.0
        assert z.n_detectors == 1

    def test_constant_multidetector_disagreement_is_finite(self):
        calib = np.zeros((100, 2))
        test = np.zeros((100, 2))
        z = compute_evidence(calib, test)
        assert np.isfinite(z.disagree)
        assert z.disagree == 0.0

    def test_constant_and_variable_detector_disagreement_is_finite(self):
        variable = np.linspace(0.0, 1.0, 100)
        calib = np.column_stack([np.zeros(100), variable])
        test = np.column_stack([np.zeros(100), variable[::-1]])
        z = compute_evidence(calib, test)
        assert np.isfinite(z.disagree)
        assert z.disagree == pytest.approx(1.0)

    def test_mismatched_detectors_raises(self):
        with pytest.raises(ValueError):
            compute_evidence(np.zeros((10, 2)), np.zeros((10, 3)))

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError):
            compute_evidence(np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Certificate radius shrinks as n grows
# ---------------------------------------------------------------------------
class TestCertificateRadius:
    def test_ebern_radius_shrinks_with_n(self):
        rng = np.random.default_rng(10)
        small = rng.normal(0.2, 0.1, size=20)
        large = rng.normal(0.2, 0.1, size=5000)
        eps_small = empirical_bernstein(small, alpha=0.1, benefit_range=2.0).epsilon
        eps_large = empirical_bernstein(large, alpha=0.1, benefit_range=2.0).epsilon
        assert eps_large < eps_small

    def test_hoeffding_radius_shrinks_with_n(self):
        rng = np.random.default_rng(11)
        small = rng.normal(0.2, 0.1, size=20)
        large = rng.normal(0.2, 0.1, size=5000)
        eps_small = hoeffding(small, alpha=0.1, benefit_range=2.0).epsilon
        eps_large = hoeffding(large, alpha=0.1, benefit_range=2.0).epsilon
        assert eps_large < eps_small

    def test_ebern_tighter_than_hoeffding_low_variance(self):
        """In the low-variance regime, empirical-Bernstein beats Hoeffding."""
        x = np.full(1000, 0.2)  # zero variance
        eps_eb = empirical_bernstein(x, alpha=0.1, benefit_range=2.0).epsilon
        eps_hoeff = hoeffding(x, alpha=0.1, benefit_range=2.0).epsilon
        assert eps_eb < eps_hoeff

    def test_conformal_radius_is_quantile(self):
        rng = np.random.default_rng(12)
        r = np.abs(rng.standard_normal(500))
        cert = conformal_split(0.2, r, alpha=0.1)
        k = min(len(r), int(np.ceil((len(r) + 1) * 0.9)))
        assert cert.epsilon == pytest.approx(float(np.sort(r)[k - 1]))

    def test_conformal_radius_shrinks_with_smaller_residuals(self):
        rng = np.random.default_rng(13)
        big = np.abs(rng.normal(0, 1.0, size=2000))
        small = np.abs(rng.normal(0, 0.1, size=2000))
        assert conformal_split(0.0, small, alpha=0.1).epsilon < conformal_split(0.0, big, alpha=0.1).epsilon

    def test_single_sample_is_uninformative_abstain(self):
        """A 1-sample empirical-Bernstein certificate has infinite radius and
        therefore ABSTAINs (no variance estimate is possible)."""
        cert = empirical_bernstein(np.array([0.5]), alpha=0.1, benefit_range=2.0)
        assert not np.isfinite(cert.epsilon)
        assert decide(cert) == Decision.ABSTAIN

    def test_ebern_lcb_matches_canonical_formula(self):
        """delta_hat - epsilon reproduces the Maurer-Pontil LCB exactly."""
        import math

        rng = np.random.default_rng(14)
        x = rng.uniform(0.0, 0.4, size=300)
        alpha = 0.05
        cert = empirical_bernstein(x, alpha=alpha, benefit_range=2.0)
        n = x.size
        mean = float(x.mean())
        var = float(x.var(ddof=1))
        ln_term = math.log(2.0 / alpha)
        expected_lcb = mean - math.sqrt(2.0 * var * ln_term / n) - 7.0 * 2.0 * ln_term / (3.0 * (n - 1))
        assert cert.lower == pytest.approx(expected_lcb, abs=1e-12)


# ---------------------------------------------------------------------------
# decide() boundaries hit exactly
# ---------------------------------------------------------------------------
class TestDecideBoundaries:
    def _cert(self, delta_hat: float, epsilon: float) -> Certificate:
        return Certificate(delta_hat=delta_hat, epsilon=epsilon, method="ebern", alpha=0.1, n=100)

    def test_adapt(self):
        assert decide(self._cert(0.15, 0.05)) == Decision.ADAPT

    def test_freeze(self):
        assert decide(self._cert(-0.15, 0.05)) == Decision.FREEZE

    def test_abstain_interior(self):
        assert decide(self._cert(0.03, 0.10)) == Decision.ABSTAIN

    def test_boundary_lower_zero_abstains(self):
        # delta_hat - epsilon == 0 exactly -> ABSTAIN (strict inequality).
        assert decide(self._cert(0.05, 0.05)) == Decision.ABSTAIN

    def test_boundary_upper_zero_abstains(self):
        # delta_hat + epsilon == 0 exactly -> ABSTAIN.
        assert decide(self._cert(-0.05, 0.05)) == Decision.ABSTAIN

    def test_just_past_lower_boundary_adapts(self):
        assert decide(self._cert(0.0500001, 0.05)) == Decision.ADAPT

    def test_just_past_upper_boundary_freezes(self):
        assert decide(self._cert(-0.0500001, 0.05)) == Decision.FREEZE

    def test_zero_epsilon_sign(self):
        assert decide(self._cert(1e-9, 0.0)) == Decision.ADAPT
        assert decide(self._cert(-1e-9, 0.0)) == Decision.FREEZE

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            decide(self._cert(0.1, -0.01))

    def test_alpha_mismatch_raises(self):
        with pytest.raises(ValueError):
            decide(self._cert(0.1, 0.05), alpha=0.2)


# ---------------------------------------------------------------------------
# Identical distributions => ABSTAIN through the full KGA pipeline
# ---------------------------------------------------------------------------
class TestKGAIdenticalAbstain:
    def test_zero_benefit_abstains(self):
        """No benefit signal (paired benefits centred at 0) -> ABSTAIN."""
        rng = np.random.default_rng(20)
        kga = KGA(alpha=0.1, method="ebern")
        benefits = rng.normal(0.0, 0.1, size=300)  # E[X] = 0 -> not certifiable
        cert = kga.certify(scores=benefits, benefit_range=2.0)
        assert kga.decide(cert) == Decision.ABSTAIN

    def test_strong_positive_benefit_adapts(self):
        rng = np.random.default_rng(21)
        kga = KGA(alpha=0.1, method="ebern")
        benefits = rng.normal(0.3, 0.05, size=400)
        cert = kga.certify(scores=benefits, benefit_range=2.0)
        assert kga.decide(cert) == Decision.ADAPT

    def test_strong_negative_benefit_freezes(self):
        rng = np.random.default_rng(22)
        kga = KGA(alpha=0.1, method="ebern")
        benefits = rng.normal(-0.3, 0.05, size=400)
        cert = kga.certify(scores=benefits, benefit_range=2.0)
        assert kga.decide(cert) == Decision.FREEZE

    def test_explain_contains_all_stages(self):
        rng = np.random.default_rng(23)
        kga = KGA(alpha=0.1, method="ebern")
        kga.evidence(rng.normal(size=(100, 2)), rng.normal(size=(100, 2)))
        kga.certify(scores=rng.normal(0.3, 0.05, size=200), benefit_range=2.0)
        kga.decide()
        info = kga.explain()
        assert info["evidence"] is not None
        assert info["certificate"] is not None
        assert info["decision"] in {"ADAPT", "FREEZE", "ABSTAIN"}
        # Must be JSON-serialisable.
        json.dumps(info)

    def test_conformal_convention_risks(self):
        """certify(adapt_risk, freeze_risk, calib_residuals) uses split-conformal."""
        rng = np.random.default_rng(24)
        kga = KGA(alpha=0.1)
        residuals = np.abs(rng.normal(0, 0.02, size=500))  # tight residuals
        cert = kga.certify(adapt_risk=0.10, freeze_risk=0.40, calib_residuals=residuals)
        # delta_hat = freeze_risk - adapt_risk = 0.30 >> epsilon -> ADAPT
        assert cert.method == "conformal"
        assert cert.delta_hat == pytest.approx(0.30)
        assert kga.decide(cert) == Decision.ADAPT

    def test_certify_requires_valid_convention(self):
        kga = KGA(alpha=0.1)
        with pytest.raises(ValueError):
            kga.certify()  # nothing supplied
        with pytest.raises(ValueError):
            kga.certify(delta_hat=0.1)  # missing residuals


# ---------------------------------------------------------------------------
# Anytime e-value certificate
# ---------------------------------------------------------------------------
class TestEValue:
    def test_strong_positive_stream_adapts(self):
        x = np.full(2000, 0.3)
        cert = evalue_anytime(x, alpha=0.1)
        assert cert.method == "evalue"
        assert cert.lower > 0
        assert decide(cert) == Decision.ADAPT

    def test_strong_negative_stream_freezes(self):
        x = np.full(2000, -0.3)
        cert = evalue_anytime(x, alpha=0.1)
        assert cert.upper < 0
        assert decide(cert) == Decision.FREEZE

    def test_zero_stream_abstains(self):
        x = np.zeros(2000)
        cert = evalue_anytime(x, alpha=0.1)
        assert decide(cert) == Decision.ABSTAIN


# ---------------------------------------------------------------------------
# Non-identifiability witness: identical Z, opposite truth => ABSTAIN
# ---------------------------------------------------------------------------
class TestNonIdentifiability:
    def test_identical_evidence_opposite_truth_both_abstain(self):
        """Two worlds with the SAME label-free evidence Z but OPPOSITE true
        benefit. Because Z is identical, any Z-only estimator gives the same
        Delta_hat for both; with a radius that brackets zero, both ABSTAIN --
        the gate cannot (and should not) commit (Theorem 1, thm:imp).
        """
        rng = np.random.default_rng(30)
        # Same observable scores => identical evidence in both worlds.
        calib = rng.normal(0.0, 1.0, size=(1000, 2))
        test = rng.normal(0.0, 1.0, size=(1000, 2))
        z_world_a = compute_evidence(calib, test)
        z_world_b = compute_evidence(calib, test)
        # Evidence is byte-for-byte identical.
        assert z_world_a.to_vector().tolist() == z_world_b.to_vector().tolist()

        # A Z-only benefit estimator must return the same delta_hat for both
        # worlds. We emulate "no separating signal" with delta_hat = 0 and a
        # conformal radius from the (shared) calibration spread.
        residuals = np.abs(calib.ravel() - np.median(calib.ravel()))
        cert_a = conformal_split(0.0, residuals, alpha=0.1)
        cert_b = conformal_split(0.0, residuals, alpha=0.1)
        assert decide(cert_a) == Decision.ABSTAIN
        assert decide(cert_b) == Decision.ABSTAIN

    def test_witness_through_kga_facade(self):
        rng = np.random.default_rng(31)
        kga = KGA(alpha=0.1)
        residuals = np.abs(rng.normal(0, 0.3, size=500))
        # Opposite hidden truths but identical observable estimate (0) + radius.
        cert = kga.certify(delta_hat=0.0, calib_residuals=residuals)
        assert kga.decide(cert) == Decision.ABSTAIN


# ---------------------------------------------------------------------------
# Empirical false-adapt rate <= alpha over >= 2000 trials (Theorem 3)
# ---------------------------------------------------------------------------
class TestFalseAdaptRate:
    @pytest.mark.parametrize("delta", [0.0, -0.05, -0.2])
    def test_false_adapt_rate_at_or_below_alpha(self, delta):
        """Under H0 (Delta <= 0), P(ADAPT) <= alpha for the empirical-Bernstein
        certificate. delta=0 is the hardest boundary case.

        Each trial draws n bounded paired benefits with mean = delta, builds the
        certificate, and decides. We assert the ADAPT fraction is <= alpha (with
        a small Monte-Carlo slack).
        """
        rng = np.random.default_rng(40)
        alpha = 0.1
        n_trials = 2000
        n = 200
        a, b = -1.0, 1.0
        kga = KGA(alpha=alpha, method="ebern")

        n_adapt = 0
        for _ in range(n_trials):
            # Two-point benefits in {a, b} with E[X] = delta (worst case for
            # boundedness; matches stream_twopoint in val_thm3_evalue.py).
            p_b = (delta - a) / (b - a)
            x = np.where(rng.random(n) < p_b, b, a).astype(float)
            cert = kga.certify(scores=x, benefit_range=2.0)
            if kga.decide(cert) == Decision.ADAPT:
                n_adapt += 1

        false_adapt_rate = n_adapt / n_trials
        # Formal bound is <= alpha; allow 1.5% MC slack above alpha.
        assert false_adapt_rate <= alpha + 0.015, (
            f"false-adapt rate {false_adapt_rate:.4f} exceeds alpha={alpha} (+slack) at delta={delta}"
        )

    def test_power_under_h1(self):
        """Sanity: with a clearly positive benefit and enough data, KGA ADAPTs
        on the large majority of trials (the certificate is not vacuous)."""
        rng = np.random.default_rng(41)
        alpha = 0.1
        kga = KGA(alpha=alpha, method="ebern")
        n_trials = 500
        n = 800
        delta = 0.3
        a, b = -1.0, 1.0
        n_adapt = 0
        for _ in range(n_trials):
            p_b = (delta - a) / (b - a)
            x = np.where(rng.random(n) < p_b, b, a).astype(float)
            if kga.decide(kga.certify(scores=x, benefit_range=2.0)) == Decision.ADAPT:
                n_adapt += 1
        assert n_adapt / n_trials > 0.8
