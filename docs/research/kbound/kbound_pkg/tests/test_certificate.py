"""Tests for kbound.certificate.

Thm thm:cert (K-Bound paper):
    conformal_radius guarantees >= 1 - alpha empirical coverage on held-out data.
"""

import math
import numpy as np
import pytest

from kbound.certificate import conformal_radius, empirical_bernstein_lcb, decide


# ---------------------------------------------------------------------------
# conformal_radius
# ---------------------------------------------------------------------------

class TestConformalRadius:
    """Coverage guarantee: the conformal radius should achieve ~ 1-alpha coverage."""

    def test_coverage_guarantee(self):
        """Thm thm:cert: P(|Bhat_test - B_test| <= eps) >= 1-alpha.

        We simulate: fit eps on a calibration set, then check that the
        test residual is covered with frequency >= 1 - alpha.
        """
        rng = np.random.default_rng(42)
        alpha = 0.10
        n_cal = 300
        n_rep = 500
        covered = 0
        for _ in range(n_rep):
            # Simulate calibration residuals (|Bhat - B|)
            cal_res = rng.exponential(scale=0.3, size=n_cal)
            test_res = rng.exponential(scale=0.3)   # fresh test point
            eps = conformal_radius(cal_res, alpha=alpha)
            if test_res <= eps:
                covered += 1
        # Empirical coverage >= 1 - alpha (allow a very small slack for MC noise)
        empirical_cov = covered / n_rep
        assert empirical_cov >= (1 - alpha) - 0.03, (
            f"Coverage {empirical_cov:.3f} < {1-alpha - 0.03:.3f} -- "
            "conformal_radius does not achieve the promised coverage"
        )

    def test_quantile_identity(self):
        """eps is the exact (1-alpha)-quantile of the residuals."""
        rng = np.random.default_rng(0)
        r = rng.standard_normal(200) ** 2   # chi-squared-like residuals
        alpha = 0.15
        eps = conformal_radius(r, alpha)
        assert abs(eps - float(np.quantile(r, 1 - alpha))) < 1e-12

    def test_output_positive(self):
        r = np.abs(np.random.default_rng(1).standard_normal(100))
        assert conformal_radius(r, 0.1) > 0

    def test_larger_alpha_smaller_eps(self):
        """More miscoverage allowed -> smaller radius."""
        rng = np.random.default_rng(2)
        r = rng.uniform(0, 1, 200)
        eps1 = conformal_radius(r, 0.05)
        eps2 = conformal_radius(r, 0.20)
        assert eps2 < eps1

    def test_invalid_alpha(self):
        r = np.ones(10)
        with pytest.raises(ValueError):
            conformal_radius(r, alpha=0.0)
        with pytest.raises(ValueError):
            conformal_radius(r, alpha=1.1)

    def test_empty_residuals(self):
        with pytest.raises(ValueError):
            conformal_radius(np.array([]), alpha=0.1)

    def test_single_value(self):
        """Single residual: eps = that value (since quantile(r, 0.9) of [x] is x)."""
        eps = conformal_radius(np.array([0.5]), 0.1)
        assert eps == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# empirical_bernstein_lcb
# ---------------------------------------------------------------------------

class TestEmpiricalBernsteinLCB:
    """Maurer-Pontil LCB -- should be below mean with probability >= 1 - alpha."""

    def test_lcb_below_mean(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(0.1, 0.9, 150)
        lcb = empirical_bernstein_lcb(x, alpha=0.05)
        assert lcb < x.mean(), "LCB should be strictly below sample mean"

    def test_coverage_mp(self):
        """P(true_mean >= LCB) >= 1 - alpha over random samples."""
        rng = np.random.default_rng(7)
        true_mean = 0.4
        alpha = 0.05
        n_rep = 500
        covered = sum(
            1 for _ in range(n_rep)
            if empirical_bernstein_lcb(rng.uniform(0, 0.8, 120), alpha) <= true_mean
        )
        cov = covered / n_rep
        # Should be >= 1 - alpha; allow a 4% MC tolerance
        assert cov >= (1 - alpha) - 0.04, (
            f"LCB coverage {cov:.3f} is too low (expected >= {1-alpha-0.04:.3f})"
        )

    def test_larger_sample_tighter_bound(self):
        rng = np.random.default_rng(9)
        x_small = rng.uniform(0.2, 0.6, 20)
        x_large = rng.uniform(0.2, 0.6, 1000)
        lcb_small = empirical_bernstein_lcb(x_small, 0.05)
        lcb_large = empirical_bernstein_lcb(x_large, 0.05)
        assert lcb_large > lcb_small, "Larger sample should give a tighter (higher) LCB"

    def test_requires_2_samples(self):
        with pytest.raises(ValueError):
            empirical_bernstein_lcb(np.array([0.5]), 0.1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            empirical_bernstein_lcb(np.ones(10), alpha=1.5)


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

class TestDecide:
    def test_adapt(self):
        assert decide(0.15, 0.05) == "adapt"

    def test_freeze(self):
        assert decide(-0.15, 0.05) == "freeze"

    def test_abstain_small_eps(self):
        assert decide(0.03, 0.10) == "abstain"

    def test_abstain_at_boundary_plus(self):
        # Bhat - eps == 0 -> abstain (strict inequality)
        assert decide(0.05, 0.05) == "abstain"

    def test_abstain_at_boundary_minus(self):
        # Bhat + eps == 0 -> abstain
        assert decide(-0.05, 0.05) == "abstain"

    def test_zero_eps_adapts_if_positive(self):
        assert decide(0.001, 0.0) == "adapt"

    def test_zero_eps_freezes_if_negative(self):
        assert decide(-0.001, 0.0) == "freeze"
