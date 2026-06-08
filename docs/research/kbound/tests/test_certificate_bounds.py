"""Empirical-Bernstein LCB (Theorem 3 certificate): valid lower bound, tightens with n."""
import numpy as np
from certification.switching_certificate import empirical_bernstein_lcb

def test_lcb_below_mean_and_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(0.2, 0.1, size=200)
    mean, lcb, var = empirical_bernstein_lcb(x, alpha=0.05, benefit_range=2.0)
    assert np.isfinite(lcb) and np.isfinite(mean) and var >= 0
    assert lcb <= mean

def test_lcb_tightens_with_more_samples():
    rng = np.random.default_rng(0)
    small = rng.normal(0.2, 0.1, size=20)
    large = rng.normal(0.2, 0.1, size=2000)
    _, lcb_s, _ = empirical_bernstein_lcb(small, alpha=0.05, benefit_range=2.0)
    _, lcb_l, _ = empirical_bernstein_lcb(large, alpha=0.05, benefit_range=2.0)
    assert lcb_l > lcb_s        # more data -> bound closer to the (positive) mean

def test_empty_returns_nan():
    mean, lcb, var = empirical_bernstein_lcb([], alpha=0.05)
    assert mean != mean and lcb != lcb     # NaN
