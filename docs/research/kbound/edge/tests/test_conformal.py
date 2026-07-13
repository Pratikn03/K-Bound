"""test_conformal -- split-conformal radius uses the CONFORMAL split only, alpha=0.10."""

import math

import numpy as np
import pytest

from kbound_edge.conformal import (
    conservative_conformal_radius, calibrate_conformal, ConformalRadius, ALPHA,
)


def test_alpha_is_fixed_at_0_10():
    assert ALPHA == 0.10


class TestConservativeRank:
    def test_is_kth_order_statistic(self):
        r = np.arange(1, 21, dtype=float) / 20.0          # 0.05 .. 1.0
        n = len(r)
        k = math.ceil((n + 1) * (1 - 0.10))               # conservative rank
        eps = conservative_conformal_radius(r, 0.10)
        assert eps == pytest.approx(np.sort(r)[k - 1])

    def test_infinite_when_too_few_points(self):
        # n=5, k=ceil(6*0.9)=6 > 5 -> cannot certify -> +inf
        eps = conservative_conformal_radius(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.10)
        assert math.isinf(eps)

    def test_smaller_alpha_gives_larger_radius(self):
        r = np.random.default_rng(0).random(100)
        assert conservative_conformal_radius(r, 0.05) >= conservative_conformal_radius(r, 0.20)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            conservative_conformal_radius(np.array([]), 0.10)


class _SpyEstimator:
    """Records the sizes of every fit/predict call so we can prove split hygiene."""
    def __init__(self):
        self.calls = []

    def fit(self, Z, B):
        self.calls.append(("fit", len(Z)))
        return self

    def predict(self, Z):
        self.calls.append(("predict", len(Z)))
        return np.zeros(len(Z))


class TestSplitHygiene:
    def test_residuals_come_only_from_conformal_split(self):
        rng = np.random.default_rng(0)
        Z_fit, B_fit = rng.standard_normal((40, 14)), rng.standard_normal(40)
        Z_conf, B_conf = rng.standard_normal((30, 14)), rng.standard_normal(30)

        est = _SpyEstimator().fit(Z_fit, B_fit)
        cr = calibrate_conformal(est, Z_conf, B_conf, alpha=ALPHA)

        assert isinstance(cr, ConformalRadius)
        # residuals are exactly |pred(Z_conf) - B_conf| -- one per conformal point
        assert len(cr.residuals) == len(B_conf) == 30
        # predict was called on the conformal split (30), never on the fit split (40)
        assert ("predict", 30) in est.calls
        assert ("predict", 40) not in est.calls
        assert cr.method == "conservative"
        assert cr.n_conformal == 30

    def test_marginal_coverage_at_least_1_minus_alpha(self):
        rng = np.random.default_rng(1)
        alpha = 0.10

        class Mean0:
            def predict(self, Z):
                return np.zeros(len(Z))

        covered, trials = 0, 400
        for _ in range(trials):
            B_conf = rng.normal(0.0, 0.3, size=60)
            cr = calibrate_conformal(Mean0(), np.zeros((60, 2)), B_conf, alpha=alpha)
            b_test = rng.normal(0.0, 0.3)         # fresh exchangeable point
            if abs(0.0 - b_test) <= cr.eps:
                covered += 1
        assert covered / trials >= (1 - alpha) - 0.05
