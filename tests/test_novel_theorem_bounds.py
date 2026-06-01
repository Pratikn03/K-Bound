"""Unit tests for the novel theorem bounds (T2/T3/T4/T6/GDR).

Pins the *corrected* derivations so a future edit can't silently reintroduce
the proposal's two errors (T3 dropped-mean term; GDR claimed p<0.05).
"""

from __future__ import annotations

import numpy as np

from elara.theory.novel_theorem_bounds import (
    bayes_optimal_tau,
    exact_binomial_p,
    ks_window_for_power,
    ks_window_power,
    mixture_ks_inflation_bound,
    stochastic_dilution_prob,
)


def test_t3_deterministic_boundary_matches_k_star():
    """sigma=0: gate silent for k<=1, fires for k>=2 at D=4, tau=0.66 (k*=1.36)."""
    assert stochastic_dilution_prob(0, 4, 0.66, 0.0) == 1.0
    assert stochastic_dilution_prob(1, 4, 0.66, 0.0) == 1.0   # mean rel 0.75 >= 0.66
    assert stochastic_dilution_prob(2, 4, 0.66, 0.0) == 0.0   # mean rel 0.50 < 0.66


def test_t3_carries_full_mean_term_not_dropped():
    """Regression guard: P(silent|k) must use mu_k=(D-k)/D, not a dropped term.
    At k=2, D=4 the mean is exactly 0.5; with tau=0.5 it sits on the boundary
    -> P(silent)=0.5 under any noise (symmetry). A dropped-mean bug breaks this."""
    p = stochastic_dilution_prob(2, 4, 0.5, sigma=0.1)
    assert abs(p - 0.5) < 1e-6


def test_t3_noise_erodes_boundary_monotonically():
    """More noise -> more silence leakage at the just-below-threshold k=1.
    (k=1 mean reliability 0.75 > tau 0.66, so the gate SHOULD stay silent; noise
    lets it occasionally fire, lowering P(silent).)"""
    p_lo = stochastic_dilution_prob(1, 4, 0.66, 0.05)
    p_hi = stochastic_dilution_prob(1, 4, 0.66, 0.30)
    assert p_lo > 0.999          # tiny noise: essentially always silent
    assert p_hi < p_lo           # more noise -> strictly less silent (monotone)
    assert p_hi < 0.99


def test_t6_power_monotone_and_window_sizing_consistent():
    powers = [ks_window_power(W, 0.15, c=1.0) for W in (32, 64, 128, 256, 512)]
    assert all(b >= a - 1e-9 for a, b in zip(powers, powers[1:]))  # monotone up
    assert powers[-1] > powers[0]
    # W* should land where power crosses the target
    W = ks_window_for_power(0.8, 0.15, c=1.0)
    assert ks_window_power(int(W) + 2, 0.15, c=1.0) >= 0.79


def test_t4_tau_monotone_in_prevalence():
    """tau* = 1 - (1-pi) q0 d0 / (pi dq1 d1): as degradation prevalence pi rises,
    the clean false-fire penalty (1-pi) shrinks and the degraded benefit (pi)
    grows, so tau* moves toward 1 (switch more readily). Pin that direction."""
    t_lo = bayes_optimal_tau(0.1, 0.05, 1.0, 0.01, 0.05)
    t_hi = bayes_optimal_tau(0.75, 0.05, 1.0, 0.01, 0.05)
    assert t_hi > t_lo            # higher prevalence -> tau* closer to 1
    assert 0.0 < t_lo < t_hi <= 1.0


def test_gdr_binomial_is_honest_0125_not_significant():
    """4/4 at chance gives p=0.125 -- NOT < 0.05. Pins the honest value."""
    p = exact_binomial_p(4, 4, 0.5)
    assert abs(p - 0.125) < 1e-9
    assert p >= 0.05  # explicitly NOT significant


def test_t2_inflation_bound_nonneg_and_scales_with_tv():
    D = np.array([[0, 0.4, 0.4], [0.4, 0, 0.4], [0.4, 0.4, 0]])
    small = mixture_ks_inflation_bound([1/3, 1/3, 1/3], [0.34, 0.33, 0.33], D)
    large = mixture_ks_inflation_bound([1/3, 1/3, 1/3], [0.8, 0.1, 0.1], D)
    assert small >= 0 and large > small  # bigger mixture shift -> bigger bound
