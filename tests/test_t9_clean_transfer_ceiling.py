"""Unit tests for T9 — clean-transfer reliability-gate impossibility.

Pins the load-bearing claims of the proof so a future edit cannot silently
weaken them:
  (i)   the sharp ceiling: Delta* <= eps_subopt = A* - A(CW), tighter than 1-A(CW);
  (ii)  homoscedastic closure: equal-weight CW equals the LDA-optimal A* when the
        modalities are exchangeable (the redundant regime);
  (iii) the Gate-E-unpassable certificate fires when eps_subopt < MDE;
  (iv)  the constructive validator certifies BOTH regimes unpassable.
"""

from __future__ import annotations

import numpy as np

from elara.theory.t9_clean_transfer_ceiling import (
    auc_empirical,
    ceiling_headroom,
    gate_e_unpassable_certificate,
    linear_rule_auc_gaussian,
    min_detectable_effect,
    neyman_pearson_auc_gaussian,
    suboptimality_gap,
    validate_t9,
)


def test_neyman_pearson_auc_matches_mahalanobis_form():
    """A* = Phi(M/sqrt 2). For mu0=0, mu1=(1,1), Sigma=I: M=sqrt2, A*=Phi(1)=0.8413."""
    a = neyman_pearson_auc_gaussian(np.zeros(2), np.ones(2), np.eye(2))
    assert abs(a - 0.8413) < 1e-3


def test_homoscedastic_closure_equal_weight_cw_is_optimal_when_exchangeable():
    """Exchangeable modalities (equal means, symmetric Sigma): the equal-weight
    CW direction IS the LDA direction, so A(CW) == A* exactly (eps_subopt=0)."""
    mu0, mu1 = np.zeros(2), np.array([1.0, 1.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    a_star = neyman_pearson_auc_gaussian(mu0, mu1, Sigma)
    a_cw = linear_rule_auc_gaussian(mu0, mu1, Sigma, np.array([0.5, 0.5]))
    assert abs(a_cw - a_star) < 1e-9
    assert suboptimality_gap(a_star, a_cw) == 0.0


def test_sharp_ceiling_is_tighter_than_headroom_to_one():
    """eps_subopt = A* - A(CW) must be <= the naive headroom 1 - A(CW)."""
    a_cw, a_star = 0.95, 0.96
    assert suboptimality_gap(a_star, a_cw) <= ceiling_headroom(a_cw) + 1e-12
    assert abs(suboptimality_gap(a_star, a_cw) - 0.01) < 1e-9  # only this slice recoverable


def test_certificate_unpassable_when_oracle_cannot_beat_cw():
    """Near-ceiling CW with an oracle that does not beat it -> Gate E unpassable."""
    cert = gate_e_unpassable_certificate(auc_cw=0.997, auc_oracle=0.9965,
                                         n_pos=300, n_neg=600)
    assert cert["eps_subopt_recoverable"] == 0.0  # oracle below CW -> clipped to 0
    assert cert["gate_e_unpassable"] is True


def test_certificate_inconclusive_when_oracle_is_underpowered():
    """2026-06-02 audit C3 guard: a weak/sabotaged oracle (detectably worse than
    CW) must yield INCONCLUSIVE (None), NOT a false 'unpassable'. Blocks
    manufacturing the impossibility by under-powering the oracle."""
    cert = gate_e_unpassable_certificate(auc_cw=0.93, auc_oracle=0.55,
                                         n_pos=600, n_neg=600)
    assert cert["gate_e_unpassable"] is None
    assert cert["oracle_is_credible_ceiling"] is False
    assert "INCONCLUSIVE" in cert["reason"]


def test_certificate_unpassable_requires_credible_oracle():
    """A credible oracle (within MDE of CW) with ~no recoverable gap -> unpassable."""
    cert = gate_e_unpassable_certificate(auc_cw=0.935, auc_oracle=0.934,
                                         n_pos=624, n_neg=754)
    assert cert["oracle_is_credible_ceiling"] is True
    assert cert["gate_e_unpassable"] is True


def test_certificate_open_when_large_recoverable_gap_and_enough_samples():
    """A big optimality gap with plenty of samples is NOT closed by T9."""
    cert = gate_e_unpassable_certificate(auc_cw=0.70, auc_oracle=0.85,
                                         n_pos=5000, n_neg=5000, rho=0.5)
    assert cert["eps_subopt_recoverable"] > cert["min_detectable_effect"]
    assert cert["gate_e_unpassable"] is False


def test_mde_positive_and_shrinks_with_n():
    """Minimum detectable effect must be positive and decrease with sample size."""
    small = min_detectable_effect(0.95, 100, 100)
    large = min_detectable_effect(0.95, 5000, 5000)
    assert small > large > 0


def test_auc_empirical_matches_known_ordering():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    # pairs: (0.35 vs 0.1)=1, (0.35 vs 0.4)=0, (0.8 vs 0.1)=1, (0.8 vs 0.4)=1 -> 3/4
    assert abs(auc_empirical(s, y) - 0.75) < 1e-9


def test_validate_t9_certifies_both_regimes_unpassable():
    out = validate_t9(seed=7)
    assert out["ceiling_bound_holds"] is True
    assert out["homoscedastic_cw_is_optimal"] is True
    assert out["gate_e_unpassable_all_regimes"] is True
    assert out["all_ok"] is True
