"""Tests for T4 prevalence sensitivity helpers."""

from __future__ import annotations

from elara.certification.risk_dominance import (
    RiskDominanceTerms,
    dominates_at_prevalence,
    prevalence_sensitivity_rows,
    risk_dominance_margin,
)


def test_risk_dominance_margin_matches_t4_inequality():
    margin = risk_dominance_margin(pi=0.1, q0=0.01, q1=0.9, delta_0=0.02, delta_1=0.05)
    assert margin > 0.0
    assert dominates_at_prevalence(pi=0.1, q0=0.01, q1=0.9, delta_0=0.02, delta_1=0.05)


def test_zero_attack_b1_terms_do_not_dominate_at_positive_prevalence():
    terms = RiskDominanceTerms(
        gate_id="G0_mean_tau66",
        scenario_id="zero_attack_k4",
        q0=0.0,
        q1=0.999375,
        delta_0=0.0,
        delta_1=-0.003857,
        pi_star=float("nan"),
        n_clean_samples=1600,
        n_degraded_samples=1600,
        notes="",
    )
    rows = prevalence_sensitivity_rows(terms, pi_values=(0.01, 0.05, 0.1))
    assert all(not row["dominates"] for row in rows)


def test_max_attack_b2_terms_dominate_for_any_positive_prevalence():
    terms = RiskDominanceTerms(
        gate_id="G0_mean_tau66",
        scenario_id="max_attack_k4",
        q0=0.0,
        q1=1.0,
        delta_0=0.0,
        delta_1=0.009461,
        pi_star=0.0,
        n_clean_samples=1600,
        n_degraded_samples=1600,
        notes="",
    )
    rows = prevalence_sensitivity_rows(terms, pi_values=(0.01, 0.05, 0.1))
    assert all(row["dominates"] for row in rows)
