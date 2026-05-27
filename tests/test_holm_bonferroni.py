"""Tests for Holm-Bonferroni multiple-comparison correction."""

from __future__ import annotations

import numpy as np

from uais.utils.stats import holm_bonferroni


def test_all_large_pvalues_no_rejection():
    p = np.array([0.5, 0.6, 0.7])
    result = holm_bonferroni(p, alpha=0.05)
    assert not result["reject"].any()
    assert result["n_tests"] == 3


def test_smallest_pvalue_rejected_with_strict_multiplier():
    # m=4; smallest p=0.01 → 0.01 * 4 = 0.04 ≤ 0.05 → reject
    p = np.array([0.5, 0.01, 0.6, 0.04])
    result = holm_bonferroni(p, alpha=0.05)
    # Only the 0.01 should be rejected; 0.04 needs 0.04*3 = 0.12 > 0.05
    assert result["reject"][1]
    assert not result["reject"][0]
    assert not result["reject"][2]
    assert not result["reject"][3]


def test_monotone_step_down_chain():
    # Step-down chain: smallest p must be rejected before larger ones
    p = np.array([0.001, 0.002, 0.003, 0.5])
    result = holm_bonferroni(p, alpha=0.05)
    # All three small ones pass step-down: 0.001*4=0.004, 0.002*3=0.006, 0.003*2=0.006
    assert result["reject"][:3].all()
    assert not result["reject"][3]


def test_adjusted_pvalues_capped_at_one():
    p = np.array([0.9, 0.95, 0.99])
    result = holm_bonferroni(p, alpha=0.05)
    assert (result["p_adjusted"] <= 1.0 + 1e-12).all()
    assert (result["p_adjusted"] >= p).all()  # adjusted ≥ raw


def test_adjusted_pvalues_monotone_after_sorting():
    p = np.array([0.5, 0.001, 0.3, 0.05])
    result = holm_bonferroni(p, alpha=0.05)
    # Sort by raw p and check adjusted is non-decreasing
    order = np.argsort(p)
    sorted_adj = result["p_adjusted"][order]
    assert (np.diff(sorted_adj) >= -1e-12).all()


def test_nan_handling():
    p = np.array([0.01, np.nan, 0.04, 0.5])
    result = holm_bonferroni(p, alpha=0.05)
    # n_tests should exclude the NaN
    assert result["n_tests"] == 3
    # Smallest of 3 valid: 0.01 * 3 = 0.03 ≤ 0.05 → reject
    assert result["reject"][0]
    assert not result["reject"][1]  # NaN entry never rejected
    assert np.isnan(result["p_adjusted"][1])


def test_empty_input():
    p = np.array([], dtype=float)
    result = holm_bonferroni(p, alpha=0.05)
    assert result["n_tests"] == 0
    assert len(result["reject"]) == 0


def test_alpha_parameter_changes_rejections():
    p = np.array([0.02, 0.03, 0.5])
    result_05 = holm_bonferroni(p, alpha=0.05)
    result_001 = holm_bonferroni(p, alpha=0.01)
    # At α=0.05, 0.02*3 = 0.06 > 0.05, so smallest not rejected
    assert not result_05["reject"][0]
    # At α=0.01, definitely not rejected either
    assert not result_001["reject"][0]
    # But at α=0.10:
    result_10 = holm_bonferroni(p, alpha=0.10)
    assert result_10["reject"][0]
