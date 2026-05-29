"""Deterministic checks for the Phase-8 / T5 false-fire vs power study.

These lock in the operating characteristic of the reliability gate: clean
false-fire (cost) and degraded detection (power) are co-monotone in the
threshold, the gate's reliability genuinely separates clean from degraded, the
budget-quantile threshold rule controls out-of-sample false-fire, and benefit is
bounded by the budget (the trade-off itself).
"""

from __future__ import annotations

from src.scripts.run_false_fire_power_study import run_study


def test_cost_and_power_monotone_in_tau():
    report = run_study(seed=0)
    assert report["ffr_monotone_in_tau"]
    assert report["tfr_monotone_in_tau"]


def test_detector_separates_and_power_dominates_cost():
    report = run_study(seed=0)
    assert report["detector_separates"]
    assert report["detector_roc_auc"] > 0.6
    assert report["power_dominates_cost"]


def test_budget_controls_out_of_sample_false_fire():
    report = run_study(seed=0)
    assert report["max_budget_calibration_error"] <= 0.05
    # the realised clean false-fire never grossly exceeds the declared budget
    for r in report["budget_curve"]:
        assert r["test_clean_false_fire"] <= r["budget"] + 0.05


def test_relaxing_budget_buys_detection_power():
    report = run_study(seed=0)
    assert report["detection_power_monotone_in_budget"]
    tfr = [r["test_degraded_detection"] for r in report["budget_curve"]]
    assert tfr[-1] > tfr[0]  # strictly more power at the loosest budget


def test_benefit_is_bounded_by_budget_tradeoff():
    report = run_study(seed=0)
    # tight budget: power-limited, benefit non-positive-ish; loose budget: positive
    assert report["delta_auc_at_tightest_budget"] <= report["delta_auc_at_loosest_budget"]
    assert report["benefit_positive_at_loosest_budget"]


def test_study_is_labelled_exploratory():
    report = run_study(seed=0)
    assert "EXPLORATORY" in report["label"].upper()
