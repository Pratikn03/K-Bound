"""Deterministic checks for the Phase-3 / T6 calibration-transfer study.

These lock in the qualitative finding: a validation-calibrated gate that helps
in-distribution becomes harmful under score-distribution shift, the source-side
signals are blind to it, and the label-free target-divergence abstention rule
covers the harmful region.
"""

from __future__ import annotations

from src.scripts.run_calibration_transfer_study import run_study


def test_help_and_hurt_regimes_both_present():
    report = run_study(seed=0)
    regimes = {r["regime"] for r in report["rows"]}
    assert "HELP" in regimes  # gate helps in-distribution
    assert "HURT" in regimes  # gate hurts under transfer drift


def test_help_occurs_at_low_divergence_and_hurt_at_higher():
    report = run_study(seed=0)
    help_div = [r["target_reference_divergence"] for r in report["rows"] if r["regime"] == "HELP"]
    hurt_div = [r["target_reference_divergence"] for r in report["rows"] if r["regime"] == "HURT"]
    assert max(help_div) < min(hurt_div)  # a separating threshold exists


def test_source_side_signals_are_blind_to_transfer_failure():
    report = run_study(seed=0)
    # Neither source-side predictor can separate HELP from HURT.
    assert report["drift_coherence_separates_help_from_hurt"] is False
    assert report["source_certificate_separates_help_from_hurt"] is False


def test_divergence_abstention_covers_all_hurt():
    report = run_study(seed=0)
    assert report["abstention_divergence_threshold"] is not None
    assert report["abstention_covers_all_hurt"] is True


def test_study_is_labeled_exploratory():
    report = run_study(seed=0)
    assert "EXPLORATORY" in report["label"]
