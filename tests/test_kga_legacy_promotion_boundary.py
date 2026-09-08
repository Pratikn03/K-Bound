"""Legacy summary flags cannot authorize confirmatory scientific claims."""

from collections import UserDict
from copy import deepcopy

import pytest

from kga.integrations import assess_promotion


def legacy_green_summary() -> dict[str, object]:
    return {
        "mode": "label_free",
        "alpha": 0.05,
        "regret_kga": 0.01,
        "regret_always_adapt": 0.10,
        "regret_always_freeze": 0.08,
        "false_adapt_rate": 0.01,
        "coverage": 0.50,
        "frozen_estimator_verified": True,
        "held_out_natural_datasets": 2,
        "frozen_before_scoring": True,
        "independent_splits": 3,
        "confidence_intervals_complete": True,
        "strong_baselines_complete": True,
        "required_tracks_complete": True,
        "integrity_failures": [],
    }


@pytest.mark.parametrize("mode", ["label_free", "target_label_light"])
def test_all_green_legacy_summary_is_diagnostic_not_promotion(mode: str) -> None:
    summary = legacy_green_summary()
    summary["mode"] = mode
    result = assess_promotion(summary)

    assert result["eligible"] is False
    assert result["legacy_criteria_passed"] is True
    assert result["assessment_scope"] == "legacy_summary_diagnostic_only"
    assert result["reasons"] == ["confirmatory_v2_evidence_not_validated"]
    assert result["claim_kind"] == mode


def test_summary_booleans_do_not_replace_missing_confirmatory_evidence() -> None:
    summary = legacy_green_summary()
    # This deliberately has no FF endpoint, action exposure, numerical paired CI,
    # checkpoint/split identities, protocol seal or outcome-order evidence.
    result = assess_promotion(summary)

    assert result["eligible"] is False
    assert "confirmatory_v2_evidence_not_validated" in result["reasons"]


def test_self_asserted_v2_flags_cannot_upgrade_a_legacy_summary() -> None:
    summary = legacy_green_summary()
    summary.update(
        {
            "protocol_locked": True,
            "false_freeze_control_pass": True,
            "adapt_exposure_pass": True,
            "freeze_exposure_pass": True,
            "strong_success": True,
        }
    )
    assert assess_promotion(summary)["eligible"] is False


@pytest.mark.parametrize("value", [None, [], (), "summary", 1, True])
def test_non_mapping_root_is_rejected_with_value_error(value: object) -> None:
    with pytest.raises(ValueError, match="mapping"):
        assess_promotion(value)


def test_mapping_input_is_not_mutated() -> None:
    summary = UserDict(legacy_green_summary())
    before = deepcopy(summary)
    result = assess_promotion(summary)

    assert summary == before
    assert result["legacy_criteria_passed"] is True
    assert result["eligible"] is False


@pytest.mark.parametrize(
    "field",
    ["alpha", "regret_kga", "regret_always_adapt", "regret_always_freeze", "false_adapt_rate", "coverage"],
)
@pytest.mark.parametrize(
    "value",
    [True, False, "0.01", None, float("nan"), float("inf"), -float("inf"), 10**1000, -(10**1000)],
    ids=["true", "false", "numeric_string", "missing", "nan", "inf", "negative_inf", "huge_int", "huge_negative_int"],
)
def test_malformed_numeric_summary_fails_without_crashing(field: str, value: object) -> None:
    summary = legacy_green_summary()
    summary[field] = value
    result = assess_promotion(summary)

    assert result["eligible"] is False
    assert result["legacy_criteria_passed"] is False
    assert "confirmatory_v2_evidence_not_validated" in result["reasons"]


@pytest.mark.parametrize("field", ["regret_kga", "regret_always_adapt", "regret_always_freeze"])
def test_negative_regret_is_not_valid_legacy_evidence(field: str) -> None:
    summary = legacy_green_summary()
    summary[field] = -0.01
    result = assess_promotion(summary)

    assert result["legacy_criteria_passed"] is False
    assert result["eligible"] is False


@pytest.mark.parametrize("field", ["false_adapt_rate", "coverage"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_rates_outside_unit_interval_fail_legacy_criteria(field: str, value: float) -> None:
    summary = legacy_green_summary()
    summary[field] = value
    result = assess_promotion(summary)

    assert result["legacy_criteria_passed"] is False
    assert result["eligible"] is False


@pytest.mark.parametrize("field", ["held_out_natural_datasets", "independent_splits"])
@pytest.mark.parametrize("value", [True, "3", 3.0, None, -1])
def test_malformed_count_does_not_pass_legacy_requirements(field: str, value: object) -> None:
    summary = legacy_green_summary()
    summary[field] = value
    result = assess_promotion(summary)

    assert result["legacy_criteria_passed"] is False
    assert result["eligible"] is False


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"regret_kga": 0.10}, "does_not_beat_always_adapt_regret"),
        ({"false_adapt_rate": 0.06}, "false_adapt_exceeds_alpha"),
        ({"coverage": 0.19}, "coverage_below_20_percent"),
        ({"integrity_failures": ["bad_hash"]}, "integrity_failure"),
        ({"frozen_before_scoring": False}, "configuration_not_frozen_before_scoring"),
        ({"confidence_intervals_complete": False}, "confidence_intervals_incomplete"),
    ],
)
def test_legacy_failures_remain_visible(change: dict[str, object], reason: str) -> None:
    summary = legacy_green_summary()
    summary.update(change)
    result = assess_promotion(summary)

    assert result["eligible"] is False
    assert result["legacy_criteria_passed"] is False
    assert reason in result["reasons"]
    assert "confirmatory_v2_evidence_not_validated" in result["reasons"]


def test_valid_boundary_values_retain_legacy_assessment_only() -> None:
    summary = legacy_green_summary()
    summary.update({"regret_kga": 0, "false_adapt_rate": 0, "coverage": 1})
    result = assess_promotion(summary)

    assert result["legacy_criteria_passed"] is True
    assert result["eligible"] is False
