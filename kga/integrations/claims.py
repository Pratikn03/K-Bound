"""Deterministic promotion rules for integrated KGA research evidence."""

from __future__ import annotations

from collections.abc import Mapping


def _as_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def assess_promotion(summary: Mapping[str, object]) -> dict[str, object]:
    """Return whether an integrated run clears the predeclared evidence bar."""

    mode = str(summary.get("mode", ""))
    alpha = _as_float(summary.get("alpha"), 0.0)
    regret_kga = _as_float(summary.get("regret_kga"), float("inf"))
    regret_adapt = _as_float(summary.get("regret_always_adapt"), float("-inf"))
    regret_freeze = _as_float(summary.get("regret_always_freeze"), float("-inf"))
    integrity_failures = _as_str_list(summary.get("integrity_failures"))
    requirements = (
        (mode in {"label_free", "target_label_light"}, "invalid_evaluation_mode"),
        (
            mode != "label_free" or bool(summary.get("frozen_estimator_verified", False)),
            "unverified_frozen_estimator",
        ),
        (
            _as_int(summary.get("held_out_natural_datasets"), 0) >= 2,
            "fewer_than_two_heldout_natural_datasets",
        ),
        (bool(summary.get("frozen_before_scoring", False)), "configuration_not_frozen_before_scoring"),
        (_as_int(summary.get("independent_splits"), 0) >= 3, "fewer_than_three_independent_splits"),
        (regret_kga < regret_adapt, "does_not_beat_always_adapt_regret"),
        (regret_kga < regret_freeze, "does_not_beat_always_freeze_regret"),
        (_as_float(summary.get("false_adapt_rate"), float("inf")) <= alpha, "false_adapt_exceeds_alpha"),
        (_as_float(summary.get("coverage"), 0.0) >= 0.20, "coverage_below_20_percent"),
        (bool(summary.get("confidence_intervals_complete", False)), "confidence_intervals_incomplete"),
        (bool(summary.get("strong_baselines_complete", False)), "strong_baselines_incomplete"),
        (bool(summary.get("required_tracks_complete", False)), "required_tracks_incomplete"),
        (not integrity_failures, "integrity_failure"),
    )
    reasons = [reason for passed, reason in requirements if not passed]
    if mode == "label_free":
        claim_kind = "label_free"
    elif mode == "target_label_light":
        claim_kind = "target_label_light"
    else:
        claim_kind = "retrospective_only"
    return {
        "eligible": not reasons,
        "claim_kind": claim_kind,
        "reasons": reasons,
    }
