"""Deterministic promotion rules for integrated KGA research evidence."""

from __future__ import annotations

from collections.abc import Mapping


def assess_promotion(summary: Mapping[str, object]) -> dict[str, object]:
    """Return whether an integrated run clears the predeclared evidence bar."""

    mode = str(summary.get("mode", ""))
    alpha = float(summary.get("alpha", 0.0))
    regret_kga = float(summary.get("regret_kga", float("inf")))
    regret_adapt = float(summary.get("regret_always_adapt", float("-inf")))
    regret_freeze = float(summary.get("regret_always_freeze", float("-inf")))
    integrity_failures = list(summary.get("integrity_failures", []) or [])
    requirements = (
        (mode in {"label_free", "target_label_light"}, "invalid_evaluation_mode"),
        (
            mode != "label_free" or bool(summary.get("frozen_estimator_verified", False)),
            "unverified_frozen_estimator",
        ),
        (
            int(summary.get("held_out_natural_datasets", 0)) >= 2,
            "fewer_than_two_heldout_natural_datasets",
        ),
        (bool(summary.get("frozen_before_scoring", False)), "configuration_not_frozen_before_scoring"),
        (int(summary.get("independent_splits", 0)) >= 3, "fewer_than_three_independent_splits"),
        (regret_kga < regret_adapt, "does_not_beat_always_adapt_regret"),
        (regret_kga < regret_freeze, "does_not_beat_always_freeze_regret"),
        (float(summary.get("false_adapt_rate", float("inf"))) <= alpha, "false_adapt_exceeds_alpha"),
        (float(summary.get("coverage", 0.0)) >= 0.20, "coverage_below_20_percent"),
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
