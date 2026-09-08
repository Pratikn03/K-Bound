"""Diagnostic assessment of legacy KGA summaries, never claim promotion."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _as_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = float(value)
        except (OverflowError, TypeError, ValueError):
            return default
        if math.isfinite(parsed):
            return parsed
    return default


def _as_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _as_str_list(value: object) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def assess_promotion(summary: object) -> dict[str, object]:
    """Assess the historical summary checklist without authorizing a claim.

    ``legacy_criteria_passed`` preserves that checklist's diagnostic result;
    ``claim_kind`` describes the reported evaluation mode, not eligibility.
    These summary scalars and booleans cannot validate confirmatory-v2 outcome
    ordering, immutable identities, symmetric directional errors, branch
    exposure, or cluster-correct inference. Consequently ``eligible`` is always
    false, including when the legacy checklist passes. This function is not a
    confirmatory-v2 evaluator.
    """

    if not isinstance(summary, Mapping):
        raise ValueError("legacy summary must be a mapping")

    mode = str(summary.get("mode", ""))
    alpha = _as_float(summary.get("alpha"), float("nan"))
    regret_kga = _as_float(summary.get("regret_kga"), float("nan"))
    regret_adapt = _as_float(summary.get("regret_always_adapt"), float("nan"))
    regret_freeze = _as_float(summary.get("regret_always_freeze"), float("nan"))
    false_adapt_rate = _as_float(summary.get("false_adapt_rate"), float("nan"))
    # Retain the legacy field name without treating it as interval coverage.
    coverage = _as_float(summary.get("coverage"), float("nan"))
    integrity_failures = _as_str_list(summary.get("integrity_failures"))
    requirements = (
        (mode in {"label_free", "target_label_light"}, "invalid_evaluation_mode"),
        (0.0 < alpha < 1.0, "invalid_alpha"),
        (
            mode != "label_free" or summary.get("frozen_estimator_verified") is True,
            "unverified_frozen_estimator",
        ),
        (
            _as_int(summary.get("held_out_natural_datasets"), 0) >= 2,
            "fewer_than_two_heldout_natural_datasets",
        ),
        (summary.get("frozen_before_scoring") is True, "configuration_not_frozen_before_scoring"),
        (_as_int(summary.get("independent_splits"), 0) >= 3, "fewer_than_three_independent_splits"),
        (math.isfinite(regret_kga) and regret_kga >= 0.0, "invalid_regret_kga"),
        (math.isfinite(regret_adapt) and regret_adapt >= 0.0, "invalid_regret_always_adapt"),
        (math.isfinite(regret_freeze) and regret_freeze >= 0.0, "invalid_regret_always_freeze"),
        (regret_kga < regret_adapt, "does_not_beat_always_adapt_regret"),
        (regret_kga < regret_freeze, "does_not_beat_always_freeze_regret"),
        (0.0 <= false_adapt_rate <= 1.0, "invalid_false_adapt_rate"),
        (false_adapt_rate <= alpha, "false_adapt_exceeds_alpha"),
        (0.0 <= coverage <= 1.0, "invalid_coverage"),
        (coverage >= 0.20, "coverage_below_20_percent"),
        (summary.get("confidence_intervals_complete") is True, "confidence_intervals_incomplete"),
        (summary.get("strong_baselines_complete") is True, "strong_baselines_incomplete"),
        (summary.get("required_tracks_complete") is True, "required_tracks_incomplete"),
        (integrity_failures is not None, "malformed_integrity_failures"),
        (integrity_failures == [], "integrity_failure"),
    )
    reasons = [reason for passed, reason in requirements if not passed]
    if mode == "label_free":
        claim_kind = "label_free"
    elif mode == "target_label_light":
        claim_kind = "target_label_light"
    else:
        claim_kind = "retrospective_only"
    return {
        "eligible": False,
        "legacy_criteria_passed": not reasons,
        "assessment_scope": "legacy_summary_diagnostic_only",
        "claim_kind": claim_kind,
        "reasons": [*reasons, "confirmatory_v2_evidence_not_validated"],
    }
