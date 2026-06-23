"""Contract tests for the optional ELARA-U integration in KGA."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from kga import Decision
from kga.integrations import elara as elara_module
from kga.integrations.claims import assess_promotion
from kga.integrations.elara import (
    ELARAKGAGuard,
    EvaluationMode,
    FrozenLinearBenefitEstimator,
)


def synthetic_scores(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_val = np.array([0, 1] * 40)
    y_test = np.array([0, 1] * 50)
    s_val = np.column_stack(
        [
            np.clip(0.15 + 0.70 * y_val + rng.normal(0, 0.04, y_val.size), 0, 1),
            np.clip(0.25 + 0.50 * y_val + rng.normal(0, 0.10, y_val.size), 0, 1),
        ]
    )
    s_test = np.column_stack(
        [
            np.clip(0.15 + 0.70 * y_test + rng.normal(0, 0.04, y_test.size), 0, 1),
            np.clip(0.25 + 0.50 * y_test + rng.normal(0, 0.10, y_test.size), 0, 1),
        ]
    )
    return s_val, y_val, s_test, y_test


def test_label_free_rejects_target_labels() -> None:
    s_val, y_val, s_test, y_test = synthetic_scores()
    guard = ELARAKGAGuard(alpha=0.1)
    with pytest.raises(ValueError, match="must not receive y_test"):
        guard.decide(
            s_val=s_val,
            y_val=y_val,
            s_test=s_test,
            mode=EvaluationMode.LABEL_FREE,
            y_test=y_test,
        )


def test_label_free_fails_closed_without_frozen_estimator() -> None:
    s_val, y_val, s_test, _ = synthetic_scores()
    guard = ELARAKGAGuard(alpha=0.1)
    with pytest.raises(ValueError, match="frozen estimator"):
        guard.decide(s_val=s_val, y_val=y_val, s_test=s_test, mode=EvaluationMode.LABEL_FREE)


def test_retrospective_is_never_claim_eligible() -> None:
    s_val, y_val, s_test, y_test = synthetic_scores()
    result = ELARAKGAGuard(alpha=0.1).decide(
        s_val=s_val,
        y_val=y_val,
        s_test=s_test,
        y_test=y_test,
        mode=EvaluationMode.RETROSPECTIVE_AUDIT,
    )
    assert result.claim_tier == "retrospective_only"
    assert result.claim_eligible is False
    assert result.labels_used_for_decision == len(y_test)


def test_target_label_light_ignores_nonprobe_labels() -> None:
    s_val, y_val, s_test, y_test = synthetic_scores()
    probe = np.arange(20)
    guard = ELARAKGAGuard(alpha=0.1, probe_seed=3)
    first = guard.decide(
        s_val=s_val,
        y_val=y_val,
        s_test=s_test,
        y_test=y_test,
        mode=EvaluationMode.TARGET_LABEL_LIGHT,
        probe_indices=probe,
    )
    permuted = y_test.copy()
    permuted[20:] = permuted[20:][::-1]
    second = guard.decide(
        s_val=s_val,
        y_val=y_val,
        s_test=s_test,
        y_test=permuted,
        mode=EvaluationMode.TARGET_LABEL_LIGHT,
        probe_indices=probe,
    )
    assert first.decision == second.decision
    assert first.certificate == second.certificate
    assert first.labels_used_for_decision == 20


def test_label_free_uses_frozen_estimator_without_labels() -> None:
    s_val, y_val, s_test, _ = synthetic_scores()
    estimator = FrozenLinearBenefitEstimator(
        feature_names=("best_val_auc",),
        weights=np.array([0.0]),
        intercept=0.40,
        residuals=np.full(200, 0.01),
        protocol_hash="calibration-protocol-sha256",
    )
    result = ELARAKGAGuard(alpha=0.1).decide(
        s_val=s_val,
        y_val=y_val,
        s_test=s_test,
        mode=EvaluationMode.LABEL_FREE,
        estimator=estimator,
    )
    assert result.decision == Decision.ADAPT
    assert result.labels_used_for_decision == 0
    assert result.claim_tier == "label_free_candidate"


def test_optional_router_import_falls_back_to_repository_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def reject_installed_namespace(name: str, *args: object, **kwargs: object):
        if name == "uais.elara_u.router":
            raise ModuleNotFoundError("forced missing installed uais package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_installed_namespace)
    RouterPolicy, _, _ = elara_module._load_router_api()
    assert RouterPolicy.__name__ == "RouterPolicy"


def eligible_summary() -> dict[str, object]:
    return {
        "mode": "label_free",
        "frozen_estimator_verified": True,
        "held_out_natural_datasets": 2,
        "frozen_before_scoring": True,
        "independent_splits": 3,
        "regret_kga": 0.01,
        "regret_always_adapt": 0.03,
        "regret_always_freeze": 0.02,
        "false_adapt_rate": 0.05,
        "alpha": 0.10,
        "coverage": 0.30,
        "confidence_intervals_complete": True,
        "strong_baselines_complete": True,
        "required_tracks_complete": True,
        "integrity_failures": [],
    }


def test_promotion_guard_accepts_only_complete_heldout_evidence() -> None:
    verdict = assess_promotion(eligible_summary())
    assert verdict["eligible"] is True
    assert verdict["reasons"] == []


def test_promotion_guard_labels_retrospective_claim_kind() -> None:
    summary = eligible_summary()
    summary["mode"] = "retrospective_audit"
    verdict = assess_promotion(summary)
    assert verdict["eligible"] is False
    assert verdict["claim_kind"] == "retrospective_only"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "retrospective_audit"),
        ("held_out_natural_datasets", 1),
        ("independent_splits", 2),
        ("coverage", 0.19),
        ("false_adapt_rate", 0.11),
        ("required_tracks_complete", False),
    ],
)
def test_promotion_guard_rejects_each_missing_requirement(field: str, value: object) -> None:
    summary = eligible_summary()
    summary[field] = value
    verdict = assess_promotion(summary)
    assert verdict["eligible"] is False
    assert verdict["reasons"]
