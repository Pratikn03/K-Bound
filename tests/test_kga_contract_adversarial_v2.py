"""Adversarial boundary tests for the legacy structural KGA contract.

These tests harden malformed-input handling only.  Passing this legacy schema
does not establish confirmatory-v2 protocol validity or scientific binding.
"""

from __future__ import annotations

import pytest

from kga.experiment_contract import (
    validate_decision_record,
    validate_joined_records,
    validate_offline_record,
    validate_protocol,
)

HASH = "a" * 64


def _protocol(**updates):
    document = {
        "schema_version": 1,
        "protocol_id": "legacy-structural-v1",
        "status": "DRAFT_UNSEALED",
        "alpha": 0.1,
        "model_seeds": [0],
        "decision_rule": {
            "adapt": "delta_hat - epsilon > 0",
            "freeze": "delta_hat + epsilon < 0",
            "otherwise": "abstain",
        },
        "inference": {
            "comparison_family": ["kga_vs_always_adapt", "kga_vs_always_freeze"],
            "multiplicity": "holm",
            "confidence_level": 0.95,
            "unit": "environment",
        },
        "primary_natural_track": {
            "dataset": "",
            "provenance_status": "UNSELECTED",
            "splits": {},
        },
        "replication_tracks": [],
        "launcher_compatibility": {},
        "execution": {"train": [], "evaluate": []},
    }
    document.update(updates)
    return document


def _decision(**updates):
    row = {
        "run_id": "run-1",
        "protocol_id": "legacy-structural-v1",
        "protocol_sha256": HASH,
        "git_sha": "deadbeef",
        "dataset_version": "dataset-v1",
        "split_role": "test",
        "unit_id": "unit-1",
        "environment_id": "env-1",
        "model_seed": 0,
        "checkpoint_sha256": HASH,
        "adapter": "tent",
        "adapter_config_sha256": HASH,
        "estimator_config_sha256": HASH,
        "estimator_artifact_sha256": HASH,
        "calibration_pool_sha256": HASH,
        "alpha": 0.1,
        "evidence_schema_version": "z-v1",
        "evidence_sha256": HASH,
        "delta_hat": 0.2,
        "epsilon": 0.1,
        "action": "ADAPT",
        "decision_timestamp_utc": "2026-08-21T00:00:00Z",
    }
    row.update(updates)
    return row


def _offline(**updates):
    row = {
        "run_id": "run-1",
        "protocol_id": "legacy-structural-v1",
        "unit_id": "unit-1",
        "delta": 0.2,
        "risk_freeze": 0.4,
        "risk_adapt": 0.2,
        "oracle_action": "ADAPT",
        "regret": 0.0,
        "false_adapt": False,
        "balanced_accuracy": 0.8,
        "macro_f1": 0.79,
        "evaluation_timestamp_utc": "2026-08-21T01:00:00Z",
    }
    row.update(updates)
    return row


def test_legacy_structural_fixtures_remain_valid():
    assert validate_protocol(_protocol()) == []
    assert validate_decision_record(_decision()) == []
    assert validate_offline_record(_offline()) == []
    assert validate_joined_records([_decision()], [_offline()]) == []


@pytest.mark.parametrize("value", ["+" + HASH, " " + HASH[1:], "١" * 64, b"a" * 64])
def test_sha256_rejects_signed_whitespace_unicode_and_nonstring(value):
    assert validate_decision_record(_decision(protocol_sha256=value))


@pytest.mark.parametrize("value", ["+deadbee", " deadbee", "١" * 7, b"deadbee"])
def test_git_sha_rejects_signed_whitespace_unicode_and_nonstring(value):
    assert validate_decision_record(_decision(git_sha=value))


@pytest.mark.parametrize("field", ["schema_version", "model_seeds"])
def test_protocol_rejects_boolean_integer_fields(field):
    value = True if field == "schema_version" else [True]
    assert validate_protocol(_protocol(**{field: value}))


def test_decision_rejects_boolean_model_seed():
    assert validate_decision_record(_decision(model_seed=True))


@pytest.mark.parametrize("value", [[], {}])
def test_protocol_rejects_unhashable_status_without_crashing(value):
    assert validate_protocol(_protocol(status=value))


@pytest.mark.parametrize(("field", "value"), [("split_role", []), ("action", {})])
def test_decision_rejects_malformed_enum_values_without_crashing(field, value):
    assert validate_decision_record(_decision(**{field: value}))


@pytest.mark.parametrize("value", [[], {}])
def test_offline_rejects_unhashable_oracle_action_without_crashing(value):
    assert validate_offline_record(_offline(oracle_action=value))


@pytest.mark.parametrize("validator", [validate_protocol, validate_decision_record, validate_offline_record])
@pytest.mark.parametrize("value", [None, [], "record"])
def test_validators_reject_nonmapping_containers(validator, value):
    assert validator(value)


def test_sealed_protocol_rejects_nonmapping_splits_without_attribute_error():
    document = _protocol(status="SEALED")
    document["primary_natural_track"]["splits"] = []
    assert validate_protocol(document, require_sealed=True)


@pytest.mark.parametrize("field", ["alpha", "delta_hat", "epsilon"])
def test_decision_rejects_giant_integers_without_overflow(field):
    assert validate_decision_record(_decision(**{field: 10**10000}))


def test_offline_rejects_giant_integer_without_overflow():
    assert validate_offline_record(_offline(delta=10**10000))


def test_protocol_rejects_giant_integer_without_overflow():
    assert validate_protocol(_protocol(alpha=10**10000))


def test_decision_rejects_giant_action_without_diagnostic_failure():
    assert validate_decision_record(_decision(action=10**10000))


def test_sealed_protocol_rejects_giant_status_without_diagnostic_failure():
    assert validate_protocol(_protocol(status=10**10000), require_sealed=True)


def test_protocol_rejects_giant_split_role_without_diagnostic_failure():
    document = _protocol()
    document["primary_natural_track"]["splits"] = {10**10000: []}
    assert validate_protocol(document)


def test_sealed_protocol_rejects_giant_launcher_name_without_diagnostic_failure():
    document = _protocol(status="SEALED", launcher_compatibility={10**10000: "BAD"})
    assert validate_protocol(document, require_sealed=True)


@pytest.mark.parametrize("value", [[], {}])
def test_join_rejects_unhashable_key_parts_without_crashing(value):
    assert validate_joined_records([_decision(unit_id=value)], [_offline(unit_id=value)])


def test_join_rejects_malformed_rows_without_crashing():
    assert validate_joined_records([[]], [{}])


@pytest.mark.parametrize("value", [None, 7])
def test_join_rejects_noniterable_containers_without_crashing(value):
    assert validate_joined_records(value, [_offline()])
    assert validate_joined_records([_decision()], value)


def test_join_rejects_empty_inputs():
    assert validate_joined_records([], [])


def test_epsilon_preserves_positive_infinity_only():
    assert validate_decision_record(_decision(epsilon=float("inf"), action="ABSTAIN")) == []
    assert validate_decision_record(_decision(epsilon=float("-inf"), action="ABSTAIN"))
    assert validate_decision_record(_decision(epsilon=float("nan"), action="ABSTAIN"))
