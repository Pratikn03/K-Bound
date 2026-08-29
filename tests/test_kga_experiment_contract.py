from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kga.experiment_contract import (
    ContractError,
    load_protocol,
    protocol_sha256,
    validate_decision_record,
    validate_joined_records,
    validate_offline_record,
    validate_protocol,
    write_new_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml"
HASH = "a" * 64


def _decision(**updates):
    row = {
        "run_id": "run-1",
        "protocol_id": "KBOUND_PROSPECTIVE_CLOSURE_v1",
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
        "protocol_id": "KBOUND_PROSPECTIVE_CLOSURE_v1",
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


def test_draft_protocol_is_structurally_valid_but_not_sealable():
    document = load_protocol(PROTOCOL)
    assert validate_protocol(document) == []
    errors = validate_protocol(document, require_sealed=True)
    assert any("primary natural dataset is not selected" in error for error in errors)
    assert any("current status is 'DRAFT_UNSEALED'" in error for error in errors)
    assert len(protocol_sha256(document)) == 64


def test_protocol_rejects_split_overlap():
    document = copy.deepcopy(load_protocol(PROTOCOL))
    document["primary_natural_track"]["splits"]["estimator_fit"] = ["same-unit"]
    document["primary_natural_track"]["splits"]["test"] = ["same-unit"]
    errors = validate_protocol(document)
    assert any("overlaps split roles" in error for error in errors)


def test_decision_record_is_label_free_and_action_consistent():
    assert validate_decision_record(_decision()) == []
    assert any("canonical action" in error for error in validate_decision_record(_decision(action="FREEZE")))
    assert any("label-bearing" in error for error in validate_decision_record(_decision(delta=0.2)))


def test_infinite_radius_forces_abstention():
    assert validate_decision_record(_decision(epsilon=float("inf"), action="ABSTAIN")) == []
    assert any(
        "canonical action" in error
        for error in validate_decision_record(_decision(epsilon=float("inf"), action="ADAPT"))
    )


def test_offline_record_requires_label_join_fields():
    assert validate_offline_record(_offline()) == []
    assert any("oracle_action" in error for error in validate_offline_record(_offline(oracle_action="ABSTAIN")))
    assert any("delta must equal" in error for error in validate_offline_record(_offline(delta=0.1)))
    assert any("in [0, 1]" in error for error in validate_offline_record(_offline(macro_f1=1.2)))


def test_join_recomputes_regret_false_adapt_and_chronology():
    assert validate_joined_records([_decision()], [_offline()]) == []
    errors = validate_joined_records([_decision()], [_offline(regret=0.1)])
    assert any("regret must be" in error for error in errors)

    harmful_decision = _decision(delta_hat=0.2, epsilon=0.1, action="ADAPT")
    harmful_offline = _offline(
        delta=-0.2,
        risk_freeze=0.2,
        risk_adapt=0.4,
        oracle_action="FREEZE",
        regret=0.2,
        false_adapt=True,
    )
    assert validate_joined_records([harmful_decision], [harmful_offline]) == []
    assert any(
        "false_adapt must be True" in error
        for error in validate_joined_records(
            [harmful_decision],
            [
                _offline(
                    delta=-0.2,
                    risk_freeze=0.2,
                    risk_adapt=0.4,
                    oracle_action="FREEZE",
                    regret=0.2,
                    false_adapt=False,
                )
            ],
        )
    )


def test_jsonl_writer_is_create_only(tmp_path):
    path = tmp_path / "decisions.jsonl"
    count = write_new_jsonl(path, [_decision()], validator=validate_decision_record)
    assert count == 1
    assert json.loads(path.read_text(encoding="utf-8"))["action"] == "ADAPT"
    with pytest.raises(FileExistsError):
        write_new_jsonl(path, [_decision()], validator=validate_decision_record)


def test_jsonl_writer_refuses_invalid_or_empty_artifacts(tmp_path):
    with pytest.raises(ContractError, match="empty evidence"):
        write_new_jsonl(tmp_path / "empty.jsonl", [], validator=validate_decision_record)
    with pytest.raises(ContractError, match="failed validation"):
        write_new_jsonl(
            tmp_path / "invalid.jsonl",
            [_decision(action="FREEZE")],
            validator=validate_decision_record,
        )
