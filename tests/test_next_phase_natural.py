"""Current-run locks must reject changed inputs and premature target access."""

from __future__ import annotations

import copy

import pytest


def runner():
    from experiments.kbound.next_phase import natural_runner

    return natural_runner


def access():
    return {
        "historical_access_status": "DOCUMENTED_ZERO_RESERVED_ACCESS_NOT_EXTERNALLY_PROVEN",
        "current_run_calibration_pixels_read": 0,
        "current_run_calibration_labels_read": 0,
        "current_run_target_pixels_read": 0,
        "current_run_target_labels_read": 0,
        "unrecognized_prior_access_artifacts": [],
    }


def test_current_run_lock_preserves_historical_uncertainty(tmp_path):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"fixed source checkpoint")
    lock = runner().make_run_lock({"id": "test"}, {"weights": source}, access())
    assert "NOT_EXTERNALLY_PROVEN" in lock["access"]["historical_access_status"]
    runner().verify_run_lock(lock, {"weights": source})


def test_checkpoint_tamper_is_rejected_before_access(tmp_path):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"original")
    lock = runner().make_run_lock({"id": "test"}, {"weights": source}, access())
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        runner().verify_run_lock(lock, {"weights": source})


def test_protocol_tamper_is_rejected(tmp_path):
    lock = runner().make_run_lock({"id": "test"}, {}, access())
    lock["protocol"]["id"] = "different"
    with pytest.raises(ValueError, match="hash"):
        runner().verify_run_lock(lock, {})


@pytest.mark.parametrize("field", ["current_run_target_pixels_read", "current_run_calibration_labels_read"])
def test_opened_current_run_cannot_be_locked(field):
    audit = access()
    audit[field] = 1
    with pytest.raises(ValueError, match="access"):
        runner().make_run_lock({"id": "test"}, {}, audit)


def test_unrecognized_access_artifact_blocks_lock():
    audit = access()
    audit["unrecognized_prior_access_artifacts"] = ["unknown_calibration.json"]
    with pytest.raises(ValueError, match="access"):
        runner().make_run_lock({"id": "test"}, {}, audit)


def test_failed_screen_never_calls_target_reader():
    calls = []
    with pytest.raises(ValueError, match="NO_TARGET_ACCESS"):
        runner().dispatch_eligible_target({"passed": False}, lambda: calls.append(1))
    assert calls == []


def test_incomplete_screen_never_calls_target_reader():
    calls = []
    with pytest.raises(ValueError, match="NO_TARGET_ACCESS"):
        runner().dispatch_eligible_target({"passed": True, "cell_count": 94}, lambda: calls.append(1))
    assert calls == []


def test_nineteen_cities_do_not_establish_ten_percent_conditional_risk():
    result = runner().zero_error_upper_bound(19, 0.05)
    assert result == pytest.approx(1 - 0.05 ** (1 / 19))
    assert result > 0.10


def test_scoring_cannot_change_sealed_decision():
    cell = {
        "city_id": "city",
        "checkpoint_id": "0",
        "action": {"decision": "ABSTAIN", "realized_action": "FREEZE", "action_sha256": "a" * 64},
        "evaluation": {
            "frozen_prediction_class_ids": [0, 0],
            "adapted_prediction_class_ids": [1, 1],
            "selected_prediction_class_ids": [0, 0],
            "sample_count": 2,
        },
    }
    original = copy.deepcopy(cell)
    before = runner().score_cell(cell, [0, 0])
    after = runner().score_cell(cell, [1, 1])
    assert before["observed_benefit"] == -1
    assert after["observed_benefit"] == 1
    assert cell == original
    assert before["action_sha256"] == after["action_sha256"]


def test_scoring_rejects_predictions_inconsistent_with_action():
    cell = {
        "city_id": "city",
        "checkpoint_id": "0",
        "action": {"decision": "ABSTAIN", "realized_action": "FREEZE", "action_sha256": "a" * 64},
        "evaluation": {
            "frozen_prediction_class_ids": [0],
            "adapted_prediction_class_ids": [1],
            "selected_prediction_class_ids": [1],
            "sample_count": 1,
        },
    }
    with pytest.raises(ValueError, match="sealed action"):
        runner().score_cell(cell, [0])


def test_receipt_consumer_returns_the_verified_document_not_a_reopen(tmp_path, monkeypatch):
    """A replacement between separate verification/read calls must not be trusted."""
    import json
    import os

    from experiments.kbound.so2sat.integrity import verify_artifact_receipt, write_immutable_json_with_receipt

    module = runner()
    artifact = tmp_path / "authority.json"
    write_immutable_json_with_receipt(artifact, {"approved": False})
    original_verify = verify_artifact_receipt

    def replace_after_verification(path):
        receipt = original_verify(path)
        os.chmod(artifact, 0o600)
        artifact.write_text(json.dumps({"approved": True}))
        return receipt

    monkeypatch.setattr(module, "verify_artifact_receipt", replace_after_verification, raising=False)
    document, _ = module.load_pair(artifact)
    assert document == {"approved": False}


def test_new_output_cannot_hide_previously_started_reserved_reads(tmp_path):
    previous = tmp_path / "earlier-run"
    previous.mkdir()
    marker = previous / "calibration_started.json"
    marker.write_text('{"target_access": false}')
    fresh = tmp_path / "different-new-run"
    fresh.mkdir()
    assert runner().prior_run_access_artifacts(tmp_path) == [str(marker)]
