from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from kbound_edge.publication import evaluate_publication_gate
from kbound_edge.real_manifest import canonical_protocol_hash, load_real_protocol
from kbound_edge.recording import build_session_checklist

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "configs" / "edge_real_phone_v1.yaml"


def valid_inputs():
    cfg = load_real_protocol(CONFIG)
    protocol_hash = canonical_protocol_hash(cfg)
    clips = []
    index = 0
    for sid in sorted(cfg["sessions"]):
        for row in build_session_checklist(cfg, sid):
            index += 1
            clips.append(
                {
                    **row,
                    "session_id": sid,
                    "capture_mode": "physical",
                    "sha256": f"{index:064x}",
                    "captured_at": ("2026-07-03T09:00:00+00:00" if sid < "S07" else "2026-07-05T09:00:00+00:00"),
                }
            )
    audit = {"checks": [{"check": f"check-{i}", "passed": True} for i in range(8)]}
    return cfg, {
        "model_card": {
            "protocol_hash": protocol_hash,
            "training_command": "03_train_source_model.py --epochs 20",
            "metrics": {"val_balanced_acc": 0.85, "val_macro_f1": 0.84},
        },
        "split_audit": {
            "sealed_splits": {"calibration_conformal": True},
            "sealed_at": "2026-07-04T09:00:00+00:00",
        },
        "inventory": {"clips": clips},
        "heldout": {"n_windows": cfg["sessions"]["S07"]["windows"] + cfg["sessions"]["S08"]["windows"]},
        "replication": {"n_windows": cfg["sessions"]["S09"]["windows"] + cfg["sessions"]["S10"]["windows"]},
        "anti_leakage": audit,
    }


def test_publication_gate_passes_complete_physical_study():
    cfg, inputs = valid_inputs()
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"]


def test_publication_gate_rejects_mock_capture():
    cfg, inputs = valid_inputs()
    inputs["inventory"]["clips"][0]["capture_mode"] = "mock"
    report = evaluate_publication_gate(cfg, **inputs)
    assert not report["passed"]
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert "all inventory clips are physical" in failed


@pytest.mark.parametrize("field", ["seal", "audit"])
@pytest.mark.parametrize("bad", ["false", "true", 1, 0, None, [], {}, [True]])
def test_publication_gate_requires_native_true_claims(field, bad):
    cfg, inputs = valid_inputs()
    if field == "seal":
        inputs["split_audit"]["sealed_splits"]["calibration_conformal"] = bad
    else:
        inputs["anti_leakage"]["checks"][0]["passed"] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("metric", ["val_balanced_acc", "val_macro_f1"])
@pytest.mark.parametrize(
    "bad", [True, False, "0.85", None, [], {}, float("inf"), -float("inf"), float("nan"), -0.1, 1.01, 10**400]
)
def test_publication_metrics_require_finite_native_unit_interval_values(metric, bad):
    cfg, inputs = valid_inputs()
    inputs["model_card"]["metrics"][metric] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("metric", [0.8, 1, 1.0])
def test_publication_metric_threshold_and_upper_boundary_are_accepted(metric):
    cfg, inputs = valid_inputs()
    inputs["model_card"]["metrics"] = {"val_balanced_acc": metric, "val_macro_f1": metric}
    assert evaluate_publication_gate(cfg, **inputs)["passed"] is True


@pytest.mark.parametrize(
    "artifact", ["model_card", "split_audit", "inventory", "heldout", "replication", "anti_leakage"]
)
@pytest.mark.parametrize("bad", [None, True, "invalid", []])
def test_malformed_top_level_artifact_returns_failed_report(artifact, bad):
    cfg, inputs = valid_inputs()
    inputs[artifact] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["metrics", "sealed_splits", "clips", "clip_row", "checks", "check_row"])
@pytest.mark.parametrize("bad", [None, True, "invalid", 42])
def test_malformed_nested_artifact_returns_failed_report(field, bad):
    cfg, inputs = valid_inputs()
    if field == "metrics":
        inputs["model_card"][field] = bad
    elif field == "sealed_splits":
        inputs["split_audit"][field] = bad
    elif field == "clips":
        inputs["inventory"][field] = bad
    elif field == "clip_row":
        inputs["inventory"]["clips"][0] = bad
    elif field == "checks":
        inputs["anti_leakage"][field] = bad
    else:
        inputs["anti_leakage"]["checks"][0] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("stream", ["heldout", "replication"])
@pytest.mark.parametrize("kind", ["float", "string", "bool", "negative", "missing"])
def test_publication_window_counts_require_exact_nonnegative_integers(stream, kind):
    cfg, inputs = valid_inputs()
    count = inputs[stream]["n_windows"]
    values = {"float": float(count), "string": str(count), "bool": True, "negative": -count, "missing": None}
    inputs[stream]["n_windows"] = values[kind]
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [None, True, [], {}, "", "03_train_source_model.py --bypass-gate"])
def test_training_command_must_be_an_explicit_non_bypassed_string(bad):
    cfg, inputs = valid_inputs()
    inputs["model_card"]["training_command"] = bad
    assert evaluate_publication_gate(cfg, **inputs)["passed"] is False


@pytest.mark.parametrize("field", ["sha256", "session_id", "capture_mode", "captured_at"])
@pytest.mark.parametrize("bad", [None, True, [], {}])
def test_clip_primitives_do_not_crash_or_pass(field, bad):
    cfg, inputs = valid_inputs()
    row = next(row for row in inputs["inventory"]["clips"] if row["session_id"] == "S07")
    row[field] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", ["", "not-a-digest", "A" * 64, "1" * 63, "1" * 65])
def test_clip_hashes_require_lowercase_sha256_strings(bad):
    cfg, inputs = valid_inputs()
    inputs["inventory"]["clips"][0]["sha256"] = bad
    assert evaluate_publication_gate(cfg, **inputs)["passed"] is False


def test_naive_seal_and_capture_times_cannot_claim_chronology():
    cfg, inputs = valid_inputs()
    inputs["split_audit"]["sealed_at"] = "2026-07-04T09:00:00"
    for row in inputs["inventory"]["clips"]:
        row["captured_at"] = row["captured_at"].replace("+00:00", "")
    assert evaluate_publication_gate(cfg, **inputs)["passed"] is False


@pytest.mark.parametrize("field", ["cfg", "sessions", "session", "windows", "classes", "objects"])
def test_malformed_config_shape_returns_failed_report(field):
    cfg, inputs = valid_inputs()
    if field == "cfg":
        cfg = []
    elif field == "sessions":
        cfg["sessions"] = []
    elif field == "session":
        cfg["sessions"]["S07"] = True
    elif field == "windows":
        cfg["sessions"]["S07"]["windows"] = float(cfg["sessions"]["S07"]["windows"])
    elif field == "classes":
        cfg["classes"] = None
    else:
        cfg["sessions"]["S07"]["objects"] = None
    # Keep the protocol hash coherent so rejection exercises the malformed
    # configuration, rather than an incidental old model-card hash.
    inputs["model_card"]["protocol_hash"] = canonical_protocol_hash(cfg)
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [True, 64.0, "64", None, -1, 0])
def test_config_window_counts_are_positive_native_integers(bad):
    cfg, inputs = valid_inputs()
    cfg["sessions"]["S07"]["windows"] = bad
    inputs["model_card"]["protocol_hash"] = canonical_protocol_hash(cfg)
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    assert any(
        row["check"] == "protocol configuration has valid shape" and row["passed"] is False for row in report["checks"]
    )


@pytest.mark.parametrize("bad", [True, None, [], {}, ""])
def test_audit_check_labels_require_nonempty_native_strings(bad):
    cfg, inputs = valid_inputs()
    inputs["anti_leakage"]["checks"][0]["check"] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kind", ["append", "replace"])
@pytest.mark.parametrize("bad", [None, True, [], {}, "", "S11", "unknown"])
def test_all_clip_rows_require_native_session_ids_declared_in_active_config(kind, bad):
    cfg, inputs = valid_inputs()
    clips = inputs["inventory"]["clips"]
    if kind == "append":
        row = deepcopy(clips[0])
        row["sha256"] = "f" * 64
        clips.append(row)
    else:
        row = clips[0]
    row["session_id"] = bad
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    json.dumps(report, allow_nan=False)
