import pytest
import os
import json
import numpy as np
import copy

from kbound_edge.evidence import EDGE_EVIDENCE_NAMES
from kbound_edge.integrity import (
    check_heldout_labels_inaccessible,
    check_feature_schema_unchanged,
    check_epsilon_conformal_split,
    check_heldout_excluded_from_calibration,
    check_identical_heldout_stream,
    check_config_hash_in_log,
    check_model_hash_in_log,
)


@pytest.fixture
def mock_cfg(tmp_path):
    # Setup paths pointing to temp files
    heldout_log = tmp_path / "heldout_online.jsonl"
    model_path = tmp_path / "f0.pt"

    cfg = {
        "num_classes": 4,
        "seed": 20260624,
        "window_size": 32,
        "image_size": 224,
        "adapter": {"lr": 0.001, "steps": 1},
        "paths": {
            "heldout_log": str(heldout_log),
            "model": str(model_path),
            "results_dir": str(tmp_path),
        }
    }

    # Save a dummy log containing clean data
    records = [
        {
            "schema_version": "kbound-edge-v1",
            "timestamp": "2026-06-25T00:00:00",
            "window_id": 0,
            "model_version": "v_hash",
            "config_hash": "c_hash",
            "decision": "adapt",
            "bhat": 0.1,
            "eps": 0.1,
            "lower": 0.0,
            "upper": 0.2,
            "reason": "x",
            "latency_ms": 10.0,
            "evidence": {name: 0.1 for name in EDGE_EVIDENCE_NAMES}
        }
    ]
    with open(heldout_log, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Save dummy calibration summary
    summary_path = tmp_path / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "conformal_sessions": ["S05", "S06"],
            "fit_sessions": ["S03", "S04"],
            "model_hash": "v_hash"
        }, f)

    # Save dummy metrics
    metrics_path = tmp_path / "heldout_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "policy_comparison": {
                "always_freeze": {"n_windows": 10},
                "always_adapt": {"n_windows": 10}
            }
        }, f)

    return cfg


def test_clean_audit_passes(mock_cfg):
    passed, _, _, _ = check_heldout_labels_inaccessible(mock_cfg)
    assert passed

    passed, _, _, _ = check_feature_schema_unchanged(mock_cfg)
    assert passed

    passed, _, _, _ = check_epsilon_conformal_split(mock_cfg["paths"]["results_dir"])
    assert passed

    passed, _, _, _ = check_heldout_excluded_from_calibration(mock_cfg["paths"]["results_dir"])
    assert passed

    passed, _, _, _ = check_identical_heldout_stream(mock_cfg["paths"]["results_dir"])
    assert passed


def test_leak_in_log_fails(mock_cfg):
    # Inject a labels key in log row
    log_path = mock_cfg["paths"]["heldout_log"]
    with open(log_path, "a") as f:
        f.write(json.dumps({"window_id": 1, "labels": [1, 2]}) + "\n")

    passed, _, _, _ = check_heldout_labels_inaccessible(mock_cfg)
    assert not passed


def test_feature_schema_change_fails(mock_cfg):
    # Modify feature schema in log row
    log_path = mock_cfg["paths"]["heldout_log"]
    with open(log_path, "w") as f:
        f.write(json.dumps({
            "evidence": {"wrong_name": 0.1}
        }) + "\n")

    passed, _, _, _ = check_feature_schema_unchanged(mock_cfg)
    assert not passed


def test_epsilon_conformal_split_fail(mock_cfg):
    # Modify calibration sessions to overlap
    summary_path = os.path.join(mock_cfg["paths"]["results_dir"], "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "conformal_sessions": ["S05", "S07"], # S07 is heldout!
            "fit_sessions": ["S03", "S04"]
        }, f)

    passed, _, _, _ = check_epsilon_conformal_split(mock_cfg["paths"]["results_dir"])
    assert not passed


def test_heldout_overlap_calibration_fails(mock_cfg):
    summary_path = os.path.join(mock_cfg["paths"]["results_dir"], "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "conformal_sessions": ["S05", "S06"],
            "fit_sessions": ["S03", "S07"] # S07 is heldout!
        }, f)

    passed, _, _, _ = check_heldout_excluded_from_calibration(mock_cfg["paths"]["results_dir"])
    assert not passed


def test_different_replay_streams_fails(mock_cfg):
    metrics_path = os.path.join(mock_cfg["paths"]["results_dir"], "heldout_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "policy_comparison": {
                "always_freeze": {"n_windows": 10},
                "always_adapt": {"n_windows": 9} # Discrepancy!
            }
        }, f)

    passed, _, _, _ = check_identical_heldout_stream(mock_cfg["paths"]["results_dir"])
    assert not passed
