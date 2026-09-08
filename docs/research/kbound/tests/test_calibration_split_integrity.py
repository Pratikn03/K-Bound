"""Synthetic regressions for maintained split checks and publication chronology.

These tests never inspect EDGE_RESULTS or open real study labels/artifacts.
Successful synthetic gate inputs are not physical-study or custody evidence.
Physical-study status remains separately unavailable/unverified until an
authorized publication/custody workflow establishes its artifacts and provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from kbound_edge.integrity import (
    check_epsilon_conformal_split,
    check_heldout_excluded_from_calibration,
)
from kbound_edge.publication import evaluate_publication_gate
from kbound_edge.real_manifest import canonical_protocol_hash
from kbound_edge.recording import build_session_checklist

# tests/ -> kbound/ -> research/ -> docs/ -> REPO. Only a path regression;
# no test below reads or writes this physical-study location.
REPO = Path(__file__).resolve().parents[4]
EDGE_RESULTS = REPO / "experiments" / "kbound" / "results" / "edge_real_phone_v1"
CHRONOLOGY_CHECK = "held-out and replication captures occurred after development seal"


def test_edge_results_dir_is_on_the_real_results_root():
    """The results root is REPO/experiments, never REPO/docs/experiments."""
    assert EDGE_RESULTS.name == "edge_real_phone_v1"
    assert EDGE_RESULTS.parents[2] == REPO / "experiments", (
        f"edge results must live under {REPO / 'experiments'}, got {EDGE_RESULTS}"
    )
    assert not str(EDGE_RESULTS).startswith(str(REPO / "docs" / "experiments"))


@pytest.fixture
def publication_inputs():
    # Small in-memory protocol/inventory: every record and hash is invented.
    cfg = {
        "seed": 7,
        "classes": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"],
        "window_size": 32,
        "sessions": {
            f"S{i:02d}": {"objects": [f"synthetic-object-{i}"], "windows": n}
            for i, n in enumerate([20, 20, 16, 44, 16, 44, 32, 56, 32, 56], start=1)
        },
    }
    clips = []
    for sid in cfg["sessions"]:
        for row in build_session_checklist(cfg, sid):
            clips.append(
                {
                    **row,
                    "session_id": sid,
                    "capture_mode": "physical",  # Synthetic field, not a capture claim.
                    "sha256": f"{len(clips) + 1:064x}",
                    "captured_at": ("2026-07-03T09:00:00+00:00" if sid < "S07" else "2026-07-05T09:00:00+00:00"),
                }
            )
    return cfg, {
        "model_card": {
            "protocol_hash": canonical_protocol_hash(cfg),
            "training_command": "synthetic-training",
            "metrics": {"val_balanced_acc": 0.85, "val_macro_f1": 0.84},
        },
        "split_audit": {
            "sealed_splits": {"calibration_conformal": True},
            "sealed_at": "2026-07-04T09:00:00+00:00",
        },
        "inventory": {"clips": clips},
        "heldout": {"n_windows": 88},
        "replication": {"n_windows": 88},
        "anti_leakage": {"checks": [{"check": f"synthetic-{i}", "passed": True} for i in range(8)]},
    }


@pytest.mark.parametrize(
    ("summary", "expected_split", "expected_exclusion"),
    [
        ({"fit_sessions": ["S03", "S04"], "conformal_sessions": ["S05", "S06"]}, True, True),
        ({"fit_sessions": ["S04", "S03"], "conformal_sessions": ["S06", "S05"]}, True, True),
        ({"fit_sessions": ["S03", "S04"], "conformal_sessions": ["S04", "S06"]}, False, True),
        ({"fit_sessions": ["S01", "S02"], "conformal_sessions": ["S05", "S06"]}, False, True),
        ({"fit_sessions": ["S03", "S04"], "conformal_sessions": []}, False, True),
        ({"fit_sessions": [], "conformal_sessions": ["S05", "S06"]}, False, True),
        ({"fit_sessions": [], "conformal_sessions": []}, False, True),
        ({"fit_sessions": ["S03", "S03", "S04"], "conformal_sessions": ["S05", "S06"]}, False, True),
        ({}, False, True),
        ({"fit_sessions": ["S03", "S07"], "conformal_sessions": ["S05", "S06"]}, False, False),
        ({"fit_sessions": ["S03", "S04"], "conformal_sessions": ["S05", "S10"]}, False, False),
    ],
    ids=[
        "declared-disjoint",
        "order-independent",
        "overlap",
        "wrong-but-disjoint",
        "empty-conformal",
        "empty-fit",
        "both-empty",
        "duplicate",
        "missing-fields",
        "heldout-in-fit",
        "replication-in-conformal",
    ],
)
def test_declared_calibration_split_and_exclusion_feed_publication_gate(
    tmp_path, publication_inputs, summary, expected_split, expected_exclusion
):
    path = tmp_path / "calibration_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    split = check_epsilon_conformal_split(str(tmp_path))
    exclusion = check_heldout_excluded_from_calibration(str(tmp_path))
    assert split[0] is expected_split
    assert exclusion[0] is expected_exclusion
    assert split[3] == exclusion[3] == str(path)

    # Exclusion alone accepts empty input (no held-out overlap); the actual
    # declared-split check must still reject it at the publication gate.
    cfg, inputs = publication_inputs
    inputs["anti_leakage"]["checks"][3]["passed"] = split[0]
    inputs["anti_leakage"]["checks"][4]["passed"] = exclusion[0]
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is expected_split
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert failed == (set() if expected_split else {"strict anti-leakage audit passes"})


def test_missing_calibration_summary_fails_both_checks_and_publication(tmp_path, publication_inputs):
    path = str(tmp_path / "calibration_summary.json")
    split = check_epsilon_conformal_split(str(tmp_path))
    exclusion = check_heldout_excluded_from_calibration(str(tmp_path))
    assert split == exclusion == (False, "calibration_summary exists", "missing", path)
    cfg, inputs = publication_inputs
    inputs["anti_leakage"]["checks"][3]["passed"] = split[0]
    inputs["anti_leakage"]["checks"][4]["passed"] = exclusion[0]
    assert evaluate_publication_gate(cfg, **inputs)["passed"] is False


@pytest.mark.parametrize("raw", ["", "\x00" * 4, "{", '{"fit_sessions": [}'])
@pytest.mark.parametrize("check", [check_epsilon_conformal_split, check_heldout_excluded_from_calibration])
def test_malformed_calibration_json_cannot_produce_a_passing_check(tmp_path, raw, check):
    # Current low-level validators propagate parse errors rather than returning
    # a typed gate report. This test does not claim strict schema validation.
    (tmp_path / "calibration_summary.json").write_text(raw, encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        check(str(tmp_path))


@pytest.mark.parametrize("session_id", ["S07", "S08", "S09", "S10"])
@pytest.mark.parametrize(
    ("captured_at", "expected"),
    [
        ("2026-07-04T08:59:59+00:00", False),
        ("2026-07-04T09:00:00+00:00", False),
        ("2026-07-04T09:00:01+00:00", True),
        ("2026-07-04T04:00:00-05:00", False),
        ("2026-07-04T04:00:01-05:00", True),
    ],
    ids=["before", "equal", "after", "equal-offset", "after-offset"],
)
def test_every_heldout_and_replication_capture_must_be_strictly_after_seal(
    publication_inputs, session_id, captured_at, expected
):
    cfg, inputs = publication_inputs
    # One offending clip is sufficient; later clips cannot mask it.
    row = next(row for row in inputs["inventory"]["clips"] if row["session_id"] == session_id)
    row["captured_at"] = captured_at
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is expected
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert failed == (set() if expected else {CHRONOLOGY_CHECK})


@pytest.mark.parametrize(
    "defect",
    [
        "missing-seal",
        "invalid-seal",
        "missing-capture",
        "invalid-capture",
        "naive-capture",
        "naive-seal",
        "empty-inventory",
        "missing-inventory-clips",
    ],
)
def test_missing_or_malformed_chronology_never_passes(publication_inputs, defect):
    cfg, inputs = publication_inputs
    row = next(row for row in inputs["inventory"]["clips"] if row["session_id"] == "S07")
    if defect == "missing-seal":
        del inputs["split_audit"]["sealed_at"]
    elif defect == "invalid-seal":
        inputs["split_audit"]["sealed_at"] = "not-a-timestamp"
    elif defect == "missing-capture":
        del row["captured_at"]
    elif defect == "invalid-capture":
        row["captured_at"] = "not-a-timestamp"
    elif defect == "naive-capture":
        row["captured_at"] = "2026-07-05T09:00:00"
    elif defect == "naive-seal":
        inputs["split_audit"]["sealed_at"] = "2026-07-04T09:00:00"
    elif defect == "empty-inventory":
        inputs["inventory"]["clips"] = []
    else:
        inputs["inventory"] = {}
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    check = next(row for row in report["checks"] if row["check"] == CHRONOLOGY_CHECK)
    assert check["passed"] is False


@pytest.mark.parametrize("sealed_splits", [{}, {"calibration_conformal": False}, {"calibration_conformal": None}])
def test_absent_or_false_seal_cannot_be_replaced_by_later_timestamps(publication_inputs, sealed_splits):
    cfg, inputs = publication_inputs
    inputs["split_audit"]["sealed_splits"] = sealed_splits
    report = evaluate_publication_gate(cfg, **inputs)
    assert report["passed"] is False
    failed = {row["check"] for row in report["checks"] if not row["passed"]}
    assert failed == {"development split sealed"}
