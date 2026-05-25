"""Phase 2.2C — Family-D v2 must not have been executed during Phase 2.2C."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"
ACCESS_LOG = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md"


def test_protocol_test_evaluation_executed_is_false():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["invariants"]["test_evaluation_executed"] is False
    assert c["provenance"]["test_evaluation_executed"] is False


def test_access_log_states_no_model_evaluation_executed():
    t = ACCESS_LOG.read_text()
    assert "No model evaluation executed" in t or "no model evaluation executed" in t.lower()
    # No performance-metric inspection in this phase
    assert "no performance metric computed" in t.lower() or "No performance metric computed" in t


def test_no_family_d_v2_execution_output_anywhere():
    # If the v3 manifest exists and test_evaluation_executed is True, execution is expected.
    manifest_path = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v3.json"
    if manifest_path.exists():
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("test_evaluation_executed", False):
            return  # skip check: execution has occurred

    forbidden_paths = [
        ROOT / "experiments" / "phase2" / "family_d",
        ROOT / "experiments" / "phase2" / "predictions" / "D-EYE-1",
        ROOT / "experiments" / "phase2" / "predictions" / "D-EYE-2",
        ROOT / "experiments" / "phase2" / "predictions" / "D-EYE-3",
    ]
    for p in forbidden_paths:
        if p.exists():
            # If a family_d directory exists (e.g., for the future hash log)
            # it must NOT contain any inference / metrics output yet.
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                name_low = f.name.lower()
                assert "inference" not in name_low
                assert "holm" not in name_low
                assert "prediction" not in name_low or "archive" in name_low  # archive dir allowed
