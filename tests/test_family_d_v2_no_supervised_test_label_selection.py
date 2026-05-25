"""Phase 2.2C — Family-D v2 method selection must not depend on test labels."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"
HYP = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_HYPOTHESES_v2.csv"
POLICY = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md"


def test_yaml_disables_rga_plus_supervised_head():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["method"]["rga_plus_supervised_head"] == "DISABLED"


def test_yaml_primary_method_is_base_rga():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["method"]["primary"] == "base_RGA"


def test_yaml_comparator_is_static_attention():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["method"]["comparator"] == "static_attention"


def test_hypotheses_every_row_forbids_test_data_access_before_execution():
    with HYP.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["test_data_access_before_execution"].upper() == "FORBIDDEN"
        assert r["test_evaluation_executed"].lower() == "false"


def test_policy_forbids_official_test_label_inputs_to_selection():
    t = POLICY.read_text()
    assert "Official anomalous test labels" in t
    assert "FORBIDDEN" in t.upper() or "forbidden" in t.lower()
