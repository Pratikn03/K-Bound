"""Phase 2.2C — Family-D v2 must use one-class multimodal protocol; no supervised label use."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"


def test_yaml_protocol_is_one_class_multimodal():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["name"] == "validation_only_degradation_calibrated_one_class_multimodal"


def test_yaml_splits_are_anomaly_free_train_val_and_held_out_test():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    splits = c["splits"]
    assert "anomaly_free" in splits["train"]
    assert "anomaly_free" in splits["validation"]
    assert "NOT_USED_BEFORE_EXECUTION" in splits["test"]


def test_yaml_primary_modalities_rgb_depth_only():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    pm = c["dataset"]["primary_modalities"]
    assert set(pm) == {"rgb", "depth"}


def test_yaml_normal_excluded_from_primary():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert "normal" in c["dataset"]["documented_but_excluded_from_primary"]
    assert "normal" not in c["dataset"]["primary_modalities"]


def test_yaml_test_evaluation_executed_false():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["invariants"]["test_evaluation_executed"] is False
    assert c["provenance"]["test_evaluation_executed"] is False
