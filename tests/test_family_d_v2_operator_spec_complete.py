"""Phase 2.2C — every degradation operator must be fully specified with no placeholders."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md"
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"

PLACEHOLDER_TOKENS = ("TBD", "TO_BE_FILLED", "TO_BE_RECORDED", "placeholder")


def test_operator_spec_has_no_placeholders():
    t = SPEC.read_text()
    for tok in PLACEHOLDER_TOKENS:
        assert tok not in t, f"operator spec contains placeholder {tok!r}"


def test_yaml_has_two_primary_operators_and_one_secondary():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    p = c["degradation_operators"]["primary_endpoints"]
    s = c["degradation_operators"]["secondary_descriptive"]
    ids = {op["id"] for op in p}
    assert ids == {"D-EYE-1", "D-EYE-2"}
    assert len(s) == 1
    assert s[0]["id"] == "D-EYE-3"


def test_every_operator_has_required_fields():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    required = (
        "id",
        "name",
        "target_modality",
        "transformation_level",
        "operator",
        "parameters",
        "seed_policy",
        "validation_use",
        "future_test_use",
        "primary_or_secondary",
    )
    for op in c["degradation_operators"]["primary_endpoints"] + c["degradation_operators"]["secondary_descriptive"]:
        for k in required:
            assert k in op, f"operator {op.get('id')!r} missing field {k!r}"
            # No placeholder string values
            v = op[k]
            if isinstance(v, str):
                for tok in PLACEHOLDER_TOKENS:
                    assert tok not in v
