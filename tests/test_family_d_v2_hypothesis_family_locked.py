"""Phase 2.2C — Family-D v2 hypothesis family invariants."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_HYPOTHESES_v2.csv"
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"


def test_hypotheses_csv_has_two_primary_and_one_secondary():
    with HYP.open() as f:
        rows = list(csv.DictReader(f))
    primary_ids = {r["cell_id"] for r in rows if r["primary_or_secondary"] == "primary"}
    secondary_ids = {r["cell_id"] for r in rows if r["primary_or_secondary"] == "secondary"}
    assert primary_ids == {"D-EYE-1", "D-EYE-2"}
    assert secondary_ids == {"D-EYE-3"}


def test_holm_family_size_is_2():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    assert c["inference"]["multiplicity"]["family"] == "D-EYE-PRIMARY"
    assert c["inference"]["multiplicity"]["family_size_K"] == 2
    assert c["inference"]["multiplicity"]["correction"] == "holm_bonferroni"


def test_primary_endpoints_share_multiplicity_family_and_comparator():
    with HYP.open() as f:
        rows = list(csv.DictReader(f))
    primary = [r for r in rows if r["primary_or_secondary"] == "primary"]
    assert len(primary) == 2
    families = {r["multiplicity_family"] for r in primary}
    comparators = {r["comparator"] for r in primary}
    methods = {r["primary_method"] for r in primary}
    assert families == {"D-EYE-PRIMARY-K2"}
    assert comparators == {"static_attention"}
    assert methods == {"base_RGA"}


def test_secondary_endpoint_is_descriptive_only():
    with HYP.open() as f:
        rows = list(csv.DictReader(f))
    sec = next(r for r in rows if r["cell_id"] == "D-EYE-3")
    assert sec["primary_or_secondary"] == "secondary"
    assert "DESCRIPTIVE" in sec["multiplicity_family"]
