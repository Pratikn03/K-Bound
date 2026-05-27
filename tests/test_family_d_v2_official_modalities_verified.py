"""Phase 2.2C — official modality + release version documentary verification."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md"
PROVENANCE = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md"
YAML_FILE = ROOT / "configs" / "phase2" / "family_d_v2_eyecandies_protocol.yaml"


def test_decision_doc_locks_release_1_0_3():
    t = DECISION.read_text()
    assert "1.0.3" in t


def test_decision_doc_names_rgb_and_depth_primary():
    t = DECISION.read_text()
    assert "RGB" in t and "depth" in t.lower()


def test_provenance_records_official_paper_and_repo():
    t = PROVENANCE.read_text()
    assert "ACCV 2022" in t
    assert "github.com/eyecan-ai/eyecandies" in t


def test_provenance_records_all_10_drive_file_ids():
    t = PROVENANCE.read_text()
    expected_ids = [
        "1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY",  # candycane
        "1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN",  # chocolatecookie
        "1dRlDAS31QJSwROgA6yFcXo85mL0EBh25",  # chocolatepraline
        "10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq",  # confetto
        "1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W",  # gummybear
        "1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm",  # hazelnuttruffle
        "1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC",  # licoricesandwich
        "1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz",  # lollipop
        "1pebIU3AegEFilqqoROaVzOZqkSgX-JTo",  # marshmallow
        "1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF",  # peppermintcandy
    ]
    for fid in expected_ids:
        assert fid in t, f"missing Drive file ID {fid}"


def test_yaml_lists_all_10_categories():
    c = yaml.safe_load(YAML_FILE.read_text())["protocol"]
    cats = c["dataset"]["categories"]
    assert len(cats) == 10
    assert set(cats) == {
        "candycane",
        "chocolatecookie",
        "chocolatepraline",
        "confetto",
        "gummybear",
        "hazelnuttruffle",
        "licoricesandwich",
        "lollipop",
        "marshmallow",
        "peppermintcandy",
    }
