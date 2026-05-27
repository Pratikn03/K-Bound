"""Phase 2.1 — registry / claim-matrix CSV schema integrity tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
CLAIMS_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_CLAIM_MATRIX_v2.csv"


@pytest.mark.parametrize("path", [REGISTRY_V2, CLAIMS_V2])
def test_csv_every_row_has_header_field_count(path: Path):
    assert path.exists(), f"{path} missing"
    with path.open() as f:
        rows = list(csv.reader(f))
    assert rows, "file is empty"
    header = rows[0]
    for i, row in enumerate(rows[1:], start=2):
        assert len(row) == len(
            header
        ), f"{path.name}:{i}: row id={row[0]!r} has {len(row)} fields, expected {len(header)}"


def test_registry_v2_has_required_columns():
    with REGISTRY_V2.open() as f:
        header = next(csv.reader(f))
    for required in (
        "experiment_id",
        "analysis_family",
        "primary_comparator",
        "analysis_surface",
        "multiplicity_family",
        "status",
    ):
        assert required in header, f"missing required column {required}"


def test_claim_matrix_v2_has_required_columns():
    with CLAIMS_V2.open() as f:
        header = next(csv.reader(f))
    for required in ("claim_id", "manuscript_location", "status"):
        assert required in header, f"missing required column {required}"


def test_registry_v2_no_unresolved_placeholders_in_family_a_cells():
    """Phase 2.1 §4: locked Family-A cells must not carry TBD placeholders."""
    with REGISTRY_V2.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["analysis_family"] == "A":
            for col in ("benchmark", "protocol", "primary_comparator"):
                assert "TBD" not in r[col].upper(), f"{r['experiment_id']}.{col} has placeholder: {r[col]!r}"
