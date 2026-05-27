"""Phase 2.2A QC — the v2 primary-surface CSV must obey the K=5 policy."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "experiments" / "phase2" / "statistics"
RAW = STATS / "family_a_v2_primary_cell_level_raw.csv"
HOLM = STATS / "family_a_v2_primary_cell_level_holm_k5.csv"


@pytest.mark.parametrize("path", [RAW, HOLM])
def test_v2_csv_exists_after_analysis(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not yet produced (Phase 2.2A in progress)")


def test_raw_csv_rows_label_primary_static_reference_audit():
    if not RAW.exists():
        pytest.skip("v2 raw CSV not yet produced")
    with RAW.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "raw CSV is empty"
    for r in rows:
        assert r["analysis_surface"] == "PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT"


def test_holm_csv_uses_k_equals_5_or_marks_partial():
    if not HOLM.exists():
        pytest.skip("v2 holm CSV not yet produced")
    with HOLM.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "holm CSV is empty"
    for r in rows:
        assert r["holm_status"] in {"K5_FULL_FAMILY", "PARTIAL_FAMILY"}, f"holm_status={r['holm_status']!r}"
        # If full family, p must be numeric; if partial, must be the literal placeholder
        if r["holm_status"] == "K5_FULL_FAMILY":
            try:
                float(r["delong_p_holm_k5"])
            except ValueError as err:
                raise AssertionError(f"K5_FULL_FAMILY row has non-numeric holm p: {r['delong_p_holm_k5']!r}") from err
        else:
            assert r["delong_p_holm_k5"] == "pending_full_family"


def test_holm_full_family_has_exactly_five_cells():
    if not HOLM.exists():
        pytest.skip("v2 holm CSV not yet produced")
    with HOLM.open() as f:
        rows = list(csv.DictReader(f))
    full = [r for r in rows if r["holm_status"] == "K5_FULL_FAMILY"]
    if not full:
        return
    assert len(full) == 5, f"K5_FULL_FAMILY rows present but count={len(full)} (expected 5)"
