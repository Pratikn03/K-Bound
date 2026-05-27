"""Phase 2.1 — registry must match the locked Family-A cell identities."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"


FAMILY_A_LOCKED = {
    "A-POWERED-1": ("MVTec 3D-AD", "PatchCore supervised-paired"),
    "A-POWERED-2": ("MVTec 3D-AD", "PatchCore held-out category"),
    "A-POWERED-3": ("MVTec LOCO-AD", "PatchCore supervised-paired"),
    "A-POWERED-4": ("VisA", "RGB+edge supervised-paired"),
    "A-POWERED-5": ("UNSW-NB15", "flow/conn/context"),
}


def _registry_rows():
    with REGISTRY_V2.open() as f:
        return list(csv.DictReader(f))


def test_family_a_cells_match_locked_identities():
    rows = {r["experiment_id"]: r for r in _registry_rows()}
    for cell_id, (bench, proto) in FAMILY_A_LOCKED.items():
        assert cell_id in rows, f"missing locked Family-A cell {cell_id}"
        r = rows[cell_id]
        assert r["analysis_family"] == "A", f"{cell_id} not in Family A"
        assert r["benchmark"] == bench, f"{cell_id}: benchmark={r['benchmark']!r} (expected {bench!r})"
        assert r["protocol"] == proto, f"{cell_id}: protocol={r['protocol']!r} (expected {proto!r})"


def test_family_a_primary_comparator_is_static_attention_everywhere():
    rows = _registry_rows()
    for r in rows:
        if r["analysis_family"] == "A":
            assert r["primary_comparator"] == "static_attention", (
                f"{r['experiment_id']}: primary_comparator={r['primary_comparator']!r} "
                f"(Phase-2 v2 locks Family-A primary comparator as static_attention)"
            )


def test_real3d_and_efficientad_not_in_family_a():
    """Phase 2.1 §3: EfficientAD / Real3D expansions belong to Family C, not A."""
    rows = _registry_rows()
    for r in rows:
        if r["analysis_family"] == "A":
            bench = r["benchmark"].lower()
            proto = r["protocol"].lower()
            assert "real3d" not in bench, f"{r['experiment_id']}: Real3D must not appear in Family A"
            assert "efficientad" not in proto, f"{r['experiment_id']}: EfficientAD must not appear in Family A"


def test_multiplicity_family_a_powered_k5_has_exactly_five_cells():
    rows = _registry_rows()
    a_powered = [r for r in rows if r["multiplicity_family"] == "A-POWERED-K5"]
    assert len(a_powered) == 5, f"A-POWERED-K5 multiplicity family must contain exactly 5 cells; found {len(a_powered)}"
