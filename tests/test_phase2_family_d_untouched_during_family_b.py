"""Phase 2.2B — Family-D files must not be modified by any Phase-2.2B
infrastructure or driver work."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"


def test_no_family_b_driver_references_family_d():
    for p in (ROOT / "src" / "scripts").glob("run_phase2_*.py"):
        # The B-CERT-1 driver mentions "Family-D" only in inert
        # documentation context — but no driver should *import* family_d.
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert "from elara.family_d" not in t
        assert "import family_d" not in t


def test_family_b_module_does_not_reference_family_d():
    for p in (ROOT / "src" / "elara" / "family_b").glob("*.py"):
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert "family_d" not in t.lower(), (
            f"{p} references family_d — Family D must remain untouched"
        )


def test_family_d_v1_invalidation_status_intact():
    p = PHASE2 / "FAMILY_D_V1_INVALIDATION_NOTICE.md"
    assert p.exists()
    assert "INVALID_FOR_EXECUTION" in p.read_text()


def test_family_d_v2_design_status_intact():
    p = PHASE2 / "FAMILY_D_V2_DESIGN_STATUS.md"
    assert p.exists()
    assert "V2_DESIGN_PENDING" in p.read_text()


def test_family_d_v2_partition_manifest_never_carries_test_execution_phase_b():
    """If the partition manifest exists (post Phase 2.2D hash-only pass), it
    must always declare test_evaluation_executed=false. Family-B work must
    never flip this flag."""
    p = PHASE2 / "FAMILY_D_PARTITION_MANIFEST_v2.json"
    if not p.exists():
        return  # acceptable pre-Phase-2.2D state
    import json
    j = json.loads(p.read_text())
    assert j.get("test_evaluation_executed") is False
