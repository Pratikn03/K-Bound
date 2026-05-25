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


def test_no_v2_family_d_freeze_artifact_was_created_during_family_b():
    """Family-B work must not produce the executable freeze artifact (the
    partition manifest with archive SHA256). Design-stage v2 files
    (hypotheses CSV, selection policy, etc.) may exist as part of the
    Phase 2.2C dataset/protocol decision; the only freeze-equivalent file
    that must remain absent until the hash-only download pass is the
    partition manifest itself."""
    forbidden = ["FAMILY_D_PARTITION_MANIFEST_v2.json"]
    leaked = [n for n in forbidden if (PHASE2 / n).exists()]
    assert not leaked, (
        f"Family-D v2 partition manifest leaked into earlier phase: {leaked}. "
        "Partition manifest may exist only after the Phase 2.2D hash-only download pass."
    )
