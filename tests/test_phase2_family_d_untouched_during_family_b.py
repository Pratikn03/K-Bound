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


def test_no_v2_family_d_artifact_was_created():
    forbidden = [
        "FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md",
        "FAMILY_D_PARTITION_MANIFEST_v2.json",
        "FAMILY_D_HYPOTHESES_v2.csv",
        "FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md",
        "FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md",
    ]
    leaked = [n for n in forbidden if (PHASE2 / n).exists()]
    assert not leaked, f"Phase 2.2B leaked Family-D v2 artifacts: {leaked}"
