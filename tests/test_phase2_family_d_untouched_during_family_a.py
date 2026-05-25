"""Phase 2.2A QC — no Family-D file may be modified or invoked during
Family-A work. v1 stays invalidated; v2 stays design-pending."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"

V1_FAMILY_D_FILES = [
    "FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md",
    "FAMILY_D_DATASET_INVENTORY.md",
    "FAMILY_D_HYPOTHESES.csv",
    "FAMILY_D_PARTITION_MANIFEST.json",
    "FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md",
    "FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md",
]


def test_no_family_d_script_executed_during_phase_2_2a():
    """Family-A drivers must not import or run any Family-D module."""
    for p in (ROOT / "src" / "scripts").glob("run_phase2_family_a_*.py"):
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert "family_d" not in t.lower(), (
            f"{p}: Family-A driver references family_d — Family D must remain untouched"
        )


def test_family_d_v1_invalidation_status_intact():
    p = PHASE2 / "FAMILY_D_V1_INVALIDATION_NOTICE.md"
    assert p.exists(), "FAMILY_D_V1_INVALIDATION_NOTICE.md missing"
    t = p.read_text()
    assert "INVALID_FOR_EXECUTION" in t


def test_family_d_v2_design_status_intact():
    p = PHASE2 / "FAMILY_D_V2_DESIGN_STATUS.md"
    assert p.exists(), "FAMILY_D_V2_DESIGN_STATUS.md missing"
    t = p.read_text()
    assert "V2_DESIGN_PENDING" in t


def test_no_v2_family_d_freeze_partition_manifest_during_family_a():
    """Phase 2.2A must not create the executable partition manifest;
    design-stage v2 files (hypotheses, selection policy, etc.) may
    legitimately exist after Phase 2.2C dataset/protocol decision."""
    forbidden = [
        "FAMILY_D_PARTITION_MANIFEST_v2.json",
    ]
    leaked = [name for name in forbidden if (PHASE2 / name).exists()]
    assert not leaked, f"Phase 2.2A leaked Family-D v2 artifacts: {leaked}"


def test_v1_family_d_files_still_present():
    """v1 files must continue to exist (preserved as historical record)."""
    missing = [name for name in V1_FAMILY_D_FILES if not (PHASE2 / name).exists()]
    assert not missing, f"v1 Family-D files removed (must be preserved): {missing}"
