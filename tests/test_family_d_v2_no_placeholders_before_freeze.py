"""Phase 2.1 — any Family-D v2 frozen manifest must contain no
placeholder text. Phase 2.1 itself does not freeze v2; this test
guarantees the next freeze attempt cannot land with placeholders."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"

PLACEHOLDERS = (
    "TO_BE_FILLED",
    "TO_BE_RECORDED",
    "TBD",
)

V2_FROZEN_CANDIDATES = [
    PHASE2 / "FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md",
    PHASE2 / "FAMILY_D_PARTITION_MANIFEST_v2.json",
    PHASE2 / "FAMILY_D_HYPOTHESES_v2.csv",
    PHASE2 / "FAMILY_D_SELECTION_AND_STATISTICAL_POLICY_v2.md",
    PHASE2 / "FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md",
]


@pytest.mark.parametrize("path", V2_FROZEN_CANDIDATES)
def test_no_placeholders_in_any_frozen_v2_file(path: Path):
    """If a Family-D v2 frozen file exists, it must contain no placeholder text."""
    if not path.exists():
        # v2 is V2_DESIGN_PENDING; absence is expected and correct in Phase 2.1
        pytest.skip(f"{path.name} not yet frozen — v2 design pending")
    t = path.read_text()
    for ph in PLACEHOLDERS:
        assert ph not in t, (
            f"{path.name} contains forbidden placeholder {ph!r} — v2 freeze invalid"
        )


def test_v2_design_status_file_exists():
    p = PHASE2 / "FAMILY_D_V2_DESIGN_STATUS.md"
    assert p.exists(), "FAMILY_D_V2_DESIGN_STATUS.md missing"
    t = p.read_text()
    assert "V2_DESIGN_PENDING" in t, (
        "design-status file must state V2_DESIGN_PENDING until v2 is fully resolved"
    )
