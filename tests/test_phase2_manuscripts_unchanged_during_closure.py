"""Phase 2.2B.2 — paper / thesis must not be edited during Phase-2 closure."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "research" / "PAPER_DRAFT_v1.tex"
THESIS = ROOT / "docs" / "research" / "THESIS_CHAPTER_v1.tex"


def test_manuscripts_contain_no_phase2_2b_decision_strings():
    """Phase-2.2B decision strings should not have leaked into the manuscripts."""
    forbidden = (
        "PHASE_2_2B",
        "PHASE_2_2A",
        "COMPARABLE_BUT_ESTIMATOR_CHANGED",
        "FAMILY_B_COMPLETE_WITH_NEGATIVE_RGA_V2",
        "EXECUTION_BLOCKED_DRIVER_SCAFFOLD",
    )
    for tex in (PAPER, THESIS):
        if not tex.exists():
            pytest.skip(f"{tex.name} missing")
        t = tex.read_text()
        for s in forbidden:
            assert s not in t, (
                f"{tex.name} contains Phase-2.2B internal label {s!r} — paper/thesis must not "
                "be edited during Phase-2 closure"
            )
