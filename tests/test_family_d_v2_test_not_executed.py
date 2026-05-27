"""Phase 2.2B.2 — Family-D v2 must not have been executed."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"


def test_no_family_d_v2_freeze_with_executed_outcomes():
    """If a FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md exists, it must
    declare test_evaluation_executed=false (i.e., no test outcomes yet
    inspected)."""
    freeze = PHASE2 / "FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md"
    if not freeze.exists():
        return  # acceptable — v2 not yet frozen
    t = freeze.read_text().lower()
    assert "test_evaluation_executed" in t
    assert "false" in t.split("test_evaluation_executed", 1)[1][:100].lower()


def test_no_family_d_execution_output_anywhere():
    """No directory or file under docs/research/phase2/ or experiments/phase2/
    may carry an indicator that Family-D was executed."""
    forbidden_markers = (
        "family_d_executed",
        "family_d_test_outcomes",
        "FAMILY_D_RESULTS",
    )
    for marker in forbidden_markers:
        for p in PHASE2.glob("**/*"):
            assert marker.lower() not in p.name.lower(), f"forbidden Family-D execution marker in file: {p}"
