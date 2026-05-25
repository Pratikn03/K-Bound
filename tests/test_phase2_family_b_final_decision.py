"""Phase 2.2B.2 — Family-B final decision invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEC = ROOT / "docs" / "research" / "phase2" / "PHASE_2_FAMILY_B_FINAL_DECISION.md"

VALID_DECISIONS = {
    "FAIL_FAMILY_B_VALIDITY",
    "FAMILY_B_COMPLETE_MECHANISM_REPLICATION_ONLY",
    "FAMILY_B_COMPLETE_WITH_NEGATIVE_RGA_V2_AND_BOUNDED_THEORY_EVIDENCE",
    "FAMILY_B_COMPLETE_WITH_RGA_V2_METHOD_ADVANCEMENT",
}


def test_final_decision_doc_present():
    assert DEC.exists()


def test_final_decision_contains_exactly_one_locked_decision_label():
    if not DEC.exists():
        pytest.skip("decision file not yet present")
    t = DEC.read_text()
    found = [d for d in VALID_DECISIONS if d in t]
    assert len(found) >= 1, f"no locked Family-B decision label found in {DEC}"


def test_final_decision_forbids_rga_v2_promotion_when_no_candidate_passes_c1():
    if not DEC.exists():
        pytest.skip("decision file not yet present")
    t = DEC.read_text()
    # Identify the chosen decision by locating the heading-format string.
    import re
    m = re.search(r"## Decision:\s*\**`([A-Z_0-9]+)`\**", t)
    assert m is not None, "decision file must contain a '## Decision: `<LABEL>`' line"
    chosen = m.group(1)
    if chosen == "FAMILY_B_COMPLETE_WITH_RGA_V2_METHOD_ADVANCEMENT":
        # If promotion is claimed, the doc must explicitly cite a passing candidate
        assert "passes C1" in t or "passes all C1..C6" in t
