"""Phase 2.2B.2 — Family-D v2 candidate eligibility invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md"
DESIGN = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DESIGN_STATUS.md"
DECISION = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md"


def test_eligibility_review_excludes_visa():
    t = REVIEW.read_text()
    visa_idx = t.find("VisA")
    assert visa_idx >= 0
    assert "INELIGIBLE_FOR_FAMILY_D" in t[visa_idx : visa_idx + 800]


def test_design_status_or_decision_doc_exists():
    if not (DESIGN.exists() or DECISION.exists()):
        pytest.fail(
            "at least one of FAMILY_D_V2_DESIGN_STATUS.md or FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md must exist"
        )


def test_no_v2_decision_doc_lists_visa_as_candidate():
    if not DECISION.exists():
        return
    t = DECISION.read_text()
    # If VisA is mentioned, it must be in an INELIGIBLE / excluded context
    if "VisA" in t:
        snippet_idx = t.find("VisA")
        ctx = t[max(0, snippet_idx - 80) : snippet_idx + 80].lower()
        assert any(
            s in ctx for s in ("ineligible", "exclude", "not be used", "not a v2", "removed")
        ), f"VisA mentioned in v2 decision without exclusion context: {ctx!r}"
