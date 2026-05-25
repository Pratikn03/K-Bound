"""Phase 2.2C — paper / thesis must not be edited with Family-D v2 design content."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "research" / "PAPER_DRAFT_v1.tex"
THESIS = ROOT / "docs" / "research" / "THESIS_CHAPTER_v1.tex"

# Phase-2.2C design strings that must not leak into the manuscripts
FORBIDDEN_PHASE_2C_STRINGS = (
    "D-EYE-1",
    "D-EYE-2",
    "D-EYE-3",
    "validation_only_degradation_calibrated_one_class_multimodal",
    "FAMILY_D_V2_FREEZE_BLOCKED",
    "D-EYE-PRIMARY-K2",
    "depth_channel_score_collapse",
)


@pytest.mark.parametrize("path", [PAPER, THESIS])
def test_manuscripts_contain_no_family_d_v2_design_strings(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    t = path.read_text()
    found = [s for s in FORBIDDEN_PHASE_2C_STRINGS if s in t]
    assert not found, f"{path.name} contains Phase-2.2C design strings: {found}"


def test_manuscripts_still_only_cite_eyecandies_in_related_work_context():
    """Eyecandies may appear in PAPER_DRAFT_v1.tex only as a related-work citation
    (e.g. \\cite{bonfiglioli2022eyecandies}) — not as an outcome reference."""
    if not PAPER.exists():
        pytest.skip("paper missing")
    t = PAPER.read_text()
    if "eyecandies" not in t.lower():
        return  # OK
    # Allowed contexts: cite{bonfiglioli2022eyecandies}, bibitem, related-work prose
    # Forbidden contexts: "delta on Eyecandies", "Eyecandies AUC", etc.
    forbidden_patterns = (
        "Eyecandies AUC",
        "delta on Eyecandies",
        "Eyecandies result",
        "Eyecandies validation",
        "Eyecandies confirmation",
    )
    for fp in forbidden_patterns:
        assert fp not in t, f"paper has outcome-level Eyecandies reference: {fp!r}"
