"""Phase 1.1 — primary ELARA-Bench-LA mechanism story is consistent."""

from __future__ import annotations

import re
from pathlib import Path


PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")


def test_paper_uses_primary_b1_b2():
    t = PAPER.read_text()
    assert re.search(r"\+0\.0506", t), "paper missing PRIMARY zero-attack delta +0.0506"
    assert re.search(r"\+0\.0319", t), "paper missing PRIMARY max-attack delta +0.0319"


def test_thesis_uses_primary_b1_b2():
    t = THESIS.read_text()
    assert re.search(r"\+0\.0506", t)
    assert re.search(r"\+0\.0319", t)
