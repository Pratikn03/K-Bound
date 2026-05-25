"""Phase 1.B — source-code + caption check: no active code may select
the headline RGA+ from test-fold metrics, and no LaTeX caption may
state `RGA+ = max(router, boost)`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(".")

# Files that historically computed `max(rga_router_test, rga_boost_test)`.
# After Phase 1.B these patterns must not appear in any *active* code path.
ACTIVE_SOURCE_FILES = [
    "src/scripts/emit_milestone2_cross_benchmark.py",
]

LATEX_FILES = [
    "docs/research/PAPER_DRAFT_v1.tex",
    "docs/research/THESIS_CHAPTER_v1.tex",
]


FORBIDDEN_SOURCE_PATTERNS = [
    # Active test-max selection of the headline RGA+ choice.
    re.compile(r"\brga_plus_candidates\s*=.*max\("),
    re.compile(r"max\(\s*rga_router(_test)?\s*,\s*rga_boost(_test)?\s*\)"),
]

FORBIDDEN_LATEX_PATTERNS = [
    re.compile(r"RGA\+\s*=\s*max\(\s*router\s*,\s*boost\s*\)", re.IGNORECASE),
    re.compile(r"MAX\(\s*router\s*,\s*boost\s*\)", re.IGNORECASE),
]


@pytest.mark.parametrize("path", ACTIVE_SOURCE_FILES)
def test_no_test_max_rga_plus_in_active_source(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"file not found: {p}")
    text = p.read_text()
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        m = pat.search(text)
        assert m is None, (
            f"{path} still contains test-set RGA+ oracle selection at offset {m.start()}: {m.group(0)!r}. "
            f"Phase 1.B requires validation-frozen RGA+ selection (see HEADLINE_METHOD_POLICY.md §1)."
        )


@pytest.mark.parametrize("path", LATEX_FILES)
def test_no_test_max_rga_plus_in_latex_caption(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"file not found: {p}")
    text = p.read_text()
    for pat in FORBIDDEN_LATEX_PATTERNS:
        m = pat.search(text)
        assert m is None, (
            f"{path} still contains 'RGA+ = max(router, boost)' or 'MAX(router, boost)' at offset {m.start()}: "
            f"{text[max(0, m.start()-40):m.end()+40]!r}. "
            f"Phase 1.B requires validation-frozen RGA+ selection."
        )
