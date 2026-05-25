"""Phase 1.D / Phase 0.6 AR-11 — no already-observed cell may be
described as "confirmatory" or "pre-registered" in the manuscript.

The words "confirmatory" and "pre-registered" are reserved exclusively
for Family D future locked confirmatory replication (which does not yet
exist).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")


def _pre_registered_forbidden_patterns():
    # These phrases imply pre-registration for already-observed cells. They may
    # still appear inside a Family D / future-work section if the surrounding
    # context narrows the claim to a not-yet-evaluated cell; the test below
    # only catches the egregious top-level versions.
    return [
        re.compile(r"\bpre-registered\s+confirmatory\b", re.IGNORECASE),
        re.compile(r"\bare\s+preregistered\b", re.IGNORECASE),
        re.compile(r"\bthese\s+(?:results|findings|cells)\s+are\s+(?:pre-?registered|confirmatory)\b", re.IGNORECASE),
        re.compile(r"\bconfirmatory\s+evidence\s+that\s+RGA", re.IGNORECASE),
    ]


@pytest.mark.parametrize("path", [PAPER, THESIS])
def test_no_retroactive_preregistration_phrase(path):
    if not path.exists():
        pytest.skip(f"{path} missing")
    text = path.read_text()
    for pat in _pre_registered_forbidden_patterns():
        m = pat.search(text)
        assert m is None, (
            f"{path} contains forbidden retroactive pre-registration phrasing at offset {m.start()}: "
            f"{text[max(0, m.start()-40):m.end()+40]!r}. AR-11: existing cells are audited reanalysis, "
            f"not confirmatory."
        )
