"""Phase 1.D — table captions and emitted artifacts must report a
Holm family size that matches the locked registry value.

The registry is the single source of truth: Family A confirmatory K=5;
Family B audited mechanism endpoints K=2; Family C K=0.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

REGISTRY = Path("experiments/audit/statistical_family_registry.csv")
PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")


@pytest.fixture(scope="module")
def k_family_a():
    if not REGISTRY.exists():
        pytest.skip(f"family registry missing: {REGISTRY}")
    with REGISTRY.open() as f:
        rows = list(csv.DictReader(f))
    a_conf = [r for r in rows if r["analysis_family"] == "A" and r["analysis_status"] == "audited primary reanalysis"]
    return len(a_conf)


def test_family_a_k_is_five(k_family_a):
    assert k_family_a == 5, f"locked Family A confirmatory K=5, got {k_family_a}"


def test_paper_does_not_state_nine_cells():
    if not PAPER.exists():
        pytest.skip(f"{PAPER} missing")
    text = PAPER.read_text()
    # The previous caption claimed "nine evaluated cells" / "9-test Holm". The
    # Phase-1 caption must reflect the locked family size K=5.
    forbidden = [
        re.compile(r"\bacross\s+all\s+nine\s+evaluated\s+cells\b", re.IGNORECASE),
        re.compile(r"\b9-test\s+Holm\b", re.IGNORECASE),
        re.compile(r"\bcorrection\s+across\s+the\s+nine\s+cells\b", re.IGNORECASE),
        re.compile(r"\bHolm\s+correction\s+across\s+all\s+(?:nine|11|eleven)\s+cells\b", re.IGNORECASE),
    ]
    for pat in forbidden:
        m = pat.search(text)
        assert m is None, (
            f"paper still claims a 9- or 11-cell Holm family at offset {m.start()}: "
            f"{text[max(0,m.start()-40):m.end()+40]!r}. Locked Family A K=5."
        )


def test_thesis_does_not_state_nine_cells():
    if not THESIS.exists():
        pytest.skip(f"{THESIS} missing")
    text = THESIS.read_text()
    forbidden = [
        re.compile(r"\bacross\s+all\s+nine\s+evaluated\s+cells\b", re.IGNORECASE),
        re.compile(r"\b9-test\s+Holm\b", re.IGNORECASE),
    ]
    for pat in forbidden:
        m = pat.search(text)
        assert m is None, f"thesis still claims a 9-cell Holm family: {text[max(0,m.start()-40):m.end()+40]!r}"
