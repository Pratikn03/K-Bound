"""Phase 1.1 Step 2 — assert exactly one PRIMARY ELARA-Bench-LA run is locked.

Reads the metrics manifest and confirms B1/B2 entries cite the k-of-D
sweep at k=4 mean-gate (the PRIMARY run per
docs/research/audit/PHASE_1_1_PRIMARY_RUN_RESOLUTION.md).
"""

from __future__ import annotations

import re
from pathlib import Path


PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")


def test_paper_uses_one_b1_b2_pair_in_abstract():
    t = PAPER.read_text()
    # Abstract block (first ~5000 chars)
    abstract = t[:5000]
    # The PRIMARY pair must be present.
    assert "+0.0506" in abstract and "+0.0319" in abstract, (
        "paper abstract missing PRIMARY B1/B2 pair (+0.0506 / +0.0319)"
    )
    # The SECONDARY (table_3 default-gate) pair must NOT appear in the abstract.
    assert "+0.0367" not in abstract and "+0.0538" not in abstract, (
        "paper abstract uses SECONDARY hard-mode deltas; only PRIMARY allowed"
    )


def test_thesis_uses_one_b1_b2_pair_in_abstract():
    t = THESIS.read_text()
    abstract = t[:5000]
    assert "+0.0506" in abstract and "+0.0319" in abstract, (
        "thesis abstract missing PRIMARY B1/B2 pair"
    )
    assert "+0.0367" not in abstract and "+0.0538" not in abstract, (
        "thesis abstract uses SECONDARY deltas"
    )
