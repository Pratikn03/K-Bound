"""Phase 2.1 — the manuscript LaTeX sources must not contain any of the
Phase-2 forbidden claim phrases verbatim."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "research" / "PAPER_DRAFT_v1.tex"
THESIS = ROOT / "docs" / "research" / "THESIS_CHAPTER_v1.tex"


# The Phase-2 forbidden claims, preserved verbatim from the contract.
FORBIDDEN_CLAIMS = [
    "ELARA is universal",
    "RGA+ beats every baseline",
    "Existing Family A cells are confirmatory",
    "Existing Family A cells are preregistered",
    "ELARA is SOTA",
    "ELARA is production-ready",
    "ELARA is deployment-ready",
    "ELARA is validated for clinical deployment",
    "Public benchmark results prove broad cross-domain superiority",
    "Real3D supports generalization",
    "Fixed-seed p-values prove robust method superiority",
]


@pytest.mark.parametrize("path", [PAPER, THESIS])
def test_manuscript_contains_no_forbidden_phase2_claim(path: Path):
    if not path.exists():
        pytest.skip(f"manuscript not present: {path.name}")
    t = path.read_text()
    offenders = [c for c in FORBIDDEN_CLAIMS if c in t]
    assert not offenders, f"{path.name} contains forbidden Phase-2 claim verbatim: {offenders}"
