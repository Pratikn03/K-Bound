"""Phase 1.1.1 — regression tests for the three residual caption / policy issues.

Tests are both source-level (LaTeX sources) and PDF-level (extracted text from
verified final PDFs). If either layer regresses, these tests fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")
PAPER_PDF = Path("output/pdf/PAPER_DRAFT_PHASE1_1_1_VERIFIED.pdf")
THESIS_PDF = Path("output/pdf/THESIS_CHAPTER_PHASE1_1_1_VERIFIED.pdf")
PAPER_PDF_STD = Path("output/pdf/PAPER_DRAFT_v1.pdf")
THESIS_PDF_STD = Path("output/pdf/THESIS_CHAPTER_v1.pdf")


def _pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    from pypdf import PdfReader

    r = PdfReader(str(path))
    return "".join(p.extract_text() + "\n" for p in r.pages)


# ---------------------------------------------------------------------------
# Issue 1: Family A is NEVER called confirmatory for existing observed cells.
# ---------------------------------------------------------------------------

# Forbidden: positive noun-phrase uses of "Family A confirmatory ..." (set/family/cells/K=N).
# Allowed: defensive disclaimers like "Family A is never called confirmatory".
ISSUE_1_PATTERNS = [
    re.compile(
        r"Family[\s~]+A\b\s+confirmatory\s+(set|family|cells?|reanalysis|K[\s_=]*\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"confirmatory\s+(set|family|cells?|reanalysis)\s+of\s+Family[\s~]+A",
        re.IGNORECASE,
    ),
    re.compile(
        r"locked\s+Family[\s~]+A\s+confirmatory",
        re.IGNORECASE,
    ),
]


def test_issue1_source_family_a_not_confirmatory():
    for path in (PAPER, THESIS):
        if not path.exists():
            continue
        text = path.read_text()
        for pat in ISSUE_1_PATTERNS:
            m = pat.search(text)
            assert m is None, (
                f"{path} still contains 'Family A confirmatory' at offset {m.start()}: "
                f"{text[max(0, m.start()-30):m.end()+30]!r}"
            )


def test_issue1_pdf_family_a_not_confirmatory():
    paper_text = _pdf_text(PAPER_PDF) or _pdf_text(PAPER_PDF_STD)
    thesis_text = _pdf_text(THESIS_PDF) or _pdf_text(THESIS_PDF_STD)
    for name, text in [("paper PDF", paper_text), ("thesis PDF", thesis_text)]:
        for pat in ISSUE_1_PATTERNS:
            assert pat.search(text) is None, f"{name} still contains 'Family A confirmatory' wording"


# ---------------------------------------------------------------------------
# Issue 2: Canonical MVTec ROC-AUC-only figure caption MUST NOT claim PR-AUC.
# ---------------------------------------------------------------------------


def test_issue2_canonical_mvtec_figure_caption_no_pr_auc():
    """Locate the canonical MVTec clean-benchmark figure caption and verify it
    does NOT claim PR-AUC is shown."""
    t = PAPER.read_text()
    # Find the figure block by its image filename, then read the next \caption{...}
    # block (matched-brace-aware).
    img_match = re.search(
        r"\\includegraphics\[[^]]*\]\{mvtec3d_clean_benchmark\.png\}",
        t,
    )
    if img_match is None:
        pytest.fail("canonical MVTec clean-benchmark figure not found in paper source")
    after_img = t[img_match.end() :]
    cap_start = re.search(r"\\caption\{", after_img)
    assert cap_start is not None, "no \\caption{ after canonical figure"
    # Find matching closing brace.
    pos = cap_start.end()
    depth = 1
    while pos < len(after_img) and depth > 0:
        ch = after_img[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        pos += 1
    caption = after_img[cap_start.end() : pos - 1]
    # The caption MAY mention PR-AUC only to say it is OMITTED. It must NOT claim
    # PR-AUC is displayed alongside ROC-AUC.
    bad_phrases = [
        re.compile(r"ROC-AUC\s+and\s+PR-AUC", re.IGNORECASE),
        re.compile(r"PR-AUC\s+and\s+ROC-AUC", re.IGNORECASE),
        re.compile(r"benchmark\s+ROC-AUC\s+and\s+PR-AUC", re.IGNORECASE),
    ]
    for pat in bad_phrases:
        m = pat.search(caption)
        assert m is None, (
            f"canonical MVTec figure caption still claims PR-AUC is displayed: "
            f"'{m.group(0)}' inside {caption[:200]!r}"
        )
    assert ("Protocol-diagnostic" in caption) or (
        "protocol-diagnostic" in caption.lower()
    ), f"canonical MVTec figure caption missing 'protocol-diagnostic' framing: {caption[:200]!r}"


# ---------------------------------------------------------------------------
# Issue 3: Tables/figure displaying +0.0367 / +0.0538 MUST contain SECONDARY label.
# ---------------------------------------------------------------------------

SECONDARY_PHRASE = re.compile(r"SECONDARY|secondary descriptive", re.IGNORECASE)


def _table_caption(text: str, label: str) -> str | None:
    """Extract the caption of the \\begin{table*?}...\\end{table*?} block whose
    \\label{...} matches `label`."""
    pattern = re.compile(
        r"\\begin\{(table\*?)\}(?P<body>.*?)\\end\{\1\}",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        body = m.group("body")
        if f"\\label{{{label}}}" not in body:
            continue
        cap_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.DOTALL)
        if cap_match:
            return cap_match.group(1)
    return None


def _figure_caption(text: str, label: str) -> str | None:
    pattern = re.compile(
        r"\\begin\{(figure\*?)\}(?P<body>.*?)\\end\{\1\}",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        body = m.group("body")
        if f"\\label{{{label}}}" not in body:
            continue
        cap_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.DOTALL)
        if cap_match:
            return cap_match.group(1)
    return None


def test_issue3_adversarial_results_table_secondary():
    t = PAPER.read_text()
    cap = _table_caption(t, "tab:adversarial-results")
    assert cap is not None, "tab:adversarial-results not found"
    assert SECONDARY_PHRASE.search(cap), f"tab:adversarial-results caption missing SECONDARY label: {cap[:200]!r}"


def test_issue3_tau_sweep_table_secondary():
    t = PAPER.read_text()
    cap = _table_caption(t, "tab:tau-sweep")
    assert cap is not None, "tab:tau-sweep not found"
    assert SECONDARY_PHRASE.search(cap), f"tab:tau-sweep caption missing SECONDARY label: {cap[:200]!r}"


def test_issue3_adversarial_figure_secondary():
    t = PAPER.read_text()
    cap = _figure_caption(t, "fig:adversarial-delta")
    assert cap is not None, "fig:adversarial-delta not found"
    assert SECONDARY_PHRASE.search(cap), f"fig:adversarial-delta caption missing SECONDARY label: {cap[:200]!r}"


def test_issue3_primary_b1_b2_preserved():
    """The abstract and primary mechanism block MUST still cite +0.0506 / +0.0319."""
    t = PAPER.read_text()
    abstract = t[:8000]
    assert "+0.0506" in abstract and "+0.0319" in abstract, "paper abstract missing PRIMARY B1/B2 deltas"


def test_issue3_thesis_primary_b1_b2_preserved():
    t = THESIS.read_text()
    abstract = t[:8000]
    assert "+0.0506" in abstract and "+0.0319" in abstract, "thesis abstract missing PRIMARY B1/B2 deltas"
