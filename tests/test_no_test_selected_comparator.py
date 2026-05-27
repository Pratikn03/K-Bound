"""Phase 1.C — assert no active code or LaTeX caption selects the
primary comparator by reading the test fold."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(".")

# Files that previously selected the comparator by reading the test fold.
# After Phase 1.C, the master comparison emit script must read from
# experiments/audit/audited_comparator_selection.csv.
ACTIVE_SOURCE_FILES = [
    "src/scripts/emit_milestone2_cross_benchmark.py",
]

LATEX_FILES = [
    "docs/research/PAPER_DRAFT_v1.tex",
    "docs/research/THESIS_CHAPTER_v1.tex",
]


# Patterns that indicate post-hoc test-winner comparator selection.
FORBIDDEN_SOURCE_PATTERNS = [
    # Loops that argmax baselines over test-fold ROC-AUC.
    re.compile(
        r"for\s+name\s*,\s*metrics\s+in\s+cs\.items\(\)\s*:\s*\n\s*if\s+name\s+in\s*\{[^}]*'?craf_attention'?[^}]*\}"
    ),
    re.compile(r"best_roc\s*=\s*None\s*\n\s*for\s+name\s*,\s*metrics\s+in\s+cs\.items"),
]


@pytest.mark.parametrize("path", ACTIVE_SOURCE_FILES)
def test_no_test_argmax_baseline_in_active_source(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"file not found: {p}")
    text = p.read_text()
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        m = pat.search(text)
        assert m is None, (
            f"{path} still contains a test-fold argmax over baselines (Phase 1.C forbids this). "
            f"Match: {text[m.start():m.end()][:120]!r}. The emit script must read the comparator "
            f"from experiments/audit/audited_comparator_selection.csv."
        )


@pytest.mark.parametrize("path", LATEX_FILES)
def test_no_best_non_router_inferential_phrase(path):
    """The phrase 'best non-router' was used to denote a test-winner
    comparator. It must not appear as part of an inferential or
    confirmatory claim in the final manuscript. (It may still appear in
    descriptive content if accompanied by audited-reanalysis language.)
    """
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"file not found: {p}")
    text = p.read_text()
    # Forbidden: "best non-router" used as an inferential framing.
    inferential_patterns = [
        re.compile(r"best\s+non-router\s+baseline\s+is\s+\w+", re.IGNORECASE),
        re.compile(r"best-non-router\s+baseline\s+is\s+\w+", re.IGNORECASE),
        re.compile(r"\\textbf\{Best\s+non-router\}", re.IGNORECASE),
    ]
    for pat in inferential_patterns:
        m = pat.search(text)
        assert m is None, (
            f"{path} contains forbidden 'best non-router' inferential framing at offset {m.start()}: "
            f"{text[max(0, m.start()-30):m.end()+30]!r}. "
            f"Phase 1.C requires the comparator to be named explicitly as the validation-frozen "
            f"primary comparator (see HEADLINE_METHOD_POLICY.md §3)."
        )
