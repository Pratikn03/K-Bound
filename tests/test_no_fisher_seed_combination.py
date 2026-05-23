"""Phase 1.D — no Fisher combination of per-seed DeLong p-values may
appear in any active manuscript-emitting code path."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(".")

ACTIVE_FILES = [
    "src/scripts/emit_milestone2_cross_benchmark.py",
    "src/scripts/emit_locked_audited_statistics.py",
]


# Forbidden patterns: anything that combines seed-level p-values into a
# Fisher chi-square. We allow definitions of `_fisher_combine` to live in
# the codebase (history) but they must not be invoked in active emit code.
FORBIDDEN_CALL_PATTERNS = [
    re.compile(r"_fisher_combine\s*\("),
    re.compile(r"fisher_combine\s*\("),
    re.compile(r"chi2\.cdf.*sum\s*\(\s*math\.log"),
]


@pytest.mark.parametrize("path", ACTIVE_FILES)
def test_no_active_fisher_invocation(path):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"file not found: {p}")
    text = p.read_text()
    # The new emit_locked_audited_statistics.py is allowed to *define*
    # nothing Fisher-related. The legacy emit_milestone2 file may still
    # contain a `def _fisher_combine` definition but must NOT call it.
    for pat in FORBIDDEN_CALL_PATTERNS:
        # Allow the function definition line itself (`def _fisher_combine(...)`)
        # but flag any *call*.
        for m in pat.finditer(text):
            start = m.start()
            # If the line begins with 'def ' before the match, skip it.
            line_start = text.rfind("\n", 0, start) + 1
            line = text[line_start:start]
            if line.strip().startswith("def "):
                continue
            raise AssertionError(
                f"{path} contains an active Fisher-combine call at offset {start}: "
                f"{text[max(0, start-30):m.end()+30]!r}. "
                f"Phase 1.D forbids Fisher combination of dependent-seed DeLong p-values."
            )
