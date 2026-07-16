"""Phase 2 — assert no Phase-2 code path selects methods/comparators from the test fold."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Active Phase-2 code that must not contain test-set selection patterns.
ACTIVE = [
    Path("src/elara/evaluation/prediction_archive.py"),
    Path("src/elara/evaluation/ensemble_inference.py"),
    Path("src/scripts/validate_phase2_prediction_archives.py"),
]

FORBIDDEN_SOURCE = [
    re.compile(r"max\(\s*rga_router_test", re.IGNORECASE),
    re.compile(r"argmax.*test_roc", re.IGNORECASE),
    re.compile(r"\bbest non-router\b", re.IGNORECASE),
]


@pytest.mark.parametrize("path", ACTIVE)
def test_phase2_source_no_test_set_selection(path):
    if not path.exists():
        pytest.skip(f"file not found: {path}")
    text = path.read_text()
    for pat in FORBIDDEN_SOURCE:
        m = pat.search(text)
        assert m is None, f"{path} contains forbidden test-set selection pattern: {m.group(0)!r}"


def test_phase2_contract_forbids_selection_used_test_metrics():
    contract = Path("docs/research/phase2/PHASE_2_STATISTICAL_POLICY.md").read_text()
    assert "validation-only" in contract.lower()
    assert "no test-selected" in contract.lower() or "forbidden" in contract.lower()
