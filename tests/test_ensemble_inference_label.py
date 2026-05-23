"""Phase 1.D — any audited inferential statistic computed on seed-averaged
predictions must be labelled as an *ensemble* audited analysis, not a
single-model claim."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

INFERENCE = Path("experiments/audit/audited_ensemble_inference_results.csv")
PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")


@pytest.fixture(scope="module")
def rows():
    if not INFERENCE.exists():
        pytest.skip(f"missing: {INFERENCE}")
    with INFERENCE.open() as f:
        return list(csv.DictReader(f))


def test_inference_rows_carry_explicit_analysis_label(rows):
    for r in rows:
        label = r.get("analysis_label", "")
        if r["analysis_status"] == "audited primary reanalysis":
            assert label and ("audited" in label.lower() or "ensemble" in label.lower() or "representative" in label.lower()), (
                f"cell {r['cell_id']} analysis_label={label!r} must contain audited/ensemble/representative wording."
            )
        elif r["analysis_status"] in {"exploratory", "protocol-diagnostic"}:
            assert label and "descriptive" in label.lower(), (
                f"cell {r['cell_id']} analysis_label={label!r} must contain 'descriptive' wording."
            )


def test_paper_does_not_imply_single_model_from_ensemble():
    """The audited single-representative-seed DeLong is labelled in the inference
    artifact; when the manuscript cites it, the cite must not imply a typical
    single-trained-model significance claim. Phase 1.G enforces this in prose;
    this test catches the most blatant patterns."""
    if not PAPER.exists():
        pytest.skip(f"{PAPER} missing")
    text = PAPER.read_text()
    forbidden = [
        re.compile(r"\bRGA\+\s+beats\s+every\s+baseline\b", re.IGNORECASE),
        re.compile(r"\bRGA\+\s+is\s+the\s+top\s+method\s+across\s+", re.IGNORECASE),
    ]
    for pat in forbidden:
        m = pat.search(text)
        assert m is None, (
            f"paper contains forbidden single-model superiority phrasing: "
            f"{text[max(0,m.start()-40):m.end()+40]!r}"
        )
