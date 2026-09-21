"""Phase 1.D — any audited inferential statistic computed on seed-averaged
predictions must be labelled as an *ensemble* audited analysis, not a
single-model claim."""

from __future__ import annotations

import csv
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
            assert label and (
                "audited" in label.lower() or "ensemble" in label.lower() or "representative" in label.lower()
            ), f"cell {r['cell_id']} analysis_label={label!r} must contain audited/ensemble/representative wording."
        elif r["analysis_status"] in {"exploratory", "protocol-diagnostic"}:
            assert (
                label and "descriptive" in label.lower()
            ), f"cell {r['cell_id']} analysis_label={label!r} must contain 'descriptive' wording."


def test_retired_elara_paper_is_not_an_active_single_model_claim():
    """ELARA was removed in 4d6bef2; its retired entrypoint must stay absent."""
    assert not PAPER.exists(), "retired ELARA paper unexpectedly restored"
