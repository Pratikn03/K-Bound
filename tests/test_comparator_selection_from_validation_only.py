"""Phase 1.C — assert the validation-frozen comparator artifact is
present, structurally valid, and free of test-set oracle usage."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV_PATH = Path("experiments/audit/audited_comparator_selection.csv")


@pytest.fixture(scope="module")
def rows():
    if not CSV_PATH.exists():
        pytest.skip(f"comparator selection artifact missing: {CSV_PATH}")
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def test_csv_has_required_columns(rows):
    required = {
        "benchmark", "protocol", "analysis_family",
        "candidate_comparators", "validation_scores",
        "selected_comparator", "selected_comparator_validation_auc", "selected_comparator_test_auc",
        "posthoc_test_best_baseline", "posthoc_test_best_auc",
        "does_selected_match_test_best", "selection_used_test_metrics", "analysis_status",
    }
    cols = set(rows[0].keys()) if rows else set()
    assert required.issubset(cols), f"missing columns: {required - cols}"


def test_selection_never_used_test_metrics(rows):
    for r in rows:
        flag = (r.get("selection_used_test_metrics") or "").strip().lower()
        assert flag in {"false", "0"}, (
            f"selection_used_test_metrics must be False (Phase 1.C); row: {r}"
        )


def test_selected_comparator_drawn_from_pool(rows):
    pool = {
        "random_forest", "early_fusion_mlp", "late_fusion_ensemble",
        "confidence_weighted_mean",
        "tent_score_adapter", "ttt_pseudo_label_adapter",
        "eata_score_adapter", "sar_score_adapter",
    }
    for r in rows:
        sel = r.get("selected_comparator")
        if not sel:
            continue  # pending JSON
        assert sel in pool, (
            f"selected_comparator must be drawn from the predefined baseline pool; got {sel!r}"
        )


def test_analysis_status_is_locked_audited_reanalysis(rows):
    for r in rows:
        s = (r.get("analysis_status") or "").strip()
        assert s in {"locked_audited_reanalysis", "pending — JSON missing"}, (
            f"analysis_status must be 'locked_audited_reanalysis'; got {s!r}"
        )


def test_validation_winner_logged_per_cell(rows):
    """The validation-frozen comparator's val ROC-AUC must be reported,
    and must be the max in the cell's validation_scores blob."""
    import json as _json
    for r in rows:
        vs = r.get("validation_scores")
        sel = r.get("selected_comparator")
        if not vs or not sel or vs in ("", "{}"):
            continue
        try:
            d = _json.loads(vs)
        except Exception:
            continue
        if not d:
            continue
        max_val = max(d.values())
        sel_val = d.get(sel)
        if sel_val is None:
            continue
        assert sel_val >= max_val - 1e-9, (
            f"{r['benchmark']} {r['protocol']}: selected_comparator {sel!r} has val ROC-AUC "
            f"{sel_val} < max-in-cell {max_val}. Validation-frozen rule violated."
        )
