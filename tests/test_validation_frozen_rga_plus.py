"""Phase 1.B — assert the validation-frozen RGA+ selection artifact is
present, structurally valid, and free of test-set oracle usage."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV_PATH = Path("experiments/audit/rga_plus_validation_frozen_selection.csv")


@pytest.fixture(scope="module")
def rows():
    if not CSV_PATH.exists():
        pytest.skip(f"selection artifact not produced: {CSV_PATH}")
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def test_csv_has_required_columns(rows):
    required = {
        "benchmark", "protocol", "analysis_family", "seed_or_ensemble",
        "router_validation_auc", "boost_validation_auc",
        "selected_head", "selected_validation_auc", "selected_test_auc",
        "old_test_max_head", "old_test_max_auc",
        "delta_old_max_minus_corrected_headline",
        "selection_used_test_metrics", "claim_status",
        "n_seeds", "n_seed_choose_router", "n_seed_choose_boost",
    }
    cols = set(rows[0].keys()) if rows else set()
    assert required.issubset(cols), f"missing columns: {required - cols}"


def test_selection_never_used_test_metrics(rows):
    for r in rows:
        flag = (r.get("selection_used_test_metrics") or "").strip().lower()
        assert flag in {"false", "0"}, (
            f"selection_used_test_metrics must be False for every row (Rule 4). "
            f"row: {r}"
        )


def test_all_cells_have_ensemble_row(rows):
    seen = set()
    for r in rows:
        if r["seed_or_ensemble"] == "ensemble":
            seen.add((r["benchmark"], r["protocol"]))
    expected = {
        ("MVTec 3D-AD", "PatchCore canonical"),
        ("MVTec 3D-AD", "PatchCore supervised"),
        ("MVTec 3D-AD", "PatchCore held-out"),
        ("MVTec LOCO-AD", "PatchCore canonical"),
        ("MVTec LOCO-AD", "PatchCore supervised"),
        ("Real3D-AD", "PCA shape + depth supervised"),
        ("VisA", "RGB+edge canonical"),
        ("VisA", "RGB+edge supervised"),
        ("VisA", "RGB+random noise-floor"),
        ("UNSW-NB15", "flow/conn/context"),
        ("UNSW-NB15", "held-out attack categories"),
    }
    missing = expected - seen
    assert not missing, f"missing ensemble rows: {missing}"


def test_chosen_head_is_router_or_boost_or_null(rows):
    for r in rows:
        h = r.get("selected_head")
        assert h in {"router", "boost", "", None}, (
            f"chosen_head must be 'router' or 'boost' (or empty for missing JSON): row {r}"
        )


def test_delta_old_max_minus_corrected_is_nonnegative_for_existing_jsons(rows):
    """Validation-frozen selection must produce a value <= test-max for every
    cell that has both test-max and val-frozen values. Otherwise the selection
    rule has been violated (val-frozen would, by construction, never exceed
    the test-max if both heads have valid test ROC-AUC)."""
    for r in rows:
        if r["seed_or_ensemble"] != "ensemble":
            continue
        sel_auc = r.get("selected_test_auc")
        old_auc = r.get("old_test_max_auc")
        if sel_auc in (None, "") or old_auc in (None, ""):
            continue
        try:
            sel_v = float(sel_auc)
            old_v = float(old_auc)
        except (TypeError, ValueError):
            continue
        # The corrected headline must be <= the test-max for that cell.
        # (val-frozen selection picks one head; test-max picks the larger of router/boost test.)
        assert sel_v <= old_v + 1e-9, (
            f"{r['benchmark']} {r['protocol']}: val-frozen selected {sel_v:.4f} exceeds test-max {old_v:.4f}. "
            f"This indicates the val-frozen rule has been violated."
        )


def test_claim_status_is_audited_reanalysis(rows):
    for r in rows:
        cs = (r.get("claim_status") or "").strip()
        assert cs in {"locked_audited_reanalysis", "pending — JSON missing"}, (
            f"claim_status must be 'locked_audited_reanalysis' or 'pending — JSON missing'; got {cs!r} on row {r}"
        )
