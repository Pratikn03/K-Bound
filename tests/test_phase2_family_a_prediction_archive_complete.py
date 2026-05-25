"""Phase 2.2A QC — every completed Family-A cell archive must contain
the required methods, the required schema, and consistent sample-ID
pairing across seeds and across methods."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ARCHIVE = ROOT / "experiments" / "phase2" / "predictions"

REQUIRED_METHODS_FOR_PRIMARY_SURFACE = {
    "rga_meta_router",
    "rga_boosted_fusion",
    "static_attention",
}


def _family_a_cells():
    with REGISTRY_V2.open() as f:
        return [r for r in csv.DictReader(f) if r["analysis_family"] == "A"]


def _cell_dir(row):
    return ARCHIVE / f"{row['experiment_id']}__{row['benchmark'].replace(' ', '_')}__{row['protocol'].replace(' ', '_')}"


@pytest.mark.parametrize("row", _family_a_cells(), ids=lambda r: r["experiment_id"])
def test_completed_cells_have_all_required_methods(row):
    d = _cell_dir(row)
    if not d.exists():
        pytest.skip(f"{row['experiment_id']} archive not present (pending compute)")
    for m in REQUIRED_METHODS_FOR_PRIMARY_SURFACE:
        assert (d / m / "test").exists(), f"{row['experiment_id']}: missing test/{m}"


@pytest.mark.parametrize("row", _family_a_cells(), ids=lambda r: r["experiment_id"])
def test_completed_cells_have_sample_id_alignment_across_methods(row):
    d = _cell_dir(row)
    if not d.exists():
        pytest.skip(f"{row['experiment_id']} archive not present")
    canonical = None
    for m in REQUIRED_METHODS_FOR_PRIMARY_SURFACE:
        seed_files = sorted((d / m / "test").glob("seed_*.parquet"))
        if not seed_files:
            pytest.skip(f"{row['experiment_id']} {m}: no seed files")
        df = pd.read_parquet(seed_files[0])
        ids = df["sample_id"].to_numpy()
        labels = df["label"].to_numpy()
        if canonical is None:
            canonical = (ids, labels)
        else:
            assert np.array_equal(ids, canonical[0]), (
                f"{row['experiment_id']} {m}: sample_id misalignment vs first method"
            )
            assert np.array_equal(labels, canonical[1]), (
                f"{row['experiment_id']} {m}: label misalignment vs first method"
            )


@pytest.mark.parametrize("row", _family_a_cells(), ids=lambda r: r["experiment_id"])
def test_no_test_set_selection_in_any_archive_row(row):
    d = _cell_dir(row)
    if not d.exists():
        pytest.skip(f"{row['experiment_id']} archive not present")
    for m in REQUIRED_METHODS_FOR_PRIMARY_SURFACE:
        for p in sorted((d / m / "test").glob("seed_*.parquet")):
            df = pd.read_parquet(p)
            assert (df["selection_used_test_metrics"] == False).all(), (  # noqa: E712
                f"{row['experiment_id']} {m} {p.name}: row with selection_used_test_metrics=True"
            )
