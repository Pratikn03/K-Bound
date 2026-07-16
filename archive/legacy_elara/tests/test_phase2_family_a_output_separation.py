"""Phase 2.2A — v2 outputs must be on separate paths from the
historical A-POWERED-1 K=10 secondary-pilot-audit outputs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "experiments" / "phase2" / "statistics"


HISTORICAL_PATHS = [
    STATS / "family_a_powered_ensemble_inference.csv",
    STATS / "family_a_powered_holm_results.csv",
    STATS / "family_a_powered_seed_metrics.csv",
    STATS / "family_a_selection_log.csv",
]

V2_PATHS = [
    STATS / "family_a_v2_primary_cell_level_raw.csv",
    STATS / "family_a_v2_primary_cell_level_holm_k5.csv",
]


def test_historical_and_v2_paths_are_distinct():
    """No filename may collide."""
    for h in HISTORICAL_PATHS:
        for v in V2_PATHS:
            assert h != v


def test_v2_csv_paths_have_v2_marker_in_filename():
    for v in V2_PATHS:
        assert "_v2_" in v.name, f"{v.name} does not contain '_v2_' marker — separation rule"


def test_historical_pilot_csvs_are_unchanged_in_v2_run():
    """If a historical CSV exists, it must contain the legacy schema
    (column 'comparator_method') — the v2 driver must not have rewritten it."""
    p = STATS / "family_a_powered_ensemble_inference.csv"
    if not p.exists():
        return
    first_line = p.read_text().splitlines()[0]
    assert (
        "comparator_method" in first_line
    ), "historical K=10 secondary-pilot-audit CSV appears to have been overwritten"
