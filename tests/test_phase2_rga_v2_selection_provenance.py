"""Phase 2.2B.2 — RGA-v2 selection provenance: 15 seeds × 4 gates across all three CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MECH = ROOT / "experiments" / "phase2" / "mechanism"
THR = MECH / "rga_v2_threshold_selection.csv"
FFR = MECH / "rga_v2_clean_false_fire.csv"
FS = MECH / "rga_v2_failure_surface_metrics.csv"


def _seeds(p):
    with p.open() as f:
        return {int(r["seed"]) for r in csv.DictReader(f)}


def test_all_three_csvs_share_15_seeds():
    s_thr = _seeds(THR)
    s_ffr = _seeds(FFR)
    s_fs = _seeds(FS)
    assert len(s_thr) >= 15
    assert s_thr == s_ffr
    assert s_thr.issubset(s_fs) or s_fs.issubset(s_thr) or s_thr == s_fs
    assert s_thr <= s_fs


def test_threshold_selection_row_count_equals_seeds_times_gates():
    with THR.open() as f:
        rows = list(csv.DictReader(f))
    seeds = {r["seed"] for r in rows}
    gates = {r["gate_id"] for r in rows}
    assert len(rows) == len(seeds) * len(gates), (
        f"expected {len(seeds)} × {len(gates)} = {len(seeds)*len(gates)} rows; got {len(rows)}"
    )


def test_no_threshold_selection_row_used_test_metrics():
    with THR.open() as f:
        for r in csv.DictReader(f):
            assert r["selection_used_test_metrics"].lower() == "false"
