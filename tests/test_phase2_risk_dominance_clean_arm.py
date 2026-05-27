"""Phase 2.2B.2 — risk-dominance terms now admissible (clean arm present)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RD_V2 = ROOT / "experiments" / "phase2" / "certification" / "risk_dominance_terms_v2.csv"
CERT_V2 = ROOT / "experiments" / "phase2" / "certification" / "switching_certificates_v2.csv"
ARCHIVE = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"


def _cell_dir():
    for d in ARCHIVE.iterdir():
        if d.is_dir() and not d.name.startswith("._"):
            return d
    return None


def test_clean_arm_methods_present_in_archive():
    cell = _cell_dir()
    if cell is None:
        pytest.skip("B-MECH-1 archive not present")
    methods = {d.name for d in cell.iterdir() if d.is_dir() and not d.name.startswith("._")}
    if not any("clean_k0" in m for m in methods):
        pytest.skip("clean k=0 arm not yet produced (run run_phase2_b_mech_1_clean_arm.py)")
    assert "static_attention__clean_k0" in methods
    assert "rga_mean_gate_tau66__clean_k0" in methods


def test_risk_dominance_v2_csv_present_and_has_q0_q1_pi_star():
    if not RD_V2.exists():
        pytest.skip("risk_dominance_terms_v2.csv not yet produced")
    with RD_V2.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "v2 risk-dominance CSV empty"
    for r in rows:
        assert "q0_clean_fire_rate" in r
        assert "q1_degraded_fire_rate" in r
        assert "pi_star_indifference_prevalence" in r
        assert "boundary_notice" in r
        assert "retrospective" in r["boundary_notice"].lower()


def test_switching_certificates_v2_present_with_boundary_notice():
    if not CERT_V2.exists():
        pytest.skip("switching_certificates_v2.csv not yet produced")
    with CERT_V2.open() as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        assert "boundary_notice" in r
        assert "retrospective" in r["boundary_notice"].lower()
