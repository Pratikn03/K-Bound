"""Phase 2.2B.2 — RGA-v2 seed-count decision: 15 is valid per contract YAML."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "phase2" / "rga_v2_gate_contract.yaml"
INF = ROOT / "experiments" / "phase2" / "mechanism" / "rga_v2_failure_surface_inference.csv"
FS = ROOT / "experiments" / "phase2" / "mechanism" / "rga_v2_failure_surface_metrics.csv"


def test_contract_yaml_locks_minimum_for_inference_at_15():
    c = yaml.safe_load(CONTRACT.read_text())["contract"]
    assert c["seeds"]["minimum_for_inference"] == 15


def test_failure_surface_csv_has_at_least_15_seeds():
    with FS.open() as f:
        seeds = {int(r["seed"]) for r in csv.DictReader(f)}
    assert len(seeds) >= 15


def test_inference_csv_g1_g2_g3_c1_all_fail():
    with INF.open() as f:
        rows = {r["gate_id"]: r for r in csv.DictReader(f)}
    for g in ("G1", "G2", "G3"):
        assert "False" in rows[g]["C1_false_fire_budget"], f"{g} should fail C1 to make the 15-seed decision final"
