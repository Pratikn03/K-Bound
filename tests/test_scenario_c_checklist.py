"""Scenario C checklist and Gate A qualification tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_audit_checklist_progress_runs():
    root = _root()
    proc = subprocess.run(
        [sys.executable, "src/scripts/scenario_c/audit_checklist_progress.py"],
        cwd=root,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads((root / "elara_master_c/audits/checklist_progress.json").read_text())
    assert report["summary"]["total"] > 10
    assert report["summary"]["execution_complete"] is True
    assert report["summary"]["m2_transfer_confirmed"] is False
    assert report["summary"]["scientific_scenario_c_ready"] is False
    assert report["summary"]["bounded_v3_evidence_ready"] is True
    assert report["summary"]["positive_transfer_confirmed"] is False
    items = {row["id"]: row for row in report["items"]}
    assert items["gate_e"]["done"] is False
    assert items["gate_f_scientific"]["done"] is False
    assert items["gate_e_bounded_v3"]["done"] is True
    assert items["gate_f_bounded_v3"]["done"] is True
    assert items["gate_e_positive_transfer"]["done"] is False
    assert items["gate_f_positive_transfer"]["done"] is False


def test_confirmatory_report_keeps_strict_and_bounded_v3_separate():
    root = _root()
    report = json.loads(
        (root / "elara_master_c/audits/confirmatory_statistics_report.json").read_text()
    )

    assert report["gate_e_m2_transfer_confirmed_strict"] is False
    assert report["gate_e_m2_transfer_confirmed"] is False
    assert report["gate_f_scenario_c_scientific"] is False
    assert report["gate_e_m2_bounded_v3_pass"] is True
    assert report["gate_f_bounded_v3"] is True
    assert report["gate_e_positive_transfer_confirmed"] is False
    assert report["gate_e_positive_transfer_official"] is False
    assert report["gate_f_positive_transfer_track"] is False


def test_dashboard_snapshot_exposes_strict_and_bounded_readiness():
    root = _root()
    snap = json.loads((root / "research_dashboard/web/data/snapshot.json").read_text())
    c = snap["confirmatory"]
    claim = snap["claim"]

    assert c["gate_e_m2_transfer_confirmed_strict"] is False
    assert c["gate_e_m2_transfer_confirmed"] is False
    assert c["gate_e_m2_bounded_v3_pass"] is True
    assert c["gate_e_positive_transfer_confirmed"] is False
    assert c["gate_e_positive_transfer_official"] is False
    assert c["gate_f_scenario_c_scientific"] is False
    assert c["gate_f_bounded_v3"] is True
    assert c["gate_f_positive_transfer_track"] is False
    assert claim["scientific_ready"] is False
    assert claim["bounded_v3_evidence_ready"] is True
    assert claim["positive_transfer_confirmed"] is False
    assert claim["readiness_tier"] != "tier_3_flagship"


def test_qualify_upstream_experts_runs():
    root = _root()
    csv = root / "experiments/fusion/mvtec3d_patchcore_v2_inputs.csv"
    if not csv.is_file():
        csv = root / "experiments/fusion/mvtec3d_patchcore_inputs.csv"
    if not csv.is_file():
        return
    proc = subprocess.run(
        [sys.executable, "src/scripts/scenario_c/qualify_upstream_experts.py"],
        cwd=root,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1)
    out = root / "elara_master_c/audits/gate_a_expert_qualification_v2.json"
    assert out.is_file()
