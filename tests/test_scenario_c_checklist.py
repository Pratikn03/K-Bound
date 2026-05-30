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
    assert report["summary"]["percent_complete"] < 100.0


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
