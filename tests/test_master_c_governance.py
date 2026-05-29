"""Tests for Master Scenario C T0 governance validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_t0_governance_validator_passes():
    root = _repo_root()
    script = root / "src/scripts/scenario_c/validate_master_c_governance.py"
    out = root / "elara_master_c/audits/t0_governance_report_test.json"
    proc = subprocess.run(
        [sys.executable, str(script), "--json-out", str(out)],
        cwd=root,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert out.is_file()


def test_dataset_registry_v2_eyecandies_development():
    root = _repo_root()
    text = (root / "research_lock/dataset_registry_v2.yaml").read_text(encoding="utf-8")
    assert "eyecandies:" in text
    idx = text.index("eyecandies:")
    block = text[idx : idx + 400]
    assert "role: development" in block
