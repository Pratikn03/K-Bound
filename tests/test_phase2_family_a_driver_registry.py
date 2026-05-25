"""Phase 2.2A — registry-driven Family-A driver must reject any non-A-POWERED-* ID."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "src" / "scripts" / "run_phase2_family_a_cell.py"


def _run(eid: str):
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, str(DRIVER), "--experiment-id", eid, "--seeds", "0"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, **env},
        timeout=30,
    )


@pytest.mark.parametrize("eid", ["B-MECH-1", "B-CERT-1", "C-EXP-EFFICIENTAD-1", "D-CONTRACT-V2"])
def test_driver_rejects_non_family_a_ids(eid: str):
    r = _run(eid)
    assert r.returncode != 0, f"driver accepted non-Family-A id {eid!r}"


def test_driver_rejects_unknown_experiment_id():
    r = _run("A-POWERED-99")
    assert r.returncode != 0
    assert "not present" in (r.stdout + r.stderr).lower()


@pytest.mark.parametrize("eid", ["A-POWERED-1", "A-POWERED-2", "A-POWERED-3", "A-POWERED-4", "A-POWERED-5"])
def test_driver_accepts_all_locked_family_a_ids(eid: str):
    """With --seeds 0 the driver should pass validation and exit cleanly without training."""
    r = _run(eid)
    # exit code 0 means no validation error; 0 seeds → no training loop body executed
    assert r.returncode == 0, (
        f"driver rejected locked Family-A id {eid!r}: "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
