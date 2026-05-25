"""Phase 2.2B — every Family-B driver must reject non-B IDs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


DRIVERS = {
    "B-MECH-1": ROOT / "src" / "scripts" / "run_phase2_mechanism_replication.py",
    "B-MECH-2": ROOT / "src" / "scripts" / "run_phase2_rga_v2_gate_sweep.py",
    "B-MECH-3": ROOT / "src" / "scripts" / "run_phase2_mixture_shift.py",
    "B-MECH-4": ROOT / "src" / "scripts" / "run_phase2_ks_power_sweep.py",
    "B-CERT-1": ROOT / "src" / "scripts" / "run_phase2_certificate_audit.py",
}


def _run(driver: Path, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(driver), *args],
        cwd=str(ROOT),
        capture_output=True, text=True, env=env, timeout=45,
    )


@pytest.mark.parametrize("driver_eid,driver_path", list(DRIVERS.items()))
@pytest.mark.parametrize("bad", ["A-POWERED-1", "C-EXP-EFFICIENTAD-1",
                                  "D-CONTRACT-V2", "B-MECH-99"])
def test_each_driver_rejects_wrong_id(driver_eid, driver_path, bad):
    if bad == driver_eid:
        pytest.skip("would be the right id")
    r = _run(driver_path, "--experiment-id", bad,
             "--dry-run" if "certificate" in driver_path.name else "--seeds", "0")
    assert r.returncode != 0, (
        f"{driver_path.name} accepted wrong id {bad}: "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )


@pytest.mark.parametrize("eid,driver_path", list(DRIVERS.items()))
def test_each_driver_accepts_its_locked_id(eid, driver_path):
    if "certificate" in driver_path.name:
        r = _run(driver_path, "--experiment-id", eid, "--dry-run")
    else:
        r = _run(driver_path, "--experiment-id", eid, "--seeds", "0")
    assert r.returncode == 0, (
        f"{driver_path.name} rejected its locked id {eid}: "
        f"stdout={r.stdout!r} stderr={r.stderr!r}"
    )
