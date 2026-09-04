"""End-to-end guard for the four current K-Bound manuscript roles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KBOUND_ROOT = ROOT / "docs/research/kbound"
AUDIT = KBOUND_ROOT / "scripts/audit_current_kbound_release.py"


def test_current_kbound_release_passes_fail_closed_audit() -> None:
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--pdf-dir", str(KBOUND_ROOT / "release/current")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
