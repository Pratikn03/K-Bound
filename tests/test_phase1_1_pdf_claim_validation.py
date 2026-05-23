"""Phase 1.1 — invoke the deterministic PDF-source claim validator.

Fails if validate_phase1_1_pdf_claims.py returns a non-zero exit code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VALIDATOR = Path("src/scripts/validate_phase1_1_pdf_claims.py")


def test_phase1_1_validator_returns_clean():
    cp = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        cwd=".",
    )
    if cp.returncode != 0:
        msg = cp.stdout + "\n---\n" + cp.stderr
        raise AssertionError("Phase 1.1 PDF claim validation failed:\n" + msg[:6000])
