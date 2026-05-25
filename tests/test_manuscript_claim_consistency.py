"""Phase 1.E — manuscript claim consistency tests.

Invokes the validator script and asserts a clean exit (0 violations).
The validator catches all forbidden tokens listed in
validate_manuscript_claims.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

VALIDATOR = Path("src/scripts/validate_manuscript_claims.py")


def test_validator_script_exists():
    assert VALIDATOR.exists(), f"validator missing: {VALIDATOR}"


def test_manuscript_validator_returns_clean():
    """End-to-end: run the validator and require exit code 0.

    Phase 1.G has to land before this passes. The test is parametrised
    so it fails loudly until the manuscript repair is done.
    """
    import sys
    cp = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        cwd=".",
    )
    if cp.returncode != 0:
        msg = cp.stdout + "\n---\n" + cp.stderr
        pytest.fail(
            "validate_manuscript_claims.py reported violations. "
            "Phase 1.G must land before this test passes.\n" + msg[:4000]
        )
