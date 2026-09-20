"""The optional full health check must call the maintained release gate."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("release_exit", [0, 23])
def test_full_health_uses_current_release_gate_and_propagates_failure(tmp_path, release_exit):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/monorepo_health.sh", scripts / "monorepo_health.sh")
    python = tmp_path / "test-python"
    python.write_text('#!/bin/bash\nprintf "python:%s\\n" "$*" >> "$CALL_LOG"\n')
    python.chmod(0o755)
    (scripts / "smoke_kbound.sh").write_text('exit 0\n')
    runbook = tmp_path / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(
        'printf "release:%s:%s\\n" "$*" "$KBOUND_PYTHON" >> "$CALL_LOG"\n'
        f'exit {release_exit}\n'
    )
    old = tmp_path / "docs/research/kbound/scripts/reproduce_submission.sh"
    old.parent.mkdir(parents=True)
    old.write_text('printf "retired\\n" >> "$CALL_LOG"\nexit 91\n')
    calls = tmp_path / "calls.log"
    env = dict(os.environ, PY=str(python), CALL_LOG=str(calls))
    completed = subprocess.run(
        ["bash", str(scripts / "monorepo_health.sh"), "--full"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10,
    )
    observed = calls.read_text().splitlines()
    assert f"release:all:{python}" in observed
    assert "retired" not in observed
    assert completed.returncode == release_exit
    assert ("Monorepo health: PASS" in completed.stdout) is (release_exit == 0)
    if release_exit:
        assert not any("audit_gate_p_production.py" in line for line in observed)
