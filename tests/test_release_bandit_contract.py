"""Security scanning excludes metadata, not executable code or credentials."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from docs.research.kbound.scripts.run_repository_verification import BANDIT_COMPATIBILITY_BOOTSTRAP

ROOT = Path(__file__).resolve().parents[1]


def _scan(directory: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            BANDIT_COMPATIBILITY_BOOTSTRAP,
            "-q",
            "-r",
            str(directory),
            "-c",
            str(ROOT / "pyproject.toml"),
            "-f",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode in (0, 1), completed.stderr
    assert "Test in comment:" not in completed.stderr, completed.stderr
    assert completed.stdout, completed.stderr
    return json.loads(completed.stdout)


def test_diagnostic_status_is_not_a_credential(tmp_path: Path) -> None:
    source = (ROOT / "kga/assumptions.py").read_text()
    status = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Status")
    fixture = tmp_path / "runtime"
    fixture.mkdir()
    (fixture / "status.py").write_text("from enum import Enum\n" + ast.get_source_segment(source, status) + "\n")
    result = _scan(fixture)
    assert result["errors"] == []
    assert result["results"] == []


def test_appledouble_sidecars_are_not_python_source(tmp_path: Path) -> None:
    fixture = tmp_path / "runtime"
    fixture.mkdir()
    (fixture / "ordinary.py").write_text("answer = 42\n")
    (fixture / "._ordinary.py").write_bytes(b"\x00\x05\x16\x07Mac metadata\x00")
    result = _scan(fixture)
    assert result["errors"] == []
    assert result["results"] == []
    assert "ordinary.py" in " ".join(result["metrics"])
    assert "._ordinary.py" not in " ".join(result["metrics"])


def test_real_hardcoded_credentials_still_reported(tmp_path: Path) -> None:
    fixture = tmp_path / "runtime"
    fixture.mkdir()
    (fixture / "credential.py").write_text("password = 'synthetic-insecure-credential'\n")
    result = _scan(fixture)
    assert result["errors"] == []
    assert any(issue["test_id"] == "B105" for issue in result["results"])
