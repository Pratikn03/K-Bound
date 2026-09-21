"""Retired orchestration must never invoke interpreters, datasets or writers."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "docs/research/kbound/g5_finalize/run_g5_finalize.sh",
        "docs/research/kbound/scripts/run_85plus_readiness.sh",
        "docs/research/kbound/scripts/run_full_panel.sh",
        "docs/research/kbound/scripts/run_final_showcase.sh",
        "docs/research/kbound/scripts/sync_from_v8.sh",
        "docs/research/kbound/runbooks/finish_empirical_training.sh",
        "scripts/rebuild_kbound.sh",
        "scripts/shrink_git_history.sh",
    ],
)
def test_retired_launcher_stops_before_any_external_command(tmp_path: Path, relative: str) -> None:
    # Run an isolated copy with no external commands available. Original scripts
    # must never be executed just to find out whether they start old experiments.
    script = tmp_path / "retired.sh"
    script.write_bytes((ROOT / relative).read_bytes())
    result = subprocess.run(
        ["/bin/bash", str(script)],
        cwd=tmp_path,
        env={**os.environ, "PATH": ""},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "retired" in result.stderr.lower()
    assert "release_candidate.sh all" in result.stderr
    assert "command not found" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["retired.sh"]


def test_legacy_headtohead_cli_stops_before_optional_imports_or_results(tmp_path: Path) -> None:
    script = tmp_path / "retired.py"
    script.write_bytes((ROOT / "docs/research/kbound/scripts/sync_protocol_b_headtohead.py").read_bytes())
    # Missing numpy is deliberate: refusal must precede optional dependencies
    # and every historical result read, not fail accidentally on absent inputs.
    command = "import runpy,sys; sys.modules['numpy']=None; runpy.run_path(sys.argv[1],run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-B", "-c", command, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "retired" in result.stderr.lower()
    assert "release_candidate.sh all" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["retired.py"]


def test_legacy_headtohead_callable_cannot_rewrite_results(monkeypatch) -> None:
    path = ROOT / "docs/research/kbound/scripts/sync_protocol_b_headtohead.py"
    source = compile(path.read_text(), str(path), "exec")
    namespace = {"__name__": "retired_sync_helper", "__file__": str(path)}
    # This double allows import only. No numpy behavior is mocked or asserted.
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    exec(source, namespace)

    def deny_result_access(*args, **kwargs):
        raise AssertionError("retired helper attempted result filesystem access")

    with monkeypatch.context() as guard:
        guard.setattr(Path, "is_file", deny_result_access)
        guard.setattr(Path, "read_text", deny_result_access)
        guard.setattr(Path, "write_text", deny_result_access)
        with pytest.raises(RuntimeError, match="(?i)retired"):
            namespace["run_sync"]()


def test_retired_headtohead_import_does_not_load_optional_runtime(tmp_path: Path) -> None:
    script = tmp_path / "retired.py"
    script.write_bytes((ROOT / "docs/research/kbound/scripts/sync_protocol_b_headtohead.py").read_bytes())
    result = subprocess.run(
        [sys.executable, "-B", "-c", "import runpy,sys; sys.modules['numpy']=None; runpy.run_path(sys.argv[1])", str(script)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["retired.py"]
