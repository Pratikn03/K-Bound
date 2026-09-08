"""Real launcher control flow with ML processes replaced at the process boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "docs/research/kbound/scripts/kbtrain.sh"


def make_executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def sandbox(tmp_path):
    repo = tmp_path / "repository with spaces"
    scripts = repo / "docs/research/kbound/scripts"
    scripts.mkdir(parents=True)
    default = tmp_path / "legacy runtime"
    selected = tmp_path / "selected runtime"
    log = tmp_path / "commands.jsonl"
    for venv, name in ((default, "default"), (selected, "selected")):
        make_executable(
            venv / "bin/python",
            f"#!{sys.executable}\n"
            "import json,os,sys\n"
            "with open(os.environ['LAUNCHER_TEST_LOG'],'a') as stream:\n"
            f"    stream.write(json.dumps([{name!r},sys.argv[1:]])+'\\n')\n",
        )
        (venv / "bin/activate").write_text('export PATH="$VENV/bin:$PATH"\n')
    # Transport-only substitution prevents access to the operator's real home
    # environment, without altering selection logic or shell failure behavior.
    script = scripts / "kbtrain.sh"
    script.write_text(LAUNCHER.read_text().replace("$HOME/.venv_wilds", str(default)))
    make_executable(
        scripts / "run_theory_v2_validators.sh",
        '#!/usr/bin/env bash\nexit "${LAUNCHER_TEST_THEORY_EXIT:-0}"\n',
    )
    stubs = tmp_path / "stubs"
    make_executable(stubs / "caffeinate", '#!/usr/bin/env bash\nshift\nexec "$@"\n')
    env = os.environ.copy()
    env.update(
        KBOUND_REPO_ROOT=str(repo),
        KBOUND_EXTERNAL_ROOT=str(tmp_path / "external"),
        KBOUND_VENV=str(selected),
        LAUNCHER_TEST_LOG=str(log),
        PATH=str(stubs) + os.pathsep + os.environ["PATH"],
        KB_SEEDS="0",
    )
    return script, env, log, selected


def execute(script, env, mode):
    return subprocess.run(
        ["bash", str(script), mode], env=env, text=True,
        capture_output=True, timeout=15, check=False,
    )


def entries(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_explicit_runtime_is_used_for_dependency_check_and_child(tmp_path):
    script, env, log, _ = sandbox(tmp_path)
    result = execute(script, env, "multicandidate-panel")
    assert result.returncode == 0, result.stderr
    calls = entries(log)
    assert len(calls) == 2
    assert [row[0] for row in calls] == ["selected", "selected"]
    assert calls[1][1] == [
        "docs/research/kbound/scripts/multicandidate_decide_kga.py", "--selftest"
    ]


def test_missing_explicit_runtime_does_not_silently_use_old_environment(tmp_path):
    script, env, log, selected = sandbox(tmp_path)
    env["KBOUND_VENV"] = str(selected / "missing")
    result = execute(script, env, "multicandidate-panel")
    assert result.returncode != 0
    assert entries(log) == []


def test_failed_first_full_stage_stops_before_later_payloads_or_success_banner(tmp_path):
    script, env, log, _ = sandbox(tmp_path)
    env["LAUNCHER_TEST_THEORY_EXIT"] = "13"
    result = execute(script, env, "final-all-v2")
    assert result.returncode == 13, result.stdout + result.stderr
    assert all(args == ["-c", "import torch, wilds"] for _, args in entries(log))
    assert "FINAL-ALL done" not in result.stdout
