"""Exercise the existing shell dispatcher without invoking ML or paper writers."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/run_final_showcase.sh"


def preview(*args):
    env = os.environ.copy()
    env.pop("KB_SEEDS", None)
    env.pop("KB_DEVICE", None)
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def test_short_preview_selects_bounded_branch_not_full_branch():
    result = preview("--smoke", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    payloads = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert len(payloads) == 1
    assert payloads[0].endswith("kbtrain.sh' smoke-all")
    assert "KB_SEEDS='0'" in payloads[0]
    assert "KB_DEVICE='cpu'" in payloads[0]


def test_short_preview_cannot_continue_to_historical_analysis_or_paper_writers():
    result = preview("--smoke")
    assert result.returncode == 0, result.stderr
    for payload in (
        "bootstrap_win_cis.py",
        "_locked_analysis_script.py",
        "run_all_headtohead.sh",
        "build_results_source.py",
        "03_make_tables.py",
        "04_make_figures.py",
        "latexmk",
        "mixed_stream_kbound.py",
    ):
        assert payload not in result.stdout


@pytest.mark.parametrize("args", [
    ("--smoke", "--seeds", "4 5"),
    ("--seeds", "4 5", "--smoke"),
])
def test_short_mode_keeps_single_seed_independent_of_flag_order(args):
    result = preview(*args)
    assert result.returncode == 0, result.stderr
    assert "KB_SEEDS='0'" in result.stdout


def test_short_mode_rejects_reusing_historical_full_run():
    result = preview("--smoke", "--skip-train")
    assert result.returncode != 0
    assert not any(line.startswith("+ ") for line in result.stdout.splitlines())


def test_full_preview_retains_existing_protocol_branch_and_explicit_seed_selection():
    result = preview("--device", "cpu", "--seeds", "0 1")
    assert result.returncode == 0, result.stderr
    payloads = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert payloads[0].endswith("kbtrain.sh' final-all-v2")
    assert "KB_SEEDS='0 1'" in payloads[0]
    assert "KB_DEVICE='cpu'" in payloads[0]


@pytest.mark.parametrize("child_exit", [0, 7])
def test_real_short_dispatch_propagates_child_status_without_running_paper_stages(
    tmp_path, child_exit
):
    # The expensive ML child is the boundary double; shell dispatch stays real.
    isolated = tmp_path / "checkout with spaces"
    scripts = isolated / "docs/research/kbound/scripts"
    scripts.mkdir(parents=True)
    local_script = scripts / SCRIPT.name
    shutil.copyfile(SCRIPT, local_script)
    observed = isolated / "child-arguments.txt"
    (scripts / "kbtrain.sh").write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$1" "$KB_SEEDS" "$KB_DEVICE" > "$DISPATCH_RECEIPT"\n'
        'exit "$DISPATCH_CHILD_EXIT"\n'
    )
    sentinel = scripts.parent / "results_source.json"
    sentinel.write_bytes(b"historical evidence must remain unchanged\n")
    env = os.environ.copy()
    env.update(
        DISPATCH_RECEIPT=str(observed), DISPATCH_CHILD_EXIT=str(child_exit)
    )
    result = subprocess.run(
        ["bash", str(local_script), "--smoke", "--device", "cpu"],
        cwd=isolated, env=env, text=True, capture_output=True, timeout=10,
        check=False,
    )
    assert result.returncode == child_exit, result.stderr
    assert observed.read_text().splitlines() == ["smoke-all", "0", "cpu"]
    assert sentinel.read_bytes() == b"historical evidence must remain unchanged\n"
    assert "latexmk" not in result.stdout
    if child_exit:
        assert "SMOKE dispatch complete" not in result.stdout
