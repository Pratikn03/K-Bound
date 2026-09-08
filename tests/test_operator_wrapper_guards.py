"""Only isolated copies, synthetic text/files and recording CLI stubs execute."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
NATURAL = "scripts/run_natural_win_v1.sh"
REMAINING = "scripts/run_remaining_gpu_experiments.sh"
POEM = "docs/research/kbound/runbooks/run_poem_imagenetc.sh"
VERIFY = "docs/research/kbound/runbooks/verify_external_repro.py"


@pytest.mark.parametrize(
    "metrics,expected",
    [({}, 1), ({"unmapped": 4}, 1), ({"cifar_tent_kga_regret": 0.0016}, 0), ({"cifar_tent_kga_regret": 0.1}, 1)],
)
def test_external_reproduction_requires_at_least_one_checked_metric(tmp_path, metrics, expected):
    numbers, outcomes = tmp_path / "macros.tex", tmp_path / "synthetic.json"
    numbers.write_text(r"\newcommand{\CIFARtentKga}{0.0016}" + "\n")
    outcomes.write_text(json.dumps(metrics))
    result = subprocess.run(
        [
            sys.executable,
            "-Werror",
            "-B",
            str(CANDIDATE / VERIFY),
            "--their-results",
            str(outcomes),
            "--numbers-tex",
            str(numbers),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert not result.stderr, result.stderr
    if expected:
        assert "attach this output" not in result.stdout


def executable(path, code):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    path.chmod(0o755)


def setup(tmp_path, relative):
    root = tmp_path / "synthetic repo with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="synthetic"\n')
    bin_dir = tmp_path / "stubs"
    log = tmp_path / "commands.jsonl"
    python = bin_dir / "python"
    executable(
        python,
        f"#!{sys.executable}\n"
        + """import json,os,sys
with open(os.environ['STUB_LOG'],'a') as stream:
    stream.write(json.dumps(sys.argv[1:])+'\\n')
args=sys.argv[1:]
if '-c' in args:
    code=args[args.index('-c')+1]
    if 'import wilds' in code: raise SystemExit(int(os.environ.get('FAIL_WILDS','0')))
    if 'import torch' in code: raise SystemExit(int(os.environ.get('FAIL_TORCH','0')))
if any('run_imagenetr_kbound.py' in x or 'run_camelyon17_kbound.py' in x or x=='main.py' for x in args):
    raise SystemExit(int(os.environ.get('FAIL_PAYLOAD','0')))
""",
    )
    executable(bin_dir / "caffeinate", '#!/bin/bash\nshift\nexec "$@"\n')
    (root / "data" / "camelyon17_v1.0").mkdir(parents=True)
    (root / "imagenetr").mkdir()
    script = root / relative
    script.parent.mkdir(parents=True, exist_ok=True)
    text = (CANDIDATE / relative).read_text()
    if relative == POEM:
        (root / "external" / "poem").mkdir(parents=True)
        (root / "external" / "poem" / "main.py").write_text("# fixture entrypoint; never executed\n")
        (root / "imagenetc" / "gaussian_noise" / "5").mkdir(parents=True)
        executable(bin_dir / "ls", f'#!{sys.executable}\nprint("class\\n" * 1000, end="")\n')
    script.write_text(text)
    environment = dict(
        os.environ,
        PY=str(python),
        CPY=str(python),
        KBOUND_POEM_PYTHON=str(python),
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        STUB_LOG=str(log),
        WILDS_DATA_ROOT=str(root / "data"),
        IMAGENETR_DIR=str(root / "imagenetr"),
        IMAGENETC_ROOT=str(root / "imagenetc"),
        SKIP_CAM="1",
        ONLY="imagenetr",
        SEEDS="0",
        SEVERITIES="5",
        CORRUPTIONS="gaussian_noise",
        TMPDIR=str(tmp_path / "tmp"),
        TORCH_HOME=str(tmp_path / "torch-cache"),
        KBOUND_REPO_ROOT=str(root),
        FAIL_PAYLOAD="0",
        FAIL_WILDS="0",
        FAIL_TORCH="0",
    )
    return root, script, environment, log


def run(script, environment):
    return subprocess.run(["/bin/bash", str(script)], env=environment, text=True, capture_output=True)


def commands(log):
    return [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []


def payloads(log):
    return [
        row
        for row in commands(log)
        if any("run_imagenetr_kbound.py" in x or "run_camelyon17_kbound.py" in x or x == "main.py" for x in row)
    ]


@pytest.mark.parametrize("failure", [0, 13])
def test_poem_child_failure_stops_without_false_completion(tmp_path, failure):
    root, script, env, log = setup(tmp_path, POEM)
    env["FAIL_PAYLOAD"] = str(failure)
    result = run(script, env)
    assert result.returncode == failure, result.stdout + result.stderr
    assert len(payloads(log)) == (1 if failure else 2)
    text = (root / "experiments/kbound/results/official_repro_v1/poem_imagenetc/poem_imagenetc.log").read_text()
    assert ("POEM runs done" in text) is (not failure)


def test_remaining_requested_missing_wilds_is_failure_before_training(tmp_path):
    _, script, env, log = setup(tmp_path, REMAINING)
    env.update(ONLY="camelyon", FAIL_WILDS="23")
    result = run(script, env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not payloads(log)
    assert "ALL REQUESTED GPU EXPERIMENTS COMPLETE" not in result.stdout


def test_remaining_missing_torch_stops_before_training(tmp_path):
    _, script, env, log = setup(tmp_path, REMAINING)
    env["FAIL_TORCH"] = "17"
    result = run(script, env)
    assert result.returncode == 17, result.stdout + result.stderr
    assert not payloads(log)


def test_remaining_child_failure_still_propagates(tmp_path):
    _, script, env, log = setup(tmp_path, REMAINING)
    env["FAIL_PAYLOAD"] = "19"
    result = run(script, env)
    assert result.returncode == 19 and len(payloads(log)) == 1
    assert "ALL REQUESTED GPU EXPERIMENTS COMPLETE" not in result.stdout


def test_remaining_unknown_mode_fails_before_any_python_invocation(tmp_path):
    _, script, env, log = setup(tmp_path, REMAINING)
    env["ONLY"] = "misspelled-mode"
    result = run(script, env)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not commands(log)
    assert "ALL REQUESTED GPU EXPERIMENTS COMPLETE" not in result.stdout


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("imagenetr", "run_imagenetr_kbound.py"),
        ("inr", "run_imagenetr_kbound.py"),
        ("b4", "run_imagenetr_kbound.py"),
        ("B4", "run_imagenetr_kbound.py"),
        ("camelyon", "run_camelyon17_kbound.py"),
        ("cam", "run_camelyon17_kbound.py"),
        ("b1", "run_camelyon17_kbound.py"),
        ("B1", "run_camelyon17_kbound.py"),
        ("both", "both"),
        (None, "both"),
    ],
)
def test_remaining_documented_modes_and_default_keep_exact_stage_selection(tmp_path, mode, expected):
    _, script, env, log = setup(tmp_path, REMAINING)
    if mode is None:
        env.pop("ONLY")
    else:
        env["ONLY"] = mode
    result = run(script, env)
    assert result.returncode == 0, result.stdout + result.stderr
    selected = payloads(log)
    if expected == "both":
        assert len(selected) == 2
        assert "run_imagenetr_kbound.py" in " ".join(selected[0])
        assert "run_camelyon17_kbound.py" in " ".join(selected[1])
    else:
        assert len(selected) == 1 and expected in " ".join(selected[0])


@pytest.mark.parametrize("output", ["missing", "repo_symlink", "valid"])
def test_natural_analysis_requires_nonempty_non_repository_result_dir(tmp_path, output):
    root, script, env, log = setup(tmp_path, NATURAL)
    directory = root / "experiments/kbound/results/natural_win_v1_imagenetr"
    if output == "repo_symlink":
        directory.parent.mkdir(parents=True)
        directory.symlink_to(root, target_is_directory=True)
    elif output == "valid":
        directory.mkdir(parents=True)
    if output != "missing":
        (directory / "per_condition_imagenet-r_fixture_seed0.json").write_text("{}")
    result = run(script, env)
    analyses = [row for row in commands(log) if any("natural_win_analysis.py" in x for x in row)]
    if output == "valid":
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(analyses) == 1  # skipped Camelyon must not analyze '.'
        path = Path(analyses[0][analyses[0].index("--run-dir") + 1])
        assert path.resolve() == directory.resolve() and path.resolve() != root.resolve()
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        assert not analyses


@pytest.mark.parametrize("cam_output", [False, True])
def test_natural_requested_camelyon_also_requires_its_own_output(tmp_path, cam_output):
    root, script, env, log = setup(tmp_path, NATURAL)
    env["SKIP_CAM"] = "0"
    base = root / "experiments/kbound/results"
    inr = base / "natural_win_v1_imagenetr"
    inr.mkdir(parents=True)
    (inr / "per_condition_imagenet-r_fixture_seed0.json").write_text("{}")
    if cam_output:
        cam = base / "natural_win_v1_camelyon"
        cam.mkdir()
        (cam / "per_condition_camelyon17_fixture_seed0.json").write_text("{}")
    result = run(script, env)
    analyses = [row for row in commands(log) if any("natural_win_analysis.py" in x for x in row)]
    if cam_output:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(analyses) == 2
        assert {row[row.index("--dataset") + 1] for row in analyses} == {"camelyon17", "imagenet-r"}
    else:
        assert result.returncode != 0 and not analyses
