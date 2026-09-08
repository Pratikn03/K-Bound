"""Exercise shell interfaces using only synthetic directories and recording CLIs."""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAR = "docs/research/kbound/runbooks/rebuild_cifar10c_sar.sh"
POEM = "docs/research/kbound/runbooks/run_poem_imagenetc.sh"
CLOSURE = "docs/research/kbound/runbooks/run_submission_closure.sh"
VERIFY = "docs/research/kbound/runbooks/verify_external_repro.py"
NATURAL = "scripts/run_natural_win_v1.sh"
REMAINING = "scripts/run_remaining_gpu_experiments.sh"


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def setup(tmp_path, relative):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    runtime = tmp_path / "runtime with spaces" / "python"
    log = tmp_path / "commands.jsonl"
    executable(
        runtime,
        f"#!{sys.executable}\n"
        + """import json, os, sys
args = sys.argv[1:]
code = sys.stdin.read() if args and args[0] == '-' else ''
with open(os.environ['STUB_LOG'], 'a') as stream:
    stream.write(json.dumps({'runtime': sys.argv[0], 'args': args, 'stdin': code}) + '\\n')
if 'glob.glob' in code:
    print('fixturemethod')
if '-c' in args and 'import wilds' in args[args.index('-c') + 1]:
    raise SystemExit(int(os.environ.get('FAIL_WILDS', '0')))
if any('run_imagenetr_kbound.py' in a or 'run_camelyon17_kbound.py' in a or a == 'main.py' for a in args):
    raise SystemExit(int(os.environ.get('FAIL_PAYLOAD', '0')))
""",
    )
    bin_dir = tmp_path / "bin"
    executable(bin_dir / "git", '#!/bin/bash\nprintf "%s\\n" "$STUB_ROOT"\n')
    executable(bin_dir / "caffeinate", '#!/bin/bash\nshift\nexec "$@"\n')
    # A missing configuration on old sources must never select a real interpreter.
    executable(bin_dir / "python3", runtime.read_text())
    for name in ("cifar_tent_mps_v2.py", "validate_cifar10c_sar_rebuild.py"):
        p = root / "docs/research/kbound/scripts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic entrypoint, never executed\n")
    executable(
        bin_dir / "shasum",
        f"#!{sys.executable}\n"
        + """import os, sys
from pathlib import Path
values = {
'cifar_tent_mps_v2.py': 'f1687904d36114340ae7da055197f6bd44c08e2f617d17703a52824765e62dbc',
'resnet18_cifar.pt': '43333456a795bbe679966c14812f9964d8b3bf060d30ca2b3d5051cb8c9d7491',
'labels.npy': 'e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b'}
print(('0' * 64 if os.environ.get('FAIL_HASH') else values[Path(sys.argv[-1]).name]) + '  ' + sys.argv[-1])
""",
    )
    data = root / "data with spaces"
    for rel in ("CIFAR-10-C", "camelyon17_v1.0", "imagenetr", "imagenetc/gaussian_noise/5"):
        (data / rel).mkdir(parents=True)
    (data / "resnet18_cifar.pt").write_text("synthetic, not a model")
    (data / "CIFAR-10-C/labels.npy").write_text("synthetic, not labels")
    (root / "external/poem").mkdir(parents=True)
    (root / "external/poem/main.py").write_text("# never executed\n")
    protocol, lock = root / "protocol with spaces.yaml", root / "lock with spaces.json"
    protocol.write_text("synthetic: true\n")
    lock.write_text("{}\n")
    if relative == POEM:
        executable(bin_dir / "ls", f'#!{sys.executable}\nprint("class\\n" * 1000, end="")\n')
    script = root / relative
    script.parent.mkdir(parents=True, exist_ok=True)
    source = (ROOT / relative).read_text()
    # Red-phase transport shield only: any removed host probes select an absent
    # fixture path, never a user's actual home/Conda runtime. Candidate has none.
    source = source.replace("$HOME/.venv_wilds/bin/python", str(tmp_path / "absent runtime"))
    source = source.replace("/opt/anaconda3/envs/poem/bin/python", str(tmp_path / "absent runtime"))
    source = source.replace("/opt/anaconda3/envs/ag311/bin/python", str(tmp_path / "absent runtime"))
    script.write_text(source)
    env = dict(os.environ)
    for key in (
        "PY",
        "CPY",
        "KBOUND_SAR_PYTHON",
        "KBOUND_POEM_PYTHON",
        "KBOUND_PYTHON",
        "KBOUND_CIFAR_ROOT",
        "KBOUND_DATA_ROOT",
        "WILDS_DATA_ROOT",
        "IMAGENETR_DIR",
        "IMAGENETC_ROOT",
    ):
        env.pop(key, None)
    env.update(
        PY=str(runtime),
        KBOUND_SAR_PYTHON=str(runtime),
        KBOUND_POEM_PYTHON=str(runtime),
        KBOUND_PYTHON=str(runtime),
        KBOUND_CIFAR_ROOT=str(data),
        KBOUND_DATA_ROOT=str(data),
        WILDS_DATA_ROOT=str(data),
        IMAGENETR_DIR=str(data / "imagenetr"),
        IMAGENETC_ROOT=str(data / "imagenetc"),
        KBOUND_CLOSURE_PROTOCOL=str(protocol),
        KBOUND_CLOSURE_LOCK=str(lock),
        KBOUND_REPO_ROOT=str(root),
        STUB_ROOT=str(root),
        STUB_LOG=str(log),
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        ONLY="imagenetr",
        SKIP_CAM="1",
        SEEDS="0",
        SEVERITIES="5",
        CORRUPTIONS="gaussian_noise",
        FAIL_PAYLOAD="0",
        FAIL_WILDS="0",
        TMPDIR=str(tmp_path / "tmp"),
        TORCH_HOME=str(tmp_path / "torch"),
    )
    results = root / "experiments/kbound/results/natural_win_v1_imagenetr"
    results.mkdir(parents=True)
    (results / "per_condition_imagenet-r_fixture_seed0.json").write_text("{}")
    return root, script, env, log


def run(script, env, mode=None):
    return subprocess.run(
        ["/bin/bash", str(script), *([mode] if mode else [])], env=env, capture_output=True, text=True, timeout=15
    )


def calls(log):
    return [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize(
    "script,var",
    [
        (SAR, "KBOUND_SAR_PYTHON"),
        (SAR, "KBOUND_CIFAR_ROOT"),
        (POEM, "KBOUND_POEM_PYTHON"),
        (POEM, "IMAGENETC_ROOT"),
        (CLOSURE, "KBOUND_PYTHON"),
        (CLOSURE, "KBOUND_DATA_ROOT"),
        (NATURAL, "PY"),
        (NATURAL, "IMAGENETR_DIR"),
        (REMAINING, "PY"),
        (REMAINING, "IMAGENETR_DIR"),
    ],
)
def test_absent_required_configuration_fails_before_python(tmp_path, script, var):
    _, path, env, log = setup(tmp_path, script)
    env.pop(var)
    result = run(path, env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not calls(log), "missing configuration must not invoke Python"
    assert var in result.stdout + result.stderr


@pytest.mark.parametrize("script", [SAR, POEM, CLOSURE, NATURAL, REMAINING])
def test_explicit_runtime_and_data_paths_with_spaces_reach_only_selected_runtime(tmp_path, script):
    _, path, env, log = setup(tmp_path, script)
    result = run(path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls(log) and {row["runtime"] for row in calls(log)} == {env["PY"]}
    if script in (POEM, NATURAL, REMAINING):
        flat = [arg for row in calls(log) for arg in row["args"]]
        expected = env["IMAGENETC_ROOT"] if script == POEM else env["IMAGENETR_DIR"]
        assert expected in flat


@pytest.mark.parametrize("script", [NATURAL, REMAINING])
def test_unrequested_camelyon_needs_neither_wilds_root_nor_cpy(tmp_path, script):
    _, path, env, log = setup(tmp_path, script)
    env.pop("WILDS_DATA_ROOT")
    env["CPY"] = str(tmp_path / "not executable")
    env["FAIL_WILDS"] = "19"
    result = run(path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all("import wilds" not in " ".join(row["args"]) for row in calls(log))


@pytest.mark.parametrize("script", [NATURAL, REMAINING])
def test_requested_camelyon_missing_data_fails_before_any_python(tmp_path, script):
    _, path, env, log = setup(tmp_path, script)
    env.update(SKIP_CAM="0", ONLY="both")
    env.pop("WILDS_DATA_ROOT")
    result = run(path, env)
    assert result.returncode != 0 and not calls(log), result.stdout + result.stderr


@pytest.mark.parametrize("script", [NATURAL, REMAINING])
def test_both_requested_missing_wilds_prevents_either_training_arm(tmp_path, script):
    _, path, env, log = setup(tmp_path, script)
    env.update(SKIP_CAM="0", ONLY="both", FAIL_WILDS="19")
    result = run(path, env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not any(
        "run_imagenetr_kbound.py" in a or "run_camelyon17_kbound.py" in a for row in calls(log) for a in row["args"]
    )


def test_poem_missing_requested_corruption_stops_before_runtime_preflight(tmp_path):
    _, path, env, log = setup(tmp_path, POEM)
    env["CORRUPTIONS"] = "gaussian_noise shot_noise"
    result = run(path, env)
    assert result.returncode != 0 and not calls(log), result.stdout + result.stderr


@pytest.mark.parametrize("script", [NATURAL, REMAINING])
def test_explicit_camelyon_runtime_overrides_only_that_arm(tmp_path, script):
    root, path, env, log = setup(tmp_path, script)
    cpy = tmp_path / "second runtime with spaces" / "python"
    executable(cpy, Path(env["PY"]).read_text())
    env.update(CPY=str(cpy), SKIP_CAM="0", ONLY="both")
    resultdir = root / "experiments/kbound/results/natural_win_v1_camelyon"
    resultdir.mkdir()
    (resultdir / "per_condition_camelyon17_fixture_seed0.json").write_text("{}")
    result = run(path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    payloads = [row for row in calls(log) if any("run_camelyon17_kbound.py" in a for a in row["args"])]
    assert len(payloads) == 1 and payloads[0]["runtime"] == str(cpy)
    inr = [row for row in calls(log) if any("run_imagenetr_kbound.py" in a for a in row["args"])]
    assert len(inr) == 1 and inr[0]["runtime"] == env["PY"]


@pytest.mark.parametrize(
    "script,var",
    [
        (SAR, "KBOUND_SAR_PYTHON"),
        (POEM, "KBOUND_POEM_PYTHON"),
        (CLOSURE, "KBOUND_PYTHON"),
        (NATURAL, "PY"),
        (REMAINING, "PY"),
    ],
)
def test_invalid_explicit_runtime_never_falls_back(tmp_path, script, var):
    _, path, env, log = setup(tmp_path, script)
    env[var] = str(tmp_path / "missing runtime")
    result = run(path, env)
    assert result.returncode != 0 and not calls(log), result.stdout + result.stderr


def test_sar_hash_mismatch_still_blocks_python(tmp_path):
    _, path, env, log = setup(tmp_path, SAR)
    env["FAIL_HASH"] = "1"
    result = run(path, env, "smoke")
    assert result.returncode == 2 and "HASH MISMATCH" in result.stderr and not calls(log)


@pytest.mark.parametrize("execute", ["0", "1"])
def test_closure_execute_flag_remains_explicit_and_lock_bound(tmp_path, execute):
    _, path, env, log = setup(tmp_path, CLOSURE)
    env["KBOUND_EXECUTE"] = execute
    result = run(path, env, "evaluate")
    assert result.returncode == 0, result.stdout + result.stderr
    stage = next(row["args"] for row in calls(log) if any("run_closure_stage.py" in a for a in row["args"]))
    assert stage[:2] == ["docs/research/kbound/scripts/run_closure_stage.py", "evaluate"]
    assert stage[stage.index("--protocol") + 1] == env["KBOUND_CLOSURE_PROTOCOL"]
    assert stage[stage.index("--lock") + 1] == env["KBOUND_CLOSURE_LOCK"]
    assert ("--execute" in stage) is (execute == "1")


def load_parser():
    spec = importlib.util.spec_from_file_location("external_fixture", ROOT / VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_macros


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\newcommand{\Alpha}{0.0016}", {"Alpha": 0.0016}),
        (r"\newcommand{\Beta}{$-12.50$}", {"Beta": -12.5}),
        (r"\newcommand{\Alpha}{x}\newcommand{\Beta}{NaN}", {}),
        (r"\newcommand{\A1}{3}\renewcommand{\Beta}{4}", {}),
        (r"\newcommand{\Alpha}{1}\newcommand{\Alpha}{2}", {"Alpha": 2.0}),
        (r"\newcommand{\Beta}{1e-3}\newcommand {\Alpha}{8}", {"Beta": 1.0}),
    ],
)
def test_tex_backslash_spelling_preserves_existing_parser_semantics(tmp_path, text, expected):
    path = tmp_path / "synthetic macros.tex"
    path.write_text(text)
    assert load_parser()(path) == expected
    old = {}
    for name, body in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text):
        match = re.search(r"-?\d+\.?\d*", body.replace("$", ""))
        if match:
            old[name] = float(match.group())
    assert load_parser()(path) == old


@pytest.mark.parametrize("relative", [SAR, POEM, CLOSURE, VERIFY, NATURAL, REMAINING])
def test_public_source_scanner_accepts_portable_interface(relative):
    from docs.research.kbound.scripts.release_privacy import scan_member

    scan_member(relative, (ROOT / relative).read_bytes())


@pytest.mark.parametrize(
    "text",
    [
        r"\\server\share\result.json",
        r"C:\Users\example\data.txt",
        "/Users/example/private.txt",
        "/private/tmp/result.json",
    ],
)
def test_real_private_paths_remain_rejected(text):
    from docs.research.kbound.scripts.release_privacy import PrivacyError, scan_member

    with pytest.raises(PrivacyError):
        scan_member("scripts/synthetic.txt", text.encode())
