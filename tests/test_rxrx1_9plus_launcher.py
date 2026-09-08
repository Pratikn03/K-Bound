from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "docs/research/kbound/scripts/run_rxrx1_9plus.sh"
KBTRAIN = ROOT / "docs/research/kbound/scripts/kbtrain.sh"


def test_rxrx1_9plus_dry_run_prints_locked_protocol_command(tmp_path) -> None:
    # Fix-queue item 30 (reproducibility hygiene).  This test used to assert on
    # "--ckpt /Users/pratik_n/kbound_rxrx1_ckpt/...", i.e. on one author's home
    # directory, which EXTERNAL_STORAGE_POLICY.md:18 bans in tracked code and
    # which makes the assertion fail for every other user.  The launcher already
    # honours RXRX1_CKPT_ROOT (run_rxrx1_9plus.sh:63), so we pin that instead and
    # assert against the value we set.
    ckpt_root = tmp_path / "kbound_rxrx1_ckpt"
    ckpt_root.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "RXRX1_MODEL_SEEDS": "0",
            "RXRX1_CONDITION_SEEDS": "0 1",
            "RXRX1_N_EVAL": "64",
            "RXRX1_RESULTS_ROOT": str(tmp_path / "rxrx1_results_test"),
            "RXRX1_CKPT_ROOT": str(ckpt_root),
        }
    )

    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    assert f"--ckpt {ckpt_root}/rxrx1_seed:0_epoch:best_model.pth" in proc.stdout
    assert "/Users/" not in proc.stdout, "launcher must not emit a machine-local home path"
    assert "--seeds 0 1" in proc.stdout
    assert "--n-eval 64" in proc.stdout
    assert "--n-batches 4" in proc.stdout
    assert "--run-name rxrx1_protocol_c_9plus_modelseed0" in proc.stdout
    assert "tent_online" not in proc.stdout  # adapters are selected by the runner, not shell text.


def test_kbtrain_exposes_rxrx1_9plus_one_command() -> None:
    text = KBTRAIN.read_text()
    assert "rxrx1-9plus)" in text
    assert "run_rxrx1_9plus.sh" in text


def _make_fake_runtime(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "fake-venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text(
        'PATH="${RXRX1_VENV}/bin:${PATH}"\nexport PATH\n',
        encoding="utf-8",
    )
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/sh
if [ "${1-}" = "-c" ]; then
  exit 0
fi
{
  printf '%s\\n' __CALL__
  printf '%s\\n' "$@"
} >> "$RXRX1_CAPTURE"
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    caffeinate = bin_dir / "caffeinate"
    caffeinate.write_text(
        """#!/bin/sh
if [ "${1-}" = "-is" ]; then
  shift
fi
exec "$@"
""",
        encoding="utf-8",
    )
    caffeinate.chmod(0o755)
    return venv, tmp_path / "child-calls.txt"


def _launcher_env(tmp_path: Path, venv: Path, capture: Path) -> dict[str, str]:
    data_root = tmp_path / "data"
    (data_root / "rxrx1_v1.0").mkdir(parents=True)
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()
    results_root = tmp_path / "results"
    env = {key: value for key, value in os.environ.items() if not key.startswith("RXRX1_")}
    env.update({
        "RXRX1_VENV": str(venv),
        "RXRX1_DATA_ROOT": str(data_root),
        "RXRX1_CKPT_ROOT": str(ckpt_root),
        "RXRX1_RESULTS_ROOT": str(results_root),
        "RXRX1_CAPTURE": str(capture),
        "RXRX1_MODEL_SEEDS": "0 1 2 3 4",
        "RXRX1_CONDITION_SEEDS": "7 9",
    })
    return env


def _checkpoint(env: dict[str, str], model_seed: int) -> Path:
    return Path(env["RXRX1_CKPT_ROOT"]) / f"rxrx1_seed:{model_seed}_epoch:best_model.pth"


def _captured_calls(capture: Path) -> list[list[str]]:
    chunks = capture.read_text(encoding="utf-8").split("__CALL__\n")
    return [chunk.strip().splitlines() for chunk in chunks if chunk.strip()]


def _value_after(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_each_checkpoint_is_launched_with_its_matching_model_seed(tmp_path):
    venv, capture = _make_fake_runtime(tmp_path)
    env = _launcher_env(tmp_path, venv, capture)
    for model_seed in range(5):
        _checkpoint(env, model_seed).touch()

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), "--allow-concurrent"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = _captured_calls(capture)
    assert len(calls) == 5
    for model_seed, args in enumerate(calls):
        assert args[0] == str(ROOT / "experiments/kbound/wilds/run_rxrx1_kbound.py")
        assert _value_after(args, "--ckpt") == str(_checkpoint(env, model_seed))
        assert _value_after(args, "--model-seed") == str(model_seed)
        assert _value_after(args, "--run-name") == f"rxrx1_protocol_c_9plus_modelseed{model_seed}"
        seeds_start = args.index("--seeds") + 1
        assert args[seeds_start:args.index("--compositions")] == ["7", "9"]


def test_missing_checkpoint_fails_before_python_child(tmp_path):
    venv, capture = _make_fake_runtime(tmp_path)
    env = _launcher_env(tmp_path, venv, capture)
    _checkpoint(env, 0).touch()

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), "--allow-concurrent"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Missing RxRx1 base-model checkpoint(s):" in completed.stdout
    for model_seed in range(1, 5):
        assert str(_checkpoint(env, model_seed)) in completed.stdout
    assert not capture.exists()
