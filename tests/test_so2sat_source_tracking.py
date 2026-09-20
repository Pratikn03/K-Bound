"""Required source must be addable without force-adding protected directories."""
import subprocess
from pathlib import Path


def test_required_so2sat_sources_are_not_ignored():
    repo = Path(__file__).resolve().parents[1]
    required = [
        "experiments/kbound/so2sat/prospective_runner_v2.py",
        "experiments/kbound/so2sat/prospective_v2.py",
        "experiments/kbound/so2sat/v2_target.py",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *required],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1 and not result.stdout, result.stdout + result.stderr


def test_protected_so2sat_payloads_remain_ignored():
    repo = Path(__file__).resolve().parents[1]
    # Paths are invented. Git checks policy without reading target payloads.
    protected = [
        "experiments/kbound/so2sat/target_data/probe.npz",
        "experiments/kbound/so2sat/gate_calibration/example.json",
        "experiments/kbound/so2sat/results/example.json",
        "experiments/kbound/so2sat/new_unreviewed.py",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *protected],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(protected)
