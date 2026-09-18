from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "docs/research/kbound/runbooks/verify_external_repro.py"
DECLARED_METRICS = {
    "cifar_tent_kga_regret": "CIFARtentKga",
    "cifar_tent_adapt_regret": "CIFARtentAdapt",
    "cifar_tent_freeze_regret": "CIFARtentFreeze",
    "cifar_tent_fa_u": "CIFARtentFA",
    "cifar_eata_kga_regret": "CIFAReataKga",
    "cifar_eata_adapt_regret": "CIFAReataAdapt",
    "cifar_eata_freeze_regret": "CIFAReataFreeze",
    "cifar_eata_fa_u": "CIFAReataFA",
    "headtohead_kga_regret": "HeadToHeadKga",
    "headtohead_poem_regret": "HeadToHeadPoem",
    "headtohead_aetta_regret": "HeadToHeadAetta",
}


@pytest.fixture
def external_repro_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, float]]:
    numbers = tmp_path / "numbers.tex"
    numbers.write_text(
        "".join(f"\\newcommand{{\\{macro}}}{{1.0000}}\n" for macro in DECLARED_METRICS.values()),
        encoding="utf-8",
    )
    theirs = tmp_path / "their_results.json"
    complete = dict.fromkeys(DECLARED_METRICS, 1.0)
    return numbers, theirs, complete


def _run_verifier(numbers: Path, theirs: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--their-results",
            str(theirs),
            "--numbers-tex",
            str(numbers),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_external_repro_accepts_one_complete_exact_metric_contract(external_repro_inputs) -> None:
    numbers, theirs, complete = external_repro_inputs
    theirs.write_text(json.dumps(complete), encoding="utf-8")

    result = _run_verifier(numbers, theirs)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "11 metrics checked, 0 failures" in result.stdout


@pytest.mark.parametrize("problem", ["empty", "missing", "extra"])
def test_external_repro_rejects_incomplete_or_extra_metric_contract(external_repro_inputs, problem: str) -> None:
    numbers, theirs, complete = external_repro_inputs
    payload = complete.copy()
    if problem == "empty":
        payload = {}
    elif problem == "missing":
        payload.pop("headtohead_aetta_regret")
    else:
        payload["undeclared_metric"] = 1.0
    theirs.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_verifier(numbers, theirs)

    assert result.returncode != 0
    assert "INVALID RESULTS" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("bad_value", [True, "1.0", None, [], {}])
def test_external_repro_rejects_non_numeric_metric_types(external_repro_inputs, bad_value: object) -> None:
    numbers, theirs, complete = external_repro_inputs
    payload: dict[str, object] = complete.copy()
    payload["cifar_tent_kga_regret"] = bad_value
    theirs.write_text(json.dumps(payload), encoding="utf-8")

    result = _run_verifier(numbers, theirs)

    assert result.returncode != 0
    assert "INVALID RESULTS" in result.stdout
    assert "Traceback" not in result.stderr


def test_external_repro_rejects_nonfinite_json_constant(external_repro_inputs) -> None:
    numbers, theirs, complete = external_repro_inputs
    serialized = json.dumps(complete).replace('"cifar_tent_kga_regret": 1.0', '"cifar_tent_kga_regret": NaN')
    theirs.write_text(serialized, encoding="utf-8")

    result = _run_verifier(numbers, theirs)

    assert result.returncode != 0
    assert "INVALID RESULTS" in result.stdout
    assert "finite" in result.stdout.lower()
    assert "Traceback" not in result.stderr


def test_external_repro_rejects_duplicate_metric_key(external_repro_inputs) -> None:
    numbers, theirs, complete = external_repro_inputs
    serialized = json.dumps(complete)
    metric = '"cifar_tent_kga_regret": 1.0'
    theirs.write_text(serialized.replace(metric, f"{metric}, {metric}", 1), encoding="utf-8")

    result = _run_verifier(numbers, theirs)

    assert result.returncode != 0
    assert "INVALID RESULTS" in result.stdout
    assert "duplicate" in result.stdout.lower()
    assert "Traceback" not in result.stderr
