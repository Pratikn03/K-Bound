from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/kbound/wilds/run_iwildcam_kga_router.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "iwildcam_router_nonpromotable_guard", RUNNER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROUTER = _load_runner()


def test_runner_fails_closed_before_device_or_output_access(monkeypatch, tmp_path):
    args = ROUTER.parse_args(["--results-root", str(tmp_path)])

    def unexpected_device_access(_device):
        raise AssertionError("device access happened before the opt-in guard")

    monkeypatch.setattr(ROUTER.tm, "pick_device", unexpected_device_access)

    with pytest.raises(RuntimeError, match="--allow-nonpromotable-experimental"):
        ROUTER.run(args)

    assert list(tmp_path.iterdir()) == []


def test_explicit_opt_in_does_not_change_nonpromotable_status():
    args = ROUTER.parse_args(["--allow-nonpromotable-experimental"])
    status = ROUTER.nonpromotable_result_status()

    assert args.allow_nonpromotable_experimental is True
    assert status["evidence_status"] == "NON_PROMOTABLE_EXPERIMENTAL"
    assert status["release_ingestion_allowed"] is False
    assert status["release_verdict"] == "NOT_ELIGIBLE_NON_PROMOTABLE"
    assert "verdict" not in status
    assert status["metric_contract"] == {
        "name": "sklearn_f1_score_average_macro",
        "official_wilds_label_present": False,
        "status": "NON_OFFICIAL",
    }


def test_opt_in_flag_is_false_by_default():
    args = ROUTER.parse_args([])
    assert args.allow_nonpromotable_experimental is False
