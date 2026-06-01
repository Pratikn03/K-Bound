from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.scripts.scenario_c.run_positive_transfer_confirmatory import (
    is_official_confirmation,
)


def test_opened_3d_adam_result_cannot_be_official(tmp_path):
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "holdout_status": "OPENED_DEVELOPMENT_ONLY",
        "natural_clean_transfer": True,
        "synthetic_or_corrupted": False,
        "gate_e_positive_transfer_confirmed": True,
        "stats": {
            "vs_sar": {"delta": 0.05, "ci95": [0.02, 0.08], "valid": True},
            "vs_cw": {"delta": 0.01, "ci95": [0.006, 0.02], "valid": True},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert is_official_confirmation(path) is False


def test_sar_pass_without_cw_pass_is_not_official(tmp_path):
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "holdout_status": "FRESH_OR_UNOPENED",
        "natural_clean_transfer": True,
        "synthetic_or_corrupted": False,
        "stats": {
            "vs_sar": {"delta": 0.05, "ci95": [0.02, 0.08], "valid": True},
            "vs_cw": {"delta": 0.001, "ci95": [-0.001, 0.003], "valid": True},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert is_official_confirmation(path) is False


def test_both_positive_fresh_clean_endpoints_are_official(tmp_path):
    result = {
        "protocol": "POSITIVE_TRANSFER_PROTOCOL_v1",
        "holdout_status": "FRESH_OR_UNOPENED",
        "natural_clean_transfer": True,
        "synthetic_or_corrupted": False,
        "stats": {
            "vs_sar": {"delta": 0.02, "ci95": [0.011, 0.04], "valid": True},
            "vs_cw": {"delta": 0.008, "ci95": [0.006, 0.012], "valid": True},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    assert is_official_confirmation(path) is True


def test_development_runner_writes_development_only_report():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "src/scripts/scenario_c/run_positive_transfer_development.py",
            "--bootstrap-iter",
            "200",
        ],
        cwd=root,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(
        (root / "elara_master_c/audits/positive_transfer_development_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "DEVELOPMENT_ONLY"
    assert report["cannot_set_gate_e"] is True
    assert report["synthetic_or_corrupted"] is False
    assert {row["dataset_id"] for row in report["datasets"]} >= {"3d_adam_v3", "mulsen_v2"}

