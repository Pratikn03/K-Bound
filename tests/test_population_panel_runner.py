from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts.run_population_panel import build_population_panel


def test_population_panel_cli_is_runnable_from_repository_root():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "docs/research/kbound/scripts/run_population_panel.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_population_panel_replays_predictions_without_outcomes():
    result = build_population_panel(
        [
            {"cell_id": "cell-a", "delta_hat": 0.30, "epsilon": 0.02, "sample_size": 2000},
            {"cell_id": "cell-b", "delta_hat": 0.00, "epsilon": 0.02, "sample_size": 2000},
        ],
        alpha_cell=0.05,
        delta_sampling=0.05,
        alpha_population=0.10,
    )
    assert result["schema"] == "kbound_population_panel_v1"
    assert result["claim_scope"].startswith("population-interval action replay")
    assert result["records"][0]["action"] == "ADAPT"
    assert result["records"][1]["action"] == "ABSTAIN"


def test_population_panel_rejects_target_dependent_inputs():
    with pytest.raises(ValueError, match="outcome-dependent"):
        build_population_panel(
            [{"cell_id": "cell-a", "delta_hat": 0.3, "epsilon": 0.02, "sample_size": 2000, "outcome": 1}],
            alpha_cell=0.05,
            delta_sampling=0.05,
            alpha_population=0.10,
        )


def test_population_panel_requires_sample_size():
    with pytest.raises(ValueError, match="sample_size"):
        build_population_panel(
            [{"cell_id": "cell-a", "delta_hat": 0.3, "epsilon": 0.02}],
            alpha_cell=0.05,
            delta_sampling=0.05,
            alpha_population=0.10,
        )


@pytest.mark.parametrize("estimate", [float("inf"), float("-inf")])
def test_population_panel_rejects_nonfinite_prediction_before_issuing_action(estimate):
    with pytest.raises(ValueError, match="delta_hat.*finite"):
        build_population_panel(
            [{"cell_id": "invalid-estimate", "delta_hat": estimate,
              "epsilon": 0.02, "sample_size": 2000}],
            alpha_cell=0.05, delta_sampling=0.05, alpha_population=0.10,
        )
