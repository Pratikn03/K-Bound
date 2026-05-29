"""Validity-guard regression tests for confirmatory_statistics.evaluate_cell.

These lock the integrity fix that prevents a degenerate seed-bootstrap CI
(deterministic methods -> identical per-seed deltas) or a below-chance cell
(AUC < 0.5) from ever reading as a passed gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.scenario_c.confirmatory_statistics import evaluate_cell


def _row(seed: int, rga: float, base: float, static: float, craf: float) -> dict:
    return {
        "seed": seed,
        "rga_boosted_fusion": {"roc_auc": rga},
        # include both candidate comparators with identical values so the test
        # is robust to whichever frozen comparator is selected.
        "sar_score_adapter": {"roc_auc": base},
        "tent_score_adapter": {"roc_auc": base},
        "static_attention": {"roc_auc": static},
        "craf_attention": {"roc_auc": craf},
    }


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"table_1_clean_performance": rows}), encoding="utf-8")
    return p


def test_degenerate_ci_zero_seed_variance_is_invalid(tmp_path: Path) -> None:
    # All five "seeds" identical (deterministic methods). rga > base and both
    # above chance, but the seed-bootstrap CI collapses to a point.
    rows = [_row(s, rga=0.83, base=0.80, static=0.80, craf=0.80) for s in range(42, 47)]
    cell = evaluate_cell(_write(tmp_path, rows), benchmark="UNIT_TEST", protocol="unit", family="M1")
    assert cell["cell_valid"] is False
    assert any("degenerate_ci" in r for r in cell["validity_reasons"])
    assert cell["gate_d_pass"] is False
    assert cell["gate_e_pass"] is False
    assert cell["t5_confirmatory_pass"] is False


def test_below_chance_cell_is_invalid(tmp_path: Path) -> None:
    # Methods below chance (AUC < 0.5) with genuine seed variance -> still invalid.
    rga_vals = [0.39, 0.40, 0.38, 0.41, 0.37]
    base_vals = [0.388, 0.395, 0.378, 0.405, 0.366]
    rows = [
        _row(42 + i, rga=rga_vals[i], base=base_vals[i], static=0.40, craf=0.40)
        for i in range(5)
    ]
    cell = evaluate_cell(_write(tmp_path, rows), benchmark="UNIT_TEST", protocol="unit", family="M2")
    assert cell["cell_valid"] is False
    assert any("below_chance" in r for r in cell["validity_reasons"])
    assert cell["gate_d_pass"] is False
    assert cell["gate_e_pass"] is False
    assert cell["t5_confirmatory_pass"] is False


def test_genuine_significant_superiority_passes(tmp_path: Path) -> None:
    # Real seed variance, clearly above chance, rga >> base, significant paired
    # test -> the cell is valid and all gates pass.
    rga_vals = [0.82, 0.83, 0.81, 0.84, 0.80]
    base_vals = [0.71, 0.70, 0.72, 0.69, 0.71]
    rows = [
        _row(42 + i, rga=rga_vals[i], base=base_vals[i], static=0.71, craf=0.71)
        for i in range(5)
    ]
    cell = evaluate_cell(_write(tmp_path, rows), benchmark="UNIT_TEST", protocol="unit", family="M1")
    assert cell["cell_valid"] is True
    assert cell["validity_reasons"] == []
    assert cell["gate_d_pass"] is True
    assert cell["gate_e_pass"] is True
    assert cell["t5_confirmatory_pass"] is True
