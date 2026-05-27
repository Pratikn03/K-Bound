"""Phase 2.2B — RGA-v2 gate-threshold selection must never read test-fold data."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "src" / "scripts" / "run_phase2_rga_v2_gate_sweep.py"


def test_selection_function_takes_only_validation_inputs():
    """The selector function signature must accept val_features / val_masks
    / val_labels and must not accept test_features / test_masks /
    test_labels."""
    src = DRIVER.read_text()
    sig = re.search(r"def _select_tau_on_validation_only\([^)]*\)", src, re.DOTALL)
    assert sig is not None, "missing _select_tau_on_validation_only"
    body = sig.group(0)
    assert "val_features" in body
    assert "val_masks" in body
    assert "val_labels" in body
    assert "test_features" not in body
    assert "test_masks" not in body


def test_validation_fold_corruption_function_does_not_accept_test_data():
    import inspect

    from elara.family_b.corruption import validation_fold_corruption_grid

    sig = inspect.signature(validation_fold_corruption_grid)
    params = set(sig.parameters)
    assert "val_features" in params
    assert "val_masks" in params
    assert "test_features" not in params
    assert "test_masks" not in params


def test_driver_stamps_selection_used_test_metrics_false_in_records():
    src = DRIVER.read_text()
    assert "selection_used_test_metrics" in src
    assert "False" in src
