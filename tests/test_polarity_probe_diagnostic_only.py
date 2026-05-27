"""Phase 1.F — polarity probe is diagnostic-only.

The probe must still emit per-seed flip logs, but the primary path
must not use them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LOG = Path("experiments/audit/polarity_diagnostic_log.csv")


def test_polarity_log_exists():
    assert LOG.exists(), f"polarity diagnostic log missing: {LOG}"


def test_log_columns():
    if not LOG.exists():
        pytest.skip("log missing")
    with LOG.open() as f:
        cols = csv.DictReader(f).fieldnames or []
    required = {
        "benchmark",
        "protocol",
        "method",
        "seed",
        "validation_probe_auc",
        "borderline_flag",
        "flip_would_have_been_applied_under_old_logic",
        "raw_test_roc_auc",
        "diagnostic_flipped_test_roc_auc",
        "raw_test_pr_auc",
        "diagnostic_flipped_test_pr_auc",
        "primary_metrics_use_flip",
    }
    assert required.issubset(set(cols)), f"missing columns: {required - set(cols)}"


def test_primary_metrics_use_flip_is_false_for_every_row():
    if not LOG.exists():
        pytest.skip("log missing")
    with LOG.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "polarity log is empty"
    for r in rows:
        flag = (r.get("primary_metrics_use_flip") or "").strip().lower()
        assert flag in {
            "false",
            "0",
        }, f"primary_metrics_use_flip must be False for every row (Phase 1.F lock). row: {r}"
