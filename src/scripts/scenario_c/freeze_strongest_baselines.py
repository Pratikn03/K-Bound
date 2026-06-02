#!/usr/bin/env python3
"""Freeze validation-selected strongest baselines per benchmark (D5 / T3)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    root = _repo_root()
    py = sys.executable
    audit_csv = root / "experiments/audit/audited_comparator_selection.csv"
    if not audit_csv.is_file():
        subprocess.check_call(
            [py, "src/scripts/select_audited_validation_frozen_comparator.py"],
            cwd=root,
            env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        )
    import pandas as pd

    df = pd.read_csv(audit_csv)
    rows = []
    for _, r in df.iterrows():
        val_auc = r.get("selected_comparator_validation_auc")
        if isinstance(val_auc, float) and math.isfinite(val_auc):
            rows.append(
                {
                    "benchmark": r["benchmark"],
                    "protocol": r["protocol"],
                    "strongest_baseline": r["selected_comparator"],
                    "validation_roc_auc": float(val_auc),
                    "selection_used_test_metrics": False,
                }
            )
    out = {
        "version": 1,
        "source": "experiments/audit/audited_comparator_selection.csv",
        "rule": "max validation ROC-AUC among non-RGA baselines",
        "cells": rows,
    }
    lock = root / "research_lock/strongest_baseline_frozen_v1.json"
    lock.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Frozen {len(rows)} cells -> {lock}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
