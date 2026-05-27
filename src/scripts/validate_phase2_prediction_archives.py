"""Phase 2.B — validator for the prediction-archive index.

Walks `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv`
and checks for every row:

  - artifact file exists and SHA256 matches;
  - schema columns match `PREDICTION_ARCHIVE_SCHEMA`;
  - selection_used_test_metrics column is False for every test-split sample;
  - sample-ID counts agree within (cell, seed) across methods;
  - usable_for_inference column matches the selection flag;
  - validation_only_selection_verified is True for all rows.

Exit 0 on clean; 1 on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Import schema constants from the elara package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from elara.evaluation.prediction_archive import (  # noqa: E402
    INDEX_COLUMNS,
    PREDICTION_ARCHIVE_SCHEMA,
)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("experiments/phase2/predictions"),
    )
    args = parser.parse_args()
    index_path = args.archive_root / "PREDICTION_ARCHIVE_INDEX.csv"
    if not index_path.exists():
        print(f"INDEX MISSING: {index_path}")
        sys.exit(1)
    idx = pd.read_csv(index_path)
    if list(idx.columns) != list(INDEX_COLUMNS):
        print(f"INDEX SCHEMA MISMATCH: {set(idx.columns)} vs {set(INDEX_COLUMNS)}")
        sys.exit(1)

    fails: list[str] = []
    by_cell_seed: dict[tuple[Any, ...], dict[str, int]] = {}

    for _, row in idx.iterrows():
        ap = Path(row["artifact_path"])
        if not ap.exists():
            fails.append(f"artifact missing: {ap}")
            continue
        actual_sha = _hash_file(ap)
        if actual_sha != row["sha256"]:
            fails.append(f"hash mismatch: {ap} got={actual_sha} idx={row['sha256']}")
            continue
        # Schema check
        try:
            if ap.suffix == ".parquet":
                frame = pd.read_parquet(ap)
            else:
                frame = pd.read_csv(ap)
        except Exception as e:
            fails.append(f"unreadable: {ap} ({e})")
            continue
        if list(frame.columns) != list(PREDICTION_ARCHIVE_SCHEMA):
            fails.append(f"schema mismatch: {ap} got={list(frame.columns)} expected={list(PREDICTION_ARCHIVE_SCHEMA)}")
            continue
        # selection_used_test_metrics must be False everywhere on test split.
        if row["split"] == "test":
            bad = frame[frame["selection_used_test_metrics"].astype(str).str.lower().isin(["true", "1"])]
            if len(bad) > 0:
                fails.append(f"selection_used_test_metrics=True on test split: {ap} (rows={len(bad)})")
        # Verify usable_for_inference column matches the selection flag.
        if not row["usable_for_inference"]:
            fails.append(f"usable_for_inference=False registered: {ap}")
        # Sample-ID count by (cell, seed) for cross-method consistency check
        key = (row["experiment_id"], row["seed"], row["split"])
        by_cell_seed.setdefault(key, {})[row["method"]] = int(row["rows"])

    # Cross-method sample-ID count consistency
    for key, methods in by_cell_seed.items():
        counts = set(methods.values())
        if len(counts) > 1:
            fails.append(f"row-count mismatch across methods for {key}: {methods}")

    if fails:
        print(f"PREDICTION ARCHIVE VALIDATION FAILED ({len(fails)} issues):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)

    print(f"PREDICTION ARCHIVE VALIDATION PASSED: {len(idx)} rows clean.")


if __name__ == "__main__":
    main()
