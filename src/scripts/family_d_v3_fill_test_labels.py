"""Phase 2.2E — Authorised final-metric label-fill step.

Reads test anomaly labels from Eyecandies metadata.yaml (the ONLY
point where anomaly labels are accessed). Updates the fusion CSV.

AUTHORISATION: This script may ONLY be called at the authorised
final-metric step in the Family-D execution sequence (Stage 5 per
FAMILY_D_EXECUTION_COMMANDS_v2_NOT_RUN.md), after model training
and validation calibration are complete.

INVARIANTS:
- Labels are read from metadata.yaml only — not from anomaly mask pixels.
- The `anomalous` field (0=normal, 1=anomalous) maps directly to label.
- train/val labels remain 0 (anomaly-free per official spec).
- Overwrites the existing fusion CSV in place.

Usage (one-time, at final-metric step only):
  PYTHONPATH=src python src/scripts/family_d_v3_fill_test_labels.py
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "raw" / "eyecandies" / "_archives"
FUSION_CSV = ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv"

CATEGORIES = [
    "CandyCane",
    "ChocolateCookie",
    "ChocolatePraline",
    "Confetto",
    "GummyBear",
    "HazelnutTruffle",
    "LicoriceSandwich",
    "Lollipop",
    "Marshmallow",
    "PeppermintCandy",
]
TEST_SPLITS = ["test_public", "test_private"]


def _extract_test_labels(cat: str) -> dict[str, dict[str, int]]:
    """Returns {split: {sample_id_str: label}} from metadata.yaml files."""
    archive = ARCHIVE_DIR / f"{cat}.tar"
    result: dict[str, dict[str, int]] = {s: {} for s in TEST_SPLITS}
    with tarfile.open(archive, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            if "_metadata.yaml" not in name:
                continue
            # Determine split
            split = None
            for s in TEST_SPLITS:
                if f"/{s}/" in name:
                    split = s
                    break
            if split is None:
                continue
            # Extract sample_id from filename e.g. 00_metadata.yaml → "00"
            base = name.rsplit("/", 1)[-1]  # "NN_metadata.yaml"
            sample_id = base.split("_")[0]
            if not sample_id.isdigit():
                continue
            # Read label
            content = tf.extractfile(m).read()
            meta = yaml.safe_load(content)
            label = int(meta.get("anomalous", 0))
            result[split][sample_id] = label
    return result


def main() -> int:
    if not FUSION_CSV.exists():
        raise SystemExit(f"Fusion CSV not found: {FUSION_CSV}")

    print(f"Loading {FUSION_CSV}...", flush=True)
    df = pd.read_csv(FUSION_CSV)

    # Verify test labels are still -1 (not yet filled)
    test_mask = df["fusion_split"] == "test"
    unique_test_labels = df.loc[test_mask, "label"].unique()
    if set(unique_test_labels) - {-1}:
        # Some labels already filled — check if this is a re-run
        non_placeholder = df.loc[test_mask & (df["label"] != -1)]
        print(
            f"WARNING: {len(non_placeholder)} test rows already have non-placeholder labels. "
            f"This may be a re-run. Proceeding to overwrite.",
            flush=True,
        )

    # Build label map for all test samples
    label_map: dict[str, int] = {}
    for cat in CATEGORIES:
        print(f"[{cat}] reading test labels from archive...", flush=True)
        cat_labels = _extract_test_labels(cat)
        for split, sid_map in cat_labels.items():
            for sid, lbl in sid_map.items():
                # sample_id format: <cat>__<split>__<sid>
                key = f"{cat}__{split}__{sid}"
                label_map[key] = lbl

    print(f"Total test label entries: {len(label_map)}", flush=True)

    # Apply: only update rows where fusion_split == "test" AND label == -1 (or all test)
    # We match on the unique part of sample_id (before domain)
    # sample_id in CSV: <cat>__<split>__<sid>  (same per rgb and depth row)
    filled = 0
    skipped = 0
    for idx in df[test_mask].index:
        sid = str(df.at[idx, "sample_id"])
        if sid in label_map:
            df.at[idx, "label"] = label_map[sid]
            filled += 1
        else:
            skipped += 1

    print(f"Filled {filled} test label cells; skipped {skipped} (no metadata found)", flush=True)

    # Verify: no -1 labels remain in test unless genuinely missing
    still_minus1 = df.loc[test_mask & (df["label"] == -1)]
    if len(still_minus1) > 0:
        print(f"WARNING: {len(still_minus1)} test rows still have label=-1 (no metadata)", flush=True)

    # Write back
    df.to_csv(FUSION_CSV, index=False)
    print(f"Updated {FUSION_CSV}", flush=True)

    # Summary
    print("\nLabel distribution after fill:")
    print(df["label"].value_counts().to_string())
    print()
    print("Test label distribution:")
    print(df.loc[test_mask, "label"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
