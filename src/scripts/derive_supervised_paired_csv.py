"""Derive a supervised-paired variant from an existing fusion CSV.

Re-uses the score and embedding columns from a canonical fusion CSV and
redistributes the test rows across train/validation/test stratified by
(category, label). This avoids re-running the slow ResNet feature
extraction when the canonical CSV already exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def derive(
    canonical_csv: Path,
    *,
    seed: int = 42,
    test_fraction: float = 0.30,
    val_fraction_of_remainder: float = 0.15 / 0.70,
) -> pd.DataFrame:
    df = pd.read_csv(canonical_csv)
    rng = np.random.default_rng(int(seed))

    # We operate at the sample_id level, then propagate the new split back to all
    # domain rows for that sample.
    samples = df.drop_duplicates("sample_id").reset_index(drop=True)
    test_mask = (samples["split"] == "test")
    test_samples = samples[test_mask].copy()
    keep_samples = samples[~test_mask].copy()

    # Stratify the test sample reassignment by (category, label).
    new_split_per_sample: dict[str, str] = dict(zip(keep_samples["sample_id"], keep_samples["split"]))
    for (_cat, _label), group in test_samples.groupby(["category", "label"], sort=False):
        ids = group["sample_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        remaining = n - n_test
        n_val = max(1, int(round(remaining * val_fraction_of_remainder))) if remaining >= 2 else 0
        for offset, sample_id in enumerate(ids):
            if offset < n_test:
                new_split_per_sample[sample_id] = "test"
            elif offset < n_test + n_val:
                new_split_per_sample[sample_id] = "validation"
            else:
                new_split_per_sample[sample_id] = "train"

    df_new = df.copy()
    df_new["split"] = df_new["sample_id"].map(new_split_per_sample).fillna("train")
    return df_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    args = parser.parse_args()

    derived = derive(args.canonical_csv, seed=args.seed, test_fraction=args.test_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    samples = derived.drop_duplicates("sample_id")
    print("Split distribution:")
    print(samples["split"].value_counts())
    print("Per-split label counts:")
    print(samples.groupby(["split", "label"]).size().unstack(fill_value=0))

    if args.metadata_output is not None:
        meta = {
            "derived_from": str(args.canonical_csv),
            "supervised_paired_protocol": {
                "test_rows_redistributed_across": ["train", "validation", "test"],
                "stratification_keys": ["category", "label"],
                "seed": int(args.seed),
                "test_fraction": float(args.test_fraction),
            },
            "n_samples": int(len(samples)),
            "split_counts": samples["split"].value_counts().to_dict(),
        }
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
