#!/usr/bin/env python3
"""Fit validation-only isotonic calibrators per domain (T2) and freeze artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uais.fusion.attention.frozen_calibrators import (
    fit_isotonic_calibrators_from_csv,
    write_calibrator_lock,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _dataset_specs(root: Path) -> list[tuple[str, str, str, tuple[str, ...], list[str] | None]]:
    """name, csv_rel, split_col, val_values, domain_order."""
    return [
        (
            "elara_bench_la",
            "experiments/fusion/real_domain_fusion_inputs.csv",
            "fusion_split",
            ("validation",),
            None,
        ),
        (
            "mvtec3d_patchcore",
            "experiments/fusion/mvtec3d_patchcore_inputs.csv",
            "split",
            ("validation",),
            None,
        ),
        (
            "mvtec3d_patchcore_v2",
            "experiments/fusion/mvtec3d_patchcore_v2_inputs.csv",
            "split",
            ("validation",),
            None,
        ),
        (
            "m2_external_3d_adam",
            "experiments/fusion/m2_external_3d_adam_sealed_inputs.csv",
            "split",
            ("validation",),
            ["rgb", "depth_or_xyz"],
        ),
        (
            "eyecandies_dev",
            "experiments/fusion/eyecandies_inputs.csv",
            "fusion_split",
            ("validation",),
            ["rgb", "depth"],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of dataset keys to fit (default: all present CSVs)",
    )
    args = parser.parse_args()
    root = _repo_root()
    bundles = {}
    for name, rel, split_col, val_values, domain_order in _dataset_specs(root):
        if args.only and name not in args.only:
            continue
        path = root / rel
        if not path.is_file():
            print(f"skip {name}: missing {path}")
            continue
        try:
            bundle = fit_isotonic_calibrators_from_csv(
                path,
                dataset_key=name,
                split_col=split_col,
                val_values=val_values,
                domain_order=domain_order,
            )
        except KeyError as exc:
            print(f"skip {name}: column error ({exc})")
            continue
        if not bundle.models:
            print(f"skip {name}: no fitted domains — {bundle.meta}")
            continue
        bundles[name] = bundle
        print(f"fitted {name}: {bundle.meta}")

    if not bundles:
        print("ERROR: no calibrator bundles fitted", file=sys.stderr)
        return 1

    merge = not bool(args.only)
    lock = write_calibrator_lock(root, bundles, merge_existing=merge)
    print(f"Calibrator lock: {lock} (wrote {len(bundles)} dataset(s); merge={merge})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
