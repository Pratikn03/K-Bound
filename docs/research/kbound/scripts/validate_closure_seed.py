#!/usr/bin/env python3
"""Fail closed when a PACS or ImageNet-R closure seed is partial or mislabeled."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PACS_DOMAINS = {"art_painting", "cartoon", "photo", "sketch"}
INR_METHODS = {
    "resnet101", "resnet152", "resnext101_32x8d", "efficientnet_b0",
    "efficientnet_b3", "convnext_tiny", "convnext_base", "vit_b_16",
    "swin_t", "swin_b",
}


def validate_pacs(path: Path, seed: int) -> None:
    data = json.loads(path.read_text())
    if data.get("dataset") != "PACS" or data.get("seed") != seed:
        raise ValueError(f"{path}: expected PACS seed {seed}")
    if set(data.get("per_domain", {})) != PACS_DOMAINS:
        raise ValueError(f"{path}: missing or unexpected PACS domains")
    override = data.get("win_hunt_v5_override", {})
    expected = {"adapt_lr": 0.004, "batch_regimes": ["tiny"],
                "aggressiveness": ["aggressive"]}
    if override != expected:
        raise ValueError(f"{path}: protocol mismatch: {override} != {expected}")
    for domain, row in data["per_domain"].items():
        if row.get("n_test_cells") != 18:
            raise ValueError(f"{path}: {domain} has {row.get('n_test_cells')} cells, expected 18")
    print(f"VALID PACS seed {seed}: 4 domains x 18 cells")


def validate_imagenetr(run_dir: Path, seed: int) -> None:
    found = set()
    for method in sorted(INR_METHODS):
        path = run_dir / f"per_condition_imagenet-r_{method}_seed{seed}.json"
        if not path.is_file():
            raise ValueError(f"missing {path}")
        data = json.loads(path.read_text())
        records = data.get("records", [])
        if len(records) != 12:
            raise ValueError(f"{path}: {len(records)} records, expected 12")
        if any(int(r.get("seed", -1)) != seed for r in records):
            raise ValueError(f"{path}: contains a wrong seed")
        keys = [r.get("condition") for r in records]
        if len(set(keys)) != 12:
            raise ValueError(f"{path}: duplicate/missing conditions")
        found.add(method)
    if found != INR_METHODS:
        raise ValueError("ImageNet-R backbone set mismatch")
    print(f"VALID ImageNet-R seed {seed}: 10 backbones x 12 conditions")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("pacs")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    i = sub.add_parser("imagenetr")
    i.add_argument("--run-dir", type=Path, required=True)
    i.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    if args.kind == "pacs":
        validate_pacs(args.file, args.seed)
    else:
        validate_imagenetr(args.run_dir, args.seed)


if __name__ == "__main__":
    main()
