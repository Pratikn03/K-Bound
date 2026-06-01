#!/usr/bin/env python3
"""Acquire (if needed), prepare, and seal external M2 v2 MulSen-AD benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root}"
    return subprocess.call(cmd, cwd=root, env=env)


def _category_ready(local_dir: Path, category: str) -> bool:
    return (local_dir / category / "RGB" / "train").is_dir() and any(
        (local_dir / category / "RGB" / "train").glob("*.png")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--feature-mode", default="patchcore", choices=["patchcore", "m3dm", "image_statistics"])
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    seal_path = root / "research_lock/M2_EXTERNAL_SEALED_v2.yaml"
    seal = yaml.safe_load(seal_path.read_text(encoding="utf-8"))
    proto = seal["heldout_protocol"]
    local_root = root / seal["source"]["local_root"]
    train_cats = proto["train_categories"]

    if not args.skip_download:
        rc = _run([py, "src/scripts/scenario_c/download_m2_external_mulsen.py"], root, "Download MulSen-AD")
        if rc != 0:
            return rc

    all_cats = sorted(set(train_cats) | set(proto["test_categories"]))
    missing = [c for c in all_cats if not _category_ready(local_root, c)]
    if missing:
        print(f"ERROR: missing MulSen categories: {missing}", file=sys.stderr)
        return 1

    prep_cmd = [
        py,
        "src/scripts/prepare_mulsen_fusion_benchmark.py",
        "--dataset-root",
        str(local_root.relative_to(root)),
        "--train-categories",
        *train_cats,
        "--feature-mode",
        args.feature_mode,
        "--embedding-dim",
        "16",
        "--patchcore-k",
        "5",
        "--patchcore-coreset-size",
        "2048",
        "--heldout-val-fraction",
        str(proto["validation_fraction_from_train_category_test_rows"]),
        "--heldout-val-seed",
        str(proto["validation_seed"]),
        "--output",
        seal["artifacts"]["inputs"],
        "--metadata",
        seal["artifacts"]["metadata"],
    ]
    rc = _run(prep_cmd, root, "Prepare MulSen external M2 fusion inputs")
    if rc != 0:
        return rc

    meta_path = root / seal["artifacts"]["metadata"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "dataset_id": seal["dataset_id"],
            "seal": str(seal_path.relative_to(root)),
            "source_repository": seal["source"]["repository"],
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rc = _run([py, "src/scripts/scenario_c/generate_split_hashes.py"], root, "Refresh split hashes")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
