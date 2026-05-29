#!/usr/bin/env python3
"""Acquire (if needed), prepare, and seal the external M2 3D-ADAM transfer benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _category_complete(local_dir: Path, category: str) -> bool:
    train_rgb = local_dir / category / "train" / "good" / "rgb"
    return train_rgb.is_dir() and any(train_rgb.glob("*.png"))


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--zip-download", action="store_true", help="Use adam3d_cropped.zip")
    parser.add_argument("--feature-mode", default="patchcore", choices=["patchcore", "m3dm", "image_statistics"])
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    seal_path = root / "research_lock/M2_EXTERNAL_SEALED_v1.yaml"
    seal = yaml.safe_load(seal_path.read_text(encoding="utf-8"))
    proto = seal["heldout_protocol"]
    local_root = root / seal["source"]["local_root"]
    train_cats = proto["train_categories"]
    val_seed = int(proto["validation_seed"])
    val_frac = float(proto["validation_fraction_from_train_category_test_rows"])

    if not args.skip_download:
        dl_cmd = [py, "src/scripts/scenario_c/download_m2_external_3d_adam.py"]
        if args.zip_download:
            dl_cmd.append("--zip")
        rc = _run(dl_cmd, root, "Download 3D-ADAM external M2")
        if rc != 0:
            return rc

    all_cats = sorted(set(train_cats) | set(proto["test_categories"]))
    missing = [c for c in all_cats if not _category_complete(local_root, c)]
    if missing:
        print(f"ERROR: missing categories on disk: {missing}", file=sys.stderr)
        print("Run download_m2_external_3d_adam.py until all categories are present.", file=sys.stderr)
        return 1

    prep_cmd = [
        py,
        "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
        "--dataset-root",
        str(local_root.relative_to(root)),
        "--feature-mode",
        args.feature_mode,
        "--embedding-dim",
        "32",
        "--patchcore-k",
        "5",
        "--patchcore-coreset-size",
        "2048",
        "--train-categories",
        *train_cats,
        "--heldout-val-fraction",
        str(val_frac),
        "--heldout-val-seed",
        str(val_seed),
        "--output",
        seal["artifacts"]["inputs"],
        "--metadata",
        seal["artifacts"]["metadata"],
    ]
    rc = _run(prep_cmd, root, "Prepare external M2 fusion inputs (category-held-out)")
    if rc != 0:
        return rc

    meta_path = root / seal["artifacts"]["metadata"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "dataset_id": seal["dataset_id"],
            "benchmark_type": "naturally_paired_3d_adam_score_fusion",
            "pairing_unit": "3D-ADAM MechMind-Nano RGB/XYZ scan (1:1 pixel mapping)",
            "source_repository": seal["source"]["repository"],
            "seal": str(seal_path.relative_to(root)),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rc = _run([py, "src/scripts/scenario_c/generate_split_hashes.py"], root, "Refresh split hashes")
    if rc != 0:
        return rc

    print("\nExternal M2 sealed. Artifacts:")
    for key in ("inputs", "metadata", "split_hash", "acquisition_record"):
        print(f"  {key}: {seal['artifacts'].get(key, seal['source'].get('local_root'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
