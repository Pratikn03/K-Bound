#!/usr/bin/env python3
"""Re-build MVTec 3D PatchCore expert scores (Gate A blocker: weak depth complement).

Runs prepare_mvtec3d_fusion_benchmark with stronger defaults, then re-runs Gate A.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--patchcore-k", type=int, default=5)
    parser.add_argument("--patchcore-coreset-size", type=int, default=2048)
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    out_csv = root / "experiments/fusion/mvtec3d_patchcore_v2_inputs.csv"
    out_meta = root / "experiments/fusion/mvtec3d_patchcore_v2_metadata.json"

    prepare = [
        py,
        "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
        "--dataset-root",
        "data/raw/mvtec3d",
        "--feature-mode",
        "patchcore",
        "--embedding-dim",
        str(args.embedding_dim),
        "--patchcore-k",
        str(args.patchcore_k),
        "--patchcore-coreset-size",
        str(args.patchcore_coreset_size),
        "--output",
        str(out_csv),
        "--metadata",
        str(out_meta),
    ]
    qualify = [
        py,
        "src/scripts/scenario_c/qualify_upstream_experts.py",
        "--csv",
        str(out_csv),
        "--metadata",
        str(out_meta),
        "--json-out",
        str(root / "elara_master_c/audits/gate_a_expert_qualification_v2.json"),
        "--export-parquet",
    ]
    for cmd in (prepare, qualify):
        print("$", " ".join(cmd))
    if args.dry_run:
        return 0
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "src")}
    if subprocess.call(prepare, cwd=root, env=env) != 0:
        return 1
    return subprocess.call(qualify, cwd=root, env=env)


if __name__ == "__main__":
    sys.exit(main())
