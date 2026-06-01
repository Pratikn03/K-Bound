#!/usr/bin/env python3
"""Emit immutable split ID hashes for Master Scenario C (T0)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _hash_split_ids(ids: list[str]) -> str:
    canonical = "\n".join(sorted(ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_dataset_csv(csv_path: Path, *, split_col: str, id_col: str = "sample_id") -> dict:
    df = pd.read_csv(csv_path, usecols=[id_col, split_col])
    out: dict[str, dict] = {}
    for split, grp in df.groupby(split_col):
        ids = grp[id_col].astype(str).unique().tolist()
        out[str(split)] = {"n_samples": len(ids), "sha256": _hash_split_ids(ids)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    root = _repo_root()
    out_dir = root / "elara_master_c/data/splits/split_hashes"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("elara_bench_la", "experiments/fusion/real_domain_fusion_inputs.csv", "fusion_split"),
        ("mvtec3d_patchcore", "experiments/fusion/mvtec3d_patchcore_inputs.csv", "split"),
        ("mvtec3d_supervised_paired", "experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv", "split"),
        ("m2_confirmatory_sealed", "experiments/fusion/m2_confirmatory_sealed_inputs.csv", "split"),
        (
            "m2_external_3d_adam_sealed",
            "experiments/fusion/m2_external_3d_adam_sealed_inputs.csv",
            "split",
        ),
        (
            "m2_external_mulsen_sealed",
            "experiments/fusion/m2_external_mulsen_sealed_inputs.csv",
            "split",
        ),
        (
            "m3_healthcare_gap1",
            "experiments/fusion/healthcare_gap1_patient_stratified_fusion_inputs.csv",
            "fusion_split",
        ),
    ]
    manifest: dict = {"version": 1, "datasets": {}}
    for name, rel, split_col in datasets:
        path = root / rel
        if not path.is_file():
            continue
        id_col = "incident_id" if "healthcare" in name else "sample_id"
        payload = {
            "csv": rel,
            "split_column": split_col,
            "id_column": id_col,
            "splits": hash_dataset_csv(path, split_col=split_col, id_col=id_col),
        }
        manifest["datasets"][name] = payload
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest['datasets'])} split hash files -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
