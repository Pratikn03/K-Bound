#!/usr/bin/env python3
"""Fit validation-only isotonic calibrators per domain (T2) and freeze artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def fit_calibrators(
    csv_path: Path,
    *,
    split_col: str,
    val_values: tuple[str, ...] = ("validation",),
) -> dict:
    df = pd.read_csv(csv_path)
    val = df[df[split_col].isin(val_values)]
    calibrators: dict[str, dict] = {}
    for domain, grp in val.groupby("domain"):
        y = grp["label"].astype(int).values
        s = grp["score"].astype(float).values
        if len(np.unique(y)) < 2 or len(s) < 20:
            calibrators[str(domain)] = {"status": "skipped", "reason": "insufficient validation data"}
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s, y)
        calibrators[str(domain)] = {
            "status": "fitted",
            "n_val": int(len(s)),
            "model": "isotonic",
        }
        calibrators[str(domain)]["_pickle"] = pickle.dumps(iso).hex()[:64]  # fingerprint only
    return calibrators


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    root = _repo_root()
    out_dir = root / "elara_master_c/models/calibrators"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("elara_bench_la", "experiments/fusion/real_domain_fusion_inputs.csv", "fusion_split"),
        ("mvtec3d_patchcore", "experiments/fusion/mvtec3d_patchcore_inputs.csv", "split"),
    ]
    manifest: dict = {"version": 1, "frozen_utc": None, "datasets": {}}
    for name, rel, split_col in specs:
        path = root / rel
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        val = df[df[split_col].isin(("validation",))]
        models: dict[str, IsotonicRegression] = {}
        meta: dict[str, dict] = {}
        for domain, grp in val.groupby("domain"):
            y = grp["label"].astype(int).values
            s = grp["score"].astype(float).values
            if len(np.unique(y)) < 2 or len(s) < 20:
                meta[str(domain)] = {"status": "skipped"}
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(s, y)
            models[str(domain)] = iso
            meta[str(domain)] = {"status": "fitted", "n_val": int(len(s))}
        if models:
            pkl = out_dir / f"{name}_isotonic.pkl"
            with pkl.open("wb") as f:
                pickle.dump(models, f)
            manifest["datasets"][name] = {"path": str(pkl.relative_to(root)), "domains": meta}

    from datetime import datetime, timezone

    manifest["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    lock = root / "elara_master_c/models/calibrators/calibrator_lock_v1.json"
    lock.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Calibrator lock: {lock} ({len(manifest['datasets'])} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
