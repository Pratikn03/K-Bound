#!/usr/bin/env python3
"""Recompute the KGA controller-only microbenchmark with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_11(p0: np.ndarray, pa: np.ndarray) -> np.ndarray:
    e0 = -(p0 * np.log(p0 + 1e-9)).sum(1)
    ea = -(pa * np.log(pa + 1e-9)).sum(1)
    c0 = p0.max(1)
    ca = pa.max(1)
    classes = p0.shape[1]
    mb0 = p0.mean(0)
    mba = pa.mean(0)
    pb0 = -(mb0 * np.log(mb0 + 1e-9)).sum() / np.log(classes)
    pba = -(mba * np.log(mba + 1e-9)).sum() / np.log(classes)
    return np.array(
        [
            e0.mean(),
            c0.mean(),
            pb0,
            ea.mean(),
            ca.mean(),
            pba,
            pb0 - pba,
            e0.mean() - ea.mean(),
            float((ca > 0.9).mean()),
            float((mba * np.log((mba + 1e-9) / (mb0 + 1e-9))).sum()),
            0.037,
        ]
    )


def timed_ms(function, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        function()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        function()
    return (time.perf_counter_ns() - start) / repeats / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/kbound/results/cifar10c_per_condition_seed0/per_condition_cifar10c_tent_seed0.json"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    records = payload["records"]
    features = np.asarray([row["Z"] for row in records], dtype=float)
    benefits = np.asarray([row["B"] for row in records], dtype=float)
    model = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    ).fit(features, benefits)

    rng = np.random.default_rng(0)
    p0 = rng.dirichlet(np.ones(10), 200)
    pa = rng.dirichlet(np.ones(10), 200)
    evidence_ms = timed_ms(lambda: evidence_11(p0, pa), args.repeats, args.warmup)
    prediction_ms = timed_ms(lambda: model.predict(features[:1]), args.repeats, args.warmup)

    checkpoint = None
    if args.checkpoint:
        checkpoint = {
            "path_basename": args.checkpoint.name,
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        }

    result = {
        "schema_version": 1,
        "scope": "controller-only microbenchmark; excludes adaptation and model inference",
        "input": {
            "path": args.input.as_posix(),
            "sha256": sha256(args.input),
            "records": len(records),
        },
        "config": {
            "batch_size": 200,
            "classes": 10,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "latency_ms": {
            "evidence_assembly": round(evidence_ms, 6),
            "benefit_model_prediction": round(prediction_ms, 6),
            "controller_total": round(evidence_ms + prediction_ms, 6),
        },
        "benefit_model_bytes": len(pickle.dumps(model)),
        "rollback_checkpoint": checkpoint,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "Timing is environment-specific. The optional checkpoint records rollback-copy "
            "storage, not live peak memory."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
