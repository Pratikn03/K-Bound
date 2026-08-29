#!/usr/bin/env python3
"""Fail closed unless model seeds resolve to distinct, hash-locked checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(template: str, seeds: list[int]) -> dict:
    if len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be unique")
    rows = []
    for seed in seeds:
        path = Path(template.format(seed=seed)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing model-seed {seed} checkpoint: {path}")
        rows.append(
            {
                "model_seed": seed,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    hashes = {row["sha256"] for row in rows}
    if len(hashes) != len(rows):
        duplicates = [row for row in rows if sum(r["sha256"] == row["sha256"] for r in rows) > 1]
        raise ValueError(f"model seeds do not resolve to distinct checkpoint bytes: {duplicates}")
    return {
        "schema": "kbound_independent_checkpoint_audit_v1",
        "status": "PASS",
        "n_model_seeds": len(rows),
        "all_hashes_distinct": True,
        "checkpoints": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-template", required=True, help="path containing {seed}")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint_template, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint audit: PASS ({len(args.seeds)} distinct model seeds) -> {args.out}")


if __name__ == "__main__":
    main()
