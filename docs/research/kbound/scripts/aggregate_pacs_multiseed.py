#!/usr/bin/env python3
"""Validate and aggregate locked PACS seed summaries without inventing observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


EXPECTED_DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
METRICS = ("FA_u", "FA_c", "coverage", "adapt_rate", "base_rate_harmful")
ROOT = Path(__file__).resolve().parents[4]


def portable_path(path: Path) -> str:
    """Record repository-relative provenance when the input is inside the checkout."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load(path: Path, seed: int) -> dict:
    data = json.loads(path.read_text())
    declared = data.get("seed", seed if seed == 0 else None)
    if declared != seed:
        raise ValueError(f"{path}: seed is {declared!r}, expected {seed}")
    if data.get("dataset") != "PACS":
        raise ValueError(f"{path}: expected dataset PACS")
    if set(data.get("per_domain", {})) != set(EXPECTED_DOMAINS):
        raise ValueError(f"{path}: incomplete PACS domains")
    return data


def signature(data: dict) -> dict:
    override = data.get("win_hunt_v5_override", {})
    return {"alpha": data.get("alpha"), "adapt_lr": override.get("adapt_lr"),
            "batch_regimes": override.get("batch_regimes"),
            "aggressiveness": override.get("aggressiveness")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=Path, required=True)
    ap.add_argument("--seed1", type=Path, required=True)
    ap.add_argument("--seed2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = [args.seed0, args.seed1, args.seed2]
    runs = [load(path, seed) for seed, path in enumerate(paths)]
    sigs = [signature(run) for run in runs]
    if any(sig != sigs[0] for sig in sigs[1:]):
        raise ValueError(f"PACS protocol mismatch across seeds: {sigs}")

    domains = {}
    for domain in EXPECTED_DOMAINS:
        rows = [run["per_domain"][domain] for run in runs]
        domain_result = {"n_test_cells_per_seed": [r["n_test_cells"] for r in rows]}
        for metric in METRICS:
            vals = [float(r[metric]) for r in rows]
            domain_result[metric] = {"per_seed": vals, "mean": mean(vals),
                                     "sd": stdev(vals) if len(vals) > 1 else 0.0}
        for policy in ("K_Bound", "always_adapt", "always_freeze"):
            vals = [float(r["regret"][policy]) for r in rows]
            domain_result[f"regret_{policy}"] = {
                "per_seed": vals, "mean": mean(vals),
                "sd": stdev(vals) if len(vals) > 1 else 0.0}
        domain_result["verdict_per_seed"] = [r["verdict"] for r in rows]
        domains[domain] = domain_result

    result = {"schema": "kbound_pacs_multiseed_v1", "dataset": "PACS",
              "seeds": [0, 1, 2], "n_seeds": 3, "protocol_signature": sigs[0],
              "source_files": [portable_path(p) for p in paths], "per_domain": domains,
              "scope": "descriptive seed stability; no accuracy-dominance claim implied"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {args.out} (PACS seeds 0,1,2; protocol validated)")


if __name__ == "__main__":
    main()
