#!/usr/bin/env python3
"""Compare a smoke / partial run manifest against locked headline artifacts.

Usage:
  python smoke_pipeline_report.py --smoke-root experiments/kbound/results/smoke_ms_20260701_120000
  python smoke_pipeline_report.py --manifest path/to/final_manifest_*.json

Prints: per-dataset coverage, delta vs locked regrets, blockers for full run.
Exit 1 if any EXPECTED dataset is missing from the smoke manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
KB = ROOT / "docs/research/kbound"

LOCKED = {
    "cifar10c": {
        "path": ROOT / "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json",
        "metric": lambda d: d["candidates"]["tent"]["kga_mean_regret"],
        "label": "CIFAR Tent regret KGA (locked Holm WIN)",
    },
    "imagenetc": {
        "path": ROOT / "docs/research/kbound/results_source.json",
        "metric": lambda d: d["corruption_grids"]["imagenetc_sar"]["regret_kga"],
        "label": "ImageNet-C grid regret KGA (noise grid; SAR faithful is separate)",
    },
    "officehome": {
        "path": ROOT / "docs/research/kbound/results_source.json",
        "metric": lambda d: d["natural_shifts"]["officehome_M_v2"]["regret_kga"],
        "label": "Office-Home M v2 regret KGA (CI no-harm headline)",
    },
    "iwildcam": {
        "path": ROOT / "docs/research/kbound/results_source.json",
        "metric": lambda d: d["natural_shifts"]["iwildcam_H_v2"]["regret_kga"],
        "label": "iWildCam H v2 regret KGA",
    },
}

EXPECTED = ["cifar10c", "imagenetc", "cifar101", "camelyon", "rxrx1",
            "imagenetr", "pacs", "iwildcam", "officehome"]


def _mean_regret(s: str) -> float:
    return float(str(s).split("+/-")[0])


def load_manifest(smoke_root: Path | None, manifest: Path | None) -> dict:
    if manifest:
        return json.loads(manifest.read_text())
    if smoke_root:
        cands = sorted(smoke_root.glob("final_manifest_*.json"))
        if not cands:
            die(f"no final_manifest_*.json under {smoke_root}")
        return json.loads(cands[-1].read_text())
    die("pass --smoke-root or --manifest")


def die(msg: str) -> None:
    print(f"[smoke_pipeline_report] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-root", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--seeds-expected", type=int, default=2,
                    help="minimum seed coverage to flag CIFAR multiseed as OK")
    args = ap.parse_args()

    man = load_manifest(args.smoke_root, args.manifest)
    rows = {r["dataset"]: r for r in man.get("rows", [])}
    files = man.get("files_scanned", [])

    # count CIFAR seeds in manifest file paths
    cifar_seeds = {p for p in files if "stress_grid_multiseed" in p and "seed" in p
                   for part in Path(p).parts if part.startswith("seed") and part[4:].isdigit()}
    cifar_seed_n = len({p.split("seed")[1].split("/")[0] for p in files
                        if ("stress_grid_multiseed" in p or "cifar10c_stress" in p) and "/seed" in p})

    print("=" * 72)
    print("SMOKE PIPELINE REPORT")
    print("=" * 72)
    print(f"manifest rows: {len(rows)}  files_scanned: {len(files)}")
    print(f"CIFAR seed dirs seen: ~{cifar_seed_n} (want >={args.seeds_expected} for multiseed smoke)")
    print()

    missing = [d for d in EXPECTED if d not in rows]
    print("DATASET COVERAGE")
    print("-" * 72)
    for ds in EXPECTED:
        if ds in rows:
            r = rows[ds]
            print(f"  OK  {ds:12} n={r['n']:3}  KGA={r['regret_kga']}  beats-both(pt)={r['beats_both']}")
        else:
            why = {
                "rxrx1": "RxRx1 data at ~/kbound_rxrx1_data/rxrx1_v1.0",
                "camelyon": "wilds_camelyon17_kga.json or cross-seed KGA (need >=2 seeds)",
            }.get(ds, "check collate patterns / runner output")
            print(f"  MISS {ds:12}  ({why})")
    print()

    print("DELTA VS LOCKED HEADLINES (smoke is indicative only)")
    print("-" * 72)
    for ds, spec in LOCKED.items():
        if ds not in rows:
            continue
        if not spec["path"].exists():
            print(f"  {ds}: locked file missing {spec['path']}")
            continue
        locked = json.loads(spec["path"].read_text())
        try:
            locked_v = float(spec["metric"](locked))
        except (KeyError, TypeError) as e:
            print(f"  {ds}: could not read locked metric: {e}")
            continue
        smoke_v = _mean_regret(rows[ds]["regret_kga"])
        print(f"  {ds}: smoke KGA={smoke_v:.4f}  locked={locked_v:.4f}  "
              f"delta={smoke_v - locked_v:+.4f}  ({spec['label']})")
    print()

    blockers = []
    if missing:
        blockers.append(f"missing datasets: {missing}")
    if cifar_seed_n < args.seeds_expected:
        blockers.append(f"CIFAR multiseed: only {cifar_seed_n} seed(s); set KB_SMOKE_SEEDS='0 1' or more")
    if not (ROOT / "docs/research/kbound/results_source.json").exists():
        blockers.append("results_source.json missing")
    rxrx = Path.home() / "kbound_rxrx1_data/rxrx1_v1.0"
    if not rxrx.is_dir():
        blockers.append(f"RxRx1 data not at {rxrx} (required for full run step 5)")

    print("FULL RUN READINESS")
    print("-" * 72)
    if blockers:
        for b in blockers:
            print(f"  ! {b}")
        print()
        print("Full command (when blockers resolved):")
        print("  caffeinate -is bash docs/research/kbound/scripts/run_final_showcase.sh \\")
        print('    --device mps --seeds "0 1 2 3 4"')
    else:
        print("  All smoke datasets present. Full showcase command is ready.")
    print("=" * 72)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
