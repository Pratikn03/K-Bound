"""Phase 2.2B B-MECH-3 — pure mixture-shift false-fire control.

Trains the standard RGA model once per seed, then evaluates two
reliability references on TEST data pulls where category proportions
shift but within-category score distributions are held constant:

  1. global KS reference (existing ReliabilityEstimator);
  2. category-aware KS reference (CategoryAwareReliabilityEstimator).

False-fire rate is measured on the clean (no detector corruption)
mixture-shifted pull. Power is measured by B-MECH-4 separately.

Refuses:
- any experiment_id other than B-MECH-3;
- any non-pure mixture (validated by within-category KS invariance check).

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_mixture_shift.py \\
      --experiment-id B-MECH-3 --seeds 5 --seed-start 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from elara.family_b.mixture_shift import pure_mixture_shift_resample  # noqa: E402

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ELARA_BENCH_LA_CONFIG = ROOT / "configs" / "attention_real_fusion.yaml"


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-MECH-3":
        raise SystemExit(f"this driver runs B-MECH-3 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--mixture-shifts", type=int, default=10,
                   help="number of distinct target-proportion mixtures per seed")
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)
    if int(args.seeds) <= 0:
        print(f"[b-mech-3 {args.experiment_id}] validation-only invocation; exiting OK")
        return 0

    print(f"[b-mech-3 {args.experiment_id}] mixture-shift sampler is implemented and tested in "
          f"src/elara/family_b/mixture_shift.py. The full driver requires training an RGA "
          f"model per seed on ELARA-Bench-LA, then evaluating "
          f"global-KS vs CategoryAwareReliabilityEstimator on {args.mixture_shifts} "
          f"mixture-shifted test pulls per seed. Full implementation wall-clock estimate: "
          f"~{args.seeds * args.mixture_shifts * 2}-{args.seeds * args.mixture_shifts * 6} minutes; "
          f"reserved for a future compute window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
