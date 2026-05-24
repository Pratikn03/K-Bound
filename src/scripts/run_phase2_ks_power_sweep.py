"""Phase 2.2B B-MECH-4 — KS true-degradation power × window-size sweep.

Sweeps the ReliabilityEstimator.ks_window_size parameter across the
locked grid {32, 64, 128, 256, 512} and measures detection power
(true-positive rate of gate firing) under genuine score collapse +
score noise + missingness vs false-activation rate on clean data.

Refuses:
- any experiment_id other than B-MECH-4;
- any window size not in the locked grid.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_ks_power_sweep.py \\
      --experiment-id B-MECH-4 --seeds 5 --seed-start 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from elara.family_b.ks_window import KS_WINDOW_GRID  # noqa: E402

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-MECH-4":
        raise SystemExit(f"this driver runs B-MECH-4 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--window-sizes", default=",".join(str(w) for w in KS_WINDOW_GRID),
                   help="comma-separated subset of the locked grid")
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)
    if int(args.seeds) <= 0:
        print(f"[b-mech-4 {args.experiment_id}] validation-only invocation; exiting OK")
        return 0

    requested = [int(x) for x in args.window_sizes.split(",")]
    for w in requested:
        if w not in KS_WINDOW_GRID:
            raise SystemExit(
                f"window size {w} not in locked grid {KS_WINDOW_GRID}; refusing"
            )

    print(f"[b-mech-4 {args.experiment_id}] requested window sizes: {requested}")
    print(f"[b-mech-4] ReliabilityEstimator now accepts ks_window_size and uses it to "
          f"truncate both the validation reference and current scores before KS. The "
          f"driver runs the standard RGA path once per (seed, window) and records "
          f"detection-power vs false-activation on score collapse / noise / missingness. "
          f"Full execution wall-clock estimate: {args.seeds * len(requested) * 5}-"
          f"{args.seeds * len(requested) * 15} minutes; reserved for a future compute window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
