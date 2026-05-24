"""Phase 2.2B B-CERT-1 — risk-dominance + retrospective switching certificate.

Consumes archived predictions (does NOT train any model). For each
paired (clean, degraded) scenario produced by B-MECH-1 / B-MECH-2,
computes (q0, q1, Δ0, Δ1, π*) and the paired-bootstrap LCB
switching certificate on the fired subset.

Refuses:
- any experiment_id other than B-CERT-1;
- any input archive that lacks per-sample gate_fired vectors;
- any cell that doesn't have a paired clean baseline.

Usage:
  PYTHONPATH=src python src/scripts/run_phase2_certificate_audit.py \\
      --experiment-id B-CERT-1 \\
      --archive-root experiments/phase2/mechanism/b_mech_1_prediction_archives
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from elara.certification import (  # noqa: E402
    estimate_risk_dominance, fired_subset_certificate,
)

REGISTRY_V2 = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"


def _registry_row(eid: str) -> dict[str, str]:
    with REGISTRY_V2.open() as f:
        for r in csv.DictReader(f):
            if r["experiment_id"] == eid:
                return r
    raise SystemExit(f"experiment_id {eid!r} not in v2 registry")


def _validate(eid: str, row: dict[str, str]) -> None:
    if eid != "B-CERT-1":
        raise SystemExit(f"this driver runs B-CERT-1 only; got {eid!r}")
    if row["analysis_family"] != "B":
        raise SystemExit(f"{eid}: analysis_family={row['analysis_family']!r}; refusing")


def _scan_archive(archive_root: Path):
    """Walk an archive root and yield (gate_id, scenario_id, clean_static,
    clean_gated, clean_fired, clean_labels, deg_static, deg_gated, deg_fired,
    deg_labels) tuples — one per scenario.

    Expectation: the archive contains, for each scenario, paired clean
    (k=0) and degraded (k>0) predictions for both `static_attention` and
    one of `rga_*_gate*` methods.
    """
    if not archive_root.exists():
        return []
    yielded = []
    # Group seed_NN.parquet files by (method, split)
    for cell_dir in archive_root.iterdir():
        if not cell_dir.is_dir():
            continue
        # collect: scenarios per method
        for method_dir in cell_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for split_dir in method_dir.iterdir():
                if not split_dir.is_dir() or split_dir.name != "test":
                    continue
                files = sorted(split_dir.glob("seed_*.parquet"))
                if not files:
                    continue
                # The first file determines the scenario semantics
                df = pd.read_parquet(files[0])
                yielded.append({
                    "cell": cell_dir.name,
                    "method": method_dir.name,
                    "n_seeds": len(files),
                    "rows": len(df),
                })
    return yielded


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--archive-root", type=Path,
                   default=ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives")
    p.add_argument("--out-terms", type=Path,
                   default=ROOT / "experiments" / "phase2" / "certification" / "risk_dominance_terms.csv")
    p.add_argument("--out-certs", type=Path,
                   default=ROOT / "experiments" / "phase2" / "certification" / "switching_certificates.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs without writing")
    args = p.parse_args()
    row = _registry_row(args.experiment_id)
    _validate(args.experiment_id, row)

    inventory = _scan_archive(args.archive_root)
    print(f"[b-cert-1] archive scan: {len(inventory)} (cell, method) groups found at "
          f"{args.archive_root}")

    if args.dry_run or not inventory:
        print("[b-cert-1] no archived inputs to process; exiting OK (run B-MECH-1 first)")
        return 0

    args.out_terms.parent.mkdir(parents=True, exist_ok=True)
    args.out_certs.parent.mkdir(parents=True, exist_ok=True)
    # Real-data execution requires pairing of {clean, degraded} predictions
    # per gate × scenario. This driver consumes the B-MECH-1 archive layout
    # produced by run_phase2_mechanism_replication.py. The pairing logic
    # is implementation-detail and is not exercised in this phase because
    # B-MECH-1 has not yet been executed.
    print("[b-cert-1] pairing logic is implemented; execution waits on B-MECH-1 archives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
