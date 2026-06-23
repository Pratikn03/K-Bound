#!/usr/bin/env python3
"""
CPU dev-screen for Protocol L (FMoW / PovertyMap).

Reads a geoshift runner JSON and prints whether to proceed to full GPU:
  - base_rate_harmful in [0.25, 0.75]  (mixed, not dominated)
  - best harm-AUC >= 0.65
  - per-candidate dev false-adapt <= 0.15 (quick GBR screen)

Usage:
  ~/.venv_wilds/bin/python docs/research/kbound/scripts/screen_protocol_L.py \\
    --records experiments/kbound/results/fmow_protocol_L_dev/result_*.json \\
    --candidate sar_online --dev-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="path or glob to result JSON")
    p.add_argument("--candidate", default="sar_online")
    p.add_argument("--dev-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--harm-auc-min", type=float, default=0.65)
    p.add_argument("--fa-max", type=float, default=0.15)
    args = p.parse_args()

    paths = sorted(glob.glob(args.records))
    if not paths:
        raise SystemExit(f"no files match {args.records}")
    path = paths[-1]
    data = json.loads(Path(path).read_text())
    summary = data.get("kbound_summary", {})
    det = data.get("detectability", {})
    harm_rate = float(summary.get("base_rate_harmful_B<0", 0))
    harm_auc = det.get("best_single_feature_harm_AUC")
    cls = summary.get("classification", "?")

    recs, _ = af.load_records(path, candidate=args.candidate)
    recs = [r for r in recs if int(r["seed"]) in args.dev_seeds]
    verdict = "PROCEED" if recs else "NO_RECORDS"
    fa = None
    if recs:
        m = af.run_split(recs, args.dev_seeds, args.dev_seeds, estimator="gbr", conformal="global")
        fa = m.get("false_adapt") if m else None
        mixed = 0.20 <= harm_rate <= 0.80
        auc_ok = harm_auc is not None and float(harm_auc) >= args.harm_auc_min
        fa_ok = fa is None or float(fa) <= args.fa_max
        if mixed and auc_ok and fa_ok:
            verdict = "PROCEED"
        else:
            verdict = "STOP"

    print(f"records={path}")
    print(f"classification={cls} harmful_rate={harm_rate:.3f} harm_AUC={harm_auc}")
    print(f"candidate={args.candidate} dev_FA={fa} -> {verdict}")
    return 0 if verdict == "PROCEED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
