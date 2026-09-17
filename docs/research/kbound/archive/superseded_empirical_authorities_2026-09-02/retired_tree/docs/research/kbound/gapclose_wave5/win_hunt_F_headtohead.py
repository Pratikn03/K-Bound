#!/usr/bin/env python3
"""WIN_HUNT_v3 Arm F — verify + report ALL mixed head-to-head configurations.

Independent recompute (per-condition paired bootstrap, 10^4) of KGA vs POEM and
AETTA for every configuration present in mixed_headtohead_v1. tent_primary was
already verified digit-for-digit; this scores the remaining arms identically.

Run (CPU, ~1 min):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_F_headtohead.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HH = ROOT / "experiments/kbound/results/mixed_headtohead_v1"
POLICIES = ["always_adapt", "always_freeze", "aetta", "poem", "kga", "oracle"]
NBOOT = 10000


def main() -> int:
    pat = str(HH / "per_condition_cifar10c_*_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    if not files:
        print(f"SCHEMA ERROR: no files match {pat}", file=sys.stderr)
        return 3
    rx = re.compile(r"per_condition_cifar10c_(.+)_(" + "|".join(POLICIES)
                    + r")_seed(\d+)\.json$")
    data: dict = defaultdict(lambda: defaultdict(dict))
    for f in files:
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        cfg, pol, seed = m.group(1), m.group(2), int(m.group(3))
        d = json.load(open(f))
        for r in d["records"]:
            orc = max(float(r["a0"]), float(r["a_adapted"]))
            acc = (float(r["a_adapted"]) if r["policy_decision"] == "ADAPT"
                   else float(r["a0"]))
            data[cfg][pol].setdefault(r["condition"], []).append(
                dict(reg=orc - acc,
                     fa=int(r["policy_decision"] == "ADAPT" and float(r["B"]) <= 0)))
    if not data:
        print("SCHEMA ERROR: filename pattern matched nothing; files: "
              + ", ".join(os.path.basename(f) for f in files[:5]), file=sys.stderr)
        return 3

    rng = np.random.default_rng(0)
    out = {"protocol": "WIN_HUNT_v3_ARM_F",
           "registered": "research_lock/WIN_HUNT_v3_PROTOCOL.yaml", "configs": {}}
    ci = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]  # noqa: E731
    for cfg, pols in sorted(data.items()):
        if "kga" not in pols:
            continue
        percond = {p: np.array([np.mean([x["reg"] for x in v])
                                for k, v in sorted(pols[p].items())])
                   for p in pols}
        fa = {p: (sum(x["fa"] for v in pols[p].values() for x in v),
                  sum(len(v) for v in pols[p].values())) for p in pols}
        entry = {"n_conditions": int(len(percond["kga"])),
                 "mean_regret": {p: float(percond[p].mean()) for p in percond},
                 "FA_rate": {p: round(fa[p][0] / max(fa[p][1], 1), 4) for p in fa}}
        rep = True
        for opp in ("poem", "aetta"):
            if opp not in percond:
                continue
            dd = percond["kga"] - percond[opp]
            idx = rng.integers(0, len(dd), size=(NBOOT, len(dd)))
            bs = dd[idx].mean(1)
            c = ci(bs)
            entry[f"kga_minus_{opp}"] = dict(mean=float(dd.mean()), ci95=c,
                                             ci_below_zero=bool(c[1] < 0))
            rep &= c[1] < 0
        entry["REPLICATED_WIN"] = bool(rep)
        out["configs"][cfg] = entry
    print(json.dumps(out, indent=1))
    p = ROOT / "research_lock/WIN_HUNT_v3_ARM_F_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
