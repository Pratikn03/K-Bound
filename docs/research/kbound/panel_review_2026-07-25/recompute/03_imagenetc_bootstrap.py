#!/usr/bin/env python3
"""Fix-queue item 3: ImageNet-C paired bootstrap at the unit of analysis the
manuscript claims.

`docs/research/kbound/scripts/g8_exactrank_ci.py:18` does

    idx = rng.integers(0, n, (5000, n))      # n = 135 cell-seed rows, i.i.d.

while `kbound_short.tex:797-802` says the pooling is over "seed-averaged
conditions, the same paired design as the CIFAR-10-C rows", i.e. the design in
`_locked_analysis_script.py:54`  (pooled = mean over seeds per condition, then
bootstrap the 27 conditions).

This script reports, for the promoted EXACT-RANK radius and for the corrected
leave-one-out-of-pool radius:
  (a) i.i.d. over 135 cell-seed rows      (what the code did)
  (b) seed-averaged over 27 conditions    (what the text describes)  <== correct unit
  (c) cluster = condition, cells kept     (cluster-robust, 27 clusters of 5)
  (d) cluster = seed                      (5 clusters)
  (e) cluster = corruption family         (9 families of 3 conditions)
20000 replicates, fixed seed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (decide, paired_boot, radii_in_pool, radii_loo, records)

ROOT = "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed"
SEEDS = [0, 1, 2, 3, 4]
NBOOT = 20000
BSEED = 20260720  # same stream seed as g8_exactrank_ci.py:3


def build(cand, rule, pool):
    """Return per-(seed, condition) regret vectors, condition order preserved."""
    reg = {"kga": [], "adapt": [], "freeze": []}
    conds = None
    for s in SEEDS:
        r = records(f"{ROOT}/per_condition_imagenetc_{cand}_seed{s}.json")
        B = np.array([x["B"] for x in r], float)
        bh = np.array([x["b_hat"] for x in r], float)
        a0 = np.array([x["a0"] for x in r], float)
        aad = np.array([x["a_adapted"] for x in r], float)
        c = [x["condition"] for x in r]
        if conds is None:
            conds = c
        assert c == conds, "condition order mismatch across seeds"
        eps = radii_in_pool(bh, B, rule) if pool == "in_pool" else radii_loo(bh, B, rule)
        dec = decide(bh, eps)
        orc = np.maximum(a0, aad)
        kga = np.where(dec == "ADAPT", aad, a0)
        reg["kga"].append(orc - kga)
        reg["adapt"].append(orc - aad)
        reg["freeze"].append(orc - a0)
    return conds, {k: np.vstack(v) for k, v in reg.items()}  # each (5, 27)


def family(c):
    return c.split("|")[0]


def run(cand, rule, pool):
    conds, reg = build(cand, rule, pool)
    out = {"candidate": cand, "rule": rule, "pool": pool,
           "n_conditions": len(conds), "n_seeds": len(SEEDS),
           "point": {k: float(np.mean(v)) for k, v in reg.items()}}

    flat = {k: v.reshape(-1) for k, v in reg.items()}          # 135, seed-major
    flat_cond = np.array([c for _s in SEEDS for c in conds])
    flat_seed = np.array([s for s in SEEDS for _c in conds])
    flat_fam = np.array([family(c) for c in flat_cond])
    avg = {k: v.mean(axis=0) for k, v in reg.items()}          # 27, seed-averaged
    avg_fam = np.array([family(c) for c in conds])

    designs = {
        "iid_135_cellseed_rows_AS_CODED": (
            {k: flat[k] for k in flat}, None),
        "seedavg_27_conditions_AS_TEXT_DESCRIBES": (
            {k: avg[k] for k in avg}, None),
        "cluster_by_condition_135_rows": (
            {k: flat[k] for k in flat}, flat_cond),
        "cluster_by_seed_135_rows": (
            {k: flat[k] for k in flat}, flat_seed),
        "cluster_by_corruption_family_135_rows": (
            {k: flat[k] for k in flat}, flat_fam),
        "cluster_by_corruption_family_27_seedavg": (
            {k: avg[k] for k in avg}, avg_fam),
    }
    out["designs"] = {}
    for name, (d, cl) in designs.items():
        ga = paired_boot(d["kga"] - d["adapt"], NBOOT, BSEED, cl)
        gf = paired_boot(d["kga"] - d["freeze"], NBOOT, BSEED, cl)
        out["designs"][name] = {
            "n_units": (len(np.unique(cl)) if cl is not None else len(d["kga"])),
            "adapt_gap": ga, "freeze_gap": gf,
            "beats_both_ci": bool(ga["hi"] < 0 and gf["hi"] < 0),
            "beats_freeze_ci": bool(gf["hi"] < 0),
            "beats_adapt_ci": bool(ga["hi"] < 0),
        }
    # per-family breakdown on the seed-averaged vector
    out["per_family_seedavg"] = {}
    for f in sorted(set(avg_fam)):
        m = avg_fam == f
        out["per_family_seedavg"][f] = {
            "n_conditions": int(m.sum()),
            "kga": float(avg["kga"][m].mean()),
            "adapt": float(avg["adapt"][m].mean()),
            "freeze": float(avg["freeze"][m].mean()),
            "gap_vs_adapt": float((avg["kga"][m] - avg["adapt"][m]).mean()),
            "gap_vs_freeze": float((avg["kga"][m] - avg["freeze"][m]).mean()),
        }
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    res = {}
    for cand in ("sar",):
        for rule, pool in (("exact", "in_pool"), ("exact", "loo"),
                           ("interp", "in_pool")):
            key = f"{cand}_{rule}_{pool}"
            res[key] = run(cand, rule, pool)
    json.dump(res, open(os.path.join(here, "out_imagenetc_boot.json"), "w"), indent=1)

    for key, o in res.items():
        print("=" * 104)
        print(f"{key}   point KGA/adapt/freeze = "
              f"{o['point']['kga']:.6f} / {o['point']['adapt']:.6f} / {o['point']['freeze']:.6f}")
        print(f"{'design':46s} {'units':>5s}  {'KGA - always-adapt':>26s}  {'KGA - always-freeze':>26s}  BB")
        for name, d in o["designs"].items():
            ga, gf = d["adapt_gap"], d["freeze_gap"]
            print(f"{name:46s} {d['n_units']:5d}  "
                  f"[{ga['lo']:+.4f}, {ga['hi']:+.4f}]{'*' if ga['hi']<0 else ' '}"
                  f"        [{gf['lo']:+.4f}, {gf['hi']:+.4f}]{'*' if gf['hi']<0 else ' '}"
                  f"      {d['beats_both_ci']}")
        print("  per-corruption-family (seed-averaged) gaps:")
        for f, v in o["per_family_seedavg"].items():
            print(f"    {f:20s} n={v['n_conditions']}  KGA {v['kga']:.5f}  adapt {v['adapt']:.5f}"
                  f"  freeze {v['freeze']:.5f}   gap_adapt {v['gap_vs_adapt']:+.5f}"
                  f"  gap_freeze {v['gap_vs_freeze']:+.5f}")
    print("\nwrote out_imagenetc_boot.json")
