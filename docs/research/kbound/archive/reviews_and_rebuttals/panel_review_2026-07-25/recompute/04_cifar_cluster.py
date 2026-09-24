#!/usr/bin/env python3
"""Fix-queue item 17, part (a): cluster-robust CIFAR-10-C intervals.

Original combined docstring:

(a) Cluster-robust paired bootstrap for CIFAR-10-C at three units:
      432 cells i.i.d. (as `_locked_analysis_script.py:58-66` runs it),
      216 twin-pairs (r0/r1 are the same design point),
      12 corruption x severity clusters,
      6 corruption-family clusters.
    Plus the r0/r1 replicate correlation that motivates it.

(b) Leave-one-corruption-out calibration: refit the shipped estimator
    (GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
    subsample=0.8, random_state=0) -- cifar_tent_mps_v2.py:151-162) under three
    partitions and report residual MAE, R^2, eps, adapt rate, FA_u, KGA regret:
      leave-one-CELL-out        (as shipped)
      leave-one-TWIN-PAIR-out   (drops the r0/r1 duplicate)
      leave-one-CORRUPTION-out  (the honest transport test)

Artifacts: experiments/kbound/results/mixed_headtohead_v1/
             per_condition_cifar10c_tent_primary_kga_seed{0..4}.json    (432 cells each)
           experiments/kbound/results/stress_grid_multiseed_v1/seed{1..4}/
             per_condition_cifar10c_tent_seed{s}.json

Run: python3 04_cifar_cluster_and_loco.py           (LOCO on seed 0 only, ~1 min)
     python3 04_cifar_cluster_and_loco.py --all-seeds
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (ALPHA, decide, eps_exact, eps_interp, paired_boot,
                       radii_in_pool, records, score)

H2H = "experiments/kbound/results/mixed_headtohead_v1"
STRESS = "experiments/kbound/results/stress_grid_multiseed_v1"
NBOOT = 20000
BSEED = 20260611  # same stream seed as _locked_analysis_script.py:13

GBR_KW = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8,
              random_state=0)


def load(path):
    r = records(path)
    return {
        "B": np.array([x["B"] for x in r], float),
        "bh": np.array([x["b_hat"] for x in r], float),
        "a0": np.array([x["a0"] for x in r], float),
        "aad": np.array([x["a_adapted"] for x in r], float),
        "Z": np.array([x["Z"] for x in r], float),
        "cond": [x["condition"] for x in r],
        "eps": np.array([x["eps_conformal"] for x in r], float),
    }


def keys(cond):
    p = cond.split("|")
    return {"corruption": p[0], "severity": p[1], "batch": p[2], "comp": p[3],
            "aggr": p[4], "rep": p[5],
            "design_point": "|".join(p[:5])}   # everything except the r0/r1 repeat


# ------------------------------------------------------------------ (a) clusters
def cluster_analysis(files, label):
    per_seed = {"kga": [], "adapt": [], "freeze": []}
    cond = None
    for f in files:
        d = load(f)
        if cond is None:
            cond = d["cond"]
        assert d["cond"] == cond
        eps = radii_in_pool(d["bh"], d["B"], "interp")
        dec = decide(d["bh"], eps)
        orc = np.maximum(d["a0"], d["aad"])
        kga = np.where(dec == "ADAPT", d["aad"], d["a0"])
        per_seed["kga"].append(orc - kga)
        per_seed["adapt"].append(orc - d["aad"])
        per_seed["freeze"].append(orc - d["a0"])
    avg = {k: np.vstack(v).mean(axis=0) for k, v in per_seed.items()}
    K = [keys(c) for c in cond]
    corr = np.array([k["corruption"] for k in K])
    csev = np.array([k["corruption"] + "|" + k["severity"] for k in K])
    dp = np.array([k["design_point"] for k in K])
    rep = np.array([k["rep"] for k in K])

    # r0/r1 replicate correlation on the gap vectors
    ga = avg["kga"] - avg["adapt"]
    gf = avg["kga"] - avg["freeze"]
    order0 = {d: i for i, d in enumerate(dp[rep == "r0"])}
    idx0 = np.where(rep == "r0")[0]
    idx1 = np.array([np.where((dp == dp[i]) & (rep == "r1"))[0][0] for i in idx0])
    rep_corr = {
        "gap_vs_adapt": float(np.corrcoef(ga[idx0], ga[idx1])[0, 1]),
        "gap_vs_freeze": float(np.corrcoef(gf[idx0], gf[idx1])[0, 1]),
        "kga_regret": float(np.corrcoef(avg["kga"][idx0], avg["kga"][idx1])[0, 1]),
        "n_pairs": len(idx0),
    }

    designs = {
        "cells_iid_432_AS_RUN": (ga, gf, None),
        "twin_pairs_216": (ga, gf, dp),
        "corruption_x_severity_12": (ga, gf, csev),
        "corruption_family_6": (ga, gf, corr),
    }
    out = {"label": label, "files": files, "n_conditions": len(cond),
           "point": {k: float(np.mean(v)) for k, v in avg.items()},
           "replicate_correlation_r0_r1": rep_corr, "designs": {}}
    for name, (a, f_, cl) in designs.items():
        out["designs"][name] = {
            "n_units": (len(np.unique(cl)) if cl is not None else len(a)),
            "adapt_gap": paired_boot(a, NBOOT, BSEED, cl),
            "freeze_gap": paired_boot(f_, NBOOT, BSEED, cl),
        }
        d = out["designs"][name]
        d["beats_both_ci"] = bool(d["adapt_gap"]["hi"] < 0 and d["freeze_gap"]["hi"] < 0)
    # per-corruption breakdown
    out["per_corruption"] = {}
    for c in sorted(set(corr)):
        m = corr == c
        out["per_corruption"][c] = {
            "n": int(m.sum()),
            "kga": float(avg["kga"][m].mean()),
            "adapt": float(avg["adapt"][m].mean()),
            "freeze": float(avg["freeze"][m].mean()),
            "gap_vs_adapt": float(ga[m].mean()),
            "gap_vs_freeze": float(gf[m].mean()),
        }
    # CI width ratios vs the as-run design
    base_a = out["designs"]["cells_iid_432_AS_RUN"]["adapt_gap"]
    base_f = out["designs"]["cells_iid_432_AS_RUN"]["freeze_gap"]
    wa = base_a["hi"] - base_a["lo"]
    wf = base_f["hi"] - base_f["lo"]
    for name, d in out["designs"].items():
        d["width_ratio_adapt"] = (d["adapt_gap"]["hi"] - d["adapt_gap"]["lo"]) / wa
        d["width_ratio_freeze"] = (d["freeze_gap"]["hi"] - d["freeze_gap"]["lo"]) / wf
    return out



def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = {}
    for lbl, files in {
        "cifar10c_tent_headtohead_seed0-4": [
            f"{H2H}/per_condition_cifar10c_tent_primary_kga_seed{s}.json" for s in range(5)],
        "cifar10c_eata_headtohead_seed0-4": [
            f"{H2H}/per_condition_cifar10c_eata_secondary_kga_seed{s}.json" for s in range(5)],
        "cifar10c_tent_stressgrid_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_tent_seed{s}.json" for s in (1, 2, 3, 4)],
        "cifar10c_eata_stressgrid_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_eata_seed{s}.json" for s in (1, 2, 3, 4)],
        "cifar10c_sar_stressgrid_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_sar_seed{s}.json" for s in (1, 2, 3, 4)],
    }.items():
        res[lbl] = cluster_analysis(files, lbl)
        o = res[lbl]
        print("=" * 100)
        print(f"{lbl}   point KGA/adapt/freeze = {o['point']['kga']:.8f} /"
              f" {o['point']['adapt']:.8f} / {o['point']['freeze']:.8f}")
        print(f"  r0/r1 replicate correlation: adapt-gap {o['replicate_correlation_r0_r1']['gap_vs_adapt']:.4f}"
              f"  freeze-gap {o['replicate_correlation_r0_r1']['gap_vs_freeze']:.4f}"
              f"  kga-regret {o['replicate_correlation_r0_r1']['kga_regret']:.4f}"
              f"  (n_pairs={o['replicate_correlation_r0_r1']['n_pairs']})")
        for name, d in o["designs"].items():
            print(f"  {name:30s} {d['n_units']:5d}  [{d['adapt_gap']['lo']:+.5f},{d['adapt_gap']['hi']:+.5f}]"
                  f" {d['width_ratio_adapt']:4.2f}x  [{d['freeze_gap']['lo']:+.5f},{d['freeze_gap']['hi']:+.5f}]"
                  f" {d['width_ratio_freeze']:4.2f}x  BB={d['beats_both_ci']}")
        print("  per-corruption gaps:")
        for c, v in o["per_corruption"].items():
            print(f"    {c:20s} n={v['n']:3d}  gap_adapt {v['gap_vs_adapt']:+.5f}"
                  f"  gap_freeze {v['gap_vs_freeze']:+.5f}"
                  f"   {'<-- KGA WORSE than always-adapt' if v['gap_vs_adapt'] > 0 else ''}")
    json.dump(res, open(os.path.join(here, "out_cifar_cluster.json"), "w"), indent=1, default=float)
    print("\nwrote out_cifar_cluster.json")


if __name__ == "__main__":
    main()
