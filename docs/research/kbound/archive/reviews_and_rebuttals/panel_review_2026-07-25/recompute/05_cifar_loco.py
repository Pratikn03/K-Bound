#!/usr/bin/env python3
"""Fix-queue item 17, part (b): leave-one-corruption-out calibration on CIFAR-10-C.

Refits the shipped benefit estimator
  GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                            subsample=0.8, random_state=0)
  -- docs/research/kbound/scripts/cifar_tent_mps_v2.py:151-162
under three calibration partitions and reports residual MAE, R^2, the realised
conformal radius, adapt rate, FA_u and the KGA regret triple.

Run: python3 05_cifar_loco.py [--all-seeds]     (~2 min/seed)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import ALPHA, decide, eps_exact, eps_interp, records, score

H2H = "experiments/kbound/results/mixed_headtohead_v1"
STRESS = "experiments/kbound/results/stress_grid_multiseed_v1"
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
    return {"corruption": p[0], "design_point": "|".join(p[:5])}


def refit(Z, B, groups):
    """Out-of-fold prediction with the shipped GBR, folds = unique(groups)."""
    from sklearn.ensemble import GradientBoostingRegressor
    Bhat = np.zeros(len(B))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        m = GradientBoostingRegressor(**GBR_KW)
        m.fit(Z[tr], B[tr])
        Bhat[te] = m.predict(Z[te])
    return Bhat


def loco(path, label):
    d = load(path)
    K = [keys(c) for c in d["cond"]]
    cellid = np.arange(len(d["B"]))
    dp = np.array([k["design_point"] for k in K])
    corr = np.array([k["corruption"] for k in K])
    out = {"label": label, "artifact": path, "n": len(d["B"]),
           "stored_eps": float(d["eps"][0])}
    schemes = {
        "leave_one_cell_out_AS_SHIPPED": cellid,
        "leave_one_twin_pair_out": dp,
        "leave_one_corruption_out": corr,
    }
    for name, groups in schemes.items():
        Bhat = refit(d["Z"], d["B"], groups)
        resid = np.abs(Bhat - d["B"])
        ss_res = float(np.sum((Bhat - d["B"]) ** 2))
        ss_tot = float(np.sum((d["B"] - d["B"].mean()) ** 2))
        row = {"n_folds": int(len(np.unique(groups))),
               "residual_MAE": float(resid.mean()),
               "R2": 1 - ss_res / ss_tot,
               "corr_with_stored_bhat": float(np.corrcoef(Bhat, d["bh"])[0, 1])}
        for rule, fn in (("interp", eps_interp), ("exact", eps_exact)):
            e = fn(resid)
            dec = decide(Bhat, np.full(len(Bhat), e))
            sc = score(dec, d["B"], d["a0"], d["aad"])
            row[rule] = {"eps": e, "adapt_rate": sc["n_adapt"] / sc["n"],
                         "adapt": sc["n_adapt"], "freeze": sc["n_freeze"],
                         "abstain": sc["n_abstain"],
                         "fa_u": sc["fa_u"], "fa_num": sc["fa_num"],
                         "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]]}
        out[name] = row
    return out



def main():
    here = os.path.dirname(os.path.abspath(__file__))
    all_seeds = "--all-seeds" in sys.argv
    res = {}
    seeds = range(5) if all_seeds else [0]
    for s in seeds:
        lbl = f"loco_cifar10c_tent_headtohead_seed{s}"
        res[lbl] = loco(f"{H2H}/per_condition_cifar10c_tent_primary_kga_seed{s}.json", lbl)
        o = res[lbl]
        print("=" * 104)
        print(f"{lbl}  (eps stored in file = {o['stored_eps']:.6f})")
        print(f"  {'calibration scheme':40s} {'folds':>5s} {'MAE':>9s} {'R2':>7s}"
              f" {'eps':>8s} {'adapt%':>7s} {'FA_u':>7s} {'KGA regret':>11s}")
        for name in ("leave_one_cell_out_AS_SHIPPED", "leave_one_twin_pair_out",
                     "leave_one_corruption_out"):
            r = o[name]
            for rule in ("interp", "exact"):
                v = r[rule]
                print(f"  {name+' ['+rule+']':40s} {r['n_folds']:5d} {r['residual_MAE']:9.5f}"
                      f" {r['R2']:7.4f} {v['eps']:8.5f} {100*v['adapt_rate']:6.1f}%"
                      f" {v['fa_u']:7.4f} {v['regret'][0]:11.6f}")
        print(f"  corr(refit b_hat, stored b_hat) LOO-cell = "
              f"{o['leave_one_cell_out_AS_SHIPPED']['corr_with_stored_bhat']:.6f}")
    json.dump(res, open(os.path.join(here, "out_cifar_loco.json"), "w"), indent=1, default=float)
    print("\nwrote out_cifar_loco.json")


if __name__ == "__main__":
    main()
