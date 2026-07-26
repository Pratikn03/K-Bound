#!/usr/bin/env python3
"""Fix-queue item 18 (F3-10, F3-11): baseline parity in `tab:gates`, and what the
conformal radius actually buys.

`docs/research/kbound/scripts/gate_baseline_comparison.py` cannot be run as
released -- its input `cifar10c_percell.json` is not in the tree (F4-7). But the
same six rules can be scored from the committed per-condition dumps, which carry
exactly the fields the gate script wants (Z, a0, a_adapted, condition).

This script regenerates `tab:gates` on real data for every available CIFAR-10-C
seed, reporting for each rule:
    regret, FA_u, FA_c, adapt rate, coverage, and FA_u restricted to harmful cells
and, for the two fitted rules, whether the gate is leave-one-TASK-out (corruption)
or leave-one-CELL-out -- the docstring claims all gates are leave-one-task-out
"exactly like KGA"; gates 1-2 are unfitted sign rules and KGA is leave-one-cell-out.

Run: python3 08_gates_and_radius_value.py            (seed 0 only, ~4 min)
     python3 08_gates_and_radius_value.py --all-seeds
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import ALPHA, clopper_pearson_upper, eps_exact, eps_interp, records

H2H = "experiments/kbound/results/mixed_headtohead_v1"

# evidence layout, cifar_tent_mps_v2.EVIDENCE_NAMES / gate_baseline_comparison.py:36
(PRE_ENT, PRE_CONF, PRE_PBAL, POST_ENT, POST_CONF, POST_PBAL,
 PBAL_DROP, ENT_DROP, FRAC_HI, MKL, UPD) = range(11)
GBR_KW = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8,
              random_state=0)


def load(path):
    r = records(path)
    return {
        "Z": np.array([x["Z"] for x in r], float),
        "a0": np.array([x["a0"] for x in r], float),
        "aa": np.array([x["a_adapted"] for x in r], float),
        "B": np.array([x["B"] for x in r], float),
        "task": np.array([x["condition"].split("|")[0] for x in r]),
    }


def _drift_tau(mkl_dev, B_dev, alpha):
    best = -np.inf
    for tau in np.sort(np.unique(mkl_dev)):
        adapt = mkl_dev <= tau
        if adapt.sum() == 0:
            continue
        if np.mean(B_dev[adapt] < 0) <= alpha:
            best = tau
    return best


def gates(d, alpha=ALPHA):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    Z, a0, aa, B, task = d["Z"], d["a0"], d["aa"], d["B"], d["task"]
    n = len(B)
    out = {}

    out["confidence gate"] = (np.where(Z[:, POST_CONF] > Z[:, PRE_CONF], "ADAPT", "FREEZE"),
                              "unfitted sign rule (no calibration at all)")
    out["entropy gate"] = (np.where(Z[:, ENT_DROP] > 0, "ADAPT", "FREEZE"),
                           "unfitted sign rule (no calibration at all)")

    dec = np.empty(n, dtype=object)
    for t in np.unique(task):
        te = task == t; dv = ~te
        tau = _drift_tau(Z[dv, MKL], B[dv], alpha)
        dec[te] = np.where(Z[te, MKL] <= tau, "ADAPT", "FREEZE")
    out["drift/KL gate"] = (dec.copy(), "leave-one-CORRUPTION-out threshold (6 folds)")

    dec = np.empty(n, dtype=object)
    for t in np.unique(task):
        te = task == t; dv = ~te
        conf_dev = np.concatenate([Z[dv, PRE_CONF], Z[dv, POST_CONF]])
        acc_dev = np.concatenate([a0[dv], aa[dv]])
        ir = IsotonicRegression(out_of_bounds="clip").fit(conf_dev, acc_dev)
        dec[te] = np.where(ir.predict(Z[te, POST_CONF]) > ir.predict(Z[te, PRE_CONF]),
                           "ADAPT", "FREEZE")
    out["ATC-style gate"] = (dec.copy(), "leave-one-CORRUPTION-out isotonic (6 folds)")

    Bhat = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        m = GradientBoostingRegressor(**GBR_KW)
        m.fit(Z[tr], B[tr]); Bhat[i] = m.predict(Z[i:i + 1])[0]
    rho = np.abs(Bhat - B)
    e_int, e_exa = eps_interp(rho), eps_exact(rho)
    out["KGA (no radius)"] = (np.where(Bhat > 0, "ADAPT", "FREEZE"),
                              "leave-one-CELL-out GBR (431 fits), no radius")
    out["KGA (certificate, interp eps)"] = (
        np.where(Bhat - e_int > 0, "ADAPT", np.where(Bhat + e_int < 0, "FREEZE", "ABSTAIN")),
        f"leave-one-CELL-out GBR (431 fits) + in-pool interpolated eps={e_int:.5f}")
    out["KGA (certificate, exact-rank eps)"] = (
        np.where(Bhat - e_exa > 0, "ADAPT", np.where(Bhat + e_exa < 0, "FREEZE", "ABSTAIN")),
        f"leave-one-CELL-out GBR (431 fits) + in-pool exact-rank eps={e_exa:.5f}")
    # leave-one-CORRUPTION-out estimator, for a genuinely apples-to-apples row
    Bhat_c = np.zeros(n)
    for t in np.unique(task):
        te = task == t
        m = GradientBoostingRegressor(**GBR_KW)
        m.fit(Z[~te], B[~te]); Bhat_c[te] = m.predict(Z[te])
    rho_c = np.abs(Bhat_c - B)
    e_c = eps_interp(rho_c)
    out["KGA (certificate, leave-one-CORRUPTION-out)"] = (
        np.where(Bhat_c - e_c > 0, "ADAPT", np.where(Bhat_c + e_c < 0, "FREEZE", "ABSTAIN")),
        f"leave-one-CORRUPTION-out GBR (6 folds, SAME budget as gates 3-4) + eps={e_c:.5f}")
    out["KGA (no radius, leave-one-CORRUPTION-out)"] = (
        np.where(Bhat_c > 0, "ADAPT", "FREEZE"),
        "leave-one-CORRUPTION-out GBR (6 folds), no radius")
    return out, Bhat, e_int, e_exa


def score(dec, a0, aa, B, idx=None):
    if idx is not None:
        dec, a0, aa, B = dec[idx], a0[idx], aa[idx], B[idx]
    adapt = dec == "ADAPT"
    realized = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    fa_num = int(np.sum(adapt & (B < 0)))
    return {"n": len(B), "regret": float((oracle - realized).mean()),
            "FA_u": float(np.mean(adapt & (B < 0))), "FA_num": fa_num,
            "FA_c": (float(np.mean(B[adapt] < 0)) if adapt.any() else None),
            "coverage": float(np.mean(dec != "ABSTAIN")),
            "adapt_rate": float(adapt.mean()), "n_adapt": int(adapt.sum()),
            "cp95_upper_FA_c": clopper_pearson_upper(fa_num, int(adapt.sum()))}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    seeds = range(5) if "--all-seeds" in sys.argv else [0]
    res = {}
    for s in seeds:
        p = f"{H2H}/per_condition_cifar10c_tent_primary_kga_seed{s}.json"
        d = load(p)
        harmful = d["B"] < 0
        g, Bhat, e_int, e_exa = gates(d)
        row = {"artifact": p, "n": len(d["B"]), "n_harmful": int(harmful.sum()),
               "eps_interp": e_int, "eps_exact": e_exa, "rules": {}}
        for name, (dec, prov) in g.items():
            dec = np.asarray(dec, dtype=object)
            row["rules"][name] = {"calibration": prov,
                                  "all": score(dec, d["a0"], d["aa"], d["B"]),
                                  "harmful_subset": score(dec, d["a0"], d["aa"], d["B"], harmful)}
        res[f"seed{s}"] = row
        print("=" * 118)
        print(f"tab:gates regenerated from {p}   n={row['n']}  harmful={row['n_harmful']}"
              f"  alpha={ALPHA}")
        print(f"{'decision rule':44s} {'regret':>8s} {'FA_u':>7s} {'FA_c':>7s} {'adapt':>6s}"
              f" {'cov':>5s} {'FA_u(harm)':>10s}  calibration")
        for name, v in row["rules"].items():
            a, h = v["all"], v["harmful_subset"]
            print(f"{name:44s} {a['regret']:8.4f} {a['FA_u']:7.3f}"
                  f" {(a['FA_c'] if a['FA_c'] is not None else float('nan')):7.3f}"
                  f" {a['adapt_rate']:6.2f} {a['coverage']:5.2f} {h['adapt_rate']:10.3f}"
                  f"  {v['calibration']}")
        r_no = row["rules"]["KGA (no radius)"]["all"]
        r_ce = row["rules"]["KGA (certificate, interp eps)"]["all"]
        print(f"\n  radius value: no-radius FA_u={r_no['FA_u']:.4f} (< alpha={ALPHA}: "
              f"{r_no['FA_u'] < ALPHA}) at regret {r_no['regret']:.4f}; certificate FA_u="
              f"{r_ce['FA_u']:.4f} at regret {r_ce['regret']:.4f} "
              f"({r_ce['regret']/max(r_no['regret'],1e-12):.2f}x higher regret, coverage "
              f"{r_ce['coverage']:.2f} vs {r_no['coverage']:.2f})")
        print(f"  harmful-cell adapt rate: no-radius {row['rules']['KGA (no radius)']['harmful_subset']['adapt_rate']:.4f}"
              f"  ->  certificate {row['rules']['KGA (certificate, interp eps)']['harmful_subset']['adapt_rate']:.4f}")
    json.dump(res, open(os.path.join(here, "out_gates.json"), "w"), indent=1, default=float)
    print("\nwrote out_gates.json")


if __name__ == "__main__":
    main()
