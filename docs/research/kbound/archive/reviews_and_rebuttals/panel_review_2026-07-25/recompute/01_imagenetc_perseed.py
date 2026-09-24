#!/usr/bin/env python3
"""Fix-queue items 2 + 4 (ImageNet-C half) + item 5.

Rebuilds `tab:imagenetc-perseed` under BOTH quantile rules and under BOTH radius
pools (in-sample as shipped, leave-one-out-of-pool as the fix requires), from

  experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/
      per_condition_imagenetc_sar_seed{0..4}.json

Also reports the same for tent and eata (the other two ImageNet-C candidates)
so the "one rule everywhere" declaration can be checked.

Run: python3 01_imagenetc_perseed.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (ALPHA, clopper_pearson_upper, decide, radii_in_pool,
                       radii_loo, records, score, wilson)

ROOT = "experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed"
SEEDS = [0, 1, 2, 3, 4]
OUT = {}


def load_seed(cand, s):
    p = f"{ROOT}/per_condition_imagenetc_{cand}_seed{s}.json"
    r = records(p)
    return (p,
            np.array([x["B"] for x in r], float),
            np.array([x["b_hat"] for x in r], float),
            np.array([x["a0"] for x in r], float),
            np.array([x["a_adapted"] for x in r], float),
            [x["condition"] for x in r],
            [x["kga_decision"] for x in r],
            np.array([x["eps_conformal"] for x in r], float))


def run(cand):
    res = {"candidate": cand, "per_seed": {}, "artifacts": []}
    variants = [("interp", "in_pool"), ("exact", "in_pool"),
                ("interp", "loo"), ("exact", "loo")]
    pooled = {v: {"B": [], "dec": [], "a0": [], "aad": [], "cond": [], "seed": []}
              for v in variants}
    for s in SEEDS:
        p, B, bh, a0, aad, cond, shipped_dec, shipped_eps = load_seed(cand, s)
        res["artifacts"].append(p)
        row = {"n": len(B), "harmful_frac": float(np.mean(B < 0)),
               "shipped_eps_in_file": float(shipped_eps[0]),
               "shipped_eps_unique": int(len(np.unique(np.round(shipped_eps, 12))))}
        for rule, pool in variants:
            eps = (radii_in_pool(bh, B, rule) if pool == "in_pool"
                   else radii_loo(bh, B, rule))
            dec = decide(bh, eps)
            sc = score(dec, B, a0, aad)
            key = f"{rule}_{pool}"
            row[key] = {
                "eps_min": float(np.min(eps)), "eps_max": float(np.max(eps)),
                "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]],
                "fa_u": sc["fa_u"], "fa_num": sc["fa_num"], "fa_c": sc["fa_c"],
                "adapt": sc["n_adapt"], "freeze": sc["n_freeze"], "abstain": sc["n_abstain"],
                "beats_both": bool(sc["regret_kga"] < sc["regret_adapt"] - 1e-12
                                   and sc["regret_kga"] < sc["regret_freeze"] - 1e-12),
                "ties_freeze_exactly": bool(abs(sc["regret_kga"] - sc["regret_freeze"]) < 1e-15),
                "ties_adapt_exactly": bool(abs(sc["regret_kga"] - sc["regret_adapt"]) < 1e-15),
                "cp95_upper_fa_c": clopper_pearson_upper(sc["fa_num"], sc["n_adapt"]),
                "cp95_upper_fa_u": clopper_pearson_upper(sc["fa_num"], sc["n"]),
            }
            pooled[(rule, pool)]["B"] += list(B)
            pooled[(rule, pool)]["dec"] += list(dec)
            pooled[(rule, pool)]["a0"] += list(a0)
            pooled[(rule, pool)]["aad"] += list(aad)
            pooled[(rule, pool)]["cond"] += cond
            pooled[(rule, pool)]["seed"] += [s] * len(B)
        # decision agreement with the file's own kga_decision field
        for rule, pool in variants:
            eps = (radii_in_pool(bh, B, rule) if pool == "in_pool"
                   else radii_loo(bh, B, rule))
            dec = decide(bh, eps)
            row[f"{rule}_{pool}"]["agree_with_shipped_decision"] = int(
                np.sum(np.array(shipped_dec, dtype=object) == dec))
        res["per_seed"][s] = row

    res["pooled"] = {}
    for v in variants:
        d = pooled[v]
        sc = score(d["dec"], d["B"], d["a0"], d["aad"])
        res["pooled"][f"{v[0]}_{v[1]}"] = {
            "n": sc["n"],
            "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]],
            "fa_u": sc["fa_u"], "fa_num": sc["fa_num"], "fa_c": sc["fa_c"],
            "adapt": sc["n_adapt"], "freeze": sc["n_freeze"], "abstain": sc["n_abstain"],
            "beats_both": bool(sc["regret_kga"] < sc["regret_adapt"] - 1e-12
                               and sc["regret_kga"] < sc["regret_freeze"] - 1e-12),
            "cp95_upper_fa_c": clopper_pearson_upper(sc["fa_num"], sc["n_adapt"]),
            "cp95_upper_fa_u": clopper_pearson_upper(sc["fa_num"], sc["n"]),
            "wilson_fa_u": wilson(sc["fa_num"], sc["n"]),
        }
    # keep the pooled per-cell vectors for the bootstrap script
    res["_pooled_vectors"] = {
        f"{v[0]}_{v[1]}": {
            "B": list(map(float, pooled[v]["B"])),
            "a0": list(map(float, pooled[v]["a0"])),
            "aad": list(map(float, pooled[v]["aad"])),
            "dec": list(pooled[v]["dec"]),
            "cond": list(pooled[v]["cond"]),
            "seed": list(map(int, pooled[v]["seed"])),
        } for v in variants}
    return res


if __name__ == "__main__":
    for cand in ("sar", "tent", "eata"):
        OUT[cand] = run(cand)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(OUT, open(os.path.join(here, "out_imagenetc_perseed.json"), "w"), indent=1)

    for cand in ("sar", "tent", "eata"):
        r = OUT[cand]
        print("=" * 100)
        print(f"ImageNet-C {cand.upper()}   (alpha={ALPHA})")
        for key, lbl in [("interp_in_pool", "INTERPOLATED, in-pool (archived / appendix table)"),
                         ("exact_in_pool", "EXACT-RANK,   in-pool (promoted / manifest)"),
                         ("interp_loo", "INTERPOLATED, leave-one-out-of-pool"),
                         ("exact_loo", "EXACT-RANK,   leave-one-out-of-pool  <== FIX")]:
            print(f"\n  --- {lbl}")
            print("   seed |    KGA     adapt    freeze |  FA_u  (k/n) | ad  fr  ab | BB | ties-freeze")
            for s in SEEDS:
                d = r["per_seed"][s][key]
                print(f"    {s}   | {d['regret'][0]:.6f} {d['regret'][1]:.6f} {d['regret'][2]:.6f}"
                      f" | {d['fa_u']:.4f} ({d['fa_num']}/27) | {d['adapt']:2d}  {d['freeze']:2d}  {d['abstain']:2d}"
                      f" | {str(d['beats_both'])[:1]}  | {d['ties_freeze_exactly']}")
            p = r["pooled"][key]
            print(f"   POOL | {p['regret'][0]:.6f} {p['regret'][1]:.6f} {p['regret'][2]:.6f}"
                  f" | {p['fa_u']:.4f} ({p['fa_num']}/{p['n']}) | {p['adapt']:2d}  {p['freeze']:2d}  {p['abstain']:3d}"
                  f" | {str(p['beats_both'])[:1]}")
            n_bb = sum(1 for s in SEEDS if r["per_seed"][s][key]["beats_both"])
            n_tie = sum(1 for s in SEEDS if r["per_seed"][s][key]["ties_freeze_exactly"])
            print(f"   seeds strictly beating BOTH: {n_bb}/5   |  bit-identical ties with always-freeze: {n_tie}/5")
