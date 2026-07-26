#!/usr/bin/env python3
"""Fix-queue item 4 (CIFAR-10-C half): leave-one-out-of-pool conformal radius.

The shipped rule (docs/research/kbound/scripts/cifar_tent_mps_v2.py:162)

    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))

takes the quantile over ALL N residuals, including the residual of the cell it
then scores. This script recomputes every CIFAR-10-C cell with the scored index
EXCLUDED from its own radius pool, under both the interpolated and the exact-rank
quantile, and counts how many decisions change.

Two trees are covered:
  A. stress_grid_multiseed_v1/seed{1..4}   -- 8 tent+eata files x 432 = 3456 cells
     (seed 0 has no per-condition dump; see fix-queue item 8)
  B. mixed_headtohead_v1  kga arm, seeds 0-4  -- 10 files x 432 = 4320 cells
     (this is the tree the PROMOTED panel numbers actually come from, F4-5)
Plus SAR in both trees, reported separately (quarantined track, item 6).

Run: python3 02_cifar_loo_radius.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (clopper_pearson_upper, decide, radii_in_pool, radii_loo,
                       records, score)

STRESS = "experiments/kbound/results/stress_grid_multiseed_v1"
H2H = "experiments/kbound/results/mixed_headtohead_v1"


def load(path):
    r = records(path)
    return (np.array([x["B"] for x in r], float),
            np.array([x["b_hat"] for x in r], float),
            np.array([x["a0"] for x in r], float),
            np.array([x["a_adapted"] for x in r], float),
            [x["condition"] for x in r],
            np.array([x["kga_decision"] for x in r], dtype=object),
            np.array([x["eps_conformal"] for x in r], float))


def one_file(path):
    B, bh, a0, aad, cond, shipped_dec, shipped_eps = load(path)
    out = {"path": path, "n": len(B), "shipped_eps": float(shipped_eps[0])}
    for rule in ("interp", "exact"):
        for pool in ("in_pool", "loo"):
            eps = radii_in_pool(bh, B, rule) if pool == "in_pool" else radii_loo(bh, B, rule)
            dec = decide(bh, eps)
            sc = score(dec, B, a0, aad)
            out[f"{rule}_{pool}"] = {
                "eps_min": float(eps.min()), "eps_max": float(eps.max()),
                "regret": [sc["regret_kga"], sc["regret_adapt"], sc["regret_freeze"]],
                "adapt": sc["n_adapt"], "freeze": sc["n_freeze"], "abstain": sc["n_abstain"],
                "fa_num": sc["fa_num"], "fa_u": sc["fa_u"], "fa_c": sc["fa_c"],
                "harmful_frac": sc["harmful_frac"],
                "_dec": dec,
            }
        a = out[f"{rule}_in_pool"]["_dec"]
        b = out[f"{rule}_loo"]["_dec"]
        out[f"{rule}_decisions_changed"] = int(np.sum(a != b))
        out[f"{rule}_change_detail"] = sorted(
            {f"{x}->{y}" for x, y in zip(a[a != b], b[a != b])})
    out["shipped_dec_matches_interp_in_pool"] = int(
        np.sum(shipped_dec == out["interp_in_pool"]["_dec"]))
    out["shipped_dec_matches_exact_in_pool"] = int(
        np.sum(shipped_dec == out["exact_in_pool"]["_dec"]))
    for rule in ("interp", "exact"):
        for pool in ("in_pool", "loo"):
            out[f"{rule}_{pool}"].pop("_dec")
    return out, {"B": B, "a0": a0, "aad": aad, "cond": cond}


def aggregate(files, label):
    per_file = []
    agg = {}
    keep = {}
    for f in files:
        o, raw = one_file(f)
        per_file.append(o)
        keep[f] = raw
    res = {"label": label, "files": [o["path"] for o in per_file],
           "n_cells_total": sum(o["n"] for o in per_file), "per_file": per_file}
    for rule in ("interp", "exact"):
        res[f"{rule}_total_decisions_changed"] = sum(
            o[f"{rule}_decisions_changed"] for o in per_file)
    # 5-seed pooled aggregate (mean over seeds of per-condition regret), the
    # design _locked_analysis_script.py:54 uses
    for rule in ("interp", "exact"):
        for pool in ("in_pool", "loo"):
            k = f"{rule}_{pool}"
            fa_num = sum(o[k]["fa_num"] for o in per_file)
            n = sum(o["n"] for o in per_file)
            n_adapt = sum(o[k]["adapt"] for o in per_file)
            agg[k] = {
                "regret_mean_over_files": [
                    float(np.mean([o[k]["regret"][i] for o in per_file])) for i in range(3)],
                "adapt": n_adapt,
                "freeze": sum(o[k]["freeze"] for o in per_file),
                "abstain": sum(o[k]["abstain"] for o in per_file),
                "fa_num": fa_num, "n": n, "fa_u": fa_num / n,
                "fa_c": (fa_num / n_adapt) if n_adapt else None,
                "cp95_upper_fa_c": clopper_pearson_upper(fa_num, n_adapt),
                "cp95_upper_fa_u": clopper_pearson_upper(fa_num, n),
                "eps_range": [min(o[k]["eps_min"] for o in per_file),
                              max(o[k]["eps_max"] for o in per_file)],
            }
    res["aggregate"] = agg
    return res, keep


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}

    groups = {
        "stress_grid_tent_eata_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_{c}_seed{s}.json"
            for c in ("tent", "eata") for s in (1, 2, 3, 4)],
        "stress_grid_tent_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_tent_seed{s}.json" for s in (1, 2, 3, 4)],
        "stress_grid_eata_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_eata_seed{s}.json" for s in (1, 2, 3, 4)],
        "stress_grid_sar_seed1-4": [
            f"{STRESS}/seed{s}/per_condition_cifar10c_sar_seed{s}.json" for s in (1, 2, 3, 4)],
        "headtohead_tent_kga_seed0-4": [
            f"{H2H}/per_condition_cifar10c_tent_primary_kga_seed{s}.json" for s in range(5)],
        "headtohead_eata_kga_seed0-4": [
            f"{H2H}/per_condition_cifar10c_eata_secondary_kga_seed{s}.json" for s in range(5)],
    }
    for name, files in groups.items():
        try:
            res, _ = aggregate(files, name)
        except IOError as e:
            out[name] = {"BLOCKED": str(e)}
            print("BLOCKED", name, e)
            continue
        out[name] = res
        print("=" * 100)
        print(f"{name}:  {len(files)} files, {res['n_cells_total']} cells")
        print(f"   decisions changed by leave-one-out-of-pool:"
              f"  interpolated {res['interp_total_decisions_changed']}"
              f" / {res['n_cells_total']}   exact-rank {res['exact_total_decisions_changed']}"
              f" / {res['n_cells_total']}")
        for k in ("interp_in_pool", "interp_loo", "exact_in_pool", "exact_loo"):
            a = res["aggregate"][k]
            print(f"   {k:16s} regret {a['regret_mean_over_files'][0]:.8f}"
                  f" / {a['regret_mean_over_files'][1]:.8f} / {a['regret_mean_over_files'][2]:.8f}"
                  f"  FA_u {a['fa_u']:.6f} ({a['fa_num']}/{a['n']})  adapt {a['adapt']}"
                  f"  CP95up(FA_c) {a['cp95_upper_fa_c']}")
    json.dump(out, open(os.path.join(here, "out_cifar_loo.json"), "w"), indent=1, default=float)
    print("\nwrote out_cifar_loo.json")


if __name__ == "__main__":
    main()
