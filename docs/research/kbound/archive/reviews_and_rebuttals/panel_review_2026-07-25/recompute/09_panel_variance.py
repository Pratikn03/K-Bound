#!/usr/bin/env python3
"""Fix-queue item 23 (F5-10, F3-15, F3-16): per-seed / per-unit spread behind
every multi-seed panel row that currently reports only a mean.

Sources:
  PACS          experiments/kbound/results/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json
                (+ the per-cell dumps in experiments/kbound/results/per_cell/, 2 of 3 seeds)
  ImageNet-R    experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/
                MULTISEED_ANALYSIS_RESULTS.json  (10 backbones x 4 seeds)
  CIFAR-10-C    experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json
                (per-seed regret + harmful base rate + eps, incl. the SAR quarantine row)
  Camelyon17    experiments/kbound/results/wilds_kbound/per_condition_camelyon17_*_seed*.json
  ImageNet-C    experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/

Run: python3 09_panel_variance.py
"""
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb_common import (ALPHA, clopper_pearson, clopper_pearson_upper, read_json,
                       wilson)

R = "experiments/kbound/results"


def spread(xs):
    xs = [float(x) for x in xs]
    return {"n": len(xs), "min": min(xs), "median": float(np.median(xs)),
            "max": max(xs), "mean": float(np.mean(xs)),
            "sd": (st.stdev(xs) if len(xs) > 1 else 0.0), "values": xs}


def pacs():
    d = read_json(f"{R}/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json")
    out = {"source": f"{R}/pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json",
           "seeds": d["seeds"], "per_domain": {}, "over_budget_cells": []}
    fa_all, k_all, n_all = [], 0, 0
    kga, ad, fr = [], [], []
    for dom, v in d["per_domain"].items():
        row = {}
        for key in ("FA_u", "FA_c", "coverage", "adapt_rate", "base_rate_harmful",
                    "regret_K_Bound", "regret_always_adapt", "regret_always_freeze"):
            row[key] = spread(v[key]["per_seed"])
        row["n_test_cells_per_seed"] = v["n_test_cells_per_seed"]
        row["verdict_per_seed"] = v["verdict_per_seed"]
        # integer counts implied by the rates (denominator = 18 cells per domain-seed)
        row["implied_fa_counts"] = [int(round(f * n)) for f, n in
                                    zip(v["FA_u"]["per_seed"], v["n_test_cells_per_seed"])]
        row["implied_adapt_counts"] = [int(round(a * n)) for a, n in
                                       zip(v["adapt_rate"]["per_seed"], v["n_test_cells_per_seed"])]
        for i, (f, n) in enumerate(zip(v["FA_u"]["per_seed"], v["n_test_cells_per_seed"])):
            if f > ALPHA:
                out["over_budget_cells"].append(
                    {"domain": dom, "seed": d["seeds"][i], "FA_u": f, "n": n,
                     "k": int(round(f * n))})
        fa_all += v["FA_u"]["per_seed"]
        k_all += sum(row["implied_fa_counts"]); n_all += sum(v["n_test_cells_per_seed"])
        kga += v["regret_K_Bound"]["per_seed"]
        ad += v["regret_always_adapt"]["per_seed"]
        fr += v["regret_always_freeze"]["per_seed"]
        out["per_domain"][dom] = row
    out["pooled"] = {
        "n_domain_seed_cells": len(fa_all),
        "mean_FA_u_over_cells": float(np.mean(fa_all)),
        "implied_pooled_fa_count": k_all, "implied_pooled_n": n_all,
        "implied_pooled_FA_u": k_all / n_all,
        "wilson95_pooled_FA_u": wilson(k_all, n_all),
        "clopper_pearson95_pooled_FA_u": clopper_pearson(k_all, n_all),
        "cp95_upper_pooled_FA_u": clopper_pearson_upper(k_all, n_all),
        "regret_kga": spread(kga), "regret_adapt": spread(ad), "regret_freeze": spread(fr),
        "panel_row_reported": [0.0431, 0.0176, 0.0446],
        "recomputed_mean_over_12_domain_seed_cells":
            [float(np.mean(kga)), float(np.mean(ad)), float(np.mean(fr))],
        "note": "PACS_MULTISEED_RESULTS.json records false_adapt_count_status='not_retained'; "
                "the integer counts above are back-derived from rate x n and are exact only "
                "because n=18 and the rates are multiples of 1/18.",
    }
    return out


def imagenetr():
    p = f"{R}/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json"
    d = read_json(p)
    out = {"source": p, "seeds": d["seeds"], "per_backbone": {}}
    kga, ad, fr, harm = [], [], [], []
    for bb, v in d["candidates"].items():
        row = {"regret": [v["kga_mean_regret"], v["adapt_mean_regret"], v["freeze_mean_regret"]],
               "harmful_base_rate_per_seed": v["harmful_base_rate_per_seed"],
               "harmful_base_rate_mean": float(np.mean(v["harmful_base_rate_per_seed"])),
               "false_adapt_num": v["false_adapt_num"], "false_adapt_den": v["false_adapt_den"],
               "fa_u": v["false_adapt_rate_pooled"],
               "cp95_upper_fa_u": clopper_pearson_upper(v["false_adapt_num"], v["false_adapt_den"]),
               "eps_per_seed": v["eps_conformal_per_seed"], "eps_cv": v["eps_cv"],
               "between_seed_std": v["between_seed_std_of_seedmeanregret"],
               "kga_worse_than_adapt": bool(v["kga_mean_regret"] > v["adapt_mean_regret"] + 1e-12),
               "ratio_kga_over_adapt": (v["kga_mean_regret"] / v["adapt_mean_regret"]
                                        if v["adapt_mean_regret"] > 0 else None),
               "degenerate_zero_harmful": bool(max(v["harmful_base_rate_per_seed"]) == 0.0)}
        out["per_backbone"][bb] = row
        kga.append(v["kga_mean_regret"]); ad.append(v["adapt_mean_regret"])
        fr.append(v["freeze_mean_regret"]); harm.append(row["harmful_base_rate_mean"])
    out["across_backbones"] = {
        "kga": spread(kga), "adapt": spread(ad), "freeze": spread(fr),
        "harmful_base_rate": spread(harm),
        "panel_row_reported": [0.0112, 0.0064, 0.0325],
        "recomputed_mean": [float(np.mean(kga)), float(np.mean(ad)), float(np.mean(fr))],
        "n_backbones_where_kga_worse_than_adapt":
            sum(1 for b in out["per_backbone"].values() if b["kga_worse_than_adapt"]),
        "n_backbones_degenerate_zero_harmful":
            sum(1 for b in out["per_backbone"].values() if b["degenerate_zero_harmful"]),
    }
    return out


def cifar_stress():
    p = f"{R}/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"
    d = read_json(p)
    out = {"source": p, "seeds": d["seeds"], "per_candidate": {}}
    rows = d["pstar_law"]["per_seed_cand"]
    for cand, v in d["candidates"].items():
        rs = [r for r in rows if r["candidate"] == cand]
        out["per_candidate"][cand] = {
            "pooled_regret": [v["kga_mean_regret"], v["adapt_mean_regret"], v["freeze_mean_regret"]],
            "per_seed_regret_kga": spread([r["regret_kga"] for r in rs]),
            "per_seed_regret_adapt": spread([r["regret_adapt"] for r in rs]),
            "per_seed_regret_freeze": spread([r["regret_freeze"] for r in rs]),
            "per_seed_harmful_frac": spread([r["harmful_frac"] for r in rs]),
            "per_seed_beats_both": {str(r["seed"]): r["beats_both"] for r in rs},
            "n_seeds_beating_both": sum(1 for r in rs if r["beats_both"]),
            "eps_per_seed": v["eps_conformal_per_seed"], "eps_cv": v["eps_cv"],
            "false_adapt": [v["false_adapt_num"], v["false_adapt_den"]],
            "cp95_upper_fa_u": clopper_pearson_upper(v["false_adapt_num"], v["false_adapt_den"]),
            "between_seed_std": v["between_seed_std_of_seedmeanregret"],
        }
    # seeds 1-4 only (the SAR quarantine restatement, fix-queue item 6)
    for cand in ("sar", "tent", "eata"):
        rs = [r for r in rows if r["candidate"] == cand and r["seed"] != 0]
        out["per_candidate"][cand]["seeds_1to4_only"] = {
            "regret": [float(np.mean([r["regret_kga"] for r in rs])),
                       float(np.mean([r["regret_adapt"] for r in rs])),
                       float(np.mean([r["regret_freeze"] for r in rs]))],
            "harmful_frac_mean": float(np.mean([r["harmful_frac"] for r in rs])),
            "kga_worse_than_adapt": bool(np.mean([r["regret_kga"] for r in rs])
                                         > np.mean([r["regret_adapt"] for r in rs])),
        }
        r0 = [r for r in rows if r["candidate"] == cand and r["seed"] == 0][0]
        out["per_candidate"][cand]["seed0_only"] = {
            "regret": [r0["regret_kga"], r0["regret_adapt"], r0["regret_freeze"]],
            "harmful_frac": r0["harmful_frac"], "beats_both": r0["beats_both"]}
        hf0 = r0["harmful_frac"]
        hf14 = [r["harmful_frac"] for r in rs]
        out["per_candidate"][cand]["seed0_harmful_ratio_vs_seeds1to4_mean"] = (
            hf0 / float(np.mean(hf14)))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"pacs": pacs(), "imagenet_r": imagenetr(), "cifar10c_stress_grid": cifar_stress()}
    json.dump(out, open(os.path.join(here, "out_panel_variance.json"), "w"),
              indent=1, default=float)

    p = out["pacs"]
    print("=" * 100)
    print("PACS  (panel row reports the mean of 12 domain-seed cells, no interval)")
    print(f"  panel row     {p['pooled']['panel_row_reported']}")
    print(f"  recomputed    {[round(x,4) for x in p['pooled']['recomputed_mean_over_12_domain_seed_cells']]}")
    print(f"  KGA regret across 12 domain-seed cells: min {p['pooled']['regret_kga']['min']:.5f}"
          f"  median {p['pooled']['regret_kga']['median']:.5f}  max {p['pooled']['regret_kga']['max']:.5f}"
          f"  sd {p['pooled']['regret_kga']['sd']:.5f}")
    print(f"  pooled FA_u   {p['pooled']['implied_pooled_fa_count']}/{p['pooled']['implied_pooled_n']}"
          f" = {p['pooled']['implied_pooled_FA_u']:.5f}"
          f"   Wilson95 {tuple(round(x,5) for x in p['pooled']['wilson95_pooled_FA_u'])}"
          f"   CP95 {tuple(round(x,5) for x in p['pooled']['clopper_pearson95_pooled_FA_u'])}")
    print(f"  cells over budget (FA_u > {ALPHA}): {p['over_budget_cells']}")
    for dom, v in p["per_domain"].items():
        print(f"    {dom:14s} FA_u/seed {v['FA_u']['values']}  adapt_rate/seed {v['adapt_rate']['values']}"
              f"  coverage/seed {v['coverage']['values']}")

    ir = out["imagenet_r"]
    print("=" * 100)
    print("ImageNet-R D  (panel row is the MEAN across 10 backbones)")
    print(f"  panel row  {ir['across_backbones']['panel_row_reported']}")
    print(f"  recomputed {[round(x,4) for x in ir['across_backbones']['recomputed_mean']]}")
    print(f"  KGA across backbones: min {ir['across_backbones']['kga']['min']:.5f}"
          f"  median {ir['across_backbones']['kga']['median']:.5f}"
          f"  max {ir['across_backbones']['kga']['max']:.5f}")
    print(f"  backbones where KGA is WORSE than always-adapt: "
          f"{ir['across_backbones']['n_backbones_where_kga_worse_than_adapt']}/10")
    print(f"  backbones with a degenerate 0% harmful base rate: "
          f"{ir['across_backbones']['n_backbones_degenerate_zero_harmful']}/10")
    print(f"  {'backbone':20s} {'KGA':>9s} {'adapt':>9s} {'freeze':>9s} {'x adapt':>8s} {'harm%':>7s}")
    for bb, v in ir["per_backbone"].items():
        rr = v["ratio_kga_over_adapt"]
        print(f"  {bb:20s} {v['regret'][0]:9.5f} {v['regret'][1]:9.5f} {v['regret'][2]:9.5f}"
              f" {(f'{rr:.2f}x' if rr else '   n/a'):>8s} {100*v['harmful_base_rate_mean']:6.1f}%")

    cs = out["cifar10c_stress_grid"]
    print("=" * 100)
    print("CIFAR-10-C stress grid, per-seed spread (LOCKED_ANALYSIS_RESULTS.json)")
    for cand, v in cs["per_candidate"].items():
        print(f"  {cand.upper()}  pooled {['%.6f'%x for x in v['pooled_regret']]}"
              f"  seeds beating both: {v['n_seeds_beating_both']}/5")
        print(f"     KGA per seed   {['%.6f'%x for x in v['per_seed_regret_kga']['values']]}")
        print(f"     adapt per seed {['%.6f'%x for x in v['per_seed_regret_adapt']['values']]}")
        print(f"     harmful/seed   {v['per_seed_harmful_frac']['values']}"
              f"   seed0/mean(1-4) = {v['seed0_harmful_ratio_vs_seeds1to4_mean']:.2f}x")
        print(f"     eps per seed   {v['eps_per_seed']}  cv {v['eps_cv']:.4f}")
        s = v["seeds_1to4_only"]
        print(f"     SEEDS 1-4 ONLY regret {['%.6f'%x for x in s['regret']]}"
              f"   KGA worse than always-adapt: {s['kga_worse_than_adapt']}")
    print("\nwrote out_panel_variance.json")


if __name__ == "__main__":
    main()
