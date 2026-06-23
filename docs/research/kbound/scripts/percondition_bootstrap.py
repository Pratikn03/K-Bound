#!/usr/bin/env python3
"""
percondition_bootstrap.py -- the STANDARD paired per-condition bootstrap for the CIFAR-10-C
stress grid, computed from the SAVED per-condition records (no re-run, no GPU, pure numpy).

Why this exists
---------------
The CIs in decisive_tta_cis.json are *design-based*: they resample the pre-registered mixing-ratio
(Pareto) distribution, so they quantify uncertainty over the DEPLOYMENT MIXTURE. That is correct and
honestly labeled, but reviewers also expect the conventional CI that resamples the CONDITIONS
themselves. The summary file did not retain per-condition a0/aa arrays, but the per_condition_*.json
records DO. This script computes the conventional CI from those records, so we can report BOTH:

  * design-based  -> uncertainty over the pre-registered harmful-fraction range (decisive_tta_cis.json)
  * per-condition -> uncertainty at the REALIZED grid composition (this script)

Each of the 432 conditions is averaged over seeds (replicates), then the 432 condition means are
resampled with replacement (B=5000). "Beats both (CI)" requires BOTH regret-gap upper bounds < 0.

As a bonus it also reports FA_u (kga adapted AND true benefit < 0) and the EMPIRICAL conformal
coverage Pr[|b_hat - B| <= eps], which backs the out-of-fold calibration claim with a real number.

Reads:  per_condition_cifar10c_<method>_seed<s>.json
Writes: percondition_bootstrap.json  (+ prints a markdown table)

Usage:
  python scripts/percondition_bootstrap.py --root experiments/kbound/results
  python scripts/percondition_bootstrap.py --root . --pattern "per_condition_cifar10c_*.json"
  python scripts/percondition_bootstrap.py --selftest
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

NBOOT = 5000
CI = 0.95


def find_files(root, pattern):
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return sorted(f for f in files if not os.path.basename(f).startswith("._"))


def load(root, pattern):
    """Return {method: {seed: [records]}} de-duplicated by (method, seed)."""
    by_method = defaultdict(dict)
    used = []
    for f in find_files(root, pattern):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        method, seed, recs = d.get("method"), d.get("seed"), d.get("records", [])
        if method is None or not recs:
            continue
        if seed in by_method[method]:
            continue  # already have this (method, seed)
        by_method[method][seed] = recs
        used.append((method, seed, len(recs), f))
    return by_method, used


def boot_ci(x, nboot=NBOOT, ci=CI, seed=0):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(means, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def analyze(recs):
    # ---- per-CELL regret (oracle + KGA decision applied PER CELL, never seed-averaged before
    #      the max()), then condition means so the bootstrap unit is the condition (block bootstrap).
    #      Applying max() per cell avoids the Jensen bias that seed-averaging-before-oracle would add
    #      (which spuriously shrinks tiny regrets, e.g. helpful-dominated SAR). ----
    def _ak(r):  # KGA realized accuracy: ADAPT -> adapted batch, else frozen (FREEZE/ABSTAIN)
        if r.get("a_kbound") is not None:
            return float(r["a_kbound"])
        return float(r["a_adapted"]) if str(r.get("kga_decision", "")).upper() == "ADAPT" else float(r["a0"])

    def _ao(r):  # oracle = better of the two fixed actions, per cell
        if r.get("a_oracle") is not None:
            return float(r["a_oracle"])
        return max(float(r["a0"]), float(r["a_adapted"]))

    by_cond = defaultdict(lambda: {"rk": [], "ra": [], "rf": []})
    for r in recs:
        a0i, aai, aki, aoi = float(r["a0"]), float(r["a_adapted"]), _ak(r), _ao(r)
        g = by_cond[r["condition"]]
        g["rk"].append(aoi - aki)   # KGA regret-to-oracle, this cell
        g["ra"].append(aoi - aai)   # always-adapt regret
        g["rf"].append(aoi - a0i)   # always-freeze regret
    conds = list(by_cond)
    reg_kga    = np.array([np.mean(by_cond[c]["rk"]) for c in conds])
    reg_adapt  = np.array([np.mean(by_cond[c]["ra"]) for c in conds])
    reg_freeze = np.array([np.mean(by_cond[c]["rf"]) for c in conds])
    d_adapt  = reg_kga - reg_adapt    # < 0  => KGA beats always-adapt
    d_freeze = reg_kga - reg_freeze   # < 0  => KGA beats always-freeze
    ga = boot_ci(d_adapt, seed=1)
    gf = boot_ci(d_freeze, seed=2)

    # ---- cell-level rates (FA_u, empirical coverage) over all seeds ----
    B   = np.array([r.get("B", np.nan) for r in recs], float)
    dec = np.array([str(r.get("kga_decision", "")).upper() for r in recs])
    bh  = np.array([r.get("b_hat", np.nan) for r in recs], float)
    eps = np.array([r.get("eps_conformal", np.nan) for r in recs], float)
    fa_u = float(np.mean((dec == "ADAPT") & (B < 0))) if np.isfinite(B).all() else None
    cov = (float(np.mean(np.abs(bh - B) <= eps))
           if np.isfinite(bh).all() and np.isfinite(eps).all() and np.isfinite(B).all() else None)

    return {
        "n_conditions": int(len(reg_kga)),
        "n_cells": int(len(recs)),
        "regret_KGA": float(reg_kga.mean()),
        "regret_always_adapt": float(reg_adapt.mean()),
        "regret_always_freeze": float(reg_freeze.mean()),
        "gap_vs_adapt":  {"mean": ga[0], "ci95_lo": ga[1], "ci95_hi": ga[2]},
        "gap_vs_freeze": {"mean": gf[0], "ci95_lo": gf[1], "ci95_hi": gf[2]},
        "beats_both_ci": bool(ga[2] < 0 and gf[2] < 0),
        "FA_u": fa_u,
        "empirical_coverage": cov,
    }


def print_table(out):
    print("\n| method | regret KGA | gap vs adapt [95% CI] | gap vs freeze [95% CI] | FA_u | cov | beats both (CI)? |")
    print("|---|---|---|---|---|---|---|")
    for m, r in out["methods"].items():
        ga, gf = r["gap_vs_adapt"], r["gap_vs_freeze"]
        fa = "n/a" if r["FA_u"] is None else f"{r['FA_u']:.3f}"
        cv = "n/a" if r["empirical_coverage"] is None else f"{r['empirical_coverage']:.3f}"
        print(f"| {m} | {r['regret_KGA']:.4f} | "
              f"{ga['mean']:+.4f} [{ga['ci95_lo']:+.4f}, {ga['ci95_hi']:+.4f}] | "
              f"{gf['mean']:+.4f} [{gf['ci95_lo']:+.4f}, {gf['ci95_hi']:+.4f}] | "
              f"{fa} | {cv} | {'YES' if r['beats_both_ci'] else 'no'} |")
    print("\nbeats_both (CI) = both regret-gap upper bounds < 0 at the realized grid composition.")


def selftest():
    # KGA = near-oracle (picks better of a0/aa per cell) -> must beat both with CI < 0.
    rng = np.random.default_rng(0)
    recs = []
    for i in range(200):
        a0 = rng.uniform(0.5, 0.9)
        B = rng.uniform(-0.1, 0.1)
        aa = a0 + B
        ak = max(a0, aa) - 1e-4            # KGA ~ oracle
        recs.append(dict(condition=f"c{i}", seed=0, a0=a0, a_adapted=aa, a_kbound=ak,
                         a_oracle=max(a0, aa), B=B,
                         kga_decision="ADAPT" if aa >= a0 else "FREEZE",
                         b_hat=B + rng.normal(0, 0.01), eps_conformal=0.05))
    r = analyze(recs)
    assert r["beats_both_ci"], r
    assert r["gap_vs_adapt"]["ci95_hi"] < 0 and r["gap_vs_freeze"]["ci95_hi"] < 0, r
    assert r["FA_u"] == 0.0, r["FA_u"]            # KGA never adapts a harmful cell here
    print("selftest OK:", json.dumps(r, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/kbound/results",
                    help="dir to search recursively for per_condition_cifar10c_*.json")
    ap.add_argument("--pattern", default="per_condition_cifar10c_*.json")
    ap.add_argument("--out", default="percondition_bootstrap.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    by_method, used = load(args.root, args.pattern)
    if not by_method:
        sys.exit(f"No files under {args.root} matching {args.pattern}. "
                 f"Try --root . or point --root at the results tree.")
    print("loaded (method, seed, n_records):")
    for m, s, n, f in used:
        print(f"  {m} seed{s}: {n}")
    out = {"nboot": NBOOT, "ci_level": CI,
           "ci_kind": "standard paired per-condition bootstrap (432 conditions seed-averaged, "
                      "resampled with replacement); complements the design-based mixing-ratio CI",
           "methods": {}}
    for method, seeds in sorted(by_method.items()):
        recs = [r for s in seeds for r in seeds[s]]
        res = analyze(recs)
        res["n_seeds"] = len(seeds)
        out["methods"][method] = res
    json.dump(out, open(args.out, "w"), indent=2)
    print_table(out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
