#!/usr/bin/env python3
"""
uniform_scorer.py  --  ONE verdict rule for ALL K-Bound datasets.

WHY: each dataset's runner prints its own beats_both/SAFETY/NULL flag from separate code.
This re-scores every dataset with IDENTICAL logic and flags any runner-vs-uniform disagreement.
That is the fix for "are different datasets scored with different logic?": here, they are not.

THE KEY INSIGHT (from the point-regret table):
    POINT beats-both (regret_kga < both fixed policies) is NECESSARY but NOT SUFFICIENT.
    On most natural datasets KGA point-beats both, yet the honest verdict is NO-HARM, because:
      (1) the gap to the BETTER policy (usually freeze) is tiny, so its 95% paired-bootstrap
          CI includes zero  -> statistically KGA only TIES freeze  -> NO-HARM, not a win; OR
      (2) the held-out set was contaminated (e.g. Camelyon17 G pooled id_val in) -> withdrawn.
    So the ONE uniform discriminator that separates real wins from no-harm is CI-ROBUSTNESS
    of the gap to the better fixed policy (plus a clean-held-out flag), NOT the point sign.

DESIGN:  per-dataset ADAPTER (locate numbers) -> ONE verdict() (decide) -> table.
FOLD-IN TODO: extend the adapters' recs[] with per-cell (acc_freeze,acc_adapt,acc_oracle,acc_kga)
    from each dataset's per_condition_*.json / records[] so the CI-robust branch fires for all.
"""
import json, os, argparse, math

ALPHA = 0.10     # false-adapt budget
TOL   = 0.003    # regret tie tolerance (point)
Z     = 1.96     # 95% CI

REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
R    = os.path.join(REPO, "experiments/kbound/results")

# name, path, family, clean_heldout(bool)  -- clean_heldout=False marks a contaminated split
DATASETS = [
    ("CIFAR-10-C",  f"{R}/win_hunt_v5/cifar10c_aggr/seed0/decisive_tta_results.json", "cifar", True),
    ("CIFAR-10.1",  f"{R}/win_hunt_v5/cifar101_aggr/seed0/decisive_tta_results.json", "cifar", True),
    ("ImageNet-C",  f"{R}/win_hunt_v5/imagenetc_aggr/decisive_tta_results.json",       "cifar", True),
    ("RxRx1",       f"{R}/win_hunt_v5/rxrx1_aggr/result_4a2840ef.json",               "wilds", True),
    ("Camelyon17",  f"{R}/wilds_kbound/result_8d3c0c41.json",                          "wilds", True),
    ("ImageNet-R",  f"{R}/imagenetr_kbound_debug_mps/result_75ee8322.json",            "wilds", True),
    ("Office-Home", f"{R}/officehome_kbound_run/result_target_test_d2f4bf2c.json",     "wilds", True),
    ("iWildCam",    f"{R}/win_hunt_v5_iwildcam/result_0ba633eb.json",                  "wilds", True),
    ("PACS",        f"{R}/win_hunt_v5/pacs_aggr/pacs_result.json",                     "pacs",  True),
]

def load(p):
    with open(p) as f: return json.load(f)

def _fa(dbr, n):
    if not dbr or not n: return None
    h = dbr.get("harmful", {})
    return sum(c for a, c in h.items() if a not in ("ABSTAIN", "FREEZE")) / n

# ---------- ADAPTERS: locate numbers only. Emit per-cell regrets if available (for CI). ----------
def _percell_regrets(unit_records):
    """Return (gap_adapt[], gap_freeze[]) per-cell regret gaps if the records expose per-cell
    freeze/adapt/oracle/kga accuracy; else None. gap_* = regret_kga - regret_policy (want <0)."""
    if not unit_records: return None, None
    ga, gf = [], []
    for r in unit_records:
        f, a, o, k = (r.get(x) for x in ("acc_freeze", "acc_adapt", "acc_oracle", "acc_kga"))
        if None in (f, a, o, k): return None, None
        ga.append((o - k) - (o - a)); gf.append((o - k) - (o - f))
    return ga, gf

def adapt_cifar(d):
    U = []
    for b, bd in d.get("benchmarks", {}).items():
        for m, md in bd.get("methods", {}).items():
            met = md["metrics"]; rv = met["regret_vs_oracle"]; dbr = met.get("decisions_by_regime", {})
            n = met.get("n") or sum(sum(v.values()) for v in dbr.values())
            U.append(dict(unit=f"{b}:{m}", r_kga=rv["K_Bound"], r_adapt=rv["always_adapt"],
                          r_freeze=rv["always_freeze"], fa_u=_fa(dbr, n), n=n,
                          printed="beats-both" if met.get("beats_both") else "no",
                          recs=None))
    return U

def adapt_wilds(d):
    U = []
    rb = d.get("routing_b_multicandidate")
    if rb and rb.get("regret_vs_oracle"):
        rv = rb["regret_vs_oracle"]; dbr = rb.get("decisions_by_regime", {})
        n = rb.get("n_conditions") or sum(sum(v.values()) for v in dbr.values())
        fa = rb.get("false_adapt_rate"); fa = fa if fa is not None else _fa(dbr, n)
        U.append(dict(unit="multicand_router", r_kga=rv.get("router"),
                      r_adapt=rv.get("best_fixed_always_adapt"), r_freeze=rv.get("always_freeze"),
                      fa_u=fa, n=n, printed="beats-both" if rb.get("beats_both") else "no", recs=None))
    for c, cd in d.get("routing_a_single_candidate", {}).items():
        k = cd.get("kga", {}); rv = k.get("regret_vs_oracle")
        if rv:
            U.append(dict(unit=f"cand:{c}", r_kga=rv["K_Bound"], r_adapt=rv["always_adapt"],
                          r_freeze=rv["always_freeze"], fa_u=None, n=k.get("n"),
                          printed="beats-both" if k.get("beats_both") else "no", recs=None))
    return U

def adapt_pacs(d):
    return [dict(unit=k, r_kga=v["regret_vs_oracle"]["K_Bound"],
                 r_adapt=v["regret_vs_oracle"]["always_adapt"],
                 r_freeze=v["regret_vs_oracle"]["always_freeze"],
                 fa_u=v.get("FA_u"), n=v.get("n"), printed=v.get("verdict"), recs=None)
            for k, v in d.get("per_domain", {}).items()]

ADAPTERS = {"cifar": adapt_cifar, "wilds": adapt_wilds, "pacs": adapt_pacs}

# ---------------- THE ONE verdict engine (identical for every dataset) ----------------
def verdict(u, clean=True, alpha=ALPHA, tol=TOL):
    rk, ra, rf, fa = u.get("r_kga"), u.get("r_adapt"), u.get("r_freeze"), u.get("fa_u")
    if rk is None or ra is None or rf is None: return "NA", ""
    if fa is not None and fa > alpha + 1e-9:   return "FAIL", "FA_u>alpha"
    if not clean:                              return "WITHDRAWN", "contaminated held-out"
    better, worse = min(ra, rf), max(ra, rf)
    point_bb = (rk < ra - tol) and (rk < rf - tol)
    # CI-robust: need per-cell gaps to test that the gap to the BETTER policy excludes 0
    ga, gf = _percell_regrets(u.get("recs"))
    if point_bb and ga and gf:
        def excl0(g):
            mu = sum(g)/len(g); sd = (sum((x-mu)**2 for x in g)/max(1,len(g)-1))**0.5
            se = sd/max(1, len(g))**0.5
            return mu + Z*se < 0            # gap<0 means KGA better; upper CI still <0
        if excl0(ga) and excl0(gf):        return "BEATS-BOTH", "CI-robust"
        return "NO-HARM", "point-win but CI to better policy includes 0"
    if point_bb:                            return "BEATS-BOTH?", "POINT only (no per-cell CI in JSON)"
    if abs(rk - better) <= tol and rk <= worse + tol: return "NO-HARM", "ties better, beats worse"
    return "NULL", ""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verbose", action="store_true"); a = ap.parse_args()
    print(f"\nUNIFORM SCORER  alpha={ALPHA} tol={TOL}  --  one rule, every dataset\n")
    h = f"{'dataset':13}{'unit':20}{'r_kga':>9}{'r_adapt':>9}{'r_freeze':>9}{'FA_u':>7}  {'VERDICT':12} why / runner"
    print(h); print("-"*len(h))
    disagree = []
    for name, path, fam, clean in DATASETS:
        if not os.path.exists(path): print(f"{name:13}<missing {path}>"); continue
        try: units = ADAPTERS[fam](load(path))
        except Exception as e: print(f"{name:13}<adapter error: {e}>"); continue
        rows = units if a.verbose else [u for u in units if u["unit"] in ("multicand_router",)] or units
        for u in rows:
            v, why = verdict(u, clean)
            fa = f"{u['fa_u']:.3f}" if u.get("fa_u") is not None else "  -  "
            runner = str(u.get("printed"))
            if ("beats-both" in runner.lower()) != (v == "BEATS-BOTH"):
                disagree.append((name, u["unit"], v, runner))
            print(f"{name:13}{u['unit']:20}{u['r_kga']:9.4f}{u['r_adapt']:9.4f}{u['r_freeze']:9.4f}{fa:>7}  {v:12} {why} | runner:{runner}")
    print("\n"+"="*60)
    print(f"{len(disagree)} runner-vs-uniform disagreement(s):" if disagree else "No disagreements.")
    for n, un, v, pr in disagree: print(f"  {n}/{un}: uniform={v} runner={pr}")
    print("\nNOTE: rows marked BEATS-BOTH? are POINT-only -- the JSON summary lacks per-cell arrays,")
    print("so the CI-robust test could not run. Point-beats-both is necessary, not sufficient.")
    print("Feed per-cell (acc_freeze,acc_adapt,acc_oracle,acc_kga) into recs[] to get the CI verdict.\n")

if __name__ == "__main__":
    main()
