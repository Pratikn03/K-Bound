#!/usr/bin/env python3
"""
K-Bound sensitivity ablations with the EXACT-RANK (order-statistic) conformal radius.

Radius rule (finite-sample split/jackknife conformal):
    sort residuals r_(1) <= ... <= r_(n);  k = ceil((n+1)(1-alpha));  eps = r_(k)  (eps=+inf if k>n).
This is the exact order-statistic quantile, NOT numpy's interpolated percentile.

Out-of-fold Bhat via K-fold cross-fitting (fast, faithful proxy for leave-one-cell-out).
Decision: adapt if Bhat-eps>0 ; freeze if Bhat+eps<0 ; else abstain.
Metrics: regret-to-oracle, FA_u=Pr(adapt & B<=0), FA_c, adapt-rate, decision coverage.

Outputs (locked): experiments/kbound/results/ablation_exactrank.json  (+ config hash, per-condition baseline).
Blocks: alpha sweep, estimator swap (GBR/Ridge/RF/MLP), evidence-family dropout, cross-adapter transfer.
Anchor: prints Tent alpha=0.10 vs the locked gate table (tab:gates: regret 0.0017, FA_u 0).

FIX-QUEUE ITEM 8 -- this script could not run as released.
    ``load()`` read ``<scripts>/../experiments/kbound/results/per_condition_cifar10c_
    {cand}_seed0.json``.  No such file exists there or anywhere else: the stress
    grid's ``seed0/`` directory contains only ``decisive_tta_results.json``,
    ``decisive_tta_table.md`` and ``result_manifest.json``.  It now searches a
    ranked list of real locations (``--input-root`` / ``--seed`` override it) and
    fails with the list of paths it tried instead of a bare FileNotFoundError.

    Second, separate defect: ``load()`` also read ``r['a_oracle']``, which is
    absent from EVERY committed 432-cell dump (both the stress grid and the
    head-to-head tree).  Repointing alone therefore still crashed.  ``a_oracle``
    is now DERIVED as ``max(a0, a_adapted)`` when absent -- which is its
    definition -- and the substitution is recorded in the output under
    ``config.a_oracle_source`` rather than being done silently.

FIX-QUEUE ITEM 4 -- the radius no longer includes the scored cell.
    Lines 88-89 used ``eps_exact_rank(resid, a)`` over the whole residual vector
    and then scored the same cells.  Radii are now leave-one-out-of-pool.
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kbound_decide import read_json, repo_path, results_root  # noqa: E402

RES = os.path.join(HERE, "..", "experiments", "kbound", "results")   # where OUTPUT is written
CROSSFIT_K = 8
GBR_CFG = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
FAMILIES = {"frozen":[0,1,2], "adapted":[3,4,5,8], "change":[6,7], "drift":[9], "update":[10]}

def gbr():   return GradientBoostingRegressor(**GBR_CFG)
def ridge(): return Ridge(alpha=1.0)
def rf():    return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)
def mlp():   return MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=0)

# Ranked search order for the per-cell input.  Every entry is a real path in the
# release; the first that exists wins.  {cand} in {tent, eata, sar}, {seed} an int.
INPUT_CANDIDATES = [
    # 432-cell stress grid, seeds 1-4 (seed 0 has no per-condition dump -- F4-7).
    ("stress_grid_multiseed_v1/seed{seed}/per_condition_cifar10c_{cand}_seed{seed}.json", 432),
    # 432-cell head-to-head tree (tent/eata only; no sar arm exists there).
    ("mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed{seed}.json", 432),
    ("mixed_headtohead_v1/per_condition_cifar10c_eata_secondary_kga_seed{seed}.json", 432),
    # 270-cell aggressive grid: DIFFERENT GRID. Only used if explicitly allowed.
    ("win_hunt_v5/cifar10c_aggr/seed{seed}/per_condition_cifar10c_{cand}_seed{seed}.json", 270),
]
_H2H_ARM = {"tent": "tent_primary_kga", "eata": "eata_secondary_kga"}

A_ORACLE_SOURCE = {}   # cand -> "file" | "derived: max(a0, a_adapted)"


def _resolve(cand, seed, root, allow_270=False):
    tried = []
    for pat, ncells in INPUT_CANDIDATES:
        if ncells == 270 and not allow_270:
            continue
        if "headtohead" in pat:
            if cand not in _H2H_ARM or _H2H_ARM[cand] not in pat:
                continue
            p = os.path.join(root, pat.format(seed=seed))
        else:
            p = os.path.join(root, pat.format(cand=cand, seed=seed))
        tried.append(p)
        if os.path.exists(p):
            return p, ncells
    raise FileNotFoundError(
        f"No per-cell CIFAR-10-C dump for candidate={cand!r} seed={seed}.\n"
        "Tried, in order:\n  " + "\n  ".join(tried) + "\n"
        "  -> pass --input-root / --seed, or see docs/research/kbound/STORAGE_MANIFEST.json.\n"
        "  -> NOTE: stress_grid_multiseed_v1/seed0/ has no per-condition dump at all;\n"
        "     seed 0 is only recoverable from LOCKED_ANALYSIS_RESULTS.json aggregates."
    )


def load(cand, seed=1, root=None, allow_270=False):
    root = root or results_root()
    f, ncells = _resolve(cand, seed, root, allow_270=allow_270)
    recs = read_json(f)["records"]
    Z  = np.array([r["Z"] for r in recs], float)
    B  = np.array([r["B"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    # a_oracle is absent from every committed 432-cell dump (fix-queue item 8).
    # It is DEFINED as max(a0, a_adapted), so derive it -- but say so out loud.
    if all("a_oracle" in r for r in recs):
        ao = np.array([r["a_oracle"] for r in recs], float)
        A_ORACLE_SOURCE[cand] = "file"
    else:
        ao = np.maximum(a0, aa)
        A_ORACLE_SOURCE[cand] = "derived: max(a0, a_adapted) -- 'a_oracle' absent from this dump"
        print(f"[ablation] {cand}: 'a_oracle' absent from {os.path.relpath(f, root)}; "
              f"derived as max(a0, a_adapted).")
    sha = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12]
    print(f"[ablation] {cand}: {len(recs)} cells (grid nominal {ncells}) <- {f}")
    return Z, B, a0, aa, ao, sha

def oof_bhat(Z, B, factory, cols=None):
    if cols is not None: Z = Z[:, cols]
    bh = np.zeros(len(B))
    for tr, te in KFold(n_splits=CROSSFIT_K, shuffle=True, random_state=0).split(Z):
        bh[te] = factory().fit(Z[tr], B[tr]).predict(Z[te])
    return bh

def eps_exact_rank(resid, alpha):
    """Exact order-statistic radius: eps = r_(k), k=ceil((n+1)(1-alpha))."""
    r = np.sort(np.asarray(resid, float)); n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(r[k-1]) if k <= n else float("inf")   # k>n => no finite radius (never adapt)

def eps_loo(resid, alpha):
    """Leave-one-out-of-pool radii (fix-queue item 4): cell i's radius is the
    exact-rank quantile of the OTHER n-1 residuals, so eps is never a function of
    the label of the cell it is used to score.  Returns one radius per cell."""
    r = np.asarray(resid, float); n = len(r)
    return np.array([eps_exact_rank(np.delete(r, i), alpha) for i in range(n)], float)

def decide(bh, eps):
    """eps may be a scalar or a per-cell array; non-finite radius => abstain."""
    bh = np.asarray(bh, float)
    e = np.broadcast_to(np.asarray(eps, float), bh.shape)
    d = np.full(len(bh), "abstain", dtype=object)
    fin = np.isfinite(e)
    d[fin & (bh - e > 0)] = "adapt"
    d[fin & (bh + e < 0)] = "freeze"
    return d

def metrics(d, B, a0, aa, ao):
    a_dec = np.where(d == "adapt", aa, a0)
    adapt = d == "adapt"
    return dict(regret=round(float(np.mean(ao - a_dec)),4),
                FA_u=round(float(np.mean(adapt & (B <= 0))),4),
                FA_c=round(float(np.mean(B[adapt] <= 0)) if adapt.any() else 0.0,4),
                adapt_rate=round(float(adapt.mean()),3),
                coverage=round(float((d!="abstain").mean()),3),
                n=int(len(B)))

def _eps(resid, alpha, calibration):
    """One radius rule, one place. 'loo' => per-cell; 'in_pool' => the old scalar."""
    return eps_loo(resid, alpha) if calibration == "loo" else eps_exact_rank(resid, alpha)


def run(seed=1, root=None, allow_270=False, calibration="loo", cands=("tent","eata","sar")):
    data = {c: load(c, seed=seed, root=root, allow_270=allow_270) for c in cands}
    out = {"config": {"radius":"exact_rank eps=r_(ceil((n+1)(1-alpha)))",
                      "calibration": calibration,
                      "calibration_note":
                          "loo = leave-one-out-of-pool: cell i's radius excludes cell i's "
                          "own residual (fix-queue item 4). in_pool reproduces the archived "
                          "rule in which eps was a function of the labels it scored.",
                      "crossfit_folds":CROSSFIT_K, "gbr":GBR_CFG,
                      "input_seed": seed,
                      "n_cells": {c: int(len(data[c][1])) for c in data},
                      "a_oracle_source": dict(A_ORACLE_SOURCE),
                      "input_sha12":{c:data[c][5] for c in data}},
           "alpha":{}, "estimator":{}, "dropout":{}, "transfer":{}}
    # (i) alpha sweep — Bhat computed once per candidate (GBR), only eps changes with alpha
    bh_cache = {}
    for c in cands:
        Z,B,a0,aa,ao,_ = data[c]; bh = oof_bhat(Z,B,gbr); bh_cache[c]=bh
        resid = np.abs(bh - B); out["alpha"][c] = {}
        for a in (0.01,0.05,0.10,0.20):
            out["alpha"][c][f"alpha={a}"] = metrics(decide(bh, _eps(resid,a,calibration)), B,a0,aa,ao)
        out["alpha"][c]["no_radius"] = metrics(decide(bh, 0.0), B,a0,aa,ao)
    # (ii) estimator swap (tent, alpha=0.10)
    Z,B,a0,aa,ao,_ = data[cands[0]]
    for name,fac in (("GBR",gbr),("Ridge_linear",ridge),("RandomForest",rf),("MLP",mlp)):
        bh = oof_bhat(Z,B,fac); out["estimator"][name] = metrics(decide(bh, _eps(np.abs(bh-B),0.10,calibration)), B,a0,aa,ao)
    # (iii) evidence-family dropout (tent, alpha=0.10, GBR)
    full=list(range(Z.shape[1])); bh=oof_bhat(Z,B,gbr,full)
    out["dropout"][f"full_{len(full)}"]=metrics(decide(bh,_eps(np.abs(bh-B),0.10,calibration)),B,a0,aa,ao)
    for fam,cols in FAMILIES.items():
        keep=[i for i in full if i not in cols]; bh=oof_bhat(Z,B,gbr,keep)
        out["dropout"][f"drop_{fam}"]=metrics(decide(bh,_eps(np.abs(bh-B),0.10,calibration)),B,a0,aa,ao)
    # (iv) cross-adapter transfer (alpha=0.10): fit on source, radius from source OOF, apply to target.
    # The transfer radius is a genuine held-out radius (it comes from the SOURCE
    # adapter's residuals, never from the target cells being scored), so the
    # in-pool defect never applied here -- one scalar is correct.
    for src in cands:
        Zs,Bs,_,_,_,_ = data[src]; eps = eps_exact_rank(np.abs(oof_bhat(Zs,Bs,gbr)-Bs),0.10)
        est = gbr().fit(Zs,Bs)
        for tgt in cands:
            if tgt==src: continue
            Zt,Bt,a0t,aat,aot,_ = data[tgt]
            out["transfer"][f"{src}->{tgt}"]=metrics(decide(est.predict(Zt),eps),Bt,a0t,aat,aot)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1,
                    help="input seed. DEFAULT 1: the stress grid's seed0/ has no "
                         "per-condition dump, so seed 0 cannot be used (fix-queue item 8).")
    ap.add_argument("--input-root", default=None,
                    help="results tree holding the per-cell dumps (default: repo "
                         "experiments/kbound/results, or $KBOUND_RESULTS_ROOT)")
    ap.add_argument("--calibration", default="loo", choices=["loo","in_pool"])
    ap.add_argument("--allow-270-cell-grid", action="store_true",
                    help="permit falling back to win_hunt_v5/cifar10c_aggr (270 cells, a "
                         "DIFFERENT grid from the 432-cell stress grid the tables report)")
    ap.add_argument("--out", default=os.path.join(RES,"ablation_exactrank.json"))
    args = ap.parse_args()
    t=time.time()
    out=run(seed=args.seed, root=args.input_root, allow_270=args.allow_270_cell_grid,
            calibration=args.calibration)
    path=args.out; os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out,open(path,"w"),indent=2)
    print(f"[done {time.time()-t:.1f}s] wrote {path}")
    a=out["alpha"]["tent"]
    print("ANCHOR tent alpha=0.10 (locked gate: regret 0.0017, FA_u 0):", a["alpha=0.1"])
    print("alpha sweep tent:", {k:(v["regret"],v["FA_u"],v["coverage"]) for k,v in a.items()})
    print("estimator:", {k:(v["regret"],v["FA_u"],v["coverage"]) for k,v in out["estimator"].items()})
    print("dropout:", {k:(v["regret"],v["FA_u"]) for k,v in out["dropout"].items()})
    print("transfer:", {k:(v["regret"],v["FA_u"]) for k,v in out["transfer"].items()})
