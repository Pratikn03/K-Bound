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
"""
import json, os, time, hashlib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "experiments", "kbound", "results")
CROSSFIT_K = 8
GBR_CFG = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
FAMILIES = {"frozen":[0,1,2], "adapted":[3,4,5,8], "change":[6,7], "drift":[9], "update":[10]}

def gbr():   return GradientBoostingRegressor(**GBR_CFG)
def ridge(): return Ridge(alpha=1.0)
def rf():    return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)
def mlp():   return MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=0)

def load(cand):
    f = os.path.join(RES, f"per_condition_cifar10c_{cand}_seed0.json")
    recs = json.load(open(f))["records"]
    Z  = np.array([r["Z"] for r in recs], float)
    B  = np.array([r["B"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    ao = np.array([r["a_oracle"] for r in recs], float)
    sha = hashlib.sha256(open(f,'rb').read()).hexdigest()[:12]
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

def decide(bh, eps):
    d = np.full(len(bh), "abstain", dtype=object)
    if np.isfinite(eps):
        d[bh - eps > 0] = "adapt"; d[bh + eps < 0] = "freeze"
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

def run():
    data = {c: load(c) for c in ("tent","eata","sar")}
    out = {"config": {"radius":"exact_rank eps=r_(ceil((n+1)(1-alpha)))",
                      "crossfit_folds":CROSSFIT_K, "gbr":GBR_CFG,
                      "input_sha12":{c:data[c][5] for c in data}},
           "alpha":{}, "estimator":{}, "dropout":{}, "transfer":{}}
    # (i) alpha sweep — Bhat computed once per candidate (GBR), only eps changes with alpha
    bh_cache = {}
    for c in ("tent","eata","sar"):
        Z,B,a0,aa,ao,_ = data[c]; bh = oof_bhat(Z,B,gbr); bh_cache[c]=bh
        resid = np.abs(bh - B); out["alpha"][c] = {}
        for a in (0.01,0.05,0.10,0.20):
            out["alpha"][c][f"alpha={a}"] = metrics(decide(bh, eps_exact_rank(resid,a)), B,a0,aa,ao)
        out["alpha"][c]["no_radius"] = metrics(decide(bh, 0.0), B,a0,aa,ao)
    # (ii) estimator swap (tent, alpha=0.10)
    Z,B,a0,aa,ao,_ = data["tent"]
    for name,fac in (("GBR",gbr),("Ridge_linear",ridge),("RandomForest",rf),("MLP",mlp)):
        bh = oof_bhat(Z,B,fac); out["estimator"][name] = metrics(decide(bh, eps_exact_rank(np.abs(bh-B),0.10)), B,a0,aa,ao)
    # (iii) evidence-family dropout (tent, alpha=0.10, GBR)
    full=list(range(11)); bh=oof_bhat(Z,B,gbr,full)
    out["dropout"]["full_11"]=metrics(decide(bh,eps_exact_rank(np.abs(bh-B),0.10)),B,a0,aa,ao)
    for fam,cols in FAMILIES.items():
        keep=[i for i in full if i not in cols]; bh=oof_bhat(Z,B,gbr,keep)
        out["dropout"][f"drop_{fam}"]=metrics(decide(bh,eps_exact_rank(np.abs(bh-B),0.10)),B,a0,aa,ao)
    # (iv) cross-adapter transfer (alpha=0.10): fit on source, radius from source OOF, apply to target
    for src in ("tent","eata","sar"):
        Zs,Bs,_,_,_,_ = data[src]; eps = eps_exact_rank(np.abs(oof_bhat(Zs,Bs,gbr)-Bs),0.10)
        est = gbr().fit(Zs,Bs)
        for tgt in ("tent","eata","sar"):
            if tgt==src: continue
            Zt,Bt,a0t,aat,aot,_ = data[tgt]
            out["transfer"][f"{src}->{tgt}"]=metrics(decide(est.predict(Zt),eps),Bt,a0t,aat,aot)
    return out

if __name__ == "__main__":
    t=time.time(); out=run()
    path=os.path.join(RES,"ablation_exactrank.json"); json.dump(out,open(path,"w"),indent=2)
    print(f"[done {time.time()-t:.1f}s] wrote {path}")
    a=out["alpha"]["tent"]
    print("ANCHOR tent alpha=0.10 (locked gate: regret 0.0017, FA_u 0):", a["alpha=0.1"])
    print("alpha sweep tent:", {k:(v["regret"],v["FA_u"],v["coverage"]) for k,v in a.items()})
    print("estimator:", {k:(v["regret"],v["FA_u"],v["coverage"]) for k,v in out["estimator"].items()})
    print("dropout:", {k:(v["regret"],v["FA_u"]) for k,v in out["dropout"].items()})
    print("transfer:", {k:(v["regret"],v["FA_u"]) for k,v in out["transfer"].items()})
