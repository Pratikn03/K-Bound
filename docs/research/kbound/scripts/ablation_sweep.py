#!/usr/bin/env python3
"""
K-Bound ablations from LOGGED per-condition evidence (no model re-run, no T9).

Reproduces the deployed KGA rule exactly:
  * leave-one-cell-out (LOCO) gradient-boosted benefit estimate  Bhat_{-i}(Z_i)
  * conformal radius  eps = (1-alpha) empirical quantile of LOO residuals |Bhat_{-j}-B_j|
  * decision: adapt if Bhat-eps>0 ; freeze if Bhat+eps<0 ; else abstain
Metrics: regret-to-oracle, unconditional false-adapt FA_u=Pr(adapt & B<=0),
conditional FA_c, adapt-rate, decision coverage.

Blocks:
  alpha      - alpha sweep {0.01,0.05,0.10,0.20} x {tent,eata,sar}
  estimator  - GBR vs Ridge vs RandomForest vs MLP  (tent, alpha=0.10)
  dropout    - drop each evidence family            (tent, alpha=0.10)
  transfer   - fit on one adapter, apply to another (alpha=0.10)
Anchor: 'alpha' block prints the tent alpha=0.10 row to compare against the
locked gate table (tab:gates: regret 0.0017, FA_u 0, adapt 0.51, coverage 0.68).
"""
import json, sys, time, glob, os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The ONE radius rule (fix-queue items 2, 4, 15) and the ranked input resolver
# that fix-queue item 8 added to ablation_exactrank.py -- reused rather than
# re-forked, because this file had exactly the same two defects.
from kbound_decide import conformal_radius as _conformal_radius  # noqa: E402
from ablation_exactrank import _resolve as _resolve_percell      # noqa: E402
from kbound_decide import read_json as _read_json                # noqa: E402
from kbound_decide import results_root as _results_root          # noqa: E402

# Ablation uses 8-fold cross-fitting as a fast, faithful proxy for the deployed
# leave-one-cell-out radius (each fold trains on 7/8 of the cells). The 'alpha'
# block prints an anchor row to confirm it reproduces the locked LOCO gate table.
CROSSFIT_K = 8

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "kbound", "results")
Z_NAMES = ["pre_entropy","pre_conf","pre_pbal","post_entropy","post_conf","post_pbal",
           "pbal_drop","entropy_drop","frac_highconf","marginal_KL","update_norm"]
FAMILIES = {  # evidence-family -> feature indices (for dropout ablation)
    "frozen":  [0,1,2],
    "adapted": [3,4,5,8],
    "change":  [6,7],
    "drift":   [9],
    "update":  [10],
}

def gbr():   return GradientBoostingRegressor(n_estimators=250, max_depth=2,
                                              learning_rate=0.05, subsample=0.8, random_state=0)
def ridge(): return Ridge(alpha=1.0, random_state=0)
def rf():    return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1)
def mlp():   return MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500, random_state=0)

def load(cand, seed=1, root=None):
    """FIX-QUEUE ITEM 8 -- this loader could not run as released, for two reasons.

    (1) It read ``<scripts>/../experiments/kbound/results/
        per_condition_cifar10c_{cand}_seed0.json``.  No such file exists there or
        anywhere else in the tree; ``stress_grid_multiseed_v1/seed0/`` has no
        per-condition dump at all.  It now uses the same ranked resolver
        ``ablation_exactrank._resolve`` (default seed 1) and fails with the list
        of paths it tried instead of a bare FileNotFoundError.
    (2) It read ``r["a_oracle"]``, which is absent from EVERY committed 432-cell
        dump, so repointing alone still crashed.  ``a_oracle`` is DEFINED as
        ``max(a0, a_adapted)``; it is derived when absent and the substitution is
        announced on stdout rather than done silently.
    """
    root = root or _results_root()
    f, ncells = _resolve_percell(cand, seed, root, allow_270=False)
    recs = _read_json(f)["records"]          # names the file if it is a placeholder
    Z  = np.array([r["Z"] for r in recs], float)
    B  = np.array([r["B"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    if all("a_oracle" in r for r in recs):
        ao = np.array([r["a_oracle"] for r in recs], float)
    else:
        ao = np.maximum(a0, aa)
        print(f"[ablation_sweep] {cand}: 'a_oracle' absent from {os.path.basename(f)}; "
              f"derived as max(a0, a_adapted).")
    print(f"[ablation_sweep] {cand}: {len(recs)} cells (grid nominal {ncells}) <- {f}")
    return Z, B, a0, aa, ao

def loco_bhat(Z, B, factory, cols=None):
    """Out-of-fold predictions via K-fold cross-fitting (proxy for leave-one-cell-out)."""
    if cols is not None: Z = Z[:, cols]
    n = len(B); bh = np.zeros(n)
    for tr, te in KFold(n_splits=CROSSFIT_K, shuffle=True, random_state=0).split(Z):
        est = factory().fit(Z[tr], B[tr])
        bh[te] = est.predict(Z[te])
    return bh

def decide(bhat, eps):
    d = np.full(len(bhat), "abstain", dtype=object)
    d[bhat - eps > 0] = "adapt"
    d[bhat + eps < 0] = "freeze"
    return d

def metrics(d, B, a0, aa, ao):
    a_dec = np.where(d == "adapt", aa, a0)          # freeze/abstain -> frozen accuracy
    regret = float(np.mean(ao - a_dec))
    adapt = d == "adapt"
    fa_u = float(np.mean(adapt & (B <= 0)))
    fa_c = float(np.mean(B[adapt] <= 0)) if adapt.any() else 0.0
    return dict(regret=round(regret,4), FA_u=round(fa_u,4), FA_c=round(fa_c,4),
                adapt_rate=round(float(adapt.mean()),3),
                coverage=round(float((d!="abstain").mean()),3),
                n=int(len(B)), harmful_frac=round(float((B<=0).mean()),3))

def eps_q(resid, alpha):
    """FIX-QUEUE ITEMS 2 + 4 + 15 -- one radius rule, computed by the shipped library.

    Was ``float(np.quantile(resid, 1 - alpha))``: numpy's linear interpolation
    between order statistics, which is not an observed residual and does not
    satisfy the finite-sample rank argument.  Now the exact split-conformal rank
    quantile ``r_(k)``, ``k = ceil((n+1)(1-alpha))``, from
    ``kbound_decide.conformal_radius`` -> ``kga.certificate``.

    NOTE on the pool: within a block this radius is in-pool (all n residuals,
    including the scored cell's own), which is fix-queue item 4.  ``run_transfer``
    is exempt -- its radius comes from the SOURCE adapter's residuals and is
    applied to a different adapter's cells, so it is a genuine held-out radius.
    For the leave-one-out-of-pool version of the alpha / estimator / dropout
    blocks use ``ablation_exactrank.py --calibration loo``, which is the script
    this one was cloned from.
    """
    return float(_conformal_radius(np.abs(np.asarray(resid, float)), alpha))

def run_alpha(cands=("tent","eata","sar")):
    out = {}
    for cand in cands:
        Z,B,a0,aa,ao = load(cand)
        bh = loco_bhat(Z, B, gbr)
        resid = np.abs(bh - B)
        out[cand] = {}
        for a in (0.01,0.05,0.10,0.20):
            eps = eps_q(resid, a)
            out[cand][f"alpha={a}"] = metrics(decide(bh, eps), B,a0,aa,ao)
        # radius-free anchor (eps=0)
        out[cand]["no_radius"] = metrics(decide(bh, 0.0), B,a0,aa,ao)
    return out

def run_estimator():
    Z,B,a0,aa,ao = load("tent"); out={}
    for name,fac in (("GBR",gbr),("Ridge_linear",ridge),("RandomForest",rf),("MLP",mlp)):
        bh = loco_bhat(Z,B,fac); resid=np.abs(bh-B)
        out[name] = metrics(decide(bh, eps_q(resid,0.10)), B,a0,aa,ao)
    return out

def run_dropout():
    Z,B,a0,aa,ao = load("tent"); out={}
    full=list(range(11))
    bh = loco_bhat(Z,B,gbr,full); out["full_11"]=metrics(decide(bh,eps_q(np.abs(bh-B),0.10)),B,a0,aa,ao)
    for fam,cols in FAMILIES.items():
        keep=[c for c in full if c not in cols]
        bh = loco_bhat(Z,B,gbr,keep)
        out[f"drop_{fam}"]=metrics(decide(bh,eps_q(np.abs(bh-B),0.10)),B,a0,aa,ao)
    return out

def run_transfer():
    data={c:load(c) for c in ("tent","eata","sar")}
    out={}
    for src in ("tent","eata","sar"):
        Zs,Bs,_,_,_ = data[src]
        eps = eps_q(np.abs(loco_bhat(Zs,Bs,gbr)-Bs), 0.10)  # radius from source LOO
        est = gbr().fit(Zs,Bs)                               # estimator on full source
        for tgt in ("tent","eata","sar"):
            if tgt==src: continue
            Zt,Bt,a0,aa,ao = data[tgt]
            out[f"{src}->{tgt}"]=metrics(decide(est.predict(Zt),eps),Bt,a0,aa,ao)
    return out

BLOCKS={"alpha":run_alpha,"estimator":run_estimator,"dropout":run_dropout,"transfer":run_transfer}
if __name__=="__main__":
    block=sys.argv[1] if len(sys.argv)>1 else "all"
    tag=block; res={}
    if block=="alpha" and len(sys.argv)>2:
        cands=tuple(sys.argv[2].split(",")); tag="alpha_"+"_".join(cands)
        t=time.time(); res["alpha"]=run_alpha(cands); print(f"[alpha {cands}] {time.time()-t:.1f}s", flush=True)
    else:
        todo=list(BLOCKS) if block=="all" else [block]
        for b in todo:
            t=time.time(); res[b]=BLOCKS[b](); print(f"[{b}] {time.time()-t:.1f}s", flush=True)
    path=os.path.join(RESULTS_DIR, f"ablation_{tag}.json")
    json.dump(res, open(path,"w"), indent=2)
    print("WROTE", path)
    print(json.dumps(res, indent=2))
