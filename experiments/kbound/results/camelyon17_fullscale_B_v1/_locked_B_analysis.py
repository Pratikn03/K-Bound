"""
LOCKED Protocol-B eps-recalibration analysis.
Pre-registered: research_lock/CAMELYON17_FULLSCALE_PROTOCOL_B_v1.yaml
Procedure = the eps-recal in docs/research/kbound/scripts/run_wilds_camelyon17.py:decide_kga
  - split BY SEED into CAL/TEST
  - fit GradientBoostingRegressor on CAL (Z->B)
  - eps = (1-alpha) quantile of CAL residuals |Bhat - B|, alpha=0.10 FIXED, freeze
  - decide on TEST: ADAPT if Bhat-eps>0, FREEZE if Bhat+eps<0, else ABSTAIN
  - metrics on TEST: false-adapt (B<0 among ADAPT), commit/coverage, KGA regret vs trivials
  - mean +/- 95% CI over all C(n_seed,2) two-seed-CAL splits
"""
import json, itertools, numpy as np
from sklearn.ensemble import GradientBoostingRegressor

ALPHA = 0.10
ROOT = "/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8"
DEBUG = f"{ROOT}/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
NEW   = f"{ROOT}/experiments/kbound/results/camelyon17_fullscale_B_v1/wilds_camelyon17_kga.json"

def gbr():
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)

def eps_recal_by_seed(records):
    """records: list of dicts with Z, B, a0, aa, seed. Split BY SEED, all C(S,2)
    two-seed-CAL assignments. Returns per-split metrics list."""
    seeds = sorted(set(r['seed'] for r in records))
    Z = np.array([r['Z'] for r in records], float)
    B = np.array([r['B'] for r in records], float)
    a0 = np.array([r['a0'] for r in records], float)
    aa = np.array([r['aa'] for r in records], float)
    sd = np.array([r['seed'] for r in records])
    out = []
    for cal_seeds in itertools.combinations(seeds, 2):
        cal = np.isin(sd, cal_seeds); tst = ~cal
        if tst.sum() == 0 or cal.sum() < 2:
            continue
        m = gbr(); m.fit(Z[cal], B[cal])
        eps = float(np.quantile(np.abs(m.predict(Z[cal]) - B[cal]), 1 - ALPHA))
        Bhat_t = m.predict(Z[tst])
        dec = np.where(Bhat_t - eps > 0, "ADAPT",
              np.where(Bhat_t + eps < 0, "FREEZE", "ABSTAIN"))
        adapt = dec == "ADAPT"
        Bt, a0t, aat = B[tst], a0[tst], aa[tst]
        kga = np.where(adapt, aat, a0t)
        oracle = np.maximum(a0t, aat)
        out.append({
            "cal_seeds": list(cal_seeds), "eps": eps,
            "n_test": int(tst.sum()),
            "coverage": float(np.mean(dec != "ABSTAIN")),
            "commit_rate": float(adapt.mean()),
            "false_adapt": float(np.mean(Bt[adapt] < 0)) if adapt.any() else None,
            "regret_kga": float((oracle - kga).mean()),
            "regret_adapt": float((oracle - aat).mean()),
            "regret_freeze": float((oracle - a0t).mean()),
        })
    return out

def ci95(x):
    x = np.array([v for v in x if v is not None], float)
    if len(x) == 0: return (None, None, None, 0)
    m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0.0
    return (float(m), float(m - 1.96*se), float(m + 1.96*se), len(x))

def summarize(splits, key):
    return ci95([s[key] for s in splits])

# ---- DEBUG (n=256): reproduce archived eps-recal on 432-cell grid ----
dbg = json.load(open(DEBUG))
dsplits = eps_recal_by_seed(dbg['records'])
print("=== DEBUG n=256 (432-cell grid, C(4,2)=6 splits) ===")
for k in ["eps","false_adapt","commit_rate","coverage","regret_kga","regret_adapt","regret_freeze"]:
    m,lo,hi,n = summarize(dsplits, k)
    print(f"  {k:14s} mean={m:.4f} CI[{lo:.4f},{hi:.4f}] n={n}")

# ---- NEW (n=1024): only 5 per-seed records per method ----
new = json.load(open(NEW))
print(f"\n=== NEW n=1024 schema: methods={list(new['methods'])} ===")
new_results = {}
for meth in new['methods']:
    recs = new['methods'][meth]['records']
    print(f"  {meth}: {len(recs)} records, Z dim={len(recs[0]['Z'])}, fields={[k for k in recs[0] if k!='Z']}")
    nsplits = eps_recal_by_seed(recs)
    res = {}
    for k in ["eps","false_adapt","commit_rate","coverage","regret_kga","regret_adapt","regret_freeze"]:
        res[k] = summarize(nsplits, k)
    new_results[meth] = {"splits": nsplits, "summary": res}
    print(f"    splits={len(nsplits)} (C(5,2)=10 two-seed-CAL)")
    for k,v in res.items():
        m,lo,hi,n = v
        ms = f"{m:.4f}" if m is not None else "n/a"
        print(f"    {k:14s} mean={ms} CI={('['+format(lo,'.4f')+','+format(hi,'.4f')+']') if lo is not None else 'n/a'} n={n}")

# ---- KEY COMPARISON ----
fa256 = summarize(dsplits, "false_adapt")[0]
eps256 = summarize(dsplits, "eps")[0]
print("\n=== KEY COMPARISON ===")
for meth in new_results:
    fa1024 = new_results[meth]["summary"]["false_adapt"][0]
    eps1024 = new_results[meth]["summary"]["eps"][0]
    rr = eps1024/eps256 if (eps256 and eps1024 is not None) else None
    print(f"  {meth}: fa256={fa256:.4f} -> fa1024={fa1024 if fa1024 is not None else 'n/a'}; "
          f"eps256={eps256:.4f} -> eps1024={eps1024:.4f}; radius_ratio={rr:.4f} (pred 0.5)")

# Save JSON
out = {
    "protocol": "research_lock/CAMELYON17_FULLSCALE_PROTOCOL_B_v1.yaml",
    "alpha": ALPHA,
    "schema_note": (
        "DEBUG run has 432 per-cell records (72 conditions x 6 TTA candidates, 4 seeds, Z dim 11). "
        "NEW n=1024 run serializes ONLY 5 per-seed aggregate records per method (tent, eata), Z dim 10. "
        "The pre-registered 72x6 composition grid was NOT written to the new output: the eps-recal "
        "can be run on the new run only at per-seed granularity (5 points, C(5,2)=10 CAL/TEST splits, "
        "1 test-seed pair held out per split), NOT at the per-cell granularity of the debug analysis."
    ),
    "debug_n256": {k: dict(zip(["mean","ci_lo","ci_hi","n"], summarize(dsplits,k)))
                   for k in ["eps","false_adapt","commit_rate","coverage","regret_kga","regret_adapt","regret_freeze"]},
    "new_n1024": {meth: {k: dict(zip(["mean","ci_lo","ci_hi","n"], v["summary"][k]))
                          for k in v["summary"]} for meth,v in new_results.items()},
    "key_comparison": {
        meth: {
            "false_adapt_256": fa256,
            "false_adapt_1024": new_results[meth]["summary"]["false_adapt"][0],
            "eps_256": eps256,
            "eps_1024": new_results[meth]["summary"]["eps"][0],
            "radius_ratio": (new_results[meth]["summary"]["eps"][0]/eps256) if eps256 else None,
            "radius_ratio_predicted": 0.5,
        } for meth in new_results
    },
}
json.dump(out, open(f"{ROOT}/experiments/kbound/results/camelyon17_fullscale_B_v1/LOCKED_B_ANALYSIS.json","w"), indent=2)
print("\nSaved LOCKED_B_ANALYSIS.json")
