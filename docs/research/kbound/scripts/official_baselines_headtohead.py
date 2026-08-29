#!/usr/bin/env python3
"""
Head-to-head harness: KGA vs POEM / AETTA on the locked CIFAR-10-C stress stream.

The scoring + statistics layer is the reusable part: it takes a per-condition DECISION
sequence from ANY policy and computes regret-to-oracle, FA_u, decisive rate, and the KGA
regret-gap with a paired condition-bootstrap CI (Holm-corrected across the baseline family).

To close the "not official reproduction" caveat, run the authors' POEM / AETTA code,
pass the native logs through the fail-closed provenance audit and converter, and then pass
the resulting per-condition decisions here:

    python3 official_baselines_headtohead.py \
        --decisions poem=/path/poem_decisions.json aetta=/path/aetta_decisions.json

where each version-2 JSON carries a ``decisions`` map over the SAME 432 conditions and the
provenance-gate verdict. Legacy flat maps are accepted but are labelled unverified. Without
--decisions, the harness falls back to protocol-matched *ports* (clearly labelled).

No fabrication: if an external decisions file is missing a condition, it errors out.
"""
import argparse, json, os, hashlib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(REPO, "experiments", "kbound", "results")
STRESS = os.path.join(RES, "stress_persample_v1")
GBR = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
Zi = dict(pre_entropy=0, pre_conf=1, post_entropy=3, post_conf=4, entropy_drop=7)

def load(cand):
    f = os.path.join(STRESS, f"per_condition_cifar10c_{cand}_seed0.json")
    recs = json.load(open(f))["records"]
    cond = [r.get("condition","") for r in recs]
    Z  = np.array([r["Z"] for r in recs], float); B = np.array([r["B"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float); aa = np.array([r["a_adapted"] for r in recs], float)
    ao = np.array([r["a_oracle"] for r in recs], float)
    sha = hashlib.sha256(open(f,'rb').read()).hexdigest()[:12]
    return cond, Z, B, a0, aa, ao, sha

# ---- policies: return a per-condition array of 'adapt'/'freeze'/'abstain' ----
def always(v, n): return np.array([v]*n, dtype=object)

def kga_exact_rank(Z, B, alpha=0.10, k=8):
    bh = np.zeros(len(B))
    for tr,te in KFold(n_splits=k, shuffle=True, random_state=0).split(Z):
        bh[te] = GradientBoostingRegressor(**GBR).fit(Z[tr],B[tr]).predict(Z[te])
    r = np.sort(np.abs(bh-B)); n=len(r); kk=int(np.ceil((n+1)*(1-alpha))); eps = r[kk-1] if kk<=n else np.inf
    d = np.full(len(B),"abstain",dtype=object)
    if np.isfinite(eps): d[bh-eps>0]="adapt"; d[bh+eps<0]="freeze"
    return d

def poem_port(Z):   # POEM-style committal gate (port): commit while adaptation lowers entropy
    d = np.where(Z[:,Zi["entropy_drop"]] > 0, "adapt", "freeze"); return d.astype(object)
def aetta_port(Z):  # AETTA-style accuracy-proxy gate (port): adapt if adapted confidence rose
    d = np.where(Z[:,Zi["post_conf"]] > Z[:,Zi["pre_conf"]], "adapt", "freeze"); return d.astype(object)

def load_external(path, cond):
    raw = json.load(open(path))
    if not isinstance(raw, dict):
        raise SystemExit(f"external decisions must be a JSON object: {path}")
    if "decisions" in raw:
        m = raw["decisions"]
        official = bool(raw.get("official_label_allowed", False))
        label = raw.get("label", "external_protocol_adapter_unverified")
    else:
        m = raw
        official = False
        label = "legacy_external_decisions_unverified"
    if not isinstance(m, dict):
        raise SystemExit(f"external decisions payload is not an object: {path}")
    miss = [c for c in cond if c not in m]
    if miss: raise SystemExit(f"external decisions missing {len(miss)} conditions e.g. {miss[:2]} in {path}")
    extra = sorted(set(m) - set(cond))
    if extra: raise SystemExit(f"external decisions contain {len(extra)} out-of-stream conditions e.g. {extra[:2]}")
    values = np.array([m[c] for c in cond], dtype=object)
    bad = sorted(set(values) - {"adapt", "freeze", "abstain"})
    if bad: raise SystemExit(f"external decisions contain invalid actions: {bad}")
    return values, official, label

# ---- scoring ----
def regret_pc(dec, a0, aa, ao): return ao - np.where(dec=="adapt", aa, a0)
def summ(dec, B, a0, aa, ao):
    adapt = dec=="adapt"
    return dict(regret=round(float(regret_pc(dec,a0,aa,ao).mean()),4),
                FA_u=round(float(np.mean(adapt & (B<=0))),4),
                decisive=round(float(np.mean(dec!="abstain")),3),
                adapt_rate=round(float(adapt.mean()),3))
def paired_boot(rk, rb, nb=5000, seed=0):   # gap = baseline_regret - kga_regret (positive => KGA better)
    rng=np.random.default_rng(seed); n=len(rk); gaps=np.empty(nb)
    for i in range(nb):
        idx=rng.integers(0,n,n); gaps[i]=rb[idx].mean()-rk[idx].mean()
    lo,hi=np.percentile(gaps,[2.5,97.5])
    return dict(gap=round(float((rb-rk).mean()),4), ci95=[round(float(lo),4),round(float(hi),4)],
                p_better=round(float(np.mean(gaps>0)),4), ci_excludes_zero=bool(lo>0))

def holm(pvals):   # returns reject/keep at 0.05 family-wise
    order=sorted(range(len(pvals)), key=lambda i:pvals[i]); m=len(pvals); rej=[False]*m
    for rank,i in enumerate(order):
        if pvals[i] <= 0.05/(m-rank): rej[i]=True
        else: break
    return rej

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate", default="tent")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--decisions", nargs="*", default=[], help="name=path.json for OFFICIAL baseline decisions")
    ap.add_argument("--out", default=os.path.join(RES,"official_headtohead.json"))
    a=ap.parse_args()
    cond,Z,B,a0,aa,ao,sha = load(a.candidate); n=len(B)
    ext = {kv.split("=",1)[0]: load_external(kv.split("=",1)[1], cond) for kv in a.decisions}

    kga = kga_exact_rank(Z,B,a.alpha); rk = regret_pc(kga,a0,aa,ao)
    policies = {"always_adapt":always("adapt",n), "always_freeze":always("freeze",n),
                "oracle":np.where(B>0,"adapt","freeze").astype(object)}
    # official if provided, else labelled ports
    if "poem" in ext:
        poem_decisions, poem_official, _ = ext["poem"]
        policies["POEM_official_adapter" if poem_official else "POEM_external_unverified"] = poem_decisions
    else:
        policies["POEM_port"] = poem_port(Z)
    if "aetta" in ext:
        aetta_decisions, aetta_official, _ = ext["aetta"]
        policies["AETTA_official_adapter" if aetta_official else "AETTA_external_unverified"] = aetta_decisions
    else:
        policies["AETTA_port"] = aetta_port(Z)

    rows={"KGA":{**summ(kga,B,a0,aa,ao),"gap_vs_KGA":"---","holm_beats":"---"}}
    compare=[k for k in policies if k not in ("oracle",)]
    boots={k:paired_boot(rk, regret_pc(policies[k],a0,aa,ao)) for k in compare}
    rej = holm([1-boots[k]["p_better"] for k in compare])  # crude p ~ 1-p_better
    for k in policies:
        s=summ(policies[k],B,a0,aa,ao)
        s["holm_beats"]= (dict(zip(compare,rej)).get(k, False)) if k in compare else "---"
        s["gap_vs_KGA"]= boots[k] if k in compare else "---"
        rows[k]=s
    out=dict(candidate=a.candidate, alpha=a.alpha, n_conditions=n, input_sha12=sha,
             official=[k for k, value in ext.items() if value[1]],
             external_labels={k: value[2] for k, value in ext.items()},
             note=("POEM/AETTA are protocol-matched ports unless a converted decision artifact "
                   "carries a passing fail-closed provenance audit."),
             rows=rows)
    json.dump(out, open(a.out,"w"), indent=2); print("wrote", a.out)
    print(f"\n{'policy':16s} {'regret':>8s} {'FA_u':>6s} {'decisive':>8s} {'gap[CI]':>22s} holm")
    for k,s in rows.items():
        g=s["gap_vs_KGA"]
        gs = f"{g['gap']:.4f} {g['ci95']}" if isinstance(g,dict) else "---"
        dec = str(s.get('decisive',''))
        print(f"{k:16s} {s['regret']:>8.4f} {s['FA_u']:>6.3f} {dec:>8s} {gs:>24s} {s['holm_beats']}")

if __name__=="__main__": main()
