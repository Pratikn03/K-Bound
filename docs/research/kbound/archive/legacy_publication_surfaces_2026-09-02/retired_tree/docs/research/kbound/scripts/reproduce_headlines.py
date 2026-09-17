#!/usr/bin/env python3
"""
Reproduce/verify the CI-confirmed K-Bound headlines from committed artifacts. Exit 0 iff all PASS.

  1. CIFAR-10-C Tent  beats-both : rebuilt from per_condition_cifar10c_tent_seed0.json (exact-rank KGA)
  2. CIFAR-10-C EATA  beats-both : rebuilt from per_condition_cifar10c_eata_seed0.json
  3. Decision-gate certificate    : gate_comparison.json -> KGA certificate FA_u == 0
  4. Source-of-truth traceability : headline numbers present in results_source.json

Rebuilding (1)-(2) from the raw per-condition logs is the strong check: the beats-both verdict is
recomputed, not read back. No fabrication; missing artifacts are reported as SKIP, not PASS.
"""
import json, math, os, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
K = os.path.join(REPO, "docs", "research", "kbound")
RES = os.path.join(REPO, "experiments", "kbound", "results")
DECISIVE_RESULTS = os.path.join(
    RES, "win_hunt_v5", "imagenetc_aggr", "decisive_tta_results.json"
)
GBR = dict(n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)

def kga_regret(cand, alpha=0.10):
    f = os.path.join(RES, f"per_condition_cifar10c_{cand}_seed0.json")
    if not os.path.exists(f): return None
    with open(f, encoding="utf-8") as stream:
        recs = json.load(stream)["records"]
    Z=np.array([r["Z"] for r in recs],float); B=np.array([r["B"] for r in recs],float)
    a0=np.array([r["a0"] for r in recs],float); aa=np.array([r["a_adapted"] for r in recs],float)
    ao=np.array([r["a_oracle"] for r in recs],float)
    bh=np.zeros(len(B))
    for tr,te in KFold(8,shuffle=True,random_state=0).split(Z):
        bh[te]=GradientBoostingRegressor(**GBR).fit(Z[tr],B[tr]).predict(Z[te])
    r=np.sort(np.abs(bh-B)); n=len(r); kk=int(np.ceil((n+1)*(1-alpha))); eps=r[kk-1] if kk<=n else np.inf
    d=np.full(len(B),"abstain",dtype=object)
    if np.isfinite(eps): d[bh-eps>0]="adapt"; d[bh+eps<0]="freeze"
    reg=lambda dec: float((ao-np.where(dec=="adapt",aa,a0)).mean())
    return dict(kga=reg(d), adapt=reg(np.array(["adapt"]*len(B))), freeze=reg(np.array(["freeze"]*len(B))),
                FA_u=float(np.mean((d=="adapt") & (B<=0))))

def flat(o, acc):
    if isinstance(o,dict):
        for v in o.values(): flat(v,acc)
    elif isinstance(o,list):
        for v in o: flat(v,acc)
    elif isinstance(o,(int,float)): acc.append(round(float(o),4))

def gate_certificate_has_zero_false_adapt(payload):
    """Validate the named certificate metrics; unrelated zeroes are not evidence."""
    if not isinstance(payload, dict):
        return False
    gates = payload.get("gates")
    certificate = gates.get("KGA (certificate)") if isinstance(gates, dict) else None
    metrics = certificate.get("all") if isinstance(certificate, dict) else None
    if not isinstance(metrics, dict):
        return False
    rate = metrics.get("FA_u")
    count = metrics.get("n_false_adapt")
    n = metrics.get("n")
    return bool(
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and math.isfinite(float(rate))
        and float(rate) == 0.0
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count == 0
        and isinstance(n, int)
        and not isinstance(n, bool)
        and n > 0
    )

def main():
    checks=[]
    for cand in ("tent","eata"):
        r=kga_regret(cand)
        if r is None: checks.append((f"CIFAR-10-C {cand.upper()} beats-both (raw rebuild)","SKIP","artifact missing")); continue
        ok = r["kga"]<r["adapt"] and r["kga"]<r["freeze"] and r["FA_u"]==0
        checks.append((f"CIFAR-10-C {cand.upper()} beats-both (raw rebuild)","PASS" if ok else "FAIL",
                       f"KGA {r['kga']:.4f} < adapt {r['adapt']:.4f}, freeze {r['freeze']:.4f}; FA_u {r['FA_u']:.3f}"))
    gc=os.path.join(K,"gate_comparison.json")
    if os.path.exists(gc):
        with open(gc, encoding="utf-8") as stream:
            g=json.load(stream)
        ok = gate_certificate_has_zero_false_adapt(g)
        checks.append(("Decision-gate certificate FA_u=0","PASS" if ok else "FAIL", os.path.basename(gc)))
    else:
        checks.append(("Decision-gate certificate FA_u=0","SKIP","gate_comparison.json missing"))
    import glob as _g
    source_files=[os.path.join(K,"results_source.json"),
                  os.path.join(REPO,"research_lock","KBOUND_MIXED_STREAM_v2.json"),
                  DECISIVE_RESULTS]
    for base in (K, os.path.join(REPO,"audits")):
        source_files += _g.glob(os.path.join(base,"**","recon_results.json"), recursive=True)
        source_files += _g.glob(os.path.join(base,"**","benchmark_verdicts.json"), recursive=True)
    acc=set()
    for p in source_files:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as stream:
                    payload = json.load(stream)
                a=[]; flat(payload,a); acc|=set(a)
            except Exception: pass
    want={"ImageNet-C SAR":[0.0108,0.0625,0.0319],"three-source mixture":[0.0059,0.0632,0.0342],
          "Camelyon17 OOD":[0.1381]}
    for name,vals in want.items():
        hit=all(v in acc for v in vals)
        checks.append((f"committed artifacts have {name}","PASS" if hit else "FAIL", str(vals)))

    print(f"{'check':44s} {'result':6s} detail")
    for name,res,det in checks: print(f"{name:44s} {res:6s} {det}")
    npass=sum(1 for _,r,_ in checks if r=="PASS"); nfail=sum(1 for _,r,_ in checks if r=="FAIL")
    nskip=sum(1 for _,r,_ in checks if r=="SKIP")
    print(f"\n{npass} PASS, {nfail} FAIL, {nskip} SKIP")
    sys.exit(1 if nfail or nskip else 0)

if __name__=="__main__": main()
