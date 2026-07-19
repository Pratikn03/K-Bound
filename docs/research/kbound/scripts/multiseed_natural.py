#!/usr/bin/env python3
"""
Multi-seed no-harm summary for a natural-shift track, computed from the committed per-seed
per-condition logs (no GPU, no raw data). For each seed it reads the deployed decision
(a_kbound / kga_decision) already stored in the log and computes regret-to-oracle for KGA,
always-adapt, and always-freeze, plus FA_u; then it aggregates across seeds.

Usage:
  python3 multiseed_natural.py --dataset camelyon17 --candidate eata \
      --dir experiments/kbound/results/wilds_kbound
"""
import argparse, glob, json, os
import numpy as np

def per_seed(path):
    d = json.load(open(path)); recs = d["records"]
    a0 = np.array([r["a0"]        for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    dec= [str(r.get("kga_decision","")).lower() for r in recs]
    adapt = np.array(["adapt" in x for x in dec])
    # oracle = best of {freeze, adapt} (single-candidate); a_kbound = a_adapted iff KGA adapted, else a0.
    # Both identities verified EXACT (0/432 mismatch, all candidates) on the complete seed0; reconstructed
    # only for seeds whose logs omit the stored field. reco counts how many records were reconstructed.
    ao = np.array([(r["a_oracle"] if r.get("a_oracle") is not None else max(r["a0"], r["a_adapted"])) for r in recs], float)
    ak = np.array([(r["a_kbound"] if r.get("a_kbound") is not None else (aa[i] if adapt[i] else a0[i])) for i, r in enumerate(recs)], float)
    reco = int(sum(1 for r in recs if r.get("a_kbound") is None or r.get("a_oracle") is None))
    B  = np.array([r["B"] for r in recs], float)
    return dict(seed=d.get("seed"), n=len(recs), reconstructed=reco,
                rk=float((ao-ak).mean()), ra=float((ao-aa).mean()), rf=float((ao-a0).mean()),
                fau=float(np.mean(adapt & (B<=0))),
                rk_pc=(ao-ak), ra_pc=(ao-aa), rf_pc=(ao-a0))

def boot(x, nb=5000, seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); n=len(x); b=np.empty(nb)
    for i in range(nb): b[i]=x[rng.integers(0,n,n)].mean()
    lo,hi=np.percentile(b,[2.5,97.5]); return [round(float(lo),4),round(float(hi),4)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True); ap.add_argument("--candidate",required=True)
    ap.add_argument("--dir",required=True); ap.add_argument("--alpha",type=float,default=0.10)
    ap.add_argument("--out",default=""); a=ap.parse_args()
    import re as _re
    pat=f"per_condition_{a.dataset}_{a.candidate}_seed*.json"
    cand=set(glob.glob(os.path.join(a.dir,pat)))|set(glob.glob(os.path.join(a.dir,"**",pat),recursive=True))
    byseed={}
    for f in cand:
        m=_re.search(r"_seed(\d+)\.json$",f)
        if not m: continue
        sd=int(m.group(1))
        try: n=len(json.load(open(f)).get("records",[]))
        except Exception: n=-1
        if sd not in byseed or n>byseed[sd][0]: byseed[sd]=(n,f)   # keep full-scale over smoke copies
    files=[byseed[s][1] for s in sorted(byseed)]
    if not files: raise SystemExit(f"no per-seed files: {pat} in {a.dir}")
    S=[per_seed(f) for f in files]
    rk=np.array([s["rk"] for s in S]); ra=np.array([s["ra"] for s in S]); rf=np.array([s["rf"] for s in S])
    fau=np.array([s["fau"] for s in S])
    better="freeze" if rf.mean()<=ra.mean() else "adapt"
    # pooled per-condition bootstrap on gaps (more data than #seeds)
    rk_pc=np.concatenate([s["rk_pc"] for s in S]); ra_pc=np.concatenate([s["ra_pc"] for s in S]); rf_pc=np.concatenate([s["rf_pc"] for s in S])
    gap_better = (rf_pc-rk_pc) if better=="freeze" else (ra_pc-rk_pc)
    gap_worse  = (ra_pc-rk_pc) if better=="freeze" else (rf_pc-rk_pc)
    ci_b=boot(gap_better); ci_w=boot(gap_worse)
    gap_vs_adapt = ra_pc - rk_pc; gap_vs_freeze = rf_pc - rk_pc
    ci_a=boot(gap_vs_adapt); ci_f=boot(gap_vs_freeze)
    ties_better = ci_b[0]<=0<=ci_b[1]; beats_worse=ci_w[0]>0; beats_both=ci_b[0]>0 and beats_worse
    fa_ok=bool(np.all(fau<=a.alpha))
    verdict=("beats-both (multi-seed)" if beats_both and fa_ok else
             "stable no-harm" if ties_better and beats_worse and fa_ok else "unstable/other")
    out=dict(dataset=a.dataset, candidate=a.candidate, seeds=[s["seed"] for s in S], n_seeds=len(S),
             conditions_per_seed=S[0]["n"], reconstructed_per_seed=[s["reconstructed"] for s in S], alpha=a.alpha,
             regret_kga=[round(float(rk.mean()),4),round(float(rk.std()),4)],
             regret_adapt=[round(float(ra.mean()),4),round(float(ra.std()),4)],
             regret_freeze=[round(float(rf.mean()),4),round(float(rf.std()),4)],
             FA_u_per_seed=[round(float(x),4) for x in fau], FA_u_max=round(float(fau.max()),4),
             better_policy=better, gap_vs_better_ci95=ci_b, gap_vs_worse_ci95=ci_w,
             gap_vs_adapt=dict(mean=round(float(gap_vs_adapt.mean()),4), ci95=ci_a),
             gap_vs_freeze=dict(mean=round(float(gap_vs_freeze.mean()),4), ci95=ci_f),
             verdict=verdict, files=[os.path.basename(f) for f in files],
             latex_row=(f"{a.dataset} ({a.candidate}) & {len(S)} & "
                        f"{rk.mean():.4f}$\\pm${rk.std():.4f} & "
                        f"{ra.mean():.4f} & {rf.mean():.4f} & {fau.max():.3f} & {verdict} \\\\"))
    o=a.out or f"multiseed_{a.dataset}_{a.candidate}.json"; json.dump(out,open(o,"w"),indent=2)
    print(json.dumps(out,indent=2))
    print("\nLaTeX row:")
    print(f"{a.dataset} ({a.candidate}) & {len(S)} & {rk.mean():.4f}$\\pm${rk.std():.4f} & "
          f"{ra.mean():.4f} & {rf.mean():.4f} & {fau.max():.3f} & {verdict} \\\\")
    print("wrote", o)

if __name__=="__main__": main()
