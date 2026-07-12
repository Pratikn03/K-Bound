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
    ao = np.array([r["a_oracle"]  for r in recs], float)
    a0 = np.array([r["a0"]        for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    ak = np.array([r.get("a_kbound", r["a0"]) for r in recs], float)
    B  = np.array([r["B"] for r in recs], float)
    dec= [str(r.get("kga_decision","")).lower() for r in recs]
    adapt = np.array(["adapt" in x for x in dec])
    return dict(seed=d.get("seed"), n=len(recs),
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
    files=sorted(glob.glob(os.path.join(a.dir,f"per_condition_{a.dataset}_{a.candidate}_seed*.json")))
    if not files: raise SystemExit(f"no per-seed files: per_condition_{a.dataset}_{a.candidate}_seed*.json in {a.dir}")
    S=[per_seed(f) for f in files]
    rk=np.array([s["rk"] for s in S]); ra=np.array([s["ra"] for s in S]); rf=np.array([s["rf"] for s in S])
    fau=np.array([s["fau"] for s in S])
    better="freeze" if rf.mean()<=ra.mean() else "adapt"
    # pooled per-condition bootstrap on gaps (more data than #seeds)
    rk_pc=np.concatenate([s["rk_pc"] for s in S]); ra_pc=np.concatenate([s["ra_pc"] for s in S]); rf_pc=np.concatenate([s["rf_pc"] for s in S])
    gap_better = (rf_pc-rk_pc) if better=="freeze" else (ra_pc-rk_pc)
    gap_worse  = (ra_pc-rk_pc) if better=="freeze" else (rf_pc-rk_pc)
    ci_b=boot(gap_better); ci_w=boot(gap_worse)
    ties_better = ci_b[0]<=0<=ci_b[1]; beats_worse=ci_w[0]>0; beats_both=ci_b[0]>0 and beats_worse
    fa_ok=bool(np.all(fau<=a.alpha))
    verdict=("beats-both (multi-seed)" if beats_both and fa_ok else
             "stable no-harm" if ties_better and beats_worse and fa_ok else "unstable/other")
    out=dict(dataset=a.dataset, candidate=a.candidate, seeds=[s["seed"] for s in S], n_seeds=len(S),
             conditions_per_seed=S[0]["n"], alpha=a.alpha,
             regret_kga=[round(float(rk.mean()),4),round(float(rk.std()),4)],
             regret_adapt=[round(float(ra.mean()),4),round(float(ra.std()),4)],
             regret_freeze=[round(float(rf.mean()),4),round(float(rf.std()),4)],
             FA_u_per_seed=[round(float(x),4) for x in fau], FA_u_max=round(float(fau.max()),4),
             better_policy=better, gap_vs_better_ci95=ci_b, gap_vs_worse_ci95=ci_w,
             verdict=verdict, files=[os.path.basename(f) for f in files])
    o=a.out or f"multiseed_{a.dataset}_{a.candidate}.json"; json.dump(out,open(o,"w"),indent=2)
    print(json.dumps(out,indent=2))
    print("\nLaTeX row:")
    print(f"{a.dataset} ({a.candidate}) & {len(S)} & {rk.mean():.4f}$\\pm${rk.std():.4f} & "
          f"{ra.mean():.4f} & {rf.mean():.4f} & {fau.max():.3f} & {verdict} \\\\")
    print("wrote", o)

if __name__=="__main__": main()
