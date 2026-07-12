#!/usr/bin/env python3
"""
Aggregate per-seed natural-shift results into a multi-seed no-harm summary.

For each track, point it at the per-seed result JSONs (produced by re-running the locked
WILDS/DomainBed protocol at seeds 0..4). Each file must expose the KGA / always-adapt /
always-freeze regrets (defensive key search) and, if present, the false-adapt rate.

Outputs, per track:
  * across-seed mean +/- std of each policy's regret and of KGA's FA_u,
  * a seed-level paired bootstrap CI on the regret gap KGA-vs-worse and KGA-vs-better policy,
  * a verdict: "stable no-harm" iff KGA ties the better fixed policy and beats the worse one
    across every seed at FA_u <= alpha, else "unstable"/"beats-both"/"harmful".
  * a LaTeX table row.

Usage:
  python3 multiseed_aggregate.py --track iWildCam --glob "experiments/kbound/results/iwildcam_*seed*/*.json"
  python3 multiseed_aggregate.py --demo        # labelled synthetic 5-seed example (no data needed)

No fabrication: real runs require real per-seed files; --demo output is clearly marked SYNTHETIC.
"""
import argparse, glob as globmod, json, os
import numpy as np

def find(d, cands):
    for holder in (d, d.get("point",{}) if isinstance(d,dict) else {}, d.get("regret",{}) if isinstance(d,dict) else {}):
        if isinstance(holder, dict):
            for c in cands:
                if c in holder: return float(holder[c])
    return None

def load_seed(path):
    d = json.load(open(path))
    kga  = find(d, ("regret_kga","kga","K_Bound","kbound"))
    adpt = find(d, ("regret_adapt","always_adapt","adapt"))
    frz  = find(d, ("regret_freeze","always_freeze","freeze"))
    fau  = find(d, ("false_adapt","FA_u","fa_u"))
    if None in (kga,adpt,frz): raise SystemExit(f"missing regret keys in {path}: kga={kga} adapt={adpt} freeze={frz}")
    return dict(kga=kga, adapt=adpt, freeze=frz, fau=(0.0 if fau is None else fau), src=os.path.basename(path))

def boot_gap(x, nb=5000, seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); n=len(x); b=np.empty(nb)
    for i in range(nb): b[i]=x[rng.integers(0,n,n)].mean()
    lo,hi=np.percentile(b,[2.5,97.5]); return round(float(x.mean()),4),[round(float(lo),4),round(float(hi),4)]

def summarize(track, seeds, alpha=0.10):
    kga=np.array([s["kga"] for s in seeds]); ad=np.array([s["adapt"] for s in seeds]); fr=np.array([s["freeze"] for s in seeds])
    fau=np.array([s["fau"] for s in seeds])
    better = "freeze" if fr.mean()<=ad.mean() else "adapt"; worse = "adapt" if better=="freeze" else "freeze"
    gb_m, gb_ci = boot_gap((fr if better=="freeze" else ad) - kga)   # gap vs better (expect ~0)
    gw_m, gw_ci = boot_gap((ad if worse=="adapt" else fr) - kga)     # gap vs worse (expect >0)
    ties_better = gb_ci[0] <= 0 <= gb_ci[1]
    beats_worse = gw_ci[0] > 0
    beats_both  = (gb_ci[0] > 0) and beats_worse
    fa_ok = bool(np.all(fau <= alpha))
    verdict = ("beats-both (multi-seed)" if beats_both and fa_ok else
               "stable no-harm" if ties_better and beats_worse and fa_ok else
               "unstable/other")
    return dict(track=track, seeds=len(seeds), alpha=alpha,
                regret_kga=[round(float(kga.mean()),4),round(float(kga.std()),4)],
                regret_adapt=[round(float(ad.mean()),4),round(float(ad.std()),4)],
                regret_freeze=[round(float(fr.mean()),4),round(float(fr.std()),4)],
                FA_u_max=round(float(fau.max()),4), better_policy=better,
                gap_vs_better=dict(mean=gb_m, ci95=gb_ci), gap_vs_worse=dict(mean=gw_m, ci95=gw_ci),
                verdict=verdict, sources=[s["src"] for s in seeds])

def latex_row(s):
    return (f"{s['track']} & {s['seeds']} & {s['regret_kga'][0]:.4f}$\\pm${s['regret_kga'][1]:.4f} & "
            f"{s['regret_adapt'][0]:.4f} & {s['regret_freeze'][0]:.4f} & {s['FA_u_max']:.3f} & {s['verdict']} \\\\")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--track",default="track"); ap.add_argument("--glob",default="")
    ap.add_argument("--alpha",type=float,default=0.10); ap.add_argument("--demo",action="store_true")
    ap.add_argument("--out",default=""); a=ap.parse_args()
    if a.demo:
        rng=np.random.default_rng(0)
        seeds=[dict(kga=float(f), adapt=float(f+0.09+0.01*rng.standard_normal()),
                    freeze=float(f+0.0002*rng.standard_normal()), fau=0.0, src=f"SYNTH_seed{i}")
               for i,f in enumerate(0.004+0.0005*rng.standard_normal(5))]
        s=summarize("iWildCam[SYNTHETIC]", seeds, a.alpha)
    else:
        files=sorted(globmod.glob(a.glob))
        if not files: raise SystemExit(f"no per-seed files matched: {a.glob!r} (or use --demo)")
        s=summarize(a.track, [load_seed(f) for f in files], a.alpha)
    out=a.out or (f"multiseed_{s['track'].split('[')[0]}.json")
    json.dump(s, open(out,"w"), indent=2)
    print(json.dumps(s, indent=2)); print("\nLaTeX row:\n"+latex_row(s)); print("wrote", out)

if __name__=="__main__": main()
