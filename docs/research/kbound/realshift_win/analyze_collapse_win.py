#!/usr/bin/env python3
"""
Analyze a two-sided iWildCam panel for a CI-robust *pure-label-free* beats-both.

Inputs: two manifests from run_iwildcam_kbound.py
  --source  (id_val)  = calibration / dev-lock
  --target  (val or test) = held-out OOD, scored ONCE

Pre-registered (frozen) choices:
  - deployed adapter = source-best by mean adapted macro-F1 on SOURCE  (--deployed to override)
  - evidence = the theory-motivated COLLAPSE-ENTROPY features (domain-invariant; shown on logged
    iWildCam to transfer source->OOD at harm-AUC ~0.64 where calibrated detectors hit ~0.43):
      f1 = -entropy(marginal predicted-class dist of the adapted model)
      f2 = f1 / log(location_classes)   (location-normalized)
  - B_hat + conformal eps fit on SOURCE only; alpha=0.10; target scored once.

No target/test tuning. A null is a legitimate outcome and is reported straight.
"""
import json, argparse, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feasibility_screen import auc, screen
from verify_realshift_win import verify
K = 182

def collapse_feats(r):
    preds = np.asarray(r['preds']); c = np.bincount(preds, minlength=K).astype(float)
    p = c / max(c.sum(), 1); nz = p[p > 0]; ent = float(-(nz * np.log(nz)).sum())
    lc = max(int(r.get('location_classes') or 1), 1)
    return [-ent, -(ent / np.log(max(lc, 2)))]

def build(recs, dep):
    rs = [r for r in recs if r['candidate'] == dep]
    if not rs: raise SystemExit(f"no records for candidate {dep}")
    return (np.array([r['B'] for r in rs]), np.array([r['a0'] for r in rs]),
            np.array([r['aa'] for r in rs]), np.array([collapse_feats(r) for r in rs]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--target', required=True)
    ap.add_argument('--deployed', default='auto'); ap.add_argument('--nboot', type=int, default=5000)
    a = ap.parse_args()
    S = json.load(open(a.source))['records']; T = json.load(open(a.target))['records']
    cands = sorted(set(r['candidate'] for r in S))
    dep = (max(cands, key=lambda c: np.mean([r['aa'] for r in S if r['candidate'] == c]))
           if a.deployed == 'auto' else a.deployed)
    print("candidates:", cands, "| DEPLOYED (source-best, dev-locked):", dep, "\n")
    Bs, a0s, aas, Cs = build(S, dep); Bt, a0t, aat, Ct = build(T, dep)
    print(f"SOURCE n={len(Bs)}  harm={np.mean(Bs<0):.2f}  meanB={Bs.mean():+.3f}")
    print(f"TARGET n={len(Bt)}  harm={np.mean(Bt<0):.2f}  meanB={Bt.mean():+.3f}")
    if not (0.25 <= np.mean(Bt < 0) <= 0.60):
        print("  [warn] target not two-sided (need 0.25-0.60 harmful for a beats-both to be reachable)")
    sc = screen(Cs[:, 0], Bs, Ct[:, 0], Bt)
    print("\nFEASIBILITY (collapse-entropy detector):",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()})
    print("\nLOCKED VERIFIER (collapse-entropy Z; fit on source; target scored once):")
    r = verify(Cs, Bs, Ct, a0t, aat, nboot=a.nboot, seed=0)
    for k in ['n', 'false_adapt', 'regret_kga', 'regret_adapt', 'regret_freeze',
              'gap_vs_adapt_CI', 'gap_vs_freeze_CI', 'beats_both_point', 'beats_both_CI_robust']:
        print(f"   {k} = {r[k]}")
    print("\nVERDICT:",
          "*** CI-ROBUST BEATS-BOTH ***" if r['beats_both_CI_robust']
          else ("point-estimate beats-both (NOT CI-robust)" if r['beats_both_point']
                else "no beats-both (null) — report straight"))

if __name__ == '__main__':
    main()
