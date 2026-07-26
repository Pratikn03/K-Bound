#!/usr/bin/env python3
"""
REAL test on logged iWildCam predictions (no GPU). Question: do SCALE / LOCATION-INVARIANT
collapse features transfer source(id_val) -> OOD(val) better than the entropy/balance detectors
that failed (documented OOD harm-AUC 0.43)?

Honest discipline: the deployed adapter is dev-locked on SOURCE (best in-dist adapted metric);
each feature's orientation is locked on SOURCE; the OOD target is scored once. No target tuning.
The NEW idea = collapse measured RELATIVE TO the location's own class count (location_classes),
which removes the camera-trap intrinsic class-imbalance confound that makes raw predicted-class
balance anti-transfer.
"""
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feasibility_screen import auc
from verify_realshift_win import verify

ROOT = KB_REPO_ROOT + '/'
SRC = ROOT + 'experiments/kbound/results/iwildcam_full_idval/result_489da28f.json'
TGT = ROOT + 'experiments/kbound/results/iwildcam_full_val/result_f08e751c.json'
K = 182

def load(p):
    d = json.load(open(p)); return d['records'], d['evidence_names']

def collapse_feats(r):
    preds = np.asarray(r['preds']); c = np.bincount(preds, minlength=K).astype(float)
    tot = max(c.sum(), 1); p = c / tot; nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum()); top = float(p.max()); ndist = int((c > 0).sum())
    lc = max(int(r.get('location_classes') or 1), 1)
    return dict(
        coll_topfrac=top,                                  # high => collapsed to one class
        coll_negentropy=-ent,                              # high => collapsed (absolute)
        coll_locnorm_negent=-(ent / np.log(max(lc, 2))),   # NEW: entropy normalized by location's #classes
        coll_locnorm_negdistinct=-(ndist / lc),            # NEW: distinct preds / available classes
        upd_norm=float(r.get('upd_norm') or 0.0),          # update-norm magnitude
    )

def build(records, deployed):
    rs = [r for r in records if r['candidate'] == deployed]
    cf = [collapse_feats(r) for r in rs]; cn = list(cf[0].keys())
    return dict(B=np.array([r['B'] for r in rs]), a0=np.array([r['a0'] for r in rs]),
                aa=np.array([r['aa'] for r in rs]), Z=np.array([r['Z'] for r in rs]),
                C=np.array([[f[n] for n in cn] for f in cf]), cn=cn)

def main():
    srec, enames = load(SRC); trec, _ = load(TGT)
    cands = sorted(set(r['candidate'] for r in srec))
    src_mean_aa = {c: float(np.mean([r['aa'] for r in srec if r['candidate'] == c])) for c in cands}
    deployed = max(src_mean_aa, key=src_mean_aa.get)
    print("source mean adapted metric by candidate:", {c: round(v, 3) for c, v in src_mean_aa.items()})
    print("DEPLOYED (source-best, dev-locked):", deployed, "\n")
    S, T = build(srec, deployed), build(trec, deployed)
    names = list(enames) + S['cn']
    Xs, Xt = np.c_[S['Z'], S['C']], np.c_[T['Z'], T['C']]
    hs, ht = S['B'] < 0, T['B'] < 0
    print(f"n_source={len(S['B'])} (harm {hs.mean():.2f}, meanB {S['B'].mean():+.3f})   "
          f"n_target={len(T['B'])} (harm {ht.mean():.2f}, meanB {T['B'].mean():+.3f})\n")
    print(f"{'feature':26s} {'srcAUC':>7s} {'OOD_AUC':>8s}  note")
    rows = []
    for j, nm in enumerate(names):
        fs, ft = Xs[:, j], Xt[:, j]
        a_s = auc(fs, hs); orient = 1.0
        if a_s == a_s and a_s < 0.5:
            orient, a_s = -1.0, auc(-fs, hs)
        a_t = auc(orient * ft, ht)
        rows.append((nm, a_s, a_t));
    for nm, a_s, a_t in sorted(rows, key=lambda x: -(x[2] if x[2] == x[2] else 0)):
        note = 'TRANSFERS' if (a_t == a_t and a_t >= 0.65) else ('~chance' if a_t == a_t and a_t < 0.55 else '')
        new = '  (NEW)' if nm.startswith('coll_locnorm') else ''
        print(f"{nm:26s} {a_s:7.3f} {a_t:8.3f}  {note}{new}")
    best_src = max(rows, key=lambda x: (x[1] if x[1] == x[1] else 0))
    print(f"\nSOURCE-selected best feature (dev-lock): {best_src[0]}  srcAUC {best_src[1]:.3f} -> OOD {best_src[2]:.3f}")

    print("\nLOCKED VERIFIER  (collapse-feature Z; fit B_hat+eps on id_val, val scored once):")
    r = verify(S['C'], S['B'], T['C'], T['a0'], T['aa'], nboot=4000, seed=0)
    for k in ['n', 'false_adapt', 'regret_kga', 'regret_adapt', 'regret_freeze',
              'gap_vs_adapt_CI', 'gap_vs_freeze_CI', 'beats_both_point', 'beats_both_CI_robust']:
        print(f"   {k} = {r[k]}")
    print("\n(For reference, prior documented detectors: best single-feature srcAUC ~0.62, "
          "certificate OOD transfer-AUC 0.43 -> NO win.)")

if __name__ == '__main__':
    main()
