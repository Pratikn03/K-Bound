#!/usr/bin/env python3
"""
GO / NO-GO feasibility screen for a CI-robust *pure-label-free* real-shift beats-both.

Run on a LABELED dev/preview split BEFORE committing the full GPU run (and again to early-stop,
"Camelyon wisdom"). Computes the THREE quantities the power phase diagram says decide reachability:
  1. target two-sided mixedness  p_harm   (need ~0.25-0.60: BOTH help and harm for the DEPLOYED adapter)
  2. in-source harm detector AUC           (sanity: is the label-free signal real at all?)
  3. OOD transfer AUC                       (the BINDING constraint: does it carry source->OOD? need >=0.70)

This is a pre-registration SCREEN, not the final verdict. The verdict comes only from the locked
verifier (verify_realshift_win.py) on the held-out test, scored once, with no target tuning.
"""
import numpy as np

def auc(score, is_pos):
    """AUC of `score` (higher => more likely harmful) at ranking the harmful conditions."""
    s = np.asarray(score, float); pos = np.asarray(is_pos, bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def screen(harm_feat_cal, B_cal, harm_feat_tgt, B_tgt, thr=0.0):
    fc, ft = np.asarray(harm_feat_cal, float), np.asarray(harm_feat_tgt, float)
    harm_cal, harm_tgt = (np.asarray(B_cal) <= thr), (np.asarray(B_tgt) <= thr)
    p = float(harm_tgt.mean())
    a_src = auc(fc, harm_cal); orient = 1.0
    if a_src == a_src and a_src < 0.5:          # LOCK the harm direction on SOURCE only
        orient, a_src = -1.0, auc(-fc, harm_cal)
    a_tgt = auc(orient * ft, harm_tgt)          # apply the source-locked direction to OOD
    go = bool(0.25 <= p <= 0.60 and (a_tgt == a_tgt) and a_tgt >= 0.70)
    return dict(target_p_harm=p, insource_AUC=a_src, OOD_transfer_AUC=a_tgt, source_orient=orient,
                mixedness_ok=bool(0.25 <= p <= 0.60), transfer_ok=bool(a_tgt >= 0.70), GO=go)

def _verdict(p, a_tgt):
    if not (0.25 <= p <= 0.60): return "NO-GO (one trivial policy ~= oracle: not two-sided mixed)"
    if a_tgt != a_tgt or a_tgt < 0.55:  return "NO-GO (harm signal does NOT transfer to OOD — the iWildCam/O-H failure)"
    if a_tgt < 0.70: return "MARGINAL (transfer borderline; needs >=0.70 and large n)"
    return "GO (two-sided + transferable -> CI-robust win reachable with n>=240)"

if __name__ == '__main__':
    # Worked examples using YOUR committed logged numbers (experiments/kbound/results/*).
    # (p_harm on target, in-source best harm-AUC, OOD source->target transfer-AUC)
    cases = [
        ("iWildCam val (deployed, entropy/cert)", 0.589, 0.619, 0.432),
        ("iWildCam preview (AETTA detector)",     0.50,  0.667, 0.375),
        ("Office-Home pooled (gradient-TTA)",     0.169, 0.812, 0.328),
        ("Office-Home Product domain (mixed)",    0.389, 0.685, 0.534),
        ("Office-Home Art domain (transfers)",    0.090, 0.901, 0.711),
        ("---- what a GO looks like ----",        0.35,  0.85,  0.80),
    ]
    print(f"{'case':42s} {'p_harm':>7s} {'srcAUC':>7s} {'OOD_AUC':>8s}   verdict")
    for name, p, a_s, a_t in cases:
        print(f"{name:42s} {p:7.3f} {a_s:7.3f} {a_t:8.3f}   {_verdict(p, a_t)}")
    print("\nBinding constraint = OOD_transfer_AUC. Every real attempt so far either lost mixedness")
    print("(Office-Home Art/Clipart) or lost transfer (iWildCam 0.43, O-H Product 0.53). The GPU run")
    print("must put BOTH >=0.30<=p_harm and OOD_AUC>=0.70 in ONE domain. Then the locked verifier decides.")
