#!/usr/bin/env python3
"""Significance test for the 3D-ADAM held-out beats-both: paired bootstrap over the 23
categories. Honest: the beats-both hinges on a +0.008 margin over always-freeze; this checks
whether it survives resampling. Reports CIs and P(KGA>freeze), P(KGA>adapt)."""
import glob, os, json, numpy as np
ALPHA = 0.10
def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); pos = (y == 1).sum(); neg = (y == 0).sum()
    if pos == 0 or neg == 0: return float('nan')
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
def cw(P):
    w = 2 * np.abs(P - 0.5); ws = w.sum(1, keepdims=True); ws[ws == 0] = 1
    return (P * w).sum(1) / ws[:, 0]
def mm(cv, ct):
    lo, hi = cv.min(), cv.max(); r = hi - lo + 1e-12; return np.clip((ct - lo) / r, 0, 1)
fs = [f for f in glob.glob("experiments/fusion/3d_adam_score_cache/*.npz") if not os.path.basename(f).startswith("._")]
C = []
for f in fs:
    d = np.load(f)
    if not all(k in d for k in ("Sval", "yval", "Stest", "ytest")): continue
    Sv, yv, St, yt = d["Sval"].astype(float), d["yval"], d["Stest"].astype(float), d["ytest"]
    if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2 or Sv.shape[1] < 2: continue
    M = Sv.shape[1]
    Pv = np.column_stack([mm(Sv[:, m], Sv[:, m]) for m in range(M)])
    Pt = np.column_stack([mm(Sv[:, m], St[:, m]) for m in range(M)])
    va = np.array([auroc(yv, Pv[:, m]) for m in range(M)]); best = int(np.nanargmax(va))
    C.append(dict(Bval=auroc(yv, cw(Pv)) - auroc(yv, Pv[:, best]),
                  Btest=auroc(yt, cw(Pt)) - auroc(yt, Pt[:, best]),
                  fused=auroc(yt, cw(Pt)), frozen=auroc(yt, Pt[:, best])))
n = len(C); Bval = np.array([c["Bval"] for c in C]); Btest = np.array([c["Btest"] for c in C])
fused = np.array([c["fused"] for c in C]); frozen = np.array([c["frozen"] for c in C])
# LOCO conformal -> per-category KGA-deployed AUROC
kga = np.empty(n)
for i in range(n):
    res = np.abs(np.delete(Bval, i) - np.delete(Btest, i)); eps = float(np.quantile(res, 1 - ALPHA))
    kga[i] = fused[i] if (Bval[i] - eps > 0) else frozen[i]   # adapt->fused; freeze/abstain->frozen
rng = np.random.default_rng(20260615); B = 10000
dkf = []; dka = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    dkf.append(kga[idx].mean() - frozen[idx].mean())
    dka.append(kga[idx].mean() - fused[idx].mean())
dkf = np.array(dkf); dka = np.array(dka)
def ci(a): return [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
out = dict(n=n, kga=float(kga.mean()), freeze=float(frozen.mean()), adapt=float(fused.mean()),
           kga_minus_freeze=float(kga.mean()-frozen.mean()), ci_kga_minus_freeze=ci(dkf), P_kga_gt_freeze=float((dkf>0).mean()),
           kga_minus_adapt=float(kga.mean()-fused.mean()), ci_kga_minus_adapt=ci(dka), P_kga_gt_adapt=float((dka>0).mean()),
           beats_both_significant=bool((dkf>0).mean()>=0.95 and (dka>0).mean()>=0.95))
json.dump(out, open("experiments/kbound/results/namedcond_3dadam/bootstrap.json","w"), indent=2, default=float)
print(f"3D-ADAM paired bootstrap (n={n} categories, {B} resamples):")
print(f"  KGA={kga.mean():.4f}  freeze={frozen.mean():.4f}  adapt={fused.mean():.4f}")
print(f"  KGA-freeze = {kga.mean()-frozen.mean():+.4f}  95% CI {ci(dkf)}  P(KGA>freeze)={(dkf>0).mean():.3f}")
print(f"  KGA-adapt  = {kga.mean()-fused.mean():+.4f}  95% CI {ci(dka)}  P(KGA>adapt)={(dka>0).mean():.3f}")
print(f"  beats_both_significant (both P>=0.95) = {out['beats_both_significant']}")
