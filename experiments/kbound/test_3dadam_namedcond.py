#!/usr/bin/env python3
"""Pre-registered held-out test of 3D-ADAM for the paper's named condition.
3D-ADAM was SELECTED from 5 on-disk caches by a named-condition screen (sign-mix +
label-free detectability) -> selection caveat noted; any positive is exploratory and
needs replication. Leave-one-category-out conformal certificate; alpha=0.10. Report as-is.
"""
import glob, os, json, numpy as np
from collections import Counter
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
cats = []
for f in fs:
    d = np.load(f)
    if not all(k in d for k in ("Sval", "yval", "Stest", "ytest")): continue
    Sv, yv, St, yt = d["Sval"].astype(float), d["yval"], d["Stest"].astype(float), d["ytest"]
    if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2 or Sv.shape[1] < 2: continue
    M = Sv.shape[1]
    Pv = np.column_stack([mm(Sv[:, m], Sv[:, m]) for m in range(M)])
    Pt = np.column_stack([mm(Sv[:, m], St[:, m]) for m in range(M)])
    va = np.array([auroc(yv, Pv[:, m]) for m in range(M)]); best = int(np.nanargmax(va))
    cats.append(dict(name=os.path.basename(f)[:-4],
                     Bval=auroc(yv, cw(Pv)) - auroc(yv, Pv[:, best]),
                     Btest=auroc(yt, cw(Pt)) - auroc(yt, Pt[:, best]),
                     fused_t=auroc(yt, cw(Pt)), frozen_t=auroc(yt, Pt[:, best])))
n = len(cats); Bval = np.array([c["Bval"] for c in cats]); Btest = np.array([c["Btest"] for c in cats])
dec = []; kga_auc = []; fa = 0; n_adapt = 0
for i, c in enumerate(cats):
    res = np.abs(np.delete(Bval, i) - np.delete(Btest, i)); eps = float(np.quantile(res, 1 - ALPHA))
    if c["Bval"] - eps > 0: d_, sc = "ADAPT", c["fused_t"]; n_adapt += 1; fa += int(c["Btest"] < 0)
    elif c["Bval"] + eps < 0: d_, sc = "FREEZE", c["frozen_t"]
    else: d_, sc = "ABSTAIN", c["frozen_t"]
    dec.append(d_); kga_auc.append(sc)
kga = float(np.mean(kga_auc)); aa = float(np.mean([c["fused_t"] for c in cats]))
af = float(np.mean([c["frozen_t"] for c in cats])); orc = float(np.mean([max(c["fused_t"], c["frozen_t"]) for c in cats]))
beats_both = bool(kga > aa + 1e-9 and kga > af + 1e-9)
out = dict(dataset="3D-ADAM", n_categories=n, alpha=ALPHA, selected_from=5,
           decisions=dict(Counter(dec)), mean_test_auroc=dict(always_adapt=aa, always_freeze=af, kga_routed=kga, oracle=orc),
           beats_both=beats_both, n_adapt=n_adapt, false_adapt=fa, false_adapt_rate=fa / max(n_adapt, 1),
           kga_minus_adapt=kga - aa, kga_minus_freeze=kga - af)
od = "experiments/kbound/results/namedcond_3dadam"; os.makedirs(od, exist_ok=True)
json.dump(out, open(od + "/results.json", "w"), indent=2, default=float)
print(f"=== 3D-ADAM held-out (LOCO conformal, {n} categories, alpha={ALPHA}) ===")
print(f"decisions: {dict(Counter(dec))}")
print(f"mean test AUROC: always-adapt(fused)={aa:.4f}  always-freeze(single)={af:.4f}  KGA={kga:.4f}  oracle={orc:.4f}")
print(f"beats_both={beats_both} | n_adapt={n_adapt} false_adapt={fa} (rate={fa/max(n_adapt,1):.3f})")
print(f"KGA-adapt={kga-aa:+.4f}  KGA-freeze={kga-af:+.4f}")
