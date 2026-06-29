#!/usr/bin/env python3
"""D25: PPI target-label-light micro-probe (corrected harness).

Faithful to research_lock/TARGET_LABEL_LIGHT_PPI_PROTOCOL_D25_v1.yaml (sealed
2026-06-15 BEFORE any result). Genuine label-free biased baseline (k=0) + PPI
debiasing from k labels. Reports whatever it computes; no per-category selection.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"
ALPHA = 0.10
PROBE_SIZES = [0, 8, 16, 32, 64]
SEED = 20260615
CACHES = [
    ("Real-IAD-D3",     "experiments/fusion/realiad_d3_score_cache"),
    ("Real-IAD-NatDeg", "experiments/fusion/realiad_natdeg_score_cache"),
    ("MVTec-3D",        "experiments/fusion/mvtec3d_score_cache"),
    ("3D-ADAM",         "experiments/fusion/3d_adam_score_cache"),
    ("MulSen-AD",       "experiments/fusion/mulsen_score_cache"),
]

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos = (y == 1).sum(); neg = (y == 0).sum()
    if pos == 0 or neg == 0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

def calib(col_val, col_test):
    lo, hi = float(np.min(col_val)), float(np.max(col_val))
    rng = hi - lo + 1e-12
    return np.clip((col_val - lo) / rng, 0, 1), np.clip((col_test - lo) / rng, 0, 1)

def cw_fuse(P):  # ELARA confidence-weighted mean; P in [0,1] (n,k)
    w = 2 * np.abs(P - 0.5); ws = w.sum(1, keepdims=True); ws[ws == 0] = 1.0
    return (P * w).sum(1) / ws[:, 0]

def feats(Pf, Pr, Pall):  # label-free per-sample features
    return np.column_stack([Pf, Pr, np.abs(Pf - Pr), Pall.mean(1), Pall.std(1)])

def emp_bernstein_halfwidth(r, alpha):
    """Empirical-Bernstein half-width for the mean of r (k samples)."""
    k = len(r)
    if k < 2: return 1.0
    v = float(np.var(r, ddof=1)); R = float(np.max(r) - np.min(r)) + 1e-12
    import math
    t = math.log(2.0 / alpha)
    return math.sqrt(2.0 * v * t / k) + 3.0 * R * t / k

# ---- Pass 1: per-category label-free prediction + truth ----
cats = []
for track, cdir in CACHES:
    for f in sorted(glob.glob(os.path.join(REPO, cdir, "*.npz"))):
        if os.path.basename(f).startswith("._"): continue
        z = np.load(f)
        if not all(k in z for k in ("Sval", "yval", "Stest", "ytest")): continue
        Sv, yv, St, yt = z["Sval"].astype(float), z["yval"], z["Stest"].astype(float), z["ytest"]
        if len(np.unique(yv)) < 2 or len(np.unique(yt)) < 2 or Sv.shape[1] < 2: continue
        M = Sv.shape[1]
        Pv = np.empty_like(Sv); Pt = np.empty_like(St)
        for m in range(M):
            Pv[:, m], Pt[:, m] = calib(Sv[:, m], St[:, m])
        valauc = np.array([auroc(yv, Pv[:, m]) for m in range(M)])
        best = int(np.nanargmax(valauc))
        # fused vs frozen probabilities
        pf_v, pfr_v = cw_fuse(Pv), Pv[:, best]
        pf_t, pfr_t = cw_fuse(Pt), Pt[:, best]
        # per-sample Brier benefit (positive => fusion helps)
        Bv = (pfr_v - yv) ** 2 - (pf_v - yv) ** 2
        Bt = (pfr_t - yt) ** 2 - (pf_t - yt) ** 2
        # label-free predictor: fit on VAL features -> Bv ; apply to TEST
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=0)
        gb.fit(feats(pf_v, pfr_v, Pv), Bv)
        Bhat_t = gb.predict(feats(pf_t, pfr_t, Pt))
        cats.append(dict(track=track, name=os.path.basename(f)[:-4],
                         theta_lf=float(np.mean(Bhat_t)), theta_true=float(np.mean(Bt)),
                         Bhat=Bhat_t, Btrue=Bt, n_test=int(len(yt))))

n_cat = len(cats)
theta_lf = np.array([c["theta_lf"] for c in cats])
theta_true = np.array([c["theta_true"] for c in cats])
abs_bias = np.abs(theta_lf - theta_true)

def loo_eps(i):  # leave-one-category-out (1-alpha) quantile of |theta_lf - theta_true|
    others = np.delete(abs_bias, i)
    return float(np.quantile(others, 1 - ALPHA))

def decide(point, eps):
    if point - eps > 0: return "ADAPT"
    if point + eps < 0: return "FREEZE"
    return "ABSTAIN"

# ---- Pass 2: per-k decisions ----
rng = np.random.default_rng(SEED)
per_k = {}
for k in PROBE_SIZES:
    rows = []
    for i, c in enumerate(cats):
        true_sign_pos = c["theta_true"] > 0
        if k == 0:
            point = c["theta_lf"]; eps = loo_eps(i)
        else:
            n = c["n_test"]; kk = min(k, n)
            idx = rng.choice(n, size=kk, replace=False)
            r = c["Bhat"][idx] - c["Btrue"][idx]            # rectifier residuals
            point = c["theta_lf"] - float(np.mean(r))        # PPI debiased
            s_lf = float(np.std(c["Bhat"], ddof=1)) / np.sqrt(n)
            eps = emp_bernstein_halfwidth(r, ALPHA) + 1.6449 * s_lf
        dec = decide(point, eps)
        committed = dec != "ABSTAIN"
        # wrong-sign commits
        false_adapt = (dec == "ADAPT" and not true_sign_pos)
        false_freeze = (dec == "FREEZE" and true_sign_pos)
        sign_ok = committed and ((dec == "ADAPT" and true_sign_pos) or (dec == "FREEZE" and not true_sign_pos))
        # regret to oracle (in Brier units): if wrong-sign commit or abstain-with-signal pay |theta_true|
        regret = 0.0 if sign_ok else abs(c["theta_true"])
        rows.append(dict(track=c["track"], name=c["name"], decision=dec, committed=committed,
                         false_adapt=false_adapt, false_freeze=false_freeze, sign_ok=sign_ok,
                         regret=regret, point=point, eps=eps, theta_true=c["theta_true"]))
    commits = [r for r in rows if r["committed"]]
    adapts = [r for r in rows if r["decision"] == "ADAPT"]
    per_k[k] = dict(
        k=k, n_cat=n_cat,
        commit_rate=len(commits) / n_cat,
        n_adapt=len(adapts), n_freeze=sum(r["decision"] == "FREEZE" for r in rows),
        n_abstain=sum(r["decision"] == "ABSTAIN" for r in rows),
        false_adapt_rate=(sum(r["false_adapt"] for r in adapts) / len(adapts)) if adapts else None,
        wrong_sign_among_commit=(sum(not r["sign_ok"] for r in commits) / len(commits)) if commits else None,
        sign_acc_among_commit=(sum(r["sign_ok"] for r in commits) / len(commits)) if commits else None,
        mean_regret=float(np.mean([r["regret"] for r in rows])),
    )

# ---- verdict (per pre-stated success criteria) ----
c0, c64 = per_k[0], per_k[64]
fa_ok = all((per_k[k]["false_adapt_rate"] is None or per_k[k]["false_adapt_rate"] <= ALPHA + 1e-9) for k in PROBE_SIZES)
S1 = c64["commit_rate"] > c0["commit_rate"]
S2 = fa_ok
S3 = c64["mean_regret"] < c0["mean_regret"]
S4 = (c64["sign_acc_among_commit"] is not None and c64["sign_acc_among_commit"] >= 1 - ALPHA)
verdict = "STRONG" if (S1 and S2 and S3 and S4) else ("STANDS" if (S2 and (S1 or S3)) else "HONEST_NEGATIVE")

out = dict(schema="ppi_micro_probe_d25", protocol="research_lock/TARGET_LABEL_LIGHT_PPI_PROTOCOL_D25_v1.yaml",
           alpha=ALPHA, probe_sizes=PROBE_SIZES, seed=SEED, n_categories=n_cat,
           per_k={str(k): v for k, v in per_k.items()},
           criteria=dict(S1_commit_increases=bool(S1), S2_false_adapt_le_alpha=bool(S2),
                         S3_regret_drops=bool(S3), S4_sign_acc=bool(S4)),
           verdict=verdict)
od = os.path.join(REPO, "experiments/kbound/results/ppi_micro_probe_d25")
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=2, default=float)

print(f"=== D25 PPI micro-probe ({n_cat} real categories, alpha={ALPHA}) ===")
print(f"{'k':>3} {'commit':>7} {'adapt':>6} {'freeze':>6} {'abst':>5} {'false_adapt':>12} {'sign_acc':>9} {'regret':>8}")
for k in PROBE_SIZES:
    v = per_k[k]
    fa = "n/a" if v["false_adapt_rate"] is None else f"{v['false_adapt_rate']:.3f}"
    sa = "n/a" if v["sign_acc_among_commit"] is None else f"{v['sign_acc_among_commit']:.3f}"
    print(f"{k:>3} {v['commit_rate']:>7.3f} {v['n_adapt']:>6} {v['n_freeze']:>6} {v['n_abstain']:>5} {fa:>12} {sa:>9} {v['mean_regret']:>8.4f}")
print(f"\nS1 commit↑={S1}  S2 false-adapt≤α={S2}  S3 regret↓={S3}  S4 sign-acc≥1-α={S4}")
print(f"VERDICT: {verdict}")
print(f"wrote {od}/results.json")
