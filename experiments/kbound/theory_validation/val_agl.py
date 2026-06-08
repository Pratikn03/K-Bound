"""val_agl.py -- disagreement-region characterization of Agreement-on-the-Line (anchored form).

reference f0, family {f_theta}.  On a distribution P:
  A(theta) = P(f_theta = f0)        agreement-with-reference (LABEL-FREE)
  a(theta) = P(f_theta = Y)         accuracy
  w(theta) = P(f_theta = Y | f_theta != f0)   win rate on the disagreement region
Binary 0/1 identity:  a = a0 + (1 - A)(2w - 1)  =>  a is affine in A with slope (1 - 2w).

  C1 sufficiency:  w constant  => exact line, slope = 1 - 2 w, through anchor (A=1, a=a0).
  C2 necessity:    w varying    => not a single line (R^2 < 1).
  C3 transfer/AGL: slope = 1 - 2 w; source & target lines coincide (AGL) iff w_S = w_T;
                   slope gap = -2 (w_T - w_S) = win-rate DRIFT = K-Bound's calibration drift gamma.
Construction fix: a0 = 0.5 (large f0-wrong & f0-right pools) and disagreement mass m <= 0.4,
so any requested win rate is exactly realizable; we MEASURE the realized w to self-check.
"""
import numpy as np, json, os
rng = np.random.default_rng(1)
N, a0 = 60000, 0.50
OUT = {}

Y = rng.integers(0, 2, N)
correct0 = rng.random(N) < a0
f0 = np.where(correct0, Y, 1 - Y)
Wrong = np.where(~correct0)[0]; Good = np.where(correct0)[0]

def make_D(m, w):
    nD = int(m * N); nw = min(len(Wrong), int(round(w * nD))); nc = min(len(Good), nD - nw)
    D = np.zeros(N, bool)
    D[rng.choice(Wrong, nw, replace=False)] = True
    D[rng.choice(Good, nc, replace=False)] = True
    return D

def stats(D):
    f = f0.copy(); f[D] = 1 - f0[D]
    return float(np.mean(f == Y)), float(np.mean(f == f0)), float(np.mean((f == Y)[D]))

def fit(A, a):
    s, b = np.polyfit(A, a, 1); pred = s * A + b
    ss = np.sum((a - a.mean()) ** 2)
    return (1 - np.sum((a - pred) ** 2) / ss if ss > 0 else 1.0), float(s), float(b)

def family(wfun, K=80):
    A, a, w = [], [], []
    for _ in range(K):
        ai, Ai, wi = stats(make_D(rng.uniform(0.05, 0.40), wfun()))
        a.append(ai); A.append(Ai); w.append(wi)
    return np.array(A), np.array(a), np.array(w)

# ---- C1: constant win rate -> exact affine law, slope 1-2w, through anchor ----
for wbar in [0.30, 0.70, 0.90]:
    A, a, w = family(lambda: wbar)
    r2, s, b = fit(A, a); wreal = float(w.mean())
    ok = (r2 > 0.999 and abs(s - (1 - 2 * wreal)) < 0.02 and abs((b + s) - a0) < 0.02
          and abs(wreal - wbar) < 0.02)
    OUT[f"C1_w={wbar}"] = dict(R2=r2, slope=s, slope_pred=1 - 2 * wreal,
                              w_realized=wreal, anchor=b + s, pass_=bool(ok))
    print(f"[C1] w={wbar} (realized {wreal:.3f}): R2={r2:.4f} slope={s:+.3f} "
          f"(pred {1-2*wreal:+.3f}) anchor b+s={b+s:.3f} -> {'PASS' if ok else 'FAIL'}")

# ---- C2: varying win rate -> no single line ----
A, a, w = family(lambda: rng.uniform(0.2, 0.9))
r2v, sv, bv = fit(A, a)
OUT["C2_varying"] = dict(R2=r2v, var_w=float(np.var(w)), pass_=bool(r2v < 0.9))
print(f"[C2] varying w (var={np.var(w):.3f}): R2={r2v:.4f} -> {'PASS (no line)' if r2v<0.9 else 'FAIL'}")

# ---- C3: AGL transfer -- slope set by realized w; gap = -2(w_T - w_S) = drift gamma ----
print("[C3] slope set by win rate; AGL (lines coincide) iff w_S=w_T; gap = drift gamma:")
c3 = []
for wS, wT in [(0.70, 0.70), (0.70, 0.80), (0.60, 0.90)]:
    AS, aS, wSr = family(lambda: wS); rS, sS, _ = fit(AS, aS)
    AT, aT, wTr = family(lambda: wT); rT, sT, _ = fit(AT, aT)
    gap_obs = sT - sS; gap_pred = -2 * (wTr.mean() - wSr.mean())
    agl = abs(gap_obs) < 0.03
    ok = abs(gap_obs - gap_pred) < 0.04 and (agl == (abs(wSr.mean() - wTr.mean()) < 0.015))
    c3.append(dict(wS=wS, wT=wT, slope_S=sS, slope_T=sT, gap_obs=gap_obs, gap_pred=gap_pred,
                   AGL=bool(agl), pass_=bool(ok)))
    print(f"     w_S={wS} w_T={wT}: slopes({sS:+.3f},{sT:+.3f}) gap={gap_obs:+.3f} pred={gap_pred:+.3f} "
          f"AGL={'YES' if agl else 'no'} -> {'PASS' if ok else 'FAIL'}")
OUT["C3_transfer"] = c3

allp = (all(OUT[k]["pass_"] for k in OUT if k.startswith("C1")) and OUT["C2_varying"]["pass_"]
        and all(r["pass_"] for r in c3))
OUT["ALL_PASS"] = bool(allp)
open(os.path.join(os.path.dirname(__file__), "results_agl.json"), "w").write(json.dumps(OUT, indent=2))
print("\nALL_PASS:", allp)
