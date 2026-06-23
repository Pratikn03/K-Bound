#!/usr/bin/env python3
"""
K-Bound real-shift WIN VERIFIER  (locked, pre-registered, integrity-clean).
Torch-free (numpy only). NO target/test tuning anywhere.

FROZEN operating point (decided on CALIBRATION before any target is scored):
  - benefit estimator B_hat(Z) = ridge on calibration (Z_cal -> B_cal) only
  - conformal radius eps = (1-alpha) quantile of K-fold residuals on CALIBRATION
  - KGA rule: adapt if B_hat-eps>0; freeze if B_hat+eps<0; else abstain
  - alpha = 0.10 fixed
VERDICT on HELD-OUT TARGET, scored once:
  - regret-to-oracle for KGA / always-adapt / always-freeze (oracle = per-condition best of frozen/adapted)
  - false-adapt rate (KGA adapts where true benefit B<=0); pre-registered bar FA <= alpha
  - beats-both (point): regret_kga < regret_adapt AND < regret_freeze AND FA<=alpha
  - CI-robust: paired CONDITION-bootstrap 95% CI of BOTH regret gaps excludes 0 (and FA<=alpha)

This file is BOTH the verifier you run on real logged GPU results AND its own self-test
(python verify_realshift_win.py  ->  proves it rewards a genuine win and rejects the
documented failure modes: helpful-dominated, all-harm, anti-transfer detector).
"""
import numpy as np

ALPHA = 0.10

# ---------- locked certificate machinery ----------
def _ridge_fit(X, y, lam=1.0):
    X1 = np.c_[np.ones(len(X)), X]
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[0, 0] -= lam   # intercept unpenalized
    return np.linalg.solve(A, X1.T @ y)

def _ridge_pred(w, X):
    return np.c_[np.ones(len(X)), X] @ w

def _kfold_resid(X, y, lam=1.0, k=5, seed=0):
    n = len(X); idx = np.random.default_rng(seed).permutation(n); res = np.empty(n)
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        res[f] = y[f] - _ridge_pred(_ridge_fit(X[tr], y[tr], lam), X[f])
    return res

def fit_certificate(Z_cal, B_cal, alpha=ALPHA, lam=1.0):
    """B_hat fit on calibration; eps = (1-alpha) quantile of |k-fold residual|. Target never seen."""
    w = _ridge_fit(Z_cal, B_cal, lam)
    eps = float(np.quantile(np.abs(_kfold_resid(Z_cal, B_cal, lam)), 1 - alpha))
    return w, eps

def kga_decide(Bhat, eps):
    return np.where(Bhat - eps > 0, 'adapt', np.where(Bhat + eps < 0, 'freeze', 'abstain'))

def verify(Z_cal, B_cal, Z_tgt, a0_tgt, aa_tgt, alpha=ALPHA, lam=0.5, nboot=5000, seed=12345):
    """Every decision parameter comes from calibration. Target is scored exactly once."""
    w, eps = fit_certificate(Z_cal, B_cal, alpha, lam)
    dec = kga_decide(_ridge_pred(w, Z_tgt), eps)
    a0, aa = np.asarray(a0_tgt, float), np.asarray(aa_tgt, float)
    B = aa - a0
    oracle = np.maximum(a0, aa)
    val_kga = np.where(dec == 'adapt', aa, a0)            # freeze/abstain keep frozen
    reg_kga, reg_ad, reg_fr = (oracle - val_kga).mean(), (oracle - aa).mean(), (oracle - a0).mean()
    fa = float(((dec == 'adapt') & (B <= 0)).mean())
    beats_point = bool(reg_kga < reg_ad and reg_kga < reg_fr and fa <= alpha)
    # paired condition bootstrap of the two regret gaps (adapt-kga, freeze-kga)
    g_a = (oracle - aa) - (oracle - val_kga)
    g_f = (oracle - a0) - (oracle - val_kga)
    n = len(a0); rng = np.random.default_rng(seed)
    S = rng.integers(0, n, (nboot, n))
    ga_bs, gf_bs = g_a[S].mean(1), g_f[S].mean(1)
    ci_a = (float(np.quantile(ga_bs, .025)), float(np.quantile(ga_bs, .975)))
    ci_f = (float(np.quantile(gf_bs, .025)), float(np.quantile(gf_bs, .975)))
    ci_robust = bool(ci_a[0] > 0 and ci_f[0] > 0 and fa <= alpha)
    return dict(n=n, eps=eps, alpha=alpha,
                regret_kga=float(reg_kga), regret_adapt=float(reg_ad), regret_freeze=float(reg_fr),
                false_adapt=fa,
                gap_vs_adapt=float(g_a.mean()), gap_vs_adapt_CI=list(ci_a),
                gap_vs_freeze=float(g_f.mean()), gap_vs_freeze_CI=list(ci_f),
                beats_both_point=beats_point, beats_both_CI_robust=ci_robust,
                n_adapt=int((dec == 'adapt').sum()), n_freeze=int((dec == 'freeze').sum()),
                n_abstain=int((dec == 'abstain').sum()))

# ---------- synthetic regimes for self-test (ground truth known) ----------
def make_regime(kind, n=200, seed=0, det_noise=0.02, transfer=1.0, p_harm=None):
    """Return (Z_cal,B_cal, Z_tgt,a0_tgt,aa_tgt).
    det_noise = label-free harm-detector quality (smaller = sharper signal).
    transfer  = how much the source->OOD signal carries (1=fully, 0=none); the binding
                constraint that killed iWildCam/Office-Home (in-source signal, anti-transfer)."""
    rng = np.random.default_rng(seed)
    # no-win regimes have ONE trivial policy already = oracle:
    #   helpful_dom (Camelyon): deployed adapter helps on ALL conditions -> always-adapt is oracle
    #   all_harm    (RxRx1):    deployed adapter hurts on ALL conditions -> always-freeze is oracle
    # win needs two-sided mixedness (goldilocks/anti_transfer) AND a transferable detector.
    p = p_harm if p_harm is not None else dict(goldilocks=.35, helpful_dom=.0, all_harm=1.0, anti_transfer=.35)[kind]
    def draw_B(ph):
        harm = rng.random(n) < ph
        return np.where(harm, -rng.uniform(.04, .18, n), rng.uniform(.04, .14, n))
    B_cal, B_tgt = draw_B(p), draw_B(p)
    a0 = rng.uniform(.30, .60, n); aa = a0 + B_tgt
    sd = max(B_tgt.std(), 1e-6)
    z1c = B_cal + rng.normal(0, det_noise, n)                       # in-source detector (sharp)
    Zc = np.c_[z1c, rng.normal(0, 1, n), rng.normal(0, 1, n)]
    tr = 0.0 if kind == 'anti_transfer' else transfer              # anti_transfer: signal dies on OOD
    z1t = tr * B_tgt + rng.normal(0, det_noise, n) + (1 - tr) * rng.normal(0, sd, n)
    Zt = np.c_[z1t, rng.normal(0, 1, n), rng.normal(0, 1, n)]
    return Zc, B_cal, Zt, a0, aa

def self_test(seeds=25):
    expect = {  # (must beats-both point?, must be CI-robust?) on the genuine-win regime only
        'goldilocks':    True,
        'helpful_dom':   False,
        'all_harm':      False,
        'anti_transfer': False,
    }
    print(f"{'regime':14s} {'beats_point%':>12s} {'CI_robust%':>11s} {'FA(mean)':>9s}  verdict")
    ok = True
    for kind, win_expected in expect.items():
        pts, cis, fas = [], [], []
        for s in range(seeds):
            Zc, Bc, Zt, a0, aa = make_regime(kind, n=180, seed=1000 + s)
            r = verify(Zc, Bc, Zt, a0, aa, nboot=2000, seed=7 + s)
            pts.append(r['beats_both_point']); cis.append(r['beats_both_CI_robust']); fas.append(r['false_adapt'])
        pp, cc, fa = 100*np.mean(pts), 100*np.mean(cis), np.mean(fas)
        if win_expected:
            good = cc >= 80                       # genuine win => CI-robust in strong majority
        else:
            good = cc <= 5                        # failure modes => essentially never a CI-robust win
        ok &= good
        print(f"{kind:14s} {pp:11.0f}% {cc:10.0f}% {fa:9.3f}  {'PASS' if good else 'FAIL'}  (expect win={win_expected})")
    print("\nSELF-TEST:", "ALL PASS — verifier rewards a genuine goldilocks win and rejects "
          "helpful-dominated / all-harm / anti-transfer." if ok else "FAIL")
    return ok

if __name__ == '__main__':
    import sys
    sys.exit(0 if self_test() else 1)
