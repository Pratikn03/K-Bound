#!/usr/bin/env python3
"""
theory_v2_validation.py
=======================
Numerical validation for K-Bound Theory V2 (T-I, T-II, T-III) and the corrected
hypothesis H (per-class symmetric accuracy).

Run with --part {H,V1,V2,V3,V4,all}. Each part keeps to <40s and writes/updates
validation_results.json and figures *.png in this directory.

Setting (binary core): Y in {0,1} on observable region D (everything conditioned on D).
Panel of K predictors f_0 (frozen), f_1..f_{K-1} (candidates), fixed maps of X.
Advantage b_j = 2 P(f_j=Y|D) - 1. Correctness C_j = 1[f_j=Y].
Agreement A_ij = P(f_i=f_j|D); c_ij = 2A_ij - 1.

Corrected hypothesis H: per-class symmetric accuracy q_j(0)=q_j(1) for all j,
                        plus conditional error-independence given Y.
Under H: c_ij = b_i b_j (exact). The deficit is c_ij - b_i b_j = pi(1-pi) delta_i delta_j,
delta_j := (2 q_j(1)-1) - (2 q_j(0)-1) = 2(q_j(1)-q_j(0)).

Author: K-Bound theory_v2 agent. Seeds fixed for reproducibility.
"""
import json, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "validation_results.json")

# ----------------------------------------------------------------------------- utils
def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return {}

def save_results(d):
    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)

def simulate_panel(pi, q0, q1, n, seed):
    """Simulate a CEI panel on D with per-class accuracies q_j(0)=q0[j], q_j(1)=q1[j].
    Returns predictions F (K x n), correctness C (K x n), labels Y (n,)."""
    rng = np.random.default_rng(seed)
    K = len(q0)
    Y = (rng.random(n) < pi).astype(np.int8)
    qmat = np.stack([np.asarray(q0), np.asarray(q1)], axis=1)  # K x 2
    qy = qmat[:, Y]                                            # K x n: P(correct)
    C = (rng.random((K, n)) < qy).astype(np.int8)             # CEI by construction
    F = np.where(C == 1, Y[None, :], 1 - Y[None, :]).astype(np.int8)
    return F, C, Y

def agreements(F):
    K = F.shape[0]
    A = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = np.mean(F[i] == F[j])
    return A

def product_ratio_b2(c, i, k, l):
    """b_i^2 = c_ik c_il / c_kl  (closed-form product-ratio estimator)."""
    return c[i, k] * c[i, l] / c[k, l]

def recover_b_up_to_flip(c, K, anchor_sign=+1):
    """Recover |b| via product ratios, fix global sign so majority(b)>0 maps to anchor_sign.
    Uses, for each i, the median over all valid (k,l) pairs of c_ik c_il / c_kl.
    Relative signs from sign(c_ij). Returns b_hat (with chosen global sign)."""
    b2 = np.zeros(K)
    for i in range(K):
        vals = []
        for k in range(K):
            for l in range(K):
                if len({i, k, l}) == 3 and abs(c[k, l]) > 1e-9:
                    vals.append(c[i, k] * c[i, l] / c[k, l])
        b2[i] = np.median(vals) if vals else np.nan
    b2 = np.clip(b2, 0.0, None)
    mag = np.sqrt(b2)
    # relative signs: pick reference index 0 positive, s_i = sign(c_{0,i})
    s = np.ones(K)
    for i in range(1, K):
        s[i] = np.sign(c[0, i]) if abs(c[0, i]) > 0 else 1.0
    b = s * mag
    # global flip: anchor by majority-above-chance (sum of b > 0)
    if np.sign(np.sum(b)) != anchor_sign and np.sum(b) != 0:
        b = -b
    return b

# ----------------------------------------------------------------------------- H check
def part_H(res):
    """Hypothesis H verification: c_ij - b_i b_j = pi(1-pi) delta_i delta_j.
    Demonstrates plain CEI is INSUFFICIENT; per-class symmetry is the minimal fix."""
    rng = np.random.default_rng(100)
    K = 4
    od = ~np.eye(K, dtype=bool)
    n = 3_000_000
    out = {}

    # (1) Symmetric accuracies (H holds), balanced and unbalanced classes
    q = rng.uniform(0.60, 0.90, size=K)
    sym_runs = {}
    for pi in [0.5, 0.25]:
        F, C, Y = simulate_panel(pi, q, q, n, seed=int(pi * 1000) + 1)
        a = C.mean(1); b = 2 * a - 1
        A = agreements(F); c = 2 * A - 1
        err_ident = float(np.max(np.abs((c - np.outer(b, b))[od])))
        sym_runs[f"pi_{pi}"] = {"b": b.round(5).tolist(),
                                "max_offdiag_abs_c_minus_bb": err_ident}
    out["symmetric_H_holds"] = sym_runs

    # (2) Asymmetric accuracies (plain CEI only): identity FAILS; deficit = pi(1-pi) delta_i delta_j
    q0 = rng.uniform(0.50, 0.95, size=K)
    q1 = rng.uniform(0.50, 0.95, size=K)
    pi = 0.5
    F, C, Y = simulate_panel(pi, q0, q1, n, seed=77)
    a = C.mean(1); b = 2 * a - 1
    A = agreements(F); c = 2 * A - 1
    deficit_emp = (c - np.outer(b, b))
    delta = (2 * q1 - 1) - (2 * q0 - 1)          # = 2(q1-q0)
    deficit_theory = pi * (1 - pi) * np.outer(delta, delta)
    out["asymmetric_plainCEI_fails"] = {
        "q0": q0.round(4).tolist(), "q1": q1.round(4).tolist(),
        "delta": delta.round(4).tolist(),
        "max_offdiag_abs_c_minus_bb": float(np.max(np.abs(deficit_emp[od]))),
        "max_offdiag_abs_deficit_minus_theory": float(np.max(np.abs((deficit_emp - deficit_theory)[od]))),
        "note": "c_ij != b_i b_j under plain CEI; deficit matches pi(1-pi) delta_i delta_j"
    }

    # (3) 2x2 minor (rank-1 test) under asymmetry: nonzero => rank-1 broken
    minor = float(c[0, 1] * c[2, 3] - c[0, 2] * c[1, 3])
    out["asymmetry_breaks_rank1_minor_0123"] = minor

    res["H_hypothesis_check"] = out
    print("[H] symmetric (H holds) max|c-bb| offdiag:",
          {k: round(v["max_offdiag_abs_c_minus_bb"], 5) for k, v in sym_runs.items()})
    print("[H] asymmetric (plain CEI) max|c-bb| offdiag:",
          round(out["asymmetric_plainCEI_fails"]["max_offdiag_abs_c_minus_bb"], 5),
          " deficit==theory to:",
          round(out["asymmetric_plainCEI_fails"]["max_offdiag_abs_deficit_minus_theory"], 6))
    print("[H] rank-1 minor under asymmetry:", round(minor, 5))
    return res

# ----------------------------------------------------------------------------- V1 flip
def part_V1(res):
    """Flip witness: under H, label-complement Y'=1-Y preserves predictions, agreements,
    the full pattern (evidence) law, and H, while b -> -b. TV=0 exactly."""
    rng = np.random.default_rng(11)
    K = 4
    od = ~np.eye(K, dtype=bool)
    n = 2_000_000
    q = rng.uniform(0.60, 0.90, size=K)
    pi = 0.45
    F, C, Y = simulate_panel(pi, q, q, n, seed=12)
    a = C.mean(1); b = 2 * a - 1
    A = agreements(F); c = 2 * A - 1

    # full evidence law = joint freq over all 2^K prediction patterns
    weights = (1 << np.arange(K))[::-1]
    codes = (F.T * weights).sum(1)
    patt = np.bincount(codes, minlength=2**K) / n

    # flip: Y' = 1-Y; predictions UNCHANGED; recompute correctness/advantages
    Yp = 1 - Y
    Cp = (F == Yp[None, :]).astype(np.int8)
    ap = Cp.mean(1); bp = 2 * ap - 1
    codes_p = (F.T * weights).sum(1)           # identical predictions
    patt_p = np.bincount(codes_p, minlength=2**K) / n

    out = {
        "b": b.round(5).tolist(),
        "b_flip": bp.round(5).tolist(),
        "max_abs_b_flip_plus_b": float(np.max(np.abs(bp + b))),       # ~0
        "max_abs_pattern_law_diff_TV": float(np.max(np.abs(patt - patt_p))),  # exactly 0
        "total_variation": float(0.5 * np.sum(np.abs(patt - patt_p))),
        "c_eq_bb_offdiag_err": float(np.max(np.abs((c - np.outer(b, b))[od]))),
        "c_eq_bflip_bflip_offdiag_err": float(np.max(np.abs((c - np.outer(bp, bp))[od]))),
        "benefits_opposite": bool(np.all(np.sign(b) == -np.sign(bp))),
    }
    res["V1_flip_witness"] = out
    print("[V1] b+b_flip max:", round(out["max_abs_b_flip_plus_b"], 6),
          " TV:", out["total_variation"],
          " products invariant err:", round(out["c_eq_bflip_bflip_offdiag_err"], 5))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(2**K)
    ax[0].bar(x - 0.2, patt, width=0.4, label="P (evidence law)")
    ax[0].bar(x + 0.2, patt_p, width=0.4, label="P' (label-complement)", alpha=0.7)
    ax[0].set_title("V1: full evidence (pattern) law\nTV(P,P')=0 exactly")
    ax[0].set_xlabel("prediction pattern (binary code over K=4 models)")
    ax[0].set_ylabel("frequency"); ax[0].legend(fontsize=8)
    idx = np.arange(K)
    ax[1].bar(idx - 0.2, b, width=0.4, label="b (advantages)")
    ax[1].bar(idx + 0.2, bp, width=0.4, label="b' = -b (flip)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("V1: flip sends b -> -b\n(every benefit sign flips, evidence unchanged)")
    ax[1].set_xlabel("model index j"); ax[1].set_ylabel("advantage b_j"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_v1_flip_witness.png"), dpi=120)
    plt.close(fig)
    return res

# ----------------------------------------------------------------------------- V2 rate
def part_V2(res):
    """Identification rate: recover b up to flip from agreements across a grid of true b;
    error vs m on log-log; fit slope ~ -1/2."""
    rng = np.random.default_rng(21)
    K = 4
    ms = [100, 300, 1000, 3000, 10000, 30000, 100000]
    n_grid = 12      # grid of true-b panels
    n_rep = 25       # repetitions per (panel, m)
    pi = 0.5
    errs = np.zeros((len(ms), n_grid * n_rep))
    panels = []
    for g in range(n_grid):
        # ensure margins bounded away from 0: |b_j| in [0.3,0.85], c_min bounded below
        q = rng.uniform(0.66, 0.92, size=K)
        panels.append(q)
    for mi, m in enumerate(ms):
        col = 0
        for g in range(n_grid):
            q = panels[g]
            b_true = 2 * q - 1
            for r in range(n_rep):
                F, C, Y = simulate_panel(pi, q, q, m, seed=10_000 * mi + 100 * g + r)
                A = agreements(F); c = 2 * A - 1
                b_hat = recover_b_up_to_flip(c, K, anchor_sign=+1)  # true majority>0
                # align flip (compare both global signs, take min — identification is up to flip)
                e1 = np.max(np.abs(b_hat - b_true))
                e2 = np.max(np.abs(-b_hat - b_true))
                errs[mi, col] = min(e1, e2)
                col += 1
    mean_err = errs.mean(1)
    # robust fit on the tail (drop smallest m where clipping noise dominates)
    logm = np.log(ms); loge = np.log(mean_err)
    slope_all = float(np.polyfit(logm, loge, 1)[0])
    slope_tail = float(np.polyfit(logm[2:], loge[2:], 1)[0])
    out = {"m": ms, "mean_max_err": mean_err.round(5).tolist(),
           "loglog_slope_all": slope_all, "loglog_slope_tail": slope_tail}
    res["V2_identification_rate"] = out
    print("[V2] mean err:", mean_err.round(4).tolist())
    print("[V2] loglog slope (all/tail):", round(slope_all, 3), round(slope_tail, 3), " (theory -0.5)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ms, mean_err, "o-", label="empirical max|b_hat - b| (up to flip)")
    ref = mean_err[3] * (np.array(ms) / ms[3]) ** (-0.5)
    ax.loglog(ms, ref, "k--", label="slope -1/2 reference")
    ax.set_xlabel("m (unlabeled samples on D)"); ax.set_ylabel("recovery error")
    ax.set_title(f"V2: identification of b up to flip\nfitted tail slope = {slope_tail:.2f} (theory -1/2)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_v2_identification_rate.png"), dpi=120)
    plt.close(fig)
    return res

# ----------------------------------------------------------------------------- radii
def hoeffding_ec(m, K, delta):
    """|hat c_ab - c_ab| <= e_c whp(1-delta), union over <=K^2 pairs. c=2A-1, A mean of m Bernoulli.
    |hat A - A| <= sqrt(log(2 K^2/delta)/(2m)); c-scale doubles it -> sqrt(2 log(2K^2/delta)/m)."""
    return np.sqrt(2.0 * np.log(2.0 * K * K / delta) / m)

def eb_radius_M(s_vals, alpha):
    """Empirical-Bernstein (Maurer-Pontil) radius for M_hat = mean(s)-1/2, s in [0,1] (range R=1).
    |hat M - M| <= sqrt(2 Vhat log(2/alpha)/m) + 7 R log(2/alpha)/(3(m-1))."""
    m = len(s_vals); R = 1.0
    Vhat = np.var(s_vals, ddof=1) if m > 1 else 0.0
    return np.sqrt(2.0 * Vhat * np.log(2.0 / alpha) / m) + 7.0 * R * np.log(2.0 / alpha) / (3.0 * (m - 1))

def b_radius(m, K, delta, cmin, bmin2):
    """Radius on hat b_i from product-ratio. e_c via Hoeffding; C2=(4cmin+2)/cmin^2 product-ratio Lipschitz
    (proven, verified in radius_derive2.py); sqrt step with floor bmin2."""
    ec = hoeffding_ec(m, K, delta)
    C2 = (4.0 * cmin + 2.0) / (cmin ** 2)
    db2 = C2 * ec                                  # bound on |hat b_i^2 - b_i^2|
    floor = max(bmin2 - db2, 1e-6)
    return db2 / (2.0 * np.sqrt(floor)), ec, C2

# ----------------------------------------------------------------------------- V3 audit
def part_V3(res):
    """Budget audit: level, power, bit-ambiguous blind zone, and stealth-CEI (tau=0) impossibility.
    gamma = b_a/2 - M (identity). Bit-robust audit rejects budget beta iff
    min_flip|gamma_hat| > beta + r_n, with r_n = radius(M) + radius(b_a)/2."""
    out = {}
    alpha = 0.05
    beta = 0.05
    cmin = 0.18                       # products bounded below
    bmin2 = 0.20 ** 2                 # |b| >= 0.20 floor
    K = 3                             # a=index 1 candidate vs f0=index 0; need k,l for product ratio -> K>=3
    delta = alpha                     # split simply; audit uses both radii at level alpha

    # Two radii: (A) PROVEN worst-case (sound, for the soundness theorem) and
    # (B) BOOTSTRAP plug-in (usable power; concentrates at the true estimation sd).
    # HONEST finding: the worst-case product-ratio radius constant C2=(4cmin+2)/cmin^2 is
    # extremely conservative (the sqrt-floor blows up when C2*ec > bmin2), so it gives
    # soundness but ~zero power at realistic m. The bootstrap radius restores power while
    # remaining (empirically) valid. We report BOTH.
    def gamma_hats(b_a_hat, M_hat):
        g_plus = +abs(b_a_hat) / 2.0 - M_hat
        g_minus = -abs(b_a_hat) / 2.0 - M_hat
        return g_plus, g_minus, min(abs(g_plus), abs(g_minus))

    def run_trial(pi, q0, q1, M_true, s_noise, m, seed, n_boot=80, use="boot"):
        rng = np.random.default_rng(seed)
        F, C, Y = simulate_panel(pi, q0, q1, m, seed=seed)
        A = agreements(F); c = 2 * A - 1
        b_hat = recover_b_up_to_flip(c, K, anchor_sign=+1)
        b_a_hat = b_hat[1]
        s = np.clip(rng.normal(M_true + 0.5, s_noise, size=m), 0, 1)
        M_hat = s.mean() - 0.5
        rM = eb_radius_M(s, alpha)
        rb_wc, ec, C2 = b_radius(m, K, delta, cmin, bmin2)         # PROVEN worst case
        rn_wc = rM + 0.5 * rb_wc
        # BOOTSTRAP radius on the bit-robust statistic min_flip|gamma|. Resample indices, recompute
        # the SAME statistic, take the (1-alpha) quantile of |boot - point| as the radius.
        gp0, gm0, mf0 = gamma_hats(b_a_hat, M_hat)
        if use == "boot":
            gboot = np.empty(n_boot)
            for bb in range(n_boot):
                ridx = rng.integers(0, m, m)
                Fb = F[:, ridx]
                Ab = agreements(Fb); cb = 2 * Ab - 1
                bh = recover_b_up_to_flip(cb, K, anchor_sign=+1)
                Mb = s[ridx].mean() - 0.5
                _, _, mfb = gamma_hats(bh[1], Mb)     # bootstrap of the min_flip statistic itself
                gboot[bb] = mfb
            rb_boot = float(np.quantile(np.abs(gboot - mf0), 1 - alpha))
            rn = rb_boot
        else:
            rn = rn_wc
        gp, gm, mf = gamma_hats(b_a_hat, M_hat)
        reject = bool(mf > beta + rn)
        reject_wc = bool(mf > beta + rn_wc)
        return reject, reject_wc, rn, rn_wc, (gp, gm, mf)

    rng = np.random.default_rng(303)
    n_trials = 600           # bootstrap is expensive; 600 trials keeps runtime < 40s
    m = 6000

    q = rng.uniform(0.62, 0.85, size=K)            # symmetric H, fixed across trials
    b_a = 2 * q[1] - 1

    # (i) LEVEL: TRUE gamma within budget (gamma=0 <= beta). Audit must reject <= alpha.
    M_true = b_a / 2 - 0.0
    rej = 0; rej_wc = 0
    for t in range(n_trials):
        r, rwc, rn, rnwc, _ = run_trial(0.5, q, q, M_true, 0.25, m, seed=5000 + t)
        rej += r; rej_wc += rwc
    out["level"] = {"true_gamma": 0.0, "beta": beta, "alpha": alpha,
                    "rejection_rate_boot": rej / n_trials,
                    "rejection_rate_worstcase": rej_wc / n_trials,
                    "n_trials": n_trials, "m": m,
                    "sound_boot": bool(rej / n_trials <= alpha + 0.02),
                    "sound_worstcase": bool(rej_wc / n_trials <= alpha)}
    print(f"[V3-i] LEVEL gamma=0<=beta: boot rej {rej/n_trials:.4f}, worstcase rej {rej_wc/n_trials:.4f} (target <= {alpha})")

    # (ii) POWER: world where BOTH bits exceed budget, i.e. min_flip|gamma| > beta+margin.
    # Cleanest: M_true=0 => g_plus=+b_a/2, g_minus=-b_a/2, min_flip=b_a/2. With b_a/2 > beta+margin
    # the audit can certify the budget is violated under the WHOLE flip class (bit-robust rejection).
    M_true2 = 0.0
    eff_min_flip = b_a / 2.0
    rej2 = 0; rej2_wc = 0
    for t in range(n_trials):
        r, rwc, rn, rnwc, _ = run_trial(0.5, q, q, M_true2, 0.25, m, seed=7000 + t)
        rej2 += r; rej2_wc += rwc
    out["power"] = {"min_flip_gamma_true": float(eff_min_flip), "beta": beta,
                    "rejection_rate_boot": rej2 / n_trials,
                    "rejection_rate_worstcase": rej2_wc / n_trials,
                    "powerful_boot": bool(rej2 / n_trials >= 0.8),
                    "note_worstcase": "worst-case radius is sound but too loose for power (proven-conservative)"}
    print(f"[V3-ii] POWER min_flip|gamma|={eff_min_flip:.2f}>beta: boot rej {rej2/n_trials:.4f}, "
          f"worstcase rej {rej2_wc/n_trials:.4f}")

    # (iii) BIT-AMBIGUOUS blind zone: min_flip|gamma|=0 <= beta < max_flip|gamma|. Audit must NOT reject.
    q3 = q.copy(); b_a3 = 2 * q3[1] - 1
    M_amb = b_a3 / 2.0                               # g_plus_true = 0
    rej3 = 0
    for t in range(n_trials):
        r, rwc, rn, rnwc, _ = run_trial(0.5, q3, q3, M_amb, 0.25, m, seed=9000 + t)
        rej3 += r
    out["bit_ambiguous_blind"] = {"g_plus_true": 0.0, "g_minus_true": float(-b_a3),
                                  "rejection_rate_boot": rej3 / n_trials,
                                  "correctly_blind": bool(rej3 / n_trials <= alpha + 0.02)}
    print(f"[V3-iii] BIT-AMBIGUOUS min_flip=0<=beta<max=|{b_a3:.2f}|: boot rej {rej3/n_trials:.4f} (honest blind, target low)")

    # (iv) STEALTH CEI VIOLATION with tau=0: load constructed law; audit + tau both blind, b biased.
    stealth = json.load(open(os.path.join(HERE, "stealth_law.json")))
    sf = json.load(open(os.path.join(HERE, "stealth_flip.json")))
    patterns4 = np.array(stealth["patterns"]); p4 = np.array(stealth["p"]); p4 = p4 / p4.sum()
    b_fit4 = np.array(stealth["b_fit"]); b_true4 = np.array(stealth["b_true"])
    rng2 = np.random.default_rng(404)
    mm = 200000
    idx = rng2.choice(len(p4), size=mm, p=p4)
    Cs = patterns4[idx]; Ss = 2 * Cs - 1
    Kf = 4
    c_emp = np.array([[np.mean(Ss[:, i] * Ss[:, j]) for j in range(Kf)] for i in range(Kf)])
    prods = [c_emp[0, 1] * c_emp[2, 3], c_emp[0, 2] * c_emp[1, 3], c_emp[0, 3] * c_emp[1, 2]]
    tau_emp = float(max(prods) - min(prods))
    b_rec = recover_b_up_to_flip(c_emp, Kf, anchor_sign=+1)
    b_true_emp = 2 * Cs.mean(0) - 1
    out["stealth_tau0"] = {
        "tau_emp": tau_emp,
        "b_recovered": b_rec.round(4).tolist(),
        "b_true_emp": b_true_emp.round(4).tolist(),
        "b_fit_design": b_fit4.round(4).tolist(),
        "b_true_design": b_true4.round(4).tolist(),
        "max_abs_bias": float(np.max(np.abs(b_rec - b_true_emp))),
        "decision_flip_witness": {
            "sign_recovered_b1_minus_b0": int(np.sign(sf["b_fit"][1] - sf["b_fit"][0])),
            "sign_true_b1_minus_b0": int(np.sign(sf["b_true"][1] - sf["b_true"][0])),
            "tau": sf["tau"]},
        "note": "tau~0 yet b biased and decision can flip => tau sound-not-complete; audit blind."}
    print(f"[V3-iv] STEALTH tau=0: tau_emp={tau_emp:.5f}, b biased by {out['stealth_tau0']['max_abs_bias']:.3f}, "
          f"decision flips (rec sign {out['stealth_tau0']['decision_flip_witness']['sign_recovered_b1_minus_b0']} "
          f"vs true {out['stealth_tau0']['decision_flip_witness']['sign_true_b1_minus_b0']})")

    # (v) tau RISES with OVERT correlation rho (detectable dependence): power of the diagnostic.
    # Construction that BREAKS rank-1 asymmetrically: a shared "hard-example" latent H~Bern(rho).
    # On hard examples models 1,2 are forced WRONG together (correlated errors on a 2-subset),
    # models 3,4 stay independent. This injects extra agreement into the (1,2) pair only,
    # so c_12 inflates while c_34 etc. do not -> the 2x2 minors split -> tau>0, monotone in rho.
    taus = []
    rhos = np.linspace(0.0, 0.6, 7)
    Kf = 4
    for rho in rhos:
        qf = np.array([0.78, 0.76, 0.74, 0.72])
        rng3 = np.random.default_rng(int(rho * 1000) + 1)
        nrho = 400000
        Y = (rng3.random(nrho) < 0.5).astype(np.int8)
        base = (rng3.random((Kf, nrho)) < qf[:, None]).astype(np.int8)   # independent correctness
        hard = (rng3.random(nrho) < rho)                                 # shared hard-example latent
        C = base.copy()
        C[0, hard] = 0; C[1, hard] = 0      # models 0,1 wrong TOGETHER on hard examples (correlated errors)
        S = 2 * C - 1
        c_emp = np.array([[np.mean(S[i] * S[j]) for j in range(Kf)] for i in range(Kf)])
        pr = [c_emp[0, 1] * c_emp[2, 3], c_emp[0, 2] * c_emp[1, 3], c_emp[0, 3] * c_emp[1, 2]]
        taus.append(float(max(pr) - min(pr)))
    pear = float(np.corrcoef(rhos, taus)[0, 1])
    out["tau_vs_rho"] = {"rho": rhos.round(3).tolist(), "tau": [round(t, 5) for t in taus],
                         "pearson": pear, "monotone": bool(np.all(np.diff(taus) >= -1e-4))}
    print(f"[V3-v] tau vs rho: Pearson {pear:.3f}, tau rises {taus[0]:.4f} -> {taus[-1]:.4f}, monotone={out['tau_vs_rho']['monotone']}")

    res["V3_audit"] = out

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    cats = ["(i) level\ngamma<=beta", "(ii) power\ngamma>beta+marg", "(iii) bit-ambig\n(blind)"]
    vals = [out["level"]["rejection_rate_boot"], out["power"]["rejection_rate_boot"],
            out["bit_ambiguous_blind"]["rejection_rate_boot"]]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    ax[0].bar(cats, vals, color=colors)
    ax[0].axhline(alpha, color="r", ls="--", label=f"alpha={alpha}")
    ax[0].set_ylabel("rejection rate (bootstrap audit)"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("V3: audit level / power / blind zone"); ax[0].legend(fontsize=8)
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    # stealth bias
    xb = np.arange(4)
    ax[1].bar(xb - 0.2, out["stealth_tau0"]["b_recovered"], 0.4, label="b recovered (rank-1 fit)")
    ax[1].bar(xb + 0.2, out["stealth_tau0"]["b_true_emp"], 0.4, label="b true")
    ax[1].set_title(f"V3(iv): stealth CEI, tau={tau_emp:.4f}~0\nyet b biased (impossibility)")
    ax[1].set_xlabel("model j"); ax[1].set_ylabel("advantage"); ax[1].legend(fontsize=8)
    # tau vs rho
    ax[2].plot(rhos, taus, "o-")
    ax[2].set_xlabel("error-correlation rho"); ax[2].set_ylabel("diagnostic residual tau")
    ax[2].set_title(f"V3(v): tau detects overt dependence\nPearson={pear:.3f}")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_v3_audit.png"), dpi=120)
    plt.close(fig)
    return res

# ----------------------------------------------------------------------------- V4 rate
def part_V4(res):
    """Minimax rate within H: empirical recovery error vs the Le Cam epsilon(m) curve, and
    labeled-vs-evidence-channel efficiency ratio (constants)."""
    out = {}
    K = 3
    pi = 0.5
    # Two H-models with same bit, |b - b'| = eps on candidate index 1; KL of pattern laws ~ C' m eps^2.
    # Empirically: smallest separable eps at sample m via product-ratio; should track eps ~ c/sqrt(m).
    ms = [200, 500, 1500, 5000, 15000, 50000]
    n_rep = 200
    base_q = np.array([0.78, 0.72, 0.70])
    eps_emp = []
    for m in ms:
        # distinguishability threshold: eps s.t. |bhat-b| ~ eps half the time. Use std of bhat_1.
        bh1 = np.zeros(n_rep)
        for r in range(n_rep):
            F, C, Y = simulate_panel(pi, base_q, base_q, m, seed=20000 + 13 * len(eps_emp) + r)
            A = agreements(F); c = 2 * A - 1
            b = recover_b_up_to_flip(c, K, anchor_sign=+1)
            bh1[r] = b[1]
        eps_emp.append(float(np.std(bh1)))     # estimation sd ~ minimax eps scale
    eps_emp = np.array(eps_emp)
    logm = np.log(ms)
    slope = float(np.polyfit(logm, np.log(eps_emp), 1)[0])
    # Le Cam predicted curve: eps(m) = c0 / sqrt(m); fit c0 from data
    c0 = float(np.median(eps_emp * np.sqrt(ms)))
    lecam = c0 / np.sqrt(np.array(ms))
    out["minimax_rate"] = {"m": ms, "eps_emp": eps_emp.round(5).tolist(),
                           "loglog_slope": slope, "c0": c0,
                           "lecam_curve": lecam.round(5).tolist()}
    print(f"[V4] minimax eps slope {slope:.3f} (theory -0.5); Le Cam c0={c0:.3f}")

    # KL between two K=3 pattern laws with b vs b' (eps apart) ~ C' m eps^2: verify per-sample KL ~ C eps^2.
    def pattern_law(b, pi=0.5):
        # under H: advantages b -> per-class symmetric q=(1+b)/2; CEI => pattern prob factorizes given Y
        K = len(b); q = (1 + b) / 2
        probs = np.zeros(2 ** K)
        pats = np.array(list(__import__("itertools").product([0, 1], repeat=K)))
        for y in [0, 1]:
            py = 0.5
            for idx, pat in enumerate(pats):
                # pat = prediction values; correctness = 1[pat==y]; P(f_i=v|Y=y): if v==y -> q_i else 1-q_i
                pr = 1.0
                for i in range(K):
                    pr *= q[i] if pat[i] == y else (1 - q[i])
                probs[idx] += py * pr
        return probs
    b0 = np.array([0.5, 0.4, 0.3]); kls = []
    epss = [0.02, 0.04, 0.08, 0.16]
    for e in epss:
        b1 = b0.copy(); b1[1] += e
        P = pattern_law(b0); Q = pattern_law(b1)
        kl = float(np.sum(P * (np.log(P + 1e-15) - np.log(Q + 1e-15))))
        kls.append(kl)
    ratio = [kls[i] / epss[i] ** 2 for i in range(len(epss))]   # KL/eps^2 ~ const
    out["kl_quadratic"] = {"eps": epss, "kl": [round(k, 6) for k in kls],
                           "kl_over_eps2": [round(r, 4) for r in ratio],
                           "approx_const": float(np.mean(ratio))}
    print(f"[V4] KL/eps^2 ~ const: {[round(r,3) for r in ratio]} (per-sample; m KL = m*this)")

    # labeled vs evidence-channel efficiency: labeled benefit estimator var vs evidence-channel var.
    # labeled: hat Delta = mean of paired benefits (var sigma^2/m); evidence: product-ratio (var via delta method).
    # Compute empirical ratio of estimation sd for sign(b_a-b_0) at fixed m, fixed truth.
    m = 5000; reps = 400
    q = np.array([0.70, 0.62, 0.66]); pi = 0.5
    diff_lab = np.zeros(reps); diff_ev = np.zeros(reps)
    for r in range(reps):
        F, C, Y = simulate_panel(pi, q, q, m, seed=30000 + r)
        # labeled paired benefit on D: delta_i = 1[f_a=Y]-1[f_0=Y], candidate a=1 vs f0=0
        lab = (C[1].astype(float) - C[0].astype(float))   # uses labels Y
        diff_lab[r] = lab.mean()                            # estimates b_1-b_0 ( = 2(a1-a0) )/... actually mean delta = a1-a0
        A = agreements(F); c = 2 * A - 1
        b = recover_b_up_to_flip(c, K, anchor_sign=+1)
        diff_ev[r] = (b[1] - b[0]) / 2.0                    # b/2 scale matches a-difference
    sd_lab = float(np.std(diff_lab)); sd_ev = float(np.std(diff_ev))
    out["labeled_vs_evidence"] = {"sd_labeled": sd_lab, "sd_evidence": sd_ev,
                                  "efficiency_ratio_sd": sd_ev / sd_lab,
                                  "m": m,
                                  "note": "ratio ~ constant => labels buy constants not rate"}
    print(f"[V4] labeled sd {sd_lab:.4f} vs evidence sd {sd_ev:.4f}, ratio {sd_ev/sd_lab:.2f}")

    res["V4_rate"] = out
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].loglog(ms, eps_emp, "o-", label="empirical minimax eps (estimation sd)")
    ax[0].loglog(ms, lecam, "k--", label=f"Le Cam c0/sqrt(m), c0={c0:.2f}")
    ax[0].set_xlabel("m"); ax[0].set_ylabel("eps(m)")
    ax[0].set_title(f"V4: minimax rate within H\nslope {slope:.2f} (theory -1/2)")
    ax[0].legend(fontsize=8); ax[0].grid(True, which="both", alpha=0.3)
    ax[1].plot(epss, kls, "o-", label="KL(P_b || P_b')")
    ax[1].plot(epss, [out["kl_quadratic"]["approx_const"] * e ** 2 for e in epss], "k--",
               label=f"{out['kl_quadratic']['approx_const']:.2f} * eps^2")
    ax[1].set_xlabel("eps = |b - b'|"); ax[1].set_ylabel("per-sample KL")
    ax[1].set_title("V4: KL ~ eps^2 (Le Cam quadratic)\n=> n^{-1/2} minimax")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_v4_rate.png"), dpi=120)
    plt.close(fig)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["H", "V1", "V2", "V3", "V4", "all"])
    args = ap.parse_args()
    res = load_results()
    res.setdefault("_meta", {})["setting"] = (
        "binary Y on D; K predictors; b_j=2P(f_j=Y|D)-1; A_ij=P(f_i=f_j|D); c_ij=2A_ij-1; "
        "H=per-class-symmetric-accuracy + CEI => c_ij=b_i b_j; deficit pi(1-pi) delta_i delta_j")
    if args.part in ("H", "all"):
        res = part_H(res)
    if args.part in ("V1", "all"):
        res = part_V1(res)
    if args.part in ("V2", "all"):
        res = part_V2(res)
    if args.part in ("V3", "all"):
        res = part_V3(res)
    if args.part in ("V4", "all"):
        res = part_V4(res)
    save_results(res)
    print("saved ->", JSON_PATH)
