"""
analysis.py - torch-free K-Bound analysis core for the WILDS Camelyon17 pipeline.

Routing variants implemented:
  (a) SINGLE-candidate certificate   -> decide_kga (LOO gradient-boosted B_hat(Z)
      + exact-rank cross-fitted empirical radius eps; ADAPT/FREEZE/ABSTAIN). Identical machinery to
      the rest of the paper (cifar_tent_mps_v2.decide_kga / run_wilds_camelyon17).
  (b) MULTI-candidate route [Theorem 1A, tau-residual] -> multicandidate_route,
      which uses rankone_fit_offdiag / minor_estimator / overdet_residual to
      recover per-candidate advantages from the label-free
      pairwise-agreement matrix on the disagreement region and tau-gate the commit.
  (c) SMOOTH-DRIFT route [Theorem 1B] -> smooth_drift_route, an explicitly
      diagnostic Brier-view bracket with a frozen-model source surrogate.

DETECTABILITY: detectability_analysis correlates each label-free Z feature (and the
LOO B_hat) with the TRUE benefit sign -> tells us whether harm, if it occurs, is
detectable label-free.  INTEGRITY: labels are used ONLY to compute B / oracle /
these correlations for evaluation; the router only ever sees Z or agreements.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from calibration import exact_rank_radius
from theory_primitives import (
    minor_estimator,
    overdet_residual,
    rankone_fit_offdiag,
    w2_gaussian,
)

ALPHA = 0.10
SEED = 0

# ============================ (a) single-candidate KGA =======================
def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED):
    """LOO gradient-boosted estimator + exact-rank empirical radius.
    Returns (Bhat, eps, decisions in {ADAPT,FREEZE,ABSTAIN})."""
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i + 1])[0]
    eps = exact_rank_radius(np.abs(Bhat - B), alpha)
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec


def policy_metrics(dec, a0, aa, B=None, alpha=ALPHA):
    """Realized accuracy + regret vs oracle for each policy. ABSTAIN/FREEZE -> frozen.
    beats_both REQUIRES the pre-registered false-adapt budget FA<=alpha, not regret alone."""
    a0 = np.asarray(a0, float); aa = np.asarray(aa, float)
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    if B is None:
        B = aa - a0
    B = np.asarray(B, float)
    return {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "abstention_rate": float(np.mean(dec == "ABSTAIN")),
        "adapt_precision_B>0": float(np.mean(B[adapt] > 0)) if adapt.any() else None,
        "false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
        "mean_acc": {
            "always_adapt": float(aa.mean()), "always_freeze": float(a0.mean()),
            "K_Bound": float(kga.mean()), "oracle": float(oracle.mean()),
        },
        "regret_vs_oracle": {
            "always_adapt": float((oracle - aa).mean()),
            "always_freeze": float((oracle - a0).mean()),
            "K_Bound": float((oracle - kga).mean()),
        },
        "worst_case_acc": {"always_adapt": float(aa.min()), "always_freeze": float(a0.min()),
                           "K_Bound": float(kga.min())},
        "alpha_false_adapt_budget": float(alpha),
        "beats_both_regret_only": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                                       (oracle - kga).mean() < (oracle - a0).mean() - 1e-9),
        "beats_both": bool((oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
                           (oracle - kga).mean() < (oracle - a0).mean() - 1e-9 and
                           adapt.any() and float(np.mean(B[adapt] < 0)) <= alpha),
    }


def label_regime(B, thr=0.02):
    return "helpful" if B > thr else ("harmful" if B < -thr else "marginal")


# ============================ detectability ==================================
def _auc(score, label):
    """AUC via Mann-Whitney U (prob. that a positive outranks a negative).
    label: 1 = harmful (B<0) [the event we want to detect], 0 = not harmful."""
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[label == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def detectability_analysis(records, evidence_names, alpha=ALPHA):
    """Does the label-free evidence Z reveal the TRUE benefit sign?

    records: list of dicts with keys 'Z' (list), 'B' (float).  Per Z-feature we report
    Pearson corr with B, point-biserial corr with the harmful event (B<0), and the
    single-feature AUC for detecting harm (the feature is sign-flipped so AUC>=.5 means
    'higher value => more harmful').  We also report the AUC of the LOO B_hat(Z) harm
    predictor (-B_hat as harm score) -> the operational detectability of the certificate.
    """
    Z = np.array([r["Z"] for r in records], float)
    B = np.array([r["B"] for r in records], float)
    harmful = (B < 0).astype(int)
    n, d = Z.shape
    out = {"n_cells": int(n), "n_harmful": int(harmful.sum()),
           "base_rate_harmful": float(harmful.mean()), "mean_B": float(B.mean()),
           "per_feature": {}}
    for k in range(d):
        zk = Z[:, k]
        pear = float(np.corrcoef(zk, B)[0, 1]) if np.std(zk) > 1e-12 else 0.0
        # orient harm-score so that higher => more harmful, then AUC
        a_pos = _auc(zk, harmful); a_neg = _auc(-zk, harmful)
        best = max([a for a in (a_pos, a_neg) if a is not None], default=None)
        name = evidence_names[k] if k < len(evidence_names) else f"z{k}"
        out["per_feature"][name] = {"pearson_corr_B": pear, "harm_AUC": best}
    # operational: LOO B_hat as harm detector
    if n >= 4 and harmful.sum() not in (0, n):
        Bhat, eps, dec = decide_kga(Z, B, alpha=alpha)
        out["certificate_harm_AUC_negBhat"] = _auc(-Bhat, harmful)
        out["certificate_eps"] = float(eps)
    # headline: is harm detectable at all?
    aucs = [v["harm_AUC"] for v in out["per_feature"].values() if v["harm_AUC"] is not None]
    out["best_single_feature_harm_AUC"] = float(max(aucs)) if aucs else None
    if out["best_single_feature_harm_AUC"] is not None:
        out["detectability_verdict"] = (
            "detectable" if out["best_single_feature_harm_AUC"] >= 0.75 else
            ("weak" if out["best_single_feature_harm_AUC"] >= 0.6 else "undetectable"))
    else:
        out["detectability_verdict"] = "n/a (no regime variation)"
    return out


# ================= (b) multi-candidate route [Theorem 1A, tau] ===============
def agreement_matrix(preds_all, D):
    """Centered pairwise prediction-agreement matrix C on region D (label-free).
    preds_all: (M, N) int predictions; row 0 = anchor (frozen f0).  C_ij = 2*Pr(f_i=f_j|D)-1."""
    P = preds_all[:, D]
    M = P.shape[0]
    eq = (P[:, None, :] == P[None, :, :]).mean(axis=2)
    C = 2.0 * eq - 1.0
    np.fill_diagonal(C, 0.0)
    return C


def multicandidate_route(preds_all, tau_star=0.08, kappa=2.5, min_D=8):
    """Theorem 1A tau-residual route over M = 1(anchor f0) + K adapted candidates.

    Label-free: uses ONLY prediction agreements on the disagreement region
    D = {x : the M predictions are not unanimous}.  Recovers advantages b_hat via
    rank-one fit (vmc.rankone_fit_offdiag) and median-of-minors (vmc.minor_estimator),
    with f0 as the above-chance anchor (index 0).  COMMIT to the adapted candidate of
    largest advantage iff tau<=tau* AND its advantage beats the anchor by the observed
    margin; else FREEZE.  tau>tau* (Def-5 violation certified) => ABSTAIN.
    Returns a dict; 'choice' is the adapted-candidate index (>=1) or None."""
    M, N = preds_all.shape
    if M < 4:
        return {"decision": "ABSTAIN", "reason": f"need M>=4 candidates incl anchor for tau route; got M={M}",
                "choice": None, "n_D": 0}
    unanimous = (preds_all == preds_all[0:1, :]).all(axis=0)
    D = np.where(~unanimous)[0]
    if len(D) < min_D:
        return {"decision": "FREEZE", "reason": f"|D|={len(D)} < min_D={min_D}; candidates ~agree, no signal",
                "choice": None, "n_D": int(len(D))}
    C = agreement_matrix(preds_all, D)
    b_hat, tau = rankone_fit_offdiag(C)
    try:
        b_tilde = minor_estimator(C)
    except Exception:
        b_tilde = b_hat
    off = ~np.eye(M, dtype=bool)
    h_hat = float(np.abs(C - np.outer(b_hat, b_hat))[off].max())
    margin = kappa * h_hat + 2.0 / np.sqrt(len(D))
    gate = bool(tau <= tau_star)
    od = float(overdet_residual(C)) if M >= 4 else 0.0
    res = {"tau": float(tau), "tau_star": float(tau_star), "gate_pass": gate,
           "overdet_residual": od, "h_hat": h_hat, "margin": float(margin),
           "b_hat": [float(x) for x in b_hat], "b_tilde": [float(x) for x in b_tilde],
           "anchor_b0": float(b_hat[0]), "n_D": int(len(D)), "M": int(M)}
    if not gate:
        res.update({"decision": "ABSTAIN", "choice": None,
                    "reason": f"tau={tau:.4f} > tau*={tau_star} certifies Def-5 violated"})
        return res
    adv = b_hat[1:] - b_hat[0]                       # advantage of each adapted cand over anchor
    committed = [i + 1 for i in range(M - 1) if adv[i] > margin and b_hat[i + 1] > 0]
    if not committed:
        res.update({"decision": "FREEZE", "choice": None,
                    "reason": "no candidate beats anchor f0 by the observed margin"})
        return res
    choice = int(max(committed, key=lambda i: b_hat[i]))
    res.update({"decision": "ADAPT", "choice": choice, "committed": committed,
                "reason": f"candidate {choice} commits (b_hat={b_hat[choice]:.3f} > anchor {b_hat[0]:.3f}+margin)"})
    return res


# ================= (c) smooth-drift route [Theorem 1B] =======================
def smooth_drift_route(f0_pos, fa_pos, stream_f0_pos, L=0.6):
    """Theorem 1B smooth-drift diagnostic bracket on real Camelyon17:
    the observable boundary
        center c = U - 2 T_S ,  reach rho = 2 (L d) W ,
        COMMIT sign(c) iff |c| > rho + eps_n  else ABSTAIN,  bracket [c-rho-eps, c+rho+eps].

    Binary classification is mapped to vsd's squared-loss (Brier) setting via the
    positive-class probabilities: f0:=P0(y=1), fa:=Pa(y=1) on the eval set (label-free);
    U=E[(f0-fa)(f0+fa)], W=E|f0-fa|.  The covariate discrepancy d is the OBSERVABLE
    stream-vs-eval shift (Gaussian W2 on frozen P(y=1) moments).

    HONEST SURROGATE / DIAGNOSTIC: g_S (source concept) is approximated by the frozen
    source model f0, which makes the center conservative (c = -E[(f0-fa)^2] <= 0).  A
    non-degenerate center needs an f0-independent source-concept estimate (the paper's
    deep-classification instantiation of 1B).  So (c) is reported as a diagnostic bracket,
    not yet a certified commit."""
    f0p = np.asarray(f0_pos, float); fap = np.asarray(fa_pos, float); n = len(f0p)
    dp = f0p - fap; sp = f0p + fap
    U = float(np.mean(dp * sp)); W = float(np.mean(np.abs(dp)))
    T_S = float(np.mean(dp * f0p))                       # g_S := f0 (conservative surrogate)
    c = U - 2.0 * T_S
    s = np.asarray(stream_f0_pos, float)
    d_obs = float(w2_gaussian(float(s.mean()), float(s.std() + 1e-9),
                              float(f0p.mean()), float(f0p.std() + 1e-9)))
    reach = 2.0 * L * d_obs * W
    eps_n = 2.0 / np.sqrt(max(n, 1))
    lo, hi = c - reach - eps_n, c + reach + eps_n
    dec = ("ADAPT" if c > 0 else "FREEZE") if abs(c) > reach + eps_n else "ABSTAIN"
    return {"decision": dec, "implemented": True, "theorem": "1B",
            "view": "brier_squared_loss", "gS_estimate": "f0_surrogate(conservative)",
            "center_c": c, "U": U, "T_S": T_S, "W": W, "d_obs": d_obs, "L": float(L),
            "reach": reach, "eps_n": float(eps_n), "bracket": [lo, hi],
            "note": "DIAGNOSTIC: g_S~f0 makes center conservative; full 1B classification "
                    "instantiation needs an f0-independent source-concept estimate."}
