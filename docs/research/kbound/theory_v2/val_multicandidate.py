#!/usr/bin/env python3
r"""
val_multicandidate.py
=====================

NUMERICAL VALIDATOR for the K-Bound MULTICANDIDATE certified-routing theorem.

Claim under test
----------------
Single-candidate certificate (thm:cert): for one adapter fa with benefit
    Delta = R_T(f0) - R_T(fa),
an estimate Delta_hat and a conformal radius eps give a marginal lower-confidence
bound  L = Delta_hat - eps  with one-sided coverage
    P(Delta >= L) >= 1 - alpha,
and the rule "ADAPT iff L > 0" has false-adapt  FA = P(ADAPT and Delta <= 0) <= alpha.

Multicandidate problem (the GAP this file demonstrates and then closes):
with K candidates {fa^(1),...,fa^(K)}, naive selection
    k_hat = argmax_k L_k = argmax_k (Delta_hat_k - eps_k(alpha)),   commit iff L_k_hat > 0
INFLATES the family-wise false-adapt
    FA_fw = P( committed candidate k_hat is harmful, i.e. Delta_{k_hat} <= 0 )
above alpha as K grows (a selection / multiple-comparisons effect).

Corrected rule (TARGET THEOREM): run each per-candidate certificate at the
Bonferroni-corrected level alpha/K (or the Sidak level 1-(1-alpha)^(1/K)), i.e.
    L_k = Delta_hat_k - eps_k(alpha/K),
form the certified-helpful set  S = { k : L_k > 0 }, and commit ANY one member of S
(here argmax_k L_k), else ABSTAIN. Then for ANY selection rule
    FA_fw = P( a committed candidate is harmful ) <= alpha.

This script demonstrates:
  (A) NAIVE selection: FA_fw climbs well above alpha as K grows (the gap), even
      though each *individual* per-candidate certificate is valid at level alpha.
  (B) BONFERRONI-corrected routing: FA_fw <= alpha for all K (selection-proof).
  (C) SIDAK-corrected routing: also <= alpha, slightly more power (smaller radius)
      when the per-candidate coverage failures are independent.
  (D) ADVERSARIAL selection (an oracle that deliberately commits a harmful
      certified candidate whenever one exists) still obeys FA_fw <= alpha under the
      correction -- confirming the bound holds for the WORST selection map, not just
      argmax. This is the load-bearing "selection-proof containment" claim.
  (E) the PRICE: corrected routing abstains more / has lower true-positive
      (commit-a-helpful-candidate) rate than naive -- power is the currency paid.

Everything is exact split-conformal: per trial we draw a fresh calibration sample
and a fresh deployment point, fit the radius on calibration ONLY, and decide on the
deployment point. No labels leak from deployment into the radius. FA_fw is the
Monte-Carlo frequency of "committed a harmful candidate" over independent trials.

Self-contained: numpy only. Run:
    python3 val_multicandidate.py
"""

from __future__ import annotations

import json
import numpy as np

# --------------------------------------------------------------------------- #
# Conformal lower-confidence bound for a single candidate (split conformal).
# --------------------------------------------------------------------------- #
# Model of one candidate's benefit evidence:
#   The benefit Delta_k is a fixed (per-trial) scalar. We observe n_cal i.i.d.
#   calibration "paired-benefit" measurements  d_{k,1..n} ~ N(Delta_k, tau^2)
#   (a stand-in for per-cell benefit estimates on a labeled calibration split),
#   and one deployment estimate  Delta_hat_k ~ N(Delta_k, tau^2)  from the same law.
#   We want a one-sided lower bound L_k = Delta_hat_k - eps with
#       P(Delta_k >= L_k) >= 1 - level,  i.e.  P(Delta_hat_k - Delta_k > eps) <= level.
#
#   Split-conformal radius: the nonconformity score is s = Delta_hat - Delta (the
#   signed error). Under exchangeability of the n_cal calibration errors and the
#   one deployment error, the one-sided split-conformal quantile
#       eps(level) = the ceil((1-level)(n_cal+1))-th smallest of the calibration
#                    |errors|  (two-sided, conservative) OR signed upper quantile
#   gives finite-sample coverage. We use the standard two-sided absolute-residual
#   quantile (|Delta_hat - Delta|), which yields P(|Delta_hat-Delta|<=eps)>=1-level
#   and hence the one-sided event at level <= level. This matches thm:cert's radius
#   (|Delta_hat - Delta| <= eps with prob >= 1-alpha).
#
#   Because the calibration errors are observable here (we know Delta_k in the
#   simulation), we can compute the exact split-conformal radius. In deployment
#   Delta_k is unknown, but the radius is fit on a *labeled* calibration split where
#   the analogous errors ARE observed -- exactly the K-Bound protocol.

def split_conformal_radius(cal_errors_abs: np.ndarray, level: float) -> float:
    """One-sided/absolute split-conformal radius at miscoverage `level`.

    cal_errors_abs : |Delta_hat - Delta| on the calibration split (n_cal values).
    Returns eps such that, adding the fresh deployment |error| as exchangeable,
    P(|Delta_hat - Delta| <= eps) >= 1 - level  (finite-sample, exact under
    exchangeability via the rank/quantile inflation (n+1)).
    """
    n = cal_errors_abs.shape[0]
    # rank index for the conformal quantile: ceil((1-level)(n+1)) th order statistic
    k = int(np.ceil((1.0 - level) * (n + 1)))
    if k > n:
        # not enough calibration points to certify at this level -> infinite radius
        # (HONEST behaviour: the certificate abstains when the corrected level alpha/K
        #  cannot be met by the calibration sample; FA stays controlled trivially.)
        return np.inf
    # k-th smallest (1-indexed) absolute residual
    return float(np.partition(cal_errors_abs, k - 1)[k - 1])


def gaussian_exact_radius(tau: float, level: float) -> float:
    """Two-sided EXACT radius when the estimation error Delta_hat - Delta ~ N(0, tau^2).

    eps(level) = z_{1-level/2} * tau  gives  P(|Delta_hat - Delta| <= eps) = 1 - level
    with NO finite-sample floor, so the corrected rule can still commit at large K.
    Uses the inverse-erf normal quantile (numpy only, no scipy dependency).
    """
    from math import erf  # noqa: F401 (kept for clarity; we use erfinv below)
    # standard normal quantile via inverse error function
    # z_{1-level/2} = sqrt(2) * erfinv(1 - level)
    # numpy has no erfinv; use a high-accuracy rational approx (Acklam) for the
    # normal inverse CDF, valid across the full (0,1) range we need.
    return float(_norm_ppf(1.0 - level / 2.0) * tau)


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm); numpy-free, ~1e-9 accurate."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


# --------------------------------------------------------------------------- #
# One Monte-Carlo trial (READABLE REFERENCE implementation).
# NOTE: the EXECUTED path is the vectorized run_sweep() below, which implements
# IDENTICAL per-trial logic batched over trials for speed. one_trial() is kept as
# a transparent, line-by-line statement of what each trial does, for verification.
# --------------------------------------------------------------------------- #
def one_trial(rng, K, n_cal, tau, alpha, delta_mean, delta_sd, harmful_frac,
              radius_mode="conformal"):
    """
    Draw K candidate benefits, simulate calibration + deployment, and return,
    for each rule, whether it (committed) and whether the committed candidate was
    harmful (Delta<=0). Returns a dict of bits.

    delta_mean, delta_sd : Gaussian prior on the true benefits Delta_k.
    harmful_frac         : fraction of candidates *forced* to be harmful (Delta<=0)
                           by shifting their mean negative, to stress the selection
                           effect (the regime the paper says inflates FA).
    radius_mode          : "conformal" -> finite-sample split-conformal radius from
                           the calibration sample (can hit the inf-radius floor at
                           large K, which is honest abstention); "gaussian_exact" ->
                           the exact N(0,tau^2) two-sided radius z_{1-l/2}*tau, which
                           has NO finite-sample floor so the corrected rule keeps
                           committing at large K and FA-control is shown WHILE
                           committing (the strongest demonstration).
    """
    # ---- true (unknown at deploy) benefits for the K candidates ----
    Delta = rng.normal(delta_mean, delta_sd, size=K)
    n_harm = int(round(harmful_frac * K))
    if n_harm > 0:
        # force the first n_harm candidates to be (mildly) harmful: Delta just <= 0.
        # Mildly harmful is the hardest case -- large |Delta| harmful candidates are
        # easy to reject; near-zero harmful candidates are where false-adapt leaks.
        Delta[:n_harm] = -np.abs(rng.normal(0.0, 0.5 * delta_sd, size=n_harm)) - 1e-9
    # shuffle so the harmful ones aren't always the same indices
    perm = rng.permutation(K)
    Delta = Delta[perm]

    # ---- calibration errors and deployment estimate, per candidate ----
    # errors ~ N(0, tau^2); deployment estimate Delta_hat = Delta + err_deploy
    cal_err = rng.normal(0.0, tau, size=(K, n_cal))          # signed cal errors
    cal_abs = np.abs(cal_err)
    err_deploy = rng.normal(0.0, tau, size=K)
    Delta_hat = Delta + err_deploy

    # ---- radii at the marginal level (naive) and corrected levels ----
    lvl_naive = alpha
    lvl_bonf = alpha / K
    lvl_sidak = 1.0 - (1.0 - alpha) ** (1.0 / K)

    if radius_mode == "gaussian_exact":
        eps_naive = np.full(K, gaussian_exact_radius(tau, lvl_naive))
        eps_bonf = np.full(K, gaussian_exact_radius(tau, lvl_bonf))
        eps_sidak = np.full(K, gaussian_exact_radius(tau, lvl_sidak))
    else:
        eps_naive = np.array([split_conformal_radius(cal_abs[k], lvl_naive) for k in range(K)])
        eps_bonf = np.array([split_conformal_radius(cal_abs[k], lvl_bonf) for k in range(K)])
        eps_sidak = np.array([split_conformal_radius(cal_abs[k], lvl_sidak) for k in range(K)])

    L_naive = Delta_hat - eps_naive
    L_bonf = Delta_hat - eps_bonf
    L_sidak = Delta_hat - eps_sidak

    out = {}

    # ---------- NAIVE: argmax of marginal lower bound, commit if > 0 ----------
    k_hat = int(np.argmax(L_naive))
    commit = L_naive[k_hat] > 0.0
    out["naive_commit"] = bool(commit)
    out["naive_falseadapt"] = bool(commit and Delta[k_hat] <= 0.0)
    out["naive_truepos"] = bool(commit and Delta[k_hat] > 0.0)

    # ---------- BONFERRONI: certified set at alpha/K, argmax member ----------
    S = np.where(L_bonf > 0.0)[0]
    commit_b = S.size > 0
    if commit_b:
        kb = int(S[np.argmax(L_bonf[S])])
    out["bonf_commit"] = bool(commit_b)
    out["bonf_falseadapt"] = bool(commit_b and Delta[kb] <= 0.0)
    out["bonf_truepos"] = bool(commit_b and Delta[kb] > 0.0)

    # ---------- SIDAK: certified set at 1-(1-alpha)^(1/K), argmax member ------
    Ss = np.where(L_sidak > 0.0)[0]
    commit_s = Ss.size > 0
    if commit_s:
        ks = int(Ss[np.argmax(L_sidak[Ss])])
    out["sidak_commit"] = bool(commit_s)
    out["sidak_falseadapt"] = bool(commit_s and Delta[ks] <= 0.0)
    out["sidak_truepos"] = bool(commit_s and Delta[ks] > 0.0)

    # ---------- ADVERSARIAL selection under the BONFERRONI correction ----------
    # The worst possible selection map: if ANY harmful candidate is certified
    # (L_bonf>0 and Delta<=0), commit THAT one (maximally trying to break FA).
    # The theorem claims FA_fw <= alpha even for THIS map -- the strongest test.
    harmful_certified = np.where((L_bonf > 0.0) & (Delta <= 0.0))[0]
    if harmful_certified.size > 0:
        out["adv_commit"] = True
        out["adv_falseadapt"] = True       # by construction it picked a harmful one
    else:
        # no harmful candidate is certified: fall back to committing a certified
        # helpful one if present (so it still "commits" when it safely can)
        Sh = np.where(L_bonf > 0.0)[0]
        out["adv_commit"] = bool(Sh.size > 0)
        out["adv_falseadapt"] = False
    out["adv_truepos"] = bool(out["adv_commit"] and not out["adv_falseadapt"])

    # also record whether at least one harmful candidate existed (for context)
    out["any_harmful"] = bool(np.any(Delta <= 0.0))
    return out


# --------------------------------------------------------------------------- #
# Vectorized sweep over K (same logic as one_trial, batched over trials).
# --------------------------------------------------------------------------- #
def _conformal_radii_batch(cal_abs, level):
    """Split-conformal radius for each (trial,candidate): the k-th smallest abs
    calibration residual, k=ceil((1-level)(n_cal+1)). cal_abs: (T,K,n_cal).
    Returns (T,K) radii; inf where k>n_cal (cannot certify at this level)."""
    T, K, n_cal = cal_abs.shape
    k = int(np.ceil((1.0 - level) * (n_cal + 1)))
    if k > n_cal:
        return np.full((T, K), np.inf)
    s = np.sort(cal_abs, axis=2)            # ascending
    return s[:, :, k - 1]                   # k-th smallest (1-indexed)


def run_sweep(Ks, n_trials, n_cal, tau, alpha, delta_mean, delta_sd, harmful_frac, seed,
              radius_mode="conformal"):
    """Vectorized Monte-Carlo. For each K, draw all `n_trials` at once and compute
    NAIVE / BONFERRONI / SIDAK / ADVERSARIAL outcomes. Equivalent to looping
    one_trial() but ~100x faster (kept fast so it finishes within the sandbox window).
    """
    rng = np.random.default_rng(seed)
    rows = []
    NEG = -1e18  # sentinel for masked-out (uncertified) lower bounds in argmax
    for K in Ks:
        T = n_trials
        # ---- true benefits Delta: (T,K) ----
        Delta = rng.normal(delta_mean, delta_sd, size=(T, K))
        n_harm = int(round(harmful_frac * K))
        if n_harm > 0:
            harm = -np.abs(rng.normal(0.0, 0.5 * delta_sd, size=(T, n_harm))) - 1e-9
            Delta[:, :n_harm] = harm
        # independent per-row permutation so harmful indices vary across trials
        order = np.argsort(rng.random((T, K)), axis=1)
        Delta = np.take_along_axis(Delta, order, axis=1)

        # ---- deployment estimates and (if needed) calibration residuals ----
        Delta_hat = Delta + rng.normal(0.0, tau, size=(T, K))

        lvl_naive = alpha
        lvl_bonf = alpha / K
        lvl_sidak = 1.0 - (1.0 - alpha) ** (1.0 / K)

        if radius_mode == "gaussian_exact":
            eps_naive = np.full((T, K), gaussian_exact_radius(tau, lvl_naive))
            eps_bonf = np.full((T, K), gaussian_exact_radius(tau, lvl_bonf))
            eps_sidak = np.full((T, K), gaussian_exact_radius(tau, lvl_sidak))
        else:
            # chunk over trials so the (chunk,K,n_cal) calibration tensor stays small
            eps_naive = np.empty((T, K)); eps_bonf = np.empty((T, K))
            eps_sidak = np.empty((T, K))
            chunk = max(1, int(2_000_000 / max(K * n_cal, 1)))   # ~2M cells per chunk
            for s0 in range(0, T, chunk):
                s1 = min(s0 + chunk, T)
                cal_abs = np.abs(rng.normal(0.0, tau, size=(s1 - s0, K, n_cal)))
                eps_naive[s0:s1] = _conformal_radii_batch(cal_abs, lvl_naive)
                eps_bonf[s0:s1] = _conformal_radii_batch(cal_abs, lvl_bonf)
                eps_sidak[s0:s1] = _conformal_radii_batch(cal_abs, lvl_sidak)

        L_naive = Delta_hat - eps_naive
        L_bonf = Delta_hat - eps_bonf
        L_sidak = Delta_hat - eps_sidak

        harmful = Delta <= 0.0
        rowcount = {}

        def summarize(L, prefix):
            """argmax-of-certified-helpful-set rule (NAIVE uses the same argmax with
            the whole set; commit iff its max LCB > 0)."""
            Lm = np.where(L > 0.0, L, NEG)           # only certified-helpful eligible
            kstar = np.argmax(Lm, axis=1)            # (T,)
            best = Lm[np.arange(T), kstar]
            commit = best > NEG / 2                  # some candidate had L>0
            chosen_harm = harmful[np.arange(T), kstar]
            fa = commit & chosen_harm
            tp = commit & ~chosen_harm
            rowcount[prefix + "_commit"] = float(np.mean(commit))
            rowcount[prefix + "_falseadapt"] = float(np.mean(fa))
            rowcount[prefix + "_truepos"] = float(np.mean(tp))

        summarize(L_naive, "naive")
        summarize(L_bonf, "bonf")
        summarize(L_sidak, "sidak")

        # ADVERSARIAL under the alpha/K correction: commit a harmful candidate iff
        # ANY harmful candidate is certified (L_bonf>0 & Delta<=0); else commit a
        # certified-helpful one if present.
        cert = L_bonf > 0.0
        adv_fa = np.any(cert & harmful, axis=1)
        adv_commit = np.any(cert, axis=1)
        rowcount["adv_commit"] = float(np.mean(adv_commit))
        rowcount["adv_falseadapt"] = float(np.mean(adv_fa))
        rowcount["adv_truepos"] = float(np.mean(adv_commit & ~adv_fa))
        rowcount["any_harmful"] = float(np.mean(np.any(harmful, axis=1)))

        row = {"K": K}
        for kk, v in rowcount.items():
            row[kk + "_rate"] = v

        def hw(p):
            return 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n_trials)
        row["naive_fa_ci"] = hw(row["naive_falseadapt_rate"])
        row["bonf_fa_ci"] = hw(row["bonf_falseadapt_rate"])
        row["sidak_fa_ci"] = hw(row["sidak_falseadapt_rate"])
        row["adv_fa_ci"] = hw(row["adv_falseadapt_rate"])
        rows.append(row)
    return rows


def report_regime(name, rows, alpha):
    """Print FA + power tables for one regime; return (naive_breaks, corrected_ok,
    max-FA dict)."""
    print()
    print("#" * 92)
    print(f"REGIME: {name}")
    print("#" * 92)
    print(f"{'K':>4} | {'NAIVE FA':>16} | {'BONF FA':>16} | "
          f"{'SIDAK FA':>16} | {'ADVERS FA':>16}")
    print(f"{'':>4} | {'(+/-95% CI)':>16} | {'(alpha/K)':>16} | "
          f"{'(Sidak)':>16} | {'(worst-case sel)':>16}")
    print("-" * 92)
    naive_breaks = False
    corrected_ok = True
    for r in rows:
        nfa = f"{r['naive_falseadapt_rate']:.4f}+-{r['naive_fa_ci']:.4f}"
        bfa = f"{r['bonf_falseadapt_rate']:.4f}+-{r['bonf_fa_ci']:.4f}"
        sfa = f"{r['sidak_falseadapt_rate']:.4f}+-{r['sidak_fa_ci']:.4f}"
        afa = f"{r['adv_falseadapt_rate']:.4f}+-{r['adv_fa_ci']:.4f}"
        flag = ""
        if r['naive_falseadapt_rate'] - r['naive_fa_ci'] > alpha:
            flag += " NAIVE>alpha!"
            naive_breaks = True
        if r['bonf_falseadapt_rate'] - r['bonf_fa_ci'] > alpha:
            corrected_ok = False
            flag += " BONF>alpha!!"
        if r['sidak_falseadapt_rate'] - r['sidak_fa_ci'] > alpha:
            corrected_ok = False
            flag += " SIDAK>alpha!!"
        if r['adv_falseadapt_rate'] - r['adv_fa_ci'] > alpha:
            corrected_ok = False
            flag += " ADV>alpha!!"
        print(f"{r['K']:>4} | {nfa:>16} | {bfa:>16} | {sfa:>16} | {afa:>16}{flag}")
    print("-" * 92)
    print("PRICE (commit rate / true-positive rate). Corrected rules commit less:")
    print(f"{'K':>4} | {'naive commit':>13} | {'bonf commit':>12} | {'sidak commit':>12} | "
          f"{'naive TP':>10} | {'bonf TP':>10} | {'sidak TP':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['K']:>4} | {r['naive_commit_rate']:>13.4f} | "
              f"{r['bonf_commit_rate']:>12.4f} | {r['sidak_commit_rate']:>12.4f} | "
              f"{r['naive_truepos_rate']:>10.4f} | {r['bonf_truepos_rate']:>10.4f} | "
              f"{r['sidak_truepos_rate']:>10.4f}")
    print("-" * 92)
    maxfa = dict(
        naive=max(r['naive_falseadapt_rate'] for r in rows),
        bonf=max(r['bonf_falseadapt_rate'] for r in rows),
        sidak=max(r['sidak_falseadapt_rate'] for r in rows),
        adv=max(r['adv_falseadapt_rate'] for r in rows),
    )
    print(f"max NAIVE FA over K : {maxfa['naive']:.4f}  (alpha={alpha}) -> "
          f"{'INFLATES > alpha (gap shown)' if naive_breaks else 'did not inflate here'}")
    print(f"max BONF  FA over K : {maxfa['bonf']:.4f}  <= alpha : {maxfa['bonf'] <= alpha + 1e-9}")
    print(f"max SIDAK FA over K : {maxfa['sidak']:.4f}  <= alpha : {maxfa['sidak'] <= alpha + 1e-9}")
    print(f"max ADV   FA over K : {maxfa['adv']:.4f}  <= alpha : {maxfa['adv'] <= alpha + 1e-9}")
    return naive_breaks, corrected_ok, maxfa


def least_favorable_tightness(Ks, n_trials, tau, alpha, seed):
    """TIGHTNESS check: is alpha a vacuous ceiling, or essentially attained?

    Least-favorable configuration for false-adapt: ALL K candidates sit exactly at
    the boundary Delta_k = 0 (the hardest null -- harmful with zero margin). With the
    Gaussian-exact radius at level alpha/K, each candidate independently clears
    (L_k > 0) with probability EXACTLY alpha/K, since
        P(L_k>0) = P(Delta_hat_k > eps) = P(N(0,tau^2) > z_{1-alpha/2K} tau) = alpha/2K...
    wait -- two-sided radius eps = z_{1-l/2} tau, so P(Delta_hat>eps)=l/2 with l=alpha/K.
    Hence the per-candidate ONE-SIDED clear prob is alpha/(2K) and the union over K
    independent candidates is  1-(1-alpha/2K)^K  ->  alpha/2  as K grows.
    The worst-case (adversarial) selector commits a harmful candidate whenever ANY
    clears, so FA_adv = 1-(1-alpha/2K)^K. This APPROACHES alpha/2 (the two-sided
    radius spends half its budget on the wrong tail); the bound alpha is therefore
    not vacuous -- it is attained up to the standard one-sided/two-sided factor 2,
    which a one-sided conformal radius would remove.
    """
    rng = np.random.default_rng(seed)
    print()
    print("#" * 92)
    print("TIGHTNESS / LEAST-FAVORABLE CHECK (all Delta_k = 0; adversarial selection)")
    print("Bound is NOT vacuous: FA_adv tracks the analytic union 1-(1-alpha/2K)^K -> alpha/2")
    print("(the factor 1/2 is the two-sided-radius budget split; a one-sided radius -> alpha).")
    print("#" * 92)
    print(f"{'K':>4} | {'FA_adv (empirical)':>20} | {'analytic union':>16} | {'<= alpha':>9}")
    print("-" * 92)
    ok = True
    for K in Ks:
        lvl_bonf = alpha / K
        eps = gaussian_exact_radius(tau, lvl_bonf)     # two-sided radius
        # vectorized: all true Delta = 0; adversarial commit iff ANY L_k>0 in the trial
        Dhat = rng.normal(0.0, tau, size=(n_trials, K))
        clears = int(np.count_nonzero(np.any(Dhat - eps > 0.0, axis=1)))
        fa = clears / n_trials
        analytic = 1.0 - (1.0 - lvl_bonf / 2.0) ** K   # one-sided clear prob = (alpha/K)/2
        ci = 1.96 * np.sqrt(max(fa * (1 - fa), 1e-12) / n_trials)
        flag = ""
        if fa - ci > alpha:
            ok = False
            flag = " >alpha!!"
        print(f"{K:>4} | {fa:.4f}+-{ci:.4f}      | {analytic:>16.4f} | "
              f"{str(fa <= alpha + 1e-9):>9}{flag}")
    print("-" * 92)
    print(f"All FA_adv <= alpha at least-favorable null : {ok}")
    print("Interpretation: the family-wise bound binds (approaches alpha/2 here under a")
    print("two-sided radius); it is a real ceiling, not a loose over-statement.")
    return ok


def equivalence_check(K, n_trials, n_cal, tau, alpha, delta_mean, delta_sd,
                      harmful_frac, seed):
    """Confirm the READABLE reference one_trial() and the EXECUTED vectorized
    run_sweep() agree on the corrected (bonf) false-adapt rate, within MC error.
    Guards against the vectorization silently diverging from the per-trial spec."""
    # scalar reference
    rng = np.random.default_rng(seed)
    fa_scalar = 0
    for _ in range(n_trials):
        o = one_trial(rng, K, n_cal, tau, alpha, delta_mean, delta_sd, harmful_frac,
                      radius_mode="gaussian_exact")
        fa_scalar += int(o["bonf_falseadapt"])
    fa_scalar /= n_trials
    # vectorized (different seed stream, so compare distributions, not draws)
    rows = run_sweep([K], n_trials, n_cal, tau, alpha, delta_mean, delta_sd,
                     harmful_frac, seed + 999, radius_mode="gaussian_exact")
    fa_vec = rows[0]["bonf_falseadapt_rate"]
    tol = 3.0 * np.sqrt(max(fa_vec * (1 - fa_vec), 1e-9) / n_trials) + \
          3.0 * np.sqrt(max(fa_scalar * (1 - fa_scalar), 1e-9) / n_trials)
    agree = abs(fa_scalar - fa_vec) <= tol
    print()
    print("#" * 92)
    print("EQUIVALENCE CHECK: scalar reference one_trial() vs vectorized run_sweep()")
    print(f"  K={K}, n_trials={n_trials}: bonf FA  scalar={fa_scalar:.4f}  "
          f"vectorized={fa_vec:.4f}  |diff|={abs(fa_scalar - fa_vec):.4f}  tol={tol:.4f}")
    print(f"  agree within MC tolerance : {agree}")
    print("#" * 92)
    return agree


def main():
    alpha = 0.10
    tau = 1.0              # benefit-estimate noise sd
    delta_mean = 0.0       # prior mean of true benefits (centered: ~half helpful)
    delta_sd = 1.0         # prior sd of true benefits
    harmful_frac = 0.5     # half the candidates forced mildly harmful (stress)
    n_trials = 40000
    seed = 20260628
    Ks = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 92)
    print("K-BOUND MULTICANDIDATE CERTIFIED-ROUTING VALIDATOR")
    print(f"alpha={alpha}  tau={tau}  n_trials={n_trials}  harmful_frac={harmful_frac}  "
          f"seed={seed}  Ks={Ks}")
    print("FA_fw := P(committed candidate is harmful, Delta<=0).  Target: FA_fw <= alpha.")
    print("Rules: NAIVE = argmax marginal LCB, commit iff >0.  BONF/SIDAK = certify each")
    print("candidate at alpha/K (resp. Sidak), commit argmax of certified-helpful set, else")
    print("abstain.  ADVERS = worst-case selection under the alpha/K correction (commits a")
    print("harmful certified candidate whenever one exists) -- tests selection-proofness.")
    print("=" * 92)

    # ---- REGIME 1: finite-sample split-conformal radius (n_cal=200). Honest;
    # at large K the corrected level alpha/K cannot be met by 200 cal points, so the
    # certificate abstains (inf radius) -> FA=0 trivially. Shows the GAP clearly at
    # small/moderate K. ----
    n_cal_1 = 200
    rows1 = run_sweep(Ks, n_trials, n_cal_1, tau, alpha, delta_mean, delta_sd,
                      harmful_frac, seed, radius_mode="conformal")
    nb1, ok1, mx1 = report_regime(
        f"split-conformal radius, n_cal={n_cal_1} (large-K abstains when alpha/K "
        f"unreachable)", rows1, alpha)

    # ---- REGIME 2: Gaussian-EXACT radius (no finite-sample floor). The corrected
    # rule keeps COMMITTING at every K, so FA-control is demonstrated WHILE the rule
    # actively commits -- the strongest test of the theorem. ----
    rows2 = run_sweep(Ks, n_trials, 200, tau, alpha, delta_mean, delta_sd,
                      harmful_frac, seed + 1, radius_mode="gaussian_exact")
    nb2, ok2, mx2 = report_regime(
        "Gaussian-exact radius z_{1-l/2}*tau (corrected rule keeps committing at all K)",
        rows2, alpha)

    # ---- REGIME 3: tightness / least-favorable null ----
    ok3 = least_favorable_tightness(Ks, n_trials, tau, alpha, seed + 2)

    # ---- equivalence: scalar reference vs vectorized executed path ----
    ok_eq = equivalence_check(8, 20000, 200, tau, alpha, delta_mean, delta_sd,
                              harmful_frac, seed + 3)

    # --------- overall verdict ----------
    print()
    print("=" * 92)
    verdict_A = nb1 or nb2          # naive inflates in at least one regime
    verdict_B = (ok1 and ok2 and ok3)   # corrected controls in ALL regimes (incl. adversarial + LFC)
    print(f"(A) NAIVE selection inflates FA_fw above alpha as K grows : "
          f"{'PASS' if verdict_A else 'FAIL'}")
    print(f"(B) CORRECTED routing keeps FA_fw <= alpha in all regimes, including the")
    print(f"    worst-case (adversarial) selection and least-favorable null : "
          f"{'PASS' if verdict_B else 'FAIL'}")
    print(f"(C) scalar reference == vectorized executed path (equivalence)  : "
          f"{'PASS' if ok_eq else 'FAIL'}")
    print("=" * 92)
    if verdict_A and verdict_B and ok_eq:
        print(">>> VALIDATOR RESULT: PASS -- gap reproduced AND closed by the alpha/K correction,")
        print("    with control holding under adversarial selection and while actively committing.")
    else:
        print(">>> VALIDATOR RESULT: FAIL -- see flags above.")

    out_path = __file__.rsplit("/", 1)[0] + "/val_multicandidate_results.json"
    summary = {
        "config": dict(alpha=alpha, tau=tau, n_trials=n_trials, harmful_frac=harmful_frac,
                       seed=seed, Ks=Ks),
        "regime_conformal": dict(n_cal=n_cal_1, rows=rows1, max_fa=mx1,
                                 naive_inflates=bool(nb1), corrected_ok=bool(ok1)),
        "regime_gaussian_exact": dict(rows=rows2, max_fa=mx2,
                                      naive_inflates=bool(nb2), corrected_ok=bool(ok2)),
        "tightness_ok": bool(ok3),
        "equivalence_ok": bool(ok_eq),
        "verdict_naive_inflates": bool(verdict_A),
        "verdict_corrected_controls": bool(verdict_B),
        "pass": bool(verdict_A and verdict_B and ok_eq),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
