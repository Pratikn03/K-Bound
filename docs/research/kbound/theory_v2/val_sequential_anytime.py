#!/usr/bin/env python3
"""
val_sequential_anytime.py
=========================
Release validator for the ANYTIME-VALID SEQUENTIAL adaptation certificate
(extends the BATCH certificate Theorem thm:cert to a stream t=1,2,3,...).

Companion theorem + proof: theory_v2/sequential_anytime_theorem.tex
Builds on: kbound_pkg/kbound/eprocess.py (EProcess) and the e-process / Ville
machinery of Ramdas-Grunwald-Vovk-Shafer (Statist. Sci. 2023),
Howard-Ramdas-McAuliffe-Sekhon (Probab. Surveys 2020), and the
predictable-plugin betting CS of Waudby-Smith & Ramdas (JRSS-B 2024).

VERDICT contract: this file FAILS LOUDLY (sys.exit(1)) if any load-bearing
claim is violated. It is intentionally adversarial.

------------------------------------------------------------------------------
THEOREM UNDER TEST (informal; see .tex for the precise statement)
------------------------------------------------------------------------------
Per-window benefits X_1, X_2, ... in [a,b] (a<0<b) are observed online; each X_t
is the (paired, label-free-estimable) benefit summary of window t, with
conditional mean Delta_t = E[X_t | F_{t-1}]. The SEQUENTIAL certificate maintains

    E_t^+ = prod_{i<=t} (1 + lam_i X_i),     lam_i in [0, c/|a|], predictable,
    E_t^- = prod_{i<=t} (1 + lam_i^- (-X_i)),

and at each t outputs:
    ADAPT   if E_t^+ >= 1/alpha
    FREEZE  if E_t^- >= 1/alpha
    ABSTAIN otherwise
The decision is COMMITTAL & ABSORBING (once it adapts/freezes it stops -- a
real controller acts once). The CLAIM is the TIME-UNIFORM false-adapt bound

    GLOBAL NULL  H0: Delta_t <= 0 for ALL t   ==>   P( exists t : decision_t = ADAPT ) <= alpha,

uniformly over ALL stopping rules tau (an adversary may stop the stream whenever
it likes, including the moment most favorable to a false adapt). Symmetrically
for FREEZE under Delta_t >= 0 for all t.

WHY THE NAIVE PER-BATCH RULE FAILS (negative control): running an INDEPENDENT
level-alpha batch test every window and adapting on the first single-window
rejection makes ~alpha errors PER WINDOW; over T windows the multiplicity
inflates the continuous-monitoring false-adapt rate toward 1, NOT alpha.

------------------------------------------------------------------------------
TESTS
------------------------------------------------------------------------------
 A  e-process is a supermartingale under H0 (E[E_t^+] <= 1, t=1..T).         [exact-in-expectation]
 B  TIME-UNIFORM false-adapt <= alpha under the hardest null Delta=0,
    AND under an explicit adversarial-stopping attacker.                       [PASS/FAIL vs alpha+MC]
 C  Same under DRIFTING / heteroscedastic nulls Delta_t <= 0 (non-i.i.d.).    [PASS/FAIL]
 D  NEGATIVE CONTROL: naive per-batch monitoring inflates above alpha.        [must inflate]
 E  POWER: under Delta>0 the rule does ADAPT (bound not vacuous).             [must fire]
 F  Symmetry: FREEZE side controlled under Delta_t>=0.                        [PASS/FAIL]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Self-contained e-process (mirrors kbound/eprocess.py; re-implemented so the
# validator is independent of the package and tests the ALGORITHM, not imports).
# Truncated predictable-plugin "aGRAPA" betting (Waudby-Smith & Ramdas 2024):
# lam_t = clip( mu_hat_{t-1} / sigma2_hat_{t-1}, 0, c/|a| ), predictable.
# ----------------------------------------------------------------------------
class EProc:
    def __init__(self, alpha=0.1, a=-1.0, b=1.0, cap=0.5, prior_var=0.25, prior_w=1.0):
        assert 0.0 < alpha < 1.0 and a < 0.0 < b
        self.alpha, self.a, self.b, self.cap = alpha, a, b, cap
        self.log_thr = math.log(1.0 / alpha)
        self.lam_max_p = cap / (-a)
        self.lam_max_m = cap / b
        self._s1 = 0.0          # sum X_j  (predictable mean numerator)
        self._s1m = 0.0
        self._s2 = prior_w * prior_var
        self._s2m = prior_w * prior_var
        self._cnt = 0.0
        self._cnt_v = prior_w
        self.logw_p = 0.0
        self.logw_m = 0.0
        self.t = 0

    def update(self, x):
        x = float(max(self.a, min(self.b, x)))
        mu = self._s1 / self._cnt if self._cnt > 0 else 0.0
        mum = self._s1m / self._cnt if self._cnt > 0 else 0.0
        s2 = self._s2 / self._cnt_v
        s2m = self._s2m / self._cnt_v
        lam_p = float(np.clip(mu / s2 if s2 > 0 else 0.0, 0.0, self.lam_max_p))
        lam_m = float(np.clip(mum / s2m if s2m > 0 else 0.0, 0.0, self.lam_max_m))
        self.logw_p += math.log(max(1.0 + lam_p * x, 1e-300))
        self.logw_m += math.log(max(1.0 + lam_m * (-x), 1e-300))
        self._s1 += x
        self._s1m += -x
        self._s2 += x * x
        self._s2m += x * x
        self._cnt += 1.0
        self._cnt_v += 1.0
        self.t += 1

    @property
    def wealth_p(self):
        return math.exp(self.logw_p)

    def decision(self):
        if self.logw_p >= self.log_thr:
            return "adapt"
        if self.logw_m >= self.log_thr:
            return "freeze"
        return "abstain"


# ----------------------------------------------------------------------------
# Vectorized engine: run ALL streams in parallel, one numpy step per time index.
# Identical algorithm to EProc (truncated predictable-plugin aGRAPA betting); this
# is also an independent cross-check of the scalar implementation. Returns, per
# run: ever_adapt, ever_freeze, and the running-max wealth (for the optional-stop
# attacker, which is exactly the object Ville bounds).
# ----------------------------------------------------------------------------
def run_eproc_vec(streams, alpha, a=-1.0, b=1.0, cap=0.5, prior_var=0.25,
                  prior_w=1.0, record_ts=None):
    streams = np.clip(np.asarray(streams, dtype=float), a, b)
    n, T = streams.shape
    log_thr = math.log(1.0 / alpha)
    lam_max_p = cap / (-a)
    lam_max_m = cap / b
    rec_set = set(record_ts) if record_ts else set()
    mean_wealth = {}
    # predictable running stats (state BEFORE seeing X_t)
    s1 = np.zeros(n); s1m = np.zeros(n)
    s2 = np.full(n, prior_w * prior_var); s2m = np.full(n, prior_w * prior_var)
    cnt = np.zeros(n); cnt_v = np.full(n, prior_w)
    logw_p = np.zeros(n); logw_m = np.zeros(n)
    ever_p = np.zeros(n, dtype=bool); ever_m = np.zeros(n, dtype=bool)
    wmax_p = np.ones(n)  # running max of E_t^+ (for optional-stopping attacker)
    for t in range(T):
        x = streams[:, t]
        mu = np.where(cnt > 0, s1 / np.where(cnt > 0, cnt, 1.0), 0.0)
        mum = np.where(cnt > 0, s1m / np.where(cnt > 0, cnt, 1.0), 0.0)
        sig2 = s2 / cnt_v
        sig2m = s2m / cnt_v
        lam_p = np.clip(np.where(sig2 > 0, mu / sig2, 0.0), 0.0, lam_max_p)
        lam_m = np.clip(np.where(sig2m > 0, mum / sig2m, 0.0), 0.0, lam_max_m)
        logw_p = logw_p + np.log(np.maximum(1.0 + lam_p * x, 1e-300))
        logw_m = logw_m + np.log(np.maximum(1.0 + lam_m * (-x), 1e-300))
        wmax_p = np.maximum(wmax_p, np.exp(np.minimum(logw_p, 700.0)))
        ever_p |= (logw_p >= log_thr)
        ever_m |= (logw_m >= log_thr)
        step = t + 1
        if step in rec_set:
            mean_wealth[step] = float(np.mean(np.exp(np.minimum(logw_p, 700.0))))
        # update predictable stats AFTER using them
        s1 += x; s1m += -x; s2 += x * x; s2m += x * x; cnt += 1.0; cnt_v += 1.0
    return {
        "ever_adapt": ever_p,
        "ever_freeze": ever_m,
        "wmax_plus": wmax_p,
        "logw_plus_final": logw_p,
        "mean_wealth": mean_wealth,
    }


# ----------------------------------------------------------------------------
# Stream generators. All bounded in [a,b]=[-1,1].
# ----------------------------------------------------------------------------
def stream_twopoint(rng, T, delta, a=-1.0, b=1.0):
    """i.i.d. two-point on {a,b} with mean exactly delta (hardest extreme support)."""
    p_b = min(max((delta - a) / (b - a), 0.0), 1.0)
    return np.where(rng.random(T) < p_b, b, a)


def stream_drift_null(rng, T, a=-1.0, b=1.0, seed_phase=0.0):
    """NON-i.i.d. heteroscedastic null: every window has Delta_t <= 0 but the
    mean and variance drift over time. Means swing in [a, 0] (always <=0),
    variance modulated. This stresses the per-window (not just i.i.d.) claim."""
    t = np.arange(T)
    # drifting conditional mean in [-0.9, -0.001]  (strictly <= 0)
    mu_t = -0.45 + 0.449 * np.sin(0.05 * t + seed_phase)   # in [-0.899, -0.001]
    mu_t = np.minimum(mu_t, -1e-3)
    # heteroscedastic: realize bounded values with the target mean via two-point
    # on {a, b} whose mixing prob gives exactly mu_t (keeps support extreme/worst).
    p_b = np.clip((mu_t - a) / (b - a), 0.0, 1.0)
    return np.where(rng.random(T) < p_b, b, a)


def stream_adversarial_null(rng, T, a=-1.0, b=1.0):
    """GENUINE adversarial null for the OPTIONAL-STOPPING attack.

    The honest attacker may NOT make the world beneficial: the null requires the
    *conditional* mean Delta_t = E[X_t | F_{t-1}] <= 0 at EVERY step. (A
    deterministic run of +b would make Delta_t = +b > 0 -- a true-positive
    stretch, not a null.) The attacker's only freedom under H0 is (i) the
    per-step law subject to mean<=0 and (ii) the STOPPING TIME.

    Strongest such attack = the maximal-variance, mean-EXACTLY-0 two-point law at
    every step (symmetric +b / a with |a|=|b|, prob 1/2 each). This maximizes the
    chance of an early upward wealth excursion that the attacker can stop on. The
    'ever-cross' accounting in the simulator already takes the sup over ALL
    stopping times, so this is the worst case the optional-stopping adversary can
    achieve under H0. Each draw is an i.i.d. Rademacher * b (martingale-difference,
    Delta_t = 0 for all t)."""
    assert abs(a + b) < 1e-12, "symmetric support required for the mean-0 attack"
    return np.where(rng.random(T) < 0.5, b, a)


def worst_stopping_false_adapt(streams, alpha, a=-1.0, b=1.0):
    """Explicit optional-stopping attacker: for each run, an adversary that PEEKS
    at the whole path and stops at the single time maximizing wealth. The false
    adapt happens iff the running max of E_t^+ EVER reaches 1/alpha. This is
    exactly the event Ville's inequality bounds: P(sup_t E_t^+ >= 1/alpha) <= alpha.
    """
    thr = 1.0 / alpha
    out = run_eproc_vec(streams, alpha, a, b)
    return float((out["wmax_plus"] >= thr).mean())


# ----------------------------------------------------------------------------
# Decision simulators (vectorized; absorbing 'act once' == 'ever crosses').
# ----------------------------------------------------------------------------
def ever_adapt_eproc(streams, alpha, a=-1.0, b=1.0, attacker_stop=False):
    """Fraction of streams in which the ANYTIME e-process EVER adapts.
    An absorbing controller that commits on first crossing adapts iff E_t^+ ever
    reaches 1/alpha; that is precisely 'ever_adapt'. (Equivalent to sup over
    stopping times, which Ville bounds.)"""
    out = run_eproc_vec(streams, alpha, a, b)
    return float(out["ever_adapt"].mean())


def ever_adapt_naive_perbatch(streams, alpha, window=25, a=-1.0, b=1.0, batch_thr=None):
    """NEGATIVE CONTROL: the naive per-batch certificate applied repeatedly.

    Partition the stream into consecutive batches of `window` samples. On EACH
    batch independently, run a per-batch certificate at level alpha: ADAPT if the
    batch mean exceeds a per-batch decision threshold. Continuous monitoring =
    ADAPT on the FIRST batch that rejects.

    CALIBRATION (honesty point): we make each per-batch test EXACTLY one-sided
    level-alpha under the null (reject if batch_mean > q_{1-alpha}, the (1-alpha)
    quantile of the null batch-mean law). Otherwise a loose Hoeffding radius would
    almost never reject under H0, masking inflation behind over-conservatism for
    the WRONG reason. With a calibrated batch test, per-batch false-adapt ~ alpha,
    so any excess is PURE MULTIPLICITY: over K=T/window batches the
    family-wise false-adapt inflates to ~ 1 - (1-alpha)^K >> alpha. THAT is the
    failure the anytime e-process repairs. `batch_thr` overrides the threshold;
    if None, a Hoeffding radius is used (kept for reference)."""
    n, T = streams.shape
    n_batches = T // window
    crossed = np.zeros(n, dtype=bool)
    if batch_thr is None:
        # Hoeffding radius (reference; conservative): eps = (b-a) sqrt(ln(2/alpha)/(2 window)).
        eps = (b - a) * math.sqrt(math.log(2.0 / alpha) / (2.0 * window))
        thr = eps
    else:
        thr = batch_thr  # calibrated exact-level-alpha threshold on the batch mean
    eps = thr
    for i in range(n):
        for k in range(n_batches):
            seg = streams[i, k * window:(k + 1) * window]
            dhat = float(seg.mean())
            # calibrated mode: reject if batch mean exceeds the level-alpha
            # threshold thr. Hoeffding mode: reject if dhat - eps > 0  (thr=eps,
            # tested as dhat > thr with thr the radius, i.e. dhat - eps > 0).
            if dhat > thr:   # batch certificate says ADAPT
                crossed[i] = True
                break
    return float(crossed.mean()), eps, n_batches


def mean_wealth_under_null(streams, alpha, a=-1.0, b=1.0, record_ts=None):
    """E[E_t^+] at several t under the null (supermartingale check: should be <=1).
    Vectorized via run_eproc_vec."""
    if record_ts is None:
        record_ts = [1, 5, 25, 100, 250, streams.shape[1]]
    out = run_eproc_vec(streams, alpha, a, b, record_ts=record_ts)
    return dict(sorted(out["mean_wealth"].items()))


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------
def main():
    ALPHA = 0.10
    A, B = -1.0, 1.0
    N_RUNS = 4000
    T = 500
    rng = np.random.default_rng(20260628)
    fail = []
    results = {
        "config": {
            "alpha": ALPHA,
            "a": A,
            "b": B,
            "n_runs": N_RUNS,
            "horizon": T,
            "seed": 20260628,
        },
        "checks": {},
    }

    def mc_slack(n):
        # 3-sigma binomial slack at rate alpha (generous; one-sided concern)
        return 3.0 * math.sqrt(ALPHA * (1 - ALPHA) / n)

    print("=" * 78)
    print("ANYTIME-VALID SEQUENTIAL CERTIFICATE -- numerical validator")
    print(f"alpha={ALPHA}  [a,b]=[{A},{B}]  n_runs={N_RUNS}  horizon T={T}")
    print("=" * 78)

    # -- TEST 0: scalar EProc vs vectorized engine agree (impl cross-check) ----
    print("\n[0] Cross-check: scalar EProc == vectorized engine (same algorithm)")
    rng0 = np.random.default_rng(7)
    s_chk = np.stack([stream_twopoint(rng0, 120, d, A, B)
                      for d in (-0.2, 0.0, 0.1, 0.25) for _ in range(8)])
    out_v = run_eproc_vec(s_chk, ALPHA, A, B)
    mism = 0
    for i in range(s_chk.shape[0]):
        ep = EProc(alpha=ALPHA, a=A, b=B)
        ever_a = False
        for x in s_chk[i]:
            ep.update(float(x))
            ever_a = ever_a or (ep.decision() == "adapt")
        if bool(ever_a) != bool(out_v["ever_adapt"][i]):
            mism += 1
    print(f"    streams checked={s_chk.shape[0]}, ever-adapt mismatches={mism}")
    if mism != 0:
        fail.append(f"0: scalar/vectorized e-process disagree on {mism} streams")
    print(f"    => {'PASS' if mism == 0 else 'FAIL'}")
    results["checks"]["scalar_vectorized_crosscheck"] = {
        "streams_checked": int(s_chk.shape[0]),
        "ever_adapt_mismatches": int(mism),
        "passed": bool(mism == 0),
    }

    # -- TEST A: supermartingale under H0 (Delta=0) ---------------------------
    print("\n[A] Supermartingale under H0 (Delta=0): E[E_t^+] should be <= 1+MC")
    s_h0 = np.stack([stream_twopoint(rng, T, 0.0, A, B) for _ in range(1500)])
    mw = mean_wealth_under_null(s_h0, ALPHA, A, B)
    a_ok = True
    for t, val in mw.items():
        flag = "ok" if val <= 1.0 + 0.15 else "VIOLATION"
        if val > 1.0 + 0.15:
            a_ok = False
        print(f"    t={t:4d}   E[E_t^+]={val:.4f}   {flag}")
    if not a_ok:
        fail.append("A: supermartingale property E[E_t^+]<=1 violated")
    print(f"    => {'PASS' if a_ok else 'FAIL'}")
    results["checks"]["supermartingale_null_wealth"] = {
        "mean_wealth": {str(k): float(v) for k, v in mw.items()},
        "passed": bool(a_ok),
    }

    # -- TEST B: time-uniform false-adapt under hardest null + attacker -------
    print("\n[B] TIME-UNIFORM false-adapt <= alpha under H0 (Delta=0):")
    s_b0 = np.stack([stream_twopoint(rng, T, 0.0, A, B) for _ in range(N_RUNS)])
    far_b0 = ever_adapt_eproc(s_b0, ALPHA, A, B)
    bound = ALPHA + mc_slack(N_RUNS)
    ok_b0 = far_b0 <= bound
    print(f"    i.i.d. Delta=0        ever-adapt={far_b0:.4f}   bound(alpha+MC)={bound:.4f}   {'PASS' if ok_b0 else 'FAIL'}")
    if not ok_b0:
        fail.append(f"B: i.i.d. Delta=0 ever-adapt {far_b0:.4f} > {bound:.4f}")

    # strictly negative null (should be even smaller)
    s_bn = np.stack([stream_twopoint(rng, T, -0.10, A, B) for _ in range(N_RUNS)])
    far_bn = ever_adapt_eproc(s_bn, ALPHA, A, B)
    ok_bn = far_bn <= bound
    print(f"    i.i.d. Delta=-0.10    ever-adapt={far_bn:.4f}   bound={bound:.4f}   {'PASS' if ok_bn else 'FAIL'}")
    if not ok_bn:
        fail.append(f"B: Delta=-0.10 ever-adapt {far_bn:.4f} > {bound:.4f}")

    # OPTIONAL-STOPPING ATTACKER: maximal-variance mean-EXACTLY-0 null
    # (i.i.d. Rademacher*b; Delta_t=0 for all t), adversary stops at the single
    # best time. 'worst_stopping_false_adapt' makes 'sup over stopping times'
    # explicit via the running max of E_t^+ -- this is precisely the event Ville
    # bounds. (A deterministic +b run is NOT a null: it has Delta_t=+b>0.)
    s_adv = np.stack([stream_adversarial_null(rng, T, A, B) for _ in range(N_RUNS)])
    far_adv = worst_stopping_false_adapt(s_adv, ALPHA, A, B)
    ok_adv = far_adv <= bound
    print(f"    optional-stop attack  ever-adapt={far_adv:.4f}   bound={bound:.4f}   {'PASS' if ok_adv else 'FAIL'}")
    print("      (i.i.d. mean-0 Rademacher null; adversary peeks & stops at argmax wealth)")
    if not ok_adv:
        fail.append(f"B: optional-stop attack ever-adapt {far_adv:.4f} > {bound:.4f}")
    results["checks"]["time_uniform_false_adapt"] = {
        "mc_bound": float(bound),
        "iid_delta_0": float(far_b0),
        "iid_delta_minus_0_10": float(far_bn),
        "optional_stop_attack": float(far_adv),
        "passed": bool(ok_b0 and ok_bn and ok_adv),
    }

    # -- TEST C: drifting / heteroscedastic null Delta_t <= 0 -----------------
    print("\n[C] TIME-UNIFORM false-adapt under DRIFTING null (Delta_t<=0, non-i.i.d.):")
    s_dr = np.stack([stream_drift_null(rng, T, A, B, seed_phase=rng.uniform(0, 6.28))
                     for _ in range(N_RUNS)])
    far_dr = ever_adapt_eproc(s_dr, ALPHA, A, B)
    ok_dr = far_dr <= bound
    print(f"    drifting Delta_t<=0   ever-adapt={far_dr:.4f}   bound={bound:.4f}   {'PASS' if ok_dr else 'FAIL'}")
    if not ok_dr:
        fail.append(f"C: drifting null ever-adapt {far_dr:.4f} > {bound:.4f}")
    results["checks"]["drifting_null"] = {
        "ever_adapt": float(far_dr),
        "mc_bound": float(bound),
        "passed": bool(ok_dr),
    }

    # -- TEST D: NEGATIVE CONTROL -- naive per-batch inflates -----------------
    print("\n[D] NEGATIVE CONTROL: naive per-batch certificate under continuous monitoring")
    print("    Each batch = EXACT one-sided level-alpha test (so per-batch FA ~ alpha);")
    print("    any excess under continuous monitoring is PURE MULTIPLICITY.")
    WIN = 20
    # Exact (1-alpha) one-sided threshold for the mean of WIN iid Rademacher*b draws.
    # #heads ~ Binom(WIN, 0.5); mean = (2*heads - WIN)/WIN * b.  Find smallest q with
    # P(mean > q) <= alpha under the null.
    from math import comb
    pmf = [comb(WIN, k) / (2.0 ** WIN) for k in range(WIN + 1)]
    means = [((2 * k - WIN) / WIN) * B for k in range(WIN + 1)]
    # one-sided upper tail: choose threshold so null reject prob ~ alpha (<= alpha)
    order = sorted(range(WIN + 1), key=lambda k: means[k], reverse=True)
    cum = 0.0
    q_thr = means[order[0]] + 1.0  # default: never reject
    per_batch_reject = 0.0
    for k in order:
        cum_next = cum + pmf[k]
        if cum_next <= ALPHA:
            cum = cum_next
            q_thr = means[k] - 1e-9   # reject means strictly above this value
            per_batch_reject = cum
        else:
            break
    naive_far, q_used, nb = ever_adapt_naive_perbatch(
        s_b0, ALPHA, window=WIN, a=A, b=B, batch_thr=q_thr)
    n_batches_500 = T // WIN
    fw_indep = 1.0 - (1.0 - per_batch_reject) ** n_batches_500
    print(f"    window={WIN}, #batches={nb}, calibrated batch threshold q={q_thr:.4f}")
    print(f"    per-batch null reject prob (exact) = {per_batch_reject:.4f}  (<= alpha={ALPHA})")
    print(f"    naive ever-adapt (continuous monitoring) = {naive_far:.4f}")
    print(f"    predicted family-wise 1-(1-p)^K        = {fw_indep:.4f}")
    print(f"    anytime e-process ever-adapt (same null) = {far_b0:.4f}")
    inflated = naive_far > ALPHA + 0.03 and naive_far > far_b0 + 0.02
    print(f"    naive inflates above alpha={ALPHA}?  {'YES (control behaves)' if inflated else 'NO -- FAILED to demonstrate inflation'}")
    if not inflated:
        fail.append(f"D: negative control failed to inflate (naive={naive_far:.4f}, alpha={ALPHA})")

    # inflation grows with horizon: more monitoring -> more error (naive) while
    # the anytime e-process stays flat at <= alpha.
    print("    inflation vs horizon (calibrated per-batch, Delta=0):")
    horizon_sweep = {}
    for Tlong in (100, 250, 500, 1000):
        s_long = np.stack([stream_twopoint(rng, Tlong, 0.0, A, B) for _ in range(1500)])
        nf, _, nbk = ever_adapt_naive_perbatch(s_long, ALPHA, window=WIN, a=A, b=B, batch_thr=q_thr)
        ef = ever_adapt_eproc(s_long, ALPHA, A, B)
        print(f"        T={Tlong:5d} ({nbk:2d} batches): naive={nf:.4f}   anytime-eproc={ef:.4f}")
        horizon_sweep[str(Tlong)] = {
            "n_batches": int(nbk),
            "naive": float(nf),
            "anytime_eprocess": float(ef),
        }
    results["checks"]["negative_control_naive_per_batch"] = {
        "window": int(WIN),
        "n_batches": int(nb),
        "batch_threshold": float(q_thr),
        "per_batch_reject_exact": float(per_batch_reject),
        "naive_ever_adapt": float(naive_far),
        "predicted_familywise": float(fw_indep),
        "anytime_ever_adapt_same_null": float(far_b0),
        "inflated": bool(inflated),
        "horizon_sweep": horizon_sweep,
        "passed": bool(inflated),
    }

    # -- TEST E: POWER (bound not vacuous) ------------------------------------
    print("\n[E] POWER under H1 (Delta>0): the rule must actually ADAPT")
    e_ok = True
    power = {}
    for d in (0.15, 0.30):
        s_p = np.stack([stream_twopoint(rng, T, d, A, B) for _ in range(1500)])
        pw = ever_adapt_eproc(s_p, ALPHA, A, B)
        good = pw > 0.5
        if not good:
            e_ok = False
        print(f"    Delta={d:+.2f}   ever-adapt(power)={pw:.4f}   {'ok' if good else 'WEAK'}")
        power[f"delta_{d:+.2f}"] = float(pw)
    if not e_ok:
        fail.append("E: power too low under H1 -- certificate may be vacuous")
    print(f"    => {'PASS' if e_ok else 'FAIL'}")
    power["passed"] = bool(e_ok)
    results["checks"]["power_under_positive_benefit"] = power

    # -- TEST F: symmetry -- FREEZE side under Delta_t >= 0 -------------------
    print("\n[F] Symmetry: FREEZE false-rate <= alpha under Delta_t>=0")
    s_f = np.stack([stream_twopoint(rng, T, 0.0, A, B) for _ in range(N_RUNS)])
    far_freeze = float(run_eproc_vec(s_f, ALPHA, A, B)["ever_freeze"].mean())
    ok_f = far_freeze <= bound
    print(f"    Delta=0 false-FREEZE  ever-freeze={far_freeze:.4f}   bound={bound:.4f}   {'PASS' if ok_f else 'FAIL'}")
    if not ok_f:
        fail.append(f"F: false-freeze {far_freeze:.4f} > {bound:.4f}")
    results["checks"]["time_uniform_false_freeze"] = {
        "ever_freeze": float(far_freeze),
        "mc_bound": float(bound),
        "passed": bool(ok_f),
    }

    # -- VERDICT --------------------------------------------------------------
    results["pass"] = not fail
    results["failures"] = fail
    out_path = Path(__file__).with_name("val_sequential_anytime_results.json")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n" + "=" * 78)
    if fail:
        print("VERDICT: FAIL")
        for f in fail:
            print("   -", f)
        print(f"saved -> {out_path}")
        print("=" * 78)
        sys.exit(1)
    print("VERDICT: PASS -- all anytime-valid claims hold; naive per-batch inflates as predicted.")
    print(f"saved -> {out_path}")
    print("=" * 78)
    sys.exit(0)


if __name__ == "__main__":
    main()
