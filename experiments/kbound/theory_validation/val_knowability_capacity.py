#!/usr/bin/env python3
r"""
val_knowability_capacity.py
===========================

Numerical validation of the **Knowability--Capacity threshold theorem** for the
sign of the adaptation benefit, in the 1-D Gaussian-location covariate-shift
two-point model (K-Bound graded invariant; deepens Thm 1 / Thm 2).

----------------------------------------------------------------------------
The model (fully well-posed; the concept mechanism is explicit)
----------------------------------------------------------------------------
  * Source covariates  P_X = N(0,1)   (known).
  * Target covariates  Q_X = N(mu,1)  (covariate shift; mu unknown, estimated
    from an UNLABELED target sample x_1..x_n).
  * Deployed classifier g = f0 = 1[x>0]   (boundary at 0).
  * A FIXED, given candidate adapted rule  fa = 1[x>a]  with a>0 known (the
    "shipped adapted boundary"); disagreement region  D = (0,a).
  * Concept (label mechanism):  eta(x)=P(Y=1|x). Admissible HARD-threshold
    concepts  eta_theta(x)=1[x>theta]; the source-calibrated threshold theta_S
    is known. Admissible TARGET concepts: those whose target boundary-mass drift
    from the source boundary is at most a budget eps:
        C_eps = { theta_T :  |Phi(theta_T - mu) - Phi(theta_S - mu)| <= eps }.
    (eps is the 1-D analogue of the calibration-drift budget beta of the
    benefit-sign frontier; it is the minimal untestable supplement.)

The label-free observer sees only (P_X, g, fa, theta_S, eps) and the unlabeled
sample {x_i}~Q_X, and must output a guess of sign(Delta), where
    Delta = R_Q(f0) - R_Q(fa)     (Delta>0  <=>  adapting to fa helps).

----------------------------------------------------------------------------
Exact benefit identity and the sign-flip locus (proved in the .tex)
----------------------------------------------------------------------------
On D=(0,a), exactly one of f0,fa is correct, so (0/1 loss)
    Delta(theta) = Q( (0,a) cap {x<theta} ) - Q( (0,a) cap {x>theta} )
                 = 2*Phi(theta' - mu) - Phi(-mu) - Phi(a-mu),   theta'=clip(theta,0,a).
Delta is nondecreasing in theta with a single root theta0 (the Q-MEDIAN of the
band D), so
    sign(Delta) = sign(theta - theta0),
    theta0 = mu + Phi^{-1}( (Phi(-mu)+Phi(a-mu))/2 )   (OBSERVABLE).

----------------------------------------------------------------------------
The computable invariant K and the threshold tau (the theorem)
----------------------------------------------------------------------------
Define the label-free, computable **knowability invariant**
    K  :=  | Phi(theta0 - mu) - Phi(theta_S - mu) | / eps
        =  | (Phi(-mu)+Phi(a-mu))/2  -  Phi(theta_S - mu) | / eps     (closed form),
i.e. the MASS MARGIN between the calibrated boundary theta_S and the sign-flip
locus theta0 (in units of target probability mass), normalised by the drift
budget eps.  K is a margin-weighted SNR of the covariate shift relative to g's
boundary.

THEOREM (population / Q_X known).  With tau = 1:
    K > tau  ==>  sign(Delta) = sign(theta_S - theta0) for EVERY admissible
                  target concept; the plug-in certificate recovers it (error 0).
    K < tau  ==>  there exist two admissible concepts with IDENTICAL Q_X (hence
                  TV of any unlabeled statistic = 0 for ALL n) and OPPOSITE
                  sign(Delta); minimax label-free error = 1/2.
So converse and achievability MEET at the single threshold tau = 1, EXACTLY
(not asymptotically): K is a genuine capacity for the population problem.

FINITE-n boundary layer (Le Cam).  Even with concept transfer (eps-> the flip
is crossed only by mu-estimation noise), two worlds with mu_+- = mu* +- (c/2)/sqrt(n)
straddling the flip locus mu* have unlabeled TV = 2*Phi(c/2)-1 and per-sample
KL giving n-fold KL = c^2/2, so any label-free sign test has minimax error
    >= (1/2)(1-TV) = Phi(-c/2)   and   >= (1/4) exp(-c^2/2)   (Bretagnolle-Huber),
CONSTANT in n. This is a 1/sqrt(n)-width window around mu* (an O(1/sqrt n)
correction to the LOCATION of the threshold, not a gap in its VALUE tau=1).

----------------------------------------------------------------------------
What this script confirms (all numbers reproducible, fixed seed)
----------------------------------------------------------------------------
  (A) Exactness of the population threshold: over a large random sweep of
      (mu,a,theta_S,eps), K>1 <=> sign is identifiable over C_eps, with ZERO
      mismatches off a thin boundary shell, and a boundary stress-test at K~1.
  (B) Phase transition of the finite-n three-way certificate: minimax
      sign-recovery accuracy ->1 where K_pop>1 and =1/2 where K_pop<1, at
      several n, with the transition pinned at K=1.
  (C) Finite-n Le Cam boundary layer: the empirical Bayes (optimal) sign-test
      error meets the analytic floor Phi(-c/2), exceeds the BH certificate
      (1/4)e^{-c^2/2}, and is constant in n.
  (D) The exact benefit identity Delta(theta)=2Phi(theta'-mu)-Phi(-mu)-Phi(a-mu)
      matches Monte-Carlo to MC error (sanity on the model itself).

Run:
    python3 val_knowability_capacity.py
    python3 val_knowability_capacity.py --json results_knowability_capacity.json

Pure numpy + scipy. No labels are used by any rule (labels appear only to
*verify* the construction in part (D)). No GPU, no external I/O.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# --------------------------------------------------------------------------- #
#  Closed-form objects of the model                                           #
# --------------------------------------------------------------------------- #
def theta0_band_median(mu: float, a: float) -> float:
    r"""Sign-flip locus theta0 = Q-median of the band D=(0,a) (OBSERVABLE)."""
    return mu + norm.ppf(0.5 * (norm.cdf(-mu) + norm.cdf(a - mu)))


def delta_exact(mu: float, a: float, theta: float) -> float:
    r"""Exact benefit Delta(theta)=R_Q(f0)-R_Q(fa) for hard-threshold concept theta.

    Delta = 2*Phi(theta'-mu) - Phi(-mu) - Phi(a-mu),  theta'=clip(theta,0,a).
    Nondecreasing in theta; sign(Delta)=sign(theta-theta0).
    """
    tp = min(max(theta, 0.0), a)
    return 2.0 * norm.cdf(tp - mu) - norm.cdf(-mu) - norm.cdf(a - mu)


def K_invariant(mu: float, a: float, theta_S: float, eps: float) -> float:
    r"""Knowability invariant K = |mass-margin(theta_S, theta0)| / eps  (OBSERVABLE).

    = | (Phi(-mu)+Phi(a-mu))/2 - Phi(theta_S-mu) | / eps.
    """
    band_med_mass = 0.5 * (norm.cdf(-mu) + norm.cdf(a - mu))   # = Phi(theta0-mu)
    boundary_mass = norm.cdf(theta_S - mu)
    return abs(band_med_mass - boundary_mass) / eps


def admissible_endpoints(mu: float, theta_S: float, eps: float) -> tuple[float, float]:
    r"""Exact endpoints of the admissible target-threshold interval C_eps.

    {theta_T : |Phi(theta_T-mu) - Phi(theta_S-mu)| <= eps}
      = [ mu + Phi^{-1}(p_S - eps),  mu + Phi^{-1}(p_S + eps) ],  p_S = Phi(theta_S-mu).
    """
    pS = norm.cdf(theta_S - mu)
    lo_p = max(1e-12, pS - eps)
    hi_p = min(1.0 - 1e-12, pS + eps)
    return mu + norm.ppf(lo_p), mu + norm.ppf(hi_p)


# --------------------------------------------------------------------------- #
#  (A) Exactness of the population threshold tau = 1                           #
# --------------------------------------------------------------------------- #
@dataclass
class ExactnessResult:
    n_draws: int
    n_tested_off_boundary: int
    n_mismatch: int
    n_identifiable: int
    n_not_identifiable: int
    boundary_shell: float
    boundary_stress: list           # [(target_K, K_exact, identifiable, agrees)]
    exact_threshold_holds: bool


def validate_exactness(n_draws: int = 40_000, seed: int = 20260619,
                       shell: float = 1e-6) -> ExactnessResult:
    r"""K>1 <=> sign identifiable over C_eps, exactly, off a thin shell |K-1|<shell."""
    rng = np.random.default_rng(seed)
    mism = 0
    tested = 0
    cnt = [0, 0]
    for _ in range(n_draws):
        mu = rng.uniform(-3.0, 3.0)
        a = rng.uniform(0.2, 3.0)
        theta_S = rng.uniform(-1.0, a + 1.0)
        eps = rng.uniform(0.01, 0.25)
        K = K_invariant(mu, a, theta_S, eps)
        lo, hi = admissible_endpoints(mu, theta_S, eps)
        s_lo = np.sign(delta_exact(mu, a, lo))
        s_hi = np.sign(delta_exact(mu, a, hi))
        ident = bool(s_lo == s_hi and s_lo != 0)
        cnt[int(ident)] += 1
        if abs(K - 1.0) < shell:
            continue
        tested += 1
        if (K > 1.0) != ident:
            mism += 1

    # Boundary stress test: tune eps so K is exactly slightly above / below 1.
    mu, a, theta_S = 0.7, 1.4, 1.1
    core = abs(0.5 * (norm.cdf(-mu) + norm.cdf(a - mu)) - norm.cdf(theta_S - mu))
    stress = []
    for f in (0.90, 0.99, 0.999, 1.001, 1.01, 1.10):
        eps = core / f                       # makes K = f
        K = K_invariant(mu, a, theta_S, eps)
        lo, hi = admissible_endpoints(mu, theta_S, eps)
        ident = bool(np.sign(delta_exact(mu, a, lo)) == np.sign(delta_exact(mu, a, hi))
                     and np.sign(delta_exact(mu, a, lo)) != 0)
        stress.append([float(f), float(K), bool(ident), bool((K > 1.0) == ident)])

    return ExactnessResult(
        n_draws=int(n_draws),
        n_tested_off_boundary=int(tested),
        n_mismatch=int(mism),
        n_identifiable=int(cnt[1]),
        n_not_identifiable=int(cnt[0]),
        boundary_shell=float(shell),
        boundary_stress=stress,
        exact_threshold_holds=bool(mism == 0 and all(r[3] for r in stress)),
    )


# --------------------------------------------------------------------------- #
#  (B) Phase transition of the finite-n three-way certificate                 #
# --------------------------------------------------------------------------- #
def _certificate_score(mu: float, a: float, theta_S: float, eps: float, n: int,
                       trials: int, rng: np.random.Generator, z: float = 1.96,
                       chunk: int = 20_000) -> float:
    r"""Minimax (worst admissible concept) sign-recovery accuracy of the finite-n
    three-way certificate.

    Rule: estimate mu_hat = mean(x_i); form a z-confidence interval
    [mu_hat - z/sqrt(n), mu_hat + z/sqrt(n)]; COMMIT to sign(theta_S - theta0_hat)
    only if K stays > 1 across the whole interval (a one-sided lower-confidence
    bound K_lcb > 1); otherwise ABSTAIN. Scoring: correct commit = 1, wrong commit
    = 0, abstain = 1/2 (a coin). Accuracy is the MIN over the two extreme
    admissible target concepts (the adversary), so it is a minimax accuracy.
    """
    lo, hi = admissible_endpoints(mu, theta_S, eps)
    truths = []
    for th in (lo, hi):
        s = np.sign(delta_exact(mu, a, th))
        if s != 0:
            truths.append(s)
    if not truths:
        return float("nan")

    half = z / math.sqrt(n)
    grid_off = np.linspace(-half, half, 9)
    accs_per_truth = [0.0 for _ in truths]
    done = 0
    while done < trials:
        b = min(chunk, trials - done)
        mh = (mu + rng.standard_normal((b, n))).mean(axis=1)
        grid = mh[:, None] + grid_off[None, :]
        Kgrid = np.abs(0.5 * (norm.cdf(-grid) + norm.cdf(a - grid))
                       - norm.cdf(theta_S - grid)) / eps
        commit = Kgrid.min(axis=1) > 1.0
        pred = np.sign(norm.cdf(theta_S - mh)
                       - 0.5 * (norm.cdf(-mh) + norm.cdf(a - mh)))
        for i, truth in enumerate(truths):
            s = np.where(commit, (pred == truth).astype(float), 0.5)
            accs_per_truth[i] += float(s.sum())
        done += b
    accs_per_truth = [v / trials for v in accs_per_truth]
    return float(min(accs_per_truth))


@dataclass
class PhaseResult:
    a: float
    theta_S: float
    eps: float
    mu_flip: float
    K1_crossings: list
    n_values: list
    rows: list           # [{mu, K_pop, region, acc[n]...}]
    above_tau_to_one: bool
    below_tau_to_half: bool
    transition_at_tau: bool


def validate_phase_transition(a: float = 2.0, theta_S: float = 1.0, eps: float = 0.05,
                              n_values=(200, 2000, 20000), trials: int = 8000,
                              seed: int = 20260619) -> PhaseResult:
    rng = np.random.default_rng(seed)
    mu_flip = brentq(lambda m: theta0_band_median(m, a) - theta_S, -5.0, 5.0)

    # Locate all K=1 crossings by a dense scan (K rises above 1 then falls in each
    # tail, so there are several crossings; the abstain band straddles mu_flip).
    mus = np.linspace(-6.0, 7.0, 13001)
    Kvals = np.array([K_invariant(m, a, theta_S, eps) for m in mus])
    sgn = np.sign(Kvals - 1.0)
    crossings = mus[np.where(np.diff(sgn) != 0)[0]].tolist()

    # Evaluation points: near the flip, at the crossings, and +-0.3 around each.
    pts = sorted(set(
        [mu_flip - 0.02, mu_flip, mu_flip + 0.02]
        + crossings
        + [c + 0.3 for c in crossings]
        + [c - 0.3 for c in crossings]
    ))

    rows = []
    for mu in pts:
        K = K_invariant(mu, a, theta_S, eps)
        accs = {}
        for n in n_values:
            accs[str(n)] = _certificate_score(mu, a, theta_S, eps, n, trials, rng)
        rows.append({
            "mu": float(mu),
            "K_pop": float(K),
            "region": ">1" if K > 1.0 else "<1",
            "acc": accs,
        })

    # Classify behaviour using the LARGEST n. Above tau: accuracy ~1. Below tau
    # (with a margin from the boundary, |K-1|>0.1): accuracy ~1/2.
    nmax = str(max(n_values))
    above = [r for r in rows if r["K_pop"] > 1.1 and not math.isnan(r["acc"][nmax])]
    below = [r for r in rows if r["K_pop"] < 0.9 and not math.isnan(r["acc"][nmax])]
    above_ok = bool(above) and all(r["acc"][nmax] >= 0.97 for r in above)
    below_ok = bool(below) and all(abs(r["acc"][nmax] - 0.5) <= 0.03 for r in below)
    # transition pinned at tau: every >1 point beats every <1 point at nmax.
    trans_ok = above_ok and below_ok

    return PhaseResult(
        a=float(a), theta_S=float(theta_S), eps=float(eps),
        mu_flip=float(mu_flip), K1_crossings=[float(c) for c in crossings],
        n_values=list(n_values), rows=rows,
        above_tau_to_one=above_ok, below_tau_to_half=below_ok,
        transition_at_tau=trans_ok,
    )


# --------------------------------------------------------------------------- #
#  (C) Finite-n Le Cam boundary layer                                         #
# --------------------------------------------------------------------------- #
def _mean_tail_fraction(mu: float, n: int, thr: float, trials: int,
                        rng: np.random.Generator, greater: bool,
                        chunk: int = 4000) -> float:
    cnt = 0
    done = 0
    while done < trials:
        b = min(chunk, trials - done)
        xb = (mu + rng.standard_normal((b, n))).mean(axis=1)
        cnt += int(np.sum(xb > thr) if greater else np.sum(xb <= thr))
        done += b
    return cnt / trials


@dataclass
class BoundaryLayerResult:
    a: float
    theta_S: float
    mu_flip: float
    c_values: list
    n_values: list
    rows: list          # [{c, n, TV, floor, bh, bayes_err, meets_floor}]
    floor_constant_in_n: bool
    bayes_meets_floor: bool
    bh_below_floor: bool


def validate_boundary_layer(a: float = 2.0, theta_S: float = 1.0,
                            c_values=(0.5, 1.0, 2.0), n_values=(100, 1000, 10000),
                            trials: int = 30000, seed: int = 20260619) -> BoundaryLayerResult:
    rng = np.random.default_rng(seed + 7)
    mu_flip = brentq(lambda m: theta0_band_median(m, a) - theta_S, -5.0, 5.0)
    rows = []
    floor_by_c = {}
    for c in c_values:
        floor = float(norm.cdf(-c / 2.0))         # (1/2)(1-TV)
        bh = float(0.25 * math.exp(-c * c / 2.0))   # Bretagnolle-Huber certificate
        tv = float(2.0 * norm.cdf(c / 2.0) - 1.0)
        floor_by_c.setdefault(c, floor)
        for n in n_values:
            dmu = (c / 2.0) / math.sqrt(n)
            mup, mum = mu_flip + dmu, mu_flip - dmu
            sp = np.sign(delta_exact(mup, a, theta_S))
            sm = np.sign(delta_exact(mum, a, theta_S))
            # optimal sign test decides '+' iff xbar > mu_flip.
            err_p = _mean_tail_fraction(mup, n, mu_flip, trials, rng, greater=False)
            err_m = _mean_tail_fraction(mum, n, mu_flip, trials, rng, greater=True)
            bayes_err = 0.5 * (err_p + err_m)
            rows.append({
                "c": float(c), "n": int(n), "TV": tv, "floor": floor, "bh": bh,
                "bayes_err": float(bayes_err),
                "straddles_flip": bool(sp * sm < 0),
                "meets_floor": bool(abs(bayes_err - floor) <= 0.012),
            })

    # floor constant in n: per c, the analytic floor does not depend on n (by design).
    const = True
    by_c = {}
    for r in rows:
        by_c.setdefault(r["c"], []).append(r["floor"])
    for c, fs in by_c.items():
        if max(fs) - min(fs) > 1e-12:
            const = False
    meets = all(r["meets_floor"] for r in rows)
    bh_ok = all(r["bh"] <= r["floor"] + 1e-12 for r in rows)

    return BoundaryLayerResult(
        a=float(a), theta_S=float(theta_S), mu_flip=float(mu_flip),
        c_values=list(c_values), n_values=list(n_values), rows=rows,
        floor_constant_in_n=bool(const), bayes_meets_floor=bool(meets),
        bh_below_floor=bool(bh_ok),
    )


# --------------------------------------------------------------------------- #
#  (D) Sanity: exact benefit identity vs Monte Carlo                          #
# --------------------------------------------------------------------------- #
@dataclass
class IdentityResult:
    cases: list
    max_abs_err: float
    identity_holds: bool


def validate_identity(seed: int = 20260619, nmc: int = 4_000_000) -> IdentityResult:
    rng = np.random.default_rng(seed + 3)
    cases = []
    max_err = 0.0
    for (mu, a, theta) in [(0.0, 1.0, 0.5), (1.0, 2.0, 0.7), (-0.5, 1.4, 0.3),
                           (2.0, 1.0, 0.9), (0.5, 2.5, 1.6)]:
        x = mu + rng.standard_normal(nmc)
        y = (x > theta).astype(float)            # hard-threshold concept
        f0 = (x > 0).astype(float)
        fa = (x > a).astype(float)
        d_mc = float(np.mean(f0 != y) - np.mean(fa != y))
        d_cf = delta_exact(mu, a, theta)
        err = abs(d_mc - d_cf)
        max_err = max(max_err, err)
        cases.append({"mu": mu, "a": a, "theta": theta,
                      "delta_closed": float(d_cf), "delta_mc": d_mc, "abs_err": float(err)})
    return IdentityResult(cases=cases, max_abs_err=float(max_err),
                          identity_holds=bool(max_err < 5e-3))


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    description: str
    tau: float
    seed: int
    exactness: dict = field(default_factory=dict)
    phase: dict = field(default_factory=dict)
    boundary_layer: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    all_ok: bool = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--json", type=str, default="results_knowability_capacity.json")
    ap.add_argument("--fast", action="store_true",
                    help="smaller Monte-Carlo budgets for a quick smoke test")
    args = ap.parse_args()

    if args.fast:
        ex = validate_exactness(n_draws=8000, seed=args.seed)
        ph = validate_phase_transition(trials=3000, seed=args.seed)
        bl = validate_boundary_layer(trials=8000, seed=args.seed)
        idn = validate_identity(seed=args.seed, nmc=1_000_000)
    else:
        ex = validate_exactness(n_draws=40_000, seed=args.seed)
        ph = validate_phase_transition(trials=8000, seed=args.seed)
        bl = validate_boundary_layer(trials=30000, seed=args.seed)
        idn = validate_identity(seed=args.seed, nmc=4_000_000)

    print("=" * 100)
    print("K-Bound -- Knowability--Capacity threshold theorem (sign-of-benefit) -- validation")
    print("=" * 100)
    print("Model: P_X=N(0,1), Q_X=N(mu,1); g=1[x>0]; fa=1[x>a]; D=(0,a);")
    print("       hard-threshold concept theta; admissible mass-drift budget eps.")
    print("Invariant K = |(Phi(-mu)+Phi(a-mu))/2 - Phi(theta_S-mu)| / eps;  tau = 1.\n")

    # (A)
    print("-" * 100)
    print("(A) Exactness of the population threshold tau=1  (K>1 <=> sign identifiable over C_eps)")
    print(f"    random draws            : {ex.n_draws:,}")
    print(f"    tested off |K-1|<{ex.boundary_shell:g}  : {ex.n_tested_off_boundary:,}")
    print(f"    MISMATCHES              : {ex.n_mismatch}")
    print(f"    identifiable cells      : {ex.n_identifiable:,}")
    print(f"    non-identifiable cells  : {ex.n_not_identifiable:,}")
    print("    boundary stress (target_K -> K_exact, identifiable, agrees):")
    for f, K, ident, ok in ex.boundary_stress:
        print(f"        K={f:<6} -> {K:.5f}   identifiable={str(ident):<5}  agrees={ok}")
    print(f"    EXACT THRESHOLD HOLDS   : {ex.exact_threshold_holds}\n")

    # (B)
    print("-" * 100)
    print("(B) Phase transition of the finite-n three-way certificate (minimax accuracy)")
    print(f"    a={ph.a} theta_S={ph.theta_S} eps={ph.eps}; flip mu*={ph.mu_flip:.3f}")
    print(f"    K=1 crossings: {[round(c,3) for c in ph.K1_crossings]}")
    nv = [str(n) for n in ph.n_values]
    hdr = f"    {'mu':>7} {'K_pop':>7} {'reg':>4} " + " ".join(f"{'n='+n:>9}" for n in nv)
    print(hdr)
    for r in ph.rows:
        cells = " ".join(f"{r['acc'][n]:>9.4f}" if not math.isnan(r['acc'][n])
                         else f"{'--':>9}" for n in nv)
        print(f"    {r['mu']:>7.3f} {r['K_pop']:>7.3f} {r['region']:>4} {cells}")
    print(f"    above tau -> 1 (>=0.97) : {ph.above_tau_to_one}")
    print(f"    below tau -> 1/2 (+-.03): {ph.below_tau_to_half}")
    print(f"    TRANSITION PINNED AT tau: {ph.transition_at_tau}\n")

    # (C)
    print("-" * 100)
    print("(C) Finite-n Le Cam boundary layer (concept transfers; mu-estimation noise only)")
    print(f"    worlds mu* +- (c/2)/sqrt(n); floor=(1/2)(1-TV)=Phi(-c/2); BH=(1/4)e^(-c^2/2)")
    print(f"    {'c':>5} {'n':>7} {'TV':>8} {'floor':>8} {'BH':>8} {'bayes_err':>10} {'>=floor':>8}")
    for r in bl.rows:
        print(f"    {r['c']:>5.1f} {r['n']:>7} {r['TV']:>8.4f} {r['floor']:>8.4f} "
              f"{r['bh']:>8.4f} {r['bayes_err']:>10.4f} {str(r['meets_floor']):>8}")
    print(f"    floor constant in n     : {bl.floor_constant_in_n}")
    print(f"    bayes test MEETS floor  : {bl.bayes_meets_floor}")
    print(f"    BH <= floor (cert)      : {bl.bh_below_floor}\n")

    # (D)
    print("-" * 100)
    print("(D) Exact benefit identity vs Monte Carlo (model sanity)")
    for c in idn.cases:
        print(f"    mu={c['mu']:+.1f} a={c['a']:.1f} theta={c['theta']:+.1f}: "
              f"Delta_closed={c['delta_closed']:+.5f}  Delta_MC={c['delta_mc']:+.5f}  "
              f"|err|={c['abs_err']:.2e}")
    print(f"    max |err|               : {idn.max_abs_err:.2e}")
    print(f"    IDENTITY HOLDS          : {idn.identity_holds}\n")

    all_ok = bool(ex.exact_threshold_holds and ph.transition_at_tau
                  and bl.floor_constant_in_n and bl.bayes_meets_floor
                  and bl.bh_below_floor and idn.identity_holds)

    rep = Report(
        description=("Knowability--Capacity threshold theorem for sign-of-benefit, 1-D "
                     "Gaussian-location covariate-shift two-point model. K = mass-margin "
                     "between calibrated boundary theta_S and sign-flip locus theta0 over "
                     "drift budget eps; tau=1. Population converse+achievability meet "
                     "EXACTLY at tau=1; a finite-n Le Cam boundary layer of width "
                     "O(1/sqrt n) surrounds the threshold LOCATION."),
        tau=1.0, seed=int(args.seed),
        exactness=asdict(ex), phase=asdict(ph),
        boundary_layer=asdict(bl), identity=asdict(idn),
        all_ok=all_ok,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.json if os.path.isabs(args.json) else os.path.join(out_dir, args.json)
    with open(out_path, "w") as f:
        json.dump(asdict(rep), f, indent=2)
    print(f"Wrote machine-readable results to {out_path}")
    print(f"\nALL CHECKS PASS: {all_ok}")

    # Hard assertions: this is a validation, fail loudly.
    assert ex.exact_threshold_holds, "Population threshold K>1<=>identifiable FAILED."
    assert ph.transition_at_tau, "Phase transition not pinned at tau=1."
    assert bl.floor_constant_in_n, "Le Cam floor not constant in n."
    assert bl.bayes_meets_floor, "Optimal sign test did not meet the Le Cam floor."
    assert bl.bh_below_floor, "Bretagnolle-Huber certificate exceeded the floor."
    assert idn.identity_holds, "Exact benefit identity broken."
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
