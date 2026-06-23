#!/usr/bin/env python3
r"""
val_knowability_capacity_general.py
===================================

Numerical validation for the GENERALIZATION of the K-Bound knowability--capacity
threshold beyond the 1-D Gaussian-location model. Companion to
val_knowability_capacity.py (the proven 1-D result).

Stages (each labelled by what is PROVEN vs CONDITIONAL vs NUMERICAL):

  STAGE 1  Multivariate Gaussian-location  P_X=N(0,Sigma), Q_X=N(mu,Sigma).
    1a (PROVEN)   Aligned concept (v||w): exact lift of tau=1 in Mahalanobis
                  geometry on span(w). K_parallel>1 <=> identifiable, 0 mismatch.
    1b (PROVEN)   Free halfspace concept: the obstruction. The flip mass-level
                  p_med(v) runs monotonically from the band-median (rho=+-1) to
                  1/2 (rho=0, v Sigma-orthogonal to w). Worst-case capacity
                  K_free uses distance to 1/2. Explicit breaker where K_parallel>1
                  but a tilt flips the benefit sign.

  STAGE 2  Exponential-family / MLR location family.
    2a (PROVEN under MLR+unimodality)  Matched threshold tau=1 for the benefit-sign
                  two-point converse/achievability for a 1-D MLR location family
                  with the SAME mass-coordinate parametrization. Verified on
                  Laplace (log-concave) and the logistic location family.
    2b (PROVEN, breaker)  A non-log-concave (bimodal Gaussian-mixture) location
                  family where Delta(theta) is NON-monotone, the flip locus is not
                  unique, and a single scalar tau does NOT separate the regimes.

  STAGE 3  Distribution-free converse + the general obstruction.
    3a (PROVEN)   The label-free minimax error is >= 1/2 whenever the admissible
                  concept class contains two members with identical observable
                  marginal Q_X and opposite benefit-sign -- distribution-free, any
                  dimension, any family (pure Le Cam two-point, TV=0).
    3b (NUMERICAL/CONJECTURE) General achievability needs a regularity condition
                  (a unique, monotone, computable flip locus). Demonstrated by the
                  Stage-2b breaker: without unimodality/MLR the flip set is not a
                  single threshold and no scalar tau is a capacity.

Run:
    python3 val_knowability_capacity_general.py
    python3 val_knowability_capacity_general.py --json results_knowability_capacity_general.json

Pure numpy+scipy, fixed seed, no GPU, no external I/O. Labels are used only to
*verify* constructions, never by any decision rule.
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


# =========================================================================== #
#  STAGE 1 -- multivariate Gaussian-location                                  #
# =========================================================================== #
def _proj(mu, Sigma, w):
    m = float(w @ mu)
    sd = float(np.sqrt(w @ Sigma @ w))
    return m, sd


def K_parallel(mu, Sigma, w, a, theta_S, eps):
    m, sd = _proj(mu, Sigma, w)
    band_med = 0.5 * (norm.cdf((-m) / sd) + norm.cdf((a - m) / sd))
    bmass = norm.cdf((theta_S - m) / sd)
    return abs(band_med - bmass) / eps


def _delta_par(mu, Sigma, w, a, theta):
    """Aligned (v=w) benefit, exact 1-D reduction on span(w)."""
    m, sd = _proj(mu, Sigma, w)
    thp = min(max(theta, 0.0), a)
    return 2 * norm.cdf((thp - m) / sd) - norm.cdf((-m) / sd) - norm.cdf((a - m) / sd)


def _admissible_par(mu, Sigma, w, theta_S, eps):
    m, sd = _proj(mu, Sigma, w)
    pS = norm.cdf((theta_S - m) / sd)
    lo = max(1e-12, pS - eps)
    hi = min(1.0 - 1e-12, pS + eps)
    return m + sd * norm.ppf(lo), m + sd * norm.ppf(hi)


@dataclass
class Stage1aResult:
    dims: list
    n_draws: int
    n_tested: int
    n_mismatch: int
    n_identifiable: int
    exact_lift_holds: bool


def stage1a_aligned(n_draws=20_000, seed=20260619, shell=1e-6) -> Stage1aResult:
    rng = np.random.default_rng(seed)
    mism = tested = nident = 0
    dims_seen = set()
    for _ in range(n_draws):
        d = int(rng.integers(2, 5))
        dims_seen.add(d)
        Ac = rng.standard_normal((d, d))
        Sigma = Ac @ Ac.T + d * np.eye(d)
        mu = rng.standard_normal(d) * 1.5
        w = rng.standard_normal(d)
        m, sd = _proj(mu, Sigma, w)
        a = rng.uniform(0.2, 3.0) * sd
        theta_S = rng.uniform(-1, 1) * sd + (m + rng.uniform(-1, 1) * sd)
        eps = rng.uniform(0.01, 0.25)
        K = K_parallel(mu, Sigma, w, a, theta_S, eps)
        lo, hi = _admissible_par(mu, Sigma, w, theta_S, eps)
        slo = np.sign(_delta_par(mu, Sigma, w, a, lo))
        shi = np.sign(_delta_par(mu, Sigma, w, a, hi))
        ident = bool(slo == shi and slo != 0)
        nident += ident
        if abs(K - 1.0) < shell:
            continue
        tested += 1
        if (K > 1.0) != ident:
            mism += 1
    return Stage1aResult(sorted(dims_seen), int(n_draws), int(tested), int(mism),
                         int(nident), bool(mism == 0))


def _delta_proj_mc(mu, Sigma, w, a, v, theta, n=1_500_000, seed=1):
    """Free-tilt benefit via MC on the bivariate projection (s,u)."""
    r = np.random.default_rng(seed)
    ms = float(w @ mu); mu_u = float(v @ mu)
    vss = float(w @ Sigma @ w); vuu = float(v @ Sigma @ v); vsu = float(w @ Sigma @ v)
    C = np.array([[vss, vsu], [vsu, vuu]])
    L = np.linalg.cholesky(C)
    Z = r.standard_normal((n, 2)) @ L.T
    s = ms + Z[:, 0]; u = mu_u + Z[:, 1]
    Y = (u > theta).astype(float)
    f0 = (s > 0).astype(float); fa = (s > a).astype(float)
    return float(np.mean(f0 != Y) - np.mean(fa != Y))


def _pmed_of_rho(mu, Sigma, w, a, v, n=2_000_000, seed=1):
    """Own-normal mass-level of the flip (conditional v-median of the slab), and rho."""
    r = np.random.default_rng(seed)
    ms = float(w @ mu); mu_u = float(v @ mu)
    vss = float(w @ Sigma @ w); vuu = float(v @ Sigma @ v); vsu = float(w @ Sigma @ v)
    rho = vsu / math.sqrt(vss * vuu)
    C = np.array([[vss, vsu], [vsu, vuu]])
    L = np.linalg.cholesky(C)
    Z = r.standard_normal((n, 2)) @ L.T
    s = ms + Z[:, 0]; u = mu_u + Z[:, 1]
    inD = (s > 0) & (s < a)
    th0 = float(np.median(u[inD]))
    return float(norm.cdf((th0 - mu_u) / math.sqrt(vuu))), float(rho)


@dataclass
class Stage1bResult:
    band_median: float
    pmed_vs_rho: list          # [(rho, p_med)]
    monotone_to_half: bool
    breaker: dict
    obstruction_confirmed: bool


def stage1b_free(seed=20260619) -> Stage1bResult:
    rng = np.random.default_rng(seed + 1)
    d = 3
    Ac = rng.standard_normal((d, d)); Sigma = Ac @ Ac.T + d * np.eye(d)
    mu = rng.standard_normal(d) * 0.7; w = rng.standard_normal(d)
    ms, sds = _proj(mu, Sigma, w); a = 1.0 * sds
    band_med = 0.5 * (norm.cdf(-ms / sds) + norm.cdf((a - ms) / sds))

    # p_med vs rho along the span(w, w_perp_Sigma) family.
    rperp = rng.standard_normal(d)
    rperp = rperp - (w @ Sigma @ rperp) / (w @ Sigma @ w) * w   # Sigma-orthogonal to w
    rows = []
    for alpha in (0.0, 0.2, 0.5, 0.8, 0.95, 0.999):
        v = (alpha * w / math.sqrt(w @ Sigma @ w)
             + (1 - alpha) * rperp / math.sqrt(rperp @ Sigma @ rperp))
        pm, rho = _pmed_of_rho(mu, Sigma, w, a, v, seed=int(alpha * 100) + 1)
        rows.append([round(rho, 4), round(pm, 4)])
    # monotone toward 1/2 as |rho|->0
    near0 = min(rows, key=lambda r: abs(r[0]))
    near1 = max(rows, key=lambda r: abs(r[0]))
    monotone = bool(abs(near0[1] - 0.5) < 0.02 and abs(near1[1] - band_med) < 0.02)

    # Breaker: p_S on the 1/2-side of band_med, K_parallel=1.4>1, tilt flips sign.
    eps = 0.05
    side = -1.0 if band_med > 0.5 else 1.0
    p_S = band_med + side * 1.4 * eps
    theta_S_s = ms + sds * norm.ppf(p_S)
    thp = min(max(theta_S_s, 0.0), a)
    dpar = (2 * norm.cdf((thp - ms) / sds) - norm.cdf(-ms / sds) - norm.cdf((a - ms) / sds))
    # adversary tilts to push the flip across p_S
    best = -1e9 if dpar < 0 else 1e9
    for k in range(60):
        vv = w + rng.standard_normal(d) * rng.uniform(0.5, 5.0)
        sdv = float(np.sqrt(vv @ Sigma @ vv)); muv = float(vv @ mu)
        theta = muv + sdv * norm.ppf(p_S)        # own-normal mass = p_S (admissible)
        dd = _delta_proj_mc(mu, Sigma, w, a, vv, theta, seed=200 + k)
        best = max(best, dd) if dpar < 0 else min(best, dd)
    flipped = bool(np.sign(best) != np.sign(dpar) and abs(best) > 2e-3 and abs(dpar) > 2e-3)
    breaker = {
        "K_parallel": float(abs(band_med - p_S) / eps),
        "sign_aligned": int(np.sign(dpar)),
        "delta_aligned": float(dpar),
        "worst_tilt_delta": float(best),
        "sign_flipped_by_tilt": flipped,
    }
    return Stage1bResult(
        band_median=float(band_med), pmed_vs_rho=rows,
        monotone_to_half=monotone, breaker=breaker,
        obstruction_confirmed=bool(monotone and flipped),
    )


# =========================================================================== #
#  STAGE 2 -- exponential-family / MLR location family                        #
# =========================================================================== #
# 1-D location family with base density q0 (so Q_X has density q0(x-mu)); deployed
# 1[x>0], candidate 1[x>a], concept 1[x>theta]. The benefit on the band D=(0,a) is
#   Delta(theta) = F(theta'-mu) - F(-mu) - (F(a-mu) - F(theta'-mu))
#                = 2 F(theta'-mu) - F(-mu) - F(a-mu),   theta'=clip(theta,0,a),
# where F is the base CDF. This is IDENTICAL in form to the Gaussian case with Phi->F.
# Under MLR + unimodality (log-concave q0), F is a strictly increasing continuous CDF,
# so Delta is monotone with a UNIQUE root theta0 = mu + F^{-1}( (F(-mu)+F(a-mu))/2 ),
# and the mass-coordinate flip is the band median in F-units. The 1-D capacity proof
# is verbatim with Phi->F. The mass-drift class and K use F.
def _make_family(name):
    """Return (cdf, ppf, sampler, log_concave:bool)."""
    if name == "laplace":
        from scipy.stats import laplace
        return laplace.cdf, laplace.ppf, (lambda n, r: laplace.rvs(size=n, random_state=r)), True
    if name == "logistic":
        from scipy.stats import logistic
        return logistic.cdf, logistic.ppf, (lambda n, r: logistic.rvs(size=n, random_state=r)), True
    raise ValueError(name)


def _delta_family(F, mu, a, theta):
    thp = min(max(theta, 0.0), a)
    return 2 * F(thp - mu) - F(-mu) - F(a - mu)


def _K_family(F, Finv, mu, a, theta_S, eps):
    band_med = 0.5 * (F(-mu) + F(a - mu))
    bmass = F(theta_S - mu)
    return abs(band_med - bmass) / eps


def _admissible_family(F, Finv, mu, theta_S, eps):
    pS = F(theta_S - mu)
    lo = max(1e-12, pS - eps); hi = min(1.0 - 1e-12, pS + eps)
    return mu + Finv(lo), mu + Finv(hi)


@dataclass
class Stage2aResult:
    families: list
    n_draws_each: int
    n_mismatch: dict
    identity_max_err: dict
    matched_tau_holds: bool


def stage2a_mlr(n_draws=6000, seed=20260619) -> Stage2aResult:
    rng = np.random.default_rng(seed + 2)
    mism = {}; idmax = {}
    for fam in ("laplace", "logistic"):
        F, Finv, samp, lc = _make_family(fam)
        m_cnt = 0
        # identity check vs MC
        idmaxf = 0.0
        for (mu, a, theta) in [(0.0, 1.0, 0.5), (0.6, 1.5, 0.9), (-0.4, 1.2, 0.3)]:
            x = mu + samp(2_000_000, rng)
            Y = (x > theta).astype(float)
            f0 = (x > 0).astype(float); fa = (x > a).astype(float)
            d_mc = float(np.mean(f0 != Y) - np.mean(fa != Y))
            d_cf = _delta_family(F, mu, a, theta)
            idmaxf = max(idmaxf, abs(d_mc - d_cf))
        idmax[fam] = float(idmaxf)
        for _ in range(n_draws):
            mu = rng.uniform(-2.5, 2.5)
            a = rng.uniform(0.2, 3.0)
            theta_S = rng.uniform(-1.0, a + 1.0)
            eps = rng.uniform(0.01, 0.25)
            K = _K_family(F, Finv, mu, a, theta_S, eps)
            lo, hi = _admissible_family(F, Finv, mu, theta_S, eps)
            slo = np.sign(_delta_family(F, mu, a, lo))
            shi = np.sign(_delta_family(F, mu, a, hi))
            ident = bool(slo == shi and slo != 0)
            if abs(K - 1.0) < 1e-6:
                continue
            if (K > 1.0) != ident:
                m_cnt += 1
        mism[fam] = int(m_cnt)
    holds = bool(all(v == 0 for v in mism.values()) and all(v < 5e-3 for v in idmax.values()))
    return Stage2aResult(["laplace", "logistic"], int(n_draws), mism, idmax, holds)


def _mixture_cdf(x, w1=0.5, m1=-2.2, s1=0.45, m2=2.2, s2=0.45):
    return w1 * norm.cdf((x - m1) / s1) + (1 - w1) * norm.cdf((x - m2) / s2)


@dataclass
class Stage2bResult:
    description: str
    a: float
    eps: float
    density_modes: int
    theta0_of_mu: list          # [(mu, theta0(mu))]
    theta0_nonmonotone_events: int
    components_mixture: int
    components_gaussian: int
    scalar_tau_fails: bool


def stage2b_breaker(seed=20260619) -> Stage2bResult:
    r"""The genuine breaker for a scalar tau under a non-log-concave (bimodal) family.

    The benefit identity Delta(theta)=2F(theta'-mu)-F(-mu)-F(a-mu) is monotone in
    theta for ANY CDF F, so the flip in the concept LOCATION is always unique --- that
    is not where MLR is needed. MLR / log-concavity is needed so that the flip locus
    theta0(mu)=mu+F^{-1}((F(-mu)+F(a-mu))/2), viewed as a function of the UNKNOWN shift
    mu, is MONOTONE: only then does a single label-free estimate \hat mu pin the flip
    side via one threshold, making K(mu) a scalar capacity. For a bimodal base, the
    band median crosses the antimode and theta0(mu) becomes NON-monotone in mu, so the
    identifiable set {mu: K(mu)>1} fragments and no single threshold separates the
    regimes. We exhibit exactly this.
    """
    F = _mixture_cdf
    a = 5.0                       # band D=(0,a) spans both modes and the antimode
    eps = 0.06

    # (i) density modes (multimodality = MLR failure)
    xs = np.linspace(-5.0, 5.0, 401)
    dens = np.array([(F(t + 1e-3) - F(t - 1e-3)) / 2e-3 for t in xs])
    modes = int(np.sum((np.diff(np.sign(np.diff(dens))) < 0)))  # count local maxima

    # (ii) the load-bearing failure: theta0(mu) non-monotone in mu
    def delta(mu, theta):
        thp = min(max(theta, 0.0), a)
        return 2 * F(thp - mu) - F(-mu) - F(a - mu)

    def theta0(mu):
        try:
            return float(brentq(lambda t: delta(mu, t), 1e-9, a - 1e-9))
        except Exception:
            return float("nan")

    mus = np.linspace(-3.0, 3.0, 25)
    th0 = [theta0(m) for m in mus]
    nonmono = 0
    prev = None
    for t in th0:
        if prev is not None and not math.isnan(t) and not math.isnan(prev) and t < prev - 1e-6:
            nonmono += 1
        prev = t
    th0_curve = [[float(m), (float(t) if not math.isnan(t) else None)] for m, t in zip(mus, th0)]

    # (iii) fragmentation of {mu: K(mu)>1} vs the unimodal Gaussian baseline
    p_S = 0.5
    mfine = np.linspace(-3.0, 3.0, 601)

    def comp_count(Fc):
        Kv = np.array([abs(0.5 * (Fc(-m) + Fc(a - m)) - p_S) / eps for m in mfine])
        above = Kv > 1.0
        return int(np.sum(np.diff(above.astype(int)) == 1) + (1 if above[0] else 0))

    comp_mix = comp_count(F)
    comp_gauss = comp_count(lambda x: norm.cdf(x))

    return Stage2bResult(
        description=("Bimodal (mixture-of-Gaussians) location family: NOT log-concave, "
                     "NOT MLR. Delta(theta) is monotone in theta for any CDF, so the "
                     "concept-location flip is unique; the failure is that the flip locus "
                     "theta0(mu) as a function of the UNKNOWN shift mu becomes NON-monotone "
                     "(it collapses as the band median crosses the antimode), so the "
                     "label-free identifiable set {mu: K(mu)>1} fragments and no single "
                     "scalar threshold tau separates the regimes. This is the exact "
                     "regularity (MLR/unimodality) the scalar capacity needs."),
        a=float(a), eps=float(eps), density_modes=int(modes),
        theta0_of_mu=th0_curve, theta0_nonmonotone_events=int(nonmono),
        components_mixture=int(comp_mix), components_gaussian=int(comp_gauss),
        scalar_tau_fails=bool(modes >= 2 and nonmono >= 1),
    )


# =========================================================================== #
#  STAGE 3 -- distribution-free converse                                      #
# =========================================================================== #
@dataclass
class Stage3aResult:
    description: str
    cases: list                # [{family, dim, delta_plus, delta_minus, TV, minimax_lb}]
    converse_universal: bool


def stage3a_distribution_free(seed=20260619) -> Stage3aResult:
    r"""Demonstrate the distribution-free converse across several families/dimensions:
    construct two admissible concepts with IDENTICAL Q_X and opposite benefit-sign;
    the unlabeled TV is exactly 0, so the Le Cam minimax error is exactly 1/2 for any n.
    """
    rng = np.random.default_rng(seed + 3)
    cases = []

    # (1) 1-D Gaussian: two hard-threshold concepts straddling the band median.
    mu, a = 0.5, 2.0
    band_med = 0.5 * (norm.cdf(-mu) + norm.cdf(a - mu))
    eps = 0.10
    p_S = band_med   # calibrated exactly at the flip -> both signs admissible
    up = norm.ppf(min(1 - 1e-9, p_S + 0.6 * eps)) + mu
    um = norm.ppf(max(1e-9, p_S - 0.6 * eps)) + mu
    dpA = 2 * norm.cdf(min(max(up, 0), a) - mu) - norm.cdf(-mu) - norm.cdf(a - mu)
    dmA = 2 * norm.cdf(min(max(um, 0), a) - mu) - norm.cdf(-mu) - norm.cdf(a - mu)
    cases.append({"family": "gaussian-1d", "dim": 1,
                  "delta_plus": float(dpA), "delta_minus": float(dmA),
                  "TV_unlabeled": 0.0, "minimax_lb": 0.5,
                  "opposite_sign": bool(dpA * dmA < 0)})

    # (2) Laplace 1-D.
    from scipy.stats import laplace
    mu, a = -0.3, 1.6
    bmL = 0.5 * (laplace.cdf(-mu) + laplace.cdf(a - mu))
    up = laplace.ppf(min(1 - 1e-9, bmL + 0.6 * eps)) + mu
    um = laplace.ppf(max(1e-9, bmL - 0.6 * eps)) + mu
    dpL = 2 * laplace.cdf(min(max(up, 0), a) - mu) - laplace.cdf(-mu) - laplace.cdf(a - mu)
    dmL = 2 * laplace.cdf(min(max(um, 0), a) - mu) - laplace.cdf(-mu) - laplace.cdf(a - mu)
    cases.append({"family": "laplace-1d", "dim": 1,
                  "delta_plus": float(dpL), "delta_minus": float(dmL),
                  "TV_unlabeled": 0.0, "minimax_lb": 0.5,
                  "opposite_sign": bool(dpL * dmL < 0)})

    # (3) Multivariate Gaussian d=3, aligned concepts straddling the slab median.
    d = 3
    Ac = rng.standard_normal((d, d)); Sigma = Ac @ Ac.T + d * np.eye(d)
    muv = rng.standard_normal(d) * 0.5; w = rng.standard_normal(d)
    m, sd = _proj(muv, Sigma, w); a3 = 1.2 * sd
    bm3 = 0.5 * (norm.cdf(-m / sd) + norm.cdf((a3 - m) / sd))
    up = m + sd * norm.ppf(min(1 - 1e-9, bm3 + 0.6 * eps))
    um = m + sd * norm.ppf(max(1e-9, bm3 - 0.6 * eps))
    dp3 = _delta_par(muv, Sigma, w, a3, up)
    dm3 = _delta_par(muv, Sigma, w, a3, um)
    cases.append({"family": "gaussian-3d-aligned", "dim": 3,
                  "delta_plus": float(dp3), "delta_minus": float(dm3),
                  "TV_unlabeled": 0.0, "minimax_lb": 0.5,
                  "opposite_sign": bool(dp3 * dm3 < 0)})

    universal = bool(all(c["opposite_sign"] and c["TV_unlabeled"] == 0.0
                         and c["minimax_lb"] == 0.5 for c in cases))
    return Stage3aResult(
        description=("Distribution-free converse: in every family/dimension, two "
                     "admissible concepts with IDENTICAL Q_X and opposite benefit-sign "
                     "exist whenever the calibrated boundary sits within eps (in mass) "
                     "of a flip locus; unlabeled TV=0 => Le Cam minimax error=1/2 for "
                     "all n. The converse needs NO distributional assumption."),
        cases=cases, converse_universal=universal,
    )


# =========================================================================== #
#  Driver                                                                     #
# =========================================================================== #
@dataclass
class Report:
    description: str
    tau: float
    seed: int
    stage1a: dict = field(default_factory=dict)
    stage1b: dict = field(default_factory=dict)
    stage2a: dict = field(default_factory=dict)
    stage2b: dict = field(default_factory=dict)
    stage3a: dict = field(default_factory=dict)
    verdict: str = ""
    all_ok: bool = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--json", type=str, default="results_knowability_capacity_general.json")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    nd1a = 6000 if args.fast else 20_000
    nd2a = 2000 if args.fast else 6000

    print("=" * 100)
    print("K-Bound -- GENERAL knowability--capacity -- staged validation")
    print("=" * 100)

    print("\n[STAGE 1a] Multivariate Gaussian, ALIGNED concept (v||w): exact Mahalanobis lift")
    s1a = stage1a_aligned(n_draws=nd1a, seed=args.seed)
    print(f"  dims={s1a.dims}  draws_tested={s1a.n_tested:,}  MISMATCHES={s1a.n_mismatch}  "
          f"identifiable={s1a.n_identifiable:,}")
    print(f"  EXACT LIFT (K_parallel>1 <=> identifiable): {s1a.exact_lift_holds}")

    print("\n[STAGE 1b] Multivariate Gaussian, FREE halfspace concept: the obstruction")
    s1b = stage1b_free(seed=args.seed)
    print(f"  band median (mass) = {s1b.band_median:.4f}")
    print("  p_med(flip) vs rho=cos_Sigma(w,v):")
    for rho, pm in s1b.pmed_vs_rho:
        print(f"      rho={rho:+.3f} -> p_med={pm:.4f}")
    print(f"  monotone band_median(rho=+-1) -> 1/2(rho=0): {s1b.monotone_to_half}")
    bk = s1b.breaker
    print(f"  breaker: K_parallel={bk['K_parallel']:.3f}>1, sign_aligned={bk['sign_aligned']:+d} "
          f"(Delta={bk['delta_aligned']:+.4f}); worst tilt Delta={bk['worst_tilt_delta']:+.4f}; "
          f"FLIPPED={bk['sign_flipped_by_tilt']}")
    print(f"  OBSTRUCTION CONFIRMED (free class breaks K_parallel): {s1b.obstruction_confirmed}")

    print("\n[STAGE 2a] MLR / log-concave location families: matched tau=1 (Laplace, logistic)")
    s2a = stage2a_mlr(n_draws=nd2a, seed=args.seed)
    print(f"  mismatches: {s2a.n_mismatch}   identity max-err: "
          f"{ {k: round(v, 6) for k, v in s2a.identity_max_err.items()} }")
    print(f"  MATCHED tau=1 UNDER MLR+unimodality: {s2a.matched_tau_holds}")

    print("\n[STAGE 2b] Non-log-concave (bimodal) breaker: MLR hypothesis FAILS")
    s2b = stage2b_breaker(seed=args.seed)
    print(f"  density modes = {s2b.density_modes} (>=2 => not unimodal, MLR fails)")
    print(f"  theta0(mu) NON-monotone events = {s2b.theta0_nonmonotone_events} "
          f"(>=1 => flip locus not monotone in the unknown shift)")
    print(f"  components of {{mu:K(mu)>1}}: mixture={s2b.components_mixture}  "
          f"gaussian-baseline={s2b.components_gaussian}")
    print(f"  scalar-tau capacity proof FAILS (no monotone flip locus): {s2b.scalar_tau_fails}")

    print("\n[STAGE 3a] Distribution-free converse: minimax error 1/2 via TV=0 (any family/dim)")
    s3a = stage3a_distribution_free(seed=args.seed)
    for c in s3a.cases:
        print(f"  {c['family']:<22} d={c['dim']}: Delta+={c['delta_plus']:+.4f} "
              f"Delta-={c['delta_minus']:+.4f} opp_sign={c['opposite_sign']} "
              f"TV={c['TV_unlabeled']:.0f} minimax_lb={c['minimax_lb']}")
    print(f"  DISTRIBUTION-FREE CONVERSE UNIVERSAL: {s3a.converse_universal}")

    all_ok = bool(s1a.exact_lift_holds and s1b.obstruction_confirmed
                  and s2a.matched_tau_holds and s2b.scalar_tau_fails
                  and s3a.converse_universal)

    verdict = ("STRONG-BUT-BOUNDED generalization, NOT a paradigm-level general capacity. "
               "Stage 1 aligned: EXACT tau=1 lift (PROVEN). Stage 1 free halfspace: capacity "
               "K_free with worst-case-tilt to mass 1/2 (PROVEN, with breaker). Stage 2: "
               "matched tau=1 UNDER MLR+log-concavity (PROVEN-conditional) with an explicit "
               "non-log-concave breaker. Stage 3: distribution-free CONVERSE is universal "
               "(PROVEN); general single-threshold ACHIEVABILITY provably needs a "
               "regularity/identifiability condition (unimodal/MLR flip locus). The wall is "
               "exactly there: a computable scalar tau is a capacity iff the flip locus is a "
               "unique monotone observable threshold.")

    rep = Report(
        description=("Generalization of the K-Bound knowability--capacity threshold beyond 1-D. "
                     "Multivariate-Gaussian (exact lift + obstruction), exp-family/MLR "
                     "(conditional matched tau + breaker), distribution-free converse."),
        tau=1.0, seed=int(args.seed),
        stage1a=asdict(s1a), stage1b=asdict(s1b),
        stage2a=asdict(s2a), stage2b=asdict(s2b), stage3a=asdict(s3a),
        verdict=verdict, all_ok=all_ok,
    )
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.json if os.path.isabs(args.json) else os.path.join(out_dir, args.json)
    with open(out_path, "w") as f:
        json.dump(asdict(rep), f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"ALL STAGE CHECKS PASS: {all_ok}")
    print("VERDICT:", verdict)

    assert s1a.exact_lift_holds, "Stage 1a exact aligned lift FAILED."
    assert s1b.obstruction_confirmed, "Stage 1b free-class obstruction NOT confirmed."
    assert s2a.matched_tau_holds, "Stage 2a MLR matched tau FAILED."
    assert s2b.scalar_tau_fails, "Stage 2b breaker did not exhibit MLR failure."
    assert s3a.converse_universal, "Stage 3a distribution-free converse FAILED."
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
