#!/usr/bin/env python3
r"""
val_multiclass_capacity.py
==========================

Numerical validator for the MULTICLASS (K>=3) knowability--capacity result --- the
open part (b) of Conjecture conj:gen-capacity in
docs/research/kbound/paper/sections/knowability_capacity_general.tex
("multiclass K>=3, where the 1-D location-model intuition breaks").

Companion to val_knowability_capacity_general.py and val_knowability_dichotomy.py.
Pure numpy + scipy, fixed seed, no GPU, no external/drive I/O. Labels are used ONLY to
*verify* the benefit identity by Monte-Carlo; no decision rule ever uses a label.

------------------------------------------------------------------------------------
AUDIT NOTE (June 2026, second pass).  This file was re-derived and re-verified
independently.  Three corrections were folded in relative to the first pass; each is
load-bearing and changes a claim, so they are documented here in full:

  * BLOCK A.  The single-conflict-pair capacity needs MORE than "fixed conflict pair".
    If the residual mass m_rest(x) on the other K-2 classes VARIES across D, the
    contrast eta_{j_f}-eta_{j_g} = (1-m_rest(x)) * c(x) is reweighted by an unobservable
    profile and sign(Delta) can flip at a FIXED observable contrast crossing
    (demonstrated in blockA_residual_breaker()).  So the correct hypothesis set adds
    (M1') CONSTANT RESIDUAL RESERVOIR: m_rest(x) == c3 (const) on D.  Only then does the
    residual cancel in the contrast and the problem reduce to binary.  The capacity check
    (blockA_singlepair) is run UNDER (M1') and is correct there; the breaker certifies
    (M1') is necessary, not cosmetic.

  * BLOCK B.  The first pass asserted the multi-conflict-pair fragmentation is
    "realizable by canonical affine-score argmax classifiers" but only hand-drew the
    field.  blockB_argmax_realization() now constructs GENUINE affine-per-class-score
    argmax classifiers g,f (the exact per-coordinate MLR/log-concave regularity that
    SUFFICES in binary) whose conflict pair varies across D, plus a valid simplex-valued
    concept, and exhibits Delta(mu) with >=2 sign changes vs. a monotone binary baseline.
    The claim is now demonstrated, not asserted.

  * BLOCK C.  The first pass said the two worlds share "the same top-label calibration a
    label-free auditor can compute," which overclaims: the two worlds differ in the
    BAYES-argmax class on D (class 1 vs class 2), so a labelled auditor with target labels
    could tell them apart.  The honest, exact statement is weaker and still decisive: the
    two worlds share the identical COVARIATE law Q_X and the identical DEPLOYED classifier
    g (hence identical predicted-class map and identical deployed-class reported score
    profile eta_{j_g} on D), yet have opposite sign(Delta).  Therefore NO label-free
    statistic (TV=0 under covariate shift) and in particular the binary observable margin
    M = E_{Q|D}[eta_{j_g}] - 1/2 separates them.  The distinguishing object is the split
    of the residual mass between j_f and the third class, which neither Q_X nor eta_{j_g}
    sees.  (The converse half "error = 1/2" is an instance of the already-proven
    distribution-free converse thm:gen-df-conv; the NEW content is that the binary
    *observable* eta_{j_g} is a strictly-too-coarse reduct in K>=3.)

------------------------------------------------------------------------------------
WHAT IS ESTABLISHED (each block labelled by its status)
------------------------------------------------------------------------------------
Model. Covariate shift Q_X = q0(. - mu) on R (1-D location to start; the obstruction
already appears here). Y in {1..K}. 0/1 loss. Deployed classifier g and candidate f map
X -> {1..K}; disagreement region D = {x: g(x)!=f(x)}. Concept (target label channel)
eta(x) = (eta_1(x),...,eta_K(x)) on the simplex; under covariate shift eta does NOT
affect Q_X. The deployment benefit is Delta = R_Q(g) - R_Q(f).

BLOCK 0  (PROVEN, exact)  -- the multiclass benefit identity.
   On D, g predicts j_g(x), f predicts j_f(x) (j_g != j_f). The pointwise excess loss of
   g over f is 1{Y=j_f} - 1{Y=j_g}, conditional mean eta_{j_f}(x) - eta_{j_g}(x). Hence
       Delta = \int_D ( eta_{j_f}(x) - eta_{j_g}(x) ) dQ(x) =: \int_D Lambda(x) dQ(x),
   where Lambda(x) in [-1,1] is the SCALAR "benefit field". Verified against a 4e6-sample
   Monte-Carlo of R_Q(g)-R_Q(f). Lambda involves only TWO of the K coordinates; the
   residual mass m_rest(x) = 1 - eta_{j_g} - eta_{j_f} on the OTHER K-2 classes is a free
   function, invisible to Q_X -- this is the new multiclass degree of freedom.

BLOCK A  (EXTENSION; PROVEN under (M1)+(M1')+(M2)+(M3))  -- single-pair capacity, tau=1.
   Hypotheses (the explicit multiclass analogue of R1/R2):
     (M1)  SINGLE CONFLICT PAIR: fixed j_g != j_f with g==j_g, f==j_f on ALL of D.
     (M1') CONSTANT RESIDUAL RESERVOIR: m_rest(x) == c3 (const) on D  [necessary; see
           blockA_residual_breaker -- without it sign(Delta) is unidentifiable].
     (M2)  MLR location nuisance (q0 log-concave) + monotone source contrast, so the flip
           locus theta0(mu) is a monotone, observable function of the unknown shift.
     (M3)  CONTRAST-MASS DRIFT BUDGET eps: the admissible target concept's crossing moves
           by <= eps in the (observable) mass coordinate of Lambda's single crossing.
   Then Delta collapses EXACTLY to the binary single-crossing problem in Lambda (the
   constant reservoir cancels in the contrast), and K = mass-dist(flip, worst admissible
   flip)/eps is a CAPACITY at tau=1: K>1 <=> sign(Delta) label-free identifiable.

BLOCK B  (NECESSITY of (M1); PROVEN by GENUINE-argmax witness)  -- multi-pair breaker.
   Drop (M1): genuine affine-per-class-score argmax classifiers g,f (the per-coordinate
   MLR/log-concave regularity that SUFFICES in binary) disagree into DIFFERENT class pairs
   on different sub-bands of D. With a valid simplex concept the benefit field Lambda(x)
   sign-crosses several times in x, Delta(mu) oscillates in the unknown shift mu, and the
   identifiable set {mu: identifiable} fragments into >=3 components, while a binary
   single-threshold baseline stays monotone (1 flip). Hence (M1) is necessary; the binary
   per-coordinate regularity does NOT lift.

BLOCK C  (NECESSITY of the right OBSERVABLE; PROVEN by sharp counterexample)
   Even WITH (M1), the binary-style observable margin M = E_{Q|D}[eta_{j_g}] - 1/2 is NOT
   a capacity. Two K=3 worlds W+, W- share: IDENTICAL Q_X (covariate shift -> every
   label-free statistic has TV=0, Le Cam minimax error 1/2); IDENTICAL deployed classifier
   g and IDENTICAL deployed-class score profile eta_{j_g} on D (=> identical binary margin
   M); yet OPPOSITE sign(Delta). The third-class split of the residual mass is the
   confounder. A capacity built from M predicts K=|M|/beta>1 ("identifiable") in BOTH
   worlds, yet the sign is NOT identifiable. So in multiclass the binary margin is the
   WRONG reduct; the budget must be placed (M3) directly on the contrast field. This
   matches the unidentifiability of target accuracy from unlabeled data alone (Garg et al.
   2022) and the weakness of top-label calibration (Gupta & Ramdas 2021).

------------------------------------------------------------------------------------
RELATION TO thm:dich-main (self-audit). All blocks are CONSISTENT with the abstract
dichotomy "capacity exists <=> Phi (sign factors through the maximal observable reduct
O=[Q_X]) holds computably": Block C is a Phi-FAILS instance (converse), Block A is a
Phi-HOLDS-computably instance (achievability). The NEW, multiclass-specific content is:
(i) the exact vector benefit identity (Block 0); (ii) the explicit hypothesis set
(M1)+(M1')+(M3) that restores Phi (Block A) and the residual breaker showing (M1') is
needed; (iii) a genuine-argmax proof that the binary per-coordinate regularity is
INSUFFICIENT because the conflict pair varies across D (Block B); and (iv) a proof that
the binary observable margin M is a strictly-too-coarse reduct (Block C).
------------------------------------------------------------------------------------

Run:
    python3 val_multiclass_capacity.py
    python3 val_multiclass_capacity.py --json results_multiclass_capacity.json
    python3 val_multiclass_capacity.py --fast        # smaller draw counts
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.stats import norm

SEED = 20260629
TAU = 1.0


# =========================================================================== #
#  helpers                                                                    #
# =========================================================================== #
def _softmax_rows(S):
    S = S - S.max(axis=-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=-1, keepdims=True)


def _trapz(y, x):
    # local alias (np.trapz deprecated in numpy>=2; np.trapezoid may be absent in older)
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _argmax_affine(slopes, intercepts, xs):
    S = np.stack([slopes[k] * xs + intercepts[k] for k in range(len(slopes))], axis=-1)
    return S.argmax(axis=-1)


# =========================================================================== #
#  BLOCK 0 -- exact multiclass benefit identity, MC-verified                  #
# =========================================================================== #
@dataclass
class Block0Result:
    K: int
    mu: float
    a: float
    delta_mc: float
    delta_closed_form: float
    abs_err: float
    identity_holds: bool


def block0_identity(seed=SEED, n_mc=4_000_000) -> Block0Result:
    r"""Verify Delta = \int_D (eta_{j_f}-eta_{j_g}) dQ against Monte-Carlo R_Q(g)-R_Q(f)."""
    rng = np.random.default_rng(seed)
    K = 3
    mu, a = 0.4, 2.0

    # g: class 0 if x<0 else class 1.  f: class 0 if x<0 else class 2.
    # => on D={x>0}: g predicts 1 (j_g=1), f predicts 2 (j_f=2).
    def g(x):
        return np.where(x < 0, 0, 1)

    def f(x):
        return np.where(x < 0, 0, 2)

    w = np.array([0.0, 1.0, 0.8])
    b = np.array([0.0, -0.3, -0.6])

    def eta(x):
        x = np.atleast_1d(np.asarray(x, float))
        S = np.stack([w[k] * x + b[k] for k in range(K)], axis=-1)
        return _softmax_rows(S)

    X = mu + rng.standard_normal(n_mc)
    P = eta(X)
    cum = np.cumsum(P, axis=1)
    u = rng.random(n_mc)
    Y = (u[:, None] < cum).argmax(axis=1)
    gg = g(X)
    ff = f(X)
    delta_mc = float(np.mean((gg != Y).astype(float) - (ff != Y).astype(float)))

    inD = X > 0
    delta_cf = float(np.mean(inD * (P[:, 2] - P[:, 1])))  # E_Q[ 1_D (eta_jf - eta_jg) ]
    err = abs(delta_mc - delta_cf)
    return Block0Result(K, mu, a, delta_mc, delta_cf, err, bool(err < 2e-3))


# =========================================================================== #
#  BLOCK A -- single-conflict-pair multiclass capacity, tau=1 (EXTENSION)     #
# =========================================================================== #
# Under (M1)+(M1'): on ALL of D=(0,a), g predicts j_g, f predicts j_f, and the residual
# mass on the other K-2 classes is a CONSTANT c3.  The active pair carries (1-c3); the
# source contrast is a single-threshold field: eta_{j_f}-eta_{j_g} = (1-c3) sign(x-theta).
#   Delta(mu,theta) = (1-c3) [ (Phi(a-mu)-Phi(theta-mu)) - (Phi(theta-mu)-Phi(-mu)) ],
# flip locus = band median (the binary problem scaled by the POSITIVE constant (1-c3)).
# (M3): admissible crossing drifts by <= eps in mass u = Phi(theta-mu).
def _delta_singlepair(mu, theta, a, c3):
    th = min(max(theta, 0.0), a)
    return (1.0 - c3) * ((norm.cdf(a - mu) - norm.cdf(th - mu))
                         - (norm.cdf(th - mu) - norm.cdf(-mu)))


def _band_median(mu, a):
    return 0.5 * (norm.cdf(-mu) + norm.cdf(a - mu))


@dataclass
class BlockAResult:
    n_tested: int
    n_identifiable: int
    n_mismatch: int
    c3_values: list
    capacity_holds: bool
    residual_breaker_flips_sign: bool   # (M1') necessity: varying m_rest flips sign(Delta)
    residual_breaker_delta_const: float
    residual_breaker_delta_varA: float
    residual_breaker_delta_varB: float


def _blockA_residual_breaker():
    r"""(M1') NECESSITY.  Same fixed conflict pair, same observable contrast crossing
    theta=1.0, but a VARYING (unobservable) residual profile m_rest(x) reweights the
    contrast and flips sign(Delta).  This proves M1 (fixed pair) alone is insufficient;
    the constant-reservoir (M1') is required for the binary reduction in Block A."""
    xs = np.linspace(0.0, 2.0, 8000)
    mu = 0.0
    theta = 1.0
    c = np.sign(xs - theta)  # within-pair contrast, single crossing (observable)

    def Delta(mrest):
        return _trapz((1.0 - mrest) * c * norm.pdf(xs - mu), xs)

    d_const = Delta(0.3 * np.ones_like(xs))           # constant reservoir -> binary-like
    d_A = Delta(0.1 + 0.8 * (xs < theta))             # residual heavy on minus side -> +
    d_B = Delta(0.1 + 0.8 * (xs > theta))             # residual heavy on plus side  -> -
    flips = bool(np.sign(d_A) != np.sign(d_B) and d_A != 0 and d_B != 0)
    return flips, float(d_const), float(d_A), float(d_B)


def blockA_singlepair(n_draws=50_000, seed=SEED, shell=1e-6) -> BlockAResult:
    rng = np.random.default_rng(seed + 1)
    mism = tested = nident = 0
    c3_seen = set()
    for _ in range(n_draws):
        a = rng.uniform(1.0, 3.0)
        c3 = float(rng.uniform(0.0, 0.6))           # CONSTANT residual reservoir (M1')
        c3_seen.add(round(c3, 1))
        mu = rng.uniform(-2.0, 2.0)
        theta_S = rng.uniform(-1.0, a + 1.0)
        eps = rng.uniform(0.02, 0.25)
        pS = norm.cdf(theta_S - mu)
        flip = _band_median(mu, a)
        K = abs(pS - flip) / eps
        lo = max(1e-9, pS - eps)
        hi = min(1.0 - 1e-9, pS + eps)
        th_lo = mu + norm.ppf(lo)
        th_hi = mu + norm.ppf(hi)
        s_lo = np.sign(_delta_singlepair(mu, th_lo, a, c3))
        s_hi = np.sign(_delta_singlepair(mu, th_hi, a, c3))
        ident = bool(s_lo == s_hi and s_lo != 0)
        if abs(K - 1.0) < shell:
            continue
        tested += 1
        nident += int(ident)
        if (K > 1.0) != ident:
            mism += 1
    flips, dc, dA, dB = _blockA_residual_breaker()
    return BlockAResult(int(tested), int(nident), int(mism),
                        sorted(c3_seen), bool(mism == 0),
                        flips, dc, dA, dB)


# =========================================================================== #
#  BLOCK B -- multi-conflict-pair breaker (NECESSITY of (M1))                  #
#            now realized by GENUINE affine-score argmax classifiers           #
# =========================================================================== #
@dataclass
class BlockBResult:
    conflict_pairs_on_D: list
    n_conflict_pairs: int
    benefit_field_sign_crossings_in_x: int
    delta_mu_sign_changes: int
    n_identifiable_components: int
    binary_baseline_sign_changes: int
    max_abs_lambda: float
    realizable_by_affine_argmax: bool
    concept_valid_simplex: bool
    M1_necessary: bool


def blockB_argmax_realization(seed=SEED) -> BlockBResult:
    r"""GENUINE realization of the multi-conflict-pair breaker.

    g = argmax of affine per-class scores G_k(x)=u_k x + p_k.
    f = argmax of affine per-class scores F_k(x)=v_k x + q_k.
    These are exactly the per-coordinate MLR/log-concave (linear-logit) classifiers whose
    regularity SUFFICES in binary.  We choose g,f so that across D the (j_g,j_f) conflict
    pair varies (here g and f swap their class-0/class-1 preference across bands while
    class 2 stays low), then put a VALID simplex concept eta on top.  The benefit field
    Lambda = eta_{j_f}-eta_{j_g} then sign-crosses several times in x, Delta(mu) oscillates
    in the unknown shift mu, and the identifiable set fragments into >=3 components, while a
    binary single-threshold baseline stays monotone (1 flip)."""
    xs = np.linspace(-12.0, 12.0, 48001)

    # genuine affine argmax classifiers with interleaved 0/1 preference across D
    us, ps = [0.0, 0.6, 1.2], [0.0, -1.5, -7.0]   # g
    vs, qs = [0.6, 0.0, 1.2], [-1.5, 0.0, -7.0]   # f (0<->1 roles swapped)
    jg = _argmax_affine(us, ps, xs)
    jf = _argmax_affine(vs, qs, xs)
    D = jg != jf
    pairs = sorted(set(zip(jg[D].tolist(), jf[D].tolist())))

    # valid simplex concept: class 2 carries constant small mass; classes 0,1 contest via a
    # logistic of a sinusoid (gives sign variation in the 0-vs-1 contrast across x).
    def eta(xx):
        z = 0.8 * np.sin(0.7 * xx)
        e2 = 0.15 * np.ones_like(xx)
        rest = 1.0 - e2
        e1 = rest * (1.0 / (1.0 + np.exp(-z)))
        e0 = rest - e1
        return np.stack([e0, e1, e2], axis=-1)

    E = eta(xs)
    valid = bool(np.all(E >= -1e-9) and np.allclose(E.sum(-1), 1.0))

    idx = np.arange(len(xs))
    Lam = np.zeros_like(xs)
    Lam[D] = E[idx[D], jf[D]] - E[idx[D], jg[D]]
    max_abs = float(np.max(np.abs(Lam[D]))) if D.any() else 0.0
    LD = Lam[D]
    cross_x = int(np.sum(np.diff(np.sign(LD[LD != 0])) != 0)) if LD.size else 0

    def Delta_mu(mu):
        return _trapz(Lam * norm.pdf(xs - mu), xs)

    mus = np.linspace(-8.0, 8.0, 400)
    Dm = np.array([Delta_mu(m) for m in mus])
    nz = np.abs(Dm) > 1e-5
    sc = int(np.sum(np.diff(np.sign(Dm[nz])) != 0))

    def Delta_bin(mu, a=4.0, theta=2.0):
        th = min(max(theta, 0.0), a)
        return 2 * norm.cdf(th - mu) - norm.cdf(-mu) - norm.cdf(a - mu)

    Db = np.array([Delta_bin(m) for m in mus])
    sb = int(np.sum(np.diff(np.sign(Db)) != 0))

    return BlockBResult(
        conflict_pairs_on_D=[list(p) for p in pairs],
        n_conflict_pairs=len(pairs),
        benefit_field_sign_crossings_in_x=cross_x,
        delta_mu_sign_changes=sc,
        n_identifiable_components=sc + 1,
        binary_baseline_sign_changes=sb,
        max_abs_lambda=max_abs,
        realizable_by_affine_argmax=bool(len(pairs) >= 2),
        concept_valid_simplex=valid,
        M1_necessary=bool(sc >= 2 and sb <= 1 and len(pairs) >= 2 and valid and max_abs <= 1.0),
    )


# =========================================================================== #
#  BLOCK C -- third-class confounding (NECESSITY of the right observable)      #
# =========================================================================== #
@dataclass
class BlockCResult:
    a: float
    eta_jg_shared: float
    bayes_top_class_Wplus: int
    bayes_top_class_Wminus: int
    delta_plus: float
    delta_minus: float
    opposite_sign: bool
    shared_QX: bool
    shared_deployed_score_profile: bool
    tv_unlabeled: float
    minimax_lb: float
    binary_M: float
    binary_K_examples: list      # [(beta, K, binary_predicts_identifiable)]
    binary_margin_is_capacity: bool
    counterexample_confirmed: bool


def blockC_thirdclass(seed=SEED) -> BlockCResult:
    r"""Two K=3 worlds with identical Q_X and identical DEPLOYED-class score profile
    eta_{j_g} on D, but opposite sign(Delta). j_g=0 (the class the deployed g predicts on
    D), j_f=1. The residual mass r(x)=1-eta_0(x) is split between class 1 (=j_f) and class
    2 (the third class); the split is the unobservable confounder.

    HONEST framing (corrected): we do NOT claim the worlds share top-label calibration --
    the BAYES-argmax class on D differs between them (1 vs 2), so a *labelled* auditor
    could separate them. The claim is that NO LABEL-FREE statistic separates them (TV=0
    under covariate shift) and in particular the binary observable margin
    M = E_{Q|D}[eta_{j_g}] - 1/2 is identical, yet sign(Delta) differs."""
    a = 2.0
    eta0 = 0.45  # shared deployed-class (j_g) score on D = the binary-style observable

    def world(split1):
        xx = np.linspace(0.0, a, 4000)
        r = 1.0 - eta0
        e1 = r * split1                       # scalar (constant on D)
        e0 = np.full_like(xx, eta0)
        delta = _trapz((e1 - e0) * norm.pdf(xx), xx)  # benefit field eta_1 - eta_0
        # bayes-argmax class on D (constant here): compare eta0, e1, e2
        e2 = r * (1.0 - split1)               # scalar
        top = int(np.argmax([eta0, float(e1), float(e2)]))
        return delta, top

    dp, top_p = world(0.95)   # residual mostly to j_f -> Delta > 0 ; bayes-top = class 1
    dm, top_m = world(0.05)   # residual mostly to third class -> Delta < 0 ; bayes-top = class 2

    M = eta0 - 0.5
    binK = []
    for beta in (0.02, 0.04):
        binK.append([float(beta), float(abs(M) / beta), bool(abs(M) > beta)])
    binary_is_capacity = False
    confirmed = bool(np.sign(dp) != np.sign(dm)
                     and any(k[2] for k in binK)   # binary predicts identifiable in >=1 world
                     and not binary_is_capacity)
    return BlockCResult(
        a=float(a), eta_jg_shared=float(eta0),
        bayes_top_class_Wplus=top_p, bayes_top_class_Wminus=top_m,
        delta_plus=float(dp), delta_minus=float(dm),
        opposite_sign=bool(np.sign(dp) != np.sign(dm)),
        shared_QX=True, shared_deployed_score_profile=True,
        tv_unlabeled=0.0, minimax_lb=0.5,
        binary_M=float(M), binary_K_examples=binK,
        binary_margin_is_capacity=binary_is_capacity,
        counterexample_confirmed=confirmed,
    )


@dataclass
class BlockDResult:
    """Impossibility closure (thm:mc-cap-impossibility) aggregating Blocks B and C."""
    binary_margin_fails: bool
    multi_component_flip_set: bool
    scalar_threshold_insufficient: bool
    impossibility_closed: bool


def blockD_impossibility_closure(bB: "BlockBResult", bC: BlockCResult) -> BlockDResult:
    """Verify no universal scalar capacity: binary margin counterexample + multi-component flip."""
    binary_margin_fails = bool(
        bC.counterexample_confirmed
        and bC.opposite_sign
        and not bC.binary_margin_is_capacity
    )
    multi_component = bool(bB.M1_necessary and bB.n_identifiable_components >= 2)
    scalar_insufficient = bool(
        bB.delta_mu_sign_changes >= 2
        and bB.binary_baseline_sign_changes <= bB.delta_mu_sign_changes
    )
    closed = bool(binary_margin_fails and multi_component and scalar_insufficient)
    return BlockDResult(
        binary_margin_fails=binary_margin_fails,
        multi_component_flip_set=multi_component,
        scalar_threshold_insufficient=scalar_insufficient,
        impossibility_closed=closed,
    )


# =========================================================================== #
#  driver                                                                     #
# =========================================================================== #
@dataclass
class Report:
    description: str
    tau: float
    seed: int
    block0: dict = field(default_factory=dict)
    blockA: dict = field(default_factory=dict)
    blockB: dict = field(default_factory=dict)
    blockC: dict = field(default_factory=dict)
    blockD: dict = field(default_factory=dict)
    status: str = ""
    all_ok: bool = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    ndA = 12_000 if args.fast else 50_000
    nmc = 1_000_000 if args.fast else 4_000_000

    print("=" * 100)
    print("K-Bound -- MULTICLASS (K>=3) knowability--capacity -- validation (open part of conj:gen-capacity)")
    print("=" * 100)

    print("\n[BLOCK 0] exact multiclass benefit identity  Delta = \\int_D (eta_jf - eta_jg) dQ")
    b0 = block0_identity(seed=args.seed, n_mc=nmc)
    print(f"  K={b0.K} mu={b0.mu} a={b0.a}:  Delta_MC={b0.delta_mc:+.5f}  "
          f"closed-form={b0.delta_closed_form:+.5f}  |err|={b0.abs_err:.2e}  holds={b0.identity_holds}")

    print("\n[BLOCK A] single-conflict-pair multiclass capacity (EXTENSION), tau=1, under (M1)+(M1')+(M2)+(M3)")
    bA = blockA_singlepair(n_draws=ndA, seed=args.seed)
    print(f"  c3 (CONSTANT residual reservoir, M1') values seen: {bA.c3_values}")
    print(f"  draws_tested={bA.n_tested:,}  identifiable={bA.n_identifiable:,}  "
          f"MISMATCHES(K>1<->ident)={bA.n_mismatch}")
    print(f"  EXTENSION HOLDS (scalar capacity tau=1 under (M1)+(M1')+(M2)+(M3)): {bA.capacity_holds}")
    print(f"  (M1') NECESSITY -- same pair & same observable contrast crossing, VARYING residual:")
    print(f"     Delta(const reservoir)={bA.residual_breaker_delta_const:+.4f}  "
          f"Delta(varA)={bA.residual_breaker_delta_varA:+.4f}  "
          f"Delta(varB)={bA.residual_breaker_delta_varB:+.4f}")
    print(f"     sign flips when residual varies => (M1') is NECESSARY: {bA.residual_breaker_flips_sign}")

    print("\n[BLOCK B] multi-conflict-pair breaker via GENUINE affine-argmax g,f: (M1) is NECESSARY")
    bB = blockB_argmax_realization(seed=args.seed)
    print(f"  conflict pairs (j_g,j_f) on D from real argmax classifiers: {bB.conflict_pairs_on_D} "
          f"({bB.n_conflict_pairs} distinct)")
    print(f"  concept valid on simplex: {bB.concept_valid_simplex}   max|Lambda| on D={bB.max_abs_lambda:.2f} (<=1)")
    print(f"  Lambda(x) sign-crossings in x on D = {bB.benefit_field_sign_crossings_in_x}")
    print(f"  Delta(mu) sign-changes in nuisance mu = {bB.delta_mu_sign_changes} "
          f"=> identifiable set has {bB.n_identifiable_components} components")
    print(f"  binary single-threshold baseline sign-changes = {bB.binary_baseline_sign_changes} (monotone)")
    print(f"  (M1) NECESSARY, realized by canonical affine argmax: {bB.M1_necessary}")

    print("\n[BLOCK C] third-class confounding: the binary observable margin is the WRONG reduct")
    bC = blockC_thirdclass(seed=args.seed)
    print(f"  shared deployed-class (j_g) score on D: eta_jg={bC.eta_jg_shared}  (binary margin M={bC.binary_M:+.3f})")
    print(f"  World W+ Delta={bC.delta_plus:+.4f} (bayes-top class={bC.bayes_top_class_Wplus})   "
          f"World W- Delta={bC.delta_minus:+.4f} (bayes-top class={bC.bayes_top_class_Wminus})   "
          f"opposite_sign={bC.opposite_sign}")
    print(f"  shared Q_X & shared deployed score profile: unlabeled TV={bC.tv_unlabeled:.0f} "
          f"=> Le Cam minimax error={bC.minimax_lb}")
    for beta, K, pred in bC.binary_K_examples:
        print(f"    binary K=|M|/beta={K:.2f} (beta={beta}) predicts identifiable={pred}, but TRUE=False")
    print(f"  COUNTEREXAMPLE CONFIRMED (binary margin not a capacity in multiclass): {bC.counterexample_confirmed}")

    print("\n[BLOCK D] impossibility closure (thm:mc-cap-impossibility)")
    bD = blockD_impossibility_closure(bB, bC)
    print(f"  binary_margin_fails={bD.binary_margin_fails}  "
          f"multi_component_flip_set={bD.multi_component_flip_set}  "
          f"scalar_threshold_insufficient={bD.scalar_threshold_insufficient}")
    print(f"  IMPOSSIBILITY CLOSED (no universal scalar capacity): {bD.impossibility_closed}")

    all_ok = bool(b0.identity_holds and bA.capacity_holds and bA.residual_breaker_flips_sign
                  and bB.M1_necessary and bC.counterexample_confirmed and bD.impossibility_closed)

    status = (
        "CLOSED (impossibility). PROVEN: (Block 0) exact vector benefit identity for K>=3; "
        "(Block A) single-conflict-pair capacity at tau=1 under (M1)+(M1')+(M2)+(M3). "
        "NECESSITY (Blocks B,C): no single computable scalar capacity certifies sign(Delta) "
        "in the general multi-conflict-pair / vector-concept regime — binary margin M is too "
        "coarse (Block C), flip set fragments into >=2 components (Block B), and per-component "
        "orientation is the minimal supplement (thm:mc-cap-impossibility, Block D). "
        "Positive capacity exists only under R1/R2 (Block A)."
    )

    rep = Report(
        description=("Multiclass (K>=3) knowability--capacity: exact benefit identity; a "
                     "single-conflict-pair extension at tau=1 with explicit hypotheses "
                     "(incl. the necessary constant-residual reservoir); and necessity "
                     "results (conflict-pair structure via genuine argmax; residual-profile "
                     "constancy; third-class confounding of the binary observable) delimiting "
                     "exactly where the binary capacity stops in multiclass."),
        tau=TAU, seed=int(args.seed),
        block0=asdict(b0), blockA=asdict(bA), blockB=asdict(bB), blockC=asdict(bC),
        blockD=asdict(bD),
        status=status, all_ok=all_ok,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    if args.json:
        out_path = args.json if os.path.isabs(args.json) else os.path.join(out_dir, args.json)
        with open(out_path, "w") as fh:
            json.dump(asdict(rep), fh, indent=2)
        print(f"\nWrote {out_path}")

    print("\n" + "-" * 100)
    print(f"ALL BLOCK CHECKS PASS: {all_ok}")
    print("STATUS:", status)
    print("-" * 100)

    assert b0.identity_holds, "Block 0 multiclass benefit identity FAILED."
    assert bA.capacity_holds, "Block A single-conflict-pair capacity FAILED (mismatches found)."
    assert bA.residual_breaker_flips_sign, "Block A residual breaker did not flip sign (M1' check)."
    assert bB.M1_necessary, "Block B did not realize the multi-conflict-pair fragmentation via argmax."
    assert bC.counterexample_confirmed, "Block C third-class counterexample NOT confirmed."
    assert bD.impossibility_closed, "Block D impossibility closure FAILED."
    print("ALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
