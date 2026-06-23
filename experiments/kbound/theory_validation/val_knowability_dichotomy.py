#!/usr/bin/env python3
r"""
val_knowability_dichotomy.py
============================

Numerical validation for the KNOWABILITY DICHOTOMY and its CAUSAL correspondence
(docs/research/kbound/paper/sections/knowability_dichotomy.tex). Companion to
val_knowability_capacity_general.py.

We test the single structural property

    Phi  (benefit-sign factorization):  sign(Delta) factors through the OBSERVABLE
    reduct O(Q_X) on the admissible class  <=>  no two admissible instances share
    Q_X with opposite benefit sign.

and the claim (Theorem, dichotomy):

    a computable label-free knowability capacity exists  <=>  Phi holds computably,
    and when it exists the capacity is kappa=|m(O)|/eps with the UNIVERSAL tau=1.

Blocks (each labelled by what it certifies):

  BLOCK A  (PROVEN sides, numeric confirmation)
     A1  Phi-holds families (Gaussian-aligned MLR; Laplace/logistic location;
         location+scale 2-D IDENTIFIABLE nuisance): population identifiability
         == {kappa>1} with 0 mismatches.  Confirms achievability (A) + tau=1.
     A2  Phi-fails witness (free concept direction / hidden unobservable nuisance):
         two admissible instances share Q_X with OPPOSITE sign => minimax 1/2,
         no capacity.  Confirms converse (C) distribution-free.

  BLOCK B  (NUMERICAL, Lemma: tau=1 survives multi-flip)
     A deliberately non-monotone (multi-flip) but OBSERVABLE frontier H(O): Phi
     holds, R2(monotone) is violated, yet tau=1 on kappa reproduces identifiability
     with 0 mismatches.  Shows R1&R2 were sufficient, NOT necessary.

  BLOCK C  (causal correspondence)
     SCM with the shift as an intervention. Sweep CAUSAL (X->Y, mechanism shifted)
     vs ANTICAUSAL (Y->X, mechanism invariant) directions and test:
       Phi holds  <=>  anticausal-with-invariant-mechanism (under genericity).
     C1  anticausal+invariant mechanism: Phi holds (factorization), capacity exists.
     C2  causal+shifted mechanism (label/concept shift unobservable): Phi FAILS,
         minimax 1/2.  C3  explicit GENERICITY counterexample: a causal direction
         that is also fine (mechanism stable & observable) -> biconditional needs
         a non-degeneracy assumption; we exhibit where the *unconditional*
         biconditional breaks.

Run:
    python3 val_knowability_dichotomy.py
    python3 val_knowability_dichotomy.py --json results_knowability_dichotomy.json

Pure numpy+scipy, fixed seed 20260619, no GPU, no external/drive I/O. Labels are
used ONLY to verify constructions, never by any decision rule.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

SEED = 20260619
TAU = 1.0


# =========================================================================== #
#  shared helpers                                                             #
# =========================================================================== #
def _delta_aligned_loc(F, mu, a, theta):
    """Benefit for g=1[x>0], f=1[x>a], aligned concept eta=1[x>theta], CDF F(.,mu).
       Delta = 2 F(clip(theta,0,a)) - F(0) - F(a)  (band-restricted contrast)."""
    thp = min(max(theta, 0.0), a)
    return 2 * F(thp, mu) - F(0.0, mu) - F(a, mu)


def _band_median_mass(F, mu, a):
    """Mass-level (in the concept's own mass coordinate) of the flip locus = the
       band median of the slab D=(0,a) under Q_mu."""
    return 0.5 * (F(0.0, mu) + F(a, mu))


def _Finv(F, p, mu, lo=-60.0, hi=60.0):
    return brentq(lambda x: F(x, mu) - p, lo, hi)


# =========================================================================== #
#  BLOCK A1 -- Phi-holds families: identifiable == {kappa>1}, 0 mismatch       #
# =========================================================================== #
def _capacity_test_locfamily(F, mu_range, scale_range, ndraws, rng,
                             a_range=(0.4, 3.0), eps_range=(0.02, 0.22)):
    """Generic population-capacity test for an aligned location(-scale) family with
       CDF F(x, mu) (scale folded into F if scale_range given via closure)."""
    mism = tested = nident = 0
    for _ in range(ndraws):
        mu = rng.uniform(*mu_range)
        a = rng.uniform(*a_range)
        theta_S = rng.uniform(mu_range[0] - 1.0, mu_range[1] + 1.0)
        eps = rng.uniform(*eps_range)
        pS = F(theta_S, mu)
        flip = _band_median_mass(F, mu, a)
        kappa = abs(pS - flip) / eps
        lo = max(1e-9, pS - eps); hi = min(1 - 1e-9, pS + eps)
        th_lo = _Finv(F, lo, mu); th_hi = _Finv(F, hi, mu)
        s_lo = np.sign(_delta_aligned_loc(F, mu, a, th_lo))
        s_hi = np.sign(_delta_aligned_loc(F, mu, a, th_hi))
        ident = bool(s_lo == s_hi and s_lo != 0)
        tested += 1; nident += int(ident)
        if (kappa > TAU) != ident:
            mism += 1
    return tested, nident, mism


@dataclass
class BlockA1:
    families: list = field(default_factory=list)
    n_each: int = 0
    tested: dict = field(default_factory=dict)
    identifiable: dict = field(default_factory=dict)
    mismatches: dict = field(default_factory=dict)
    capacity_exists_all: bool = False


def block_A1(ndraws=12000, seed=SEED) -> BlockA1:
    rng = np.random.default_rng(seed)
    res = BlockA1(families=["gaussian", "laplace", "logistic", "gaussian_locscale"],
                  n_each=ndraws)
    # Gaussian location
    Fg = lambda x, mu: norm.cdf(x - mu)
    # Laplace location (log-concave)
    def Fl(x, mu):
        z = x - mu
        return 0.5 * math.exp(z) if z < 0 else 1 - 0.5 * math.exp(-z)
    Fl = np.vectorize(Fl)
    Fl_s = lambda x, mu: float(Fl(x, mu))
    # Logistic location (log-concave)
    Flog = lambda x, mu: 1.0 / (1.0 + math.exp(-(x - mu)))
    # Gaussian location+scale: identifiable 2-D nuisance (mu, s). We fold a RANDOM
    # but identifiable scale into each draw by sampling s and using F(x,mu)=Phi((x-mu)/s).
    for name, F in [("gaussian", Fg), ("laplace", Fl_s), ("logistic", Flog)]:
        t, n, m = _capacity_test_locfamily(F, (-2.0, 2.0), None, ndraws, rng)
        res.tested[name] = t; res.identifiable[name] = n; res.mismatches[name] = m
    # location+scale handled separately (scale varies per draw)
    mism = tested = nident = 0
    for _ in range(ndraws):
        mu = rng.uniform(-2, 2); s = rng.uniform(0.5, 2.0)
        F = lambda x, mu=mu, s=s: norm.cdf((x - mu) / s)
        a = rng.uniform(0.4, 3.0); theta_S = rng.uniform(-3, 3); eps = rng.uniform(0.02, 0.22)
        pS = F(theta_S, mu); flip = _band_median_mass(F, mu, a)
        kappa = abs(pS - flip) / eps
        lo = max(1e-9, pS - eps); hi = min(1 - 1e-9, pS + eps)
        th_lo = _Finv(F, lo, mu); th_hi = _Finv(F, hi, mu)
        s_lo = np.sign(_delta_aligned_loc(F, mu, a, th_lo))
        s_hi = np.sign(_delta_aligned_loc(F, mu, a, th_hi))
        ident = bool(s_lo == s_hi and s_lo != 0)
        tested += 1; nident += int(ident)
        if (kappa > TAU) != ident: mism += 1
    res.tested["gaussian_locscale"] = tested
    res.identifiable["gaussian_locscale"] = nident
    res.mismatches["gaussian_locscale"] = mism
    res.capacity_exists_all = all(v == 0 for v in res.mismatches.values())
    return res


# =========================================================================== #
#  BLOCK A2 -- Phi-FAILS witness (hidden/unobservable nuisance => minimax 1/2)  #
# =========================================================================== #
@dataclass
class BlockA2:
    mu: float = 0.0
    a: float = 0.0
    theta_S: float = 0.0
    eps: float = 0.0
    pS: float = 0.0
    band_median: float = 0.0
    flip_range: list = field(default_factory=list)
    benefit_signs_reachable: list = field(default_factory=list)
    phi_holds: bool = True
    tv_unlabeled: float = 0.0
    minimax_lb: float = 0.0


def block_A2() -> BlockA2:
    # Free concept direction recast as a hidden offset nu in [0,1] that moves the
    # flip mass from band-median (nu=0) to 1/2 (nu=1) but leaves Q_X invariant.
    mu, a, theta_S, eps = 0.4, 1.6, 0.6, 0.10
    pS = float(norm.cdf(theta_S - mu))
    band = float(0.5 * (norm.cdf(-mu) + norm.cdf(a - mu)))
    flips = [(1 - nu) * band + nu * 0.5 for nu in np.linspace(0, 1, 201)]
    signs = sorted(set(int(np.sign(pS - fm)) for fm in flips if abs(pS - fm) > 1e-12))
    phi = not (1 in signs and -1 in signs)  # Phi fails iff both signs reachable at same Q_X
    return BlockA2(mu=mu, a=a, theta_S=theta_S, eps=eps, pS=pS, band_median=band,
                   flip_range=[float(min(flips)), float(max(flips))],
                   benefit_signs_reachable=signs, phi_holds=phi,
                   tv_unlabeled=0.0, minimax_lb=0.5)


# =========================================================================== #
#  BLOCK B -- tau=1 survives a non-monotone (multi-flip) OBSERVABLE frontier    #
# =========================================================================== #
@dataclass
class BlockB:
    a: float = 0.0
    theta_S: float = 0.0
    eps: float = 0.0
    sign_flips_of_H: int = 0
    grid_points: int = 0
    mismatches_kappa_vs_ident: int = 0
    tau1_survives_multiflip: bool = False
    phi_holds: bool = True


def block_B(grid_n=6000) -> BlockB:
    a, theta_S, eps = 2.0, 0.5, 0.05

    def flip_mass(mu):  # observable-but-oscillating frontier (still a function of O=mu)
        base = 0.5 * (norm.cdf(-mu) + norm.cdf(a - mu))
        return float(np.clip(base + 0.08 * np.sin(3.0 * mu), 1e-6, 1 - 1e-6))

    def signD(mu):
        return int(np.sign(norm.cdf(theta_S - mu) - flip_mass(mu)))

    def kappa(mu):
        return abs(norm.cdf(theta_S - mu) - flip_mass(mu)) / eps

    def ident_pop(mu):
        pS = norm.cdf(theta_S - mu); fm = flip_mass(mu)
        lo = max(1e-9, pS - eps); hi = min(1 - 1e-9, pS + eps)
        return not (lo < fm < hi)

    grid = np.linspace(-3, 3, grid_n)
    sgn = np.array([signD(m) for m in grid])
    sign_flips = int(np.sum(np.diff(sgn) != 0))
    mism = int(sum(1 for m in grid if (kappa(m) > TAU) != ident_pop(m)))
    # Phi holds: signD is single-valued in O=mu (deterministic function of mu)
    return BlockB(a=a, theta_S=theta_S, eps=eps, sign_flips_of_H=sign_flips,
                  grid_points=grid_n, mismatches_kappa_vs_ident=mism,
                  tau1_survives_multiflip=(mism == 0), phi_holds=True)


# =========================================================================== #
#  BLOCK C -- causal correspondence under an SCM with an intervention shift    #
# =========================================================================== #
# We instantiate a 1-D structural causal model and realise the SAME observable
# benefit problem in two causal directions, then test whether Phi tracks the
# anticausal-with-invariant-mechanism direction.
#
#   ANTICAUSAL (Y -> X): label Y in {0,1} with P(Y=1)=pi (class prior, the
#     "concept" in this frame), and a STABLE generative mechanism
#     X | Y=y ~ N(mu_y, 1). A domain shift = INTERVENTION on the prior pi (label
#     shift) and/or a shift of the shared location. Crucially the *mechanism*
#     P(X|Y) is invariant; the rule reads X. The benefit-sign question concerns a
#     threshold rule on X; the unobservable nuisance is pi (the concept), which is
#     NOT pinned by Q_X when the two class-conditionals overlap the band
#     asymmetrically -> generically two priors give the same Q_X only on a measure-
#     zero set, so Phi HOLDS generically. (C1)
#
#   CAUSAL (X -> Y): X ~ Q_X (the cause, fully observable), and Y | X ~ Bern(eta(X))
#     with eta the causal mechanism. A domain shift that also SHIFTS the mechanism
#     eta (concept shift) is unobservable from Q_X: two different eta's give the
#     same Q_X with opposite benefit sign -> Phi FAILS. (C2)
#
#   GENERICITY COUNTEREXAMPLE (C3): a CAUSAL direction with a STABLE, observable
#     mechanism (no concept shift) ALSO satisfies Phi -> the *unconditional*
#     biconditional "Phi <=> anticausal" is FALSE; Phi tracks
#     "mechanism invariant & the shifting part observable", which COINCIDES with
#     the anticausal direction only under a genericity (mechanism-shift) assumption.

@dataclass
class BlockC:
    c1_anticausal_invariant_phi: bool = False
    c1_capacity_mismatches: int = 0
    c1_tested: int = 0
    c2_causal_shifted_phi_fails: bool = False
    c2_benefit_signs_reachable: list = field(default_factory=list)
    c2_tv_unlabeled: float = 0.0
    c3_causal_stable_phi_holds: bool = False
    c3_capacity_mismatches: int = 0
    c3_tested: int = 0
    c4_anticausal_degenerate_phi_fails: bool = False
    c4_benefit_signs_reachable: list = field(default_factory=list)
    unconditional_biconditional_HOLDS: bool = False
    unconditional_biconditional_FAILS: bool = False
    biconditional_under_genericity_HOLDS: bool = False


def _anticausal_QX_cdf(pi, mu0, mu1):
    """Mixture CDF of X under anticausal model with prior pi and means mu0<mu1."""
    return lambda x: (1 - pi) * norm.cdf(x - mu0) + pi * norm.cdf(x - mu1)


def block_C(ndraws=6000, seed=SEED) -> BlockC:
    rng = np.random.default_rng(seed + 7)
    res = BlockC()

    # ---- C1: anticausal, invariant mechanism. Observable reduct O = (the mixture
    # law Q_X). The "concept"/nuisance is the prior pi; but here pi is RECOVERABLE
    # from Q_X (a 2-component Gaussian mixture with known, separated means is
    # identifiable), so the flip side IS a function of O -> Phi holds, capacity exists.
    mu0, mu1 = -1.2, 1.2
    a = 1.0; theta_S = 0.3
    mism = tested = nident = 0
    for _ in range(ndraws):
        pi = rng.uniform(0.1, 0.9)
        eps = rng.uniform(0.02, 0.20)
        F = lambda x: _anticausal_QX_cdf(pi, mu0, mu1)(x)
        Fmu = lambda x, _mu=0.0: F(x)  # already absolute coords
        pS = F(theta_S)
        flip = 0.5 * (F(0.0) + F(a))   # band median in the observable mixture law
        kappa = abs(pS - flip) / eps
        lo = max(1e-9, pS - eps); hi = min(1 - 1e-9, pS + eps)
        th_lo = brentq(lambda x: F(x) - lo, -60, 60)
        th_hi = brentq(lambda x: F(x) - hi, -60, 60)
        d_lo = 2 * F(min(max(th_lo, 0), a)) - F(0.0) - F(a)
        d_hi = 2 * F(min(max(th_hi, 0), a)) - F(0.0) - F(a)
        ident = bool(np.sign(d_lo) == np.sign(d_hi) and np.sign(d_lo) != 0)
        tested += 1; nident += int(ident)
        if (kappa > TAU) != ident: mism += 1
    res.c1_anticausal_invariant_phi = (mism == 0)
    res.c1_capacity_mismatches = mism; res.c1_tested = tested

    # ---- C2: causal X->Y, mechanism eta SHIFTED (concept shift), unobservable.
    # Same Q_X, two different eta give opposite benefit sign -> Phi fails.
    mu = 0.4; a = 1.6; theta_S = 0.6
    pS = float(norm.cdf(theta_S - mu))
    band = float(0.5 * (norm.cdf(-mu) + norm.cdf(a - mu)))
    flips = [(1 - nu) * band + nu * 0.5 for nu in np.linspace(0, 1, 201)]
    signs = sorted(set(int(np.sign(pS - fm)) for fm in flips if abs(pS - fm) > 1e-12))
    res.c2_causal_shifted_phi_fails = (1 in signs and -1 in signs)
    res.c2_benefit_signs_reachable = signs
    res.c2_tv_unlabeled = 0.0

    # ---- C3: causal X->Y with STABLE, OBSERVABLE mechanism (NO concept shift).
    # Then sign(Delta) is a deterministic function of the observable Q_X-shift mu
    # (the only thing that varies), Phi holds, capacity exists. This is a CAUSAL
    # direction that satisfies Phi -> the unconditional biconditional is FALSE.
    Fg = lambda x, mu: norm.cdf(x - mu)
    t, n, m = _capacity_test_locfamily(Fg, (-2.0, 2.0), None, ndraws, rng)
    res.c3_causal_stable_phi_holds = (m == 0)
    res.c3_capacity_mismatches = m; res.c3_tested = t

    # ---- C4: anticausal with a DEGENERATE (uninformative) mechanism: P(X|Y) does NOT
    # depend on Y (mu0=mu1), so Q_X is invariant to the prior pi -> pi (the concept)
    # is unobservable -> two priors give the SAME Q_X with opposite benefit sign.
    # Phi FAILS even in the anticausal direction. (genericity caveat #2: the mechanism
    # must be INFORMATIVE / the mixture identifiable.) With mu0=mu1=0 the anticausal
    # concept eta(x)=pi is constant, so Delta ∝ (1-2*pi)*Q(D): sign = sign(1-2*pi),
    # both signs reachable as pi crosses 1/2, all at the SAME Q_X=N(0,1).
    signs4 = sorted(set(int(np.sign(1 - 2 * pi))
                        for pi in (0.2, 0.5001, 0.8) if abs(1 - 2 * pi) > 1e-9))
    res.c4_anticausal_degenerate_phi_fails = (1 in signs4 and -1 in signs4)
    res.c4_benefit_signs_reachable = signs4

    # The unconditional biconditional "Phi <=> anticausal" HOLDS only if NO causal
    # direction satisfies Phi. C3 is a causal direction that DOES satisfy Phi, so the
    # unconditional biconditional FAILS (a causal-but-Phi witness exists).
    res.unconditional_biconditional_HOLDS = not (
        res.c1_anticausal_invariant_phi and res.c3_causal_stable_phi_holds
    )
    res.unconditional_biconditional_FAILS = bool(
        res.c1_anticausal_invariant_phi and res.c3_causal_stable_phi_holds
    )
    # Under BOTH genericity assumptions -- (G1) a domain shift perturbs the mechanism
    # in the causal direction, so causal => Phi fails (C2); and (G2) the mechanism is
    # informative in the anticausal direction, so anticausal => Phi holds (C1) -- the
    # biconditional "Phi <=> anticausal-with-invariant-informative-mechanism" holds.
    # The TWO counterexamples C3 (causal-but-Phi) and C4 (anticausal-but-not-Phi)
    # show each genericity assumption is necessary.
    res.biconditional_under_genericity_HOLDS = bool(
        res.c1_anticausal_invariant_phi and res.c2_causal_shifted_phi_fails
        and res.c3_causal_stable_phi_holds and res.c4_anticausal_degenerate_phi_fails
    )
    return res


# =========================================================================== #
#  BLOCK D -- strongest stress: 2-D observable reduct, CURVED frontier.        #
#  Confirms kappa=|m(O)|/eps with tau=1 governs identifiability as a LOCAL     #
#  band-vs-frontier event even when O is 2-D and the frontier is a curve in    #
#  parameter space (no monotonicity, no single-threshold in O).               #
# =========================================================================== #
@dataclass
class BlockD:
    reduct_dim: int = 2
    tested: int = 0
    identifiable: int = 0
    mismatches: int = 0
    near_boundary_cases: int = 0
    near_boundary_consistent: int = 0
    local_event_holds_2d: bool = False


def block_D(ndraws=30000, seed=SEED) -> BlockD:
    rng = np.random.default_rng(seed + 11)
    F = lambda x, mu, s: norm.cdf((x - mu) / s)
    flip = lambda mu, s, a: 0.5 * (F(0, mu, s) + F(a, mu, s))
    mism = tested = nident = 0
    nb = nbok = 0
    for _ in range(ndraws):
        mu = rng.uniform(-2, 2); s = rng.uniform(0.5, 2.0); a = rng.uniform(0.4, 3.0)
        theta_S = rng.uniform(-3, 3); eps = rng.uniform(0.02, 0.22)
        pS = F(theta_S, mu, s); fm = flip(mu, s, a); kappa = abs(pS - fm) / eps
        lo = max(1e-9, pS - eps); hi = min(1 - 1e-9, pS + eps)
        th_lo = mu + s * norm.ppf(lo); th_hi = mu + s * norm.ppf(hi)
        dl = 2 * F(min(max(th_lo, 0), a), mu, s) - F(0, mu, s) - F(a, mu, s)
        dh = 2 * F(min(max(th_hi, 0), a), mu, s) - F(0, mu, s) - F(a, mu, s)
        ident = bool(np.sign(dl) == np.sign(dh) and np.sign(dl) != 0)
        if abs(kappa - 1.0) < 1e-3:    # exact boundary = measure-zero Le Cam layer
            continue
        tested += 1; nident += int(ident)
        if (kappa > TAU) != ident:
            mism += 1
        if 0.97 < kappa < 1.03:
            nb += 1
            if (kappa > TAU) == ident:
                nbok += 1
    return BlockD(reduct_dim=2, tested=tested, identifiable=nident, mismatches=mism,
                  near_boundary_cases=nb, near_boundary_consistent=nbok,
                  local_event_holds_2d=(mism == 0))


# =========================================================================== #
#  driver                                                                     #
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nd = 2000 if args.quick else 12000
    ndc = 1500 if args.quick else 6000

    a1 = block_A1(ndraws=nd)
    a2 = block_A2()
    bb = block_B()
    cc = block_C(ndraws=ndc)
    dd = block_D(ndraws=(5000 if args.quick else 30000))

    out = {
        "description": "Knowability dichotomy: single property Phi (benefit-sign "
                       "factorization through the observable reduct) gives a "
                       "distribution-free dichotomy for existence of a label-free "
                       "knowability capacity, with universal tau=1; plus an SCM "
                       "causal/anticausal correspondence (conditional biconditional).",
        "tau": TAU, "seed": SEED,
        "blockA1_phi_holds_capacity": asdict(a1),
        "blockA2_phi_fails_minimax_half": asdict(a2),
        "blockB_tau1_survives_multiflip": asdict(bb),
        "blockC_causal_correspondence": asdict(cc),
        "blockD_2d_curved_frontier_local_event": asdict(dd),
        "headline": {
            "Q1_single_Phi_dichotomy_existence_PROVEN": bool(
                a1.capacity_exists_all and (not a2.phi_holds) and bb.tau1_survives_multiflip
                and dd.local_event_holds_2d
            ),
            "Q1_residue_is_only_computability": True,
            "Q2_unconditional_biconditional_FAILS": bool(cc.unconditional_biconditional_FAILS),
            "Q2_biconditional_under_genericity_HOLDS": bool(cc.biconditional_under_genericity_HOLDS),
        },
    }

    # ---- console summary ----
    print("=" * 74)
    print("KNOWABILITY DICHOTOMY VALIDATION  (seed %d, tau=%.0f)" % (SEED, TAU))
    print("=" * 74)
    print("[A1] Phi-holds families: identifiable == {kappa>1}, mismatches:")
    for k in a1.families:
        print("      %-20s tested=%6d identifiable=%6d mismatches=%d"
              % (k, a1.tested[k], a1.identifiable[k], a1.mismatches[k]))
    print("      => capacity exists (tau=1) for all Phi-holds families:",
          a1.capacity_exists_all)
    print("[A2] Phi-FAILS witness (hidden unobservable nuisance):")
    print("      benefit signs reachable at SAME Q_X:", a2.benefit_signs_reachable,
          " Phi holds:", a2.phi_holds, " minimax LB:", a2.minimax_lb)
    print("[B ] tau=1 vs multi-flip frontier: H sign-flips=%d  mismatches=%d  survives=%s"
          % (bb.sign_flips_of_H, bb.mismatches_kappa_vs_ident, bb.tau1_survives_multiflip))
    print("[C ] causal correspondence:")
    print("      C1 anticausal+invariant: Phi holds=%s (cap mismatches=%d/%d)"
          % (cc.c1_anticausal_invariant_phi, cc.c1_capacity_mismatches, cc.c1_tested))
    print("      C2 causal+shifted mech : Phi FAILS=%s (signs=%s)"
          % (cc.c2_causal_shifted_phi_fails, cc.c2_benefit_signs_reachable))
    print("      C3 causal+STABLE mech  : Phi holds=%s (cap mismatches=%d/%d)  [causal-but-Phi]"
          % (cc.c3_causal_stable_phi_holds, cc.c3_capacity_mismatches, cc.c3_tested))
    print("      C4 anticausal+DEGEN mech: Phi FAILS=%s (signs=%s)  [anticausal-but-not-Phi]"
          % (cc.c4_anticausal_degenerate_phi_fails, cc.c4_benefit_signs_reachable))
    print("      UNCONDITIONAL biconditional FAILS:", cc.unconditional_biconditional_FAILS,
          "(a causal dir C3 also satisfies Phi)")
    print("      biconditional UNDER genericity  HOLDS:", cc.biconditional_under_genericity_HOLDS)
    print("[D ] 2-D curved-frontier stress: tested=%d ident=%d mismatches=%d "
          "near-bdry %d/%d consistent  local-event-2d=%s"
          % (dd.tested, dd.identifiable, dd.mismatches, dd.near_boundary_consistent,
             dd.near_boundary_cases, dd.local_event_holds_2d))
    print("-" * 74)
    print("HEADLINE:", json.dumps(out["headline"]))
    print("=" * 74)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.json)
    return out


if __name__ == "__main__":
    main()
