#!/usr/bin/env python3
r"""
val_thm2_lecam_finite_n.py
==========================

Numerical validation of the *finite-sample minimax regret lower bound* that
upgrades K-Bound Theorem 2 (``thm:gate``, the plug-in regret identity) from an
exact identity + an asymptotic/plug-in minimax floor to a genuine FINITE-n
two-point Le Cam lower bound on the EXPECTED REGRET of any label-free gate.

This is the script behind Proposition~\ref{prop:lecam-finite} /
Corollary~\ref{cor:lecam-regret-floor} in paper/sections/main_theory_5.tex.

----------------------------------------------------------------------------
The statement under test
----------------------------------------------------------------------------
Two target worlds theta in {-1, +1} share the same disagreement structure
(f0(x)=1[x>0], fa(x)=1[x<0], so D = {x != 0} has probability 1) but differ in:

  * the covariate (hence label-free evidence) law:  X ~ N(theta * d_n, 1),
  * the target label rule, chosen so the benefit Delta has magnitude Lambda=1
    and sign exactly theta:
        theta = +1  ->  Y = 1[x<0] = fa  ->  Delta = R(f0) - R(fa) = +1,
        theta = -1  ->  Y = 1[x>0] = f0  ->  Delta = R(f0) - R(fa) = -1.

(|Delta| = 1 exactly in BOTH worlds and for every d_n, because each label rule
makes one of f0, fa exactly correct on a probability-1 event; see
`world_delta_closed_form`.)

A *label-free* gate is any  ghat : (Z_1,...,Z_n) -> {adapt(=1), freeze(=0)}
where Z_i = X_i are the unlabeled observations (no Y is ever seen). The Bayes
gate is the constant correct action in each world: g* = adapt under theta=+1,
g* = freeze under theta=-1.

By the EXACT regret identity of Theorem 2 (thm:gate), since |Delta| == Lambda
is constant and g* is the constant correct action,

    R_{P_theta}(ghat) - R_{P_theta}(g*)
        = E[ |Delta| * 1{sign ghat != sign Delta} ]
        = Lambda * Q_theta^{(n)}( ghat commits to the WRONG sign ),

where Q_theta^{(n)} is the law of the n unlabeled observations under world theta.
Averaging over the uniform prior on theta and applying Le Cam's two-point
testing-affinity identity (the same one used for Theorem 1, ``thm:imp``(ii)):

    (1/2) sum_theta [ R_{P_theta}(ghat) - R_{P_theta}(g*) ]
        = (Lambda/2) * M(ghat)
        >= (Lambda/2) * ( 1 - TV( Q_{-1}^{(n)}, Q_{+1}^{(n)} ) ),

where M(ghat) = Q_{+1}(ghat=freeze) + Q_{-1}(ghat=adapt) is the mixed committal
error. Therefore the WORST-CASE (minimax) regret obeys, for EVERY label-free gate
and EVERY finite n,

    ┌──────────────────────────────────────────────────────────────────────┐
    │  inf_ghat  max_theta [ R_{P_theta}(ghat) - R_{P_theta}(g*) ]          │
    │            >=  (Lambda / 2) * ( 1 - TV( Q_{-1}^{(n)}, Q_{+1}^{(n)} ) ) │
    └──────────────────────────────────────────────────────────────────────┘

This is a finite-n statement in REGRET units (Theorem 1's bound is on the error
PROBABILITY M; here it is multiplied through by the |Delta|-weight of Theorem 2).

Two exact, closed-form facts make the family a clean Le Cam construction:

  (1) Sufficiency / exact TV.  The n-fold evidence sufficient statistic is the
      sample mean Xbar ~ N(theta*d_n, 1/n), so
          TV( Q_{-1}^{(n)}, Q_{+1}^{(n)} ) = 2*Phi( sqrt(n) * d_n ) - 1.
      Choosing  d_n = c / sqrt(n)  pins  sqrt(n)*d_n = c, so the TV (and the
      floor) are CONSTANT in n: the worlds stay exactly equally hard at every
      sample size. The floor is then  (Lambda/2)(1 - (2Phi(c)-1)) = Lambda*Phi(-c).

  (2) Bretagnolle-Huber certificate.  With per-observation
      KL(N(+d_n,1) || N(-d_n,1)) = 2 d_n^2, the n-fold KL is 2 n d_n^2 = 2 c^2
      (constant), and  1 - TV >= (1/2) exp(-2 c^2), giving a closed-form,
      n-independent NON-ZERO regret floor  >= (Lambda/4) exp(-2 c^2)  for all n.

----------------------------------------------------------------------------
What this script confirms (all numbers reproducible, fixed seeds)
----------------------------------------------------------------------------
  (a) Builds the two-point family for several c and several n.
  (b) Verifies |Delta| = 1 exactly in both worlds (closed form AND Monte Carlo).
  (c) Verifies the analytic floor (Lambda/2)(1-TV) matches the empirical mixed
      committal error of the OPTIMAL (Bayes) label-free rule -- i.e. the bound
      is TIGHT and the optimal rule MEETS it (so no rule can beat it).
  (d) BRUTE-FORCE search over a broad family of label-free committal rules
      (all sign-threshold rules, both polarities, on a panel of label-free
      statistics: sample mean, trimmed mean, median, fraction-positive, ...):
      confirms the realized minimax regret of EVERY candidate is >= the floor
      (up to Monte-Carlo error), so the lower bound is not beaten empirically.
  (e) Verifies the Bretagnolle-Huber certificate (Lambda/4)exp(-2c^2) lower-
      bounds the floor, and that the floor is constant in n.

Run:
    python val_thm2_lecam_finite_n.py
    python val_thm2_lecam_finite_n.py --json results_thm2_lecam_finite_n.json

Pure numpy + scipy (already in the repo env). No labels, no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
#  Closed-form facts of the two-point family                                  #
# --------------------------------------------------------------------------- #
def world_delta_closed_form(theta: int) -> float:
    r"""Exact benefit Delta = R_T(f0) - R_T(fa) for world theta, 0/1 loss.

    f0(x)=1[x>0], fa(x)=1[x<0].  X ~ N(theta*d, 1) (any d).
      theta = +1 : Y = 1[x<0] = fa  => R(fa)=0, R(f0)=P(x!=0)=1 => Delta = +1.
      theta = -1 : Y = 1[x>0] = f0  => R(f0)=0, R(fa)=1          => Delta = -1.
    Independent of d because both events have probability 1 under any N(mu,1).
    So sign(Delta)=theta and |Delta| = Lambda = 1 EXACTLY in both worlds.
    """
    return 1.0 if theta > 0 else -1.0


def tv_nfold_exact(c: float) -> float:
    r"""TV( N(+d,1)^{⊗n}, N(-d,1)^{⊗n} ) with d = c/sqrt(n).

    Sufficient statistic Xbar ~ N(+-d, 1/n); TV of two Gaussians with equal
    variance sig^2 and means m1<m2 is 2*Phi((m2-m1)/(2 sig)) - 1. Here
    (m2-m1)=2d, sig=1/sqrt(n), so the argument is d*sqrt(n) = c. CONSTANT in n.
    """
    return float(2.0 * norm.cdf(c) - 1.0)


def kl_nfold_exact(c: float) -> float:
    r"""KL( N(+d,1)^{⊗n} || N(-d,1)^{⊗n} ) = n * (2d)^2/2 = 2 n d^2 = 2 c^2."""
    return float(2.0 * c * c)


def bh_lower_bound_on_one_minus_tv(c: float) -> float:
    r"""Bretagnolle-Huber:  1 - TV(P,Q) >= (1/2) exp(-KL(P||Q))."""
    return float(0.5 * math.exp(-kl_nfold_exact(c)))


def analytic_regret_floor(c: float, Lambda: float = 1.0) -> float:
    r"""(Lambda/2)*(1 - TV) = Lambda * Phi(-c)  (the minimax regret floor)."""
    return float(0.5 * Lambda * (1.0 - tv_nfold_exact(c)))


# --------------------------------------------------------------------------- #
#  Sampling the two-point family at finite n                                   #
# --------------------------------------------------------------------------- #
def _stats_block(X: np.ndarray) -> dict[str, np.ndarray]:
    """Label-free statistics for a single (block, n) array of observations."""
    n = X.shape[1]
    xbar = X.mean(axis=1)                          # sufficient statistic (optimal)
    med = np.median(X, axis=1)
    frac_pos = (X > 0).mean(axis=1)                # P(predict f0 class)
    k = max(1, int(0.1 * n))                        # 10% trimmed mean
    if n - 2 * k > 0:
        Xs = np.sort(X, axis=1)
        trimmed = Xs[:, k:n - k].mean(axis=1)
    else:
        trimmed = xbar
    conf = np.tanh(X).mean(axis=1)                  # bounded monotone "confidence"
    return {
        "xbar": xbar,
        "median": med,
        "frac_pos": frac_pos - 0.5,   # center so 0 is the natural threshold
        "trimmed_mean": trimmed,
        "tanh_mean": conf,
    }


def sample_world_statistics(theta: int, d: float, n: int, n_batches: int,
                            rng: np.random.Generator,
                            chunk: int = 20_000) -> dict[str, np.ndarray]:
    r"""Compute the panel of LABEL-FREE batch statistics for world theta
    (X ~ N(theta*d, 1)), processing the n_batches experiments in memory-bounded
    chunks so peak memory is O(chunk * n) regardless of n_batches.

    Each statistic maps one experiment of n unlabeled observations to a scalar;
    we return an (n_batches,) vector per statistic. Labels are NEVER used here
    (label-free rules cannot see Y); the benefit sign is a property of theta.
    """
    keys = ("xbar", "median", "frac_pos", "trimmed_mean", "tanh_mean")
    acc: dict[str, list] = {k: [] for k in keys}
    done = 0
    while done < n_batches:
        b = min(chunk, n_batches - done)
        X = theta * d + rng.standard_normal((b, n))
        s = _stats_block(X)
        for k in keys:
            acc[k].append(s[k])
        done += b
        del X
    return {k: np.concatenate(acc[k]) for k in keys}


@dataclass
class CellResult:
    c: float
    n: int
    d: float
    Lambda: float
    n_batches: int
    tv_exact: float
    kl_exact: float
    floor_analytic: float            # (Lambda/2)(1-TV)
    bh_floor: float                  # (Lambda/4) exp(-2c^2) <= floor_analytic
    # closed-form / MC checks of |Delta|
    delta_plus_closed: float
    delta_minus_closed: float
    abs_delta_mc_max_err: float      # max |  |Delta_mc| - 1 | over both worlds
    # Bayes (optimal) label-free rule
    bayes_M_empirical: float         # mixed committal error of Xbar-threshold rule
    bayes_regret_empirical: float    # (Lambda/2) * bayes_M_empirical
    bayes_M_analytic: float          # 2*Phi(-c)
    bayes_floor_gap: float           # |bayes_regret_empirical - floor_analytic|
    # brute-force minimax over the rule panel
    best_rule_name: str
    min_minimax_regret_over_panel: float   # smallest worst-case regret any rule got
    floor_respected: bool            # min_minimax_regret >= floor - mc_tol
    mc_tol: float


def run_cell(c: float, n: int, *, Lambda: float = 1.0, n_batches: int = 60_000,
             seed: int = 0) -> CellResult:
    rng = np.random.default_rng(seed)
    d = c / math.sqrt(n)

    tv = tv_nfold_exact(c)
    kl = kl_nfold_exact(c)
    floor = analytic_regret_floor(c, Lambda)
    bh = 0.5 * Lambda * bh_lower_bound_on_one_minus_tv(c)  # (Lambda/2)*(1/2)e^{-KL}

    dpc = world_delta_closed_form(+1)
    dmc = world_delta_closed_form(-1)

    # ---- MC sanity: |Delta| = 1 in both worlds (uses labels ONLY to verify the
    #      construction; the gates below never touch labels) ---------------------
    big = 2_000_000
    abs_err = 0.0
    for theta in (+1, -1):
        x = theta * d + rng.standard_normal(big)
        y = (x < 0).astype(float) if theta > 0 else (x > 0).astype(float)
        f0 = (x > 0).astype(float)
        fa = (x < 0).astype(float)
        delta_mc = float(np.mean(f0 != y) - np.mean(fa != y))
        abs_err = max(abs_err, abs(abs(delta_mc) - 1.0))

    # ---- Sample the two worlds (label-free observations only), chunked --------
    # theta=+1: Delta=+1, correct=adapt.   theta=-1: Delta=-1, correct=freeze.
    stats_p = sample_world_statistics(+1, d, n, n_batches, rng)
    stats_m = sample_world_statistics(-1, d, n, n_batches, rng)

    # ---- Bayes (optimal) label-free rule: adapt iff Xbar > 0 ------------------
    # theta=+1 correct action = adapt; WRONG commit = freeze = {Xbar <= 0}.
    # theta=-1 correct action = freeze; WRONG commit = adapt = {Xbar  > 0}.
    err_p = float(np.mean(stats_p["xbar"] <= 0.0))   # P_{+1}(freeze) = wrong
    err_m = float(np.mean(stats_m["xbar"] > 0.0))    # P_{-1}(adapt)  = wrong
    bayes_M = err_p + err_m
    bayes_regret = 0.5 * Lambda * bayes_M
    bayes_M_analytic = float(2.0 * norm.cdf(-c))

    # Monte-Carlo tolerance. The reported regrets are (Lambda/2)*M or Lambda*P with
    # P a Bernoulli mean over n_batches experiments; SE(P) <= sqrt(1/(4 n_batches)).
    # The minimax regret involves max of two such means, so we allow a generous
    # 6-sigma band on the Lambda*P scale.
    mc_tol = 6.0 * Lambda * math.sqrt(0.25 / n_batches)

    # ---- Brute-force minimax over a broad label-free rule family --------------
    # For each statistic T and polarity s in {+1,-1}, the rule is "adapt iff s*T > tau".
    # The minimax (worst-of-two-worlds) regret is
    #     regret(tau) = max( Lambda*P_{+1}(s*Tp <= tau), Lambda*P_{-1}(s*Tm > tau) ),
    # because theta=+1 errs by freezing ({s*Tp <= tau}) and theta=-1 errs by adapting
    # ({s*Tm > tau}). We minimize over a fine tau grid, VECTORIZED via sorted-array
    # CDFs (searchsorted), and take the min over all (T, s). The Le Cam bound says
    # this empirical minimax minimum is >= floor (no label-free rule beats the floor).
    best_name = ""
    best_minimax = math.inf
    B = n_batches
    for name in stats_p:
        tp = stats_p[name]
        tm = stats_m[name]
        lo = min(tp.min(), tm.min())
        hi = max(tp.max(), tm.max())
        taus = np.linspace(lo, hi, 257)
        for s in (+1.0, -1.0):
            sp = np.sort(s * tp)
            sm = np.sort(s * tm)
            # P_{+1}(s*Tp <= tau) = (# sp <= tau)/B  -> searchsorted 'right'
            cdf_p = np.searchsorted(sp, taus, side="right") / B
            # P_{-1}(s*Tm  > tau) = 1 - (# sm <= tau)/B
            tail_m = 1.0 - np.searchsorted(sm, taus, side="right") / B
            minimax = np.maximum(Lambda * cdf_p, Lambda * tail_m)
            j = int(np.argmin(minimax))
            if minimax[j] < best_minimax:
                best_minimax = float(minimax[j])
                best_name = f"{name}|s={int(s):+d}|tau={taus[j]:.4f}"

    floor_ok = bool(best_minimax >= floor - mc_tol)

    return CellResult(
        c=float(c), n=int(n), d=float(d), Lambda=float(Lambda),
        n_batches=int(n_batches),
        tv_exact=float(tv), kl_exact=float(kl),
        floor_analytic=float(floor), bh_floor=float(bh),
        delta_plus_closed=float(dpc), delta_minus_closed=float(dmc),
        abs_delta_mc_max_err=float(abs_err),
        bayes_M_empirical=float(bayes_M),
        bayes_regret_empirical=float(bayes_regret),
        bayes_M_analytic=float(bayes_M_analytic),
        bayes_floor_gap=float(abs(bayes_regret - floor)),
        best_rule_name=best_name,
        min_minimax_regret_over_panel=float(best_minimax),
        floor_respected=floor_ok,
        mc_tol=float(mc_tol),
    )


@dataclass
class Report:
    description: str
    Lambda: float
    seed: int
    c_values: list
    n_values: list
    n_batches: int
    cells: list = field(default_factory=list)
    floor_constant_in_n: bool = False
    bh_certifies_all: bool = False
    bayes_meets_floor_all: bool = False
    no_rule_beats_floor_all: bool = False
    abs_delta_exact_all: bool = False
    all_ok: bool = False


def validate(c_values=(0.25, 0.5, 1.0, 1.5), n_values=(10, 50, 200, 1000),
             Lambda: float = 1.0, n_batches: int = 60_000, seed: int = 20260619) -> Report:
    cells: list[CellResult] = []
    # Deterministic per-cell seeds derived from the master seed.
    for ci, c in enumerate(c_values):
        for ni, n in enumerate(n_values):
            cell_seed = seed + 1000 * ci + ni
            cells.append(run_cell(c, n, Lambda=Lambda, n_batches=n_batches, seed=cell_seed))

    # (e) floor constant in n: for each c, all n share the same analytic floor.
    floor_const = True
    by_c: dict[float, list[float]] = {}
    for cell in cells:
        by_c.setdefault(cell.c, []).append(cell.floor_analytic)
    for c, floors in by_c.items():
        if max(floors) - min(floors) > 1e-12:
            floor_const = False

    bh_ok = all(cell.bh_floor <= cell.floor_analytic + 1e-12 for cell in cells)
    bayes_ok = all(cell.bayes_floor_gap <= cell.mc_tol for cell in cells)
    norule_ok = all(cell.floor_respected for cell in cells)
    absd_ok = all(cell.abs_delta_mc_max_err < 5e-3 for cell in cells)  # MC of |Delta|

    rep = Report(
        description=("Finite-n two-point Le Cam LOWER BOUND on the expected regret "
                     "of any label-free gate (upgrade of Thm 2 thm:gate). Family: "
                     "X~N(theta*c/sqrt(n),1), Delta=theta (|Delta|=1), evidence-TV "
                     "constant in n. Floor = (Lambda/2)(1-TV) = Lambda*Phi(-c)."),
        Lambda=float(Lambda), seed=int(seed),
        c_values=list(c_values), n_values=list(n_values), n_batches=int(n_batches),
        cells=[asdict(c) for c in cells],
        floor_constant_in_n=bool(floor_const),
        bh_certifies_all=bool(bh_ok),
        bayes_meets_floor_all=bool(bayes_ok),
        no_rule_beats_floor_all=bool(norule_ok),
        abs_delta_exact_all=bool(absd_ok),
    )
    rep.all_ok = bool(floor_const and bh_ok and bayes_ok and norule_ok and absd_ok)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--json", type=str, default="results_thm2_lecam_finite_n.json")
    args = ap.parse_args()

    rep = validate(n_batches=args.n_batches, seed=args.seed)

    print("=" * 100)
    print("K-Bound Theorem 2 -- FINITE-n two-point Le Cam regret LOWER BOUND -- validation")
    print("=" * 100)
    print(rep.description)
    print(f"Lambda = {rep.Lambda}   seed = {rep.seed}   batches/world = {rep.n_batches:,}")
    print()
    print("Family: world theta in {-1,+1};  X ~ N(theta*d_n, 1),  d_n = c/sqrt(n);")
    print("        Delta(theta) = theta  (|Delta| = 1 exactly);  correct action = sign(theta).")
    print("Bound : inf_ghat max_theta [R(ghat)-R(g*)]  >=  (Lambda/2)(1 - TV)  =  Lambda*Phi(-c).")
    print()
    hdr = (f"{'c':>5} | {'n':>5} | {'d_n':>8} | {'TV(n-fold)':>10} | {'floor':>9} | "
           f"{'BH cert':>9} | {'Bayes regret':>12} | {'min over panel':>14} | {'floor ok':>8}")
    print(hdr)
    print("-" * len(hdr))
    for cdict in rep.cells:
        print(f"{cdict['c']:>5.2f} | {cdict['n']:>5d} | {cdict['d']:>8.4f} | "
              f"{cdict['tv_exact']:>10.4f} | {cdict['floor_analytic']:>9.4f} | "
              f"{cdict['bh_floor']:>9.4f} | {cdict['bayes_regret_empirical']:>12.4f} | "
              f"{cdict['min_minimax_regret_over_panel']:>14.4f} | "
              f"{str(cdict['floor_respected']):>8}")
    print()
    print("Interpretation:")
    print("  * TV(n-fold) is CONSTANT across n for each c  => the worlds stay equally")
    print("    hard at every sample size; the regret floor does not vanish with n.")
    print("  * 'Bayes regret' (optimal Xbar-threshold rule) == 'floor' to MC error:")
    print("    the lower bound is TIGHT and is MET by the optimal label-free rule.")
    print("  * 'min over panel' (best of a broad rule family, all thresholds/polarities")
    print("    on 5 label-free statistics) stays >= floor: no rule beats the bound.")
    print("  * 'BH cert' = (Lambda/4)exp(-2c^2) is a closed-form n-free lower bound on")
    print("    the floor (Bretagnolle-Huber), certifying a strictly positive floor.")
    print()
    print(f"floor constant in n (per c)      : {rep.floor_constant_in_n}")
    print(f"Bretagnolle-Huber <= floor (all) : {rep.bh_certifies_all}")
    print(f"Bayes rule MEETS floor (all)     : {rep.bayes_meets_floor_all}")
    print(f"NO panel rule beats floor (all)  : {rep.no_rule_beats_floor_all}")
    print(f"|Delta| = 1 exactly (MC, all)    : {rep.abs_delta_exact_all}")
    print(f"ALL CHECKS PASS                  : {rep.all_ok}")
    print()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.json if os.path.isabs(args.json) else os.path.join(out_dir, args.json)
    with open(out_path, "w") as f:
        json.dump(asdict(rep), f, indent=2)
    print(f"Wrote machine-readable results to {out_path}")

    # ---- Hard assertions (this is a VALIDATION: fail loudly) ------------------
    assert rep.abs_delta_exact_all, "Construction broken: |Delta| != 1 in some world"
    assert rep.floor_constant_in_n, "Floor not constant in n -- d_n = c/sqrt(n) scaling broken"
    assert rep.bh_certifies_all, "Bretagnolle-Huber bound exceeded the analytic floor"
    assert rep.bayes_meets_floor_all, (
        "Bayes (optimal) label-free rule did not meet the floor within MC tolerance")
    assert rep.no_rule_beats_floor_all, (
        "A label-free rule BEAT the Le Cam regret floor -- bound or construction is WRONG")
    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
