#!/usr/bin/env python3
r"""
val_conj1_genpos.py
===================

HONEST OBSTRUCTION WITNESS for the general-position assumption in the conditional
resolution of Conjecture 1 (weakest one-bit class), i.e. Assumption~\ref{asm:genpos}
in paper/sections/weakest_class.tex and the minimality clause
Theorem~\ref{thm:cmono-weakest}(iii).

WHAT IS OPEN, AND WHY
---------------------
Theorem~\ref{thm:cmono-weakest} parts (i)-(ii) are UNCONDITIONAL: one declared
orientation bit certifies sign(Delta) on the margin-monotone class
C_mono = { a = sigma * c  mu-a.e. on D }.  Part (iii) -- that C_mono is a WEAKEST
falsifiable one-bit class (any strictly larger falsifiable class needs >= 2 bits) --
is proved only UNDER the general-position Assumption.

This script verifies that the assumption is GENUINELY NEEDED: it exhibits an explicit
falsifiable class  C_dom  that is

    (1) STRICTLY LARGER than C_mono (contains non-margin-monotone targets),
    (2) still FALSIFIABLE by a finite labelled probe, and
    (3) certified by ONE declared bit,

which DIRECTLY CONTRADICTS minimality once general position is dropped. Hence
general position cannot simply be removed -- the unconditional weakest class
remains open, and the obstruction is exactly an E-certifiable *dominant region*.

THE EXPLICIT CONSTRUCTION
-------------------------
Disagreement region  D = R0 ∪ R1  with mu(R0) = mu(R1) = 1/2. The observable
relative-calibration field c is

    c(x) = +1 on R0   (a "calibrated" region: margin m - m* = +1),
    c(x) =  0 on R1   (the anchor region: NO observable benefit signal).

Benefit field a(x) = 2*eta_a(x) - 1 with Delta = ∫_D a dmu = (1/2)(a(R0) + a(R1)).

  * C_mono forces a = sigma * c, i.e. a(R0) = sigma, a(R1) = 0. (R1 contributes 0.)
  * C_dom RELAXES this on the anchor region while keeping R0 dominant:
        a(R0) = sigma           (pinned to the orientation bit, calibrated)
        a(R1) ∈ [-rho, +rho]    ARBITRARY, with a declared constant rho < 1.
    Because mu(R0) = mu(R1) and |a(R1)| <= rho < 1 = |a(R0)|, R0's benefit mass
    DOMINATES, so
        sign(Delta) = sign( (1/2)(sigma + a(R1)) ) = sigma     for every a(R1),
    and the SINGLE bit sigma certifies sign(Delta) on all of C_dom.

C_dom is strictly larger than C_mono (it contains targets with a(R1) != 0, which are
NOT margin-monotone: on R1 the sign of a is unconstrained while sign(c) = 0), and it
is falsifiable (reject if R0 is not calibrated or if |a(R1)| > rho on a labelled
probe). Thus C_mono is not minimal once the dominant-region case is allowed: general
position is the precise hypothesis that excludes this loophole.

We ALSO confirm the boundary is sharp: drop the domination bound (rho >= 1) and one
bit no longer certifies sign(Delta) -- exactly the K=2 / two-bit regime of
Lemma~\ref{lem:bitcomplexity}. This is the minimal counterexample referenced in
Remark~\ref{rmk:genpos}.

This validator therefore does NOT close the open problem. It MACHINE-CHECKS the
honest claim that general position is necessary, by exhibiting the obstruction.

Run:
    python val_conj1_genpos.py
    python val_conj1_genpos.py --json results_conj1_genpos.json

Pure numpy. No labels used by any "rule" (the bit is declared, not estimated); we
use labels only to MEASURE Delta when verifying the construction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field

import numpy as np


# --------------------------------------------------------------------------- #
#  The two-region evidence fibre                                              #
# --------------------------------------------------------------------------- #
@dataclass
class TwoRegionTarget:
    """A target specified by its per-region benefit (a on R0, a on R1).

    mu(R0)=mu(R1)=1/2. Observable field c = (+1 on R0, 0 on R1). The label-free
    evidence E sees (mu, c) and the predictions, NOT a directly -- a is a property
    of the (hidden) target labels. Two targets sharing (mu, c) are evidence-identical
    on this fibre, so any region where c is uninformative (R1, c=0) is sign-free.
    """
    a_R0: float
    a_R1: float

    def delta(self) -> float:
        return 0.5 * self.a_R0 + 0.5 * self.a_R1

    def is_margin_monotone(self, sigma: float, tol: float = 1e-9) -> bool:
        """C_mono: a = sigma*c everywhere => a_R0 = sigma*(+1), a_R1 = sigma*0 = 0."""
        return abs(self.a_R0 - sigma) <= tol and abs(self.a_R1) <= tol


def one_bit_decode(sigma: float) -> float:
    """The single declared bit certifies sign(Delta) := sigma (no labels read)."""
    return sigma


# --------------------------------------------------------------------------- #
#  Class samplers                                                             #
# --------------------------------------------------------------------------- #
def sample_C_dom(sigma: float, rho: float, rng: np.random.Generator) -> TwoRegionTarget:
    """C_dom: a_R0 = sigma (calibrated, dominant), a_R1 ~ Uniform[-rho, rho], rho<1."""
    return TwoRegionTarget(a_R0=sigma, a_R1=float(rng.uniform(-rho, rho)))


def sample_no_domination(sigma: float, spread: float, rng: np.random.Generator) -> TwoRegionTarget:
    """Relaxation WITHOUT domination (spread >= 1): a_R1 may exceed |a_R0| in mass."""
    return TwoRegionTarget(a_R0=sigma, a_R1=float(rng.uniform(-spread, spread)))


# --------------------------------------------------------------------------- #
#  Checks                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    description: str
    seed: int
    rho: float
    n_trials: int
    # (1) C_dom strictly larger than C_mono
    c_dom_contains_non_mono: bool
    n_non_mono_witnessed: int
    # (2) C_dom falsifiable (a labelled probe rejects out-of-class targets)
    falsifiable_probe_rejects_outside: bool
    probe_reject_rate_outside: float
    probe_accept_rate_inside: float
    # (3) one bit certifies sign(Delta) on C_dom
    one_bit_certifies_on_C_dom: bool
    one_bit_error_rate_C_dom: float
    # sharpness: drop domination => one bit fails (>=2 bits / K=2 regime)
    one_bit_fails_without_domination: bool
    one_bit_error_rate_no_domination: float
    # explicit minimal counterexample (matches Prop. cmono-witness numbers)
    witness_dom_example: dict
    witness_nodom_example: dict
    conclusion: str
    obstruction_confirmed: bool


def labelled_probe_in_class(t: TwoRegionTarget, sigma: float, rho: float,
                            tol: float = 1e-9) -> bool:
    """A finite labelled probe estimates per-region benefits and accepts the target
    as a member of C_dom iff R0 is calibrated to the bit (a_R0 = sigma) AND the
    anchor region respects the domination bound (|a_R1| <= rho). This is the
    falsifiability mechanism: out-of-class targets are rejected.
    """
    return (abs(t.a_R0 - sigma) <= tol) and (abs(t.a_R1) <= rho + tol)


def run(rho: float = 0.6, n_trials: int = 200_000, seed: int = 20260619) -> Report:
    rng = np.random.default_rng(seed)

    # ---- (1) + (3): C_dom is strictly larger AND one-bit-certified -----------
    n_non_mono = 0
    one_bit_errors = 0
    accept_inside = 0
    for _ in range(n_trials):
        sigma = float(rng.choice([-1.0, 1.0]))
        t = sample_C_dom(sigma, rho, rng)
        if not t.is_margin_monotone(sigma):
            n_non_mono += 1                       # strictly outside C_mono
        # one declared bit certifies the sign?
        if np.sign(t.delta()) != one_bit_decode(sigma):
            one_bit_errors += 1
        # probe accepts genuine members of C_dom
        if labelled_probe_in_class(t, sigma, rho):
            accept_inside += 1

    one_bit_err_rate = one_bit_errors / n_trials
    accept_rate_inside = accept_inside / n_trials

    # ---- (2): falsifiability -- probe rejects out-of-class targets -----------
    # Out-of-class draws: violate calibration on R0 OR exceed domination on R1.
    reject_outside = 0
    n_out = 50_000
    for _ in range(n_out):
        sigma = float(rng.choice([-1.0, 1.0]))
        if rng.random() < 0.5:
            # break calibration on R0
            t = TwoRegionTarget(a_R0=sigma * float(rng.uniform(0.0, 0.8)),
                                a_R1=float(rng.uniform(-rho, rho)))
        else:
            # exceed the domination bound on R1
            t = TwoRegionTarget(a_R0=sigma,
                                a_R1=float(rng.uniform(1.0, 2.0)) * float(rng.choice([-1, 1])))
        if not labelled_probe_in_class(t, sigma, rho):
            reject_outside += 1
    reject_rate_outside = reject_outside / n_out

    # ---- sharpness: drop domination (spread=2) => one bit FAILS --------------
    nodom_errors = 0
    for _ in range(n_trials):
        sigma = float(rng.choice([-1.0, 1.0]))
        t = sample_no_domination(sigma, spread=2.0, rng=rng)
        if np.sign(t.delta()) != one_bit_decode(sigma):
            nodom_errors += 1
    nodom_err_rate = nodom_errors / n_trials

    # ---- explicit minimal counterexample numbers (mirror prop:cmono-witness) -
    # Dominant case: a=(+1, +0.5) and (+1, -0.5) -> Delta=+0.75 and +0.25, both +,
    # so bit sigma=+1 certifies; both are OUTSIDE C_mono (a_R1 != 0). One bit works.
    w_dom = {
        "a_plus": [1.0, 0.5], "delta_plus": 0.75,
        "a_minus": [1.0, -0.5], "delta_minus": 0.25,
        "both_same_sign": True,
        "one_bit_sigma": 1.0,
        "note": "R0 dominates (|a_R1|<=0.5<1): sign(Delta)=sigma for BOTH -> 1 bit suffices, "
                "yet both are non-margin-monotone (a_R1 != 0). C_mono not minimal here.",
    }
    # No-domination case: a=(+1,+2) vs (+1,-2) -> Delta=+1.5 vs -0.5 : OPPOSITE signs,
    # so NO single bit certifies (this is the K=2 / two-bit regime).
    w_nodom = {
        "a_plus": [1.0, 2.0], "delta_plus": 1.5,
        "a_minus": [1.0, -2.0], "delta_minus": -0.5,
        "both_same_sign": False,
        "note": "Without domination the anchor region flips Delta's sign -> two bits "
                "necessary (K=2). This is exactly what general position rules out.",
    }

    obstruction = bool(
        n_non_mono > 0 and                       # strictly larger
        reject_rate_outside > 0.99 and           # falsifiable
        accept_rate_inside > 0.99 and            # probe sound on members
        one_bit_err_rate == 0.0 and              # one bit suffices on C_dom
        nodom_err_rate > 0.05                     # and FAILS without domination
    )

    conclusion = (
        "General position is NECESSARY for minimality (iii): C_dom is a strictly larger, "
        "falsifiable, ONE-bit class, contradicting minimality when an E-certifiable dominant "
        "region is allowed. The UNCONDITIONAL weakest class therefore remains OPEN; the "
        "obstruction is exactly the dominant-region loophole excluded by Assumption~genpos."
    )

    return Report(
        description=("Honest obstruction witness for Assumption~genpos / "
                     "Thm cmono-weakest(iii). Confirms general position cannot be removed."),
        seed=int(seed), rho=float(rho), n_trials=int(n_trials),
        c_dom_contains_non_mono=bool(n_non_mono > 0),
        n_non_mono_witnessed=int(n_non_mono),
        falsifiable_probe_rejects_outside=bool(reject_rate_outside > 0.99),
        probe_reject_rate_outside=float(reject_rate_outside),
        probe_accept_rate_inside=float(accept_rate_inside),
        one_bit_certifies_on_C_dom=bool(one_bit_err_rate == 0.0),
        one_bit_error_rate_C_dom=float(one_bit_err_rate),
        one_bit_fails_without_domination=bool(nodom_err_rate > 0.05),
        one_bit_error_rate_no_domination=float(nodom_err_rate),
        witness_dom_example=w_dom,
        witness_nodom_example=w_nodom,
        conclusion=conclusion,
        obstruction_confirmed=obstruction,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--n-trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260619)
    ap.add_argument("--json", type=str, default="results_conj1_genpos.json")
    args = ap.parse_args()

    rep = run(rho=args.rho, n_trials=args.n_trials, seed=args.seed)

    print("=" * 100)
    print("Conjecture 1 -- general-position OBSTRUCTION witness (Thm cmono-weakest(iii))")
    print("=" * 100)
    print(rep.description)
    print(f"seed = {rep.seed}   rho = {rep.rho}   trials = {rep.n_trials:,}")
    print()
    print("Explicit fibre:  D = R0 ∪ R1, mu(R0)=mu(R1)=1/2,  c = (+1 on R0, 0 on R1).")
    print("C_dom:  a_R0 = sigma (calibrated, dominant),  a_R1 ∈ [-rho,rho], rho<1.")
    print()
    print(f"(1) C_dom strictly larger than C_mono     : {rep.c_dom_contains_non_mono} "
          f"({rep.n_non_mono_witnessed:,} non-margin-monotone targets witnessed)")
    print(f"(2) falsifiable -- probe rejects outside  : {rep.falsifiable_probe_rejects_outside} "
          f"(reject rate {rep.probe_reject_rate_outside:.4f}; "
          f"accept-inside {rep.probe_accept_rate_inside:.4f})")
    print(f"(3) ONE bit certifies sign(Delta) on C_dom: {rep.one_bit_certifies_on_C_dom} "
          f"(error rate {rep.one_bit_error_rate_C_dom:.6f})")
    print(f"(sharp) one bit FAILS without domination  : {rep.one_bit_fails_without_domination} "
          f"(error rate {rep.one_bit_error_rate_no_domination:.4f}; expect ~0.25)")
    print()
    print("Minimal counterexample (dominant -> one bit works, both outside C_mono):")
    wd = rep.witness_dom_example
    print(f"   a=(+1,+0.5)->Delta={wd['delta_plus']},  a=(+1,-0.5)->Delta={wd['delta_minus']}  "
          f"(both sign +, bit sigma=+1 certifies)")
    wn = rep.witness_nodom_example
    print(f"   without domination: a=(+1,+2)->Delta={wn['delta_plus']}, "
          f"a=(+1,-2)->Delta={wn['delta_minus']}  (opposite signs -> 2 bits)")
    print()
    print(f"OBSTRUCTION CONFIRMED (general position necessary): {rep.obstruction_confirmed}")
    print()
    print("CONCLUSION:", rep.conclusion)
    print()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.json if os.path.isabs(args.json) else os.path.join(out_dir, args.json)
    with open(out_path, "w") as f:
        json.dump(asdict(rep), f, indent=2)
    print(f"Wrote machine-readable results to {out_path}")

    # ---- Hard assertions: the WITNESS must hold (we are validating the honest
    #      "general position is necessary / weakest class open" claim) ----------
    assert rep.c_dom_contains_non_mono, "C_dom should strictly contain C_mono"
    assert rep.falsifiable_probe_rejects_outside, "C_dom must be falsifiable"
    assert rep.probe_accept_rate_inside > 0.99, "probe must accept genuine C_dom members"
    assert rep.one_bit_certifies_on_C_dom, "one bit must certify sign(Delta) on C_dom"
    assert rep.one_bit_fails_without_domination, "sharpness: one bit must fail w/o domination"
    assert rep.obstruction_confirmed, "obstruction (general position necessary) not confirmed"
    print("\nALL CHECKS PASSED (general-position assumption shown NECESSARY; "
          "unconditional weakest class remains OPEN).")


if __name__ == "__main__":
    main()
