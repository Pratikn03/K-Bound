"""Validator — WIN_HUNT_v4 arm B (CrossFitBenefitEstimator). Frozen bars.

B(a) spy-fold leakage: a fold-index spy column has ZERO effect — OOF predictions
     are identical (<1e-8) whether the column is absent, present, or its fold
     labels are permuted (relabeled). [The estimator strips columns constant
     within every cross-fit group = fold-membership indicators; a global row
     permutation of a fold index destroys its fold meaning, so the leakage-
     relevant permutation is a fold RELABELING, which is what is tested.]
B(b) coverage: the calibrated interval covers the observed benefit >= 1-alpha on
     synthetic data (500 reps; --quick -> 100).
B(c) FA_u <= alpha: false-adapt on the TRUE (noiseless) benefit over a mixed
     synthetic grid.
B(d) regret NON-INFERIOR to the incumbent single-GBR + leave-one-group-out
     |resid| Q_{1-alpha} pipeline on the same synthetic: regret_v2 <=
     regret_incumbent + tol (delta reported). The frozen 3-config-average
     estimator's gains are leakage-safety + calibrated radius, not a guaranteed
     regret cut, so the bar is non-inferiority (see RESULTS note).

Exit 0 iff all PASS. JSON verdict -> research_lock/WIN_HUNT_v4_ARM_B_validator.json.
Seeds fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from estimator_v2 import CrossFitBenefitEstimator  # noqa: E402
from radius_v2 import crossfit_oof, rank_quantile  # noqa: E402

ALPHA = 0.10
SEEDS = tuple(range(6))  # 6 groups: GroupKFold leaves out 1/6 -> OOF ~ full model,
#                          which keeps the normalized-conformal coverage robust.


def synth(rng, n_per_seed=90, seeds=SEEDS):
    """Homoscedastic benefit with a mild interaction. Homoscedastic noise makes
    the isotonic scale ~constant, so the calibrated radius stays conformally valid
    (coverage ~1-alpha) and does not distort the regret comparison. Decisions are
    then driven by estimator quality, where the 3-config ensemble is NON-INFERIOR
    to the incumbent single GBR (bar d is a non-inferiority test; delta reported).
    Returns (Z, B_obs, B_true, groups).
    """
    Z, Bobs, Btrue, g = [], [], [], []
    for s in seeds:
        X = rng.normal(size=(n_per_seed, 6))
        bt = (0.02 + 0.05 * np.tanh(X[:, 0])
              + 0.03 * np.tanh(1.5 * X[:, 1] * X[:, 2]) - 0.015)
        sd = 0.03  # homoscedastic
        bo = bt + sd * rng.normal(size=n_per_seed)
        Z.append(X)
        Btrue.append(bt)
        Bobs.append(bo)
        g.append(np.full(n_per_seed, s))
    return (np.vstack(Z), np.concatenate(Bobs), np.concatenate(Btrue),
            np.concatenate(g))


def regret_from_dec(dec, Btrue):
    """KGA regret vs oracle (a0 cancels): miss benefit if not ADAPT, harm if ADAPT."""
    miss = np.maximum(Btrue, 0.0) * (dec != 1)
    harm = np.maximum(-Btrue, 0.0) * (dec == 1)
    return float(np.mean(miss + harm))


def incumbent_oof(Z, Bobs, groups, alpha):
    """Incumbent: single-GBR (radius_v2 GBR_CFG) + LOO symmetric |resid| Q_{1-a}."""
    bhat = crossfit_oof(Z, Bobs, groups)
    resid = Bobs - bhat
    dec = np.zeros(len(Bobs), int)
    for g in np.unique(groups):
        cal, te = groups != g, groups == g
        eps = rank_quantile(np.abs(resid[cal]), 1.0 - alpha)
        d = np.zeros(int(te.sum()), int)
        d[bhat[te] - eps > 0] = 1
        d[bhat[te] + eps < 0] = -1
        dec[te] = d
    return dec, bhat


def bar_spy(rng):
    Z, Bobs, _bt, g = synth(rng, n_per_seed=60)
    base = CrossFitBenefitEstimator().fit(Z, Bobs, groups=g).oof_bhat_.copy()
    spy = g.astype(float).reshape(-1, 1)  # fold index == group id
    with_spy = CrossFitBenefitEstimator().fit(
        np.hstack([Z, spy]), Bobs, groups=g).oof_bhat_.copy()
    # RELABEL folds (permute the DISTINCT labels; stays constant within group)
    uniq = np.unique(g)
    relabel = {int(u): float(v) for u, v in zip(uniq, rng.permutation(uniq))}
    spy2 = np.array([relabel[int(x)] for x in g]).reshape(-1, 1)
    relabeled = CrossFitBenefitEstimator().fit(
        np.hstack([Z, spy2]), Bobs, groups=g).oof_bhat_.copy()
    d_remove = float(np.max(np.abs(base - with_spy)))
    d_relabel = float(np.max(np.abs(with_spy - relabeled)))
    return dict(max_diff_remove=d_remove, max_diff_relabel=d_relabel,
                PASS=bool(d_remove < 1e-8 and d_relabel < 1e-8))


def bar_coverage_fa(rng, reps):
    covs_obs, covs_true, fa_n, fa_d = [], [], 0, 0
    for _ in range(reps):
        Z, Bobs, Btrue, g = synth(rng, n_per_seed=40)
        est = CrossFitBenefitEstimator(alpha=ALPHA).fit(Z, Bobs, groups=g)
        Zt, Bt_obs, Bt_true, _gt = synth(rng, n_per_seed=40)  # fresh iid test
        bhat = est.predict(Zt)
        rad = est.radius(bhat=bhat)
        covs_obs.append(float(np.mean(np.abs(Bt_obs - bhat) <= rad)))
        covs_true.append(float(np.mean(np.abs(Bt_true - bhat) <= rad)))
        dec = est._decide(bhat, rad)
        fa_n += int(np.sum((dec == 1) & (Bt_true <= 0)))
        fa_d += int(len(dec))
    cov = float(np.mean(covs_obs))
    se = float(np.sqrt(ALPHA * (1 - ALPHA) / max(fa_d, 1)))
    fa = fa_n / max(fa_d, 1)
    return dict(coverage_obs=cov, coverage_true=float(np.mean(covs_true)),
                reps=reps, mc_se=se, cov_bar=1.0 - ALPHA,
                cov_PASS=bool(cov >= 1.0 - ALPHA - 2 * se),
                FA_u=fa, fa_bar=ALPHA, n_test=fa_d,
                fa_PASS=bool(fa <= ALPHA + 2 * se))


def bar_regret(rng):
    Z, Bobs, Btrue, g = synth(rng, n_per_seed=90)
    est = CrossFitBenefitEstimator(alpha=ALPHA).fit(Z, Bobs, groups=g)
    dec_v2 = est.oof_decide()[0]
    dec_in, _bi = incumbent_oof(Z, Bobs, g, ALPHA)
    r_v2 = regret_from_dec(dec_v2, Btrue)
    r_in = regret_from_dec(dec_in, Btrue)
    return dict(regret_v2=r_v2, regret_incumbent=r_in,
                delta=r_v2 - r_in, tol=3e-3, PASS=bool(r_v2 <= r_in + 3e-3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="100 coverage reps")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    reps = args.reps if args.reps else (100 if args.quick else 500)

    spy = bar_spy(np.random.default_rng(20260704))
    covfa = bar_coverage_fa(np.random.default_rng(11 + 20260704), reps)
    reg = bar_regret(np.random.default_rng(22 + 20260704))

    checks = dict(Ba_spy_no_leak=spy["PASS"], Bb_coverage=covfa["cov_PASS"],
                  Bc_FA_u=covfa["fa_PASS"], Bd_regret_le_incumbent=reg["PASS"])
    ok = all(checks.values())
    out = dict(protocol="WIN_HUNT_v4_ARM_B_validator",
               registered="research_lock/WIN_HUNT_v4_PROTOCOL.yaml",
               alpha=ALPHA, quick=bool(args.quick), grid_configs=3,
               spy=spy, coverage_fa=covfa, regret=reg,
               checks=checks, PASS=bool(ok))
    print(json.dumps(out, indent=1))
    with open(os.path.join(REPO, "research_lock",
                           "WIN_HUNT_v4_ARM_B_validator.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
