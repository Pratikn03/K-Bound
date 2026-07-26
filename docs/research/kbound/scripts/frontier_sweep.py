#!/usr/bin/env python3
"""frontier_sweep.py -- THE REAL beta-sweep frontier experiment.  NOT YET RUN.

Fix-queue item 14 (F5-4 / F1-4), and the single biggest upgrade available to the
paper.  ``frontier_validation.py`` is an illustration on a circular DGP: its Z is
four noisy copies of M, so eps -> 0.9*beta by algebra and the frontier "lands at
beta" by construction.  This script is the design that would actually test the
population rule, on the CIFAR-10-C stress grid, against real benefits.

>>> STATUS: SCAFFOLD.  It writes NO results file and fabricates NO numbers.
>>> Running it without --i-have-run-the-real-thing prints this plan and exits 2.

---------------------------------------------------------------------------
THE DESIGN
---------------------------------------------------------------------------
The population frontier rule (thm:headline) says: commit iff |M| > beta, where

    M     = the observable margin,
    beta  = the declared budget on the unobserved calibration drift gamma,
    Delta = M + gamma  is the true benefit of adapting.

Nothing in the nine benchmark tracks ever supplies beta to KGA
(``kbound_short.tex:596-598``), so the population theory has zero contact with
the empirical section.  This closes that gap:

  (1) M FROM REAL DATA, NOT FROM THE LABELS.
      Compute M with the ATC-style source-calibrated score already described at
      ``kbound_short.tex:364``: fit an isotonic confidence->accuracy map on the
      SOURCE/dev split only, apply it to the target cell's pre- and post-adapt
      confidences, and take M = ahat_adapted - ahat_frozen.  M must be a function
      of label-free target evidence plus source labels only.
      Crucially M must NOT be a noisy copy of Delta -- report corr(M, Delta) and
      the residual spread; if |corr| > 0.99 the experiment has reproduced the
      circularity it exists to escape and must be reported as such.

  (2) BETA DECLARED, NOT FITTED.
      beta comes from HISTORICAL dev-to-deployment gaps -- e.g. the observed
      |ahat - a| discrepancy of the ATC map on held-out source corruptions -- and
      is fixed BEFORE any target cell is scored.  A beta tuned on the sweep is
      the fatal flaw the project has so far avoided (R3 grepped for it and found
      none); do not introduce it here.

  (3) SWEEP beta in {0, 0.02, 0.05, 0.10, 0.20} and run the population rule
      against the finite-sample certificate Deltahat +/- eps:

          population rule : commit iff |M| > beta
          KGA rule        : ADAPT if Deltahat - eps > 0, FREEZE if + eps < 0
          agreement       : per beta, the 2x2 table of (commit?) x (KGA commits?)
                            plus regret and FA_u of each rule

  (4) HELD-OUT CALIBRATION.  eps is calibrated on a corruption-disjoint split
      (leave-one-corruption-out, 6 folds), not leave-one-cell-out: at cell level
      the residual MAE is 0.0096 and eps 0.021, at corruption level 0.0309 and
      0.0972 (NUMBERS_PACK.md sec. 7.1).  Which of those two is the honest
      operating point is exactly what this experiment decides.

  (5) REPORT THE NEGATIVE.  If the population rule and the certificate disagree,
      or if the frontier does not track beta on real data, that is a publishable
      result and a far more informative one than the current silence (F5-5).
      Pre-commit to reporting the sweep whatever it shows.

---------------------------------------------------------------------------
WHAT IS ALREADY WIRED
---------------------------------------------------------------------------
  * ``load_grid()``   -- reads the committed 432-cell per-condition dumps.
  * ``atc_margin()``  -- the ATC-style M, computed source-only.  IMPLEMENTED but
                         UNVALIDATED: its isotonic map is fitted on the cells of
                         the held-out corruptions, which is the right split but
                         has never been checked against a source-only run.
  * ``sweep()``       -- the beta loop and the agreement table.
  * NOT wired: the declaration of beta.  ``--beta-source`` has no default on
                         purpose.  Supplying a beta is a research decision, not a
                         default.

TODO BEFORE THIS PRODUCES A NUMBER
  [ ] Decide and DOCUMENT the beta declaration procedure; record it in
      research_lock/ before running (the pre-registration forbids re-tuning).
  [ ] Verify atc_margin() against a genuine source-split fit, not a
      leave-one-corruption-out fit on target cells.
  [ ] Decide whether the eval-time confidences in Z are the right ATC input;
      Z[:, PRE_CONF] / Z[:, POST_CONF] are means over the eval pool, and ATC
      normally needs the full score distribution.
  [ ] Run on all 5 seeds and report per-seed spread, not a pooled mean.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbound_decide import decide, loo_bhat, conformal_radius, false_adapt, records, results_root  # noqa: E402

ALPHA = 0.10
BETAS = (0.0, 0.02, 0.05, 0.10, 0.20)
# Evidence layout, identical to cifar_tent_mps_v2.EVIDENCE_NAMES.
(PRE_ENT, PRE_CONF, PRE_PBAL, POST_ENT, POST_CONF, POST_PBAL,
 PBAL_DROP, ENT_DROP, FRAC_HI, MKL, UPD) = range(11)


def load_grid(cand="tent", seed=1, root=None):
    """Load one 432-cell CIFAR-10-C stress-grid seed."""
    root = root or results_root()
    pat = os.path.join(root, "stress_grid_multiseed_v1", f"seed{seed}",
                       f"per_condition_cifar10c_{cand}_seed{seed}.json")
    files = glob.glob(pat)
    if not files:
        raise FileNotFoundError(
            f"No per-condition dump at {pat}\n"
            f"  -> set KBOUND_RESULTS_ROOT, or pick a seed in 1..4 "
            f"(seed 0 has no per-condition dump; see fix-queue item 8)."
        )
    recs = records(files[0])
    Z = np.array([r["Z"] for r in recs], float)
    B = np.array([r["B"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["a_adapted"] for r in recs], float)
    corr = np.array([r["condition"].split("|")[0] for r in recs])
    return Z, B, a0, aa, corr


def atc_margin(Z, a0, aa, corr):
    """ATC-style source-calibrated margin M = ahat_adapted - ahat_frozen.

    UNVALIDATED (see the TODO list).  The isotonic confidence->accuracy map is
    fitted leave-one-corruption-out, so no cell contributes to its own map, but
    the fit still uses TARGET-corruption labels from the other folds.  A genuine
    source-only fit needs an in-distribution dev split that the committed dumps
    do not carry.  Do not report M from this function as label-free until that is
    resolved.
    """
    from sklearn.isotonic import IsotonicRegression
    M = np.full(len(a0), np.nan)
    for t in np.unique(corr):
        te = corr == t
        dv = ~te
        conf = np.concatenate([Z[dv, PRE_CONF], Z[dv, POST_CONF]])
        acc = np.concatenate([a0[dv], aa[dv]])
        ir = IsotonicRegression(out_of_bounds="clip").fit(conf, acc)
        M[te] = ir.predict(Z[te, POST_CONF]) - ir.predict(Z[te, PRE_CONF])
    return M


def loco_certificate(Z, B, corr, alpha=ALPHA):
    """Deltahat + eps under leave-one-CORRUPTION-out calibration (design point 4).

    Returns (Deltahat, eps_per_cell): a cell's radius comes only from folds that
    exclude its corruption, so neither the estimator nor the radius has seen the
    cell or any of its twins.
    """
    bh = loo_bhat(Z, B)
    resid = np.abs(bh - B)
    eps = np.empty(len(B))
    for t in np.unique(corr):
        te = corr == t
        eps[te] = conformal_radius(resid[~te], alpha=alpha)
    return bh, eps


def sweep(M, bh, eps, B, a0, aa, betas=BETAS, alpha=ALPHA):
    """Population rule vs certificate, per beta.  Returns a list of row dicts."""
    kga_dec = decide(bh, eps, alpha=alpha)
    orc = np.maximum(a0, aa)
    rows = []
    for beta in betas:
        pop_commit = np.abs(M) > beta
        pop_dec = np.where(pop_commit & (M > 0), "ADAPT",
                           np.where(pop_commit & (M < 0), "FREEZE", "ABSTAIN"))
        pop_acc = np.where(pop_dec == "ADAPT", aa, a0)
        kga_acc = np.where(kga_dec == "ADAPT", aa, a0)
        rows.append({
            "beta": beta,
            "pop_commit_rate": float(pop_commit.mean()),
            "kga_commit_rate": float(np.mean(kga_dec != "ABSTAIN")),
            "agree_commit": float(np.mean(pop_commit == (kga_dec != "ABSTAIN"))),
            "agree_action": float(np.mean(pop_dec == kga_dec)),
            "pop_regret": float(np.mean(orc - pop_acc)),
            "kga_regret": float(np.mean(orc - kga_acc)),
            "pop_fa_u": false_adapt(pop_dec, B)["fa_u"],
            "kga_fa_u": false_adapt(kga_dec, B)["fa_u"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", default="tent", choices=["tent", "eata", "sar"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default=None)
    ap.add_argument("--beta-source", default=None,
                    help="REQUIRED for a real run: a short string naming where beta was "
                         "declared from (e.g. 'historical dev-to-deployment ATC gap, "
                         "research_lock/frontier_beta_v1.yaml'). There is no default: "
                         "declaring beta is a research decision.")
    ap.add_argument("--i-have-run-the-real-thing", action="store_true",
                    help="acknowledge that the TODO list at the top of this file has been "
                         "worked through. Without it the script prints the plan and exits.")
    a = ap.parse_args()

    if not a.i_have_run_the_real_thing or not a.beta_source:
        print(__doc__)
        print("\n" + "=" * 72)
        print("SCAFFOLD ONLY -- no results were computed and no file was written.")
        print("Supply --beta-source '<where beta came from>' and "
              "--i-have-run-the-real-thing after working through the TODO list.")
        print("=" * 72)
        return 2

    Z, B, a0, aa, corr = load_grid(a.candidate, a.seed, a.root)
    M = atc_margin(Z, a0, aa, corr)
    r = float(np.corrcoef(M, B)[0, 1])
    print(f"corr(M, Delta) = {r:.4f}   "
          f"{'<-- CIRCULAR, report as such' if abs(r) > 0.99 else 'OK: M is not a copy of Delta'}")
    bh, eps = loco_certificate(Z, B, corr)
    print(f"leave-one-corruption-out eps: mean {eps.mean():.5f} "
          f"[{eps.min():.5f}, {eps.max():.5f}]")
    print(f"beta declared from: {a.beta_source}")
    print(f"\n{'beta':>6} {'pop_commit':>11} {'kga_commit':>11} {'agree_act':>10} "
          f"{'pop_regret':>11} {'kga_regret':>11} {'pop_FA_u':>9} {'kga_FA_u':>9}")
    for row in sweep(M, bh, eps, B, a0, aa):
        print(f"{row['beta']:>6.2f} {row['pop_commit_rate']:>11.3f} "
              f"{row['kga_commit_rate']:>11.3f} {row['agree_action']:>10.3f} "
              f"{row['pop_regret']:>11.5f} {row['kga_regret']:>11.5f} "
              f"{row['pop_fa_u']:>9.4f} {row['kga_fa_u']:>9.4f}")
    print("\nReport this table whatever it shows (design point 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
