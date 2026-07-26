#!/usr/bin/env python3
"""Probe 2: reproduce the reported per-candidate ``eps_conformal`` and harm-AUC on the
Camelyon17 MPS debug run, and show what the fixed radius rule does to them.

FIX-QUEUE ITEM 15.  This file carried ``decide_kga`` fork #7 of seven -- a private
copy of the LOO-GBR estimator plus the conformal radius plus the decision rule.
The body is gone.  Everything now goes through
``docs/research/kbound/scripts/kbound_decide.py``, which calls the shipped
``kga.certificate`` / ``kga.policy``.

FIX-QUEUE ITEM 4.  This probe's *job* is to reproduce an archived number, and that
number was produced by the leaky in-pool interpolated rule.  So, unlike every
other driver, the archived rule stays available here -- but it is now (a)
explicit, (b) labelled, and (c) printed side by side with the
leave-one-out-of-pool exact-rank radius the paper should be quoting.  A probe
that silently reproduces a defect is how the defect survives.

FIX-QUEUE ITEM 30.  ``P`` was a hard-coded absolute path into one machine's
session mount, so the probe could not run anywhere else.  It now resolves inside
the repository, with ``KBOUND_RESULTS_ROOT`` / ``--result`` as overrides.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, *[os.pardir] * 3, "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from kbound_decide import (  # noqa: E402
    conformal_radius, decide_from_records, loo_bhat, read_json, results_root,
)

ALPHA = 0.10
DEFAULT_RESULT = os.path.join(
    results_root(), "wilds_kbound_debug_mps", "result_73add410.json")
CANDIDATES = ["tent_online", "tent_episodic", "eata_online",
              "eata_episodic", "sar_online", "sar_episodic"]


def auc(score, label):
    """Mann-Whitney U AUC: P(a positive outranks a negative)."""
    score = np.asarray(score, float)
    label = np.asarray(label, int)
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    U = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return U / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", default=DEFAULT_RESULT,
                    help="Camelyon17 debug result JSON (default: repo copy; "
                         "override the results tree with KBOUND_RESULTS_ROOT)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--use-stored-bhat", action="store_true",
                    help="use the artifact's stored per-record b_hat instead of "
                         "refitting the LOO GBR (faster). On result_73add410.json the "
                         "two paths agree to the printed precision; they are not "
                         "guaranteed to, because b_hat was written under an unpinned "
                         "scikit-learn (fix-queue item 19).")
    a = ap.parse_args()

    # read_json names the file and says what to do when the artifact is absent or
    # is a NUL-filled iCloud placeholder, instead of a bare OSError.
    d = read_json(a.result)
    recs = d["records"]
    det = d["detectability"]
    ra = d["routing_a_single_candidate"]

    by_cand = defaultdict(list)
    for r in recs:
        by_cand[r["candidate"]].append(r)

    all_Bhat, all_B = [], []
    print("Radius rules: 'archived' = in-pool INTERPOLATED np.quantile, the rule that\n"
          "produced the reported column (fix-queue item 4 calls it leaky);\n"
          "'exact/in-pool' and 'exact/LOO' are the promoted exact-rank rule without and\n"
          "with the scored cell removed from its own calibration pool.\n")
    print("candidate           reported_eps  archived_eps  exact/in-pool  exact/LOO(mean)"
          "  beats_both(rep)")
    for cand in CANDIDATES:
        rs = by_cand.get(cand, [])
        if not rs:
            print(f"{cand:<18}  (absent from this result file)")
            continue
        Z = np.array([r["Z"] for r in rs], float)
        B = np.array([r["B"] for r in rs], float)
        # DEFAULT: refit the LOO GBR, which is what this probe exists to check.
        if a.use_stored_bhat and all("b_hat" in r for r in rs):
            Bhat = np.array([r["b_hat"] for r in rs], float)
        else:
            Bhat = loo_bhat(Z, B)
        resid = np.abs(Bhat - B)
        eps_archived = float(np.quantile(resid, 1 - a.alpha))   # the superseded rule
        eps_inpool = conformal_radius(resid, a.alpha)
        eps_loo, _ = decide_from_records(Bhat, B, alpha=a.alpha, calibration="loo")
        rep = ra[cand]["kga"]["eps_conformal"]
        bb = ra[cand]["kga"]["beats_both"]
        print("%-18s  %.5f      %.5f      %.5f       %.5f        %s"
              % (cand, rep, eps_archived, eps_inpool, float(np.mean(eps_loo)), bb))
        all_Bhat.append(Bhat)
        all_B.append(B)

    if not all_Bhat:
        print("\nNo candidates found; nothing to pool.")
        return
    all_Bhat = np.concatenate(all_Bhat)
    all_B = np.concatenate(all_B)
    lab = (all_B < 0).astype(int)
    print(f"\nPOOLED over {len(lab)} cells:")
    print("  NOTE: the artifact's `detectability` block fits ONE estimator over all cells")
    print("        at once, while this probe concatenates SIX per-candidate LOO fits, so")
    print("        the two harm-AUCs are different estimands and are not expected to match.")
    print("  reported certificate_harm_AUC_negBhat:", det["certificate_harm_AUC_negBhat"])
    print("  reproduced harm-AUC(-Bhat)           :", auc(-all_Bhat, lab))
    print("  reported certificate_eps             :", det["certificate_eps"])
    print("  reproduced pooled eps (archived rule):",
          float(np.quantile(np.abs(all_Bhat - all_B), 1 - a.alpha)))
    print("  reproduced pooled eps (exact rank)   :",
          conformal_radius(np.abs(all_Bhat - all_B), a.alpha))
    print("  n_harmful B<0:", int(lab.sum()), "of", len(lab))


if __name__ == "__main__":
    main()
