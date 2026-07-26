#!/usr/bin/env python3
"""g8_canonical_pooling.py -- pooled ImageNet-C regret under ONE declared radius rule.

Fix-queue items 4, 15 and 30.  What changed:

  * item 4 -- the radius.  The old line 11 was::

        rho = np.abs(bh - B)
        eps = cexact(rho) if use_exact else float(np.quantile(rho, 1 - A))

    i.e. one radius per file computed from ALL residuals in that file, then used
    to decide every cell in the same file including the cell whose residual is in
    the pool.  eps was therefore a function of the very test labels that the
    FA_u <= alpha guarantee attaches to.  The default is now
    leave-one-out-of-pool: cell i's radius uses the other n-1 residuals only.
    Effect on this track (NUMBERS_PACK.md sec. 4.2): SAR regret 0.026422 ->
    0.028893, FA_u 0/135 -> 1/135, ADAPT 12 -> 13, ABSTAIN 109 -> 107.

  * items 2 + 4 -- one rule.  The interpolated ``np.quantile`` branch is gone
    from the default path.  The exact split-conformal rank quantile
    ``k = ceil((n+1)(1-alpha))`` is the promoted rule and is now the only rule
    this script computes unless you explicitly ask for the archived comparison
    with ``--show-archived-interpolated``.

  * item 15 -- the radius and the decision rule come from ``kbound_decide``,
    which calls ``kga.certificate`` / ``kga.policy``.

  * item 30 -- ``R`` no longer hard-codes a machine-local checkout path,
    which is banned by ``EXTERNAL_STORAGE_POLICY.md:18``.  Override the results
    tree with ``KBOUND_RESULTS_ROOT``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbound_decide import decide_from_records, false_adapt, records, results_root  # noqa: E402

ALPHA = 0.10
POOLED = os.path.join(results_root(), "win_hunt_v5_imagenetc_ms", "pooled_5seed")


def pool(cand, calibration="loo", root=POOLED):
    """Pool one candidate across seeds. Radius is refit per FILE (= per seed),
    which is the shipped convention; only the in-pool/LOO choice changed."""
    files = sorted(glob.glob(os.path.join(root, f"per_condition_imagenetc_{cand}_seed*.json")))
    if not files:
        raise FileNotFoundError(
            f"No per-condition dumps for {cand!r} under {root}.\n"
            f"  -> set KBOUND_RESULTS_ROOT to a results tree containing "
            f"win_hunt_v5_imagenetc_ms/pooled_5seed/."
        )
    allB, allDec, allEps = [], [], []
    for f in files:
        recs = records(f)
        B = np.array([x["B"] for x in recs], float)
        bh = np.array([x["b_hat"] for x in recs], float)
        eps, dec = decide_from_records(bh, B, alpha=ALPHA, calibration=calibration)
        allB += list(B)
        allDec += list(dec)
        allEps += list(np.atleast_1d(eps))
    B = np.array(allB, float)
    dec = np.array(allDec, dtype=object)
    act = np.where(dec == "ADAPT", "ADAPT", "FREEZE")
    orc = np.where(B > 0, "ADAPT", "FREEZE")
    fa = false_adapt(dec, B)
    return {
        "n_files": len(files),
        "regret_kga": float(np.mean(np.abs(B) * (act != orc))),
        "regret_adapt": float(np.mean(np.abs(B) * ("ADAPT" != orc))),
        "regret_freeze": float(np.mean(np.abs(B) * ("FREEZE" != orc))),
        "counts": {d: int((dec == d).sum()) for d in ("ADAPT", "FREEZE", "ABSTAIN")},
        "eps_min": float(np.min(allEps)), "eps_max": float(np.max(allEps)),
        **fa,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default="loo", choices=["loo", "in_pool"],
                    help="DEFAULT 'loo' (leave-one-out-of-pool). 'in_pool' reproduces "
                         "the archived, leaky radius for provenance checks only.")
    ap.add_argument("--candidates", nargs="+", default=["sar", "eata", "tent"])
    ap.add_argument("--root", default=POOLED)
    ap.add_argument("--show-archived-interpolated", action="store_true",
                    help="also print the superseded np.quantile-interpolated rule that "
                         "SUBMISSION_LEDGER.md:88-89 ordered dropped. For provenance "
                         "diffs only -- do not publish either column from this flag.")
    a = ap.parse_args()

    print(f"ImageNet-C pooled regret | rule=exact-rank k=ceil((n+1)(1-alpha)) "
          f"| calibration={a.calibration} | alpha={ALPHA}")
    for cand in a.candidates:
        r = pool(cand, calibration=a.calibration, root=a.root)
        print(f"\n{cand.upper()}  ({r['n_files']} seed files, n={r['n']})")
        print(f"  regret: KGA={r['regret_kga']:.6f}  adapt={r['regret_adapt']:.6f}  "
              f"freeze={r['regret_freeze']:.6f}")
        print(f"  actions: {r['counts']}   eps in [{r['eps_min']:.5f}, {r['eps_max']:.5f}]")
        fac = "n/a" if r["fa_c"] is None else f"{r['fa_c']:.4f}"
        print(f"  false adapts (ADAPT and B<=0): {r['n_false_adapt']}  "
              f"FA_u={r['fa_u']:.4f}  FA_c={fac}")
        print(f"  point-estimate beats both: "
              f"{r['regret_kga'] < r['regret_adapt'] - 1e-9 and r['regret_kga'] < r['regret_freeze'] - 1e-9}"
              f"   (this is a POINT estimate; for intervals run g8_exactrank_ci.py --unit condition)")
        if a.show_archived_interpolated:
            print("  [archived, superseded] the interpolated np.quantile rule is not "
                  "recomputed here on purpose; it is the rule the ledger ordered dropped.")


if __name__ == "__main__":
    main()
