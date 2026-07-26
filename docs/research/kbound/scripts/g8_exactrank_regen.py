#!/usr/bin/env python3
"""g8_exactrank_regen.py -- regenerate every track under ONE declared radius rule.

Fix-queue items 2, 4, 15 and 30.

  * item 4 -- the old line 26 computed both radii from the WHOLE residual vector::

        rho = np.abs(bh - B); ei = float(np.quantile(rho, 1 - ALPHA)); ee = cexact(rho)

    and then scored the same cells those residuals came from.  The default here
    is leave-one-out-of-pool: cell i's radius is the exact conformal rank
    quantile of the other n-1 residuals.

  * item 2 -- the exact split-conformal rank rule ``k = ceil((n+1)(1-alpha))`` is
    the promoted rule.  The interpolated ``np.quantile`` column is still printed
    because this script's job is to show what "one rule everywhere" costs
    (NUMBERS_PACK.md sec. 0.4 lists five published rows that move), but it is
    labelled SUPERSEDED and it is never the basis of a verdict here.

  * item 5 -- every track now prints its ADAPT/FREEZE/ABSTAIN composition and the
    structural FA_u ceiling ``(n-k)/n``.  Under in-pool rank calibration FA_u
    cannot exceed that ceiling for ANY data (0.0972 at n=432, 0.0370 at n=27,
    exactly 0 at n<=9), so "FA_u <= alpha" is an identity, not a measurement.
    Reporting FA_u without the ceiling next to it overstates the evidence.

  * item 30 -- ``R`` no longer hard-codes a machine-local checkout path.
    Override with ``KBOUND_RESULTS_ROOT``.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbound_decide import (  # noqa: E402
    conformal_radius, decide, decide_from_records, fa_ceiling, false_adapt,
    records, results_root,
)

ALPHA = 0.10
R = results_root()


def _regret(dec, B):
    B = np.asarray(B, float)
    orc = np.where(B > 0, "ADAPT", "FREEZE")
    act = np.where(np.asarray(dec) == "ADAPT", "ADAPT", "FREEZE")
    return float(np.mean(np.abs(B) * (act != orc)))


def pick_seeds(files):
    by = {}
    for f in files:
        m = re.search(r"seed(\d+)", os.path.basename(f))
        if m:
            by.setdefault(int(m.group(1)), f)   # first hit per seed
    return [by[s] for s in sorted(by)]


def track(name, globs, calibration="loo", show_interp=True):
    files = []
    for g in globs:
        files += glob.glob(g, recursive=True)
    fs = pick_seeds(files)
    if not fs:
        print(f"\n=== {name} ===\n  [SKIP] no per-condition dumps matched:")
        for g in globs:
            print(f"         {g}")
        print("         -> set KBOUND_RESULTS_ROOT, or this track is not in the release.")
        return
    fa_e = fa_i = n_tot = 0
    rk, ra, rf = [], [], []
    counts = {"ADAPT": 0, "FREEZE": 0, "ABSTAIN": 0}
    n_per = None
    for fp in fs:
        recs = records(fp)
        B = np.array([x["B"] for x in recs], float)
        bh = np.array([x.get("b_hat", x.get("Bhat")) for x in recs], float)
        n_per = len(B)
        # promoted rule, chosen calibration
        _eps_e, de = decide_from_records(bh, B, alpha=ALPHA, calibration=calibration)
        fa_e += false_adapt(de, B)["n_false_adapt"]
        n_tot += len(B)
        for d in counts:
            counts[d] += int(np.sum(np.asarray(de) == d))
        rk.append(_regret(de, B))
        ra.append(_regret(["ADAPT"] * len(B), B))
        rf.append(_regret(["FREEZE"] * len(B), B))
        if show_interp:
            # SUPERSEDED archived rule, reported only to price "one rule everywhere".
            rho = np.abs(bh - B)
            ei = float(np.quantile(rho, 1 - ALPHA))
            fa_i += false_adapt(decide(bh, ei, alpha=ALPHA), B)["n_false_adapt"]

    ceil_e = fa_ceiling(n_per, ALPHA)
    print(f"\n=== {name}: {len(fs)} seeds, n/seed={n_per} ===")
    print(f"  calibration={calibration}  rule=exact-rank k=ceil((n+1)(1-alpha))")
    print(f"  actions pooled: {counts}")
    print(f"  FA_u exact-rank = {fa_e}/{n_tot} = {fa_e/n_tot:.4f}   "
          f"(in-pool structural ceiling at n={n_per} is (n-k)/n = {ceil_e:.4f}"
          f"{' -- BELOW alpha, so FA_u<=alpha is an identity there' if ceil_e <= ALPHA else ''})")
    if counts["ADAPT"] < 10:
        print(f"  [!] only {counts['ADAPT']} ADAPT decisions: the false-adapt guarantee is "
              f"UNTESTED on this track (fix-queue item 5c).")
    if show_interp:
        print(f"  FA_u interpolated (SUPERSEDED rule) = {fa_i}/{n_tot} = {fa_i/n_tot:.4f}")
    print(f"  regret: KGA={np.mean(rk):.6f}  adapt={np.mean(ra):.6f}  freeze={np.mean(rf):.6f}")
    bb = np.mean(rk) < np.mean(ra) - 1e-9 and np.mean(rk) < np.mean(rf) - 1e-9
    print(f"  POINT-ESTIMATE beats-both: {bb}  "
          f"(no interval is claimed here; see g8_exactrank_ci.py --unit condition)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", default="loo", choices=["loo", "in_pool"])
    ap.add_argument("--no-interpolated", action="store_true",
                    help="suppress the superseded interpolated-rule column")
    a = ap.parse_args()
    show = not a.no_interpolated

    IC = os.path.join(R, "win_hunt_v5_imagenetc_ms")
    for cand in ("sar", "eata", "tent"):
        track(f"ImageNet-C {cand.upper()}",
              [os.path.join(IC, "seed*", f"per_condition_imagenetc_{cand}_seed*.json"),
               os.path.join(IC, "pooled_5seed", f"per_condition_imagenetc_{cand}_seed*.json")],
              calibration=a.calibration, show_interp=show)

    SG = os.path.join(R, "stress_grid_multiseed_v1")
    for cand in ("tent", "eata"):
        track(f"CIFAR-10-C {cand.upper()}",
              [os.path.join(SG, "seed*", f"per_condition_cifar10c_{cand}_seed*.json")],
              calibration=a.calibration, show_interp=show)


if __name__ == "__main__":
    main()
