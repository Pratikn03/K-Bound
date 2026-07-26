#!/usr/bin/env python3
"""g8_exactrank_ci.py -- ImageNet-C gap intervals at the UNIT OF ANALYSIS THE TEXT CLAIMS.

Fix-queue item 3.  The previous body of this file was::

    def boot(g):
        idx = rng.integers(0, n, (5000, n)); ms = g[idx].mean(1)
        return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))

with ``n = 135``: it resampled the 135 **cell-seed rows** i.i.d.  Those rows are
not independent -- they are 27 conditions observed under 5 seeds, and the 5 rows
sharing a condition are the same corruption at the same severity.  Meanwhile
``kbound_short.tex:797-802`` describes a **seed-averaged** design, and
``_locked_analysis_script.py:54`` already does exactly that for the CIFAR rows.
Bootstrapping 135 correlated rows as independent understates the interval by
roughly the design effect.

WHY THE CELL-SEED UNIT IS WRONG, CONCRETELY
    A bootstrap replicate must be exchangeable with the observed sample under the
    sampling design.  The design here draws 27 conditions once and then re-runs
    each of them under 5 seeds; the seeds are a within-condition replication, not
    27x5 = 135 fresh draws from the condition population.  Resampling rows i.i.d.
    lets one condition appear with all 5 of its seeds while another vanishes, and
    it treats the 5 near-duplicate rows as 5 independent pieces of evidence about
    the condition population.  The inferential target -- "does KGA beat the fixed
    policy on a NEW corruption condition?" -- is a statement about conditions, so
    the resampling unit must be the condition.

    Empirically this is not academic.  Under the promoted in-pool exact-rank
    radius the KGA - always-adapt gap goes from [-0.0519, -0.0034] (135 rows,
    excludes zero) to [-0.0806, +0.0175] (27 conditions, INCLUDES zero).

DEFAULT: --unit condition.  The other units are available for the sensitivity
table but must not be reported as the primary interval; --unit cell_seed
reproduces the shipped (wrong) number and says so on stdout.

20,000 paired percentile-bootstrap replicates, fixed seed, resampling the SAME
unit indices for KGA and for the fixed policy so the comparison stays paired.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbound_decide import decide_from_records, records, results_root  # noqa: E402

ALPHA = 0.10
NBOOT = 20000
SEED = 20260720   # the stream seed the shipped script used; kept so results are comparable

# fix-queue item 30: no machine-local path.  Override with KBOUND_RESULTS_ROOT.
POOLED = os.path.join(results_root(), "win_hunt_v5_imagenetc_ms", "pooled_5seed")

# The ImageNet-C grid is 3 corruption families x 3 severities x 3 batch/composition
# cells = 27 conditions.  Family is the first token of the condition string.
_FAMILY = re.compile(r"^([a-z_]+)")


def load_track(cand, calibration="loo", root=POOLED):
    """Return per-row (condition, seed, regret_kga, regret_adapt, regret_freeze).

    Regret convention is ``g8_canonical_pooling``'s "Method B":
    ``|B| * (action != oracle_action)``.
    """
    files = sorted(glob.glob(os.path.join(root, f"per_condition_imagenetc_{cand}_seed*.json")))
    if not files:
        raise FileNotFoundError(
            f"No per-condition dumps for candidate {cand!r} under {root}.\n"
            f"  -> set KBOUND_RESULTS_ROOT to the results tree that contains\n"
            f"     win_hunt_v5_imagenetc_ms/pooled_5seed/."
        )
    conds, seeds, rk, ra, rf = [], [], [], [], []
    for f in files:
        m = re.search(r"seed(\d+)", os.path.basename(f))
        s = int(m.group(1)) if m else -1
        recs = records(f)
        B = np.array([x["B"] for x in recs], float)
        bh = np.array([x["b_hat"] for x in recs], float)
        _eps, dec = decide_from_records(bh, B, alpha=ALPHA, calibration=calibration)
        act = np.where(np.asarray(dec) == "ADAPT", "ADAPT", "FREEZE")
        orc = np.where(B > 0, "ADAPT", "FREEZE")
        rk += list(np.abs(B) * (act != orc))
        ra += list(np.abs(B) * ("ADAPT" != orc))
        rf += list(np.abs(B) * ("FREEZE" != orc))
        conds += [x.get("condition", f"cond{i}") for i, x in enumerate(recs)]
        seeds += [s] * len(recs)
    return (np.array(conds), np.array(seeds), np.array(rk, float),
            np.array(ra, float), np.array(rf, float))


def _unit_labels(conds, seeds, unit):
    if unit == "condition":
        return conds
    if unit == "cell_seed":
        return np.array([f"{c}#{s}" for c, s in zip(conds, seeds)])
    if unit == "seed":
        return seeds.astype(str)
    if unit == "family":
        return np.array([_FAMILY.match(c).group(1) if _FAMILY.match(c) else c for c in conds])
    raise ValueError(f"unknown unit {unit!r}")


def paired_boot(diff, labels, nboot=NBOOT, seed=SEED):
    """Paired percentile bootstrap of mean(diff), resampling whole units.

    ``diff`` is the per-row paired difference (KGA regret minus the fixed
    policy's regret on the SAME row), so the pairing is preserved by
    construction.  For unit = condition the rows of a unit are first averaged
    (seed-averaging), which is the design ``kbound_short.tex:797-802`` describes.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    per_unit = np.array([diff[labels == u].mean() for u in uniq], float)
    m = per_unit.size
    idx = rng.integers(0, m, size=(nboot, m))
    bs = per_unit[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"units": int(m), "obs": float(per_unit.mean()),
            "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(hi < 0 or lo > 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", default="condition",
                    choices=["condition", "cell_seed", "seed", "family"],
                    help="bootstrap resampling unit. DEFAULT 'condition' (27 units, "
                         "seed-averaged) -- the design the manuscript describes. "
                         "'cell_seed' (135 units) reproduces the shipped number and is "
                         "WRONG: it treats 5 seeds of one condition as 5 independent "
                         "conditions.")
    ap.add_argument("--calibration", default="loo", choices=["loo", "in_pool"],
                    help="conformal radius calibration (fix-queue item 4). 'loo' removes "
                         "the scored cell from its own pool; 'in_pool' reproduces the "
                         "archived rule.")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--candidates", nargs="+", default=["sar", "eata", "tent"])
    ap.add_argument("--root", default=POOLED)
    a = ap.parse_args()

    if a.nboot < 20000:
        print(f"[warn] --nboot {a.nboot} < 20000; percentile endpoints will be noisy "
              f"at the 2.5/97.5 tails.")
    if a.unit == "cell_seed":
        print("[WARNING] --unit cell_seed resamples 135 correlated rows as if they were "
              "independent. It reproduces the shipped interval; it is not a valid "
              "primary interval. See this file's docstring.")
    if a.unit == "family":
        print("[WARNING] --unit family has only 3 clusters on this grid "
              "(gaussian_noise / shot_noise / impulse_noise). Do not report as primary.")

    print(f"ImageNet-C gap CIs | rule=exact-rank | calibration={a.calibration} | "
          f"unit={a.unit} | nboot={a.nboot} | seed={a.seed}")
    for cand in a.candidates:
        conds, seeds, rk, ra, rf = load_track(cand, calibration=a.calibration, root=a.root)
        labels = _unit_labels(conds, seeds, a.unit)
        ga = paired_boot(rk - ra, labels, a.nboot, a.seed)   # KGA minus always-adapt
        gf = paired_boot(rk - rf, labels, a.nboot, a.seed)   # KGA minus always-freeze
        print(f"\n{cand.upper()}  (n_rows={len(rk)}, units={ga['units']})")
        print(f"  point regret: KGA={rk.mean():.6f}  adapt={ra.mean():.6f}  freeze={rf.mean():.6f}")
        print(f"  KGA - always-adapt : {ga['obs']:+.6f}  CI95 [{ga['lo']:+.4f}, {ga['hi']:+.4f}]"
              f"  {'EXCLUDES 0' if ga['excludes_zero'] else 'includes 0'}")
        print(f"  KGA - always-freeze: {gf['obs']:+.6f}  CI95 [{gf['lo']:+.4f}, {gf['hi']:+.4f}]"
              f"  {'EXCLUDES 0' if gf['excludes_zero'] else 'includes 0'}")
        bb = ga["excludes_zero"] and gf["excludes_zero"] and ga["hi"] < 0 and gf["hi"] < 0
        print(f"  CI-supported beats-both at unit={a.unit}: {bb}")


if __name__ == "__main__":
    main()
