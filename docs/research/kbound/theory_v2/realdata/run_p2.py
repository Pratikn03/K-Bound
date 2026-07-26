"""
P2: per-condition paired bootstrap + Holm on the REAL CIFAR-10-C 65-cell grid.

Replaces the flagged synthetic-stream Pareto bootstrap (decisive_tta_cis.json,
ci_source='pareto_bootstrap_curve') with a genuine per-condition paired test on
cifar10c_65cells.csv (per-cell accuracy for frozen/tent/eata/sar/kga/oracle).

Estimand per method m in {tent,eata,sar}: regret_m(cell) = acc_oracle - acc_m.
KGA regret = acc_oracle - acc_kga (KGA picks adapt-or-freeze per cell via the
gamma/agreement rule). Headline comparisons:
   KGA vs always-adapt (= the TTA method run on every cell)
   KGA vs always-freeze (= frozen accuracy on every cell)
Paired difference per cell: d(cell) = regret_KGA(cell) - regret_baseline(cell)
                                    = acc_baseline(cell) - acc_kga(cell)  (oracle cancels)
Negative mean d => KGA has lower regret (better).

Per-condition paired bootstrap (resample the 65 cells with replacement, recompute
mean d) gives a BCa-free percentile CI; paired t over cells gives p; Holm corrects
across the family of comparisons. We report which headline comparisons survive Holm.
"""
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import csv, json, os
import numpy as np
from scipy import stats

CSV = KB_REPO_ROOT + "/experiments/kbound/results/cifar10c_65cells.csv"
RNG = np.random.default_rng(42)
NBOOT = 10000
ALPHA = 0.05

rows = list(csv.DictReader(open(CSV)))
cols = {k: np.array([float(r[k]) for r in rows]) for k in
        ["frozen", "tent", "eata", "sar", "kga", "oracle"]}
ncell = len(rows)
oracle = cols["oracle"]; kga = cols["kga"]; frozen = cols["frozen"]

# Per-cell regrets
reg = {m: oracle - cols[m] for m in ["frozen", "tent", "eata", "sar", "kga"]}

def paired_bootstrap(d, nboot=NBOOT, alpha=ALPHA):
    """d = per-cell paired differences (regret_KGA - regret_baseline). Resample cells."""
    n = len(d)
    means = np.empty(nboot)
    idxs = RNG.integers(0, n, size=(nboot, n))
    for b in range(nboot):
        means[b] = d[idxs[b]].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    obs = float(d.mean())
    # two-sided bootstrap p (proportion of resamples on the other side of 0, x2)
    p_boot = 2 * min((means >= 0).mean(), (means <= 0).mean())
    p_boot = min(1.0, p_boot)
    return obs, float(lo), float(hi), float(p_boot)

def holm(pvals, names, alpha=ALPHA):
    order = np.argsort(pvals)
    m = len(pvals)
    out = {}
    reject_so_far = True
    for rank, idx in enumerate(order):
        thr = alpha / (m - rank)
        rej = bool(pvals[idx] < thr) and reject_so_far
        if not rej:
            reject_so_far = False
        out[names[idx]] = {"p": float(pvals[idx]), "holm_threshold": float(thr),
                           "reject_holm": rej}
    return out

# Build the comparison family: KGA vs always-adapt(method) and KGA vs always-freeze,
# for each of the 3 TTA methods => 6 comparisons (matches the paper's Table 7 family).
comparisons = []
for meth in ["tent", "eata", "sar"]:
    # KGA-under-this-method: the 65-cell CSV has a single 'kga' column (the realized KGA
    # decisions). KGA chose per-cell between this method's adapted acc and frozen acc.
    # vs always-adapt = this TTA method on all cells:
    d_adapt = reg["kga"] - reg[meth]      # = acc_meth - acc_kga
    comparisons.append((f"{meth}: KGA vs always-adapt", d_adapt))
    # vs always-freeze = frozen on all cells:
    d_freeze = reg["kga"] - reg["frozen"]  # = acc_frozen - acc_kga
    comparisons.append((f"{meth}: KGA vs always-freeze", d_freeze))

res = {}
pvals = []; names = []
for name, d in comparisons:
    obs, lo, hi, pb = paired_bootstrap(d)
    t, pt = stats.ttest_rel(reg["kga"], reg["kga"] - d)  # paired t on the two regret arrays
    # equivalently ttest_1samp on d:
    t1, pt1 = stats.ttest_1samp(d, 0.0)
    res[name] = {
        "n_cells": ncell,
        "mean_regret_KGA": round(float(reg["kga"].mean()), 5),
        "mean_diff_KGA_minus_baseline": round(obs, 5),
        "boot95_CI": [round(lo, 5), round(hi, 5)],
        "p_boot": round(pb, 5),
        "p_paired_t": round(float(pt1), 5),
        "t_stat": round(float(t1), 3),
        "direction": "KGA better" if obs < 0 else ("tie" if abs(obs) < 1e-9 else "KGA worse"),
    }
    pvals.append(float(pt1)); names.append(name)

holm_res = holm(np.array(pvals), names, ALPHA)
for name in res:
    res[name]["holm"] = holm_res[name]

# also raw per-method aggregate regrets for context
agg = {m: round(float(reg[m].mean()), 5) for m in ["frozen", "tent", "eata", "sar", "kga"]}

out = {
    "P2": {
        "source_csv": CSV, "n_cells": ncell, "n_boot": NBOOT, "alpha": ALPHA,
        "method": ("per-condition paired bootstrap over the 65 real CIFAR-10-C cells "
                   "(corruption x severity); Holm across the 6-comparison family. "
                   "Replaces decisive_tta_cis.json pareto_bootstrap_curve."),
        "mean_regret_to_oracle_per_method": agg,
        "comparisons": res,
        "n_survive_holm": int(sum(1 for v in holm_res.values() if v["reject_holm"])),
        "survivors_holm": [k for k, v in holm_res.items() if v["reject_holm"]],
    }
}
print(json.dumps(out, indent=2))
json.dump(out, open(KB_REPO_ROOT + "/docs/research/kbound/theory_v2/realdata/_p2_partial.json", "w"), indent=2)
print("\nwrote _p2_partial.json")
