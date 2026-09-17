"""Arm A logged re-analysis — jackknife+ radius vs incumbent KGA radius.

Frozen per research_lock/WIN_HUNT_v4_PROTOCOL.yaml (arm_A_jackknife_plus_radius,
bar (ii)). CPU, LOGGED DATA ONLY: no new fitting on held-out labels, identical
splits to the original protocols. The synthetic validator (val_jackknife_plus.py)
must PASS before this is run.

What it does
------------
1. Loads the logged per-condition JSONs for --run-dir with the SAME loader as
   natural_win_analysis.py (records carry B, Z, [Z_ev2], a0, a_adapted, seed,
   condition; per_panel_* collapse supported via --panel).
2. Recomputes, under the ORIGINAL leave-one-seed split:
     - INCUMBENT radius decision = KGA V3 (radius_v2 crossfit_oof + Mondrian signed
       conformal), i.e. the radius layer of the existing pipeline.
     - JACKKNIFE+ radius decision = leave-one-seed CV+ (JackknifePlusGate): for each
       held-out seed, calibrate on the OTHER seeds and decide the held-out cells.
   Only the RADIUS is swapped; the gate/routing layers are out of arm A scope, so
   both policies share the same leave-one-seed cross-fit and the same alpha.
3. Scores regret (vs the per-cell oracle max(a0, a_adapted)) and FA_u for each,
   paired-bootstraps (10^4) the per-cell accuracy improvement (jk+ - incumbent),
   and writes research_lock/WIN_HUNT_v4_ARM_A_<dataset>_result.json.

Bar (ii): FA_u <= alpha AND regret_jk <= regret_incumbent;
CI_ROBUST_IMPROVEMENT iff the regret-gap CI vs the incumbent excludes zero.

SCHEMA-DEFENSIVE: on any missing required field, prints the available keys and
exits 3; fabricates nothing.

Run (from repo root):
  .venv/bin/python docs/research/kbound/gapclose_wave5/rerun_A_jkplus_logged.py \
      --run-dir <dir> --dataset camelyon17 --alpha 0.10
  .venv/bin/python docs/research/kbound/gapclose_wave5/rerun_A_jkplus_logged.py \
      --run-dir <dir> --dataset imagenet-r --panel --alpha 0.10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from radius_v2 import crossfit_oof, mondrian_bounds  # noqa: E402
from radius_jackknife_plus import JackknifePlusGate  # noqa: E402

REQUIRED = ("Z", "B", "a0", "a_adapted", "seed", "condition")


def _die_schema(msg: str, rows=None) -> None:
    print(f"SCHEMA ERROR: {msg}", file=sys.stderr)
    if rows:
        keys = sorted({k for r in rows[:200] for k in r.keys()})
        print(f"available keys: {keys}", file=sys.stderr)
    sys.exit(3)


def load_records(run_dir: str, dataset: str, method, panel: bool):
    if panel:
        pat = os.path.join(run_dir, f"per_panel_{dataset}_seed*.json")
        files = [f for f in sorted(glob.glob(pat))
                 if not os.path.basename(f).startswith("._")]
        rows = []
        for f in files:
            rows.extend(json.load(open(f)).get("records", []))
        if rows:
            return rows
        _die_schema(f"--panel but no per_panel files match {pat}")
    pat = os.path.join(run_dir, f"per_condition_{dataset}_*_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    if method:
        files = [f for f in files if f"_{method}_" in os.path.basename(f)]
    if not files:
        _die_schema(f"no per_condition files match {pat}")
    rows = []
    for f in files:
        rows.extend(json.load(open(f)).get("records", []))
    return rows


def build_arrays(rows):
    if not rows:
        _die_schema("zero records loaded")
    for k in REQUIRED:
        if any(r.get(k) is None for r in rows):
            _die_schema(f"required field '{k}' missing in some records", rows)
    has_ev2 = any(r.get("Z_ev2") for r in rows)
    dim_ev2 = len(next((r["Z_ev2"] for r in rows if r.get("Z_ev2")), [])) if has_ev2 else 0
    Z = np.array([list(map(float, r["Z"]))
                  + (list(map(float, r["Z_ev2"])) if r.get("Z_ev2") else [0.0] * dim_ev2)
                  for r in rows])
    B = np.array([float(r["B"]) for r in rows])
    g = np.array([int(r["seed"]) for r in rows])
    a0 = np.array([float(r["a0"]) for r in rows])
    aa = np.array([float(r["a_adapted"]) for r in rows])
    if np.unique(g).size < 2:
        _die_schema(f"leave-one-seed needs >=2 seeds; got {np.unique(g).tolist()}")
    return Z, B, g, a0, aa, has_ev2


def incumbent_v3(Z, B, g, alpha):
    """KGA V3 radius decision (leave-one-seed): 1=ADAPT, -1=FREEZE, 0=ABSTAIN."""
    Bhat = crossfit_oof(Z, B, g)
    resid = B - Bhat
    dec = np.zeros(B.shape[0], dtype=int)
    for s in np.unique(g):
        cal, te = g != s, g == s
        lo, hi = mondrian_bounds(Bhat[cal], resid[cal], Bhat[te], alpha)
        d = np.zeros(int(te.sum()), dtype=int)
        d[Bhat[te] + lo > 0] = 1
        d[Bhat[te] + hi < 0] = -1
        dec[te] = d
    return dec


def jkplus_leave_one_seed(Z, B, g, alpha):
    """Jackknife+ CV+ decision under leave-one-seed: calibrate on OTHER seeds."""
    dec = np.zeros(B.shape[0], dtype=int)
    for s in np.unique(g):
        te = g == s
        other = ~te
        go = g[other]
        gate = JackknifePlusGate(alpha)
        if np.unique(go).size >= 2:
            gate.fit_grouped(Z[other], B[other], go)
        else:  # only one remaining group -> per-point LOO fallback
            gate.fit(Z[other], B[other])
        codes, _lo, _hi = gate.decide_batch(Z[te])
        dec[te] = codes
    return dec


def paired_bootstrap(diff, nboot, rng):
    n = len(diff)
    idx = rng.integers(0, n, size=(nboot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2.0 * min((means >= 0).mean(), (means <= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(max(p, 1.0 / nboot))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="per-direction level for BOTH radii (KGA operating point)")
    ap.add_argument("--nboot", type=int, default=10000)
    args = ap.parse_args()

    rows = load_records(args.run_dir, args.dataset, args.method, args.panel)
    Z, B, g, a0, aa, has_ev2 = build_arrays(rows)

    dec_inc = incumbent_v3(Z, B, g, args.alpha)
    dec_jk = jkplus_leave_one_seed(Z, B, g, args.alpha)

    oracle = np.maximum(a0, aa)
    acc_inc = np.where(dec_inc == 1, aa, a0)
    acc_jk = np.where(dec_jk == 1, aa, a0)
    reg_inc = float(np.mean(oracle - acc_inc))
    reg_jk = float(np.mean(oracle - acc_jk))
    fa_inc = float(np.mean((dec_inc == 1) & (B <= 0)))
    fa_jk = float(np.mean((dec_jk == 1) & (B <= 0)))

    improvement = acc_jk - acc_inc  # per cell; >0 means jk+ recovers more accuracy
    rng = np.random.default_rng(20260704)
    mean_imp, lo, hi, p = paired_bootstrap(improvement, args.nboot, rng)

    bar_ii_pass = bool(fa_jk <= args.alpha and reg_jk <= reg_inc + 1e-12)
    ci_robust = bool(mean_imp > 0.0 and lo > 0.0 and fa_jk <= args.alpha)

    out = dict(
        protocol="WIN_HUNT_v4", arm="arm_A_jackknife_plus_radius", bar="(ii)",
        dataset=args.dataset, run_dir=args.run_dir, method=args.method,
        scoring_mode="panel" if args.panel else "per_candidate",
        alpha=args.alpha, n_records=int(B.shape[0]),
        n_seeds=int(np.unique(g).size), evidence_dims=int(Z.shape[1]),
        used_ev2=bool(has_ev2),
        incumbent=dict(regret=reg_inc, FA_u=fa_inc,
                       adapt=float((dec_inc == 1).mean()),
                       freeze=float((dec_inc == -1).mean()),
                       abstain=float((dec_inc == 0).mean())),
        jackknife_plus=dict(regret=reg_jk, FA_u=fa_jk,
                            adapt=float((dec_jk == 1).mean()),
                            freeze=float((dec_jk == -1).mean()),
                            abstain=float((dec_jk == 0).mean())),
        regret_improvement=dict(mean=mean_imp, ci=[lo, hi], p=p, nboot=args.nboot),
        bar_ii=dict(FA_u_le_alpha=bool(fa_jk <= args.alpha),
                    regret_le_incumbent=bool(reg_jk <= reg_inc + 1e-12),
                    PASS=bar_ii_pass),
        CI_ROBUST_IMPROVEMENT=ci_robust,
    )
    print(json.dumps(out, indent=1))
    out_dir = os.path.join(REPO, "research_lock")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,
                           f"WIN_HUNT_v4_ARM_A_{args.dataset}_result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
