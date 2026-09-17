#!/usr/bin/env python3
"""WIN_HUNT_v2 Arm C — certified per-candidate selection on the 10-backbone panel.

Pre-registered in research_lock/WIN_HUNT_v2_PROTOCOL.yaml. Consumes the logged
natural_win_v1_imagenetr per-candidate files (10 backbones x 3 seeds, Wave-5
schema). Located failures fixed:
  (a) selection now uses per-candidate BENEFIT certificates — leave-one-seed
      cross-fitted GBR per backbone + SIGNED conformal lower bound at alpha/K
      (K=10, Bonferroni; the paper's multicandidate theorem) — commit set
      {b : LCB_b > 0}, choose argmax B-hat within it, else freeze;
  (b) baselines are IMPLEMENTABLE: always-freeze, and the single backbone with
      the best mean accuracy on the OTHER seeds (deployed unchanged).
Oracle (per-cell best) reported as ceiling only, not a bar.

Run (CPU, ~1 min):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_C_panel_select.py \
      --run-dir experiments/kbound/results/natural_win_v1_imagenetr
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).resolve().parents[3]
sys.path.insert(0, HERE)
from radius_v2 import crossfit_oof, rank_quantile  # noqa: E402

ALPHA = 0.10
NBOOT = 5000
BACKBONES = ["resnet101", "resnet152", "resnext101_32x8d", "efficientnet_b0",
             "efficientnet_b3", "convnext_tiny", "convnext_base", "vit_b_16",
             "swin_t", "swin_b"]


def load(run_dir: str, dataset: str = "imagenet-r"):
    pat = os.path.join(run_dir, f"per_condition_{dataset}_*_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    if not files:
        print(f"SCHEMA ERROR: no files match {pat}", file=sys.stderr)
        sys.exit(3)
    cells: dict[tuple, dict] = defaultdict(dict)
    for f in files:
        d = json.load(open(f))
        b = d["method"]
        for r in d["records"]:
            cells[(int(r["seed"]), r["condition"])][b] = r
    rows = []
    for (seed, cond), by_b in sorted(cells.items()):
        missing = [b for b in BACKBONES if b not in by_b]
        if missing:
            print(f"SCHEMA ERROR: cell {seed}/{cond} missing {missing}",
                  file=sys.stderr)
            sys.exit(3)
        rows.append(dict(seed=seed, cond=cond,
                         a0=float(by_b[BACKBONES[0]]["a0"]),
                         aa={b: float(by_b[b]["a_adapted"]) for b in BACKBONES},
                         Z={b: list(map(float, by_b[b]["Z"])) for b in BACKBONES},
                         B={b: float(by_b[b]["B"]) for b in BACKBONES}))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(
        ROOT / "experiments/kbound/results/natural_win_v1_imagenetr"))
    args = ap.parse_args()

    rows = load(args.run_dir)
    seeds = sorted({r["seed"] for r in rows})
    n = len(rows)
    K = len(BACKBONES)
    a_bonf = ALPHA / K
    print(f"cells={n} seeds={seeds} K={K} alpha={ALPHA} (per-candidate {a_bonf:.4f})")

    # per-candidate cross-fitted benefit + signed Bonferroni lower bound
    g = np.array([r["seed"] for r in rows])
    lcb = np.zeros((n, K))
    bhat = np.zeros((n, K))
    for j, b in enumerate(BACKBONES):
        Z = np.array([r["Z"][b] for r in rows])
        B = np.array([r["B"][b] for r in rows])
        pred = crossfit_oof(Z, B, g)
        bhat[:, j] = pred
        resid = B - pred
        for s in seeds:
            cal, te = g != s, g == s
            q_lo = rank_quantile(resid[cal], a_bonf)   # signed lower quantile
            lcb[te, j] = pred[te] + q_lo

    # decisions: commit set {LCB>0}; argmax bhat within it; else freeze
    a0 = np.array([r["a0"] for r in rows])
    aa = np.array([[r["aa"][b] for b in BACKBONES] for r in rows])
    Btrue = np.array([[r["B"][b] for b in BACKBONES] for r in rows])
    committed = lcb > 0
    choice = np.full(n, -1)
    for i in range(n):
        js = np.where(committed[i])[0]
        if len(js):
            choice[i] = js[np.argmax(bhat[i, js])]
    acc_sel = np.where(choice >= 0, aa[np.arange(n), np.clip(choice, 0, K - 1)], a0)
    fa = float(np.mean((choice >= 0)
                       & (Btrue[np.arange(n), np.clip(choice, 0, K - 1)] <= 0)))

    # implementable baselines
    acc_freeze = a0
    acc_bestfix = np.empty(n)
    bestfix_name = {}
    for s in seeds:
        cal, te = g != s, g == s
        jbest = int(np.argmax(aa[cal].mean(axis=0)))
        bestfix_name[int(s)] = BACKBONES[jbest]
        acc_bestfix[te] = aa[te, jbest]
    oracle = np.maximum(a0, aa.max(axis=1))

    rk = oracle - acc_sel
    rf = oracle - acc_freeze
    rb = oracle - acc_bestfix
    rng = np.random.default_rng(20260703)
    idx = rng.integers(0, n, (NBOOT, n))
    gf, gb = (rf - rk)[idx].mean(1), (rb - rk)[idx].mean(1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731

    win = (rk.mean() < rf.mean()) and (rk.mean() < rb.mean()) and fa <= ALPHA
    ci_robust = win and np.percentile(gf, 2.5) > 0 and np.percentile(gb, 2.5) > 0
    verdict = "CI_ROBUST_WIN" if ci_robust else ("WIN" if win else "NO_WIN")

    out = dict(
        protocol="WIN_HUNT_v2_ARM_C",
        registered="research_lock/WIN_HUNT_v2_PROTOCOL.yaml",
        run_dir=args.run_dir, n_cells=n, K=K, alpha=ALPHA,
        bonferroni_alpha=a_bonf, false_adapt=fa,
        commit_rate=float((choice >= 0).mean()),
        best_fixed_backbone_by_heldout_seed=bestfix_name,
        regret=dict(certified_select=float(rk.mean()),
                    always_freeze=float(rf.mean()),
                    best_fixed_backbone=float(rb.mean()),
                    oracle_ceiling=0.0),
        vs_freeze=dict(mean=float((rf - rk).mean()), ci95=ci(gf)),
        vs_best_fixed=dict(mean=float((rb - rk).mean()), ci95=ci(gb)),
        VERDICT=verdict,
    )
    print(json.dumps(out, indent=1))
    p = Path(ROOT) / "research_lock/WIN_HUNT_v2_ARM_C_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
