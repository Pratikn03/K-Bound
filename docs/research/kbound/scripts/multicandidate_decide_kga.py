#!/usr/bin/env python3
"""Leave-one-out GBR + Bonferroni multicandidate routing for stress-grid conditions.

Mirrors ``decide_kga`` in ``cifar_tent_mps_v2.py`` but selects among K adapter
candidates per condition with family-wise false-adapt control (``thm:multicand``,
``thm:multiclass-multicand``).

Usage (synthetic self-test, no torch):
    python docs/research/kbound/scripts/multicandidate_decide_kga.py --selftest

Usage (from locked per-candidate benefits):
    python multicandidate_decide_kga.py --benefits-npz path.npz --alpha 0.1
    # npz keys: Z (n,d), B (K,n) per-candidate benefits, optional names (K,)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from kga.routing import route_panel  # noqa: E402


def loo_gbr_predict(Z: np.ndarray, B: np.ndarray, seed: int = 0) -> np.ndarray:
    """LOO benefit predictions for one candidate across n conditions."""
    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float).ravel()
    n = B.size
    out = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        m = GradientBoostingRegressor(
            n_estimators=250, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed
        )
        m.fit(Z[tr], B[tr])
        out[i] = m.predict(Z[i : i + 1])[0]
    return out


def decide_multicandidate_panel(
    Z: np.ndarray,
    B_candidates: np.ndarray,
    *,
    alpha: float = 0.1,
    seed: int = 0,
) -> dict:
    """Full panel: LOO GBR per candidate + Bonferroni route at each condition."""
    Z = np.asarray(Z, dtype=float)
    B_candidates = np.asarray(B_candidates, dtype=float)
    if B_candidates.ndim != 2:
        raise ValueError("B_candidates must be shape (K, n_conditions)")
    k, n = B_candidates.shape
    if Z.shape[0] != n:
        raise ValueError("Z rows must match B_candidates columns")

    deploy = np.zeros(k)
    cal_scores = np.zeros((k, n))
    cal_truth = B_candidates.copy()
    for j in range(k):
        bhat = loo_gbr_predict(Z, B_candidates[j], seed=seed)
        cal_scores[j] = bhat
        deploy[j] = bhat[-1] if n > 0 else 0.0

    # Deploy on last condition index (held-out style): use LOO preds as scores.
    deploy_scores = np.array([cal_scores[j, -1] for j in range(k)])
    # Calibration = all but last
    if n < 3:
        raise ValueError("need at least 3 conditions for multicandidate LOO calibration")
    cal_scores = cal_scores[:, :-1]
    cal_truth = cal_truth[:, :-1]

    decision = route_panel(deploy_scores, cal_scores, cal_truth, alpha=alpha)
    return {
        "alpha": alpha,
        "K": k,
        "selected": decision.selected,
        "decision": decision.decision,
        "bonferroni_alpha": decision.bonferroni_alpha,
        "lcbs": [c.lcb for c in decision.certificates],
        "delta_hats": [c.delta_hat for c in decision.certificates],
        "epsilons": [c.epsilon for c in decision.certificates],
    }


def selftest() -> int:
    rng = np.random.default_rng(20260701)
    alpha = 0.1
    n_cond = 40
    k = 8
    Z = rng.standard_normal((n_cond, 6))
    B = rng.uniform(-0.1, 0.35, (k, n_cond))
    B[0] += 0.15  # one clearly helpful candidate
    out = decide_multicandidate_panel(Z, B, alpha=alpha)
    ok = out["decision"] in ("adapt", "abstain") and out["K"] == k
    print(json.dumps(out, indent=2))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benefits-npz", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.benefits_npz is None:
        ap.error("provide --benefits-npz or --selftest")
    data = np.load(args.benefits_npz)
    Z = data["Z"]
    B = data["B"]
    out = decide_multicandidate_panel(Z, B, alpha=args.alpha, seed=args.seed)
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
