#!/usr/bin/env python3
"""WIN_HUNT_v3 Arm G — real-data certified willingness-to-pay (TTC Prop 1 demo).

For each logged condition with certificate fields (b_hat, eps_conformal) and a
compute-cost proxy c = upd_norm (normalized to mean 1 per dataset), the
certified spend price is
    lambda*(cond) = max{ lambda >= 0 : b_hat - lambda*c - eps > 0 }  (0 if none)
and at each swept lambda the priced decision SPEND iff b_hat - lambda*c - eps > 0
must keep false-spend FA(lambda) = P(SPEND and B - lambda*c <= 0) <= alpha.

Datasets: stress_grid_multiseed_v1 seeds 0-4 (cifar10c, all methods) and
natural_win_v1_camelyon. Run (CPU, seconds):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_G_lambda_real.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ALPHA = 0.10
LAMS = np.linspace(0.0, 2.0, 41)
SETS = {
    "cifar10c_stress": "experiments/kbound/results/stress_grid_multiseed_v1/seed*/per_condition_cifar10c_*_seed*.json",
    "camelyon17_v1": "experiments/kbound/results/natural_win_v1_camelyon/per_condition_camelyon17_*_seed*.json",
}


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(ROOT / pat))):
        if os.path.basename(f).startswith("._"):
            continue
        d = json.load(open(f))
        for r in d["records"]:
            if r.get("b_hat") is None or r.get("eps_conformal") is None:
                continue
            # cost proxy: update norm if present in Z tail or record
            c = r.get("upd_norm")
            if c is None and isinstance(r.get("Z"), list) and len(r["Z"]) >= 11:
                c = r["Z"][10]  # update_norm is the 11th base evidence dim
            if c is None:
                continue
            rows.append(dict(b_hat=float(r["b_hat"]), eps=float(r["eps_conformal"]),
                             B=float(r["B"]), c=max(float(c), 0.0)))
    return rows


def main() -> int:
    out = {"protocol": "WIN_HUNT_v3_ARM_G",
           "registered": "research_lock/WIN_HUNT_v3_PROTOCOL.yaml",
           "alpha": ALPHA, "datasets": {}}
    ok_all = True
    for name, pat in SETS.items():
        rows = load(pat)
        if not rows:
            print(f"SCHEMA ERROR: no usable records for {name} ({pat})",
                  file=sys.stderr)
            return 3
        b = np.array([r["b_hat"] for r in rows])
        e = np.array([r["eps"] for r in rows])
        B = np.array([r["B"] for r in rows])
        c = np.array([r["c"] for r in rows])
        c = c / max(c.mean(), 1e-9)  # normalize cost to mean 1
        lam_star = np.where(b - e > 0, (b - e) / np.maximum(c, 1e-9), 0.0)
        fa_curve, spend_curve = [], []
        valid = True
        for lam in LAMS:
            spend = (b - lam * c - e) > 0
            fa = float(np.mean(spend & (B - lam * c <= 0)))
            fa_curve.append(round(fa, 4))
            spend_curve.append(round(float(spend.mean()), 4))
            valid &= fa <= ALPHA
        ok_all &= valid
        out["datasets"][name] = dict(
            n_records=len(rows),
            lambda_star=dict(mean=float(lam_star.mean()),
                             median=float(np.median(lam_star)),
                             q90=float(np.quantile(lam_star, 0.9)),
                             frac_zero=float((lam_star == 0).mean())),
            lambda_grid=[round(float(x), 3) for x in LAMS],
            spend_rate=spend_curve, false_spend=fa_curve,
            FA_valid_all_lambda=bool(valid))
    out["PASS_priced_validity"] = bool(ok_all)
    print(json.dumps(out, indent=1))
    p = ROOT / "research_lock/WIN_HUNT_v3_ARM_G_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
