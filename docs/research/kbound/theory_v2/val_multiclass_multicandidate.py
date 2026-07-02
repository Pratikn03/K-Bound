#!/usr/bin/env python3
"""
val_multiclass_multicandidate.py
================================
Validator for theory_v2/multiclass_multicandidate_theorem.tex (thm:multiclass-multicand).

Combines multiclass benefit Delta_k = mu_D * (p_a^k - p_0) with Bonferroni routing.
Checks family-wise false-adapt AND false-harm (p_a <= p_0) <= alpha.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "val_multiclass_multicandidate_results.json")

ALPHA = 0.1
N_TRIALS = 50000
SEED = 20260701
KS = (1, 4, 8, 16, 32, 64)
MU_D = 0.35
TAU = 0.12  # noise on benefit estimates


def split_conformal_radius(cal_errs: np.ndarray, level: float) -> float:
    n = len(cal_errs)
    k = int(np.ceil((1 - level) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.sort(np.abs(cal_errs))[k - 1])


def run_for_K(K: int, rng: np.random.Generator) -> dict:
    naive_fa = bonf_fa = 0
    naive_fh = bonf_fh = 0
    naive_commit = bonf_commit = 0

    for _ in range(N_TRIALS):
        harmful = rng.random(K) < 0.5
        p0 = float(rng.uniform(0.45, 0.85))
        pa = np.where(
            harmful,
            p0 - rng.uniform(0.02, 0.18, K),
            p0 + rng.uniform(0.02, 0.22, K),
        )
        pa = np.clip(pa, 0.05, 0.95)
        delta = MU_D * (pa - p0)

        n_cal = 80
        L_naive = []
        L_bonf = []
        for k in range(K):
            dk = float(delta[k])
            cal = dk + TAU * rng.standard_normal(n_cal)
            dep = dk + TAU * rng.standard_normal()
            eps_naive = split_conformal_radius(cal - dk, ALPHA)
            eps_bonf = split_conformal_radius(cal - dk, ALPHA / K)
            L_naive.append(dep - eps_naive)
            L_bonf.append(dep - eps_bonf)
        S_naive = [k for k, L in enumerate(L_naive) if L > 0]
        S_bonf = [k for k, L in enumerate(L_bonf) if L > 0]
        if S_naive:
            naive_commit += 1
            sig = int(max(S_naive, key=lambda k: L_naive[k]))
            if delta[sig] <= 0:
                naive_fa += 1
            if pa[sig] <= p0:
                naive_fh += 1
        if S_bonf:
            bonf_commit += 1
            sig = int(max(S_bonf, key=lambda k: L_bonf[k]))
            if delta[sig] <= 0:
                bonf_fa += 1
            if pa[sig] <= p0:
                bonf_fh += 1

    return {
        "K": K,
        "naive_falseadapt_rate": naive_fa / N_TRIALS,
        "bonf_falseadapt_rate": bonf_fa / N_TRIALS,
        "naive_falseharm_rate": naive_fh / N_TRIALS,
        "bonf_falseharm_rate": bonf_fh / N_TRIALS,
        "naive_commit_rate": naive_commit / N_TRIALS,
        "bonf_commit_rate": bonf_commit / N_TRIALS,
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = [run_for_K(K, rng) for K in KS]
    naive_inflates = any(
        r["naive_falseadapt_rate"] > ALPHA + 0.02 for r in rows if r["K"] >= 8
    )
    bonf_fa_ok = all(r["bonf_falseadapt_rate"] <= ALPHA + 0.012 for r in rows)
    bonf_fh_ok = all(r["bonf_falseharm_rate"] <= ALPHA + 0.012 for r in rows)

    out = {
        "config": {"alpha": ALPHA, "mu_D": MU_D, "n_trials": N_TRIALS, "Ks": list(KS)},
        "rows": rows,
        "checks": {
            "naive_inflates_with_K": naive_inflates,
            "bonferroni_falseadapt_le_alpha": bonf_fa_ok,
            "bonferroni_falseharm_le_alpha": bonf_fh_ok,
            "multiclass_harm_equiv_delta_le_0": True,
        },
        "VERDICT": "PASS" if (naive_inflates and bonf_fa_ok and bonf_fh_ok) else "FAIL",
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["checks"], indent=2))
    print("saved ->", JSON_PATH)
    ok = out["VERDICT"] == "PASS"
    print(f"==== MULTICLASS MULTICANDIDATE: {'PASS' if ok else 'FAIL'} ====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
