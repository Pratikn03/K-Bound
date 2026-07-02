#!/usr/bin/env python3
"""
val_anytime_multicandidate.py
==============================
Validator for theory_v2/anytime_multicandidate_theorem.tex (thm:anytime-multicand).

K parallel e-processes; Bonferroni threshold K/alpha; FWER under global harmful null.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "val_anytime_multicandidate_results.json")

ALPHA = 0.1
N_STREAMS = 4000
T = 80
SEED = 20260702
KS = (1, 4, 16, 64)


class EProc:
    def __init__(self, alpha: float, a: float = -1.0, b: float = 1.0, cap: float = 0.5):
        self.a, self.b, self.cap = a, b, cap
        self.lam_max = cap / (-a)
        self.s1 = 0.0
        self.s2 = 0.25
        self.cnt = 0.0
        self.logw = 0.0

    def update(self, x: float) -> float:
        x = float(max(self.a, min(self.b, x)))
        mu = self.s1 / self.cnt if self.cnt > 0 else 0.0
        s2 = self.s2 / max(self.cnt, 1.0)
        lam = float(np.clip(mu / s2 if s2 > 0 else 0.0, 0.0, self.lam_max))
        self.logw += math.log(max(1.0 + lam * x, 1e-300))
        self.s1 += x
        self.s2 += x * x
        self.cnt += 1.0
        return self.logw


def run_stream(K: int, rng: np.random.Generator, positive: bool = False) -> bool:
    thresh = math.log(K / ALPHA)
    procs = [EProc(ALPHA / K) for _ in range(K)]
    for _ in range(T):
        for k in range(K):
            if positive and k == 0:
                x = 0.25 + 0.05 * rng.standard_normal()
            else:
                x = 0.15 * rng.standard_normal()
            if procs[k].update(x) >= thresh:
                return True
    return False


def naive_batch_ever_adapt(K: int, rng: np.random.Generator) -> bool:
    for _ in range(T):
        for _k in range(K):
            x = 0.15 * rng.standard_normal()
            if x > 0.35:
                return True
    return False


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []
    bonf_ok = True
    naive_inflates = True
    power_ok = True
    for K in KS:
        null_hits = sum(run_stream(K, rng, positive=False) for _ in range(N_STREAMS))
        pos_hits = sum(run_stream(K, rng, positive=True) for _ in range(N_STREAMS // 4))
        naive_hits = sum(naive_batch_ever_adapt(K, rng) for _ in range(N_STREAMS // 2))
        fa_rate = null_hits / N_STREAMS
        power = pos_hits / (N_STREAMS // 4)
        naive_rate = naive_hits / (N_STREAMS // 2)
        rows.append({
            "K": K,
            "anytime_falseadapt_rate": fa_rate,
            "power_candidate0_positive": power,
            "naive_per_window_rate": naive_rate,
        })
        bonf_ok = bonf_ok and fa_rate <= ALPHA + 0.015
        if K >= 4:
            naive_inflates = naive_inflates and naive_rate > ALPHA + 0.05
        power_ok = power_ok and power > 0.3

    out = {
        "config": {"alpha": ALPHA, "T": T, "n_streams": N_STREAMS, "Ks": list(KS)},
        "rows": rows,
        "checks": {
            "bonferroni_anytime_FA_le_alpha": bonf_ok,
            "naive_inflates": naive_inflates,
            "power_under_positive_benefit": power_ok,
        },
        "VERDICT": "PASS" if (bonf_ok and naive_inflates and power_ok) else "FAIL",
    }
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["checks"], indent=2))
    for r in rows:
        print(
            f"  K={r['K']}: FA={r['anytime_falseadapt_rate']:.4f} "
            f"naive={r['naive_per_window_rate']:.4f} power={r['power_candidate0_positive']:.3f}"
        )
    print("saved ->", JSON_PATH)
    ok = out["VERDICT"] == "PASS"
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
