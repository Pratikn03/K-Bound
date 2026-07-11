"""Validator — WIN_HUNT_v4 arm C (drift-conditioned tau'). Frozen bars.

C1 level <= 0.07 at nominal alpha=0.05 on INDEPENDENT H-true panels, >= 480 reps
   stratified across 3 drift strata (weak / mid / strong agreement).
C2 power >= 0.9 on co-adapted twin-agreement panels.

Generators are REUSED (imported), not duplicated:
   simulate_H_panel  <- tau_selfnorm     (H-true / level panels)
   coadapted_panel   <- val_gapB_tau     (co-adapted / power panels)
Tercile thresholds are frozen on a DISJOINT dev set; level/power panels are
drawn independently. --quick relaxes reps for iteration (NOT the registered
verdict: the >=480-rep bar requires the default full run).

Exit 0 iff PASS. JSON verdict -> research_lock/WIN_HUNT_v4_ARM_C_validator.json.
Seeds fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from tau_selfnorm import simulate_H_panel  # noqa: E402
from val_gapB_tau import coadapted_panel  # noqa: E402
from tau_adaptive import calibrate_dev_terciles, tau_adaptive  # noqa: E402

ALPHA = 0.05
K_PANEL = 6
DRIFT_BS = (0.2, 0.5, 0.8)  # 3 drift strata via label-free agreement scale


def h_panel(bs, m, rng):
    b = bs * (0.6 + 0.4 * rng.random(K_PANEL))
    return simulate_H_panel(b, 0.5, m, rng)


def build_calib(n_per_stratum, m, n_sim, seed):
    rng = np.random.default_rng(seed)
    Cs, ms = [], []
    for bs in DRIFT_BS:
        for _ in range(n_per_stratum):
            Cs.append(h_panel(bs, m, rng))
            ms.append(m)
    return calibrate_dev_terciles(Cs, ms, ALPHA, n_sim=n_sim, seed=seed + 1)


def run_level(calib, reps_per, m, n_sim, seed):
    rng = np.random.default_rng(seed)
    by, total_rej, total = {}, 0, 0
    for bs in DRIFT_BS:
        rej = 0
        for _ in range(reps_per):
            C = h_panel(bs, m, rng)
            r = tau_adaptive(C, m, alpha=ALPHA, n_sim=n_sim,
                             seed=int(rng.integers(1 << 30)), calib=calib)
            rej += int(r["reject_H"])
        by[f"bs{bs}"] = rej / reps_per
        total_rej += rej
        total += reps_per
    return by, total_rej / total, total


def run_power(calib, reps, m, n_sim, seed):
    rng = np.random.default_rng(seed)
    pw = 0
    for _ in range(reps):
        C = coadapted_panel(K_PANEL, m, 0.25, 0.5, rng)
        r = tau_adaptive(C, m, alpha=ALPHA, n_sim=n_sim,
                         seed=int(rng.integers(1 << 30)), calib=calib)
        pw += int(r["reject_H"])
    return pw / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--m", type=int, default=1500)
    args = ap.parse_args()
    quick = args.quick
    reps_per = 80 if quick else 160          # per stratum (3*160 = 480 full)
    pow_reps = 60 if quick else 120
    dev_per = 25 if quick else 40
    n_sim = 100 if quick else 130
    m = args.m
    m_pow = 2000  # proven power regime (val_gapB_tau)

    calib = build_calib(dev_per, m, n_sim, seed=7)
    level_by, level, n_level = run_level(calib, reps_per, m, n_sim, seed=101)
    power = run_power(calib, pow_reps, m_pow, n_sim, seed=202)

    c1 = (level <= 0.07) and (quick or n_level >= 480)
    c2 = power >= 0.9
    out = dict(protocol="WIN_HUNT_v4_ARM_C_validator",
               registered="research_lock/WIN_HUNT_v4_PROTOCOL.yaml",
               alpha=ALPHA, quick=quick, K=K_PANEL,
               calib=dict(cuts=calib["cuts"], mult=calib["mult"],
                          n_dev=calib["n_dev"]),
               level=dict(overall=level, n_reps=n_level, by_stratum=level_by,
                          bar_le_0p07=bool(level <= 0.07),
                          reps_ge_480=bool(n_level >= 480)),
               power=dict(coadapted=power, m=m_pow, n_reps=pow_reps,
                          bar_ge_0p9=bool(power >= 0.9)),
               checks=dict(C1_level=bool(c1), C2_power=bool(c2)),
               PASS=bool(c1 and c2))
    print(json.dumps(out, indent=1))
    with open(os.path.join(REPO, "research_lock",
                           "WIN_HUNT_v4_ARM_C_validator.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if out["PASS"] else 3


if __name__ == "__main__":
    sys.exit(main())
