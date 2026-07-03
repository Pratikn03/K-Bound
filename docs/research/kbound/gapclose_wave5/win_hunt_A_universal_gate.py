#!/usr/bin/env python3
"""WIN_HUNT_v2 Arm A — ONE universal gate on the pooled natural deployment.

Pre-registered in research_lock/WIN_HUNT_v2_PROTOCOL.yaml. Data, splits and
candidates are IDENTICAL to mixed_stream_kbound.py (mixed_protocol_oof_v2);
the single change under test: one GBR benefit model + one LOO-conformal radius
fit on the POOLED dev records of all three datasets over the SHARED base-11
evidence dims, applied once to the pooled held-out test conditions.

Run (CPU, seconds):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_A_universal_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402  (fit_point, decide_global, ALPHA, load_records)
import mixed_stream_kbound as ms  # noqa: E402  (DATASETS, load — reuse verbatim)

BASE_DIMS = 11  # shared EVIDENCE_NAMES prefix across all WILDS runners
NBOOT = 5000


def base_z(recs):
    return np.array([list(map(float, r["Z"][:BASE_DIMS])) for r in recs])


def main() -> int:
    rng = np.random.default_rng(13)
    cal_all, test_all, test_meta = [], [], []
    for ds in ms.DATASETS:
        cal, test = ms.load(ds)
        for r in cal:
            if len(r["Z"]) < BASE_DIMS:
                print(f"SCHEMA ERROR: {ds['name']} record Z dim {len(r['Z'])} < {BASE_DIMS}",
                      file=sys.stderr)
                return 3
        cal_all.extend(cal)
        test_all.extend(test)
        test_meta.extend([ds["name"]] * len(test))
        print(f"loaded {ds['name']:11s} cal={len(cal):3d} test={len(test):3d}")

    Zc, Bc = base_z(cal_all), np.array([float(r["B"]) for r in cal_all])
    Zt = base_z(test_all)
    Bt = np.array([float(r["B"]) for r in test_all])
    a0 = np.array([float(r["a0"]) for r in test_all])
    aa = np.array([float(r["aa"]) for r in test_all])

    # ONE gate: pooled fit + pooled LOO conformal radius (analyze_F recipe)
    m = af.fit_point(Zc, Bc)
    loo = np.empty(len(Bc))
    for i in range(len(Bc)):
        tr = np.arange(len(Bc)) != i
        loo[i] = af.fit_point(Zc[tr], Bc[tr]).predict(Zc[i:i + 1])[0]
    eps = float(np.quantile(np.abs(loo - Bc), 1 - af.ALPHA))
    dec = af.decide_global(m.predict(Zt), eps)
    adapt = dec == "ADAPT"

    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    rk, ra, rf = oracle - kga, oracle - aa, oracle - a0
    fa = float((Bt[adapt] <= 0).mean()) if adapt.any() else 0.0

    gap_f, gap_a = rf - rk, ra - rk
    idx = rng.integers(0, len(rk), (NBOOT, len(rk)))
    cf, ca = gap_f[idx].mean(1), gap_a[idx].mean(1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731

    win = (rk.mean() < ra.mean()) and (rk.mean() < rf.mean()) and fa <= af.ALPHA
    ci_robust = win and np.percentile(cf, 2.5) > 0 and np.percentile(ca, 2.5) > 0
    verdict = "CI_ROBUST_WIN" if ci_robust else ("WIN" if win else "NO_WIN")

    out = dict(
        protocol="WIN_HUNT_v2_ARM_A", registered="research_lock/WIN_HUNT_v2_PROTOCOL.yaml",
        gate="single pooled GBR + single pooled LOO conformal radius, base-11 Z",
        n_conditions=int(len(rk)),
        composition={n: int(sum(1 for t in test_meta if t == n))
                     for n in dict.fromkeys(test_meta)},
        alpha=af.ALPHA, eps_universal=eps, false_adapt=fa,
        adapt_rate=float(adapt.mean()),
        regret_kga=float(rk.mean()), regret_adapt=float(ra.mean()),
        regret_freeze=float(rf.mean()),
        kga_vs_freeze=dict(mean=float(gap_f.mean()), ci95=ci(cf)),
        kga_vs_adapt=dict(mean=float(gap_a.mean()), ci95=ci(ca)),
        per_dataset={
            n: dict(regret_kga=float(rk[[t == n for t in test_meta]].mean()),
                    regret_adapt=float(ra[[t == n for t in test_meta]].mean()),
                    regret_freeze=float(rf[[t == n for t in test_meta]].mean()))
            for n in dict.fromkeys(test_meta)},
        VERDICT=verdict,
    )
    print(json.dumps(out, indent=1))
    p = ROOT / "research_lock/WIN_HUNT_v2_ARM_A_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
