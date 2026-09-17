#!/usr/bin/env python3
"""WIN_HUNT_v3 Arm E — one universal gate across SEVEN natural sources.

Pre-registered in research_lock/WIN_HUNT_v3_PROTOCOL.yaml. Identical recipe to
Arm A (single pooled GBR + single pooled LOO conformal radius, shared base-11 Z)
with four PACS leave-one-domain percell sources added:
  cal = repeat r0 conditions, test = repeat r1 (frozen in the protocol).

Run (CPU, ~1-2 min):
  python3 docs/research/kbound/gapclose_wave5/win_hunt_E_universal7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # noqa: E402
import mixed_stream_kbound as ms  # noqa: E402

BASE_DIMS = 11
NBOOT = 5000
PACS = ["art_painting", "cartoon", "photo", "sketch"]


def base_z(recs):
    return np.array([list(map(float, r["Z"][:BASE_DIMS])) for r in recs])


def load_pacs(domain: str):
    p = ROOT / f"docs/research/kbound/pacs_{domain}_percell.json"
    if not p.exists():
        print(f"SCHEMA ERROR: missing {p}", file=sys.stderr)
        sys.exit(3)
    recs = json.load(open(p))
    cal, test = [], []
    for r in recs:
        if len(r.get("Z", [])) < BASE_DIMS or "B" not in r:
            print(f"SCHEMA ERROR: bad PACS record in {p}", file=sys.stderr)
            sys.exit(3)
        row = dict(Z=r["Z"], B=float(r["B"]), a0=float(r["a0"]),
                   aa=float(r["aa"]))
        if "|r0|" in r["condition"]:
            cal.append(row)
        elif "|r1|" in r["condition"]:
            test.append(row)
    if not cal or not test:
        print(f"SCHEMA ERROR: PACS {domain} r0/r1 split empty", file=sys.stderr)
        sys.exit(3)
    return cal, test


def main() -> int:
    rng = np.random.default_rng(13)
    cal_all, test_all, meta = [], [], []
    for ds in ms.DATASETS:
        cal, test = ms.load(ds)
        cal_all.extend(cal); test_all.extend(test)
        meta.extend([ds["name"]] * len(test))
        print(f"loaded {ds['name']:12s} cal={len(cal):3d} test={len(test):3d}")
    for dom in PACS:
        cal, test = load_pacs(dom)
        cal_all.extend(cal); test_all.extend(test)
        meta.extend([f"PACS_{dom}"] * len(test))
        print(f"loaded PACS_{dom:11s} cal={len(cal):3d} test={len(test):3d}")

    Zc, Bc = base_z(cal_all), np.array([float(r["B"]) for r in cal_all])
    Zt = base_z(test_all)
    Bt = np.array([float(r["B"]) for r in test_all])
    a0 = np.array([float(r["a0"]) for r in test_all])
    aa = np.array([float(r["aa"]) for r in test_all])

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

    win = rk.mean() < ra.mean() and rk.mean() < rf.mean() and fa <= af.ALPHA
    ci_robust = win and np.percentile(cf, 2.5) > 0 and np.percentile(ca, 2.5) > 0
    verdict = "CI_ROBUST_WIN" if ci_robust else ("WIN" if win else "NO_WIN")

    names = list(dict.fromkeys(meta))
    out = dict(protocol="WIN_HUNT_v3_ARM_E",
               registered="research_lock/WIN_HUNT_v3_PROTOCOL.yaml",
               n_conditions=int(len(rk)),
               composition={n: int(sum(1 for t in meta if t == n)) for n in names},
               alpha=af.ALPHA, eps_universal=eps, false_adapt=fa,
               adapt_rate=float(adapt.mean()),
               regret_kga=float(rk.mean()), regret_adapt=float(ra.mean()),
               regret_freeze=float(rf.mean()),
               kga_vs_freeze=dict(mean=float(gap_f.mean()), ci95=ci(cf)),
               kga_vs_adapt=dict(mean=float(gap_a.mean()), ci95=ci(ca)),
               per_source={n: dict(
                   regret_kga=float(rk[[t == n for t in meta]].mean()),
                   regret_adapt=float(ra[[t == n for t in meta]].mean()),
                   regret_freeze=float(rf[[t == n for t in meta]].mean()))
                   for n in names},
               VERDICT=verdict)
    print(json.dumps(out, indent=1))
    p = ROOT / "research_lock/WIN_HUNT_v3_ARM_E_result.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"saved {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
