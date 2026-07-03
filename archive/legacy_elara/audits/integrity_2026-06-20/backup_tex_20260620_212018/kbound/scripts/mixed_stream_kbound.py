#!/usr/bin/env python3
"""Mixed-stream (heterogeneous deployment) evaluation for KGA.

Thesis under test: under a deployment that MIXES adapt-favorable and
freeze-favorable shift, NO single global policy (always-adapt / always-freeze)
can be near-oracle, but KGA -- which decides per condition from label-free
evidence with false-adapt control -- is. We pool the held-out TEST conditions of
three real datasets:
  * Camelyon17  (adapt-favorable: freezing leaves large regret)
  * OfficeHome  (freeze-favorable: adapting often hurts)
  * iWildCam    (freeze-favorable: adapting often hurts)
Each dataset's KGA gate is the SAME locked recipe (GBR benefit estimator + global
conformal eps, alpha=0.10), fit on that dataset's DEV seeds and applied ONCE to
its TEST seeds -- identical to the locked single-dataset protocols. We then pool
the per-condition realized regrets and bootstrap the pooled regret gaps.

No new GPU passes; no recalibration on test; per-dataset means reproduce the
locked protocol_result.json to 4 decimals (asserted below).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/research/kbound/scripts"))
import analyze_F as af  # fit_point, decide_global, ALPHA, load_records


def filt(recs, seeds):
    s = set(seeds)
    return [r for r in recs if int(r["seed"]) in s]


def per_condition(cal, test):
    """Return per-condition regret triples + benefit, faithful to analyze_F.metrics."""
    Zc = np.array([r["Z"] for r in cal], float); Bc = np.array([r["B"] for r in cal], float)
    Zt = np.array([r["Z"] for r in test], float)
    Bt = np.array([r["B"] for r in test], float)
    a0 = np.array([r["a0"] for r in test], float); aa = np.array([r["aa"] for r in test], float)
    m = af.fit_point(Zc, Bc)
    eps = float(np.quantile(np.abs(m.predict(Zc) - Bc), 1 - af.ALPHA))
    dec = af.decide_global(m.predict(Zt), eps)
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    return dict(reg_kga=oracle - kga, reg_adapt=oracle - aa, reg_freeze=oracle - a0,
                B=Bt, adapt=adapt, n=len(test))


# (dataset, records_or_cal/test, candidate, dev_seeds, test_seeds, stored_regret_kga)
DATASETS = [
    dict(name="Camelyon17", kind="single",
         rec="experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json",
         cand="eata_online", dev=[0, 1], test=[2, 3, 4], favors="adapt"),
    dict(name="OfficeHome", kind="transfer",
         cal="experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json",
         test="experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json",
         cand="sar_online_aggressive", dev=[0, 1], test_seeds=[0, 1], favors="freeze"),
    dict(name="iWildCam", kind="single",
         rec="experiments/kbound/results/iwildcam_full_test/result_e40faf29.json",
         cand="tent_episodic", dev=[0], test=[1], favors="freeze"),
]


def load(ds):
    if ds["kind"] == "transfer":
        cal = filt(af.load_records(str(ROOT / ds["cal"]), candidate=ds["cand"])[0], ds["dev"])
        test = filt(af.load_records(str(ROOT / ds["test"]), candidate=ds["cand"])[0], ds["test_seeds"])
    else:
        recs = af.load_records(str(ROOT / ds["rec"]), candidate=ds["cand"])[0]
        cal = filt(recs, ds["dev"]); test = filt(recs, ds["test"])
    return cal, test


def main():
    rng = np.random.default_rng(13)
    parts = {}
    print("per-dataset (means must match locked protocol):")
    for ds in DATASETS:
        cal, test = load(ds)
        pc = per_condition(cal, test)
        parts[ds["name"]] = pc
        fa = float((pc["B"][pc["adapt"]] < 0).mean()) if pc["adapt"].any() else 0.0
        print(f"  {ds['name']:11s} favors={ds['favors']:6s} n={pc['n']:3d} "
              f"regret kga={pc['reg_kga'].mean():.4f} adapt={pc['reg_adapt'].mean():.4f} "
              f"freeze={pc['reg_freeze'].mean():.4f}  FA={fa:.3f}")
    # pool
    rk = np.concatenate([p["reg_kga"] for p in parts.values()])
    ra = np.concatenate([p["reg_adapt"] for p in parts.values()])
    rf = np.concatenate([p["reg_freeze"] for p in parts.values()])
    Bp = np.concatenate([p["B"] for p in parts.values()])
    adp = np.concatenate([p["adapt"] for p in parts.values()])
    n = len(rk)
    fa_pool = float((Bp[adp] < 0).mean()) if adp.any() else 0.0
    gap_f = rf - rk; gap_a = ra - rk
    B = 5000; idx = rng.integers(0, n, (B, n))
    cf = gap_f[idx].mean(1); ca = gap_a[idx].mean(1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = dict(
        stream="pooled heterogeneous deployment (Camelyon17 + OfficeHome + iWildCam test conditions)",
        n_conditions=int(n), composition={k: int(p["n"]) for k, p in parts.items()},
        alpha=af.ALPHA, false_adapt=fa_pool,
        regret_kga=float(rk.mean()), regret_adapt=float(ra.mean()), regret_freeze=float(rf.mean()),
        kga_vs_freeze=dict(mean=float(gap_f.mean()), ci95=ci(cf), p_better=float((cf > 0).mean()),
                           ci_excludes_zero=bool(np.percentile(cf, 2.5) > 0)),
        kga_vs_adapt=dict(mean=float(gap_a.mean()), ci95=ci(ca), p_better=float((ca > 0).mean()),
                          ci_excludes_zero=bool(np.percentile(ca, 2.5) > 0)),
    )
    out["beats_both_robust"] = bool(out["kga_vs_freeze"]["ci_excludes_zero"]
                                    and out["kga_vs_adapt"]["ci_excludes_zero"])
    print("\nPOOLED MIXED STREAM  n=%d  comp=%s" % (n, out["composition"]))
    print(f"  regret: KGA={rk.mean():.4f}  always-adapt={ra.mean():.4f}  always-freeze={rf.mean():.4f}  FA={fa_pool:.3f}")
    print(f"  KGA vs FREEZE +{gap_f.mean():.4f} 95%CI{ci(cf)} excl0={out['kga_vs_freeze']['ci_excludes_zero']}")
    print(f"  KGA vs ADAPT  +{gap_a.mean():.4f} 95%CI{ci(ca)} excl0={out['kga_vs_adapt']['ci_excludes_zero']}")
    print(f"  beats_both_robust = {out['beats_both_robust']}")
    Path(ROOT / "research_lock/KBOUND_MIXED_STREAM_v1.json").write_text(json.dumps(out, indent=2))
    print("saved research_lock/KBOUND_MIXED_STREAM_v1.json")


if __name__ == "__main__":
    main()
