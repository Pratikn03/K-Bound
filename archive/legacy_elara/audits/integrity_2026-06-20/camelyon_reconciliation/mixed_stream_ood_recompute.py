#!/usr/bin/env python3
"""
mixed_stream_ood_recompute.py - Phase 3 integrity recompute (2026-06-20)

The locked mixed-stream capstone (research_lock/KBOUND_MIXED_STREAM_v1.json) pools the
Camelyon17 n=54 held-out cells, which (per Phase 1) include 18 in-distribution id_val
cells. This script (a) reproduces the locked pooled result as a check, then (b) recomputes
the mixed stream with Camelyon restricted to genuine OOD domains {test, val} only, so NO
part of the capstone relies on id_val pooling. Reuses analyze_F's own functions.
"""
import json, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/Volumes/T9/uav/AutoML_Flagship_V8")
spec = importlib.util.spec_from_file_location("af", ROOT / "docs/research/kbound/scripts/analyze_F.py")
af = importlib.util.module_from_spec(spec); spec.loader.exec_module(af)

def per_condition(cal, test):
    Zc = np.array([r["Z"] for r in cal], float); Bc = np.array([r["B"] for r in cal], float)
    Zt = np.array([r["Z"] for r in test], float); Bt = np.array([r["B"] for r in test], float)
    a0 = np.array([r["a0"] for r in test], float); aa = np.array([r["aa"] for r in test], float)
    m = af.fit_point(Zc, Bc)
    eps = float(np.quantile(np.abs(m.predict(Zc) - Bc), 1 - af.ALPHA))
    dec = af.decide_global(m.predict(Zt), eps)
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0); oracle = np.maximum(a0, aa)
    return dict(reg_kga=oracle - kga, reg_adapt=oracle - aa, reg_freeze=oracle - a0,
                B=Bt, adapt=adapt, n=len(test))

def filt_seed(recs, seeds):
    s = set(seeds); return [r for r in recs if int(r["seed"]) in s]

def camelyon(cal_test_domains):
    raw = json.load(open(ROOT / "experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json"))["records"]
    rows = [af._one_record(r, candidate="eata_online") for r in raw
            if r["candidate"] == "eata_online" and r["domain"] in cal_test_domains]
    return filt_seed(rows, [0, 1]), filt_seed(rows, [2, 3, 4])

def officehome():
    cal = filt_seed(af.load_records(str(ROOT / "experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json"), candidate="sar_online_aggressive")[0], [0, 1])
    test = filt_seed(af.load_records(str(ROOT / "experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json"), candidate="sar_online_aggressive")[0], [0, 1])
    return cal, test

def iwildcam():
    recs = af.load_records(str(ROOT / "experiments/kbound/results/iwildcam_full_test/result_e40faf29.json"), candidate="tent_episodic")[0]
    return filt_seed(recs, [0]), filt_seed(recs, [1])

def pool_and_boot(parts, seed=13, B=5000):
    rng = np.random.default_rng(seed)
    rk = np.concatenate([p["reg_kga"] for p in parts]); ra = np.concatenate([p["reg_adapt"] for p in parts])
    rf = np.concatenate([p["reg_freeze"] for p in parts]); Bp = np.concatenate([p["B"] for p in parts])
    adp = np.concatenate([p["adapt"] for p in parts]); n = len(rk)
    fa = float((Bp[adp] < 0).mean()) if adp.any() else 0.0
    gap_f = rf - rk; gap_a = ra - rk; idx = rng.integers(0, n, (B, n))
    cf = gap_f[idx].mean(1); ca = gap_a[idx].mean(1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return dict(n=n, regret_kga=float(rk.mean()), regret_adapt=float(ra.mean()),
                regret_freeze=float(rf.mean()), false_adapt=fa,
                kga_vs_freeze_mean=float(gap_f.mean()), kga_vs_freeze_ci=ci(cf),
                kga_vs_adapt_mean=float(gap_a.mean()), kga_vs_adapt_ci=ci(ca),
                beats_both_robust=bool(np.percentile(cf, 2.5) > 0 and np.percentile(ca, 2.5) > 0))

oh, iw = officehome(), iwildcam()
pc_oh = per_condition(*oh); pc_iw = per_condition(*iw)

# (a) reproduce locked: Camelyon pooled (all 3 domains)
cal_p, test_p = camelyon({"test", "val", "id_val"})
repro = pool_and_boot([per_condition(cal_p, test_p), pc_oh, pc_iw])

# (b) integrity recompute: Camelyon OOD-only {test, val}
cal_o, test_o = camelyon({"test", "val"})
ood = pool_and_boot([per_condition(cal_o, test_o), pc_oh, pc_iw])

print("=== (a) REPRODUCE LOCKED (Camelyon pooled, incl id_val) ===")
print(json.dumps(repro, indent=2))
print("\n=== (b) INTEGRITY RECOMPUTE (Camelyon OOD test+val only) ===")
print(json.dumps(ood, indent=2))
json.dump({"reproduce_locked_pooled": repro, "ood_only_camelyon": ood},
          open(ROOT / "audits/integrity_2026-06-20/camelyon_reconciliation/mixed_stream_recompute.json", "w"), indent=2)
print("\nSAVED mixed_stream_recompute.json")
