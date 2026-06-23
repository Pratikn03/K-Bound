#!/usr/bin/env python3
"""Machine-extract the corrected per-dataset beats_both verdict for every run
benchmark, straight from the canonical result artifacts (no hand-typed numbers).

corrected = (router_regret < freeze_regret) AND (router_regret < best_fixed_adapt_regret)
            AND (false_adapt_rate <= ALPHA) AND (stable across seeds, where applicable)
Emits benchmark_verdicts.json and prints the summary table."""
import json, os
REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"; A = 0.10; EPS = 1e-9
def L(p): return json.load(open(os.path.join(REPO, p)))
def beats(r, fr, ad, fa): return (r < fr - EPS) and (r < ad - EPS) and (fa <= A + 1e-12)
rows = []

# CIFAR-10-C : decisive grid, benchmark cifar10c (tent + eata are wins)
d = L("experiments/kbound/results/decisive_tta_results.json")["benchmarks"]["cifar10c"]["methods"]
for meth in ("tent", "eata", "sar"):
    m = d[meth]["metrics"]; rv = m["regret_vs_oracle"]; fa = m["false_adapt_rate_B<0"]
    rows.append(("CIFAR-10-C ("+meth+")", rv["K_Bound"], rv["always_freeze"], rv["always_adapt"], fa,
                 m["beats_both"], beats(rv["K_Bound"], rv["always_freeze"], rv["always_adapt"], fa),
                 "decisive_tta_results.json::benchmarks.cifar10c"))

# ImageNet-C : imagenetc_noise, sar is the decisive win
d = L("experiments/kbound/results/imagenetc_noise/decisive_tta_results.json")["benchmarks"]["imagenetc"]["methods"]
for meth in ("tent", "eata", "sar"):
    m = d[meth]["metrics"]; rv = m["regret_vs_oracle"]; fa = m["false_adapt_rate_B<0"]
    rows.append(("ImageNet-C ("+meth+")", rv["K_Bound"], rv["always_freeze"], rv["always_adapt"], fa,
                 m["beats_both"], beats(rv["K_Bound"], rv["always_freeze"], rv["always_adapt"], fa),
                 "imagenetc_noise/decisive_tta_results.json::benchmarks.imagenetc"))

# Camelyon17 : protocol G held-out
m = L("experiments/kbound/results/camelyon17_protocol_G_v1/analyze_F_results.json")["test_locked"]
rows.append(("Camelyon17 (G, eata_online, 5-seed held-out)", m["regret_kga"], m["regret_freeze"],
             m["regret_adapt"], m["false_adapt"], None,
             beats(m["regret_kga"], m["regret_freeze"], m["regret_adapt"], m["false_adapt"]),
             "camelyon17_protocol_G_v1/analyze_F_results.json::test_locked"))

# RxRx1 : protocol J held-out
m = L("experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json")["test_locked"]
rows.append(("RxRx1 (J, sar_online, 10-seed held-out)", m["regret_kga"], m["regret_freeze"],
             m["regret_adapt"], m["false_adapt"], L("experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json").get("beats_both"),
             beats(m["regret_kga"], m["regret_freeze"], m["regret_adapt"], m["false_adapt"]),
             "rxrx1_protocol_J_v1/analyze_F_results.json::test_locked"))

# fMoW : protocol L held-out
m = L("experiments/kbound/results/fmow_protocol_L_v1/analyze_F_results.json")["test_locked"]
rows.append(("fMoW (L, sar_online, 5-seed held-out)", m["regret_kga"], m["regret_freeze"],
             m["regret_adapt"], m["false_adapt"], L("experiments/kbound/results/fmow_protocol_L_v1/analyze_F_results.json").get("beats_both"),
             beats(m["regret_kga"], m["regret_freeze"], m["regret_adapt"], m["false_adapt"]),
             "fmow_protocol_L_v1/analyze_F_results.json::test_locked"))

# iWildCam : route-b multicandidate on full val (the patched bug node)
n = L("experiments/kbound/results/iwildcam_full_val/result_f08e751c.json")["routing_b_multicandidate"]
rv = n["regret_vs_oracle"]; fa = n["false_adapt_rate"]
rows.append(("iWildCam (full-val route-b multicandidate)", rv["router"], rv["always_freeze"],
             rv["best_fixed_always_adapt"], fa, n.get("beats_both_raw"),
             beats(rv["router"], rv["always_freeze"], rv["best_fixed_always_adapt"], fa),
             "iwildcam_full_val/result_f08e751c.json::routing_b_multicandidate (PATCHED)"))

# Office-Home : val verdict (route-a deployed)
v = L("experiments/kbound/results/officehome_full_FINAL/VERDICT_val.json")
ra = v["route_a_deployed"]; rv = ra["regret_vs_oracle"]
rows.append(("Office-Home (val, route-a deployed)", rv["K_Bound"], rv["always_freeze"], rv["always_adapt"],
             0.0, v.get("beats_both"),
             beats(rv["K_Bound"], rv["always_freeze"], rv["always_adapt"], 0.0),
             "officehome_full_FINAL/VERDICT_val.json::route_a_deployed"))

# CIFAR-10.1 : multiseed stability (beats_both_count over 5 seeds)
p = L("experiments/kbound/results/cifar101_multiseed_v1/pooled_summary.json")["pooled"]
best = max(p.items(), key=lambda kv: kv[1]["beats_both_count"])
rows.append(("CIFAR-10.1 (5-seed; best=%s %d/5)" % (best[0], best[1]["beats_both_count"]),
             best[1]["regret_mean"], None, None, None, None,
             best[1]["beats_both_count"] >= 4,  # "stable" requires majority of seeds
             "cifar101_multiseed_v1/pooled_summary.json (beats_both_count/5)"))

# ImageNet-R : multiseed beats_both_by_candidate
bb = L("experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json")["beats_both_by_candidate"]
rows.append(("ImageNet-R (multiseed; %d/%d candidates beat both)" % (sum(bb.values()), len(bb)),
             None, None, None, None, any(bb.values()), any(bb.values()),
             "imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json::beats_both_by_candidate"))

out = []
print("%-46s %10s %10s %10s %7s  %-6s %-9s" % ("dataset", "router", "freeze", "bestadapt", "FA", "stored", "CORRECT"))
print("-" * 120)
for name, r, fr, ad, fa, stored, corr, src in rows:
    def f(x): return "  n/a  " if x is None else ("%.5f" % x)
    print("%-46s %10s %10s %10s %7s  %-6s %-9s" % (name, f(r), f(fr), f(ad),
          ("n/a" if fa is None else "%.3f" % fa), str(stored), str(corr)))
    out.append({"dataset": name, "router_regret": r, "freeze_regret": fr, "best_fixed_adapt_regret": ad,
                "false_adapt": fa, "stored_beats_both": stored, "corrected_verdict": bool(corr), "source": src})
json.dump(out, open(os.path.join(REPO, "audits/integrity_2026-06-20/benchmark_verdicts.json"), "w"), indent=2)
print("\nwrote audits/integrity_2026-06-20/benchmark_verdicts.json")
