import json
RX = json.load(open("/Users/pratik_n/kbound_rxrx1_results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json"))
IR = json.load(open("/Users/pratik_n/kbound_inr_results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json"))

def show(d, tag):
    ks = d["kbound_summary"]; b = d["baselines"]; ra = d["routing_a_single_candidate"]; rb = d["routing_b_multicandidate"]; rc = d["routing_c_smooth_drift"]
    td = d["tau_distribution"]
    print("="*30, tag, "="*30)
    print("classification:", ks["classification"])
    print("harmful base rate B<0:", round(ks["base_rate_harmful_B<0"],4))
    print("mean_B:", round(ks["mean_B"],4))
    print("detect verdict:", ks["detectability_verdict"], "best harm-AUC:", round(ks["best_single_feature_harm_AUC"],4))
    print("tau range:", round(min(td),3), "-", round(max(td),3), "n=", len(td))
    print("routing_b abstain_rate:", rb.get("abstention_rate"), "mean_tau:", round(rb.get("mean_tau") or 0,3))
    print("smooth-drift decisions:", rc.get("decision_counts"), "bracket_cov:", rc.get("bracket_coverage_trueB"))
    print("always_freeze mean acc:", round(b["always_freeze_mean_acc"],4))
    print("oracle mean acc:", round(b["per_condition_oracle_mean_acc"],4))
    aa = b["per_candidate_always_adapt_mean_acc"]
    print("always-adapt per cand:", {k: round(v,4) for k,v in aa.items()})
    print("  best adapter:", max(aa,key=aa.get), round(max(aa.values()),4), "| worst:", min(aa,key=aa.get), round(min(aa.values()),4))
    print("--- routing_a per-candidate KGA ---")
    for c, e in ra.items():
        kga = e.get("kga", {})
        if "mean_acc" in kga:
            ma = kga["mean_acc"]; rg = kga["regret_vs_oracle"]
            print(f"  {c}: freeze={ma['always_freeze']:.3f} adapt={ma['always_adapt']:.3f} KGA={ma['K_Bound']:.3f} oracle={ma['oracle']:.3f} | regret KGA={rg['K_Bound']:.4f} adapt={rg['always_adapt']:.4f} freeze={rg['always_freeze']:.4f} | harmfulB<0={e['base_rate_harmful_B<0']:.2f} false_adapt={kga.get('false_adapt_rate_B<0')} beats_both={kga.get('beats_both')} abstain={kga['decision_counts'].get('ABSTAIN')}/{sum(kga['decision_counts'].values())}")
        else:
            print(f"  {c}: {kga.get('note')}")

show(RX, "RxRx1")
show(IR, "ImageNet-R (cross-check vs paper)")
