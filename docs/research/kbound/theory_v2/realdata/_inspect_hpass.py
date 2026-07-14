import json
o = json.load(open("/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/realdata/_p1_partial.json"))
for mode in ["per_task_median", "per_task_val_opt"]:
    print(f"\n##### H-PASS tasks ({mode}) #####")
    for r in o["P1"][mode]:
        if "skipped" in r: continue
        if r["H_reject"] is False:
            print(f"{r['task']:20s} |D|={r['n_D']:4d} piD={r['pi_D']:.3f} tau={r['tau']:.4f} "
                  f"q95={r['tau_null_q95']:.4f} sgn_rec={r['sign_rec_ba_minus_b0']:+d} "
                  f"sgn_true={r['sign_true_ba_minus_b0']:+d} ok={r['sign_ok']} errB={r['err_abs_b_max']:.3f}")
    # also count tasks where sign_true==0 (excluded from accuracy)
    used = [r for r in o["P1"][mode] if "skipped" not in r]
    n0 = sum(1 for r in used if r["sign_ok"] is None)
    print(f"  (tasks with sign_true==0 excluded from accuracy: {n0})")
