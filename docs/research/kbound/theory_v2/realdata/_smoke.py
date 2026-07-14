import sys, glob, os, json
sys.path.insert(0, "/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/realdata")
from realdata_audit import audit_task, ARCHIVE
# smoke: 10 ADBench tabular tasks (clean 6-detector schema, varied N and prevalence)
files = sorted(glob.glob(os.path.join(ARCHIVE, "adb_*.npz")))[:10]
print(f"smoke on {len(files)} tasks\n")
for f in files:
    r = audit_task(f, thr_mode="val_opt", boot=200)
    if "skipped" in r:
        print(f"{r['task']:24s} SKIP {r['skipped']}")
    else:
        print(f"{r['task']:22s} K={r['K']} |D|={r['n_D']:4d} piD={r['pi_D']:.3f} "
              f"tau={r['tau']} q95={r['tau_null_q95']} Hrej={r['H_reject']} "
              f"sgn_rec={r['sign_rec_ba_minus_b0']:+d} sgn_true={r['sign_true_ba_minus_b0']:+d} "
              f"ok={r['sign_ok']} errB={r['err_abs_b_max']}")
