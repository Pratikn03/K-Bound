import sys, glob, os, json
import numpy as np
sys.path.insert(0, "/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/realdata")
from realdata_audit import audit_task, ARCHIVE

# 62 torch-free usable tasks: 42 ADBench tabular + 20 special domains (exclude cv_* image OOD)
adb = sorted(glob.glob(os.path.join(ARCHIVE, "adb_*.npz")))
special = []
for pat in ["creditcard.npz", "online_shoppers.npz", "nlp_*.npz", "unsw_*.npz"]:
    special += sorted(glob.glob(os.path.join(ARCHIVE, pat)))
files = adb + special
print(f"P1: auditing {len(files)} tasks ({len(adb)} adbench + {len(special)} special)")

results = {"val_opt": [], "median": []}
for mode in ["val_opt", "median"]:
    for f in files:
        r = audit_task(f, thr_mode=mode, min_D=40, boot=300)
        results[mode].append(r)

# ---- aggregate ----
def summarize(rows, tag):
    used = [r for r in rows if "skipped" not in r]
    skipped = [r for r in rows if "skipped" in r]
    n = len(used)
    Hrej = [r for r in used if r["H_reject"] is True]
    Hpass = [r for r in used if r["H_reject"] is False]
    def sign_acc(group):
        sc = [r["sign_ok"] for r in group if r["sign_ok"] is not None]
        return (float(np.mean(sc)), len(sc)) if sc else (None, 0)
    sa_all = sign_acc(used); sa_rej = sign_acc(Hrej); sa_pass = sign_acc(Hpass)
    errs = [r["err_abs_b_max"] for r in used]
    gaps = [r["gamma_recoverable_gap"] for r in used]
    summ = {
        "tag": tag, "n_used": n, "n_skipped": len(skipped),
        "H_reject_rate": round(len(Hrej) / n, 4) if n else None,
        "n_H_reject": len(Hrej), "n_H_pass": len(Hpass),
        "sign_acc_all": sa_all, "sign_acc_H_reject": sa_rej, "sign_acc_H_pass": sa_pass,
        "median_err_abs_b": round(float(np.median(errs)), 4) if errs else None,
        "mean_err_abs_b": round(float(np.mean(errs)), 4) if errs else None,
        "median_gamma_gap": round(float(np.median(gaps)), 4) if gaps else None,
    }
    return summ

summ_valopt = summarize(results["val_opt"], "val_opt")
summ_median = summarize(results["median"], "median")
print("\n=== P1 SUMMARY (val-optimal thresholds) ===")
print(json.dumps(summ_valopt, indent=2))
print("\n=== P1 SUMMARY (median thresholds) ===")
print(json.dumps(summ_median, indent=2))

out = {"P1": {"per_task_val_opt": results["val_opt"],
              "per_task_median": results["median"],
              "summary_val_opt": summ_valopt,
              "summary_median": summ_median,
              "notes": ("f0=best-val-AUC detector; f_a=second-best-val detector; "
                        "D={f_a != f0} on TEST; labels used ONLY to score b_true/sign_true. "
                        "tau null = best-fit CEI/per-class-symmetric H-model bootstrap (300 draws), "
                        "H_reject if observed tau > null q95. Identification is up to global flip; "
                        "sign scored is relative sign(b_a-b_0) fixed by majority-above-chance anchor.")}}
json.dump(out, open("/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/realdata/_p1_partial.json", "w"), indent=2)
print("\nwrote _p1_partial.json")
