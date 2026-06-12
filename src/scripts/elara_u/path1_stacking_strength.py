"""Path 1: strengthen the stacking result.

(a) Stronger stacker baselines: is "stacking beats selection" robust across stacker
    families (logistic / random forest / gradient boosting), and does it beat the
    strong meta-baselines (auto-select, simple average, best fixed detector)?
(b) Calibration: fit a validation-only isotonic calibrator and report ECE / Brier /
    NLL for the stack vs auto-select (raw and calibrated).

Operates on the cached 123-task score archive. No test labels are used for fitting
(validation labels train stackers/calibrators; test labels score only).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "experiments/elara_u/score_archive"
RNG = 0


def _stack(ctor, Sv, yv, St):
    if len(np.unique(yv)) < 2:
        return St.mean(1)
    return ctor().fit(Sv, yv).predict_proba(St)[:, 1]


STACKERS = {
    "stack_logistic": lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    "stack_rf": lambda: RandomForestClassifier(n_estimators=200, random_state=RNG, class_weight="balanced"),
    "stack_gbm": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=RNG),
}


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    tasks = []
    for f in sorted(ARCHIVE.glob("*.npz")):
        if f.name.startswith("._"):
            continue
        z = np.load(f, allow_pickle=True)
        if len(np.unique(z["ytest"])) > 1 and len(np.unique(z["yval"])) > 1:
            tasks.append((z["Sval"], z["yval"].astype(int), z["Stest"], z["ytest"].astype(int),
                          z["val_auc"], list(z["det_names"])))
    dets = tasks[0][5]
    methods = list(dets) + ["auto_select", "avg"] + list(STACKERS)
    auc = {m: [] for m in methods}
    cal = {"stack_logistic_raw": [], "stack_logistic_isotonic": [], "auto_select": []}  # (brier, nll, ece) tuples

    def ece(s, y, bins=10):
        s = np.clip(s, 0, 1); e = 0.0
        for i in range(bins):
            lo, hi = i / bins, (i + 1) / bins
            m = (s >= lo) & (s < hi if i < bins - 1 else s <= hi)
            if m.any():
                e += m.mean() * abs(s[m].mean() - y[m].mean())
        return e

    for Sv, yv, St, yt, vauc, _ in tasks:
        for j, d in enumerate(dets):
            auc[d].append(roc_auc_score(yt, St[:, j]))
        auc["auto_select"].append(roc_auc_score(yt, St[:, int(np.argmax(vauc))]))
        auc["avg"].append(roc_auc_score(yt, St.mean(1)))
        stack_test = {}
        for name, ctor in STACKERS.items():
            sc = _stack(ctor, Sv, yv, St)
            stack_test[name] = sc
            auc[name].append(roc_auc_score(yt, sc))
        # calibration on the logistic stack: raw vs isotonic (fit on val)
        sv_log = _stack(STACKERS["stack_logistic"], Sv, yv, Sv)   # stack scores on val
        st_log = stack_test["stack_logistic"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(sv_log, yv)
        st_iso = iso.predict(st_log)
        st_auto = St[:, int(np.argmax(vauc))]
        for key, sc in [("stack_logistic_raw", st_log), ("stack_logistic_isotonic", np.clip(st_iso, 0, 1)),
                        ("auto_select", st_auto)]:
            cal[key].append((brier_score_loss(yt, np.clip(sc, 0, 1)),
                             log_loss(yt, np.clip(sc, 1e-6, 1 - 1e-6), labels=[0, 1]),
                             ece(sc, yt)))

    pa = {m: np.array(v) for m, v in auc.items()}
    best_fixed = max(dets, key=lambda d: pa[d].mean())
    contrasts = {}
    for st in list(STACKERS) + ["avg"]:
        for base in ["auto_select", best_fixed]:
            mean, lo, hi = _boot(pa[st] - pa[base])
            contrasts[f"{st}_vs_{base}"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                            "ci_excludes_zero": lo > 0 or hi < 0}
    # are the stacker families equivalent?
    m, lo, hi = _boot(pa["stack_rf"] - pa["stack_logistic"])
    contrasts["stack_rf_vs_stack_logistic"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                               "ci_excludes_zero": lo > 0 or hi < 0}
    m, lo, hi = _boot(pa["stack_gbm"] - pa["stack_logistic"])
    contrasts["stack_gbm_vs_stack_logistic"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                               "ci_excludes_zero": lo > 0 or hi < 0}

    calib = {k: {"brier": round(float(np.mean([x[0] for x in v])), 4),
                 "nll": round(float(np.mean([x[1] for x in v])), 4),
                 "ece": round(float(np.mean([x[2] for x in v])), 4)} for k, v in cal.items()}

    res = {"protocol": "PATH1_STACKING_STRENGTH_v1", "n_tasks": len(tasks),
           "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
           "best_fixed": best_fixed, "stacker_and_baseline_contrasts": contrasts, "calibration": calib}
    out = ROOT / "experiments/elara_u/path1_stacking_strength.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"=== Path 1: stronger stackers + calibration ({len(tasks)} tasks) ===")
    for m in ["auto_select", "avg", best_fixed] + list(STACKERS):
        print(f"  {m:16} mean AUROC {res['mean_auroc'][m]:.4f}")
    print("\ncontrasts (method - baseline):")
    for k, v in contrasts.items():
        print(f"  {k:34} {v['mean']:+.4f} CI {v['ci95']} excl0={v['ci_excludes_zero']}")
    print("\ncalibration (lower better):")
    for k, v in calib.items():
        print(f"  {k:26} Brier={v['brier']:.4f} NLL={v['nll']:.4f} ECE={v['ece']:.4f}")
    allbeat = all(contrasts[f"{st}_vs_auto_select"]["mean"] > 0 and contrasts[f"{st}_vs_auto_select"]["ci_excludes_zero"]
                  for st in STACKERS)
    print(f"\nALL stacker families beat auto-select (CI excl 0)? {'YES' if allbeat else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
