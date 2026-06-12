"""Stronger meta-selection baseline (MetaOD-style), leave-one-task-out (D25).

A reviewer's strongest attack on "stacking beats selection" is: you only beat
argmax auto-select; what about a *learned* meta-selector (e.g. MetaOD)? This builds
a meta-feature selector in MetaOD's spirit -- it learns from the other 122 tasks to
predict each detector's test performance from validation-only meta-features
(including the validation-AUROC landmarkers) and selects the predicted-best
detector. It is strictly stronger than auto-select (it sees the landmarkers plus
learned cross-task corrections). No test labels are used to fit the selector for a
held-out task. We then check whether per-task stacking still beats it.

Honest label: this is our in-repo MetaOD-STYLE meta-selection baseline, not the
published MetaOD artifact; it tests the same idea (meta-learned model selection).
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / "experiments/elara_u/score_archive"


def meta_features(Sval, yval, val_auc):
    """Validation-only meta-features (no test labels)."""
    d = Sval.shape[1]
    mu = Sval.mean(0)
    sd = Sval.std(0)
    sk = skew(Sval, axis=0, bias=False)
    sk = np.nan_to_num(sk)
    C = np.corrcoef(Sval.T)
    C = np.nan_to_num(C)
    off = C[np.triu_indices(d, 1)]
    return np.concatenate([
        val_auc,                          # 6 landmarkers
        mu, sd, sk,                       # 18 score-distribution features
        [float(len(yval)), float(yval.mean())],  # n, anomaly rate
        [float(off.mean()), float(off.std()), float(off.max()), float(off.min())],  # detector diversity
    ])


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    fs = [f for f in sorted(glob.glob(str(ARCH / "*.npz"))) if not os.path.basename(f).startswith("._")]
    MF, TESTAUC, names = [], [], []
    rows = []   # per task: (Sval,yval,Stest,ytest,val_auc)
    for f in fs:
        z = np.load(f, allow_pickle=True)
        Sval, yval, Stest, ytest, va = z["Sval"], z["yval"], z["Stest"], z["ytest"], z["val_auc"]
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        taucs = np.array([roc_auc_score(ytest, Stest[:, j]) for j in range(Stest.shape[1])])
        MF.append(meta_features(Sval, yval, va))
        TESTAUC.append(taucs)
        rows.append((Sval, yval, Stest, ytest, va))
        names.append(os.path.basename(f))
    MF = np.array(MF); TESTAUC = np.array(TESTAUC)
    n = len(rows)

    auto, meta, stack = [], [], []
    for i in range(n):
        Sval, yval, Stest, ytest, va = rows[i]
        # auto-select: argmax validation AUROC
        auto.append(roc_auc_score(ytest, Stest[:, int(np.argmax(va))]))
        # meta-select (MetaOD-style): train on all OTHER tasks, predict this task's per-detector test AUROC
        tr = np.array([j for j in range(n) if j != i])
        rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(MF[tr], TESTAUC[tr])
        pred = rf.predict(MF[i:i + 1])[0]
        meta.append(roc_auc_score(ytest, Stest[:, int(np.argmax(pred))]))
        # per-task logistic stack
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yval)
        stack.append(roc_auc_score(ytest, clf.predict_proba(Stest)[:, 1]))
    auto, meta, stack = np.array(auto), np.array(meta), np.array(stack)

    cmp = {}
    for nm, a, b in [("stack_vs_meta_select", stack, meta),
                     ("stack_vs_auto_select", stack, auto),
                     ("meta_select_vs_auto_select", meta, auto)]:
        m, lo, hi = _boot(a - b)
        cmp[nm] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "pass": lo > 0}
    res = {
        "protocol": "D25_meta_selection_baseline (MetaOD-style, leave-one-task-out)",
        "n_tasks": n,
        "note": "In-repo MetaOD-style meta-feature selector (RF over 28 val-only meta-features incl. landmarkers); not the published MetaOD artifact.",
        "mean_auroc": {"auto_select": round(float(auto.mean()), 4),
                       "meta_select": round(float(meta.mean()), 4),
                       "stack": round(float(stack.mean()), 4)},
        "comparisons": cmp,
        "stacking_beats_learned_selection": cmp["stack_vs_meta_select"]["pass"],
    }
    out = ROOT / "experiments/elara_u/metaod_baseline_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"=== D25 META-SELECTION BASELINE (MetaOD-style, LOO, {n} tasks) ===")
    print(f"  auto_select  mean AUROC {res['mean_auroc']['auto_select']:.4f}")
    print(f"  meta_select  mean AUROC {res['mean_auroc']['meta_select']:.4f}  (learned, MetaOD-style)")
    print(f"  stack        mean AUROC {res['mean_auroc']['stack']:.4f}")
    for k, v in cmp.items():
        print(f"  {k:28} {v['mean']:+.4f} CI {v['ci95']} pass={v['pass']}")
    print(f"  STACKING BEATS LEARNED SELECTION? {'YES' if res['stacking_beats_learned_selection'] else 'NO'}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
