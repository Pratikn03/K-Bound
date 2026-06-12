"""Time-series family benchmark on TSB-AD-M (multivariate). Tests whether the
rank-normalized stack beats validation-AUROC selection on real multivariate
time-series anomaly tasks --- reported as a SEPARATE family (not merged into the
123-task headline), like the industrial family.

Each TSB-AD-M CSV is a multivariate series with a Label column and a temporal train
split encoded in the filename (..._tr_<N>_...). We fit the detector zoo on the first
N (normal) rows, then evaluate point-wise on the remainder with a stratified val/test
split (anomalies are rare, so stratification keeps both classes in each split). No
test labels are used to fit any router.
"""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
TSB = ROOT / "data/raw/tsb_ad"
OUT = ROOT / "experiments/elara_u/tsb_timeseries_results.json"
RNG = 0


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _rn(S):
    return np.column_stack([rankdata(S[:, j]) / len(S) for j in range(S.shape[1])])


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else 0.5


def score_series(csv):
    df = pd.read_csv(csv)
    lab = [c for c in df.columns if c.lower() == "label"]
    if not lab:
        return None
    y = df[lab[0]].to_numpy().astype(int)
    X = df.drop(columns=lab).to_numpy(dtype=float)
    X = np.nan_to_num(X)
    m = re.search(r"_tr_(\d+)", os.path.basename(csv))
    tr = int(m.group(1)) if m else len(X) // 3
    tr = min(max(tr, 50), len(X) - 60)
    Xtr = X[:tr]
    Xrest, yrest = X[tr:], y[tr:]
    if int(yrest.sum()) < 12 or len(np.unique(yrest)) < 2:
        return None
    try:
        Xv, Xte, yv, yte = train_test_split(Xrest, yrest, test_size=0.5, random_state=RNG, stratify=yrest)
    except ValueError:
        return None
    if len(np.unique(yv)) < 2 or len(np.unique(yte)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s, Xte_s = sc.transform(Xtr), sc.transform(Xv), sc.transform(Xte)
    Sval, Stest, vauc, dets = [], [], [], []
    for name, ctor in detector_zoo().items():
        try:
            mdl = ctor().fit(Xtr_s)
            ref = mdl.decision_function(Xtr_s)
            v = _zsig(mdl.decision_function(Xv_s), ref); t = _zsig(mdl.decision_function(Xte_s), ref)
        except Exception:
            v = np.full(len(yv), 0.5); t = np.full(len(yte), 0.5)
        Sval.append(v); Stest.append(t); vauc.append(_auc(yv, v)); dets.append(name)
    Sval, Stest, vauc = np.column_stack(Sval), np.column_stack(Stest), np.array(vauc)
    # strategies
    js = int(np.argmax(vauc))
    out = {f"fixed/{dets[j]}": _auc(yte, Stest[:, j]) for j in range(len(dets))}
    out["auto_select"] = _auc(yte, Stest[:, js])
    Xrn_v, Xrn_t = _rn(Sval), _rn(Stest)
    if len(np.unique(yv)) == 2:
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xrn_v, yv)
        out["stack"] = _auc(yte, clf.predict_proba(Xrn_t)[:, 1])
    else:
        out["stack"] = _auc(yte, Xrn_t.mean(1))
    out["_dets"] = dets
    return out


def _boot(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d)
    rng = np.random.default_rng(RNG)
    bb = [d[rng.integers(0, n, n)].mean() for _ in range(10000)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
            "sig": bool(np.percentile(bb, 2.5) > 0), "win": float(np.mean(d > 0))}


def main():
    files = [f for f in sorted(glob.glob(str(TSB / "**/TSB-AD-M/*.csv"), recursive=True))
             if not os.path.basename(f).startswith("._")]
    print(f"found {len(files)} TSB-AD-M series", flush=True)
    rows = []
    for i, f in enumerate(files):
        try:
            r = score_series(f)
        except Exception:
            r = None
        if r:
            rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)} ({len(rows)} usable)", flush=True)
    if len(rows) < 5:
        print("too few usable series"); return 1
    dets = rows[0]["_dets"]
    col = lambda s: np.array([r[s] for r in rows])
    bestfix = max((f"fixed/{d}" for d in dets), key=lambda s: col(s).mean())
    res = {"protocol": "ELARA_U_TIMESERIES_TSB_AD_M (separate family)", "n_tasks": len(rows),
           "mean_auroc": {"stack": float(col("stack").mean()), "auto_select": float(col("auto_select").mean()),
                          "best_fixed": float(col(bestfix).mean())}, "best_fixed": bestfix,
           "stack_vs_auto_select": _boot(col("stack"), col("auto_select")),
           "stack_vs_best_fixed": _boot(col("stack"), col(bestfix))}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n=== TSB-AD-M time-series ({len(rows)} tasks) ===")
    print(f"  stack={res['mean_auroc']['stack']:.3f} auto={res['mean_auroc']['auto_select']:.3f} "
          f"fixed={res['mean_auroc']['best_fixed']:.3f}")
    print(f"  stack vs auto: {res['stack_vs_auto_select']['mean']:+.4f} {res['stack_vs_auto_select']['ci95']} "
          f"sig={res['stack_vs_auto_select']['sig']}")
    print(f"  stack vs fixed: {res['stack_vs_best_fixed']['mean']:+.4f} {res['stack_vs_best_fixed']['ci95']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
