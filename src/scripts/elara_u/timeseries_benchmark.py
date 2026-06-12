"""Time-series anomaly family (NAB) for the ELARA-U benchmark.

Each NAB series -> sliding-window features -> a windowed anomaly-classification
task, scored by the same detector zoo and combiners as the other families. Tests
whether the core finding (stacking beats validation auto-selection) GENERALISES to
a temporal domain. No test labels are used for any method (eval only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo

ROOT = Path(__file__).resolve().parents[3]
NAB = ROOT / "data/raw/nab"
W, STRIDE, RNG = 128, 8, 0


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _windows(v, anom):
    feats, labs = [], []
    for i in range(0, len(v) - W, STRIDE):
        w = v[i:i + W]
        x = np.arange(W)
        slope = float(np.polyfit(x, w, 1)[0])
        feats.append([w.mean(), w.std(), w.min(), w.max(), w[-1], slope,
                      w.max() - w.min(), np.abs(np.diff(w)).mean()])
        labs.append(int(anom[i:i + W].any()))
    return np.array(feats, float), np.array(labs, int)


def load_tasks():
    wins = json.loads((NAB / "combined_windows.json").read_text())
    tasks = []
    for key, awins in wins.items():
        if not awins:
            continue
        f = NAB / "data" / key
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["timestamp"])
        v = d["value"].to_numpy(float)
        anom = np.zeros(len(d), bool)
        ts = d["timestamp"]
        for a, b in awins:
            anom |= (ts >= pd.Timestamp(a)) & (ts <= pd.Timestamp(b))
        X, y = _windows(v, anom.to_numpy() if hasattr(anom, "to_numpy") else anom)
        if len(y) >= 80 and int(y.sum()) >= 12 and len(np.unique(y)) == 2:
            tasks.append((key.split("/")[-1].replace(".csv", ""), X, y))
    return tasks


def score(Xtr, Xva, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    sv, st = [], []
    for _, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr); ref = m.decision_function(Xtr)
            sv.append(_zsig(m.decision_function(Xva), ref)); st.append(_zsig(m.decision_function(Xte), ref))
        except Exception:
            sv.append(np.full(len(Xva), 0.5)); st.append(np.full(len(Xte), 0.5))
    return np.column_stack(sv), np.column_stack(st)


def _stack(Sv, yv, St):
    if len(np.unique(yv)) < 2:
        return St.mean(1)
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sv, yv).predict_proba(St)[:, 1]


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    tasks = load_tasks()
    per = {m: [] for m in ["auto_select", "stack", "rank_mean"]}
    for name, X, y in tasks:
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
        Sval, Stest = score(Xtr, Xva, Xte)
        vauc = np.array([roc_auc_score(yva, Sval[:, j]) if len(np.unique(yva)) > 1 else 0.5
                         for j in range(Sval.shape[1])])
        a = {"auto_select": roc_auc_score(yte, Stest[:, int(np.argmax(vauc))]),
             "stack": roc_auc_score(yte, _stack(Sval, yva, Stest)),
             "rank_mean": roc_auc_score(yte, (np.argsort(np.argsort(Stest, 0), 0) / (len(yte) - 1)).mean(1))}
        for m in per:
            per[m].append(float(a[m]))
        print(f"[{name:34}] auto={a['auto_select']:.3f} stack={a['stack']:.3f} ({len(y)} win, {int(y.sum())} anom)", flush=True)
    pa = {m: np.array(v) for m, v in per.items()}
    vs = _boot(pa["stack"] - pa["auto_select"])
    res = {"protocol": "TIMESERIES_NAB_v1", "n_tasks": len(tasks),
           "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
           "stack_vs_auto_select": {"mean": round(vs[0], 4), "ci95": [round(vs[1], 4), round(vs[2], 4)],
                                    "ci_excludes_zero": vs[1] > 0 or vs[2] < 0}}
    out = ROOT / "experiments/elara_u/timeseries_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== TIME-SERIES (NAB, {len(tasks)} tasks) ===")
    print("mean AUROC:", res["mean_auroc"])
    print(f"stack - auto_select: {vs[0]:+.4f} CI [{vs[1]:+.4f},{vs[2]:+.4f}] excl0={res['stack_vs_auto_select']['ci_excludes_zero']}")
    print(f"GENERALISES (stacking beats selection on time-series)? {'YES' if (vs[0]>0 and res['stack_vs_auto_select']['ci_excludes_zero']) else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
