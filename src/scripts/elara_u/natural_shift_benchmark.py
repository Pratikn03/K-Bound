"""Natural temporal-shift SEALED benchmark (D22).

Tests whether drift-aware reliability routing beats BOTH auto-select and plain
stacking when validation goes stale under real distribution shift. Sources:
UNSW-NB15 chronological partitions (early=_1 train/val, late=_3 sealed test),
per attack category; plus creditcard temporal (early 60% / late 40%). No test
labels are used for selection; drift detection uses the test feature distribution
only. ONE-SHOT on the late period.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo
from uais.elara_u.router import RouterPolicy, super_route

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
RNG = 0
CAP_TRAIN, CAP_EVAL = 4000, 2000
CAT_EXCLUDE = {"srcip", "sport", "dstip", "dsport", "proto", "state", "service",
               "Stime", "Ltime", "attack_cat", "Label"}


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _unsw(path, n=180000):
    names = [str(x).strip() for x in
             pd.read_csv(RAW / "cyber/NUSW-NB15_features.csv", encoding="latin-1").iloc[:, 1]]
    d = pd.read_csv(path, header=None, names=names, nrows=n, low_memory=False)
    feats = [c for c in names if c not in CAT_EXCLUDE]
    X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cat = (d["attack_cat"].fillna("normal").astype(str).str.strip().str.lower()
           .replace({"nan": "normal", "": "normal", " ": "normal"}))
    return X.to_numpy(float), cat.to_numpy(), d["Label"].to_numpy(int)


def unsw_tasks():
    Xe, ce, le = _unsw(RAW / "cyber/UNSW-NB15_1.csv")
    Xl, cl, ll = _unsw(RAW / "cyber/UNSW-NB15_3.csv")
    rng = np.random.default_rng(RNG)
    norm_e, norm_l = ce == "normal", cl == "normal"
    out = []
    for cat in ["exploits", "dos", "fuzzers", "reconnaissance", "generic", "backdoors", "analysis"]:
        ae, al = ce == cat, cl == cat
        if ae.sum() < 50 or al.sum() < 50:
            continue
        # early: normal (train) + normal+attack (val); late: normal+attack (sealed)
        ntr = np.where(norm_e)[0]
        ntr = rng.choice(ntr, min(len(ntr), CAP_TRAIN), replace=False)
        ve = np.concatenate([rng.choice(np.where(norm_e)[0], min(norm_e.sum(), CAP_EVAL // 2), replace=False),
                             rng.choice(np.where(ae)[0], min(ae.sum(), CAP_EVAL // 2), replace=False)])
        te = np.concatenate([rng.choice(np.where(norm_l)[0], min(norm_l.sum(), CAP_EVAL // 2), replace=False),
                             rng.choice(np.where(al)[0], min(al.sum(), CAP_EVAL // 2), replace=False)])
        out.append((f"unsw_{cat}", Xe[ntr], Xe[ve], le[ve], Xl[te], ll[te]))
    return out


def creditcard_task():
    d = pd.read_csv(RAW / "fraud/creditcard.csv").sort_values("Time")
    X = d.drop(columns=["Time", "Class"]).to_numpy(float)
    y = d["Class"].to_numpy(int)
    cut = int(0.6 * len(d))
    Xe, ye, Xl, yl = X[:cut], y[:cut], X[cut:], y[cut:]
    rng = np.random.default_rng(RNG)
    tr = rng.choice(np.where(ye == 0)[0], min((ye == 0).sum(), CAP_TRAIN), replace=False)
    ve = np.concatenate([rng.choice(np.where(ye == 0)[0], CAP_EVAL, replace=False), np.where(ye == 1)[0]])
    te = np.concatenate([rng.choice(np.where(yl == 0)[0], CAP_EVAL, replace=False), np.where(yl == 1)[0]])
    return [("creditcard_temporal", Xe[tr], Xe[ve], ye[ve], Xl[te], yl[te])]


def score_task(Xtr, Xval, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xval, Xte = sc.transform(Xtr), sc.transform(Xval), sc.transform(Xte)
    sval, stest = [], []
    for _, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr)
            ref = m.decision_function(Xtr)
            sval.append(_zsig(m.decision_function(Xval), ref))
            stest.append(_zsig(m.decision_function(Xte), ref))
        except Exception:
            sval.append(np.full(len(Xval), 0.5)); stest.append(np.full(len(Xte), 0.5))
    return np.column_stack(sval), np.column_stack(stest)


def _plain_stack(Sv, yv, St):
    if len(np.unique(yv)) < 2:
        return St.mean(1)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sv, yv)
    return clf.predict_proba(St)[:, 1]


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(np.mean(diff)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    tasks = unsw_tasks() + creditcard_task()
    rows, per = {}, {m: [] for m in ["auto_select", "plain_stack", "drift_stack"]}
    for name, Xtr, Xval, yval, Xte, yte in tasks:
        t0 = time.time()
        Sval, Stest = score_task(Xtr, Xval, Xte)
        vauc = np.array([roc_auc_score(yval, Sval[:, j]) if len(np.unique(yval)) > 1 else 0.5
                         for j in range(Sval.shape[1])])
        auto = Stest[:, int(np.argmax(vauc))]
        plain = _plain_stack(Sval, yval, Stest)
        drift, act = super_route(Sval, yval, Stest, RouterPolicy())
        a = {m: float(roc_auc_score(yte, s)) for m, s in
             [("auto_select", auto), ("plain_stack", plain), ("drift_stack", drift)]}
        for m in per:
            per[m].append(a[m])
        rows[name] = a | {"action": act}
        print(f"[{name:22}] auto={a['auto_select']:.3f} plain_stack={a['plain_stack']:.3f} "
              f"drift_stack={a['drift_stack']:.3f} ({act}, {time.time()-t0:.0f}s)", flush=True)

    pa = {m: np.array(v) for m, v in per.items()}
    vs_auto = _boot(pa["drift_stack"] - pa["auto_select"])
    vs_plain = _boot(pa["drift_stack"] - pa["plain_stack"])
    res = {
        "protocol": "NATURAL_SHIFT_SEALED_v1 (D22)", "n_tasks": len(rows),
        "mean_auroc": {m: round(float(np.mean(v)), 4) for m, v in pa.items()},
        "drift_stack_vs_auto_select": {"mean": vs_auto[0], "ci95": vs_auto[1:],
                                       "ci_excludes_zero": vs_auto[1] > 0 or vs_auto[2] < 0},
        "drift_stack_vs_plain_stack": {"mean": vs_plain[0], "ci95": vs_plain[1:],
                                       "ci_excludes_zero": vs_plain[1] > 0 or vs_plain[2] < 0},
        "per_task": rows,
    }
    out = ROOT / "experiments/elara_u/natural_shift_results.json"
    out.write_text(json.dumps(res, indent=2))
    print("\n=== NATURAL-SHIFT SEALED (D22) ===")
    print("mean AUROC:", res["mean_auroc"])
    print(f"drift_stack - auto_select : {vs_auto[0]:+.4f} CI [{vs_auto[1]:+.4f},{vs_auto[2]:+.4f}] "
          f"excl0={res['drift_stack_vs_auto_select']['ci_excludes_zero']}")
    print(f"drift_stack - plain_stack : {vs_plain[0]:+.4f} CI [{vs_plain[1]:+.4f},{vs_plain[2]:+.4f}] "
          f"excl0={res['drift_stack_vs_plain_stack']['ci_excludes_zero']}")
    both = (res["drift_stack_vs_auto_select"]["ci_excludes_zero"] and vs_auto[0] > 0
            and res["drift_stack_vs_plain_stack"]["ci_excludes_zero"] and vs_plain[0] > 0)
    print(f"GATE D22 (router beats BOTH meta-baselines): {'PASS' if both else 'FAIL'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
