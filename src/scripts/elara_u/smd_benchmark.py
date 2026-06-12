"""SMD (Server Machine Dataset) multivariate time-series anomaly family.

TSB-AD's data is gated (HuggingFace auth); SMD is an openly available standard
multivariate TS anomaly benchmark (28 machines x 38 dims) that is part of TSB-AD's
suite. Each machine -> multivariate sliding-window features -> a windowed
anomaly-classification task, scored by the same detector zoo and combiners. Tests
whether stacking-beats-selection holds on multivariate time series. Detectors are
fit on the all-normal training period; no test labels are used for any method.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
SMD = ROOT / "data/raw/smd"
W, STRIDE, RNG = 100, 10, 0


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _windows(M, lab=None):
    feats, labs = [], []
    for i in range(0, len(M) - W, STRIDE):
        w = M[i:i + W]
        feats.append(np.concatenate([w.mean(0), w.std(0)]))   # per-dim mean+std
        if lab is not None:
            labs.append(int(lab[i:i + W].any()))
    return np.array(feats, float), (np.array(labs, int) if lab is not None else None)


def task(machine):
    tr = np.loadtxt(SMD / "train" / f"{machine}.txt", delimiter=",")
    te = np.loadtxt(SMD / "test" / f"{machine}.txt", delimiter=",")
    lb = np.loadtxt(SMD / "test_label" / f"{machine}.txt", delimiter=",")
    Xtr, _ = _windows(tr)                       # all-normal training windows
    Xall, yall = _windows(te, lb)
    if len(np.unique(yall)) < 2 or int(yall.sum()) < 12:
        return None
    Xva, Xte, yva, yte = train_test_split(Xall, yall, test_size=0.5, random_state=RNG, stratify=yall)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    sval, stest = [], []
    for _, ctor in detector_zoo().items():
        try:
            m = ctor().fit(Xtr); ref = m.decision_function(Xtr)
            sval.append(_zsig(m.decision_function(Xva), ref)); stest.append(_zsig(m.decision_function(Xte), ref))
        except Exception:
            sval.append(np.full(len(yva), 0.5)); stest.append(np.full(len(yte), 0.5))
    return np.column_stack(sval), yva, np.column_stack(stest), yte


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    machines = sorted(p.stem for p in (SMD / "train").glob("*.txt") if not p.name.startswith("._"))
    per = {m: [] for m in ["auto_select", "stack", "rank_mean"]}
    n = 0
    for mc in machines:
        out = task(mc)
        if out is None:
            continue
        Sval, yva, Stest, yte = out
        n += 1
        vauc = np.array([roc_auc_score(yva, Sval[:, j]) if len(np.unique(yva)) > 1 else 0.5 for j in range(Sval.shape[1])])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yva)
        a = {"auto_select": roc_auc_score(yte, Stest[:, int(np.argmax(vauc))]),
             "stack": roc_auc_score(yte, clf.predict_proba(Stest)[:, 1]),
             "rank_mean": roc_auc_score(yte, (np.argsort(np.argsort(Stest, 0), 0) / (len(yte) - 1)).mean(1))}
        for m in per:
            per[m].append(float(a[m]))
        print(f"[{mc}] auto={a['auto_select']:.3f} stack={a['stack']:.3f}", flush=True)
    pa = {m: np.array(v) for m, v in per.items()}
    vs = _boot(pa["stack"] - pa["auto_select"])
    res = {"protocol": "SMD_MULTIVARIATE_TS_v1 (TSB-AD-suite, openly available)", "n_tasks": n,
           "note": "TSB-AD proper is gated; SMD is the open standard substitute (part of TSB-AD suite).",
           "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
           "stack_vs_auto_select": {"mean": round(vs[0], 4), "ci95": [round(vs[1], 4), round(vs[2], 4)],
                                    "ci_excludes_zero": vs[1] > 0 or vs[2] < 0}}
    out = ROOT / "experiments/elara_u/smd_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== SMD multivariate TS ({n} machines) ===")
    print("mean AUROC:", res["mean_auroc"])
    print(f"stack - auto_select: {vs[0]:+.4f} CI [{vs[1]:+.4f},{vs[2]:+.4f}] excl0={res['stack_vs_auto_select']['ci_excludes_zero']}")
    print(f"stacking beats selection on multivariate TS? {'YES' if (vs[0]>0 and res['stack_vs_auto_select']['ci_excludes_zero']) else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
