"""Path 1c: does stacking still beat selection with a DEEP detector in the zoo?

Robustness check against "classical detectors are weak, so routing is trivial."
Adds DeepSVDD to the 6 classical detectors and re-runs the contract. If a strong
deep detector dominated everywhere, auto-select would catch up; if stacking still
beats auto-select with the deep detector present, the win is robust to zoo strength.
No test labels are used for fitting.
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
from pyod.models.deep_svdd import DeepSVDD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.elara_u.gate_u_seed_eval import detector_zoo, load_tasks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
RNG = 0


def _zsig(raw, ref):
    mu, sd = float(np.mean(ref)), float(np.std(ref) + 1e-6)
    return 1.0 / (1.0 + np.exp(-(raw - mu) / sd))


def _deepsvdd_scores(Xtr, Xva, Xte):
    with contextlib.redirect_stdout(io.StringIO()):
        m = DeepSVDD(n_features=Xtr.shape[1], epochs=20, verbose=0)
        m.fit(Xtr)
        ref = m.decision_function(Xtr)
        return _zsig(m.decision_function(Xva), ref), _zsig(m.decision_function(Xte), ref)


def score(X, y):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=RNG, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=RNG, stratify=ytmp)
    sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    sval, stest, names = [], [], []
    for nm, ctor in detector_zoo().items():
        try:
            mdl = ctor().fit(Xtr); ref = mdl.decision_function(Xtr)
            sval.append(_zsig(mdl.decision_function(Xva), ref)); stest.append(_zsig(mdl.decision_function(Xte), ref))
        except Exception:
            sval.append(np.full(len(yva), 0.5)); stest.append(np.full(len(yte), 0.5))
        names.append(nm)
    try:
        dv, dt = _deepsvdd_scores(Xtr, Xva, Xte)
    except Exception:
        dv, dt = np.full(len(yva), 0.5), np.full(len(yte), 0.5)
    sval.append(dv); stest.append(dt); names.append("DeepSVDD")
    return np.column_stack(sval), yva, np.column_stack(stest), yte, names


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    tasks = load_tasks()
    per = {m: [] for m in ["auto_select", "stack", "DeepSVDD", "best_classical"]}
    n = 0
    for name, dom, X, y in tasks:
        Sval, yva, Stest, yte, names = score(X, y)
        if len(np.unique(yte)) < 2 or len(np.unique(yva)) < 2:
            continue
        n += 1
        vauc = np.array([roc_auc_score(yva, Sval[:, j]) if len(np.unique(yva)) > 1 else 0.5 for j in range(Sval.shape[1])])
        di = names.index("DeepSVDD")
        per["auto_select"].append(roc_auc_score(yte, Stest[:, int(np.argmax(vauc))]))
        per["DeepSVDD"].append(roc_auc_score(yte, Stest[:, di]))
        classical_auc = [roc_auc_score(yte, Stest[:, j]) for j in range(len(names)) if names[j] != "DeepSVDD"]
        per["best_classical"].append(max(classical_auc))
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yva)
        per["stack"].append(roc_auc_score(yte, clf.predict_proba(Stest)[:, 1]))
        if n % 25 == 0:
            print(f"  ...{n} tasks", flush=True)
    pa = {m: np.array(v) for m, v in per.items()}
    vs_auto = _boot(pa["stack"] - pa["auto_select"])
    res = {"protocol": "PATH1C_DEEP_ZOO_v1", "n_tasks": n, "zoo": "6 classical + DeepSVDD",
           "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
           "stack_vs_auto_select": {"mean": round(vs_auto[0], 4), "ci95": [round(vs_auto[1], 4), round(vs_auto[2], 4)],
                                    "ci_excludes_zero": vs_auto[1] > 0 or vs_auto[2] < 0}}
    out = ROOT / "experiments/elara_u/deep_zoo_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== Path 1c: deep zoo ({n} tasks, 6 classical + DeepSVDD) ===")
    print("mean AUROC:", res["mean_auroc"])
    print(f"stack - auto_select: {vs_auto[0]:+.4f} CI [{vs_auto[1]:+.4f},{vs_auto[2]:+.4f}] excl0={res['stack_vs_auto_select']['ci_excludes_zero']}")
    print(f"stacking still beats selection with deep detector in zoo? {'YES' if (vs_auto[0]>0 and res['stack_vs_auto_select']['ci_excludes_zero']) else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
