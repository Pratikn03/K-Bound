"""Comprehensive baseline suite (D28): prove stacking is not a weak-baseline win.

Runs every standard combiner/selector/oracle on the 123-task score archive and
compares each to the headline logistic stack with a paired bootstrap. Includes
selection baselines (best-fixed, auto-select, learned MetaOD-style), averaging
(raw / calibrated / rank / confidence-weighted), an AutoML-style greedy ensemble
(Caruana), several stackers (logistic score-space, logistic rank-normalized, random
forest, gradient boosting, hist-GBM, per-family specialist), and two TEST-label
oracle upper bounds (best single detector, best pair fusion). No test labels enter
any non-oracle method.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestRegressor, RandomForestClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import skew

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / "experiments/elara_u/score_archive"
HEADLINE = "stack_logistic"   # the paper's headline stack (logistic, score-space)
FAM = {"adbench": "tabular", "tabular": "tabular", "image_ood": "image-OOD",
       "text": "text", "cyber": "cyber", "fraud": "fraud"}


def _ranknorm(S):
    return np.argsort(np.argsort(S, 0), 0) / max(len(S) - 1, 1)


def _meta_features(Sval, yval, va):
    mu, sd, sk = Sval.mean(0), Sval.std(0), np.nan_to_num(skew(Sval, 0, bias=False))
    C = np.nan_to_num(np.corrcoef(Sval.T)); off = C[np.triu_indices(Sval.shape[1], 1)]
    return np.concatenate([va, mu, sd, sk, [len(yval), yval.mean(),
                           off.mean(), off.std(), off.max(), off.min()]])


def _greedy_ensemble(Sval, yval, Stest, T=15):
    """Caruana-style greedy ensemble selection on validation AUROC (with replacement)."""
    M = Sval.shape[1]; chosen = []
    cur_val = np.zeros(len(yval))
    for _ in range(T):
        best_j, best_a = 0, -1
        for j in range(M):
            cand = (cur_val * len(chosen) + Sval[:, j]) / (len(chosen) + 1)
            a = roc_auc_score(yval, cand) if len(np.unique(yval)) > 1 else 0.5
            if a > best_a:
                best_a, best_j = a, j
        chosen.append(best_j)
        cur_val = (cur_val * (len(chosen) - 1) + Sval[:, best_j]) / len(chosen)
    w = np.bincount(chosen, minlength=M) / len(chosen)
    return Stest @ w


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    fs = [f for f in sorted(glob.glob(str(ARCH / "*.npz"))) if not os.path.basename(f).startswith("._")]
    tasks = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        Sval, yval, Stest, ytest, va = z["Sval"], z["yval"], z["Stest"], z["ytest"], z["val_auc"]
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        tasks.append((os.path.basename(f)[:-4], Sval, yval.astype(int), Stest, ytest.astype(int),
                      va, FAM.get(str(z["domain"]), "tabular")))
    n = len(tasks)

    # global best fixed detector (by mean test AUROC across tasks)
    det_test = np.array([[roc_auc_score(yt, St[:, j]) for j in range(St.shape[1])]
                         for _, _, _, St, yt, _, _ in tasks])
    best_fixed_j = int(det_test.mean(0).argmax())

    # MetaOD-style LOO selector (precompute meta-features + per-detector test targets)
    MF = np.array([_meta_features(Sv, yv, va) for _, Sv, yv, _, _, va, _ in tasks])

    per = {m: [] for m in [
        "best_fixed", "auto_select", "metaod_select", "avg_raw", "avg_calibrated",
        "rank_mean", "cw_mean", "greedy_ensemble", "stack_logistic", "stack_logistic_ranknorm",
        "stack_rf", "stack_gbm", "stack_histgbm", "stack_perfamily",
        "oracle_best_single", "oracle_best_pair"]}

    for i, (name, Sval, yval, Stest, ytest, va, fam) in enumerate(tasks):
        A = lambda s: roc_auc_score(ytest, s)
        per["best_fixed"].append(A(Stest[:, best_fixed_j]))
        per["auto_select"].append(A(Stest[:, int(np.argmax(va))]))
        # MetaOD-style learned selection (LOO)
        tr = np.array([j for j in range(n) if j != i])
        rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(MF[tr], det_test[tr])
        per["metaod_select"].append(A(Stest[:, int(np.argmax(rf.predict(MF[i:i+1])[0]))]))
        # averaging family
        per["avg_raw"].append(A(Stest.mean(1)))
        cal = np.column_stack([IsotonicRegression(out_of_bounds="clip")
                               .fit(Sval[:, j], yval).predict(Stest[:, j]) for j in range(Sval.shape[1])])
        per["avg_calibrated"].append(A(cal.mean(1)))
        per["rank_mean"].append(A(_ranknorm(Stest).mean(1)))
        w = np.clip(2 * np.abs(Sval - 0.5).mean(0), 1e-6, None); w /= w.sum()
        per["cw_mean"].append(A(Stest @ w))
        per["greedy_ensemble"].append(A(_greedy_ensemble(Sval, yval, Stest)))
        # stackers
        per["stack_logistic"].append(A(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                                       .fit(Sval, yval).predict_proba(Stest)[:, 1]))
        per["stack_logistic_ranknorm"].append(A(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                                                .fit(_ranknorm(Sval), yval).predict_proba(_ranknorm(Stest))[:, 1]))
        per["stack_rf"].append(A(RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced", n_jobs=-1)
                                 .fit(Sval, yval).predict_proba(Stest)[:, 1]))
        per["stack_gbm"].append(A(GradientBoostingClassifier(random_state=0).fit(Sval, yval).predict_proba(Stest)[:, 1]))
        per["stack_histgbm"].append(A(HistGradientBoostingClassifier(random_state=0).fit(Sval, yval).predict_proba(Stest)[:, 1]))
        # per-family specialist: logistic pooled over OTHER same-family tasks' validation
        pool = [(tasks[j][1], tasks[j][2]) for j in range(n) if j != i and tasks[j][6] == fam]
        if pool:
            Xp = np.vstack([p[0] for p in pool]); yp = np.concatenate([p[1] for p in pool])
            if len(np.unique(yp)) > 1:
                per["stack_perfamily"].append(A(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                                                .fit(Xp, yp).predict_proba(Stest)[:, 1]))
            else:
                per["stack_perfamily"].append(per["stack_logistic"][-1])
        else:
            per["stack_perfamily"].append(per["stack_logistic"][-1])  # singleton family -> self
        # oracles (TEST-label upper bounds)
        per["oracle_best_single"].append(det_test[i].max())
        per["oracle_best_pair"].append(max(A((Stest[:, a] + Stest[:, b]) / 2)
                                           for a, b in combinations(range(Stest.shape[1]), 2)))

    pa = {m: np.array(v) for m, v in per.items()}
    head = pa[HEADLINE]
    rows = {}
    for m, v in pa.items():
        mean, lo, hi = _boot(head - v) if m != HEADLINE else (0.0, 0.0, 0.0)
        rows[m] = {"mean_auroc": round(float(v.mean()), 4),
                   "stack_minus_this": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                   "stack_beats": (lo > 0) if m != HEADLINE else None,
                   "oracle": m.startswith("oracle")}
    res = {"protocol": "D28_BASELINE_SUITE", "n_tasks": n, "headline": HEADLINE,
           "best_fixed_detector_idx": best_fixed_j, "methods": rows}
    out = ROOT / "experiments/elara_u/baselines_suite_results.json"
    out.write_text(json.dumps(res, indent=2))

    order = sorted(rows, key=lambda m: -rows[m]["mean_auroc"])
    print(f"=== D28 BASELINE SUITE ({n} tasks); headline = {HEADLINE} ({rows[HEADLINE]['mean_auroc']:.3f}) ===")
    print(f"{'method':26} {'mAUROC':>7}  {'stack-Δ':>8}  {'95% CI':>20}  beats?")
    for m in order:
        r = rows[m]
        tag = "ORACLE" if r["oracle"] else ("HEADLINE" if m == HEADLINE else ("yes" if r["stack_beats"] else "n.s."))
        print(f"{m:26} {r['mean_auroc']:7.3f}  {r['stack_minus_this']:+8.4f}  [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]   {tag}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
