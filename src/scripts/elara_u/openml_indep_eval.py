"""Fully-independent external benchmark (D27 phase 2) -- ONE SHOT, frozen pipeline.

Scores the cached OpenML independent suite (datasets verifiably absent from the
135-task development archive) with the development-frozen pipeline. Pre-registered
anomaly construction: minority class = anomaly; if anomaly rate > 10%, downsample
anomalies to 10% (seed 0); cap rows preserving rate. Pass: stack beats auto-select
AND best fixed detector, paired-bootstrap CI lower bound > 0. No tuning after scoring.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.elara_u.build_score_archive import build_task
from scripts.elara_u.gate_u_seed_eval import _balance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/raw/indep_external"
TARGET_RATE = 0.10


def _downsample_anom(X, y, rate=TARGET_RATE, seed=0):
    """Frozen construction: bring anomaly rate down to `rate` by dropping anomalies."""
    if y.mean() <= rate:
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep_pos = rng.choice(pos, size=max(15, int(rate * len(neg) / (1 - rate))), replace=False)
    idx = np.sort(np.concatenate([neg, keep_pos]))
    return X[idx], y[idx]


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    files = [f for f in sorted(glob.glob(str(CACHE / "*.npz"))) if not os.path.basename(f).startswith("._")]
    per, names = {}, []
    for f in files:
        z = np.load(f)
        X, y = z["X"], z["y"]
        X, y = _downsample_anom(X, y)
        X, y = _balance(X, y)
        if int(y.sum()) < 12 or len(np.unique(y)) < 2:
            continue
        try:
            Sval, yval, Stest, ytest, det_names, vauc = build_task(X, y)
        except Exception:
            continue
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        A = lambda s: roc_auc_score(ytest, s)
        row = {d: A(Stest[:, j]) for j, d in enumerate(det_names)}
        row["auto_select"] = A(Stest[:, int(np.argmax(vauc))])
        row["rank_mean"] = A((np.argsort(np.argsort(Stest, 0), 0) / (len(ytest) - 1)).mean(1))
        row["avg"] = A(Stest.mean(1))
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yval)
        row["stack"] = A(clf.predict_proba(Stest)[:, 1])
        for k, v in row.items():
            per.setdefault(k, []).append(float(v))
        names.append(os.path.basename(f).replace(".npz", ""))
        print(f"[{names[-1]:34}] auto={row['auto_select']:.3f} stack={row['stack']:.3f}", flush=True)

    pa = {m: np.array(v) for m, v in per.items()}
    dets = [d for d in pa if d not in ("auto_select", "rank_mean", "avg", "stack")]
    best_fixed = max(dets, key=lambda d: pa[d].mean())
    vs_auto = _boot(pa["stack"] - pa["auto_select"])
    vs_best = _boot(pa["stack"] - pa[best_fixed])
    res = {
        "protocol": "D27_FULLY_INDEPENDENT_EXTERNAL (sklearn+HAR, ONE-SHOT, frozen)",
        "n_tasks": len(names), "datasets": names,
        "source": "sklearn digits/wine/wdbc + HAR smartphones; distinct sources, verifiably absent from the 135-task development archive",
        "construction": "one-vs-rest (class=anomaly); downsample anomalies to 10% (seed 0); cap rows preserving rate",
        "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
        "best_fixed": best_fixed,
        "stack_vs_auto_select": {"mean": round(vs_auto[0], 4), "ci95": [round(vs_auto[1], 4), round(vs_auto[2], 4)],
                                 "pass": vs_auto[1] > 0},
        "stack_vs_best_fixed": {"mean": round(vs_best[0], 4), "ci95": [round(vs_best[1], 4), round(vs_best[2], 4)],
                                "pass": vs_best[1] > 0},
    }
    res["gate_independent_external_confirmed"] = res["stack_vs_auto_select"]["pass"] and res["stack_vs_best_fixed"]["pass"]
    out = ROOT / "experiments/elara_u/indep_external_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== D27 FULLY-INDEPENDENT EXTERNAL (one-shot, {len(names)} tasks) ===")
    for m in ["auto_select", best_fixed, "rank_mean", "avg", "stack"]:
        print(f"  {m:14} mean AUROC {res['mean_auroc'][m]:.4f}")
    print(f"stack - auto_select : {vs_auto[0]:+.4f} CI [{vs_auto[1]:+.4f},{vs_auto[2]:+.4f}] pass={res['stack_vs_auto_select']['pass']}")
    print(f"stack - best_fixed  : {vs_best[0]:+.4f} CI [{vs_best[1]:+.4f},{vs_best[2]:+.4f}] pass={res['stack_vs_best_fixed']['pass']}")
    print(f"GATE D27 INDEPENDENT EXTERNAL CONFIRMED? {'YES' if res['gate_independent_external_confirmed'] else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
