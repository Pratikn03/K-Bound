"""BAF (Bank Account Fraud, Feedzai/NeurIPS-2022) fraud-family expansion (D26).

The fraud family was underpowered (n=1: credit-card) in leave-family-out CV. BAF
provides six bias-controlled variants (Base, I--V) of a large tabular fraud problem,
each a distinct distribution -> six additional fraud tasks (a second source). We run
the development-frozen pipeline (build_task scoring + rank-normalized logistic
stacking) on each variant and test whether stacking-beats-selection holds on fraud
beyond the single credit-card task. No test labels are used for any method.

Honest scope: the six variants share one base generator (bias-controlled), so they
are related rather than fully independent; we report them as a within-source fraud
panel that lifts the fraud family from n=1 to n=7.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.elara_u.build_score_archive import build_task
from scripts.elara_u.gate_u_seed_eval import _balance

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
BAF = ROOT / "data/raw/baf"
LABEL = "fraud_bool"


def _encode(df: pd.DataFrame):
    y = df[LABEL].astype(int).to_numpy()
    X = df.drop(columns=[LABEL])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):     # pyarrow-backed strings aren't 'object'
            X[c] = pd.factorize(X[c].astype(str))[0]
    return np.nan_to_num(X.to_numpy(dtype=float)), y


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    files = [f for f in sorted(glob.glob(str(BAF / "*.csv"))) if not os.path.basename(f).startswith("._")]
    per = {}
    names = []
    for f in files:
        df = pd.read_csv(f)
        X, y = _encode(df)
        X, y = _balance(X, y)                      # cap rows, preserve fraud rate (frozen pipeline)
        Sval, yval, Stest, ytest, det_names, vauc = build_task(X, y)
        if len(np.unique(ytest)) < 2 or len(np.unique(yval)) < 2:
            continue
        A = lambda s: roc_auc_score(ytest, s)
        row = {d: A(Stest[:, j]) for j, d in enumerate(det_names)}
        row["auto_select"] = A(Stest[:, int(np.argmax(vauc))])
        row["rank_mean"] = A((np.argsort(np.argsort(Stest, 0), 0) / (len(ytest) - 1)).mean(1))
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Sval, yval)
        row["stack"] = A(clf.predict_proba(Stest)[:, 1])
        for k, v in row.items():
            per.setdefault(k, []).append(float(v))
        names.append(os.path.basename(f).replace(".csv", ""))
        print(f"[{names[-1]:11}] auto={row['auto_select']:.3f} stack={row['stack']:.3f}", flush=True)

    pa = {m: np.array(v) for m, v in per.items()}
    dets = [d for d in pa if d not in ("auto_select", "rank_mean", "stack")]
    best_fixed = max(dets, key=lambda d: pa[d].mean())
    vs_auto = _boot(pa["stack"] - pa["auto_select"])
    vs_best = _boot(pa["stack"] - pa[best_fixed])
    res = {
        "protocol": "D26_BAF_FRAUD_PANEL (Feedzai NeurIPS-2022)", "n_tasks": len(names),
        "variants": names,
        "note": "6 bias-controlled variants of one base generator; related within-source fraud panel, lifts fraud family from n=1 to n=7 (with credit-card).",
        "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
        "best_fixed": best_fixed,
        "stack_vs_auto_select": {"mean": round(vs_auto[0], 4), "ci95": [round(vs_auto[1], 4), round(vs_auto[2], 4)],
                                 "ci_excludes_zero": vs_auto[1] > 0 or vs_auto[2] < 0},
        "stack_vs_best_fixed": {"mean": round(vs_best[0], 4), "ci95": [round(vs_best[1], 4), round(vs_best[2], 4)],
                                "ci_excludes_zero": vs_best[1] > 0 or vs_best[2] < 0},
    }
    out = ROOT / "experiments/elara_u/baf_fraud_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== D26 BAF FRAUD PANEL ({len(names)} variants) ===")
    for m in ["auto_select", best_fixed, "rank_mean", "stack"]:
        print(f"  {m:12} mean AUROC {res['mean_auroc'][m]:.4f}")
    print(f"stack - auto_select : {vs_auto[0]:+.4f} CI [{vs_auto[1]:+.4f},{vs_auto[2]:+.4f}] excl0={res['stack_vs_auto_select']['ci_excludes_zero']}")
    print(f"stack - best_fixed  : {vs_best[0]:+.4f} CI [{vs_best[1]:+.4f},{vs_best[2]:+.4f}] excl0={res['stack_vs_best_fixed']['ci_excludes_zero']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
