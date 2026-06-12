"""Sealed external holdout evaluation (D24) -- ONE SHOT, frozen pipeline.

Runs the development-frozen pipeline (build_task scoring + rank-normalized logistic
stacking) on UNTOUCHED ADBench feature sets (CV_by_ViT, NLP_by_RoBERTa) that were
never loaded or tuned during development. Pass: stack beats auto-select AND best
fixed detector with paired-bootstrap CI lower bound > 0. No tuning after scoring.
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
SEALED = [("data/raw/adbench_sealed_cv", "image_ood_vit"),
          ("data/raw/adbench_sealed_nlp", "text_roberta")]


def _boot(diff, seed=0):
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    return float(diff.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    per = {}
    fams = []
    for sub, fam in SEALED:
        for f in sorted(glob.glob(str(ROOT / sub / "*.npz"))):
            if os.path.basename(f).startswith("._"):
                continue
            z = np.load(f)
            X, y = np.nan_to_num(z["X"].astype(float)), z["y"].astype(int)
            if X.shape[0] < 80 or int(y.sum()) < 12 or len(np.unique(y)) < 2:
                continue
            X, y = _balance(X, y)
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
            fams.append(fam)

    pa = {m: np.array(v) for m, v in per.items()}
    dets = [d for d in pa if d not in ("auto_select", "rank_mean", "avg", "stack")]
    best_fixed = max(dets, key=lambda d: pa[d].mean())
    vs_auto = _boot(pa["stack"] - pa["auto_select"])
    vs_best = _boot(pa["stack"] - pa[best_fixed])
    n = len(fams)
    res = {
        "protocol": "D24_SEALED_EXTERNAL (ONE-SHOT, frozen)", "n_tasks": n,
        "holdout": "ADBench CV_by_ViT + NLP_by_RoBERTa (untouched extractors)",
        "families": {f: fams.count(f) for f in set(fams)},
        "mean_auroc": {m: round(float(v.mean()), 4) for m, v in pa.items()},
        "best_fixed": best_fixed,
        "stack_vs_auto_select": {"mean": round(vs_auto[0], 4), "ci95": [round(vs_auto[1], 4), round(vs_auto[2], 4)],
                                 "pass": vs_auto[1] > 0},
        "stack_vs_best_fixed": {"mean": round(vs_best[0], 4), "ci95": [round(vs_best[1], 4), round(vs_best[2], 4)],
                                "pass": vs_best[1] > 0},
    }
    res["gate_sealed_external_confirmed"] = res["stack_vs_auto_select"]["pass"] and res["stack_vs_best_fixed"]["pass"]
    out = ROOT / "experiments/elara_u/sealed_external_results.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"=== D24 SEALED EXTERNAL (one-shot, {n} untouched tasks) ===")
    print(f"holdout: {res['holdout']}  families: {res['families']}")
    for m in ["auto_select", best_fixed, "rank_mean", "avg", "stack"]:
        print(f"  {m:14} mean AUROC {res['mean_auroc'][m]:.4f}")
    print(f"\nstack - auto_select : {vs_auto[0]:+.4f} CI [{vs_auto[1]:+.4f},{vs_auto[2]:+.4f}] pass={res['stack_vs_auto_select']['pass']}")
    print(f"stack - best_fixed  : {vs_best[0]:+.4f} CI [{vs_best[1]:+.4f},{vs_best[2]:+.4f}] pass={res['stack_vs_best_fixed']['pass']}")
    print(f"GATE D24 SEALED CONFIRMED? {'YES' if res['gate_sealed_external_confirmed'] else 'NO'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
