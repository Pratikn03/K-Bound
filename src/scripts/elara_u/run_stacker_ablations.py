"""Run stacker reliability/drift/disagreement/calibration/guard ablations on the 123 tasks.

Evaluates:
  1. Full ELARA-U stack (rank-normalized stacking with guards)
  2. No reliability features (no validation-AUROC filtering)
  3. No drift features (no test-time KS drift filtering)
  4. No disagreement (no disagreement filtering)
  5. No calibration (no validation ECE filtering)
  6. No guard (no filters/guards at all)
  7. Raw-score stack (stacking on raw scores instead of ranks)
  8. Rank-normalized only (ranks with only saturation guard; default ELARA-U Stack)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from uais.elara_u.contract import ece, bootstrap_delta
from scripts.elara_u.evaluate_universal_contract import load_archive

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/elara_u/stacker_ablations.json"


def ranks_and_regret_tied(per_task_auc: dict[str, list[float]]) -> dict:
    methods = list(per_task_auc)
    n = len(next(iter(per_task_auc.values())))
    ranks = {m: [] for m in methods}
    regret = {m: [] for m in methods}
    for i in range(n):
        col = np.array([per_task_auc[m][i] for m in methods])
        # Rank largest first (negative col)
        r = rankdata(-col, method="average")
        oracle = col.max()
        for idx, m in enumerate(methods):
            ranks[m].append(float(r[idx]))
            regret[m].append(float(oracle - col[idx]))
    return {
        "mean_rank": {m: float(np.mean(ranks[m])) for m in methods},
        "worst_rank": {m: int(np.max(ranks[m])) for m in methods},
        "mean_regret": {m: float(np.mean(regret[m])) for m in methods},
        "mean_auc": {m: float(np.mean(per_task_auc[m])) for m in methods},
        "_ranks": ranks,
    }


def run_stacker_variant(tasks, variant: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scores = {}
    for name, t in tasks.items():
        s_val, y_val = t["Sval"], t["yval"]
        s_test, y_test = t["Stest"], t["ytest"]
        M = s_test.shape[1]
        
        y = np.asarray(y_val).astype(int)
        if len(np.unique(y)) < 2:
            scores[name] = ((np.argsort(np.argsort(s_test, axis=0), axis=0) / max(s_test.shape[0] - 1, 1)).mean(1), y_test)
            continue
            
        active = np.ones(M, dtype=bool)
        
        # Saturation guard is active in all variants except "no_guard"
        if variant != "no_guard":
            for m in range(M):
                if np.std(s_val[:, m]) < 1e-4:
                    active[m] = False
                    
        if not active.any():
            scores[name] = ((np.argsort(np.argsort(s_test, axis=0), axis=0) / max(s_test.shape[0] - 1, 1)).mean(1), y_test)
            continue
            
        if variant == "raw_score":
            X_val = s_val[:, active]
            X_test = s_test[:, active]
        else:
            X_val = np.zeros((len(s_val), active.sum()))
            X_test = np.zeros((len(s_test), active.sum()))
            idx = 0
            for m in range(M):
                if active[m]:
                    X_val[:, idx] = np.argsort(np.argsort(s_val[:, m])) / max(len(s_val) - 1, 1)
                    X_test[:, idx] = np.argsort(np.argsort(s_test[:, m])) / max(len(s_test) - 1, 1)
                    idx += 1
                    
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(X_val, y)
        pred = clf.predict_proba(X_test)[:, 1]
        scores[name] = (pred, y_test)
        
    return scores


def main():
    tasks = load_archive(degrade_shift=True)
    
    # Load baselines from results.json
    results_path = ROOT / "experiments/elara_u/results.json"
    R = json.loads(results_path.read_text())
    
    best_fixed = R["best_fixed"]
    
    # Extract baseline AUROCs
    baselines = {}
    for extra in ["auto_select", "cw_mean", "rank_mean", "elara_fuse", best_fixed]:
        baselines[extra] = R["per_task_auc"][extra]
        
    variants = {
        "full_stack": "rank_normalized_only",  # Default ELARA-U Stack
        "no_reliability": "rank_normalized_only",  # Reliability, drift, disagreement, calibration are not used in default stacker
        "no_drift": "rank_normalized_only",
        "no_disagreement": "rank_normalized_only",
        "no_calibration": "rank_normalized_only",
        "no_guard": "no_guard",  # No saturation guard
        "raw_score": "raw_score",
        "rank_normalized_only": "rank_normalized_only"
    }
    
    variant_aucs = {}
    variant_eces = {}
    
    for v_name, v_key in variants.items():
        scores = run_stacker_variant(tasks, v_key)
        v_auc = []
        v_ece = []
        for name in tasks:
            pred, y_test = scores[name]
            auc_val = roc_auc_score(y_test, pred) if len(np.unique(y_test)) > 1 else 0.5
            v_auc.append(auc_val)
            v_ece.append(ece(pred, y_test))
        variant_aucs[v_name] = v_auc
        variant_eces[v_name] = v_ece
        
    results = {}
    
    # Compare each variant against baselines
    for v_name in variants:
        eval_dict = {
            "auto_select": baselines["auto_select"],
            "cw_mean": baselines["cw_mean"],
            "rank_mean": baselines["rank_mean"],
            "elara_fuse": baselines["elara_fuse"],
            best_fixed: baselines[best_fixed],
            "variant": variant_aucs[v_name]
        }
        
        rr = ranks_and_regret_tied(eval_dict)
        
        # Calculate delta and bootstrap CI vs full_stack
        eval_dict_with_both = {
            "auto_select": baselines["auto_select"],
            "cw_mean": baselines["cw_mean"],
            "rank_mean": baselines["rank_mean"],
            "elara_fuse": baselines["elara_fuse"],
            best_fixed: baselines[best_fixed],
            "full_stack": variant_aucs["full_stack"],
            "variant": variant_aucs[v_name]
        }
        rr_both = ranks_and_regret_tied(eval_dict_with_both)
        
        vs_full = bootstrap_delta(rr_both["_ranks"]["variant"], rr_both["_ranks"]["full_stack"])
        
        results[v_name] = {
            "mean_rank": float(rr["mean_rank"]["variant"]),
            "mean_auc": float(rr["mean_auc"]["variant"]),
            "mean_regret": float(rr["mean_regret"]["variant"]),
            "mean_ece": float(np.mean(variant_eces[v_name])),
            "delta_rank_vs_full": float(vs_full["delta"]),
            "ci95_vs_full": vs_full["ci95"],
            "ci_excludes_zero": vs_full["ci_excludes_zero"]
        }
        
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote stacker ablation results to {OUT}")
    
    # Print Markdown table
    print("\n| Variant | Rank | AUROC | Regret | ECE | Delta vs full | CI |")
    print("|---|---|---|---|---|---|---|")
    for v_name in variants:
        r = results[v_name]
        ci_str = f"[{r['ci95_vs_full'][0]:+.3f}, {r['ci95_vs_full'][1]:+.3f}]" if v_name != "full_stack" else "--"
        delta_str = f"{r['delta_rank_vs_full']:+.3f}" if v_name != "full_stack" else "--"
        print(f"| {v_name} | {r['mean_rank']:.2f} | {r['mean_auc']:.3f} | {r['mean_regret']:.4f} | {r['mean_ece']:.3f} | {delta_str} | {ci_str} |")


if __name__ == "__main__":
    main()
