"""Regression guard: ELARA-U routing/stacking scores must not depend on TEST labels.
Permuting ytest must leave every routed test score bit-identical (ytest is only used
to compute final AUROC, never to route)."""
import numpy as np
import pytest
from scripts.elara_u.calibration_eval import task_probs


def _synthetic_task(seed=0, n=200, M=4):
    rng = np.random.default_rng(seed)
    yval = (rng.random(n) < 0.25).astype(int)
    ytest = (rng.random(n) < 0.25).astype(int)
    d = rng.uniform(1.2, 2.4, M)
    sig = lambda y: np.column_stack([1 / (1 + np.exp(-(d[m] * y + rng.standard_normal(n)))) for m in range(M)])
    Sval, Stest = sig(yval), sig(ytest)
    from sklearn.metrics import roc_auc_score
    val_auc = np.array([roc_auc_score(yval, Sval[:, m]) for m in range(M)])
    return {"Sval": Sval, "yval": yval, "Stest": Stest, "ytest": ytest, "val_auc": val_auc}


def test_routed_scores_invariant_to_test_label_permutation():
    t = _synthetic_task()
    base = {k: v[1].copy() for k, v in task_probs(t).items()}   # test_prob per method
    t2 = dict(t); rng = np.random.default_rng(99)
    t2["ytest"] = t["ytest"][rng.permutation(len(t["ytest"]))]   # shuffle TEST labels only
    after = {k: v[1] for k, v in task_probs(t2).items()}
    for m in base:
        np.testing.assert_array_equal(base[m], after[m],
                                      err_msg=f"{m} test scores changed when ytest was permuted (leakage!)")
