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


# --- Router (D23 reliability path) ---------------------------------------------

def test_router_functions_take_no_test_labels():
    """Static guard: routing entry points must not accept any test-label argument."""
    import inspect
    from uais.elara_u import router
    for fn in (router.route, router.super_route, router.fuse, router.select,
               router.reliability_features):
        params = set(inspect.signature(fn).parameters)
        leaky = {p for p in params if p.lower() in {"y_test", "ytest", "y_te", "labels_test"}}
        assert not leaky, f"{fn.__name__} exposes test-label argument(s): {leaky}"


def test_super_route_scores_depend_only_on_val_and_test_scores():
    """route()/super_route() take (s_val, y_val, s_test, policy) only; the routed test
    score must be identical no matter what (separate) test labels exist."""
    from uais.elara_u.router import route, RouterPolicy
    t = _synthetic_task(seed=3, n=300, M=5)
    pol = RouterPolicy()
    s1, a1 = route(t["Sval"], t["yval"], t["Stest"], pol, action="hybrid")
    # an entirely different ytest must not (and cannot) reach the router:
    s2, a2 = route(t["Sval"], t["yval"], t["Stest"], pol, action="hybrid")
    np.testing.assert_array_equal(s1, s2)
    assert a1 == a2


def test_auto_select_index_invariant_to_test_label_permutation():
    """Auto-select uses validation AUROC only; the chosen detector must not move when
    test labels are permuted."""
    from uais.elara_u.router import select
    t = _synthetic_task(seed=5)
    base = select(t["Stest"], t["val_auc"])
    # val_auc is computed from yval; permuting ytest changes nothing upstream of select
    np.testing.assert_array_equal(base, select(t["Stest"], t["val_auc"]))


# --- D25 meta-selection (MetaOD-style) -----------------------------------------

def test_meta_features_take_no_test_labels():
    import inspect
    from scripts.elara_u.metaod_baseline import meta_features
    params = set(inspect.signature(meta_features).parameters)
    assert params == {"Sval", "yval", "val_auc"}, f"meta_features signature drifted: {params}"


def test_meta_selection_choice_invariant_to_heldout_test_labels():
    """The MetaOD-style selector's pick for a held-out task is a function of its
    validation-only meta-features plus OTHER tasks' history; permuting the held-out
    task's test labels must not change which detector is selected for it."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import roc_auc_score
    from scripts.elara_u.metaod_baseline import meta_features

    tasks = [_synthetic_task(seed=s, n=300, M=5) for s in range(8)]
    MF = np.array([meta_features(t["Sval"], t["yval"], t["val_auc"]) for t in tasks])
    TESTAUC = np.array([[roc_auc_score(t["ytest"], t["Stest"][:, j]) for j in range(5)] for t in tasks])
    i = 0  # held-out task

    def pick(testauc):
        tr = np.array([j for j in range(len(tasks)) if j != i])
        rf = RandomForestRegressor(n_estimators=60, random_state=0).fit(MF[tr], testauc[tr])
        return int(np.argmax(rf.predict(MF[i:i + 1])[0]))

    base = pick(TESTAUC)
    perm = TESTAUC.copy()
    rng = np.random.default_rng(7)
    yperm = tasks[i]["ytest"][rng.permutation(len(tasks[i]["ytest"]))]   # permute held-out labels
    perm[i] = np.array([roc_auc_score(yperm, tasks[i]["Stest"][:, j]) for j in range(5)])
    assert pick(perm) == base, "meta-selector pick changed when held-out test labels were permuted (leakage!)"
