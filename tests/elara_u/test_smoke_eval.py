"""Smoke test: the multimodal reliability contract logic runs and is internally
consistent on synthetic independent-failure data (no archive/GPU needed)."""
import numpy as np
from scripts.elara_u.multimodal_reliability_test import run_category


def test_reliability_gate_recovers_under_independent_failure():
    rng = np.random.default_rng(0)
    n, M = 300, 3
    yval = (rng.random(n) < 0.3).astype(int)
    ytest = (rng.random(n) < 0.3).astype(int)
    d = np.array([2.4, 1.2, 0.6])   # modality 0 strongest on val
    sig = lambda y: np.column_stack([1 / (1 + np.exp(-(d[m] * y + rng.standard_normal(n)))) for m in range(M)])
    Sval, Stest = sig(yval), sig(ytest)
    from sklearn.metrics import roc_auc_score
    vauc = np.array([roc_auc_score(yval, Sval[:, m]) for m in range(M)])
    clean = run_category(Sval, yval, Stest, ytest, vauc, fail=False)
    fail = run_category(Sval, yval, Stest, ytest, vauc, fail=True)
    # clean: gate inert (== validation-only fusion); failure: gate recovers above stale-select
    assert abs(clean["reliability_gate"] - clean["no_reliability"]) < 1e-9
    assert fail["reliability_gate"] >= fail["stale_auto_select"]
