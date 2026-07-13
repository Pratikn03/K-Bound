import pytest
import numpy as np

from kbound_edge.metrics import emitted_predictions, evaluate

@pytest.fixture
def example_outcome():
    class Outcome:
        p0 = np.array([[0.1, 0.9], [0.8, 0.2]])
        pa = np.array([[0.9, 0.1], [0.2, 0.8]])
    return Outcome()

def test_abstain_emits_frozen_prediction(example_outcome):
    emitted = emitted_predictions("abstain", example_outcome.p0, example_outcome.pa)
    np.testing.assert_array_equal(emitted, example_outcome.p0)

def test_false_adapt_definitions():
    decisions = ["adapt", "adapt", "freeze", "abstain"]
    delta = np.array([0.2, -0.1, -0.2, 0.3])
    m = evaluate(decisions, delta, np.zeros(4))
    assert m["false_adapt_uncond"] == 0.25
    assert m["false_adapt_cond"] == 0.50

def test_bootstrap_real_metrics():
    from kbound_edge.metrics import bootstrap_real_metrics
    class Outcome:
        def __init__(self, p0, pa):
            self.p0 = p0
            self.pa = pa

    outcomes = [
        Outcome(np.array([[0.1, 0.9]]), np.array([[0.9, 0.1]])),
        Outcome(np.array([[0.8, 0.2]]), np.array([[0.2, 0.8]])),
    ]
    true_labels = [np.array([0]), np.array([1])]
    window_metadata = [
        {"session_id": "S07", "object_id": "P09", "shift_id": "mild_light"},
        {"session_id": "S07", "object_id": "P09", "shift_id": "mild_light"},
    ]
    policy_decisions = {
        "kga_full": ["adapt", "freeze"]
    }
    latencies_ms = [50.0, 60.0]

    res = bootstrap_real_metrics(
        outcomes, true_labels, window_metadata, policy_decisions, latencies_ms, n_boot=10, seed=123
    )
    assert "kga_full" in res
    assert "balanced_acc" in res["kga_full"]
    assert "val" in res["kga_full"]["balanced_acc"]
    assert "ci" in res["kga_full"]["balanced_acc"]
