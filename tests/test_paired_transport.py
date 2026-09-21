"""Independent analytic and fault-injection checks for transport certificates."""

import importlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("kga.paired_transport") is not None, "transport implementation missing"
    return importlib.import_module("kga.paired_transport")


def example(rho=0.0, contract="synthetic-fixed-pair-iid-transport"):
    p = api()
    a = np.array([[0.2, 0.2, 0], [0.8, 0.8, 0], [0, 0, 1.0]])
    q = np.array([0.2, 0.8, 0.0])
    c = p.accuracy_contrasts([(2, 0), (2, 1), (2, 2)], 3)
    return p.TransportSpec(a, a, q, q, c, rho=rho, assumption_contract=contract)


def test_rank_deficient_model_can_identify_positive_benefit():
    result = api().solve_benefit(example())
    assert result.verified and result.action == "ADAPT"
    assert result.lower == pytest.approx(0.2, abs=1e-8)
    assert result.upper == pytest.approx(0.8, abs=1e-8)
    assert result.lower <= 0.2 and result.upper >= 0.8


def test_transport_budget_crosses_zero_at_one_sixth():
    p = api()
    result = p.break_even_budget(example())
    assert result.verified
    assert result.lower <= 1 / 6
    assert result.upper is None  # approximate primal feasibility is not a rigorous upper bound
    assert result.lower == pytest.approx(1 / 6, abs=1e-7)
    assert p.solve_benefit(example(0.10)).lower == pytest.approx(0.08, abs=1e-8)
    assert p.solve_benefit(example(1 / 6)).action == "ABSTAIN"
    assert p.solve_benefit(example(0.20)).action == "ABSTAIN"


def test_unrestricted_violation_restores_both_signs():
    result = api().solve_benefit(example(1.0))
    assert result.lower == pytest.approx(-1.0)
    assert result.upper == pytest.approx(1.0)


def test_declared_assumptions_are_required_for_action():
    result = api().solve_benefit(example(contract=None))
    assert result.action == "ABSTAIN"
    assert result.status == "assumptions_not_declared"


def test_negative_benefit_and_zero_ties_use_strict_semantics():
    p = api()
    c = p.accuracy_contrasts([(0, 1), (1, 0)], 2)
    spec = p.TransportSpec(np.eye(2), np.eye(2), [0.5, 0.5], [0.5, 0.5], c, assumption_contract="analytic")
    assert p.solve_benefit(spec).action == "FREEZE"
    zero = p.TransportSpec(
        np.eye(2), np.eye(2), [0.5, 0.5], [0.5, 0.5], np.zeros((2, 2)), assumption_contract="analytic"
    )
    result = p.solve_benefit(zero)
    assert result.action == "ABSTAIN" and result.lower <= 0 <= result.upper


def test_empty_polytope_fails_closed():
    p = api()
    a = np.array([[1.0, 1.0], [0.0, 0.0]])
    spec = p.TransportSpec(a, a, [0.0, 1.0], [0.0, 1.0], np.zeros((2, 2)), assumption_contract="analytic")
    result = p.solve_benefit(spec)
    assert result.status == "infeasible"
    assert not result.verified and result.action == "ABSTAIN"
    assert result.lower is None and result.upper is None


def test_missing_source_labels_do_not_get_zero_uncertainty():
    p = api()
    bands = p.confidence_bounds(np.zeros((1, 2), dtype=int), [100], alpha=0.05)
    assert np.array_equal(bands.source_lower, [[0.0, 0.0]])
    assert np.array_equal(bands.source_upper, [[1.0, 1.0]])
    spec = p.TransportSpec.from_confidence(bands, p.accuracy_contrasts([(0, 1)], 2), assumption_contract="analytic")
    result = p.solve_benefit(spec)
    assert result.lower == pytest.approx(-1.0) and result.upper == pytest.approx(1.0)
    assert result.action == "ABSTAIN"


def test_exact_binomial_intervals_have_finite_sample_coverage():
    p = api()
    n, alpha = 8, 0.05
    intervals = [p.binomial_interval(x, n, alpha) for x in range(n + 1)]
    assert intervals[0][0] == 0 and intervals[-1][1] == 1
    for truth in np.linspace(0.01, 0.99, 43):
        covered = sum(
            math.comb(n, x) * truth**x * (1 - truth) ** (n - x)
            for x, (lo, hi) in enumerate(intervals)
            if lo <= truth <= hi
        )
        assert covered >= 1 - alpha - 1e-12


@pytest.mark.parametrize("counts", [[[1.5, 2]], [[-1, 2]], [[True, False]], [[True, 2]], [[float("nan"), 1]]])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises((ValueError, TypeError)):
        api().confidence_bounds(counts, [10], alpha=0.05)


def test_incoherent_or_nonfinite_probability_boxes_are_rejected():
    p = api()
    with pytest.raises(ValueError):
        p.TransportSpec([[0.8], [0.8]], [[1.0], [1.0]], [0.5, 0.5], [0.5, 0.5], [[0.0], [0.0]])
    with pytest.raises(ValueError):
        p.TransportSpec([[float("nan")]], [[1.0]], [1.0], [1.0], [[0.0]])
    with pytest.raises(ValueError):
        p.TransportSpec([[0.0]], [[1.0]], [1.0], [1.0], [[0.0]], rho=-0.1)


def test_pair_coupling_is_never_worse_than_separate_accuracy_extrema():
    p = api()
    # Identical predictors: paired benefit is exactly zero even with uncertain accuracy.
    a = np.ones((1, 2))
    spec = p.TransportSpec(a, a, [1.0], [1.0], [[0.0, 0.0]], assumption_contract="analytic")
    paired = p.solve_benefit(spec)
    separate = p.separate_accuracy_interval(spec, [[1.0, 0.0]], [[1.0, 0.0]])
    assert paired.lower >= -1e-8 and paired.upper <= 1e-8
    assert separate.lower == pytest.approx(-1.0) and separate.upper == pytest.approx(1.0)
    with pytest.raises(ValueError):
        p.separate_accuracy_interval(spec, [[1.0, 0.0]], [[0.0, 1.0]])


def test_dual_certificate_corrects_infeasible_stationarity():
    p = api()
    # min x, -x <= -.25, 0 <= x <= 1 has exact minimum .25.
    bound = p.conservative_dual_bound([1.0], [[-1.0]], [-0.25], [], [], [-1.0], [])
    assert bound <= 0.25 and bound == pytest.approx(0.25)
    # An inaccurate dual remains safe after residual correction.
    perturbed = p.conservative_dual_bound([1.0], [[-1.0]], [-0.25], [], [], [-1.01], [])
    assert perturbed < 0.25
    with pytest.raises(ValueError):
        p.conservative_dual_bound([1.0], [[-1.0]], [-0.25], [], [], [float("nan")], [])


def test_corrupted_solver_witness_cannot_produce_a_certificate(monkeypatch):
    p = api()
    original = p.linprog

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.success:
            result.x[:] = 2.0
        return result

    monkeypatch.setattr(p, "linprog", corrupt)
    result = p.solve_benefit(example())
    assert not result.verified and result.action == "ABSTAIN"
    assert result.status == "numerical_verification_failed"


def test_interval_contains_every_prior_on_a_small_exact_grid():
    p = api()
    a = np.array([[0.2, 0.2, 0], [0.8, 0.8, 0], [0, 0, 1.0]])
    c = p.accuracy_contrasts([(2, 0), (2, 1), (2, 2)], 3)
    for i in range(5):
        for j in range(5 - i):
            pi = np.array([i, j, 4 - i - j]) / 4
            q = a @ pi
            spec = p.TransportSpec(a, a, q, q, c, assumption_contract="analytic")
            result = p.solve_benefit(spec)
            truth = float(np.sum(c * a * pi))
            assert result.verified
            assert result.lower <= truth + 1e-12 <= result.upper + 1e-12


def test_diagnostic_requires_immutable_seal_and_refuses_occupied_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/run_paired_transport_diagnostic.py"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 9,
                "alpha": 0.05,
                "repetitions": 1,
                "sample_sizes": [10],
                "rho_grid": [0.0],
                "scenarios": [
                    {
                        "id": "tiny",
                        "evidence_group": "same",
                        "pairs": [[0, 1]],
                        "source_conditional": [[1.0, 1.0]],
                        "target_joint": [[0.2, 0.8]],
                    }
                ],
            }
        )
    )
    out = tmp_path / "run"
    command = [sys.executable, str(script)]
    sealed = subprocess.run(
        command + ["seal", "--protocol", str(protocol), "--output", str(out)], capture_output=True, text=True
    )
    assert sealed.returncode == 0, sealed.stderr
    again = subprocess.run(
        command + ["seal", "--protocol", str(protocol), "--output", str(out)], capture_output=True, text=True
    )
    assert again.returncode != 0
    (out / "protocol.json").write_text("{}")
    run = subprocess.run(command + ["run", "--output", str(out)], capture_output=True, text=True)
    assert run.returncode != 0
    assert not (out / "decisions.jsonl").exists()


def test_diagnostic_decisions_do_not_change_with_hidden_labels(tmp_path):
    script = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/run_paired_transport_diagnostic.py"
    protocol = tmp_path / "protocol.json"
    scenarios = [
        {
            "id": name,
            "evidence_group": "same",
            "pairs": [[0, 1]],
            "source_conditional": [[1.0, 1.0]],
            "target_joint": [joint],
        }
        for name, joint in [("helpful", [0.2, 0.8]), ("harmful", [0.8, 0.2])]
    ]
    protocol.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 9,
                "alpha": 0.05,
                "repetitions": 1,
                "sample_sizes": [10],
                "rho_grid": [0.0],
                "scenarios": scenarios,
            }
        )
    )
    out = tmp_path / "run"
    command = [sys.executable, str(script)]
    for args in (["seal", "--protocol", str(protocol), "--output", str(out)], ["run", "--output", str(out)]):
        result = subprocess.run(command + args, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    decisions = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    for arm in ("paired_lp", "separate_accuracy", "plugin"):
        chosen = [r for r in decisions if r["arm"] == arm]
        assert len(chosen) == 2
        for field in ("lower", "upper", "action", "input_sha256"):
            assert chosen[0][field] == chosen[1][field]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    again = subprocess.run(command + ["run", "--output", str(out)], capture_output=True, text=True)
    assert again.returncode != 0
