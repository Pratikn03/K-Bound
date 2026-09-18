"""Real fit/calibrate/route/score regressions; fixtures are synthetic, not evidence."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/task3_common_scoring.py"


@pytest.fixture
def api():
    if not SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("task3_common_scoring", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def panel(ncal=9):
    cells, features, outcomes = [], [], []
    for role, count in [("fit", 6), ("calibrate", ncal), ("score", 4)]:
        for i in range(count):
            identity = {"cell_id": f"{role}:{i}", "environment_id": f"{role}:env:{i // 2}",
                        "sample_ids": [f"{role}:{i}:sample:{j}" for j in range(4)],
                        "model_id": "resnet50_bn", "frozen_checkpoint_sha256": "a" * 64,
                        "candidate_checkpoint_sha256": "b" * 64,
                        "trajectory_id": f"trajectory:{role}:{i}", "candidate_method": "POEM"}
            cells.append({"role": role, "identity": identity})
            z = [0.2, 0.5, 0.1, 0.1, 0.6, 0.1, 0., 0.1, 0.7, 0.1, float(i % 2)]
            features.append({"identity": deepcopy(identity), "Z": z})
            outcomes.append({"identity": deepcopy(identity), "frozen_correct": [1, 1, 0, 0],
                             "candidate_correct": [1, 1, 1, 0] if i % 2 else [1, 0, 0, 0]})
    manifest = {"schema": "task3-common-panel-v1", "feature_names": [
        "pre_entropy", "pre_conf", "pre_pbal", "post_entropy", "post_conf", "post_pbal",
        "pbal_drop", "entropy_drop", "frac_highconf", "marginal_KL", "update_norm"],
        "cells": cells}
    dev = features[:-4]
    return manifest, dev, outcomes[:6], outcomes[6:-4], features[-4:], outcomes[-4:]


def run(api, data):
    manifest, dev, fit, cal, features, outcomes = data
    gate = api.fit_gate(manifest, dev, fit, cal)
    packet = api.decide_gate(gate, features)
    return gate, packet, api.score_decisions(manifest, packet, outcomes)


def native_pair(identity, method="AETTA"):
    return {"identity": deepcopy(identity), "method": method,
            "implementation": "native_component", "source_sha256": "c" * 64,
            "invocation_sha256": "d" * 64, "frozen_estimated_accuracy": 0.4,
            "candidate_estimated_accuracy": 0.6}


def test_full_pipeline_scores_fixed_point_kga_on_identical_pairs(api):
    assert api is not None, "missing executable outcome-disjoint common scorer"
    _, packet, scores = run(api, panel())
    assert scores["policies"]["always_freeze"]["mean_accuracy"] == 0.5
    assert scores["policies"]["always_adapt"]["mean_accuracy"] == 0.5
    assert scores["policies"]["point_benefit"]["mean_accuracy"] == 0.625
    assert scores["policies"]["KGA"]["mean_accuracy"] == 0.625
    assert [r["actions"]["point_benefit"] for r in packet["rows"]] == ["FREEZE", "ADAPT", "FREEZE", "ADAPT"]
    assert all(r["actions"]["confidence_gain"] == "ADAPT" for r in packet["rows"])
    assert all(r["actions"]["entropy_progress"] == "ADAPT" for r in packet["rows"])


def test_exact_rank_infeasible_means_infinite_radius_and_abstain(api):
    _, packet, scores = run(api, panel(ncal=8))
    assert packet["radius"] == "infinity"
    assert all(r["actions"]["KGA"] == "ABSTAIN" for r in packet["rows"])
    assert scores["policies"]["KGA"]["commitment_rate"] == 0
    assert "coverage" not in scores["policies"]["KGA"]
    assert scores["policies"]["KGA"]["mean_accuracy"] == 0.5


def test_exact_rank_not_interpolation_or_clamped(api):
    assert api.exact_radius([i / 10 for i in range(9)]) == 0.8
    assert math.isinf(api.exact_radius([0.1] * 8))
    assert api.exact_radius([i / 20 for i in range(19)]) == 0.85


def test_score_outcome_and_environment_interventions_never_change_full_gate(api):
    data = panel()
    gate, packet, _ = run(api, data)
    for indexes in ([0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]):
        altered = deepcopy(data)
        for i in indexes:
            altered[-1][i]["frozen_correct"] = [0, 0, 0, 0]
            altered[-1][i]["candidate_correct"] = [1, 1, 1, 1]
        changed_gate, changed_packet, _ = run(api, altered)
        assert changed_packet == packet
        assert changed_gate.fit_binding == gate.fit_binding
        assert changed_gate.calibration_binding == gate.calibration_binding


def test_score_environment_features_do_not_refit_or_change_other_environment(api):
    data = panel()
    _, packet, _ = run(api, data)
    altered = deepcopy(data)
    for row in altered[-2][:2]:
        row["Z"] = [0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0., 0.1, 0.2, 0.1, 2.]
    _, changed, _ = run(api, altered)
    assert changed["radius"] == packet["radius"]
    assert changed["fit_binding"] == packet["fit_binding"]
    assert changed["calibration_binding"] == packet["calibration_binding"]
    assert changed["rows"][2:] == packet["rows"][2:]


def test_calibration_labels_change_radius_but_never_prediction_fit(api):
    data = panel()
    _, before, _ = run(api, data)
    for outcome in data[3]:
        outcome["candidate_correct"] = [1, 1, 1, 1]
    _, after, _ = run(api, data)
    assert before["fit_binding"] == after["fit_binding"]
    assert [r["prediction"] for r in before["rows"]] == [r["prediction"] for r in after["rows"]]
    assert after["radius"] > before["radius"]


def test_fit_labels_actually_drive_predictions(api):
    data = panel()
    _, before, _ = run(api, data)
    for outcome in data[2]:
        outcome["candidate_correct"] = [1, 1, 1, 1]
    _, after, _ = run(api, data)
    assert [r["prediction"] for r in before["rows"]] != [r["prediction"] for r in after["rows"]]


def test_native_estimates_are_paired_local_controllers_and_score_same_candidate(api):
    data = panel()
    gate = api.fit_gate(*data[:4])
    native = [native_pair(row["identity"], method) for method in ("AETTA", "Baek_ALine") for row in data[-2]]
    packet = api.decide_gate(gate, data[-2], native)
    scores = api.score_decisions(data[0], packet, data[-1])
    for method in ("AETTA_controller", "Baek_ALine_controller"):
        assert scores["policies"][method]["mean_accuracy"] == 0.5
        assert scores["policies"][method]["false_adapt_unconditional"] == 0.5
        assert "local estimated-benefit" in packet["policy_descriptions"][method]
    assert packet["native_provenance"]["independently_authenticated"] is False


@pytest.mark.parametrize("key,value", [("frozen_correct", [1]*4), ("a0", 0.5), ("label", 2)])
def test_decision_stage_rejects_outcomes_not_silently_ignores_them(api, key, value):
    data = panel()
    gate = api.fit_gate(*data[:4])
    data[-2][0][key] = value
    with pytest.raises(ValueError):
        api.decide_gate(gate, data[-2])


@pytest.mark.parametrize("change", ["environment", "sample", "duplicate_cell", "role", "hash", "empty_id"])
def test_manifest_rejects_ambiguous_identity_and_role_overlap(api, change):
    data = panel()
    cells = data[0]["cells"]
    if change == "environment":
        cells[-1]["identity"]["environment_id"] = cells[0]["identity"]["environment_id"]
    elif change == "sample":
        cells[-1]["identity"]["sample_ids"][0] = cells[0]["identity"]["sample_ids"][0]
    elif change == "duplicate_cell":
        cells.append(deepcopy(cells[-1]))
    elif change == "role":
        cells[-1]["role"] = "test"
    elif change == "hash":
        cells[0]["identity"]["frozen_checkpoint_sha256"] = "unverified"
    else:
        cells[0]["identity"]["trajectory_id"] = ""
    with pytest.raises(ValueError):
        api.fit_gate(*data[:4])


@pytest.mark.parametrize("stage", ["fit_features", "fit_outcomes", "calibration", "score_features", "score_outcomes"])
def test_each_stage_rejects_wrong_role_duplicate_missing_or_mismatched_rows(api, stage):
    for change in ("duplicate", "missing", "trajectory", "checkpoint", "sample_order"):
        data = panel()
        gate = api.fit_gate(*data[:4])
        rows = data[{"fit_features": 1, "fit_outcomes": 2, "calibration": 3, "score_features": 4, "score_outcomes": 5}[stage]]
        if change == "duplicate":
            rows.append(deepcopy(rows[0]))
        elif change == "missing":
            rows.pop()
        elif change == "trajectory":
            rows[0]["identity"]["trajectory_id"] = "wrong-model-timing"
        elif change == "checkpoint":
            rows[0]["identity"]["candidate_checkpoint_sha256"] = "e" * 64
        else:
            rows[0]["identity"]["sample_ids"].reverse()
        with pytest.raises(ValueError):
            if stage == "score_features":
                api.decide_gate(gate, data[-2])
            elif stage == "score_outcomes":
                api.score_decisions(data[0], api.decide_gate(gate, data[-2]), data[-1])
            else:
                api.fit_gate(*data[:4])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.1", True, None])
def test_features_and_outcomes_reject_non_numeric_non_finite_values(api, value):
    data = panel()
    data[1][0]["Z"][0] = value
    with pytest.raises(ValueError):
        api.fit_gate(*data[:4])
    data = panel()
    data[2][0]["frozen_correct"][0] = value
    with pytest.raises(ValueError):
        api.fit_gate(*data[:4])


@pytest.mark.parametrize("field,value", [("implementation", "port"), ("method", "Kim_TTALine"),
    ("candidate_estimated_accuracy", 1.1), ("frozen_estimated_accuracy", -0.1),
    ("candidate_estimated_accuracy", True), ("invocation_sha256", "unknown")])
def test_native_rejects_ports_misattribution_and_malformed_estimates(api, field, value):
    data = panel()
    gate = api.fit_gate(*data[:4])
    native = [native_pair(row["identity"]) for row in data[-2]]
    native[0][field] = value
    with pytest.raises(ValueError):
        api.decide_gate(gate, data[-2], native)


def test_poem_independent_trajectory_cannot_borrow_other_candidate_outcomes(api):
    data = panel()
    _, packet, _ = run(api, data)
    data[-1][0]["identity"]["candidate_method"] = "SAR"
    with pytest.raises(ValueError):
        api.score_decisions(data[0], packet, data[-1])


def test_packet_mutation_is_detected_and_manifest_is_snapshotted(api):
    data = panel()
    gate, packet, _ = run(api, data)
    data[0]["cells"][0]["identity"]["model_id"] = "new-model"
    assert api.decide_gate(gate, data[-2]) == packet
    packet["rows"][0]["actions"]["KGA"] = "ADAPT"
    with pytest.raises(ValueError):
        api.score_decisions(panel()[0], packet, data[-1])


def test_score_rejects_rehashed_unknown_policies_and_wrong_interval_rule(api):
    # A checksum is not authenticity, but even rehashed packets must obey the schema.
    import hashlib
    import json
    for change in ("unknown_policy", "alpha", "negative_radius", "wrong_action"):
        data = panel()
        _, packet, _ = run(api, data)
        if change == "unknown_policy":
            packet["policy_descriptions"]["claimed_native_Kim"] = "invented"
            for row in packet["rows"]:
                row["actions"]["claimed_native_Kim"] = "ADAPT"
        elif change == "alpha":
            packet["alpha"] = 0.2
        elif change == "negative_radius":
            packet["radius"] = -0.1
        else:
            packet["rows"][0]["actions"]["KGA"] = "ADAPT"
        packet.pop("packet_sha256")
        packet["packet_sha256"] = hashlib.sha256(json.dumps(
            packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with pytest.raises(ValueError):
            api.score_decisions(data[0], packet, data[-1])


def test_scored_outcome_rows_rejected_in_fit_and_calibration(api):
    for position in (2, 3):
        data = list(panel())
        data[position] = data[-1]
        with pytest.raises(ValueError):
            api.fit_gate(*data[:4])


def test_native_missing_duplicate_mismatched_pair_is_rejected(api):
    for change in ("missing", "duplicate", "trajectory"):
        data = panel()
        gate = api.fit_gate(*data[:4])
        native = [native_pair(row["identity"]) for row in data[-2]]
        if change == "missing":
            native.pop()
        elif change == "duplicate":
            native.append(deepcopy(native[0]))
        else:
            native[0]["identity"]["trajectory_id"] = "another-POEM-trajectory"
        with pytest.raises(ValueError):
            api.decide_gate(gate, data[-2], native)


def test_zero_benefit_and_native_ties_have_explicit_boundary_semantics(api):
    data = panel()
    for rows in (data[2], data[3], data[-1]):
        for row in rows:
            row["candidate_correct"] = row["frozen_correct"][:]
    gate = api.fit_gate(*data[:4])
    native = [native_pair(row["identity"]) for row in data[-2]]
    for row in native:
        row["candidate_estimated_accuracy"] = row["frozen_estimated_accuracy"]
    packet = api.decide_gate(gate, data[-2], native)
    scores = api.score_decisions(data[0], packet, data[-1])
    assert all(r["actions"]["point_benefit"] == "FREEZE" for r in packet["rows"])
    assert all(r["actions"]["KGA"] == "ABSTAIN" for r in packet["rows"])
    assert all(r["actions"]["AETTA_controller"] == "FREEZE" for r in packet["rows"])
    assert scores["policies"]["always_adapt"]["false_adapt_unconditional"] == 1.0
    assert scores["policies"]["always_adapt"]["false_adapt_conditional"] == 1.0


@pytest.mark.parametrize("value", [-0.1, 0.5, 1.1])
def test_correctness_requires_binary_values(api, value):
    data = panel()
    data[2][0]["candidate_correct"][0] = value
    with pytest.raises(ValueError):
        api.fit_gate(*data[:4])
