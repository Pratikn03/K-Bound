"""Outcome isolation and statistical edge cases, before implementation."""

import math

import pytest

from kga.calibration_value import (
    calibrate_fold,
    conditional_risk_upper,
    decide,
    exact_radius,
    grouped_roles,
    matched_exposure_ids,
    outcome_free,
    score_decisions,
    sha256_file,
    validate_roles,
    verify_sha256,
    write_fresh_json,
)


def cells(n=60):
    return [
        {
            "id": f"cell{i}",
            "group": f"g{i}",
            "environment": f"e{i // 10}",
            "checkpoint": "m0",
            "features": [i / n, (i % 3) / 3],
            "frozen_score": 0.5,
            "candidate_score": 0.5 + 0.3 * (i / n - 0.5),
        }
        for i in range(n)
    ]


def test_exact_rank_never_clips_unavailable_rank():
    assert math.isinf(exact_radius([0.1] * 8, 0.1))
    assert exact_radius([0.1] * 9, 0.1) == 0.1
    assert math.isinf(exact_radius([], 0.1))
    with pytest.raises(ValueError):
        exact_radius([float("nan")], 0.1)


def test_group_roles_keep_repeated_cells_together():
    data = cells()
    repeated = [dict(r, id=r["id"] + "rep") for r in data]
    roles = grouped_roles(data + repeated, salt="unit")
    validate_roles(**roles)
    sets = [{r["group"] for r in rows} for rows in roles.values()]
    assert sum(map(len, sets)) == 60
    assert all(len(v) for v in roles.values())


def test_duplicate_and_group_overlap_are_rejected():
    x = cells(5)
    with pytest.raises(ValueError, match="group"):
        validate_roles(fit=x[:2], cal=x[1:3])
    with pytest.raises(ValueError, match="duplicate"):
        validate_roles(fit=x + x)


def test_scored_outcomes_cannot_change_fit_or_decisions():
    data = cells(120)
    fit, tune, cal, score = data[:50], data[50:70], data[70:100], data[100:]
    trained = calibrate_fold(fit, tune, cal)
    before = decide(trained, outcome_free(score))
    altered = [dict(r, frozen_score=1 - r["frozen_score"], candidate_score=1 - r["candidate_score"]) for r in score]
    assert decide(trained, outcome_free(altered)) == before
    with pytest.raises(ValueError, match="outcome-free"):
        decide(trained, score)
    assert len({r["predictor_sha256"] for r in before}) == 1


def test_constant_radius_preserves_matched_exposure_ranking():
    scores = [0.2, 0.2, -0.1, 0.8, 0.03]
    ids = ["b", "a", "c", "d", "e"]
    assert matched_exposure_ids(scores, ids, 3) == matched_exposure_ids([p - 0.4 for p in scores], ids, 3)


def test_zero_exposure_and_tie_metrics_are_defined():
    outcomes = [dict(r, candidate_score=r["frozen_score"]) for r in cells(3)]
    decisions = [{"id": r["id"], "action": "ABSTAIN", "lower": None, "upper": None} for r in outcomes]
    score = score_decisions(decisions, outcomes)
    assert score["false_adapt_conditional"] is None
    assert score["false_adapt_unconditional"] == 0
    assert score["oracle_regret"] == 0
    assert score["interval_inclusion"] is None


def test_risk_upper_is_simultaneous_conservative_and_zero_exposure_unavailable():
    assert conditional_risk_upper(0, 0, 0.05) == 1
    assert conditional_risk_upper(0, 100, 0.05 / 13) > conditional_risk_upper(0, 100, 0.05)
    assert conditional_risk_upper(0, 100, 0.05) == pytest.approx(1 - 0.05**0.01)


def test_fresh_outputs_and_tampering(tmp_path):
    p = tmp_path / "x.json"
    write_fresh_json(p, {"radius": None, "status": "infinite"})
    verify_sha256(p, sha256_file(p))
    with pytest.raises(FileExistsError):
        write_fresh_json(p, {})
    with pytest.raises(ValueError, match="hash"):
        verify_sha256(p, "0" * 64)
    with pytest.raises(ValueError):
        write_fresh_json(tmp_path / "bad.json", {"bad": float("inf")})


def test_outer_environment_and_joint_checkpoint_isolation():
    from kga.calibration_value import build_folds

    data = []
    for m in range(3):
        for env in range(3):
            for group in range(12):
                data.append(
                    {
                        "id": f"m{m}e{env}g{group}",
                        "group": f"e{env}g{group}",
                        "environment": f"e{env}",
                        "checkpoint": f"m{m}",
                        "features": [group / 12],
                        "frozen_score": 0.5,
                        "candidate_score": 0.5,
                    }
                )
    for kind in ["environment", "checkpoint", "joint"]:
        folds = build_folds(data, kind)
        scored = []
        for f in folds:
            validate_roles(fit=f["fit"], tune=f["tune"], cal=f["cal"], score=f["score_features"])
            scored += [r["id"] for r in f["score_features"]]
            if kind in ["checkpoint", "joint"]:
                trained = {r["checkpoint"] for role in ["fit", "tune", "cal"] for r in f[role]}
                assert not trained & {r["checkpoint"] for r in f["score_features"]}
            if kind in ["environment", "joint"]:
                trained = {r["environment"] for role in ["fit", "tune", "cal"] for r in f[role]}
                assert not trained & {r["environment"] for r in f["score_features"]}
        assert sorted(scored) == sorted(r["id"] for r in data)


def test_full_outer_fold_refitting_ignores_its_score_outcomes():
    from kga.calibration_value import build_folds

    data = cells(90)
    first = build_folds(data, "environment")[0]
    held = {r["id"] for r in first["score_features"]}
    altered = [dict(r, candidate_score=1 - r["candidate_score"]) if r["id"] in held else r for r in data]
    second = build_folds(altered, "environment")[0]
    before = calibrate_fold(first["fit"], first["tune"], first["cal"])
    after = calibrate_fold(second["fit"], second["tune"], second["cal"])
    assert before["receipt"]["predictor_sha256"] == after["receipt"]["predictor_sha256"]
    assert before["receipt"]["radii"] == after["receipt"]["radii"]
    assert before["receipt"]["fixed_margins"] == after["receipt"]["fixed_margins"]
    assert before["receipt"]["risk_selected"] == after["receipt"]["risk_selected"]
    assert decide(before, first["score_features"]) == decide(after, second["score_features"])


def test_environment_rank_uses_only_fit_tune_disjoint_environments():
    data = cells(100)
    # Deliberately same environments represented in fit/tune/cal; no honest
    # held-out environment calibration units exist despite 20 residual rows.
    for r in data:
        r["environment"] = "same_environment"
    trained = calibrate_fold(data[:50], data[50:70], data[70:90])
    assert trained["receipt"]["independent_calibration_environment_count"] == 0
    decisions = decide(trained, outcome_free(data[90:]))
    env = [d for d in decisions if d["arm"].startswith("environment_max")]
    assert env and all(d["action"] == "ABSTAIN" for d in env)


def test_full_calibration_receipt_is_strict_json_serializable():
    import json

    data = [dict(r, candidate_score=0.8) for r in cells(100)]
    data[75]["candidate_score"] = 0.2
    trained = calibrate_fold(data[:50], data[50:70], data[70:90])
    json.dumps(trained["receipt"], allow_nan=False)
    json.dumps(decide(trained, outcome_free(data[90:])), allow_nan=False)


def test_directional_false_freeze_excludes_abstention_and_distinguishes_zero_harm():
    rows = cells(6)
    for i, b in enumerate([0.1, 0.1, 0.1, 0, 0, -0.1]):
        rows[i]["candidate_score"] = rows[i]["frozen_score"] + b
    decisions = [
        {"id": r["id"], "action": a, "lower": None, "upper": None}
        for r, a in zip(rows, ["FREEZE", "ABSTAIN", "ABSTAIN", "FREEZE", "ADAPT", "ADAPT"])
    ]
    metrics = score_decisions(decisions, rows)
    assert metrics["false_freeze_count"] == 2  # positive and zero committed FREEZE
    assert metrics["forgone_helpful_nonadapt_count"] == 3  # positive FREEZE and ABSTAIN
    assert metrics["false_adapt_count"] == 2  # nonpositive ADAPT
    assert metrics["harmful_adapt_count"] == 1  # strictly negative ADAPT
    assert metrics["zero_benefit_adapt_count"] == 1


def test_semantic_group_strips_only_an_actual_terminal_repeat_field():
    from kga.calibration_value import semantic_group

    assert semantic_group("fog|s1|small|iid|mild|r0") == "fog|s1|small|iid|mild"
    assert (
        semantic_group("gaussian_noise|s3|small|aggressive|single_class")
        == "gaussian_noise|s3|small|aggressive|single_class"
    )


def test_heldout_semantic_group_risk_uses_fold_and_group_not_cell_counts():
    from kga.calibration_value import score_semantic_groups

    specifications = [
        ("f1", "g1", "a", 0.2, "ADAPT"),
        ("f1", "g1", "b", -0.1, "ADAPT"),
        ("f1", "g2", "c", 0.4, "FREEZE"),
        ("f1", "g2", "d", 0.0, "ADAPT"),
        ("f2", "g1", "a", 0.3, "ADAPT"),
        ("f1", "g3", "e", 0.1, "ABSTAIN"),
        ("f1", "g3", "f", 0.1, "ABSTAIN"),
        ("f1", "g3", "g", 0.1, "ABSTAIN"),
    ]
    decisions = [{"fold_id": f, "group": g, "id": i, "action": a} for f, g, i, b, a in specifications]
    outcomes = [
        {"fold_id": f, "group": g, "id": i, "frozen_score": 0.5, "candidate_score": 0.5 + b}
        for f, g, i, b, a in specifications
    ]
    metrics = score_semantic_groups(decisions, outcomes)
    assert metrics["n_all_groups"] == 4
    assert metrics["n_adapted_groups"] == 3
    assert metrics["n_error_groups"] == 2
    assert metrics["false_adapt_conditional"] == pytest.approx(2 / 3)
    assert metrics["false_adapt_unconditional"] == 0.5
    assert metrics["n_strictly_harmful_adapted_groups"] == 1
    assert metrics["n_zero_only_error_groups"] == 1
    assert metrics["equal_group_harm_weighted_losses"] == pytest.approx({"1": 0.0875, "5": 0.1375, "20": 0.325})
    # Unequal group sizes must not turn this into equal-cell aggregation.
    assert metrics["equal_group_harm_weighted_losses"]["1"] != pytest.approx(0.1)


def test_heldout_group_risk_zero_exposure_null_and_identity_checks():
    import json

    from kga.calibration_value import score_semantic_groups

    ds = [{"fold_id": "f", "group": "g", "id": "x", "action": "ABSTAIN"}]
    ys = [{"fold_id": "f", "group": "g", "id": "x", "frozen_score": 0.5, "candidate_score": 0.7}]
    metrics = score_semantic_groups(ds, ys)
    assert metrics["false_adapt_conditional"] is None
    assert metrics["false_adapt_unconditional"] == 0
    assert metrics["n_all_groups"] == 1
    json.dumps(metrics, allow_nan=False)
    with pytest.raises(ValueError, match="duplicate"):
        score_semantic_groups(ds + ds, ys)
    with pytest.raises(ValueError, match="group"):
        score_semantic_groups(ds, [dict(ys[0], group="different")])
    with pytest.raises(ValueError, match="ID"):
        score_semantic_groups(ds, [dict(ys[0], id="different")])
