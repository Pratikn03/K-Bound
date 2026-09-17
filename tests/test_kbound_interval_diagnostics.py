"""Synthetic QA for retrospective interval reporting; no model or target data."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from docs.research.kbound.scripts import build_current_policy_interval_diagnostics as diagnostics


def raw_records(n: int = 20, *, seed: int = 0, candidate: str = "tent") -> list[dict]:
    records = []
    for index in range(n):
        benefit = 0.125 if index % 2 == 0 else -0.125
        family = "contrast" if index % 2 == 0 else "fog"
        records.append(
            {
                "seed": seed,
                "method": candidate,
                "benchmark": "cifar10c",
                "condition": f"{family}|synthetic{index}",
                "B": benefit,
                "b_hat": benefit,
                "a0": 0.5,
                "a_adapted": 0.5 + benefit,
                # Deliberately unusable historical fields: they must not enter
                # the current replay or its validation requirements.
                "eps_conformal": "obsolete radius, do not read",
                "kga_decision": "obsolete action, do not read",
            }
        )
    return records


def summary_record(benefit: float, prediction: float, radius: float, decision: str) -> dict:
    return {
        "B": benefit,
        "b_hat": prediction,
        "epsilon": radius,
        "decision": decision,
        "a0": 0.5,
        "a_adapted": 0.5 + benefit,
    }


def test_replay_uses_released_loo_policy_and_ignores_historical_fields(monkeypatch) -> None:
    original = diagnostics._released_policy()
    calls = []

    def spy(prediction, benefit, **kwargs):
        calls.append(kwargs)
        return original(prediction, benefit, **kwargs)

    monkeypatch.setattr(diagnostics, "_released_policy", lambda: spy)
    raw = raw_records()
    unchanged = copy.deepcopy(raw)
    replay = diagnostics.replay_records(raw, "tent", 0)
    assert raw == unchanged
    assert calls == [{"alpha": 0.1, "calibration": "loo"}]
    assert len(replay) == len(raw)
    assert [row["epsilon"] for row in replay] == [0.0] * 20
    assert [row["decision"] for row in replay] == ["ADAPT", "FREEZE"] * 10
    assert all("eps_conformal" not in row and "kga_decision" not in row for row in replay)


def test_family_summaries_keep_the_original_per_seed_calibration_pool() -> None:
    raw = raw_records(40)
    for index, row in enumerate(raw):
        row["b_hat"] += index / 512
    replay = diagnostics.replay_records(raw, "tent", 0)
    before = copy.deepcopy(replay)
    original_epsilon, _ = diagnostics._released_policy()(
        [row["b_hat"] for row in raw], [row["B"] for row in raw], alpha=0.1, calibration="loo"
    )
    for family in ("contrast", "fog"):
        indexes = [index for index, row in enumerate(replay) if row["family"] == family]
        group = [replay[index] for index in indexes]
        result = diagnostics.summarize(group)
        assert result["n"] == 20
        assert result["full_interval_width"]["mean"]["value"] == float(np.mean(2 * original_epsilon[indexes]))
    assert replay == before


def test_inclusion_uses_closed_endpoints_and_full_unclipped_width() -> None:
    rows = [
        summary_record(0.125, 0.0, 0.125, "ABSTAIN"),
        summary_record(-0.125, 0.0, 0.125, "ABSTAIN"),
        summary_record(0.0, 0.125, 0.125, "ABSTAIN"),
        summary_record(0.25, 0.0, 0.125, "ABSTAIN"),
    ]
    result = diagnostics.summarize(rows)
    assert result["nominal_inclusion_target"] == 0.9
    assert result["observed_inclusion"]["numerator"] == 3
    assert result["observed_inclusion"]["denominator"] == 4
    assert result["observed_inclusion"]["value"] == 0.75
    assert result["full_interval_width"]["mean"] == {"value": 0.25, "status": "finite"}
    assert result["full_interval_width"]["median"]["value"] == 0.25
    assert result["finite_interval_count"] == 4
    assert result["infinite_interval_count"] == 0
    assert result["commitment"]["value"] == 0
    for name, action in (("false_adapt", "ADAPT"), ("false_freeze", "FREEZE")):
        assert result[name]["marginal"]["value"] == 0
        conditional = result[name]["conditional"]
        assert conditional == {
            "numerator": 0,
            "denominator": 0,
            "value": None,
            "defined": False,
            "undefined_reason": f"no {action} decisions",
        }


def test_false_adapt_and_freeze_use_their_own_exposure_denominators() -> None:
    rows = [
        summary_record(-0.125, 0.25, 0.0625, "ADAPT"),
        summary_record(0.0, 0.25, 0.0625, "ADAPT"),
        summary_record(0.125, 0.25, 0.0625, "ADAPT"),
        summary_record(0.125, -0.25, 0.0625, "FREEZE"),
        summary_record(0.0, -0.25, 0.0625, "FREEZE"),
        summary_record(-0.125, -0.25, 0.0625, "FREEZE"),
        summary_record(0.0, 0.0, 0.0625, "ABSTAIN"),
    ]
    result = diagnostics.summarize(rows)
    assert result["commitment"]["value"] == 6 / 7
    for name in ("false_adapt", "false_freeze"):
        assert result[name]["marginal"]["numerator"] == 2
        assert result[name]["marginal"]["value"] == 2 / 7
        assert result[name]["conditional"]["denominator"] == 3
        assert result[name]["conditional"]["value"] == 2 / 3


def test_infinite_intervals_are_counted_not_dropped_and_json_is_strict() -> None:
    with pytest.warns(UserWarning):
        replay = diagnostics.replay_records(raw_records(3), "tent", 0)
    result = diagnostics.summarize(replay)
    assert result["n"] == 3
    assert result["observed_inclusion"]["value"] == 1
    assert result["finite_interval_count"] == 0
    assert result["infinite_interval_count"] == 3
    assert result["finite_interval_inclusion"]["defined"] is False
    assert result["actions"]["ABSTAIN"]["numerator"] == 3
    assert result["commitment"]["value"] == 0
    for summary in result["full_interval_width"].values():
        if isinstance(summary, dict):
            assert summary == {"value": None, "status": "positive_infinity"}
    encoded = json.dumps(result, allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded


@pytest.mark.parametrize("field", ["B", "b_hat", "a0", "a_adapted"])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), float("-inf"), True, "0.125"])
def test_unavailable_or_malformed_values_fail_without_dropping_a_cell(field, invalid) -> None:
    raw = raw_records()
    raw[0][field] = invalid
    with pytest.raises(ValueError, match="finite number"):
        diagnostics.replay_records(raw, "tent", 0)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("seed", 1, "seed mismatch"),
        ("seed", True, "seed mismatch"),
        ("method", "sar", "candidate/benchmark"),
        ("benchmark", "new_target", "candidate/benchmark"),
        ("condition", "unknown|1", "corruption family"),
        ("condition", "contrast", "condition identity"),
        ("a0", 1.5, "accuracy values"),
        ("B", 0.5, "B disagrees"),
    ],
)
def test_unit_and_score_contract_failures(field, invalid, message) -> None:
    raw = raw_records()
    raw[0][field] = invalid
    with pytest.raises(ValueError, match=message):
        diagnostics.replay_records(raw, "tent", 0)


def test_duplicate_cells_and_empty_pools_fail() -> None:
    raw = raw_records()
    raw[1]["condition"] = raw[0]["condition"]
    with pytest.raises(ValueError, match="duplicate condition"):
        diagnostics.replay_records(raw, "tent", 0)
    with pytest.raises(ValueError, match="nonempty"):
        diagnostics.replay_records([], "tent", 0)
    with pytest.raises(ValueError, match="empty group"):
        diagnostics.summarize([])
    with pytest.raises(ValueError, match="empty pool"):
        diagnostics.replay_score([])


@pytest.mark.parametrize("radius", [float("nan"), -0.1, float("-inf")])
def test_summary_rejects_invalid_radii(radius) -> None:
    with pytest.raises(ValueError, match="invalid radius"):
        diagnostics.summarize([summary_record(0, 0, radius, "ABSTAIN")])


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '[]'])
def test_json_reader_rejects_ambiguous_or_nonfinite_input(tmp_path, text) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        diagnostics.read_json(path, tmp_path)


def test_source_seal_is_for_the_exact_resident_bytes(tmp_path) -> None:
    path = tmp_path / "source.json"
    data = b'{"records": [1, 2]}\n'
    path.write_bytes(data)
    payload, seal = diagnostics.read_json(path, tmp_path)
    assert payload == {"records": [1, 2]}
    assert seal == {"path": "source.json", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_dataless_input_is_rejected_before_read(monkeypatch, tmp_path) -> None:
    def dataless_stat(self, **kwargs):
        return SimpleNamespace(st_flags=diagnostics.DATALESS_FLAG, st_mode=stat.S_IFREG)

    def forbidden_read(self):
        raise AssertionError("must not read or hydrate a dataless input")

    monkeypatch.setattr(Path, "stat", dataless_stat)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="nonresident/dataless"):
        diagnostics.read_json(tmp_path / "cloud.json", tmp_path)


def test_symlink_and_missing_inputs_are_not_followed_or_restored(tmp_path) -> None:
    target = tmp_path / "source.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="not a symlink"):
        diagnostics.read_json(link, tmp_path)
    with pytest.raises(FileNotFoundError):
        diagnostics.read_json(tmp_path / "missing.json", tmp_path)
    with pytest.raises(FileNotFoundError):
        diagnostics.build_artifact(root=tmp_path)


def scientific_fixture() -> tuple[list[dict], dict, dict]:
    per_seed = [diagnostics.replay_records(raw_records(seed=seed), "tent", seed) for seed in (0, 1)]
    rows = [row for group in per_seed for row in group]
    canonical = diagnostics.replay_score(rows)
    canonical["per_file"] = [
        {"seed": seed, "score": diagnostics.replay_score(group), "epsilon_min": 0.0, "epsilon_mean": 0.0, "epsilon_max": 0.0}
        for seed, group in enumerate(per_seed)
    ]
    inference = {
        "decision_counts": {"ADAPT": 20, "FREEZE": 20, "ABSTAIN": 0},
        "adapt_exposure": 0.5,
        "freeze_exposure": 0.5,
        "strict_decision_coverage": 1.0,
        "grain": {"n_records": 40, "n_run_seeds": 2, "n_conditions_per_seed": 20, "n_inference_units": 2, "families": ["contrast", "fog"]},
        "comparisons": {
            "always_adapt": {"point": 0.0625, "family_effects": {"contrast": 0.0, "fog": 0.125}},
            "always_freeze": {"point": 0.0625, "family_effects": {"contrast": 0.125, "fog": 0.0}},
        },
    }
    return rows, canonical, inference


def test_scientific_equality_checks_candidate_seed_radii_and_family_effects() -> None:
    rows, canonical, inference = scientific_fixture()
    checks = diagnostics.verify_scientific_equality(rows, canonical, inference)
    assert checks and all(check["passed"] for check in checks)
    names = {check["check"] for check in checks}
    assert {"canonical.regret", "canonical.seed0.epsilon_mean", "current_policy.always_adapt.family_effects"} <= names
    assert all(check["comparison"] == "exact_equality" for check in checks)


@pytest.mark.parametrize("change", ["count", "regret", "radius", "family", "point", "inference_action", "duplicate_seed"])
def test_any_scientific_authority_drift_blocks_reporting(change) -> None:
    rows, canonical, inference = scientific_fixture()
    if change == "count":
        canonical["adapt_count"] += 1
    elif change == "regret":
        canonical["regret"]["kga"] = 1e-20
    elif change == "radius":
        canonical["per_file"][0]["epsilon_mean"] = 1e-20
    elif change == "family":
        inference["comparisons"]["always_adapt"]["family_effects"]["fog"] = 0.25
    elif change == "point":
        inference["comparisons"]["always_adapt"]["point"] = math.nextafter(0.0625, 1)
    elif change == "inference_action":
        inference["decision_counts"]["ADAPT"] += 1
    else:
        canonical["per_file"].append(copy.deepcopy(canonical["per_file"][0]))
    with pytest.raises(ValueError, match="mismatch|duplicate"):
        diagnostics.verify_scientific_equality(rows, canonical, inference)


def test_numeric_equality_does_not_hide_drift_behind_rounding_or_tolerance() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        diagnostics.exact_match([], "small_drift", 0.1, math.nextafter(0.1, 1))


def test_latex_includes_every_candidate_and_group_with_undefined_rates() -> None:
    summary = diagnostics.summarize([summary_record(0, 0, 1, "ABSTAIN")])
    artifact = {"candidates": {candidate: {"summary": summary, "by_corruption_family": {family: summary for family in diagnostics.FAMILIES}} for candidate in diagnostics.CANDIDATES}}
    latex = diagnostics.render_latex(artifact)
    groups = diagnostics.render_groups_latex(artifact)
    assert groups.count(" & gaussian\\_noise & ") == 3
    assert "0.0000/-- & 0.0000/--" in latex
    assert "0/0; -- & 0/0; --" in groups
    assert "RETROSPECTIVE" in latex and "rank-constrains" in latex
    assert "RETROSPECTIVE" in groups and "rank-constrained" in groups
    for candidate in diagnostics.CANDIDATES:
        assert latex.count(f"{candidate.upper()} & ") == 1
        assert groups.count(f"{candidate.upper()} & ") == 6


def test_failed_validation_creates_no_outputs(monkeypatch, tmp_path) -> None:
    output_json = tmp_path / "diagnostics.json"
    output_tex = tmp_path / "diagnostics.tex"

    def fail():
        raise ValueError("canonical count mismatch")

    monkeypatch.setattr(diagnostics, "build_artifact", fail)
    monkeypatch.setattr("sys.argv", ["diagnostics", "--output-json", str(output_json), "--output-tex", str(output_tex)])
    assert diagnostics.main() == 1
    assert not output_json.exists() and not output_tex.exists()


def owned_json(value: int) -> str:
    return json.dumps({"schema": diagnostics.SCHEMA, "analysis_script": "docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py", "synthetic": value}) + "\n"


def test_explicit_refresh_updates_only_recognized_generated_outputs(tmp_path) -> None:
    path = tmp_path / "diagnostics.json"
    path.write_text(owned_json(1), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit --refresh-existing"):
        diagnostics.write_outputs({path: owned_json(2)})
    assert path.read_text() == owned_json(1)
    diagnostics.write_outputs({path: owned_json(2)}, refresh_existing=True)
    assert path.read_text() == owned_json(2)
    diagnostics.write_outputs({path: owned_json(2)}, check=True)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("suffix,existing", [(".json", "{}"), (".tex", "% user-authored table\n")])
def test_refresh_refuses_unrecognized_files_and_preflights_all_outputs(tmp_path, suffix, existing) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / ("unrecognized" + suffix)
    first.write_text(owned_json(1), encoding="utf-8")
    second.write_text(existing, encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized"):
        diagnostics.write_outputs({first: owned_json(2), second: "replacement"}, refresh_existing=True)
    assert first.read_text() == owned_json(1)
    assert second.read_text() == existing


def test_refresh_symlink_and_protected_path_guards(tmp_path) -> None:
    source = tmp_path / "authority.json"
    source.write_text(owned_json(1), encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        diagnostics.write_outputs({link: owned_json(2)}, refresh_existing=True)
    with pytest.raises(ValueError, match="input authority or code"):
        diagnostics.write_outputs({source: owned_json(2)}, refresh_existing=True, protected_paths=(source,))
    directory = tmp_path / "linked-directory"
    directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        diagnostics.write_outputs({directory / "new.json": owned_json(2)}, refresh_existing=True)
    assert source.read_text() == owned_json(1)
    assert not (tmp_path / "new.json").exists()


def test_check_and_refresh_are_mutually_exclusive(monkeypatch, tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        diagnostics.write_outputs({}, check=True, refresh_existing=True)
    monkeypatch.setattr("sys.argv", ["diagnostics", "--check", "--refresh-existing"])
    with pytest.raises(SystemExit) as stopped:
        diagnostics.main()
    assert stopped.value.code == 2


def test_failed_scientific_validation_never_refreshes_existing_output(monkeypatch, tmp_path) -> None:
    path = tmp_path / "diagnostics.json"
    path.write_text(owned_json(1), encoding="utf-8")

    def fail():
        raise ValueError("canonical regret mismatch")

    monkeypatch.setattr(diagnostics, "build_artifact", fail)
    monkeypatch.setattr("sys.argv", ["diagnostics", "--refresh-existing", "--output-json", str(path)])
    assert diagnostics.main() == 1
    assert path.read_text() == owned_json(1)
