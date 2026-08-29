from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "docs/research/kbound/scripts/generate_exact_confirmation_units.py"
ANALYZER = REPO / "docs/research/kbound/scripts/analyze_exact_confirmation.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(tmp_path: Path, *, n_calibration: int):
    generator = load_module("exact_generator", GENERATOR)
    manifest = generator.generate_manifest(
        seed=7,
        n_fit=12,
        n_calibration=n_calibration,
        n_test=5,
        alpha=0.10,
    )
    manifest["status"] = "SEALED"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = generator.canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fit_cal = []
    test = []
    labels = []
    role_index = {"estimator_fit": 0, "residual_calibration": 0, "test": 0}
    for unit in manifest["units"]:
        role = unit["role"]
        i = role_index[role]
        role_index[role] += 1
        z = [float(i), float((i * i) % 7), float(unit["severity"])]
        delta = 0.03 * z[0] - 0.01 * z[1] + 0.005 * z[2] - 0.08
        if role == "test":
            test.append({"unit_id": unit["unit_id"], "role": role, "Z": z})
            labels.append(
                {
                    "unit_id": unit["unit_id"],
                    "role": role,
                    "delta": delta,
                    "risk_freeze": 0.4,
                    "risk_adapt": 0.4 - delta,
                }
            )
        else:
            fit_cal.append({"unit_id": unit["unit_id"], "role": role, "Z": z, "delta": delta})
    fit_path = tmp_path / "fit_cal.json"
    test_path = tmp_path / "test_evidence.json"
    labels_path = tmp_path / "test_labels.json"
    fit_path.write_text(json.dumps({"records": fit_cal}), encoding="utf-8")
    test_path.write_text(json.dumps({"records": test}), encoding="utf-8")
    labels_path.write_text(json.dumps({"records": labels}), encoding="utf-8")
    return manifest, manifest_path, fit_path, test_path, labels_path


def test_generator_is_deterministic_role_disjoint_and_hash_valid() -> None:
    generator = load_module("exact_generator_determinism", GENERATOR)
    first = generator.generate_manifest(seed=11, n_fit=3, n_calibration=4, n_test=5, alpha=0.1)
    second = generator.generate_manifest(seed=11, n_fit=3, n_calibration=4, n_test=5, alpha=0.1)
    assert first == second
    assert generator.validate_manifest(first) == []
    by_role = {
        role: {unit["unit_id"] for unit in first["units"] if unit["role"] == role}
        for role in ("estimator_fit", "residual_calibration", "test")
    }
    assert by_role["estimator_fit"].isdisjoint(by_role["residual_calibration"])
    assert by_role["estimator_fit"].isdisjoint(by_role["test"])
    assert by_role["residual_calibration"].isdisjoint(by_role["test"])


def test_feasible_exact_rank_decide_then_offline_join(tmp_path: Path) -> None:
    analyzer = load_module("exact_analyzer_feasible", ANALYZER)
    _, manifest_path, fit_path, test_path, labels_path = write_fixture(tmp_path, n_calibration=9)
    decisions = analyzer.decide(manifest_path, fit_path, test_path)
    assert decisions["exact_rank"] == 9
    assert math.isfinite(decisions["epsilon"])
    assert all(not (analyzer.LABEL_FIELDS & set(row)) for row in decisions["decisions"])
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    result = analyzer.evaluate(manifest_path, decisions_path, labels_path)
    assert result["summary"]["n"] == 5
    assert 0.0 <= result["summary"]["interval_coverage"] <= 1.0
    assert result["summary"]["beats_both_inference"] == "pending_predeclared_paired_analysis"


def test_infeasible_exact_rank_forces_abstention(tmp_path: Path) -> None:
    analyzer = load_module("exact_analyzer_infeasible", ANALYZER)
    _, manifest_path, fit_path, test_path, _ = write_fixture(tmp_path, n_calibration=8)
    with pytest.warns(UserWarning, match="needs n >="):
        decisions = analyzer.decide(manifest_path, fit_path, test_path)
    assert math.isinf(decisions["epsilon"])
    assert {row["action"] for row in decisions["decisions"]} == {"ABSTAIN"}


def test_test_evidence_rejects_label_bearing_fields(tmp_path: Path) -> None:
    analyzer = load_module("exact_analyzer_leakage", ANALYZER)
    _, manifest_path, fit_path, test_path, _ = write_fixture(tmp_path, n_calibration=9)
    payload = json.loads(test_path.read_text(encoding="utf-8"))
    payload["records"][0]["delta"] = 0.1
    test_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label-bearing"):
        analyzer.decide(manifest_path, fit_path, test_path)
