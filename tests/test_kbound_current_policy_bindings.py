"""Canonical replay provenance must cover every declared live numeric dependency.

Only synthetic source/data fixtures are used. The in-memory artifact-builder
test stubs candidate analysis; no canonical result data or generator CLI is run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest

from docs.research.kbound.scripts import analyze_current_policy_cluster_inference as producer
from docs.research.kbound.scripts import build_result_manifest as manifest


def _load_sync_module():
    """Avoid the separate ``src/scripts`` package on pytest's import path."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_reconciled_panels.py"
    spec = importlib.util.spec_from_file_location("_test_current_policy_binding_sync", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load_sync_module()


EXPECTED_BINDINGS = {
    "policy": "kga/policy.py",
    "certificate": "kga/certificate.py",
    "numeric_validation": "kga/_validation.py",
    "preregistered_protocol": "research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml",
}
CONSUMERS = (manifest, sync)


@pytest.fixture
def bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    for module in (producer, *CONSUMERS):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    for name, relative_path in EXPECTED_BINDINGS.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source for {name}\n", encoding="utf-8")
    return producer.current_policy_code_bindings()


def test_producer_and_consumers_require_the_same_exact_four_bindings(bindings: dict) -> None:
    for module in (producer, *CONSUMERS):
        assert module.CURRENT_POLICY_BINDING_PATHS == EXPECTED_BINDINGS
    assert set(bindings) == set(EXPECTED_BINDINGS)
    for name, relative_path in EXPECTED_BINDINGS.items():
        expected_hash = hashlib.sha256((producer.ROOT / relative_path).read_bytes()).hexdigest()
        assert bindings[name] == {"path": relative_path, "sha256": expected_hash}
    for module in CONSUMERS:
        assert module._validated_current_policy_bindings(bindings) is bindings


def test_in_memory_artifact_builder_records_the_complete_binding_inventory(
    bindings: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = producer.ROOT / "synthetic-analysis.py"
    script.write_text("synthetic analysis source\n", encoding="utf-8")
    monkeypatch.setattr(producer, "__file__", str(script))
    monkeypatch.setattr(producer, "git_head", lambda: "synthetic-commit")

    def synthetic_candidate(source_dir, candidate, **kwargs):
        assert source_dir == producer.ROOT / "unused-synthetic-panel"
        return {
            "candidate": candidate,
            "comparisons": {
                baseline: {"p_value_one_sided_exact_sign_flip": 0.5}
                for baseline in producer.BASELINES
            },
            "gate": {"both_pointwise_95pct_cluster_bootstrap_intervals_positive": False},
        }

    monkeypatch.setattr(producer, "analyze_candidate", synthetic_candidate)
    artifact = producer.build_artifact(
        Namespace(
            source_dir=producer.ROOT / "unused-synthetic-panel",
            candidates=["tent", "eata", "sar"],
            n_boot=10,
            seed=1,
            ci_level=0.95,
        )
    )
    assert artifact["live_code_bindings"] == bindings
    assert artifact[producer.FAMILY_FIELD]["family_size"] == 6


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
@pytest.mark.parametrize("missing", tuple(EXPECTED_BINDINGS))
def test_each_missing_binding_fails_closed(bindings: dict, module, missing: str) -> None:
    incomplete = copy.deepcopy(bindings)
    incomplete.pop(missing)
    with pytest.raises(ValueError, match="bind exactly"):
        module._validated_current_policy_bindings(incomplete)


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
@pytest.mark.parametrize("invalid", [None, {}, [], "bindings"])
def test_absent_empty_or_nonmapping_bindings_fail_closed(module, invalid) -> None:
    with pytest.raises(ValueError, match="bind exactly"):
        module._validated_current_policy_bindings(invalid)


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
def test_extra_unrecognized_binding_is_not_an_accepted_inventory(bindings: dict, module) -> None:
    extra = copy.deepcopy(bindings)
    extra["other_helper"] = extra["policy"]
    with pytest.raises(ValueError, match="bind exactly"):
        module._validated_current_policy_bindings(extra)


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
@pytest.mark.parametrize("name", tuple(EXPECTED_BINDINGS))
def test_a_valid_hash_for_the_wrong_file_cannot_substitute_a_dependency(bindings: dict, module, name: str) -> None:
    substituted = copy.deepcopy(bindings)
    wrong_name = "certificate" if name == "policy" else "policy"
    substituted[name] = dict(bindings[wrong_name])
    with pytest.raises(ValueError, match=f"{name} binding has invalid path/hash metadata"):
        module._validated_current_policy_bindings(substituted)


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
@pytest.mark.parametrize("value", [None, [], {}, {"path": "kga/_validation.py", "sha256": None}])
def test_malformed_binding_row_fails_with_validation_error(bindings: dict, module, value) -> None:
    malformed = copy.deepcopy(bindings)
    malformed["numeric_validation"] = value
    with pytest.raises(ValueError, match="numeric_validation binding"):
        module._validated_current_policy_bindings(malformed)


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
def test_changing_only_numeric_validation_invalidates_old_provenance(bindings: dict, module) -> None:
    helper = producer.ROOT / EXPECTED_BINDINGS["numeric_validation"]
    helper.write_text("synthetic changed mask handling\n", encoding="utf-8")
    refreshed = producer.current_policy_code_bindings()
    assert refreshed["policy"] == bindings["policy"]
    assert refreshed["certificate"] == bindings["certificate"]
    assert refreshed["preregistered_protocol"] == bindings["preregistered_protocol"]
    assert refreshed["numeric_validation"]["sha256"] != bindings["numeric_validation"]["sha256"]
    with pytest.raises(ValueError, match="numeric_validation binding"):
        module._validated_current_policy_bindings(bindings)
    assert module._validated_current_policy_bindings(refreshed) is refreshed


@pytest.mark.parametrize("module", CONSUMERS, ids=["manifest", "sync"])
def test_missing_helper_source_file_invalidates_an_otherwise_complete_binding(bindings: dict, module) -> None:
    helper = producer.ROOT / EXPECTED_BINDINGS["numeric_validation"]
    helper.rename(helper.with_suffix(".unavailable"))
    with pytest.raises(ValueError, match="numeric_validation binding"):
        module._validated_current_policy_bindings(bindings)


def _minimal_metadata(bindings: dict) -> dict:
    return {
        "schema": producer.SCHEMA,
        "contrast_convention": producer.CI_CONVENTION,
        producer.FAMILY_FIELD: {"family_size": 6, "alpha": 0.05},
        "live_code_bindings": bindings,
    }


def test_sync_entrypoint_checks_bindings_before_using_candidate_metrics(bindings: dict) -> None:
    incomplete = copy.deepcopy(bindings)
    incomplete.pop("numeric_validation")
    with pytest.raises(ValueError, match="bind exactly"):
        sync._normalized_current_cluster(_minimal_metadata(incomplete), "tent")


def test_manifest_entrypoint_checks_bindings_before_using_candidate_metrics(
    bindings: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = copy.deepcopy(bindings)
    incomplete.pop("numeric_validation")
    metadata = _minimal_metadata(incomplete)
    analysis_script = manifest.ROOT / "analysis.py"
    analysis_script.write_text("synthetic analysis source\n", encoding="utf-8")
    metadata["analysis_script"] = "analysis.py"
    metadata["analysis_script_sha256"] = hashlib.sha256(analysis_script.read_bytes()).hexdigest()
    artifact = manifest.ROOT / "synthetic-inference.json"
    artifact.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(manifest, "CURRENT_CLUSTER", artifact)
    with pytest.raises(ValueError, match="bind exactly"):
        manifest.current_cluster_metrics()
