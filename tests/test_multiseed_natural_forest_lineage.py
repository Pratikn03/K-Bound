import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "docs/research/kbound/scripts/make_multiseed_natural_forest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_multiseed_forest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


forest_module = _load_module()

HELDOUT_SOURCE_SCHEMA = "kbound_natural_locked_holdout_source_v1"
DECISION_LOCK_SCHEMA = "kbound_natural_decision_lock_v3"
CALIBRATION_SOURCE_SCHEMA = "kbound_natural_calibration_source_v1"
ESTIMATOR_ARTIFACT_SCHEMA = "kbound_natural_locked_estimator_v1"
PROTOCOL_ARTIFACT_SCHEMA = "kbound_natural_locked_protocol_v1"
PREOPENING_RECEIPT_SCHEMA = "kbound_natural_preopening_receipt_v1"
METRIC_CONTRACT = {
    "name": "accuracy",
    "official": True,
    "direction": "higher_is_better",
    "unit": "per_condition",
    "range": [0.0, 1.0],
}
TTA_PROTOCOL = {
    "schema": "kbound_tta_candidate_protocol_v1",
    "mode": "online",
    "semantics": "online_disjoint_stream_update_then_transductive_bn_evaluation",
    "requires_auxiliary_stream_eval_disjoint": True,
    "gradient_update_reads_eval_x": False,
    "prediction_uses_eval_batch_statistics": True,
    "candidate_evaluation_is_transductive": True,
    "candidate_adaptation_eval_disjoint": False,
    "target_labels_used_for_adaptation_or_prediction": False,
}


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _lineage_bundle(tmp_path, *, scope="development", seed_kind="model_seed"):
    source = tmp_path / "source.json"
    heldout = scope == "heldout"
    candidate = "sar_online_aggressive"
    dataset = "officehome"
    partition = "target_test" if heldout else "target_val"
    helpful_condition = f"Art|{partition}|iid|tiny"
    harmful_condition = f"Clipart|{partition}|iid|tiny"
    source_records = []
    for seed in [0, 1]:
        source_records.extend(
            [
                {
                    "model_seed": seed,
                    "seed": 17,
                    "candidate": candidate,
                    "domain": "Art",
                    "split": partition,
                    "comp": "iid",
                    "regime": "tiny",
                    "a0": 0.5,
                    "aa": 0.6,
                    "B": 0.1,
                    "tta_protocol": TTA_PROTOCOL,
                },
                {
                    "model_seed": seed,
                    "seed": 17,
                    "candidate": candidate,
                    "domain": "Clipart",
                    "split": partition,
                    "comp": "iid",
                    "regime": "tiny",
                    "a0": 0.6,
                    "aa": 0.5,
                    "B": -0.1,
                    "tta_protocol": TTA_PROTOCOL,
                },
            ]
        )
    source_document = {"records": source_records}
    checkpoint_hashes = {}
    checkpoint_tensor_hashes = {}
    if heldout:
        checkpoint_artifacts = {}
        for seed in [0, 1]:
            checkpoint_path = tmp_path / f"checkpoint_seed{seed}.pt"
            torch.save(
                {"model": {"weight": torch.tensor([float(seed), 1.0])}},
                checkpoint_path,
            )
            checkpoint_hash = forest_module._sha256_file(checkpoint_path)
            checkpoint_tensor_hash = forest_module._checkpoint_tensor_sha256(checkpoint_path)
            checkpoint_hashes[str(seed)] = checkpoint_hash
            checkpoint_tensor_hashes[str(seed)] = checkpoint_tensor_hash
            checkpoint_artifacts[str(seed)] = {
                "file": checkpoint_path.name,
                "sha256": checkpoint_hash,
                "tensor_sha256": checkpoint_tensor_hash,
                "model_seed": seed,
            }
        for record in source_records:
            seed = str(record["model_seed"])
            record["checkpoint_sha256"] = checkpoint_hashes[seed]
            record["checkpoint_tensor_sha256"] = checkpoint_tensor_hashes[seed]
        source_document.update(
            {
                "schema": HELDOUT_SOURCE_SCHEMA,
                "dataset": dataset,
                "evaluation_partition": partition,
                "metric_contract": METRIC_CONTRACT,
                "publication_eligible": False,
                "computational_candidate_ready": True,
                "checkpoint_artifacts_by_seed": checkpoint_artifacts,
                "conditions": [
                    {
                        "record_index": index,
                        "model_seed": int(record["model_seed"]),
                        "candidate": candidate,
                        "condition": (
                            helpful_condition
                            if record["domain"] == "Art"
                            else harmful_condition
                        ),
                        "status": "complete",
                    }
                    for index, record in enumerate(source_records)
                ],
                "completion_ledger": {
                    "status": "complete",
                    "execution_complete": True,
                    "expected": 4,
                    "completed": 4,
                    "failed": 0,
                    "pending": 0,
                    "failure_history": [],
                },
            }
        )
    _write_json(source, source_document)
    source_sha = forest_module._sha256_file(source)

    decision_lock = None
    if heldout:
        calibration_source = tmp_path / "calibration_source.json"
        estimator_artifact = tmp_path / "locked_estimator.json"
        protocol_artifact = tmp_path / "locked_protocol.json"
        receipt_path = tmp_path / "preopening_receipt.json"
        locked_routes = [
            {
                "seed": seed,
                "condition": condition,
                "b_hat": 0.2 if condition == helpful_condition else -0.2,
                "eps_conformal": 0.05,
                "kga_decision": "ADAPT" if condition == helpful_condition else "FREEZE",
            }
            for seed in [0, 1]
            for condition in [helpful_condition, harmful_condition]
        ]
        calibration_rows = [
            {
                "seed": route["seed"],
                "condition": route["condition"],
                "b_hat": route["b_hat"],
                "eps_conformal": route["eps_conformal"],
                "kga_decision": route["kga_decision"],
                "absolute_residuals": [0.05] * 10,
            }
            for route in locked_routes
        ]
        shared_lock_contract = {
            "dataset": dataset,
            "candidate": candidate,
            "metric_contract": METRIC_CONTRACT,
            "alpha": 0.1,
            "model_seeds": [0, 1],
            "calibration_partition": "target_val",
            "evaluation_partition": partition,
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 0,
        }
        _write_json(
            calibration_source,
            {
                "schema": CALIBRATION_SOURCE_SCHEMA,
                **shared_lock_contract,
                "calibration_rows": calibration_rows,
            },
        )
        _write_json(
            estimator_artifact,
            {
                "schema": ESTIMATOR_ARTIFACT_SCHEMA,
                **shared_lock_contract,
                "estimator_backend": forest_module.LOCKED_BACKEND,
                "locked_routes": locked_routes,
            },
        )
        expected_conditions = [
            {"seed": route["seed"], "condition": route["condition"]}
            for route in locked_routes
        ]
        _write_json(
            protocol_artifact,
            {
                "schema": PROTOCOL_ARTIFACT_SCHEMA,
                **shared_lock_contract,
                "expected_conditions": expected_conditions,
            },
        )
        calibration_sha = forest_module._sha256_file(calibration_source)
        estimator_sha = forest_module._sha256_file(estimator_artifact)
        protocol_sha = forest_module._sha256_file(protocol_artifact)
        _write_json(
            receipt_path,
            {
                "schema": PREOPENING_RECEIPT_SCHEMA,
                **shared_lock_contract,
                "calibration_source_sha256": calibration_sha,
                "estimator_artifact_sha256": estimator_sha,
                "protocol_sha256": protocol_sha,
                "locked_routes_sha256": forest_module._sha256_json(locked_routes),
                "immutable_uri": "https://osf.io/abc12/",
                "provider": "osf",
                "locked_at_utc": "2026-08-26T12:00:00Z",
                "target_opened_at_utc": None,
            },
        )
        lock_path = tmp_path / "decision_lock.json"
        _write_json(
            lock_path,
            {
                "schema": DECISION_LOCK_SCHEMA,
                **shared_lock_contract,
                "decisions_locked_before_evaluation": True,
                "target_opened_before_lock": False,
                "estimator_backend": forest_module.LOCKED_BACKEND,
                "calibration_source": {
                    "file": calibration_source.name,
                    "sha256": calibration_sha,
                },
                "estimator_artifact": {
                    "file": estimator_artifact.name,
                    "sha256": estimator_sha,
                },
                "protocol_artifact": {
                    "file": protocol_artifact.name,
                    "sha256": protocol_sha,
                },
                "immutable_receipt": {
                    "file": receipt_path.name,
                    "sha256": forest_module._sha256_file(receipt_path),
                },
                "locked_routes": locked_routes,
                "locked_routes_sha256": forest_module._sha256_json(locked_routes),
            },
        )
        decision_lock = {
            "file": lock_path.name,
            "sha256": forest_module._sha256_file(lock_path),
        }
    per_files = []
    per_hashes = {}
    checkpoints = {}
    checkpoint_tensors = {}
    serialization_generation_id = "d" * 64
    for seed in [0, 1]:
        name = f"per_condition_{dataset}_{candidate}_seed{seed}.json"
        path = tmp_path / name
        checkpoint = None
        if seed_kind == "model_seed":
            checkpoint = (
                checkpoint_hashes[str(seed)]
                if heldout
                else ("a" if seed == 0 else "b") * 64
            )
            checkpoint_tensor = (
                checkpoint_tensor_hashes[str(seed)]
                if heldout
                else ("c" if seed == 0 else "d") * 64
            )
        else:
            checkpoint_tensor = None
        _write_json(
            path,
            {
                "extract_contract": forest_module.LINEAGE_CONTRACT,
                "benchmark": dataset,
                "method": candidate,
                "seed": seed,
                "model_seed": seed if seed_kind == "model_seed" else None,
                "seed_kind": seed_kind,
                "evaluation_partition": "target_test" if scope == "heldout" else "target_val",
                "metric_contract": METRIC_CONTRACT if heldout else None,
                "alpha": 0.1,
                "checkpoint_sha256": checkpoint,
                "checkpoint_tensor_sha256": checkpoint_tensor,
                "tta_protocol": TTA_PROTOCOL,
                "tta_protocol_sha256": forest_module._sha256_json(TTA_PROTOCOL),
                "serialization_generation_id": serialization_generation_id,
                "decision_lock_sha256": decision_lock["sha256"] if heldout else None,
                "kga_backend": (
                    forest_module.LOCKED_BACKEND
                    if heldout
                    else "sklearn_gradient_boost_crossfit_split"
                ),
                "estimator_publication_eligible": False,
                "estimator_computationally_locked": heldout,
                "n_calibration_infeasible": 0,
                "records": [
                    {
                        "condition": helpful_condition,
                        "B": 0.1,
                        "a0": 0.5,
                        "a_adapted": 0.6,
                        "a_kbound": 0.6,
                        "a_oracle": 0.6,
                        "b_hat": 0.2,
                        "eps_conformal": 0.05,
                        "kga_decision": "ADAPT",
                        "calibration_feasible": True,
                        "radius_status": "FINITE",
                        "source_file_sha256": source_sha,
                        "source_record_index": seed * 2,
                        "source_record_sha256": forest_module._sha256_json(
                            source_records[seed * 2]
                        ),
                        "tta_protocol": TTA_PROTOCOL,
                        "tta_protocol_sha256": forest_module._sha256_json(TTA_PROTOCOL),
                    },
                    {
                        "condition": harmful_condition,
                        "B": -0.1,
                        "a0": 0.6,
                        "a_adapted": 0.5,
                        "a_kbound": 0.6,
                        "a_oracle": 0.6,
                        "b_hat": -0.2,
                        "eps_conformal": 0.05,
                        "kga_decision": "FREEZE",
                        "calibration_feasible": True,
                        "radius_status": "FINITE",
                        "source_file_sha256": source_sha,
                        "source_record_index": seed * 2 + 1,
                        "source_record_sha256": forest_module._sha256_json(
                            source_records[seed * 2 + 1]
                        ),
                        "tta_protocol": TTA_PROTOCOL,
                        "tta_protocol_sha256": forest_module._sha256_json(TTA_PROTOCOL),
                    },
                ],
            },
        )
        per_files.append(name)
        per_hashes[name] = forest_module._sha256_file(path)
        if checkpoint:
            checkpoints[str(seed)] = checkpoint
            checkpoint_tensors[str(seed)] = checkpoint_tensor

    aggregate = {
        "schema": forest_module.AGGREGATE_SCHEMA,
        "lineage_contract": forest_module.LINEAGE_CONTRACT,
        "dataset": dataset,
        "candidate": candidate,
        "tta_protocol": TTA_PROTOCOL,
        "tta_protocol_sha256": forest_module._sha256_json(TTA_PROTOCOL),
        "metric_contract": METRIC_CONTRACT if heldout else None,
        "analysis": (
            "validation_locked_disjoint_test"
            if heldout
            else "crossfit_split_within_development_partition_single_candidate"
        ),
        "seed_kind": seed_kind,
        "inference_unit": (
            "independent model checkpoint" if seed_kind == "model_seed" else "stream order"
        ),
        "evaluation_partition": "target_test" if heldout else "target_val",
        "calibration_partition": "target_val" if heldout else None,
        "decisions_locked_before_evaluation": heldout,
        "target_opened_before_lock": False if heldout else None,
        "decision_lock": decision_lock,
        "external_authenticity_verified": False,
        "publication_eligible": False,
        "sources_publication_ready": False,
        "sources_computationally_ready": heldout,
        "estimator_publication_eligible": False,
        "estimator_computationally_locked": heldout,
        "calibration_feasible_all": True,
        "n_calibration_infeasible_total": 0,
        "statistical_verdict_withheld": heldout,
        "model_seed_ci_eligible": (not heldout) and seed_kind == "model_seed",
        "confirmatory_ci_eligible": False,
        "heldout_promotion_eligible": False,
        "beats_both_promoted": False,
        "seeds": [0, 1],
        "n_seeds": 2,
        "alpha": 0.1,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 0,
        "regret_kga": [0.0, 0.0],
        "regret_adapt": [0.05, 0.0],
        "regret_freeze": [0.05, 0.0],
        "gap_vs_adapt": {"mean": 0.05, "ci95": [0.05, 0.05]},
        "gap_vs_freeze": {"mean": 0.05, "ci95": [0.05, 0.05]},
        "gap_vs_better_ci95": [0.05, 0.05],
        "gap_vs_worse_ci95": [0.05, 0.05],
        "better_policy": "freeze",
        "FA_u_per_seed": [0.0, 0.0],
        "FA_u_max": 0.0,
        "verdict_code": (
            "WITHHELD_PENDING_INDEPENDENT_AUTHENTICITY_AUDIT"
            if heldout
            else "DEVELOPMENT_BEATS_BOTH_DIAGNOSTIC"
        ),
        "verdict": (
            "WITHHELD_PENDING_INDEPENDENT_AUTHENTICITY_AUDIT"
            if heldout
            else "DEVELOPMENT_BEATS_BOTH_DIAGNOSTIC"
        ),
        "files": per_files,
        "file_sha256": per_hashes,
        "checkpoint_sha256_by_seed": checkpoints,
        "checkpoint_tensor_sha256_by_seed": checkpoint_tensors,
        "kga_backend": [
            forest_module.LOCKED_BACKEND
            if heldout
            else "sklearn_gradient_boost_crossfit_split"
        ],
    }
    aggregate_path = tmp_path / f"multiseed_{dataset}_{candidate}.json"
    _write_json(aggregate_path, aggregate)
    serializer_manifest_path = tmp_path / f"per_condition_{dataset}_manifest.json"
    _write_json(
        serializer_manifest_path,
        {
            "schema": "kbound_per_condition_generation_v1",
            "generation_id": serialization_generation_id,
            "generation_committed": True,
            "dataset": dataset,
            "methods": [candidate],
            "seeds": [0, 1],
            "expected_cells": 2,
            "validated_tta_protocol_by_candidate": {candidate: TTA_PROTOCOL},
            "files": {
                name: {
                    "sha256": forest_module._sha256_file(tmp_path / name),
                    "n_conditions": 2,
                }
                for name in per_files
            },
        },
    )
    manifest = {
        "schema": forest_module.EXTRACT_SCHEMA,
        "track": dataset,
        "requested_candidates": [candidate],
        "validated_tta_protocol_by_candidate": {candidate: TTA_PROTOCOL},
        "expected_seeds": [0, 1],
        "seed_kind": seed_kind,
        "sources": [
            {
                "path": str(source.resolve()),
                "sha256": forest_module._sha256_file(source),
                "partition": aggregate["evaluation_partition"],
                "publication": {
                    "publication_ready": False,
                    "computational_candidate_ready": heldout,
                },
            }
        ],
        "aggregates": [aggregate_path.name],
        "serialize": {
            "written": per_files,
            "manifest": serializer_manifest_path.name,
            "generation_id": serialization_generation_id,
        },
        "aggregate_sha256": {
            aggregate_path.name: forest_module._sha256_file(aggregate_path)
        },
    }
    manifest_path = tmp_path / f"extract_manifest_{dataset}.json"
    _write_json(manifest_path, manifest)
    return aggregate_path, aggregate, manifest_path


def _rewrite_aggregate(path, aggregate, manifest_path):
    _write_json(path, aggregate)
    manifest = json.loads(manifest_path.read_text())
    serializer_path = manifest_path.parent / manifest["serialize"]["manifest"]
    serializer = json.loads(serializer_path.read_text())
    for name in serializer["files"]:
        serializer["files"][name]["sha256"] = forest_module._sha256_file(
            manifest_path.parent / name
        )
    _write_json(serializer_path, serializer)
    manifest["aggregate_sha256"][path.name] = forest_module._sha256_file(path)
    _write_json(manifest_path, manifest)


def _refresh_heldout_bundle(tmp_path, aggregate_path, manifest_path):
    """Refresh every downstream hash after an adversarial fixture mutation."""
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    source_sha = forest_module._sha256_file(source_path)

    lock_path = tmp_path / "decision_lock.json"
    lock = json.loads(lock_path.read_text())
    for field in ("calibration_source", "estimator_artifact", "protocol_artifact"):
        reference = lock.get(field)
        if isinstance(reference, dict) and reference.get("file"):
            reference["sha256"] = forest_module._sha256_file(tmp_path / reference["file"])

    receipt_reference = lock.get("immutable_receipt")
    if isinstance(receipt_reference, dict) and receipt_reference.get("file"):
        receipt_path = tmp_path / receipt_reference["file"]
        receipt = json.loads(receipt_path.read_text())
        artifact_fields = {
            "calibration_source_sha256": "calibration_source",
            "estimator_artifact_sha256": "estimator_artifact",
            "protocol_sha256": "protocol_artifact",
        }
        for receipt_field, lock_field in artifact_fields.items():
            if receipt_field in receipt and isinstance(lock.get(lock_field), dict):
                receipt[receipt_field] = lock[lock_field]["sha256"]
        if "locked_routes_sha256" in receipt:
            receipt["locked_routes_sha256"] = forest_module._sha256_json(
                lock.get("locked_routes", [])
            )
        _write_json(receipt_path, receipt)
        receipt_reference["sha256"] = forest_module._sha256_file(receipt_path)

    _write_json(lock_path, lock)
    lock_sha = forest_module._sha256_file(lock_path)

    aggregate = json.loads(aggregate_path.read_text())
    if isinstance(aggregate.get("decision_lock"), dict):
        aggregate["decision_lock"]["sha256"] = lock_sha
    per_hashes = {}
    for name in aggregate["files"]:
        per_path = tmp_path / name
        document = json.loads(per_path.read_text())
        if "decision_lock_sha256" in document:
            document["decision_lock_sha256"] = lock_sha
        for record in document.get("records", []):
            record["source_file_sha256"] = source_sha
            index = int(record["source_record_index"])
            record["source_record_sha256"] = forest_module._sha256_json(
                source["records"][index]
            )
        _write_json(per_path, document)
        per_hashes[name] = forest_module._sha256_file(per_path)
    aggregate["file_sha256"] = per_hashes
    _write_json(aggregate_path, aggregate)

    manifest = json.loads(manifest_path.read_text())
    serializer_path = tmp_path / manifest["serialize"]["manifest"]
    serializer = json.loads(serializer_path.read_text())
    for name, digest in per_hashes.items():
        serializer["files"][name]["sha256"] = digest
    _write_json(serializer_path, serializer)
    manifest["sources"][0]["sha256"] = source_sha
    manifest["aggregate_sha256"][aggregate_path.name] = forest_module._sha256_file(
        aggregate_path
    )
    _write_json(manifest_path, manifest)


def test_default_scope_accepts_only_disjoint_validation_locked_test(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="heldout")
    loaded = forest_module._load_aggs([str(path)])
    assert len(loaded) == 1
    assert loaded[0][1]["heldout_promotion_eligible"] is False


@pytest.mark.parametrize(
    "field",
    [
        "external_authenticity_verified",
        "publication_eligible",
        "sources_publication_ready",
        "confirmatory_ci_eligible",
    ],
)
def test_development_scope_rejects_publication_or_authenticity_self_promotion(
    tmp_path, field
):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="development")
    aggregate[field] = True
    _rewrite_aggregate(path, aggregate, manifest_path)

    with pytest.raises(
        forest_module.LineageError,
        match="development diagnostic carries publication/confirmatory flags",
    ):
        forest_module._load_aggs([str(path)], scope="development-diagnostic")


def test_heldout_rejects_coherent_out_of_range_accuracy_with_refreshed_hashes(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    for record in source["records"]:
        record["a0"] += 1.0
        record["aa"] += 1.0
    _write_json(source_path, source)

    # An affine shift preserves every benefit, regret, route, and aggregate CI;
    # only the reviewed accuracy range distinguishes it from valid evidence.
    for name in aggregate["files"]:
        per_path = tmp_path / name
        document = json.loads(per_path.read_text())
        for record in document["records"]:
            for field in ("a0", "a_adapted", "a_kbound", "a_oracle"):
                record[field] += 1.0
        _write_json(per_path, document)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError, match="outside the reviewed metric range"):
        forest_module._load_aggs([str(path)])


@pytest.mark.parametrize("record_split", ["target_val", None])
def test_heldout_rejects_nonheldout_or_missing_record_split_under_test_envelope(
    tmp_path, record_split
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    for record in source["records"]:
        if record_split is None:
            record.pop("split")
        else:
            record["split"] = record_split
    _write_json(source_path, source)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(
        forest_module.LineageError,
        match="partition disagrees|cannot form a condition key",
    ):
        forest_module._load_aggs([str(path)])


def test_heldout_rejects_target_scored_competing_candidate_even_with_complete_ledger(
    tmp_path,
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    for seed in [0, 1]:
        checkpoint = source["checkpoint_artifacts_by_seed"][str(seed)]
        record_index = len(source["records"])
        source["records"].append(
            {
                "model_seed": seed,
                "seed": 17,
                "candidate": "target_selected_competitor",
                "domain": "Product",
                "split": "target_test",
                "comp": "iid",
                "regime": "tiny",
                "a0": 0.5,
                "aa": 0.9,
                "B": 0.4,
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
                "tta_protocol": TTA_PROTOCOL,
            }
        )
        source["conditions"].append(
            {
                "record_index": record_index,
                "model_seed": seed,
                "candidate": "target_selected_competitor",
                "condition": "Product|target_test|iid|tiny",
                "status": "complete",
            }
        )
    source["completion_ledger"]["expected"] = len(source["records"])
    source["completion_ledger"]["completed"] = len(source["records"])
    _write_json(source_path, source)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError, match="unregistered competing candidate"):
        forest_module._load_aggs([str(path)])


def test_heldout_rejects_extra_unscored_harmful_source_rows_even_with_refreshed_hashes(
    tmp_path,
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    for seed in [0, 1]:
        checkpoint = source["checkpoint_artifacts_by_seed"][str(seed)]
        source["records"].append(
            {
                "model_seed": seed,
                "seed": 17,
                "candidate": "sar_online_aggressive",
                "domain": "Product",
                "split": "target_test",
                "comp": "iid",
                "regime": "tiny",
                "a0": 0.9,
                "aa": 0.0,
                "B": -0.9,
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_tensor_sha256": checkpoint["tensor_sha256"],
                "tta_protocol": TTA_PROTOCOL,
            }
        )
    _write_json(source_path, source)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError):
        forest_module._load_aggs([str(path)])


@pytest.mark.parametrize("mutation", ["finder_schema", "unofficial_metric"])
def test_heldout_rejects_diagnostic_source_schema_or_metric_even_if_self_promoted(
    tmp_path, mutation
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    if mutation == "finder_schema":
        source["schema"] = "kbound_officehome_v4"
    else:
        source["metric_contract"] = {
            **METRIC_CONTRACT,
            "name": "diagnostic_per_cell_macro_f1",
            "official": False,
        }
    _write_json(source_path, source)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError):
        forest_module._load_aggs([str(path)])


@pytest.mark.parametrize("mutation", ["placeholder_artifacts", "missing_receipt"])
def test_heldout_rejects_placeholder_lock_artifacts_or_missing_immutable_receipt(
    tmp_path, mutation
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    if mutation == "placeholder_artifacts":
        _write_json(tmp_path / "calibration_source.json", {"partition": "target_val", "n": 20})
        _write_json(
            tmp_path / "locked_estimator.json",
            {"backend": forest_module.LOCKED_BACKEND},
        )
    else:
        lock_path = tmp_path / "decision_lock.json"
        lock = json.loads(lock_path.read_text())
        lock.pop("immutable_receipt")
        _write_json(lock_path, lock)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError):
        forest_module._load_aggs([str(path)])


@pytest.mark.parametrize("mutation", ["missing_metric", "unbound_checkpoints"])
def test_heldout_rejects_missing_metric_or_arbitrary_unbound_checkpoint_hashes(
    tmp_path, mutation
):
    path, _, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    if mutation == "missing_metric":
        for name in [
            "source.json",
            "calibration_source.json",
            "locked_estimator.json",
            "locked_protocol.json",
            "preopening_receipt.json",
            "decision_lock.json",
        ]:
            artifact_path = tmp_path / name
            artifact = json.loads(artifact_path.read_text())
            artifact.pop("metric_contract", None)
            _write_json(artifact_path, artifact)
        aggregate = json.loads(path.read_text())
        aggregate.pop("metric_contract", None)
        for name in aggregate["files"]:
            per_path = tmp_path / name
            document = json.loads(per_path.read_text())
            document.pop("metric_contract", None)
            _write_json(per_path, document)
        _write_json(path, aggregate)
    else:
        aggregate = json.loads(path.read_text())
        arbitrary = {"0": "a" * 64, "1": "b" * 64}
        aggregate["checkpoint_sha256_by_seed"] = arbitrary
        for name in aggregate["files"]:
            per_path = tmp_path / name
            document = json.loads(per_path.read_text())
            document["checkpoint_sha256"] = arbitrary[str(document["seed"])]
            _write_json(per_path, document)
        _write_json(path, aggregate)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError):
        forest_module._load_aggs([str(path)])


def test_heldout_scope_rejects_legacy_source_even_with_complete_ledger(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="heldout")
    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    source["schema"] = "legacy_natural_result_v1"
    _write_json(source_path, source)
    manifest_path = tmp_path / "extract_manifest_officehome.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][0]["sha256"] = forest_module._sha256_file(source_path)
    manifest["sources"][0]["publication"]["publication_ready"] = False
    _write_json(manifest_path, manifest)
    with pytest.raises(
        forest_module.LineageError,
        match="computational-candidate|dedicated locked scorer",
    ):
        forest_module._load_aggs([str(path)])


def test_default_scope_rejects_development_loo(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="development")
    with pytest.raises(
        forest_module.LineageError,
        match="publication_eligible|development/opened partition",
    ):
        forest_module._load_aggs([str(path)])
    loaded = forest_module._load_aggs([str(path)], scope="development-diagnostic")
    assert len(loaded) == 1


def test_stream_seed_evidence_is_never_admitted_to_ci_forest(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="development", seed_kind="stream_seed")
    with pytest.raises(forest_module.LineageError, match="stream-seed evidence"):
        forest_module._load_aggs([str(path)], scope="development-diagnostic")


def test_opened_test_is_rejected_even_if_other_flags_claim_promotable(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    aggregate["target_opened_before_lock"] = True
    _rewrite_aggregate(path, aggregate, manifest_path)
    with pytest.raises(forest_module.LineageError, match="opened before"):
        forest_module._load_aggs([str(path)])


def test_repeated_checkpoint_cannot_masquerade_as_model_seed_replication(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    aggregate["checkpoint_sha256_by_seed"]["1"] = aggregate["checkpoint_sha256_by_seed"]["0"]
    # Keep the per-condition payload internally aligned so the dedicated
    # independent-checkpoint guard is what rejects this artifact.
    per_seed1 = tmp_path / aggregate["files"][1]
    document = json.loads(per_seed1.read_text())
    document["checkpoint_sha256"] = aggregate["checkpoint_sha256_by_seed"]["1"]
    _write_json(per_seed1, document)
    aggregate["file_sha256"][per_seed1.name] = forest_module._sha256_file(per_seed1)
    _rewrite_aggregate(path, aggregate, manifest_path)
    with pytest.raises(forest_module.LineageError, match="valid and unique"):
        forest_module._load_aggs([str(path)])


def test_byte_different_containers_with_same_tensor_state_fail_forest(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    seed0_path = tmp_path / "checkpoint_seed0.pt"
    seed1_path = tmp_path / "checkpoint_seed1.pt"
    seed0_payload = torch.load(seed0_path, map_location="cpu", weights_only=True)
    torch.save(
        {"model": seed0_payload["model"], "container_metadata": {"declared_seed": 1}},
        seed1_path,
    )
    assert forest_module._sha256_file(seed0_path) != forest_module._sha256_file(seed1_path)
    tensor_sha = forest_module._checkpoint_tensor_sha256(seed0_path)
    assert forest_module._checkpoint_tensor_sha256(seed1_path) == tensor_sha

    source_path = tmp_path / "source.json"
    source = json.loads(source_path.read_text())
    seed1_identity = source["checkpoint_artifacts_by_seed"]["1"]
    seed1_identity["sha256"] = forest_module._sha256_file(seed1_path)
    seed1_identity["tensor_sha256"] = tensor_sha
    for record in source["records"]:
        if int(record["model_seed"]) == 1:
            record["checkpoint_sha256"] = seed1_identity["sha256"]
            record["checkpoint_tensor_sha256"] = tensor_sha
    _write_json(source_path, source)

    aggregate["checkpoint_sha256_by_seed"]["1"] = seed1_identity["sha256"]
    aggregate["checkpoint_tensor_sha256_by_seed"]["1"] = tensor_sha
    per_seed1 = tmp_path / aggregate["files"][1]
    document = json.loads(per_seed1.read_text())
    document["checkpoint_sha256"] = seed1_identity["sha256"]
    document["checkpoint_tensor_sha256"] = tensor_sha
    _write_json(per_seed1, document)
    aggregate["file_sha256"][per_seed1.name] = forest_module._sha256_file(per_seed1)
    _write_json(path, aggregate)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError, match="tensor-state hashes.*unique"):
        forest_module._load_aggs([str(path)])


def test_per_condition_model_seed_must_match_checkpoint_inventory_key(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    per_seed0 = tmp_path / aggregate["files"][0]
    document = json.loads(per_seed0.read_text())
    document["model_seed"] = 1
    _write_json(per_seed0, document)
    aggregate["file_sha256"][per_seed0.name] = forest_module._sha256_file(per_seed0)
    _write_json(path, aggregate)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError, match="seed/model_seed binding mismatch"):
        forest_module._load_aggs([str(path)])


def test_committed_extracted_row_cannot_tamper_tta_protocol(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    per_seed0 = tmp_path / aggregate["files"][0]
    document = json.loads(per_seed0.read_text())
    document["records"][0]["tta_protocol"] = {
        **TTA_PROTOCOL,
        "gradient_update_reads_eval_x": True,
    }
    document["records"][0]["tta_protocol_sha256"] = forest_module._sha256_json(
        document["records"][0]["tta_protocol"]
    )
    _write_json(per_seed0, document)
    aggregate["file_sha256"][per_seed0.name] = forest_module._sha256_file(per_seed0)
    _write_json(path, aggregate)
    _refresh_heldout_bundle(tmp_path, path, manifest_path)

    with pytest.raises(forest_module.LineageError, match="does not preserve candidate tta_protocol"):
        forest_module._load_aggs([str(path)])


def test_tampered_per_condition_file_fails_hash_verification(tmp_path):
    path, aggregate, _ = _lineage_bundle(tmp_path, scope="heldout")
    per_path = tmp_path / aggregate["files"][0]
    per_path.write_text(per_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="per-condition hash mismatch"):
        forest_module._load_aggs([str(path)])


def test_tampered_source_file_fails_hash_verification(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="heldout")
    (tmp_path / "source.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="source hash mismatch"):
        forest_module._load_aggs([str(path)])


def test_tampered_decision_lock_fails_before_confirmatory_render(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="heldout")
    (tmp_path / "decision_lock.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="decision-lock hash mismatch"):
        forest_module._load_aggs([str(path)])


def test_self_hashed_fabricated_aggregate_metrics_are_rejected(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    aggregate["regret_kga"] = [0.99, 0.0]
    aggregate["gap_vs_adapt"] = {"mean": 0.99, "ci95": [0.99, 0.99]}
    aggregate["gap_vs_freeze"] = {"mean": 0.99, "ci95": [0.99, 0.99]}
    aggregate["gap_vs_better_ci95"] = [0.99, 0.99]
    aggregate["gap_vs_worse_ci95"] = [0.99, 0.99]
    aggregate["FA_u_per_seed"] = [1.0, 1.0]
    aggregate["FA_u_max"] = 1.0
    aggregate["verdict_code"] = "HELDOUT_BEATS_BOTH"
    aggregate["verdict"] = "HELDOUT_BEATS_BOTH"
    _rewrite_aggregate(path, aggregate, manifest_path)

    with pytest.raises(forest_module.LineageError, match="recomputation|per-condition"):
        forest_module._load_aggs([str(path)])


def test_refreshed_hash_cannot_hide_route_that_differs_from_lock(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    per_path = tmp_path / aggregate["files"][0]
    document = json.loads(per_path.read_text())
    document["records"][0].update(
        {
            "b_hat": -0.2,
            "kga_decision": "FREEZE",
            "a_kbound": 0.5,
        }
    )
    _write_json(per_path, document)
    aggregate["file_sha256"][per_path.name] = forest_module._sha256_file(per_path)
    _rewrite_aggregate(path, aggregate, manifest_path)

    with pytest.raises(forest_module.LineageError, match="differs from decision lock"):
        forest_module._load_aggs([str(path)])


def test_coherent_scored_metrics_must_match_hashed_raw_source(tmp_path):
    path, aggregate, manifest_path = _lineage_bundle(tmp_path, scope="heldout")
    per_path = tmp_path / aggregate["files"][0]
    document = json.loads(per_path.read_text())
    document["records"][0].update(
        {
            "a0": 0.4,
            "a_adapted": 0.6,
            "B": 0.2,
            "a_oracle": 0.6,
            "a_kbound": 0.6,
        }
    )
    _write_json(per_path, document)
    aggregate["file_sha256"][per_path.name] = forest_module._sha256_file(per_path)
    _rewrite_aggregate(path, aggregate, manifest_path)

    with pytest.raises(forest_module.LineageError, match="differs from hashed source record"):
        forest_module._load_aggs([str(path)])


def test_strict_loader_rejects_nonstandard_json_constants(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"epsilon": Infinity}\n', encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="non-standard JSON constant"):
        forest_module._strict_json_load(path)


def test_strict_loader_rejects_finite_syntax_that_overflows_to_infinity(tmp_path):
    path = tmp_path / "overflow.json"
    path.write_text('{"nested": {"x": 1e999}}\n', encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="non-finite JSON number"):
        forest_module._strict_json_load(path)


def test_legacy_point_only_aggregate_is_not_silently_reconstructed(tmp_path):
    path = tmp_path / "multiseed_legacy.json"
    _write_json(
        path,
        {
            "dataset": "camelyon17",
            "candidate": "tent",
            "regret_kga": [0.01, 0.0],
            "regret_adapt": [0.02, 0.0],
            "regret_freeze": [0.03, 0.0],
        },
    )
    with pytest.raises(forest_module.LineageError, match="legacy/unknown aggregate schema"):
        forest_module._load_aggs([str(path)])


def test_latex_table_has_seven_columns_and_escapes_candidate(tmp_path):
    path, aggregate, _ = _lineage_bundle(tmp_path, scope="development")
    out = tmp_path / "table.tex"
    forest_module.latex_table(
        [(str(path), aggregate)], out, scope="development-diagnostic"
    )
    text = out.read_text(encoding="utf-8")
    assert r"\begin{tabular}{lcccccl}" in text
    assert r"sar\_online\_aggressive" in text
    header = next(line for line in text.splitlines() if "track (candidate)" in line)
    assert header.count("&") == 6


def test_end_to_end_confirmatory_render_writes_strict_verified_payload(tmp_path):
    path, _, _ = _lineage_bundle(tmp_path, scope="heldout")
    figure = tmp_path / "forest.png"
    table = tmp_path / "forest.tex"
    payload_path = tmp_path / "forest.json"
    assert (
        forest_module.main(
            [
                "--agg",
                str(path),
                "--out-fig",
                str(figure),
                "--out-tex",
                str(table),
                "--out-json",
                str(payload_path),
            ]
        )
        == 0
    )
    payload = json.loads(
        payload_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    assert payload["schema"] == "kbound_natural_multiseed_forest_payload_v2"
    assert payload["scope"] == "heldout-candidate"
    assert payload["lineage_verified"] is False
    assert payload["computational_lineage_verified"] is True
    assert payload["chronology_independently_verified"] is False
    assert payload["generation_committed"] is True
    assert len(payload["generation_id"]) == 64
    assert payload["figure"]["sha256"] == forest_module._sha256_file(figure)
    assert payload["table"]["sha256"] == forest_module._sha256_file(table)
    assert f"% Generation: {payload['generation_id']}" in table.read_text()
    assert forest_module.verify_committed_generation(payload_path) == payload
    assert figure.stat().st_size > 0
    table_text = table.read_text()
    assert r"\begin{tabular}{lcccccl}" in table_text
    rendered_strings = []

    def collect_strings(value, key=None):
        if key in {"source", "figure", "table"}:
            return
        if isinstance(value, str):
            rendered_strings.append(value)
        elif isinstance(value, dict):
            for child_key, child in value.items():
                collect_strings(child, child_key)
        elif isinstance(value, list):
            for child in value:
                collect_strings(child)

    collect_strings(payload)
    public_surface = (table_text + "\n" + "\n".join(rendered_strings)).upper()
    for forbidden in ("BEATS_BOTH", "WIN", "CONFIRMATORY"):
        assert forbidden not in public_surface
    heldout_aggregate = json.loads(path.read_text())
    for field in (
        "external_authenticity_verified",
        "publication_eligible",
        "sources_publication_ready",
        "estimator_publication_eligible",
        "model_seed_ci_eligible",
        "confirmatory_ci_eligible",
        "heldout_promotion_eligible",
        "beats_both_promoted",
    ):
        assert heldout_aggregate[field] is False
    assert not list(tmp_path.glob(".*.json.*"))

    table.write_text(table.read_text() + "% tampered\n", encoding="utf-8")
    with pytest.raises(forest_module.LineageError, match="table hash mismatch"):
        forest_module.verify_committed_generation(payload_path)


@pytest.mark.parametrize(
    "runbook",
    ["run_iwildcam_episodic.sh", "run_iwildcam_locked.sh"],
)
def test_retired_runbooks_fail_before_any_work(runbook, tmp_path):
    path = REPO / "docs/research/kbound/runbooks" / runbook
    result = subprocess.run(
        ["bash", str(path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "RETIRED WORKFLOW" in result.stderr
    assert "intentionally performs no" in result.stderr
    assert list(tmp_path.iterdir()) == []
