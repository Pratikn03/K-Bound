"""Immutable execution-protocol seals and byte-level artifact receipts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .integrity import IntegrityError, file_sha256, require_sha256, stable_sha256
from .label_free_traces import FEATURE_NAMES
from .tent_official import (
    OFFICIAL_TENT_COMMIT,
    TENT_BETAS,
    TENT_EPISODIC,
    TENT_LR,
    TENT_STEPS,
    TENT_WEIGHT_DECAY,
)

PROTOCOL_ID = "KBOUND_CCT20_PROSPECTIVE_CONFIRMATION_v1"
TARGET_SELECTION_SHA256 = "1283205124f086dd4af291c302a7c55bb9dd2c5c164111702dcdfe9f50f2c183"
METADATA_ADDENDUM_SHA256 = "a79fc983573cce4d4fa3e5896130dda7683d2f0c4c852002b3cf802d667fbe08"
LABEL_CONTRACT_ADDENDUM_SHA256 = "295403f261932df1e3118225b30fe813a51247cc56ae4759bba9c19f92aa79c1"
AUTHORITATIVE_PROTOCOL_FILE_SHA256 = "dc6f5da269b7e12523c036030f60b504fa46ca7170f15f9506004aa6e49041a5"
AUTHORITATIVE_PROTOCOL_DOCUMENT_SHA256 = "5ab62a278d750ebb77872894ba150b4319e185ec00367f9dc5ef46c324e0f66c"
EXPECTED_TARGET_IMAGES = 23_275
EXPECTED_CLASS_COUNT = 16
EXPECTED_MODEL_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_LOCATION_COUNT = 9
EXPECTED_DEVELOPMENT_TRACE_COUNT = 55
EXPECTED_TARGET_LOCATIONS = ("0", "7", "28", "40", "46", "78", "100", "105", "130")
REQUIRED_DATA_DEPENDENCY_NAMES = frozenset(
    (
        "target_annotations_json",
        "label_free_target_manifest",
        "official_image_archive",
        "official_annotation_archive",
    )
)
REQUIRED_CODE_DEPENDENCY_NAMES = frozenset(
    (
        "checkpoint_audit",
        "development_gate",
        "development_gate_receipt",
        "development_trace_collection",
        "development_trace_collection_receipt",
        "prospective_protocol",
        "seal_cct20_execution",
        "run_development_gate",
        "run_locked_target",
        "runner_runtime",
        "target_executor",
        "prediction_artifacts",
        "label_free_traces",
        "ridge_gate",
        "tent_official_binding",
        "protocol_seal",
        "integrity",
        "audit_checkpoints",
        "prospective_data",
        "score_once",
        "two_way_inference",
        "train_source",
        "official_tent_py",
        "source_training_seal",
        "source_training_seal_receipt",
        "shared_runtime_identity",
        "shared_runtime_identity_receipt",
        "downstream_execution_runtime_addendum",
        *(f"development_trace_{index:02d}" for index in range(EXPECTED_DEVELOPMENT_TRACE_COUNT)),
        *(f"development_trace_receipt_{index:02d}" for index in range(EXPECTED_DEVELOPMENT_TRACE_COUNT)),
    )
)


def _authoritative_lock_identities() -> list[dict[str, Any]]:
    repository_root = Path(__file__).resolve().parents[3]
    expected = {
        "target_selection": (
            repository_root / "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1.yaml",
            TARGET_SELECTION_SHA256,
        ),
        "metadata_disclosure_addendum": (
            repository_root / "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_ADDENDUM.yaml",
            METADATA_ADDENDUM_SHA256,
        ),
        "label_contract_addendum": (
            repository_root / "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_LABEL_CONTRACT_ADDENDUM.yaml",
            LABEL_CONTRACT_ADDENDUM_SHA256,
        ),
    }
    rows = []
    for name, (path, expected_hash) in expected.items():
        if not path.is_file():
            raise IntegrityError(f"authoritative lock is missing: {path}")
        observed = file_sha256(path)
        if observed != expected_hash:
            raise IntegrityError(f"authoritative lock {name} hash mismatch: {observed} != {expected_hash}")
        rows.append({"name": name, "path": str(path), "sha256": observed})
    return rows


def _authoritative_protocol_identity() -> dict[str, Any]:
    path = Path(__file__).resolve().with_name("prospective_protocol_v1.yaml")
    if not path.is_file():
        raise IntegrityError(f"authoritative prospective protocol is missing: {path}")
    observed = file_sha256(path)
    if observed != AUTHORITATIVE_PROTOCOL_FILE_SHA256:
        raise IntegrityError(
            f"authoritative prospective protocol byte hash mismatch: {observed} != {AUTHORITATIVE_PROTOCOL_FILE_SHA256}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
        "canonical_document_sha256": AUTHORITATIVE_PROTOCOL_DOCUMENT_SHA256,
        "status": "SEALED_BEFORE_SOURCE_TRAINING_AND_TARGET_OUTCOMES",
    }


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrityError(f"refusing to overwrite immutable artifact: {path}") from exc
    finally:
        if descriptor is not None:  # pragma: no cover - exceptional cleanup
            os.close(descriptor)


def write_immutable_json_with_receipt(
    path: str | Path,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable JSON artifact and a separate immutable receipt."""

    destination = Path(path).expanduser().resolve()
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or receipt_path.exists():
        raise IntegrityError(f"refusing to overwrite immutable artifact or receipt for {destination}")
    payload = (
        json.dumps(
            dict(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    _exclusive_write(destination, payload)
    receipt = {
        "schema": "kbound_cct20_artifact_receipt_v1",
        "artifact_path": str(destination),
        "artifact_bytes": len(payload),
        "artifact_sha256": file_sha256(destination),
        "canonical_document_sha256": stable_sha256(dict(document)),
    }
    try:
        _exclusive_write(
            receipt_path,
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n",
        )
    except Exception:
        # Keep the artifact: deleting it after a receipt failure would make an
        # overwrite possible.  The missing receipt is an explicit fail-closed state.
        raise
    return receipt


def verify_artifact_receipt(
    artifact_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_path).expanduser().resolve()
    receipt_file = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else artifact.with_name(artifact.name + ".receipt.json")
    )
    if not artifact.is_file() or not receipt_file.is_file():
        raise IntegrityError(f"artifact/receipt pair is incomplete: {artifact}, {receipt_file}")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        document = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot verify artifact receipt: {exc}") from exc
    if receipt.get("schema") != "kbound_cct20_artifact_receipt_v1":
        raise IntegrityError("unknown artifact receipt schema")
    if receipt.get("artifact_path") != str(artifact):
        raise IntegrityError("receipt artifact_path mismatch")
    if receipt.get("artifact_bytes") != artifact.stat().st_size:
        raise IntegrityError("receipt byte count mismatch")
    if receipt.get("artifact_sha256") != file_sha256(artifact):
        raise IntegrityError("receipt file SHA-256 mismatch")
    if receipt.get("canonical_document_sha256") != stable_sha256(document):
        raise IntegrityError("receipt canonical document SHA-256 mismatch")
    return receipt


def _artifact_identities(paths: Mapping[str, str | Path]) -> list[dict[str, Any]]:
    rows = []
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise IntegrityError(f"sealed dependency {name!r} is missing: {path}")
        rows.append(
            {
                "name": str(name),
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not rows:
        raise IntegrityError("a protocol seal needs at least one named dependency")
    return rows


def _validate_dependency_identities(
    rows: Any,
    *,
    field: str,
    verify_files: bool,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise IntegrityError(f"execution seal {field} must be a non-empty list")
    names: set[str] = set()
    paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise IntegrityError(f"execution seal {field}[{index}] is not a mapping")
        name = str(row.get("name", ""))
        path_value = str(row.get("path", ""))
        byte_count = row.get("bytes")
        digest = require_sha256(row.get("sha256"), field=f"{field}[{index}].sha256")
        if not name or name in names:
            raise IntegrityError(f"execution seal {field} has an empty/duplicate name {name!r}")
        if not path_value or path_value in paths:
            raise IntegrityError(f"execution seal {field} has an empty/duplicate path {path_value!r}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise IntegrityError(f"execution seal {field}[{index}] has an invalid byte count")
        names.add(name)
        paths.add(path_value)
        if verify_files:
            path = Path(path_value)
            if not path.is_file():
                raise IntegrityError(f"sealed dependency is missing at execution time: {path}")
            if path.stat().st_size != byte_count:
                raise IntegrityError(f"sealed dependency byte count changed: {path}")
            if file_sha256(path) != digest:
                raise IntegrityError(f"sealed dependency SHA-256 changed: {path}")
    required_names = (
        REQUIRED_DATA_DEPENDENCY_NAMES
        if field == "dataset_dependencies"
        else REQUIRED_CODE_DEPENDENCY_NAMES
        if field == "code_dependencies"
        else None
    )
    if required_names is not None and names != required_names:
        raise IntegrityError(
            f"execution seal {field} names drift; missing={sorted(required_names - names)}, "
            f"extra={sorted(names - required_names)}"
        )


def _validate_checkpoint_audit(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        audit.get("schema") != "kbound_cct20_independent_checkpoint_audit_v1"
        or audit.get("status") != "PASS"
        or audit.get("required_model_seeds") != list(EXPECTED_MODEL_SEEDS)
        or audit.get("n_checkpoints") != len(EXPECTED_MODEL_SEEDS)
    ):
        raise IntegrityError("checkpoint audit must be the complete five-seed PASS document")
    rows = list(audit.get("checkpoints", ()))
    if not all(isinstance(row, Mapping) for row in rows):
        raise IntegrityError("checkpoint audit rows must be mappings")
    seeds = tuple(row.get("model_seed") for row in rows)
    if seeds != EXPECTED_MODEL_SEEDS:
        raise IntegrityError(f"checkpoint seeds must be {EXPECTED_MODEL_SEEDS}, found {seeds}")
    identity_fields = (
        "file_sha256",
        "tensor_sha256",
        "initial_tensor_sha256",
        "imagenet_backbone_tensor_sha256",
        "config_sha256",
        "config_recipe_sha256",
        "data_sha256",
        "code_sha256",
    )
    for row in rows:
        for field in identity_fields:
            require_sha256(row.get(field), field=f"checkpoint.{field}")
    for field, claim in (
        ("file_sha256", "all_file_hashes_distinct"),
        ("tensor_sha256", "all_tensor_hashes_distinct"),
        ("initial_tensor_sha256", "all_initial_tensor_hashes_distinct"),
        ("config_sha256", "all_config_hashes_distinct"),
    ):
        if audit.get(claim) is not True or len({row[field] for row in rows}) != len(EXPECTED_MODEL_SEEDS):
            raise IntegrityError(f"checkpoint audit does not establish {claim}")
    for field, claim in (
        ("config_recipe_sha256", "shared_config_recipe_sha256"),
        ("imagenet_backbone_tensor_sha256", "shared_imagenet_backbone_tensor_sha256"),
        ("data_sha256", "shared_data_sha256"),
        ("code_sha256", "shared_code_sha256"),
    ):
        values = {row[field] for row in rows}
        if len(values) != 1 or audit.get(claim) != next(iter(values)):
            raise IntegrityError(f"checkpoint audit {claim} does not reconcile to its rows")
    return [dict(row) for row in rows]


def _validate_protocol_config(
    protocol_config: Mapping[str, Any],
    *,
    target_locations: Sequence[str],
) -> None:
    """Require the byte-pinned authoritative protocol document."""

    if protocol_config.get("schema") != "kbound-cct20-prospective-protocol-v1":
        raise IntegrityError("execution seal requires the authoritative protocol schema")
    observed_document_hash = stable_sha256(dict(protocol_config))
    if observed_document_hash != AUTHORITATIVE_PROTOCOL_DOCUMENT_SHA256:
        raise IntegrityError(
            "authoritative protocol document hash mismatch: "
            f"{observed_document_hash} != {AUTHORITATIVE_PROTOCOL_DOCUMENT_SHA256}"
        )

    if protocol_config.get("schema") == "kbound-cct20-prospective-protocol-v1":
        if protocol_config.get("protocol_id") != PROTOCOL_ID:
            raise IntegrityError("authoritative protocol_id drift")
        if protocol_config.get("status") != "SEALED_BEFORE_SOURCE_TRAINING_AND_TARGET_OUTCOMES":
            raise IntegrityError("authoritative protocol is not in its final sealed state")
        dataset = protocol_config.get("dataset", {})
        if dataset.get("output_classes") != 16 or dataset.get("target_split") != "trans_test":
            raise IntegrityError("authoritative dataset task drift")
        if [str(value) for value in dataset.get("target_locations", ())] != list(target_locations):
            raise IntegrityError("authoritative target location order drift")
        partition = protocol_config.get("roles", {}).get("probe_evaluation_partition", {})
        if (
            partition.get("probe_fraction") != 0.30
            or partition.get("salt") != "KBOUND_CCT20_PROBE_EVAL_v1"
            or partition.get("hash_payload") != "UTF8(salt + NUL + str(location) + NUL + seq_id)"
            or partition.get("hash_fraction") != "big_endian_integer_of_full_sha256_digest_divided_by_2^256"
            or partition.get("rule") != "hash_fraction_strictly_below_0.30_is_probe_else_evaluation"
            or partition.get("labels_used") is not False
        ):
            raise IntegrityError("authoritative probe/evaluation partition drift")
        source = protocol_config.get("source_model", {})
        if (
            source.get("implementation") != "torchvision.models.resnet50"
            or source.get("pretrained_weights") != "IMAGENET1K_V2"
            or source.get("seeds") != list(EXPECTED_MODEL_SEEDS)
        ):
            raise IntegrityError("authoritative source-model contract drift")
        adapter = protocol_config.get("adapter", {})
        optimizer = adapter.get("optimizer", {})
        if not (
            adapter.get("name") == "official_tent"
            and adapter.get("commit") == OFFICIAL_TENT_COMMIT
            and optimizer.get("name") == "Adam"
            and optimizer.get("learning_rate") == TENT_LR
            and optimizer.get("betas") == list(TENT_BETAS)
            and optimizer.get("weight_decay") == TENT_WEIGHT_DECAY
            and adapter.get("steps") == TENT_STEPS
            and adapter.get("episodic") is TENT_EPISODIC
            and adapter.get("reset_boundary") == "checkpoint_x_location"
        ):
            raise IntegrityError("authoritative official-Tent contract drift")
        gate = protocol_config.get("gate", {})
        formulas = gate.get("feature_formulas", {})
        if (
            gate.get("features") != list(FEATURE_NAMES)
            or gate.get("benefit_estimator", {}).get("alpha") != 10.0
            or gate.get("support", {}).get("primary") != "finite_values_and_exact_feature_schema"
            or gate.get("calibration", {}).get("exact_rank_n") != 9
            or gate.get("calibration", {}).get("exact_rank_k") != 9
            or formulas.get("logarithm") != "natural"
            or formulas.get("aggregation") != "image_weighted_over_probe_only"
            or formulas.get("entropy_change") != "frozen_mean_entropy - tent_mean_entropy"
            or formulas.get("confidence_change") != "tent_mean_confidence - frozen_mean_confidence"
            or "/ log(2)" not in str(formulas.get("marginal_jensen_shannon_divergence"))
            or "/ 16" not in str(formulas.get("normalized_predicted_class_effective_count"))
            or "max(L2(theta_BN_affine_before_probe), 1e-12)" not in str(formulas.get("normalized_tent_update_norm"))
            or "Channel-weighted mean Gaussian KL"
            not in str(formulas.get("batchnorm_batch_source_statistic_divergence"))
            or formulas.get("numeric_clipping") != "probabilities_only_at_machine_tiny_before_log"
        ):
            raise IntegrityError("authoritative ridge-gate contract drift")
        target = protocol_config.get("target_execution", {})
        if (
            target.get("probe_predictions_scored") is not False
            or target.get("evaluation_predictions_scored") is not True
            or target.get("execution_order")
            != [
                "process every probe sequence in native stream order",
                "compute evidence and seal the checkpoint-by-location action",
                "process every evaluation sequence in native stream order",
            ]
            or target.get("globally_interleaved_probe_and_evaluation_stream") != "forbidden"
        ):
            raise IntegrityError("authoritative target scoring-role contract drift")
        inference = protocol_config.get("inference", {})
        bootstrap = inference.get("bootstrap", {})
        if (
            inference.get("matrix_shape") != [5, 9]
            or bootstrap.get("replicates") != 20_000
            or bootstrap.get("random_seed") != 20_260_828
            or bootstrap.get("simultaneous_bonferroni_ci_level_per_comparison") != 0.975
        ):
            raise IntegrityError("authoritative two-way inference contract drift")
        return


def build_execution_seal(
    *,
    target_location_ids: Sequence[str | int],
    target_manifest_sha256: str,
    dataset_dependencies: Mapping[str, str | Path],
    code_dependencies: Mapping[str, str | Path],
    checkpoint_audit: Mapping[str, Any],
    gate_sha256: str,
    protocol_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build (but do not yet write) the fully frozen pre-inference seal."""

    locations = tuple(str(value) for value in target_location_ids)
    if locations != EXPECTED_TARGET_LOCATIONS:
        raise IntegrityError(
            f"execution seal target locations/order must be exactly {EXPECTED_TARGET_LOCATIONS}, found {locations}"
        )
    checkpoints = _validate_checkpoint_audit(checkpoint_audit)
    manifest_hash = require_sha256(target_manifest_sha256, field="target_manifest_sha256")
    gate_hash = require_sha256(gate_sha256, field="gate_sha256")
    _validate_protocol_config(protocol_config, target_locations=locations)
    document = {
        "schema": "kbound_cct20_execution_seal_v1",
        "status": "SEALED_BEFORE_TARGET_INFERENCE",
        "protocol_id": PROTOCOL_ID,
        "immutable_parent_locks": {
            "target_selection_sha256": TARGET_SELECTION_SHA256,
            "metadata_disclosure_addendum_sha256": METADATA_ADDENDUM_SHA256,
            "label_contract_addendum_sha256": LABEL_CONTRACT_ADDENDUM_SHA256,
        },
        "authoritative_lock_files": _authoritative_lock_identities(),
        "authoritative_protocol_file": _authoritative_protocol_identity(),
        "population": {
            "target_split": "trans_test",
            "expected_images": EXPECTED_TARGET_IMAGES,
            "n_classes": EXPECTED_CLASS_COUNT,
            "target_location_ids": list(locations),
            "target_location_count": EXPECTED_LOCATION_COUNT,
            "target_manifest_sha256": manifest_hash,
        },
        "gate_sha256": gate_hash,
        "checkpoint_audit": dict(checkpoint_audit),
        "checkpoint_audit_sha256": stable_sha256(dict(checkpoint_audit)),
        "checkpoints": checkpoints,
        "dataset_dependencies": _artifact_identities(dataset_dependencies),
        "code_dependencies": _artifact_identities(code_dependencies),
        "protocol_config": dict(protocol_config),
        "protocol_config_sha256": stable_sha256(dict(protocol_config)),
        "firewall": {
            "target_runner_imports_scorer": False,
            "target_label_fields_in_prediction_artifacts": False,
            "predictions_and_actions_sealed_before_scoring": True,
            "all_results_reported_regardless_of_direction": True,
        },
    }
    document["seal_payload_sha256"] = stable_sha256(document)
    validate_execution_seal(document)
    return document


def validate_execution_seal(document: Mapping[str, Any]) -> None:
    if document.get("schema") != "kbound_cct20_execution_seal_v1":
        raise IntegrityError("unknown CCT-20 execution seal schema")
    if document.get("status") != "SEALED_BEFORE_TARGET_INFERENCE":
        raise IntegrityError("execution seal is not in its pre-inference state")
    if document.get("protocol_id") != PROTOCOL_ID:
        raise IntegrityError("execution seal protocol_id drift")
    expected_parent_locks = {
        "target_selection_sha256": TARGET_SELECTION_SHA256,
        "metadata_disclosure_addendum_sha256": METADATA_ADDENDUM_SHA256,
        "label_contract_addendum_sha256": LABEL_CONTRACT_ADDENDUM_SHA256,
    }
    if document.get("immutable_parent_locks") != expected_parent_locks:
        raise IntegrityError("execution seal parent-lock identities drift")
    if document.get("authoritative_lock_files") != _authoritative_lock_identities():
        raise IntegrityError("execution seal authoritative lock-file identities drift")
    if document.get("authoritative_protocol_file") != _authoritative_protocol_identity():
        raise IntegrityError("execution seal authoritative protocol-file identity drift")
    _validate_protocol_config(
        document.get("protocol_config", {}),
        target_locations=[str(value) for value in document.get("population", {}).get("target_location_ids", ())],
    )
    if document.get("protocol_config_sha256") != stable_sha256(document.get("protocol_config", {})):
        raise IntegrityError("execution seal protocol-config hash mismatch")
    population = document.get("population", {})
    if (
        population.get("target_split") != "trans_test"
        or population.get("expected_images") != EXPECTED_TARGET_IMAGES
        or population.get("n_classes") != EXPECTED_CLASS_COUNT
        or population.get("target_location_count") != EXPECTED_LOCATION_COUNT
    ):
        raise IntegrityError("execution seal target population contract drift")
    require_sha256(population.get("target_manifest_sha256"), field="target_manifest_sha256")
    require_sha256(document.get("gate_sha256"), field="gate_sha256")
    locations = population.get("target_location_ids", ())
    if tuple(str(value) for value in locations) != EXPECTED_TARGET_LOCATIONS:
        raise IntegrityError("execution seal target location set/order drift")
    checkpoint_audit = document.get("checkpoint_audit")
    if not isinstance(checkpoint_audit, Mapping):
        raise IntegrityError("execution seal lacks the checkpoint-audit document")
    if document.get("checkpoint_audit_sha256") != stable_sha256(dict(checkpoint_audit)):
        raise IntegrityError("execution seal checkpoint-audit hash mismatch")
    audited_checkpoints = _validate_checkpoint_audit(checkpoint_audit)
    if document.get("checkpoints") != audited_checkpoints:
        raise IntegrityError("execution seal checkpoints differ from the sealed checkpoint audit")
    _validate_dependency_identities(
        document.get("dataset_dependencies"),
        field="dataset_dependencies",
        verify_files=False,
    )
    expected_firewall = {
        "target_runner_imports_scorer": False,
        "target_label_fields_in_prediction_artifacts": False,
        "predictions_and_actions_sealed_before_scoring": True,
        "all_results_reported_regardless_of_direction": True,
    }
    if document.get("firewall") != expected_firewall:
        raise IntegrityError("execution seal target/scoring firewall drift")
    _validate_dependency_identities(
        document.get("code_dependencies"),
        field="code_dependencies",
        verify_files=False,
    )
    claimed = document.get("seal_payload_sha256")
    unsigned = dict(document)
    unsigned.pop("seal_payload_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("execution seal payload hash mismatch")


def verify_execution_environment(document: Mapping[str, Any]) -> None:
    """Revalidate every mutable path immediately before target execution/scoring."""

    validate_execution_seal(document)
    _authoritative_lock_identities()
    _authoritative_protocol_identity()
    _validate_dependency_identities(
        document.get("dataset_dependencies"),
        field="dataset_dependencies",
        verify_files=True,
    )
    _validate_dependency_identities(
        document.get("code_dependencies"),
        field="code_dependencies",
        verify_files=True,
    )
    for row in document["checkpoints"]:
        path_value = row.get("path")
        byte_count = row.get("bytes")
        if not isinstance(path_value, str) or not path_value:
            raise IntegrityError("checkpoint audit lacks an execution-time path")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise IntegrityError("checkpoint audit lacks an execution-time byte count")
        path = Path(path_value)
        if not path.is_file():
            raise IntegrityError(f"sealed checkpoint is missing at execution time: {path}")
        if path.stat().st_size != byte_count:
            raise IntegrityError(f"sealed checkpoint byte count changed: {path}")
        if file_sha256(path) != row["file_sha256"]:
            raise IntegrityError(f"sealed checkpoint SHA-256 changed: {path}")
