"""Shared label-free model and image runtime for sealed CCT-20 execution.

This module contains no development outcome loader and no post-seal scorer.
The held-out target entry point can therefore import it without making target
outcomes reachable in its module graph.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm
import torchvision.transforms as transforms
import yaml
from PIL import Image, UnidentifiedImageError
from PIL import __version__ as pillow_version
from yaml import YAMLError

from .audit_checkpoints import (
    CANONICAL_MODEL_SEEDS,
    CHECKPOINT_SCHEMA,
    checkpoint_identity,
    tensor_state_sha256,
)
from .integrity import IntegrityError, file_sha256, require_sha256, stable_sha256
from .label_free_traces import TARGET_BATCH_SIZE, assert_label_free
from .protocol_seal import verify_artifact_receipt, write_immutable_json_with_receipt
from .tent_official import (
    BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
    BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
    BN_GAUSSIAN_KL_TAYLOR_TERMS,
    BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
    LOCKED_BACKEND_STRATEGY,
    OFFICIAL_TENT_COMMIT,
    OFFICIAL_TENT_FILE_SHA256,
    OFFICIAL_TENT_TREE,
)

EXPECTED_OUTPUTS = 16
EVALUATION_RESIZE = 232
EVALUATION_CROP = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RUNTIME_SCHEMA = "kbound_cct20_shared_runtime_identity_v2"
EXPECTED_INFERENCE_DEVICE = "mps"
EXPECTED_TORCH_INTRAOP_THREADS = 4
EXPECTED_TORCH_INTEROP_THREADS = 10
SOURCE_TRAINING_SEAL_ARTIFACT_SHA256 = "31556ee57d65b0dba192139ab7a1257968b56a1172d2858d0b7cd5de150fa334"
SOURCE_TRAINING_SEAL_DOCUMENT_SHA256 = "6beae941d68ef2d9e346ca74bcf2b3211e9cd74ac28ee193a23d1eac5da58fee"
SUPERSEDED_V1_RUNTIME_SHA256 = "15caaf0b78dab6aa2e700247974f9a76f04c42846e7a69d7511a8d9352834e34"
SUPERSEDED_V1_RUNTIME_ARTIFACT_SHA256 = "5113e6ac4fde63efc1c32ae21748e481066297e15a94fb4576f5d52cd958a28f"
SUPERSEDED_V1_ADDENDUM_SHA256 = "67c28ab6410e8595a13827ccf10906c4b053b2f6fcc56e001e18de3147595ac0"
SUPERSESSION_RECORD_SHA256 = "ab64d1dd4ef6ed8495326427bd5d9c2687d5cec6eee7c474a70cfc26d1a77027"
SUPERSEDED_V1_SCIENTIFIC_ARCHIVE_SHA256 = "ab490a8cce03b75ba45748b0457e398b90be0fb37e2303c54eb38e5970971bd2"
SUPERSEDED_V1_CODE_ARCHIVE_SHA256 = "a75378a2f454be69ca64c0a9f1522596e1ab80a531157da057be8b654a6b7a98"
RUNTIME_ADDENDUM_SCHEMA = "kbound-cct20-execution-runtime-addendum-v2"
RUNTIME_ADDENDUM_STATUS = "SEALED_AFTER_15_DEVELOPMENT_TRACES_BEFORE_GATE_AND_TARGET_INFERENCE"
RUNTIME_ADDENDUM_RELATIVE_PATH = PurePosixPath("research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml")
RUNTIME_ADDENDUM_CODE_PATHS = frozenset(
    {
        "experiments/kbound/cct20/audit_checkpoints.py",
        "experiments/kbound/cct20/integrity.py",
        "experiments/kbound/cct20/label_free_traces.py",
        "experiments/kbound/cct20/prediction_artifacts.py",
        "experiments/kbound/cct20/prospective_data.py",
        "experiments/kbound/cct20/prospective_protocol_v1.yaml",
        "experiments/kbound/cct20/protocol_seal.py",
        "experiments/kbound/cct20/ridge_gate.py",
        "experiments/kbound/cct20/run_development_gate.py",
        "experiments/kbound/cct20/run_locked_target.py",
        "experiments/kbound/cct20/runner_runtime.py",
        "experiments/kbound/cct20/score_once.py",
        "experiments/kbound/cct20/seal_cct20_execution.py",
        "experiments/kbound/cct20/target_executor.py",
        "experiments/kbound/cct20/tent_official.py",
        "experiments/kbound/cct20/train_source.py",
        "experiments/kbound/cct20/two_way_inference.py",
        "external/tent_official/tent.py",
    }
)
SHARED_RUNTIME_DEPENDENCY_NAMES = frozenset(
    (
        "run_development_gate",
        "runner_runtime",
        "tent_official",
        "label_free_traces",
        "ridge_gate",
        "integrity",
        "audit_checkpoints",
        "train_source",
        "prospective_protocol",
        "official_tent_py",
        "downstream_execution_runtime_addendum",
    )
)


def expected_backend_strategy() -> dict[str, Any]:
    """Return the exact reviewed MPS/CPU hybrid execution contract."""

    return {
        "schema": "kbound_cct20_backend_strategy_v2",
        "strategy": LOCKED_BACKEND_STRATEGY,
        "requested_device": "mps",
        "frozen_model_device": "mps",
        "adapted_model_default_device": "mps",
        "fallback_layer": "bn1",
        "fallback_module_class": "KBoundCPUFallbackBatchNorm2d",
        "fallback_compute_device": "cpu",
        "fallback_input_device": "mps",
        "fallback_output_device": "mps",
        "official_tent_parameter_devices_expected": ["cpu", "mps"],
        "sequence_atomic_max_images": TARGET_BATCH_SIZE,
        "bn_gaussian_kl_numeric_implementation": BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
        "bn_gaussian_kl_taylor_threshold": BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
        "bn_gaussian_kl_taylor_terms": BN_GAUSSIAN_KL_TAYLOR_TERMS,
        "bn_gaussian_kl_numeric_clipping": BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
        "reason": "sealed_runtime_mps_root_batchnorm_backward_nonfinite",
    }


def _to_cpu_float64(value: torch.Tensor) -> torch.Tensor:
    """Transfer before widening because MPS cannot materialize float64 tensors."""

    return value.to(device="cpu").to(dtype=torch.float64)


def select_inference_device(requested: str) -> torch.device:
    """Require the explicitly sealed accelerator without automatic selection."""

    if requested != EXPECTED_INFERENCE_DEVICE:
        raise IntegrityError(
            f"CCT-20 execution requires explicit --device {EXPECTED_INFERENCE_DEVICE}; found {requested!r}"
        )
    if not torch.backends.mps.is_available():
        raise IntegrityError("MPS requested but unavailable")
    return torch.device(EXPECTED_INFERENCE_DEVICE)


def configure_deterministic_inference() -> None:
    """Use the same fail-closed deterministic setting as source training."""

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"runtime addendum {field} must be a mapping")
    return value


def validate_runtime_addendum(
    repository_root: str | Path,
    *,
    tent_repo: str | Path,
    runtime_addendum: str | Path,
) -> dict[str, Any]:
    """Replay the amendment's checksum, semantics, anchors, and code ledger."""

    root = Path(repository_root).expanduser().resolve()
    addendum_path = Path(runtime_addendum).expanduser().resolve()
    expected_addendum_path = (root / RUNTIME_ADDENDUM_RELATIVE_PATH).resolve()
    if addendum_path != expected_addendum_path:
        raise IntegrityError("runtime addendum path differs from the locked repository path")
    if not addendum_path.is_file() or addendum_path.stat().st_size < 1:
        raise IntegrityError("runtime addendum is missing or empty")

    addendum_sha256 = file_sha256(addendum_path)
    sidecar_path = addendum_path.with_name(addendum_path.name + ".sha256")
    expected_sidecar = f"{addendum_sha256}  {addendum_path.name}\n"
    try:
        observed_sidecar = sidecar_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise IntegrityError("runtime addendum checksum sidecar is unreadable") from exc
    if observed_sidecar != expected_sidecar:
        raise IntegrityError("runtime addendum checksum sidecar mismatch")

    try:
        document = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, YAMLError) as exc:
        raise IntegrityError("runtime addendum is not valid UTF-8 YAML") from exc
    document = dict(_require_mapping(document, field="document"))
    if document.get("schema") != RUNTIME_ADDENDUM_SCHEMA or document.get("status") != RUNTIME_ADDENDUM_STATUS:
        raise IntegrityError("runtime addendum schema/status mismatch")

    authority = _require_mapping(document.get("authority"), field="authority")
    if (
        authority.get("classification") != "versioned_downstream_execution_protocol_amendment"
        or authority.get("original_protocol_preserved_byte_identically") is not True
        or authority.get("source_training_seal_preserved_byte_identically") is not True
        or authority.get("source_checkpoints_reused_byte_identically") is not True
    ):
        raise IntegrityError("runtime addendum authority contract mismatch")

    anchors = _require_mapping(document.get("immutable_anchors"), field="immutable_anchors")
    anchor_specs = (
        (
            "authoritative_protocol",
            "experiments/kbound/cct20/prospective_protocol_v1.yaml",
            "file_sha256",
        ),
        (
            "source_training_seal",
            "research_lock/KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json",
            "artifact_sha256",
        ),
        (
            "source_training_seal_receipt",
            "research_lock/KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json.receipt.json",
            "file_sha256",
        ),
        ("official_tent", "external/tent_official/tent.py", "file_sha256"),
    )
    for name, expected_relative_path, hash_field in anchor_specs:
        row = _require_mapping(anchors.get(name), field=f"immutable_anchors.{name}")
        if row.get("path") != expected_relative_path:
            raise IntegrityError(f"runtime addendum {name} path mismatch")
        expected_hash = require_sha256(row.get(hash_field), field=f"{name}.{hash_field}")
        if file_sha256(root / expected_relative_path) != expected_hash:
            raise IntegrityError(f"runtime addendum {name} file hash mismatch")
    official_tent = _require_mapping(anchors.get("official_tent"), field="immutable_anchors.official_tent")
    if (
        official_tent.get("commit") != OFFICIAL_TENT_COMMIT
        or official_tent.get("tree") != OFFICIAL_TENT_TREE
        or official_tent.get("file_sha256") != OFFICIAL_TENT_FILE_SHA256
        or (Path(tent_repo).expanduser().resolve() / "tent.py") != (root / "external/tent_official/tent.py").resolve()
    ):
        raise IntegrityError("runtime addendum official Tent identity mismatch")

    amendments = _require_mapping(
        document.get("explicit_protocol_amendments"),
        field="explicit_protocol_amendments",
    )
    batching = _require_mapping(amendments.get("downstream_batching"), field="downstream_batching")
    backend = _require_mapping(amendments.get("execution_backend"), field="execution_backend")
    if (
        batching.get("original_target_images_per_batch") != 128
        or batching.get("amended_sequence_atomic_max_images") != TARGET_BATCH_SIZE
        or batching.get("sequence_atomicity_unchanged") is not True
        or batching.get("cross_location_batches_forbidden") is not True
        or backend.get("strategy") != LOCKED_BACKEND_STRATEGY
        or backend.get("requested_device_argument") != EXPECTED_INFERENCE_DEVICE
        or backend.get("automatic_device_selection") != "forbidden"
        or backend.get("fallback_layer") != "bn1"
        or backend.get("fallback_module_class") != "KBoundCPUFallbackBatchNorm2d"
        or backend.get("fallback_compute_device") != "cpu"
        or backend.get("fallback_parameter_device") != "cpu"
        or backend.get("fallback_input_device") != "mps"
        or backend.get("fallback_output_device") != "mps"
        or backend.get("bn_gaussian_kl_numeric_implementation") != BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION
        or backend.get("bn_gaussian_kl_taylor_threshold") != BN_GAUSSIAN_KL_TAYLOR_THRESHOLD
        or backend.get("bn_gaussian_kl_taylor_terms") != BN_GAUSSIAN_KL_TAYLOR_TERMS
        or backend.get("bn_gaussian_kl_numeric_clipping") != BN_GAUSSIAN_KL_NUMERIC_CLIPPING
    ):
        raise IntegrityError("runtime addendum batch/backend contract mismatch")

    superseded = _require_mapping(
        document.get("superseded_development_attempt"),
        field="superseded_development_attempt",
    )
    if (
        superseded.get("attempt_id") != "development_gate_v1_partial_15_traces"
        or superseded.get("successful_trace_count") != 15
        or superseded.get("failed_cell") != "cis_test:43:checkpoint-0"
        or superseded.get("failed_cell_trace_created") is not False
        or superseded.get("gate_created") is not False
        or superseded.get("development_collection_created") is not False
        or superseded.get("target_runner_invoked") is not False
        or superseded.get("target_action_count") != 0
        or superseded.get("target_prediction_count") != 0
        or superseded.get("target_outcomes_accessed") is not False
        or superseded.get("reuse_in_v2_gate_or_target") != "forbidden"
        or superseded.get("v1_runtime_sha256") != SUPERSEDED_V1_RUNTIME_SHA256
        or superseded.get("v1_runtime_artifact_sha256") != SUPERSEDED_V1_RUNTIME_ARTIFACT_SHA256
    ):
        raise IntegrityError("runtime addendum superseded-attempt boundary mismatch")
    v1_addendum = _require_mapping(
        superseded.get("v1_runtime_addendum"),
        field="superseded_development_attempt.v1_runtime_addendum",
    )
    v1_addendum_path = root / "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v1.yaml"
    if (
        v1_addendum.get("path") != "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v1.yaml"
        or v1_addendum.get("sha256") != SUPERSEDED_V1_ADDENDUM_SHA256
        or file_sha256(v1_addendum_path) != SUPERSEDED_V1_ADDENDUM_SHA256
    ):
        raise IntegrityError("runtime addendum v1 parent identity mismatch")

    record = _require_mapping(
        superseded.get("supersession_record"),
        field="superseded_development_attempt.supersession_record",
    )
    record_path = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        record.get("sha256") != SUPERSESSION_RECORD_SHA256
        or not record_path.is_file()
        or file_sha256(record_path) != SUPERSESSION_RECORD_SHA256
    ):
        raise IntegrityError("runtime addendum supersession record identity mismatch")
    record_receipt = verify_artifact_receipt(record_path)
    if record_receipt.get("artifact_sha256") != SUPERSESSION_RECORD_SHA256:
        raise IntegrityError("runtime addendum supersession receipt mismatch")
    archives = _require_mapping(
        superseded.get("preservation_archives"),
        field="superseded_development_attempt.preservation_archives",
    )
    for name, expected_hash in (
        ("scientific_artifacts", SUPERSEDED_V1_SCIENTIFIC_ARCHIVE_SHA256),
        ("runtime_code_and_locks", SUPERSEDED_V1_CODE_ARCHIVE_SHA256),
    ):
        row = _require_mapping(archives.get(name), field=f"preservation_archives.{name}")
        archive_path = Path(str(row.get("path", ""))).expanduser().resolve()
        if (
            row.get("sha256") != expected_hash
            or not archive_path.is_file()
            or file_sha256(archive_path) != expected_hash
        ):
            raise IntegrityError(f"runtime addendum {name} archive identity mismatch")

    corrections = _require_mapping(
        document.get("implementation_corrections"),
        field="implementation_corrections",
    )
    stable_kl = _require_mapping(
        corrections.get("stable_gaussian_kl"),
        field="implementation_corrections.stable_gaussian_kl",
    )
    if (
        stable_kl.get("numeric_implementation") != BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION
        or stable_kl.get("relative_variance") != "(probe_var-source_var)/(source_var+eps)"
        or stable_kl.get("variance_gap") != "x-log1p(x)"
        or stable_kl.get("taylor_threshold") != BN_GAUSSIAN_KL_TAYLOR_THRESHOLD
        or stable_kl.get("taylor_terms") != BN_GAUSSIAN_KL_TAYLOR_TERMS
        or stable_kl.get("numeric_clipping") != BN_GAUSSIAN_KL_NUMERIC_CLIPPING
        or stable_kl.get("feature_definition_changed") is not False
        or stable_kl.get("regenerate_all_55_development_cells") is not True
    ):
        raise IntegrityError("runtime addendum stable Gaussian-KL contract mismatch")

    locked_runtime = _require_mapping(document.get("locked_runtime"), field="locked_runtime")
    software = _require_mapping(locked_runtime.get("software"), field="locked_runtime.software")
    if software.get("pyyaml") != yaml.__version__:
        raise IntegrityError("runtime addendum PyYAML version mismatch")
    threads = _require_mapping(
        locked_runtime.get("pytorch_threads"),
        field="locked_runtime.pytorch_threads",
    )
    if (
        threads.get("intraop_threads") != EXPECTED_TORCH_INTRAOP_THREADS
        or threads.get("interop_threads") != EXPECTED_TORCH_INTEROP_THREADS
        or threads.get("record_and_replay_required") is not True
    ):
        raise IntegrityError("runtime addendum thread contract mismatch")

    scope = _require_mapping(document.get("execution_scope"), field="execution_scope")
    if (
        scope.get("development_cells") != 55
        or scope.get("target_cells") != 45
        or scope.get("total_cells_under_one_shared_runtime_identity") != 100
        or scope.get("mixed_runtime_artifacts") != "forbidden"
    ):
        raise IntegrityError("runtime addendum execution scope mismatch")
    artifact_binding = _require_mapping(document.get("artifact_binding"), field="artifact_binding")
    dependency_counts = _require_mapping(
        artifact_binding.get("expected_execution_dependency_counts"),
        field="artifact_binding.expected_execution_dependency_counts",
    )
    if (
        artifact_binding.get("addendum_path") != str(RUNTIME_ADDENDUM_RELATIVE_PATH)
        or artifact_binding.get("execution_dependency_name") != "downstream_execution_runtime_addendum"
        or dependency_counts.get("dataset") != 4
        or dependency_counts.get("code_and_artifact") != 138
        or dependency_counts.get("total") != 142
    ):
        raise IntegrityError("runtime addendum artifact-binding contract mismatch")

    code_identities = _require_mapping(document.get("code_identities"), field="code_identities")
    files = _require_mapping(code_identities.get("files"), field="code_identities.files")
    if set(files) != RUNTIME_ADDENDUM_CODE_PATHS:
        raise IntegrityError("runtime addendum code-path ledger mismatch")
    normalized_files: dict[str, str] = {}
    for relative_path in sorted(files):
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts or str(pure_path) != relative_path:
            raise IntegrityError("runtime addendum code path is not canonical and repository-relative")
        expected_hash = require_sha256(files[relative_path], field=f"code_identities.files.{relative_path}")
        observed_path = (root / pure_path).resolve()
        if not observed_path.is_relative_to(root):
            raise IntegrityError("runtime addendum code path escapes the repository")
        if not observed_path.is_file() or file_sha256(observed_path) != expected_hash:
            raise IntegrityError(f"runtime addendum code identity mismatch: {relative_path}")
        normalized_files[relative_path] = expected_hash
    if code_identities.get("hash_algorithm") != "sha256_of_exact_file_bytes" or require_sha256(
        code_identities.get("aggregate_sha256"),
        field="code_identities.aggregate_sha256",
    ) != stable_sha256(normalized_files):
        raise IntegrityError("runtime addendum code aggregate mismatch")
    return document


def shared_runtime_dependency_paths(
    repository_root: str | Path,
    *,
    tent_repo: str | Path,
    runtime_addendum: str | Path,
) -> dict[str, Path]:
    """Return the exact pre-development dependency set for the shared runtime."""

    root = Path(repository_root).expanduser().resolve()
    runtime_addendum_path = Path(runtime_addendum).expanduser().resolve()
    validate_runtime_addendum(
        root,
        tent_repo=tent_repo,
        runtime_addendum=runtime_addendum_path,
    )
    directory = root / "experiments" / "kbound" / "cct20"
    return {
        "run_development_gate": directory / "run_development_gate.py",
        "runner_runtime": directory / "runner_runtime.py",
        "tent_official": directory / "tent_official.py",
        "label_free_traces": directory / "label_free_traces.py",
        "ridge_gate": directory / "ridge_gate.py",
        "integrity": directory / "integrity.py",
        "audit_checkpoints": directory / "audit_checkpoints.py",
        "train_source": directory / "train_source.py",
        "prospective_protocol": directory / "prospective_protocol_v1.yaml",
        "official_tent_py": Path(tent_repo).expanduser().resolve() / "tent.py",
        "downstream_execution_runtime_addendum": runtime_addendum_path,
    }


def _runtime_dependency_identities(
    dependency_paths: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    names = set(dependency_paths)
    if names != SHARED_RUNTIME_DEPENDENCY_NAMES:
        raise IntegrityError(
            "shared runtime dependency names drift; "
            f"missing={sorted(SHARED_RUNTIME_DEPENDENCY_NAMES - names)}, "
            f"extra={sorted(names - SHARED_RUNTIME_DEPENDENCY_NAMES)}"
        )
    identities = []
    paths: set[str] = set()
    for name in sorted(SHARED_RUNTIME_DEPENDENCY_NAMES):
        path = Path(dependency_paths[name]).expanduser().resolve()
        path_value = str(path)
        if path_value in paths:
            raise IntegrityError(f"shared runtime dependency path is duplicated: {path}")
        if not path.is_file():
            raise IntegrityError(f"shared runtime dependency is missing: {path}")
        byte_count = path.stat().st_size
        if byte_count < 1:
            raise IntegrityError(f"shared runtime dependency is empty: {path}")
        identities.append(
            {
                "name": name,
                "path": path_value,
                "bytes": byte_count,
                "sha256": file_sha256(path),
            }
        )
        paths.add(path_value)
    return identities


def runtime_dependency_paths_from_identity(
    document: Mapping[str, Any],
) -> dict[str, Path]:
    """Extract a strict dependency-path map from a shared-runtime document."""

    rows = document.get("dependencies")
    if not isinstance(rows, list) or len(rows) != len(SHARED_RUNTIME_DEPENDENCY_NAMES):
        raise IntegrityError("shared runtime identity has an incomplete dependency ledger")
    result: dict[str, Path] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"name", "path", "bytes", "sha256"}:
            raise IntegrityError(f"shared runtime dependency {index} schema drift")
        name = str(row.get("name", ""))
        if name not in SHARED_RUNTIME_DEPENDENCY_NAMES or name in result:
            raise IntegrityError("shared runtime dependency name is unknown or duplicated")
        path_value = str(row.get("path", ""))
        if not path_value or Path(path_value).expanduser().resolve() != Path(path_value):
            raise IntegrityError("shared runtime dependency path is not absolute/canonical")
        if isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int):
            raise IntegrityError("shared runtime dependency byte count is invalid")
        if int(row["bytes"]) < 1:
            raise IntegrityError("shared runtime dependency byte count is invalid")
        require_sha256(row.get("sha256"), field=f"dependencies[{index}].sha256")
        result[name] = Path(path_value)
    if [row["name"] for row in rows] != sorted(SHARED_RUNTIME_DEPENDENCY_NAMES):
        raise IntegrityError("shared runtime dependency order is not canonical")
    return result


def _validate_backend_runtime(device: torch.device) -> None:
    if device.type != EXPECTED_INFERENCE_DEVICE or str(device) != EXPECTED_INFERENCE_DEVICE:
        raise IntegrityError(f"CCT-20 shared runtime requires {EXPECTED_INFERENCE_DEVICE}, found {device}")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise IntegrityError("CCT-20 shared runtime requires an available MPS backend")
    intraop = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    if intraop != EXPECTED_TORCH_INTRAOP_THREADS or interop != EXPECTED_TORCH_INTEROP_THREADS:
        raise IntegrityError(
            "CCT-20 shared runtime requires fresh-process torch thread counts "
            f"intra-op={EXPECTED_TORCH_INTRAOP_THREADS}, "
            f"inter-op={EXPECTED_TORCH_INTEROP_THREADS}; found {intraop}/{interop}"
        )


def build_shared_runtime_identity(
    device: torch.device,
    *,
    source_training_seal_artifact_sha256: str,
    source_training_seal_document_sha256: str,
    dependency_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Capture one runtime before development traces and reuse it for target work."""

    _validate_backend_runtime(device)
    artifact_hash = require_sha256(
        source_training_seal_artifact_sha256,
        field="source_training_seal_artifact_sha256",
    )
    document_hash = require_sha256(
        source_training_seal_document_sha256,
        field="source_training_seal_document_sha256",
    )
    core = {
        "schema": RUNTIME_SCHEMA,
        "status": "SEALED_BEFORE_DEVELOPMENT_TRACES_AND_TARGET_INFERENCE",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "inference_device_type": device.type,
        "inference_device": str(device),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "backend_strategy": expected_backend_strategy(),
        "source_training_seal_artifact_sha256": artifact_hash,
        "source_training_seal_document_sha256": document_hash,
        "dependencies": _runtime_dependency_identities(dependency_paths),
    }
    if core["deterministic_algorithms_enabled"] is not True:
        raise IntegrityError("shared runtime identity requires deterministic algorithms")
    core["runtime_sha256"] = stable_sha256(core)
    return core


def validate_source_training_seal_identity(
    document: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    """Require the fixed source-training parent before deriving shared provenance."""

    if (
        document.get("schema") != "kbound_cct20_source_training_seal_v1"
        or document.get("status") != "SEALED_BEFORE_SOURCE_TRAINING_AND_TARGET_OUTCOMES"
        or receipt.get("artifact_sha256") != SOURCE_TRAINING_SEAL_ARTIFACT_SHA256
        or receipt.get("canonical_document_sha256") != SOURCE_TRAINING_SEAL_DOCUMENT_SHA256
    ):
        raise IntegrityError("immutable source-training seal identity/status mismatch")


def validate_shared_runtime_identity_artifact(document: Mapping[str, Any]) -> None:
    """Validate the sealed document without asserting the current live process."""

    expected_fields = {
        "schema",
        "status",
        "python",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "pyyaml",
        "platform",
        "mps_available",
        "mps_built",
        "inference_device_type",
        "inference_device",
        "deterministic_algorithms_enabled",
        "torch_intraop_threads",
        "torch_interop_threads",
        "backend_strategy",
        "source_training_seal_artifact_sha256",
        "source_training_seal_document_sha256",
        "dependencies",
        "runtime_sha256",
    }
    if set(document) != expected_fields:
        raise IntegrityError("shared runtime identity field schema drift")
    if (
        document.get("schema") != RUNTIME_SCHEMA
        or document.get("status") != "SEALED_BEFORE_DEVELOPMENT_TRACES_AND_TARGET_INFERENCE"
    ):
        raise IntegrityError("unknown or unsealed CCT-20 shared runtime identity")
    unsigned = dict(document)
    claimed = unsigned.pop("runtime_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("shared runtime identity SHA-256 mismatch")
    for field in ("python", "torch", "torchvision", "numpy", "pillow", "pyyaml", "platform"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise IntegrityError(f"shared runtime identity lacks {field}")
    if (
        document.get("mps_available") is not True
        or document.get("mps_built") is not True
        or document.get("inference_device_type") != EXPECTED_INFERENCE_DEVICE
        or document.get("inference_device") != EXPECTED_INFERENCE_DEVICE
        or document.get("deterministic_algorithms_enabled") is not True
        or document.get("torch_intraop_threads") != EXPECTED_TORCH_INTRAOP_THREADS
        or document.get("torch_interop_threads") != EXPECTED_TORCH_INTEROP_THREADS
        or document.get("backend_strategy") != expected_backend_strategy()
    ):
        raise IntegrityError("shared runtime backend/determinism/thread contract drift")
    require_sha256(
        document.get("source_training_seal_artifact_sha256"),
        field="source_training_seal_artifact_sha256",
    )
    require_sha256(
        document.get("source_training_seal_document_sha256"),
        field="source_training_seal_document_sha256",
    )
    runtime_dependency_paths_from_identity(document)


def validate_shared_runtime_identity(
    document: Mapping[str, Any],
    *,
    device: torch.device,
    dependency_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Require the live process and dependencies to equal the pre-development seal."""

    validate_shared_runtime_identity_artifact(document)
    sealed_dependency_paths = runtime_dependency_paths_from_identity(document)
    expected_dependency_paths = sealed_dependency_paths if dependency_paths is None else dependency_paths
    replay = build_shared_runtime_identity(
        device,
        source_training_seal_artifact_sha256=str(document.get("source_training_seal_artifact_sha256", "")),
        source_training_seal_document_sha256=str(document.get("source_training_seal_document_sha256", "")),
        dependency_paths=expected_dependency_paths,
    )
    if dict(document) != replay:
        changed = sorted(key for key in set(document) | set(replay) if document.get(key) != replay.get(key))
        raise IntegrityError(f"live shared runtime differs from its seal: {changed}")


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(EVALUATION_RESIZE, antialias=True),
            transforms.CenterCrop(EVALUATION_CROP),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"cannot load CCT-20 checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise IntegrityError(f"checkpoint has the wrong schema: {path}")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise IntegrityError(f"checkpoint lacks model_state: {path}")
    claimed = require_sha256(payload.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256")
    observed = tensor_state_sha256(state)
    if observed != claimed:
        raise IntegrityError(f"checkpoint tensor identity mismatch: {path}")
    if payload.get("architecture") != "resnet50" or payload.get("num_classes") != 16:
        raise IntegrityError("checkpoint architecture/output count differs from the seal")
    return payload


def verify_checkpoint_audit_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Replay the independent-checkpoint audit against files currently on disk."""

    if (
        document.get("schema") != "kbound_cct20_independent_checkpoint_audit_v1"
        or document.get("status") != "PASS"
        or document.get("required_model_seeds") != list(CANONICAL_MODEL_SEEDS)
        or document.get("n_checkpoints") != len(CANONICAL_MODEL_SEEDS)
    ):
        raise IntegrityError("checkpoint audit is not the sealed five-seed PASS document")
    raw_rows = document.get("checkpoints")
    if not isinstance(raw_rows, list) or not all(isinstance(row, Mapping) for row in raw_rows):
        raise IntegrityError("checkpoint audit rows must be a list of mappings")
    rows = [dict(row) for row in raw_rows]
    if [row.get("model_seed") for row in rows] != list(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("checkpoint audit rows are not ordered seeds 0..4")
    replayed = []
    for row in rows:
        identity = checkpoint_identity(row.get("path", ""))
        if row != identity:
            changed = sorted(key for key in set(row) | set(identity) if row.get(key) != identity.get(key))
            raise IntegrityError(
                "checkpoint audit row differs from its replayed checkpoint identity "
                f"for seed {row.get('model_seed')}: {changed}"
            )
        replayed.append(identity)
    for field, claim in (
        ("file_sha256", "all_file_hashes_distinct"),
        ("tensor_sha256", "all_tensor_hashes_distinct"),
        ("initial_tensor_sha256", "all_initial_tensor_hashes_distinct"),
        ("config_sha256", "all_config_hashes_distinct"),
    ):
        if document.get(claim) is not True or len({row[field] for row in replayed}) != 5:
            raise IntegrityError(f"checkpoint audit does not establish {claim}")
    for field, claim in (
        ("config_recipe_sha256", "shared_config_recipe_sha256"),
        ("imagenet_backbone_tensor_sha256", "shared_imagenet_backbone_tensor_sha256"),
        ("data_sha256", "shared_data_sha256"),
        ("code_sha256", "shared_code_sha256"),
    ):
        values = {row[field] for row in replayed}
        if len(values) != 1 or document.get(claim) != next(iter(values)):
            raise IntegrityError(f"checkpoint audit {claim} does not reconcile to its rows")
    return rows


def load_checkpoint_model_pair(
    checkpoint_row: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """Load two independent model objects from one hash-verified checkpoint."""

    path = Path(str(checkpoint_row.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise IntegrityError(f"sealed checkpoint is missing: {path}")
    expected_bytes = checkpoint_row.get("bytes")
    if expected_bytes != path.stat().st_size:
        raise IntegrityError(f"sealed checkpoint byte count changed: {path}")
    expected_file_hash = require_sha256(checkpoint_row.get("file_sha256"), field="checkpoint.file_sha256")
    if file_sha256(path) != expected_file_hash:
        raise IntegrityError(f"sealed checkpoint file hash changed: {path}")
    payload = _load_checkpoint_payload(path)
    if payload.get("model_seed") != checkpoint_row.get("model_seed"):
        raise IntegrityError("checkpoint seed differs from its audit row")
    if payload.get("checkpoint_tensor_sha256") != checkpoint_row.get("tensor_sha256"):
        raise IntegrityError("checkpoint tensor differs from its audit row")
    state = payload["model_state"]

    def instantiate() -> nn.Module:
        model = tvm.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, EXPECTED_OUTPUTS)
        try:
            result = model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise IntegrityError(f"checkpoint state does not load strictly: {exc}") from exc
        if result.missing_keys or result.unexpected_keys:  # pragma: no cover - strict guard
            raise IntegrityError("checkpoint state has missing or unexpected tensors")
        return model.to(device)

    frozen = instantiate()
    adapted_source = instantiate()
    frozen.eval()
    adapted_source.eval()
    return frozen, adapted_source


def _relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError("image file_name must be a non-empty relative path")
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise IntegrityError(f"unsafe image path {value!r}")
    return Path(*posix.parts)


class VerifiedImageStore:
    """Resolve exact manifest members and verify bytes before every forward."""

    def __init__(
        self,
        image_root: str | Path,
        expected_samples: Iterable[Mapping[str, Any]],
    ) -> None:
        self.root = Path(image_root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise IntegrityError(f"image root is not a directory: {self.root}")
        self._expected: dict[str, dict[str, Any]] = {}
        for position, raw in enumerate(expected_samples):
            row = dict(raw)
            image_id = str(row.get("image_id", row.get("id", "")))
            file_name = str(row.get("file_name", ""))
            byte_count = row.get("image_bytes")
            digest = require_sha256(row.get("image_sha256"), field=f"expected_samples[{position}].image_sha256")
            if not image_id or image_id in self._expected:
                raise IntegrityError("image store has an empty or duplicate image id")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
                raise IntegrityError("image store requires a positive image byte count")
            _relative_path(file_name)
            self._expected[image_id] = {
                "file_name": file_name,
                "image_bytes": byte_count,
                "image_sha256": digest,
            }
        if not self._expected:
            raise IntegrityError("image store cannot be empty")
        self.transform = evaluation_transform()

    @property
    def population_sha256(self) -> str:
        return stable_sha256([{"image_id": key, **self._expected[key]} for key in sorted(self._expected)])

    def tensor_batch(self, rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        assert_label_free(rows, path="image_batch_metadata")
        if not rows:
            raise IntegrityError("cannot load an empty image batch")
        tensors = []
        for row in rows:
            image_id = str(row.get("image_id", row.get("id", "")))
            expected = self._expected.get(image_id)
            if expected is None:
                raise IntegrityError(f"image {image_id!r} is outside the sealed population")
            file_name = str(row.get("file_name", ""))
            if file_name != expected["file_name"]:
                raise IntegrityError(f"image {image_id!r} path differs from the sealed manifest")
            relative = _relative_path(file_name)
            try:
                path = (self.root / relative).resolve(strict=True)
            except FileNotFoundError as exc:
                raise IntegrityError(f"sealed image is missing: {file_name}") from exc
            if self.root != path and self.root not in path.parents:
                raise IntegrityError(f"sealed image escapes image root: {file_name}")
            if not path.is_file():
                raise IntegrityError(f"sealed image is not a regular file: {path}")
            try:
                payload = path.read_bytes()
            except OSError as exc:
                raise IntegrityError(f"cannot read sealed image {path}: {exc}") from exc
            if len(payload) != expected["image_bytes"]:
                raise IntegrityError(f"sealed image byte count changed: {file_name}")
            if hashlib.sha256(payload).hexdigest() != expected["image_sha256"]:
                raise IntegrityError(f"sealed image SHA-256 changed: {file_name}")
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    tensors.append(self.transform(image.convert("RGB")))
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                raise IntegrityError(f"sealed image no longer decodes: {file_name}") from exc
        return torch.stack(tensors, dim=0)


def paired_forward(
    frozen_model: nn.Module,
    adapted_model: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen and official-adapter pre-update logits for one batch."""

    value = images.to(device)
    with torch.no_grad():
        frozen = frozen_model(value)
    with torch.enable_grad():
        adapted = adapted_model(value)
    if frozen.shape != adapted.shape or tuple(frozen.shape[1:]) != (EXPECTED_OUTPUTS,):
        raise IntegrityError(f"paired logits must have shape (n, 16), found {frozen.shape}, {adapted.shape}")
    if not torch.isfinite(frozen).all() or not torch.isfinite(adapted).all():
        raise IntegrityError("paired inference produced NaN or Infinity")
    return (
        _to_cpu_float64(frozen.detach()).numpy(),
        _to_cpu_float64(adapted.detach()).numpy(),
    )


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON artifact is not an object: {source}")
    return value


def load_sealed_json_object(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = verify_artifact_receipt(path)
    return load_json_object(path), receipt


def write_or_verify_immutable_json(
    path: str | Path,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Create once, or verify a byte-semantically identical completed artifact."""

    destination = Path(path).expanduser().resolve()
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    exists = (destination.exists(), receipt_path.exists())
    if exists == (False, False):
        return write_immutable_json_with_receipt(destination, document)
    if exists != (True, True):
        raise IntegrityError(f"incomplete immutable artifact/receipt pair: {destination}")
    observed, receipt = load_sealed_json_object(destination)
    if observed != dict(document):
        raise IntegrityError(f"existing immutable artifact differs from replay: {destination}")
    return receipt
