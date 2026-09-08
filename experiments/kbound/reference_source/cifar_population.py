"""Fail-closed CIFAR-10-C/v6 CIFAR-10.1 inventory and base-ID partitions.

This module prepares inputs only.  It neither constructs a model nor reads or
produces target predictions, adaptation benefits, KGA decisions, or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STANDARD_TEST_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
VALIDATION_CORRUPTIONS = (
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
)
SEVERITIES = (1, 2, 3, 4, 5)
PARTITION_PROPORTIONS = {"fit": 0.6, "cal": 0.2, "check": 0.2}
PARTITION_ROLES = tuple(PARTITION_PROPORTIONS)
PARTITION_HASH_PREFIX = "kbound-cifar-base-partition-v1"


class CifarPopulationError(RuntimeError):
    """A CIFAR input or identity contract failed closed."""


@dataclass(frozen=True)
class Cifar10CScope:
    standard_corruptions: tuple[str, ...]
    validation_corruptions: tuple[str, ...]
    severities: tuple[int, ...]
    base_count: int
    full: bool

    @property
    def corruptions(self) -> tuple[str, ...]:
        return self.standard_corruptions + self.validation_corruptions


FULL_CIFAR10C_SCOPE = Cifar10CScope(
    standard_corruptions=STANDARD_TEST_CORRUPTIONS,
    validation_corruptions=VALIDATION_CORRUPTIONS,
    severities=SEVERITIES,
    base_count=10_000,
    full=True,
)
FULL_CIFAR101_COUNT = 2_000


@dataclass(frozen=True)
class CifarPreparationResult:
    data_acceptance_path: Path
    partition_receipt_path: Path
    eligible_for_full_campaign: bool


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_directory(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} root is missing: {resolved}")
    if resolved.is_symlink():
        raise CifarPopulationError(f"{label} root must not be a symlink: {resolved}")
    return resolved.resolve()


def _required_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"required CIFAR input is missing: {path}")
    if path.is_symlink():
        raise CifarPopulationError(f"required CIFAR input must not be a symlink: {path}")
    return path.resolve()


def _file_keys(scope: Cifar10CScope) -> tuple[str, ...]:
    return (
        *(f"cifar10c/{name}.npy" for name in scope.corruptions),
        "cifar10c/labels.npy",
        "cifar101/cifar10.1_v6_data.npy",
        "cifar101/cifar10.1_v6_labels.npy",
    )


def _validate_scope(scope: Cifar10CScope, cifar101_count: int) -> None:
    if not isinstance(scope, Cifar10CScope):
        raise CifarPopulationError("CIFAR-10-C scope must be a Cifar10CScope")
    if type(scope.base_count) is not int or scope.base_count <= 0:
        raise CifarPopulationError("CIFAR-10-C base count must be positive")
    if scope.base_count % 5 or type(cifar101_count) is not int or cifar101_count <= 0 or cifar101_count % 5:
        raise CifarPopulationError("CIFAR base population counts must be positive and divisible by 5")
    if not scope.corruptions or len(scope.corruptions) != len(set(scope.corruptions)):
        raise CifarPopulationError("CIFAR-10-C corruption names must be nonempty and unique")
    if any(not isinstance(name, str) or not name for name in scope.corruptions):
        raise CifarPopulationError("CIFAR-10-C corruption names must be nonempty strings")
    if not scope.severities or len(scope.severities) != len(set(scope.severities)):
        raise CifarPopulationError("CIFAR-10-C severities must be nonempty and unique")
    if any(type(severity) is not int or severity <= 0 for severity in scope.severities):
        raise CifarPopulationError("CIFAR-10-C severities must be positive integers")


def _validate_expected_hashes(document: Mapping[str, Any], scope: Cifar10CScope) -> dict[str, str]:
    if document.get("schema") != "kbound_cifar_expected_hashes_v1":
        raise CifarPopulationError("expected-hash manifest has an unsupported schema")
    files = document.get("files")
    if not isinstance(files, Mapping):
        raise CifarPopulationError("expected-hash manifest files must be an object")
    required = set(_file_keys(scope))
    observed = set(files)
    if observed != required:
        missing = sorted(required - observed)
        unexpected = sorted(observed - required)
        scope_label = "full CIFAR" if scope.full else "configured CIFAR"
        raise CifarPopulationError(
            f"{scope_label} expected-hash manifest mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    validated: dict[str, str] = {}
    for name in sorted(required):
        digest = files[name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise CifarPopulationError(f"invalid expected SHA-256 for {name}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise CifarPopulationError(f"non-hexadecimal expected SHA-256 for {name}") from exc
        validated[name] = digest.lower()
    return validated


def _verify_hash(path: Path, key: str, expected: Mapping[str, str]) -> str:
    actual = _streaming_sha256(path)
    if actual != expected[key]:
        raise CifarPopulationError(f"SHA-256 mismatch for {key}: expected {expected[key]}, observed {actual}")
    return actual


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CifarPopulationError(f"cannot safely load NumPy input {path}: {exc}") from exc


def _validate_image_array(array: np.ndarray, expected_count: int, label: str) -> None:
    expected_shape = (expected_count, 32, 32, 3)
    if tuple(array.shape) != expected_shape:
        raise CifarPopulationError(
            f"{label} shape {tuple(array.shape)} does not provide expected severity blocks/population {expected_shape}"
        )
    if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
        raise CifarPopulationError(f"{label} contains non-finite pixels")
    if array.dtype != np.dtype("uint8"):
        raise CifarPopulationError(f"{label} dtype must be uint8, observed {array.dtype}")


def _label_acceptance(
    path: Path,
    digest: str,
    expected_count: int,
    label: str,
    *,
    repeated_blocks: int | None = None,
    require_all_classes: bool,
) -> dict[str, Any]:
    array = _load_npy(path)
    if tuple(array.shape) != (expected_count,):
        raise CifarPopulationError(
            f"{label} label population shape must be {(expected_count,)}, observed {tuple(array.shape)}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
            raise CifarPopulationError(f"{label} labels contain non-finite values")
        raise CifarPopulationError(f"{label} label dtype must be integral, observed {array.dtype}")
    class_min = int(array.min())
    class_max = int(array.max())
    if class_min < 0 or class_max > 9:
        raise CifarPopulationError(f"{label} label values must lie in [0,9], observed [{class_min},{class_max}]")
    if repeated_blocks is not None:
        block_size = expected_count // repeated_blocks
        reference = array[:block_size]
        for block in range(1, repeated_blocks):
            if not bool(np.array_equal(reference, array[block * block_size : (block + 1) * block_size])):
                raise CifarPopulationError(f"{label} labels differ across severity blocks")
    if require_all_classes and {int(value) for value in np.unique(array)} != set(range(10)):
        raise CifarPopulationError(f"{label} label population does not cover all 10 classes")
    return {
        "path": str(path),
        "sha256": digest,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "class_min": class_min,
        "class_max": class_max,
        "values_exported": False,
    }


def _base_identity_sha256(dataset: str, count: int) -> str:
    digest = hashlib.sha256()
    for base_id in range(count):
        digest.update(f"{dataset}\0{base_id}\n".encode())
    return digest.hexdigest()


def _expanded_cifar10c_sha256(scope: Cifar10CScope) -> str:
    digest = hashlib.sha256()
    for corruption in scope.corruptions:
        for severity_index, severity in enumerate(scope.severities):
            for base_id in range(scope.base_count):
                row_index = severity_index * scope.base_count + base_id
                digest.update(f"{corruption}\0{severity}\0{base_id}\0{row_index}\n".encode())
    return digest.hexdigest()


def partition_base_ids(
    dataset: str, base_ids: Sequence[int], *, seed: int, expected_count: int
) -> dict[str, list[int]]:
    """Assign complete base identities to exact 60/20/20 roles."""

    if type(seed) is not int:
        raise CifarPopulationError("partition seed must be an integer")
    if type(expected_count) is not int or expected_count <= 0:
        raise CifarPopulationError("expected base population count must be positive")
    if expected_count % 5:
        raise CifarPopulationError("exact 60/20/20 partition requires a population divisible by 5")
    if not isinstance(dataset, str) or not dataset:
        raise CifarPopulationError("partition dataset identity must be a nonempty string")
    ids = list(base_ids)
    if any(type(value) is not int for value in ids):
        raise CifarPopulationError("base IDs must be integers")
    if len(ids) != len(set(ids)):
        raise CifarPopulationError("duplicate base IDs are forbidden")
    if set(ids) != set(range(expected_count)):
        raise CifarPopulationError(f"base IDs must exactly cover 0..{expected_count - 1} with no missing identity")

    def order_key(base_id: int) -> tuple[bytes, int]:
        payload = f"{PARTITION_HASH_PREFIX}\0{seed}\0{dataset}\0{base_id}".encode()
        return hashlib.sha256(payload).digest(), base_id

    ordered = sorted(ids, key=order_key)
    fit_end = expected_count * 3 // 5
    cal_end = fit_end + expected_count // 5
    roles = {
        "fit": sorted(ordered[:fit_end]),
        "cal": sorted(ordered[fit_end:cal_end]),
        "check": sorted(ordered[cal_end:]),
    }
    assigned = roles["fit"] + roles["cal"] + roles["check"]
    if len(assigned) != expected_count or len(set(assigned)) != expected_count:
        raise CifarPopulationError("internal partition coverage/disjointness failure")
    return roles


def validate_distinct_windows(
    adaptation_ids: Sequence[int], evaluation_ids: Sequence[int], role_membership: Sequence[int]
) -> dict[str, int | bool]:
    """Validate one downstream role's adaptation/evaluation window identities."""

    adaptation = list(adaptation_ids)
    evaluation = list(evaluation_ids)
    role = list(role_membership)
    if not adaptation or not evaluation:
        raise CifarPopulationError("adaptation and evaluation windows must both be nonempty")
    if any(type(value) is not int or value < 0 for values in (adaptation, evaluation, role) for value in values):
        raise CifarPopulationError("window and role IDs must be nonnegative type-exact integers")
    if len(role) != len(set(role)):
        raise CifarPopulationError("duplicate identity in role membership")
    membership = set(role)
    if len(adaptation) != len(set(adaptation)) or len(evaluation) != len(set(evaluation)):
        raise CifarPopulationError("duplicate identity within an adaptation/evaluation window")
    if set(adaptation) & set(evaluation):
        raise CifarPopulationError("adaptation and evaluation windows overlap")
    if not set(adaptation).issubset(membership) or not set(evaluation).issubset(membership):
        raise CifarPopulationError("adaptation/evaluation window contains an identity outside its role")
    return {
        "adaptation_count": len(adaptation),
        "evaluation_count": len(evaluation),
        "disjoint": True,
    }


def iter_cifar10c_role_members(partition_receipt: Mapping[str, Any], role: str) -> Iterator[dict[str, Any]]:
    """Expand a receipt role without changing its cross-corruption assignment."""

    if role not in PARTITION_ROLES:
        raise CifarPopulationError(f"unknown partition role: {role}")
    if partition_receipt.get("schema") != "kbound_cifar_kga_partition_v1":
        raise CifarPopulationError("invalid CIFAR partition receipt schema")
    all_membership = partition_receipt.get("membership", {}).get("cifar10c")
    membership = all_membership.get(role) if isinstance(all_membership, Mapping) else None
    scope = partition_receipt.get("cifar10c_scope")
    if not isinstance(membership, list) or not isinstance(scope, Mapping):
        raise CifarPopulationError("partition receipt is missing CIFAR-10-C membership/scope")
    base_count = scope.get("base_count")
    corruptions = scope.get("corruptions")
    severities = scope.get("severities")
    if type(base_count) is not int or not isinstance(corruptions, list) or not isinstance(severities, list):
        raise CifarPopulationError("partition receipt has malformed CIFAR-10-C scope")
    standard, validation = scope.get("standard_test_corruptions"), scope.get("validation_corruptions")
    if (not isinstance(standard, list) or not isinstance(validation, list)
            or standard + validation != corruptions or not corruptions or not severities
            or any(not isinstance(c, str) for c in corruptions)
            or len(corruptions) != len(set(corruptions))
            or any(type(s) is not int or s not in SEVERITIES for s in severities)
            or len(severities) != len(set(severities))
            or not set(standard).issubset(STANDARD_TEST_CORRUPTIONS)
            or not set(validation).issubset(VALIDATION_CORRUPTIONS)):
        raise CifarPopulationError("partition receipt has inconsistent CIFAR-10-C scope")
    if partition_receipt.get("eligible_for_full_campaign") is True and (
            standard != list(STANDARD_TEST_CORRUPTIONS) or validation != list(VALIDATION_CORRUPTIONS)
            or severities != list(SEVERITIES) or base_count != FULL_CIFAR10C_SCOPE.base_count):
        raise CifarPopulationError("full receipt has incomplete CIFAR-10-C scope")
    if set(all_membership) != set(PARTITION_ROLES) or any(
        not isinstance(all_membership[name], list) for name in PARTITION_ROLES
    ):
        raise CifarPopulationError("partition receipt has malformed CIFAR-10-C role membership")
    flattened = [base_id for name in PARTITION_ROLES for base_id in all_membership[name]]
    expected_membership = partition_base_ids(
        "cifar10c",
        flattened,
        seed=partition_receipt.get("partition_seed"),
        expected_count=base_count,
    )
    if dict(all_membership) != expected_membership:
        raise CifarPopulationError("partition receipt base-ID assignment mismatch")
    for corruption in corruptions:
        for severity_index, severity in enumerate(severities):
            for base_id in membership:
                yield {
                    "corruption": corruption,
                    "severity": severity,
                    "base_id": base_id,
                    "row_index": severity_index * base_count + base_id,
                    "role": role,
                }


def _inspect_inputs(
    cifar10c_root: Path,
    cifar101_root: Path,
    scope: Cifar10CScope,
    cifar101_count: int,
    expected: Mapping[str, str],
    eligible_for_full_campaign: bool,
) -> dict[str, Any]:
    expected_c10c_names = {f"{name}.npy" for name in scope.corruptions} | {"labels.npy"}
    observed_c10c_names = {
        path.name for path in cifar10c_root.glob("*.npy") if path.is_file() and not path.name.startswith("._")
    }
    if observed_c10c_names != expected_c10c_names:
        missing = sorted(expected_c10c_names - observed_c10c_names)
        unexpected = sorted(observed_c10c_names - expected_c10c_names)
        message = f"CIFAR-10-C declared array set mismatch: missing={missing}, unexpected={unexpected}"
        if missing:
            raise FileNotFoundError(message)
        raise CifarPopulationError(message)
    expected_c101_names = {"cifar10.1_v6_data.npy", "cifar10.1_v6_labels.npy"}
    observed_c101_names = {
        path.name for path in cifar101_root.glob("*.npy") if path.is_file() and not path.name.startswith("._")
    }
    if observed_c101_names != expected_c101_names:
        missing = sorted(expected_c101_names - observed_c101_names)
        unexpected = sorted(observed_c101_names - expected_c101_names)
        message = f"CIFAR-10.1 v6 declared array set mismatch: missing={missing}, unexpected={unexpected}"
        if missing:
            raise FileNotFoundError(message)
        raise CifarPopulationError(message)

    corruption_rows = []
    corruption_count = scope.base_count * len(scope.severities)
    for group, names in (
        ("standard_test", scope.standard_corruptions),
        ("validation", scope.validation_corruptions),
    ):
        for name in names:
            path = _required_file(cifar10c_root, f"{name}.npy")
            key = f"cifar10c/{name}.npy"
            digest = _verify_hash(path, key, expected)
            array = _load_npy(path)
            _validate_image_array(array, corruption_count, f"CIFAR-10-C {name}")
            corruption_rows.append(
                {
                    "name": name,
                    "group": group,
                    "path": str(path),
                    "sha256": digest,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "severity_block_order": list(scope.severities),
                    "base_ids_per_severity": scope.base_count,
                }
            )

    c10c_label_path = _required_file(cifar10c_root, "labels.npy")
    c10c_label_hash = _verify_hash(c10c_label_path, "cifar10c/labels.npy", expected)
    c10c_labels = _label_acceptance(
        c10c_label_path,
        c10c_label_hash,
        corruption_count,
        "CIFAR-10-C",
        repeated_blocks=len(scope.severities),
        require_all_classes=eligible_for_full_campaign,
    )

    c101_data_path = _required_file(cifar101_root, "cifar10.1_v6_data.npy")
    c101_data_hash = _verify_hash(c101_data_path, "cifar101/cifar10.1_v6_data.npy", expected)
    c101_data = _load_npy(c101_data_path)
    _validate_image_array(c101_data, cifar101_count, "CIFAR-10.1 v6 data")
    c101_label_path = _required_file(cifar101_root, "cifar10.1_v6_labels.npy")
    c101_label_hash = _verify_hash(c101_label_path, "cifar101/cifar10.1_v6_labels.npy", expected)
    c101_labels = _label_acceptance(
        c101_label_path,
        c101_label_hash,
        cifar101_count,
        "CIFAR-10.1 v6",
        require_all_classes=eligible_for_full_campaign,
    )

    return {
        "schema": "kbound_cifar_data_acceptance_v1",
        "status": "FULL_INPUTS_ACCEPTED" if eligible_for_full_campaign else "NON_FULL_TEST_FIXTURE",
        "eligible_for_full_campaign": eligible_for_full_campaign,
        "upstream_download_authenticated": False,
        "hash_authority": "caller-supplied expected hashes; publisher provenance not established by this module",
        "outcome_use": "schema/range/repeated-block validation only; no target scoring or model selection",
        "datasets": {
            "cifar10c": {
                "root": str(cifar10c_root),
                "base_image_count": scope.base_count,
                "corruption_count": len(scope.corruptions),
                "standard_test_corruption_count": len(scope.standard_corruptions),
                "validation_corruption_count": len(scope.validation_corruptions),
                "severity_count": len(scope.severities),
                "expanded_image_count": len(scope.corruptions) * len(scope.severities) * scope.base_count,
                "base_identity_sha256": _base_identity_sha256("cifar10c", scope.base_count),
                "expanded_identity_sha256": _expanded_cifar10c_sha256(scope),
                "corruption_arrays": corruption_rows,
                "label_acceptance": c10c_labels,
            },
            "cifar101": {
                "variant": "v6",
                "root": str(cifar101_root),
                "base_image_count": cifar101_count,
                "base_identity_sha256": _base_identity_sha256("cifar101-v6", cifar101_count),
                "data_array": {
                    "path": str(c101_data_path),
                    "sha256": c101_data_hash,
                    "shape": list(c101_data.shape),
                    "dtype": str(c101_data.dtype),
                },
                "label_acceptance": c101_labels,
            },
        },
        "ignored_files": {
            "appledouble_metadata_count": sum(
                1 for root in (cifar10c_root, cifar101_root) for path in root.glob("._*.npy") if path.is_file()
            )
        },
    }


def _partition_receipt(
    acceptance: Mapping[str, Any],
    scope: Cifar10CScope,
    cifar101_count: int,
    seed: int,
    acceptance_sha256: str,
    eligible_for_full_campaign: bool,
) -> dict[str, Any]:
    memberships = {
        "cifar10c": partition_base_ids(
            "cifar10c", list(range(scope.base_count)), seed=seed, expected_count=scope.base_count
        ),
        "cifar101": partition_base_ids(
            "cifar101-v6", list(range(cifar101_count)), seed=seed, expected_count=cifar101_count
        ),
    }
    counts = {dataset: {role: len(ids) for role, ids in roles.items()} for dataset, roles in memberships.items()}
    identity = {
        dataset: acceptance["datasets"][dataset]["base_identity_sha256"] for dataset in ("cifar10c", "cifar101")
    }
    receipt = {
        "schema": "kbound_cifar_kga_partition_v1",
        "status": "SEALED_INPUT_PARTITION" if eligible_for_full_campaign else "NON_FULL_TEST_FIXTURE",
        "eligible_for_full_campaign": eligible_for_full_campaign,
        "partition_seed": seed,
        "partition_algorithm": {
            "name": "sha256_order_exact_counts",
            "prefix": PARTITION_HASH_PREFIX,
            "key": "SHA256(prefix + NUL + seed + NUL + dataset + NUL + base_id), then base_id",
            "assignment": "first 60% fit, next 20% cal, final 20% check; role lists sorted by base_id",
        },
        "proportions": dict(PARTITION_PROPORTIONS),
        "membership": memberships,
        "membership_counts": counts,
        "base_population_identity_sha256": identity,
        "data_acceptance_sha256": acceptance_sha256,
        "cifar10c_scope": {
            "standard_test_corruptions": list(scope.standard_corruptions),
            "validation_corruptions": list(scope.validation_corruptions),
            "corruptions": list(scope.corruptions),
            "severities": list(scope.severities),
            "base_count": scope.base_count,
            "assignment_reused_across_every_corruption_and_severity": True,
        },
        "cifar101_scope": {"variant": "v6", "base_count": cifar101_count},
        "data_use_boundaries": {
            "source_model_selection_exposure": "NOT_VERIFIED_BY_INPUT_INDEX",
            "historical_target_outcome_status": "OPENED_BEFORE_THIS_CAMPAIGN",
            "partitioning_restores_unopened_status": False,
            "adaptation_evaluation_windows": "must be separately disjoint within each role; use validate_distinct_windows",
            "identity_partition_proves_theorem_sampling_assumptions": False,
            "older_five_fold_cell_protocol_equivalence": False,
        },
    }
    receipt["partition_identity_sha256"] = _json_sha256(
        {
            "partition_seed": seed,
            "proportions": PARTITION_PROPORTIONS,
            "membership": memberships,
            "base_population_identity_sha256": identity,
        }
    )
    return receipt


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_cifar_inputs(
    cifar10c_root: str | Path,
    cifar101_root: str | Path,
    output_dir: str | Path,
    partition_seed: int,
    expected_hashes: Mapping[str, Any],
    *,
    cifar10c_scope: Cifar10CScope = FULL_CIFAR10C_SCOPE,
    cifar101_count: int = FULL_CIFAR101_COUNT,
    require_full: bool = True,
) -> CifarPreparationResult:
    """Authenticate complete arrays, then create a fresh separated receipt."""

    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh output directory already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory is missing: {output.parent}")
    if type(partition_seed) is not int:
        raise CifarPopulationError("partition seed must be an integer")
    _validate_scope(cifar10c_scope, cifar101_count)
    eligible = cifar10c_scope == FULL_CIFAR10C_SCOPE and cifar101_count == FULL_CIFAR101_COUNT
    if require_full and not eligible:
        raise CifarPopulationError("full CIFAR preparation requires exact 19x5x10000 CIFAR-10-C and 2000 v6 IDs")
    expected = _validate_expected_hashes(expected_hashes, cifar10c_scope)
    c10c_root = _require_directory(cifar10c_root, "CIFAR-10-C")
    c101_root = _require_directory(cifar101_root, "CIFAR-10.1")
    acceptance = _inspect_inputs(
        c10c_root,
        c101_root,
        cifar10c_scope,
        cifar101_count,
        expected,
        eligible,
    )
    acceptance["expected_hash_manifest_sha256"] = _json_sha256(
        {"schema": "kbound_cifar_expected_hashes_v1", "files": expected}
    )
    acceptance_payload = _json_bytes(acceptance)
    acceptance_sha256 = hashlib.sha256(acceptance_payload).hexdigest()
    partition = _partition_receipt(
        acceptance,
        cifar10c_scope,
        cifar101_count,
        partition_seed,
        acceptance_sha256,
        eligible,
    )
    partition_payload = _json_bytes(partition)

    output.mkdir(mode=0o755)
    acceptance_path = output / "data-acceptance.json"
    partition_path = output / "kga-partition-receipt.json"
    _atomic_write(acceptance_path, acceptance_payload)
    _atomic_write(partition_path, partition_payload)
    if _streaming_sha256(acceptance_path) != acceptance_sha256:
        raise CifarPopulationError("data acceptance changed during output verification")
    if json.loads(partition_path.read_text(encoding="utf-8")) != partition:
        raise CifarPopulationError("partition receipt changed during output verification")
    return CifarPreparationResult(acceptance_path, partition_path, eligible)


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CifarPopulationError(f"duplicate JSON key in expected-hash manifest: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CifarPopulationError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CifarPopulationError(f"cannot read expected-hash manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CifarPopulationError("expected-hash manifest must be a JSON object")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authenticate and partition full CIFAR target inputs")
    parser.add_argument("--cifar10c-root", required=True)
    parser.add_argument("--cifar101-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition-seed", required=True, type=int)
    parser.add_argument("--expected-hashes", required=True)
    args = parser.parse_args(argv)
    result = prepare_cifar_inputs(
        cifar10c_root=args.cifar10c_root,
        cifar101_root=args.cifar101_root,
        output_dir=args.output_dir,
        partition_seed=args.partition_seed,
        expected_hashes=_strict_json(Path(args.expected_hashes)),
    )
    print(
        json.dumps(
            {
                "status": "OK",
                "eligible_for_full_campaign": result.eligible_for_full_campaign,
                "data_acceptance": str(result.data_acceptance_path),
                "partition_receipt": str(result.partition_receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
