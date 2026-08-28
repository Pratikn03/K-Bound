#!/usr/bin/env python3
"""Chunked, source-only data-quality preflight for So2Sat LCZ42.

The public entry point accepts exactly one image container, ``training.h5``,
and one receipt-verified population manifest.  The safe ``training_geo.h5``
path is derived (never supplied by a caller) and content-verified against the
manifest.  Validation/testing image containers and their labels are outside
the representable interface.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import (
    IntegrityError,
    canonical_json_bytes,
    file_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .label_firewall import VerifiedTrainingGeoIndex
from .metadata_manifest import (
    GEO_BASENAMES,
    CityAllocationContract,
)
from .protocol import GATE_CITY_SALT, PROTOCOL_ID


SCHEMA = "kbound_so2sat_source_data_preflight_v1"
TRAINING_DATA_BASENAME = "training.h5"
TRAINING_GEO_BASENAME = GEO_BASENAMES["training"]
EXPECTED_DATASETS = ("label", "sen1", "sen2")
SAMPLE_ROLES = (
    "source_train",
    "source_monitor",
    "gate_fit_probe",
    "gate_fit_evaluation",
    "gate_cal_probe",
    "gate_cal_evaluation",
)
CITY_ROLES = ("source_fit_ineligible", "source_fit_core", "gate_fit", "gate_cal")
NUM_CLASSES = 17
PATCH_HEIGHT = 32
PATCH_WIDTH = 32
CHANNELS = {"sen1": 8, "sen2": 10}
DEFAULT_CHUNK_ROWS = 256
DEFAULT_DUPLICATE_SAMPLE_ROWS = 2_048
ONE_HOT_ATOL = 1e-6
_TARGET_SPLIT_TOKENS = frozenset({"validation", "testing"})

H5Factory = Callable[[Path], AbstractContextManager[Any]]


def _default_h5_factory(path: Path) -> AbstractContextManager[Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on research environment
        raise RuntimeError("So2Sat source preflight requires h5py") from exc
    return h5py.File(path, "r")


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for component in path.parts:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", component.casefold()) if token)
    return tokens


def require_source_training_path(path: str | os.PathLike[str]) -> Path:
    """Resolve and validate the only image-container path this module accepts."""

    source = Path(path).expanduser().resolve()
    if source.name != TRAINING_DATA_BASENAME:
        raise IntegrityError(
            f"source preflight accepts only {TRAINING_DATA_BASENAME!r}; found {source.name!r}"
        )
    forbidden = sorted(_path_tokens(source) & _TARGET_SPLIT_TOKENS)
    if forbidden:
        raise IntegrityError(
            "source preflight refuses a path containing target split token(s): "
            + ", ".join(forbidden)
        )
    if not source.is_file():
        raise FileNotFoundError(f"missing source training container: {source}")
    return source


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IntegrityError(f"{field} must be a positive integer")
    return value


def _manifest_contract(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, int], frozenset[str], frozenset[str], CityAllocationContract]:
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise IntegrityError("population manifest has an unknown So2Sat protocol id")
    splits = manifest.get("splits")
    cities = manifest.get("cities")
    partition = manifest.get("partition_contract")
    if not isinstance(splits, Mapping) or set(splits) != {"training", "validation", "testing"}:
        raise IntegrityError("population manifest must contain exactly the three official splits")
    if not isinstance(cities, Mapping) or not isinstance(partition, Mapping):
        raise IntegrityError("population manifest lacks city/partition contracts")
    counts: dict[str, int] = {}
    for split in ("training", "validation", "testing"):
        split_doc = splits.get(split)
        if not isinstance(split_doc, Mapping):
            raise IntegrityError(f"population manifest {split} split must be a mapping")
        expected = _positive_int(
            split_doc.get("expected_samples"), field=f"{split}.expected_samples"
        )
        observed = _positive_int(
            split_doc.get("observed_samples"), field=f"{split}.observed_samples"
        )
        if observed != expected:
            raise IntegrityError(f"population manifest {split} expected/observed counts differ")
        counts[split] = observed
    training_cities = cities.get("training")
    target_cities = cities.get("target")
    if (
        not isinstance(training_cities, list)
        or not training_cities
        or any(not isinstance(city, str) or not city for city in training_cities)
        or len(training_cities) != len(set(training_cities))
    ):
        raise IntegrityError("population manifest has invalid training-city identities")
    if (
        not isinstance(target_cities, list)
        or not target_cities
        or any(not isinstance(city, str) or not city for city in target_cities)
        or len(target_cities) != len(set(target_cities))
    ):
        raise IntegrityError("population manifest has invalid target-city identities")
    if set(training_cities) & set(target_cities):
        raise IntegrityError("population manifest source/target city sets overlap")
    if partition.get("gate_city_salt_sha256") != stable_sha256(GATE_CITY_SALT):
        raise IntegrityError("population manifest gate-city salt identity drift")
    allocation = CityAllocationContract(
        minimum_eligible_rows=_positive_int(
            partition.get("minimum_eligible_rows"), field="minimum_eligible_rows"
        ),
        expected_ineligible_city_count=_positive_int(
            partition.get("expected_ineligible_city_count"),
            field="expected_ineligible_city_count",
        ),
        source_fit_core_count=_positive_int(
            partition.get("source_fit_core_count"), field="source_fit_core_count"
        ),
        gate_fit_count=_positive_int(partition.get("gate_fit_count"), field="gate_fit_count"),
        gate_cal_count=_positive_int(partition.get("gate_cal_count"), field="gate_cal_count"),
        gate_salt=GATE_CITY_SALT,
    )
    return counts, frozenset(training_cities), frozenset(target_cities), allocation


def _load_sealed_manifest(
    path: str | os.PathLike[str],
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, int],
    frozenset[str],
    frozenset[str],
    CityAllocationContract,
]:
    manifest_path = Path(path).expanduser().resolve()
    verify_artifact_receipt(manifest_path)
    document = strict_json_load(manifest_path)
    if not isinstance(document, dict):
        raise IntegrityError("So2Sat population manifest must be a JSON mapping")
    claimed = require_sha256(document.get("manifest_sha256"), field="manifest_sha256")
    unsigned = dict(document)
    unsigned.pop("manifest_sha256", None)
    if stable_sha256(unsigned) != claimed:
        raise IntegrityError("So2Sat population manifest self-hash mismatch")
    counts, training_cities, target_cities, allocation = _manifest_contract(document)
    return manifest_path, document, counts, training_cities, target_cities, allocation


def _audit_configuration(*, chunk_rows: int, duplicate_sample_rows: int) -> dict[str, Any]:
    _positive_int(chunk_rows, field="chunk_rows")
    _positive_int(duplicate_sample_rows, field="duplicate_sample_rows")
    if chunk_rows > 8_192:
        raise IntegrityError("chunk_rows may not exceed 8192")
    return {
        "schema": "kbound_so2sat_source_data_preflight_config_v1",
        "expected_hdf5_keys": list(EXPECTED_DATASETS),
        "expected_shapes": {
            "sen1": ["training_rows", PATCH_HEIGHT, PATCH_WIDTH, CHANNELS["sen1"]],
            "sen2": ["training_rows", PATCH_HEIGHT, PATCH_WIDTH, CHANNELS["sen2"]],
            "label": ["training_rows", NUM_CLASSES],
        },
        "chunk_rows": chunk_rows,
        "one_hot_absolute_tolerance": ONE_HOT_ATOL,
        "pixel_requirements": {
            "numeric_real_dtype": True,
            "finite": True,
            "hard_physical_range": None,
            "range_policy": "report_empirical_min_max_per_band_no_undocumented_cutoff",
        },
        "label_requirements": {
            "numeric_real_dtype": True,
            "finite": True,
            "one_hot_17_classes": True,
            "all_classes_present_in_complete_training_population": True,
        },
        "duplicate_screen": {
            "method": "evenly_spaced_rows_sha256_exact_bytes",
            "maximum_rows_per_sensor": duplicate_sample_rows,
            "also_check_all_zero_and_adjacent_exact_duplicate_rows": True,
            "interpretation": "screen_not_an_exhaustive_global_duplicate_test",
        },
    }


def _code_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    files = (
        "source_preflight.py",
        "integrity.py",
        "metadata_manifest.py",
        "label_firewall.py",
        "protocol.py",
    )
    rows = []
    for basename in files:
        path = package / basename
        rows.append(
            {
                "artifact": f"experiments/kbound/so2sat/{basename}",
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "files": rows,
        "aggregate_sha256": stable_sha256(rows),
    }


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _build_geo_alignment(
    geo_index: VerifiedTrainingGeoIndex,
    *,
    expected_rows: int,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    role_to_code = {role: index for index, role in enumerate(SAMPLE_ROLES)}
    city_role_to_code = {role: index for index, role in enumerate(CITY_ROLES)}
    sample_role_codes = np.empty(expected_rows, dtype=np.uint8)
    city_role_codes = np.empty(expected_rows, dtype=np.uint8)
    geo_fingerprints = np.empty(expected_rows, dtype=np.uint64)
    ordered_role_digest = hashlib.sha256()
    sample_role_counts: Counter[str] = Counter()
    city_role_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    blocks: set[str] = set()
    observed = 0
    for row_index, record in enumerate(geo_index.iter_records()):
        if row_index >= expected_rows:
            raise IntegrityError("training geography exceeds the sealed image population")
        if record.row_index != row_index:
            raise IntegrityError("training geography row order is not contiguous")
        if record.sample_id != f"training:{row_index:06d}" or record.official_split != "training":
            raise IntegrityError("training geography sample identity is not row-aligned")
        if record.sample_role not in role_to_code:
            raise IntegrityError(
                f"training geography has unknown sample role {record.sample_role!r}"
            )
        if record.city_role not in city_role_to_code:
            raise IntegrityError(f"training geography has unknown city role {record.city_role!r}")
        sample_role_codes[row_index] = role_to_code[record.sample_role]
        city_role_codes[row_index] = city_role_to_code[record.city_role]
        identity = {
            "city_id": record.city_id,
            "epsg": record.epsg,
            "tfw": list(record.tfw),
        }
        fingerprint = hashlib.blake2b(canonical_json_bytes(identity), digest_size=8).digest()
        geo_fingerprints[row_index] = int.from_bytes(fingerprint, "big")
        ordered_role_digest.update(
            canonical_json_bytes(
                {
                    "row_index": row_index,
                    "sample_id": record.sample_id,
                    "city_id": record.city_id,
                    "city_role": record.city_role,
                    "sample_role": record.sample_role,
                }
            )
        )
        ordered_role_digest.update(b"\n")
        sample_role_counts[record.sample_role] += 1
        city_role_counts[record.city_role] += 1
        city_counts[record.city_id] += 1
        blocks.add(record.spatial_block_id)
        observed += 1
    if observed != expected_rows:
        raise IntegrityError(
            f"training geography has {observed} rows but training.h5 requires {expected_rows}"
        )
    training_doc = manifest["splits"]["training"]
    expected_sample_roles = training_doc.get("sample_role_counts")
    expected_city_roles = training_doc.get("city_role_counts")
    expected_city_counts = training_doc.get("city_counts")
    if dict(sorted(sample_role_counts.items())) != expected_sample_roles:
        raise IntegrityError("row-reconstructed sample-role counts differ from the population seal")
    if dict(sorted(city_role_counts.items())) != expected_city_roles:
        raise IntegrityError("row-reconstructed city-role counts differ from the population seal")
    if dict(sorted(city_counts.items())) != expected_city_counts:
        raise IntegrityError("row-reconstructed city counts differ from the population seal")
    sorted_fingerprints = np.sort(geo_fingerprints)
    duplicate_fingerprint_rows = int(
        np.count_nonzero(sorted_fingerprints[1:] == sorted_fingerprints[:-1])
    )
    alignment = {
        "status": "VERIFIED",
        "training_rows": expected_rows,
        "contiguous_row_indices": True,
        "sample_id_matches_row_index": True,
        "sample_role_counts": dict(sorted(sample_role_counts.items())),
        "city_role_counts": dict(sorted(city_role_counts.items())),
        "city_counts": dict(sorted(city_counts.items())),
        "distinct_spatial_blocks": len(blocks),
        "ordered_row_role_sha256": ordered_role_digest.hexdigest(),
        "geo_identity_duplicate_screen": {
            "method": "blake2b_64_of_normalized_city_epsg_tfw",
            "rows": expected_rows,
            "duplicate_fingerprint_rows": duplicate_fingerprint_rows,
            "hash_collision_caveat": True,
        },
    }
    return alignment, sample_role_codes, city_role_codes


def _require_real_numeric_dataset(dataset: Any, *, name: str) -> np.dtype[Any]:
    try:
        dtype = np.dtype(dataset.dtype)
    except (AttributeError, TypeError) as exc:
        raise IntegrityError(
            f"training.h5/{name} does not expose a NumPy-compatible dtype"
        ) from exc
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise IntegrityError(f"training.h5/{name} must use a real numeric dtype, found {dtype}")
    return dtype


def _sample_indices(population_n: int, maximum_rows: int) -> np.ndarray[Any, np.dtype[np.int64]]:
    count = min(population_n, maximum_rows)
    if count == population_n:
        return np.arange(population_n, dtype=np.int64)
    return np.unique(np.linspace(0, population_n - 1, num=count, dtype=np.int64))


def _scan_pixels(
    dataset: Any,
    *,
    name: str,
    population_n: int,
    chunk_rows: int,
    duplicate_sample_rows: int,
) -> dict[str, Any]:
    dtype = _require_real_numeric_dataset(dataset, name=name)
    bands = CHANNELS[name]
    expected_shape = (population_n, PATCH_HEIGHT, PATCH_WIDTH, bands)
    if tuple(dataset.shape) != expected_shape:
        raise IntegrityError(
            f"training.h5/{name} shape drift: expected {expected_shape}, found {dataset.shape}"
        )
    finite_counts = np.zeros(bands, dtype=np.int64)
    sums = np.zeros(bands, dtype=np.float64)
    sum_squares = np.zeros(bands, dtype=np.float64)
    minima = np.full(bands, np.inf, dtype=np.float64)
    maxima = np.full(bands, -np.inf, dtype=np.float64)
    zero_counts = np.zeros(bands, dtype=np.int64)
    negative_counts = np.zeros(bands, dtype=np.int64)
    nonfinite_rows = 0
    all_zero_rows = 0
    adjacent_duplicate_rows = 0
    previous_last: np.ndarray[Any, Any] | None = None
    sampled = _sample_indices(population_n, duplicate_sample_rows)
    sampled_set = set(int(index) for index in sampled)
    fingerprints: list[str] = []
    for start in range(0, population_n, chunk_rows):
        stop = min(start + chunk_rows, population_n)
        values = np.asarray(dataset[start:stop])
        if values.shape != (stop - start, PATCH_HEIGHT, PATCH_WIDTH, bands):
            raise IntegrityError(f"training.h5/{name} returned a truncated or reshaped chunk")
        finite = np.isfinite(values)
        finite_rows = np.all(finite, axis=(1, 2, 3))
        nonfinite_rows += int(np.count_nonzero(~finite_rows))
        all_zero_rows += int(np.count_nonzero(np.all(values == 0, axis=(1, 2, 3))))
        if previous_last is not None and np.array_equal(previous_last, values[0]):
            adjacent_duplicate_rows += 1
        if values.shape[0] > 1:
            adjacent_duplicate_rows += int(
                np.count_nonzero(np.all(values[1:] == values[:-1], axis=(1, 2, 3)))
            )
        previous_last = np.array(values[-1], copy=True)
        flattened = values.reshape(-1, bands)
        finite_flat = finite.reshape(-1, bands)
        finite_counts += finite_flat.sum(axis=0, dtype=np.int64)
        safe = np.where(finite_flat, flattened, 0.0).astype(np.float64, copy=False)
        sums += safe.sum(axis=0, dtype=np.float64)
        sum_squares += np.square(safe).sum(axis=0, dtype=np.float64)
        finite_min_input = np.where(finite_flat, flattened, np.inf)
        finite_max_input = np.where(finite_flat, flattened, -np.inf)
        minima = np.minimum(minima, finite_min_input.min(axis=0))
        maxima = np.maximum(maxima, finite_max_input.max(axis=0))
        zero_counts += np.count_nonzero(finite_flat & (flattened == 0), axis=0)
        negative_counts += np.count_nonzero(finite_flat & (flattened < 0), axis=0)
        for row_index in range(start, stop):
            if row_index in sampled_set:
                row = np.ascontiguousarray(values[row_index - start])
                digest = hashlib.sha256()
                digest.update(str(dtype).encode("ascii"))
                digest.update(canonical_json_bytes(list(row.shape)))
                digest.update(row.tobytes(order="C"))
                fingerprints.append(digest.hexdigest())
    if np.any(finite_counts == 0):
        raise IntegrityError(f"training.h5/{name} has a band with no finite observations")
    means = sums / finite_counts
    variances = np.maximum(0.0, sum_squares / finite_counts - np.square(means))
    fingerprint_counts = Counter(fingerprints)
    duplicate_groups = sum(count > 1 for count in fingerprint_counts.values())
    duplicate_rows = sum(count - 1 for count in fingerprint_counts.values() if count > 1)
    per_band = []
    pixels_per_band = population_n * PATCH_HEIGHT * PATCH_WIDTH
    for band in range(bands):
        per_band.append(
            {
                "band_index": band,
                "finite_elements": int(finite_counts[band]),
                "nonfinite_elements": int(pixels_per_band - finite_counts[band]),
                "minimum": float(minima[band]),
                "maximum": float(maxima[band]),
                "mean": float(means[band]),
                "standard_deviation": float(math.sqrt(float(variances[band]))),
                "zero_elements": int(zero_counts[band]),
                "negative_elements": int(negative_counts[band]),
            }
        )
    total_elements = pixels_per_band * bands
    total_finite = int(finite_counts.sum())
    return {
        "shape": list(expected_shape),
        "dtype": str(dtype),
        "elements": total_elements,
        "finite_elements": total_finite,
        "nonfinite_elements": total_elements - total_finite,
        "finite_rate": total_finite / total_elements,
        "minimum": float(np.min(minima)),
        "maximum": float(np.max(maxima)),
        "all_zero_rows": all_zero_rows,
        "nonfinite_rows": nonfinite_rows,
        "adjacent_exact_duplicate_rows": adjacent_duplicate_rows,
        "sampled_exact_duplicate_screen": {
            "method": "sha256_dtype_shape_and_c_order_bytes",
            "sampled_rows": len(fingerprints),
            "requested_maximum_rows": duplicate_sample_rows,
            "sample_index_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(),
            "duplicate_groups": duplicate_groups,
            "duplicate_rows_beyond_first": duplicate_rows,
            "exhaustive": len(fingerprints) == population_n,
        },
        "per_band": per_band,
    }


def _class_count_document(counts: np.ndarray[Any, Any]) -> dict[str, int]:
    return {str(class_index): int(counts[class_index]) for class_index in range(NUM_CLASSES)}


def _scan_labels(
    dataset: Any,
    *,
    population_n: int,
    chunk_rows: int,
    sample_role_codes: np.ndarray[Any, Any],
    city_role_codes: np.ndarray[Any, Any],
) -> dict[str, Any]:
    dtype = _require_real_numeric_dataset(dataset, name="label")
    expected_shape = (population_n, NUM_CLASSES)
    if tuple(dataset.shape) != expected_shape:
        raise IntegrityError(
            f"training.h5/label shape drift: expected {expected_shape}, found {dataset.shape}"
        )
    total_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    by_sample_role = np.zeros((len(SAMPLE_ROLES), NUM_CLASSES), dtype=np.int64)
    by_city_role = np.zeros((len(CITY_ROLES), NUM_CLASSES), dtype=np.int64)
    invalid_by_sample_role = np.zeros(len(SAMPLE_ROLES), dtype=np.int64)
    finite_elements = 0
    nonfinite_rows = 0
    invalid_binary_rows = 0
    invalid_sum_rows = 0
    valid_rows = 0
    minimum = np.inf
    maximum = -np.inf
    minimum_row_sum = np.inf
    maximum_row_sum = -np.inf
    for start in range(0, population_n, chunk_rows):
        stop = min(start + chunk_rows, population_n)
        values = np.asarray(dataset[start:stop])
        if values.shape != (stop - start, NUM_CLASSES):
            raise IntegrityError("training.h5/label returned a truncated or reshaped chunk")
        finite = np.isfinite(values)
        row_finite = np.all(finite, axis=1)
        finite_elements += int(np.count_nonzero(finite))
        nonfinite_rows += int(np.count_nonzero(~row_finite))
        finite_values = values[finite]
        if finite_values.size:
            minimum = min(minimum, float(np.min(finite_values)))
            maximum = max(maximum, float(np.max(finite_values)))
        near_zero = np.isclose(values, 0.0, rtol=0.0, atol=ONE_HOT_ATOL)
        near_one = np.isclose(values, 1.0, rtol=0.0, atol=ONE_HOT_ATOL)
        binary_rows = row_finite & np.all(near_zero | near_one, axis=1)
        row_sums = np.where(finite, values, 0.0).sum(axis=1, dtype=np.float64)
        if np.any(row_finite):
            minimum_row_sum = min(minimum_row_sum, float(np.min(row_sums[row_finite])))
            maximum_row_sum = max(maximum_row_sum, float(np.max(row_sums[row_finite])))
        sum_rows = row_finite & np.isclose(row_sums, 1.0, rtol=0.0, atol=ONE_HOT_ATOL)
        valid = binary_rows & sum_rows
        invalid_binary_rows += int(np.count_nonzero(row_finite & ~binary_rows))
        invalid_sum_rows += int(np.count_nonzero(row_finite & ~sum_rows))
        valid_rows += int(np.count_nonzero(valid))
        classes = np.argmax(values, axis=1)
        total_counts += np.bincount(classes[valid], minlength=NUM_CLASSES)
        role_slice = sample_role_codes[start:stop]
        city_role_slice = city_role_codes[start:stop]
        for role_code in range(len(SAMPLE_ROLES)):
            mask = valid & (role_slice == role_code)
            by_sample_role[role_code] += np.bincount(classes[mask], minlength=NUM_CLASSES)
            invalid_by_sample_role[role_code] += int(
                np.count_nonzero((~valid) & (role_slice == role_code))
            )
        for role_code in range(len(CITY_ROLES)):
            mask = valid & (city_role_slice == role_code)
            by_city_role[role_code] += np.bincount(classes[mask], minlength=NUM_CLASSES)
    total_elements = population_n * NUM_CLASSES
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise IntegrityError("training.h5/label has no finite values")
    return {
        "shape": list(expected_shape),
        "dtype": str(dtype),
        "elements": total_elements,
        "finite_elements": finite_elements,
        "nonfinite_elements": total_elements - finite_elements,
        "nonfinite_rows": nonfinite_rows,
        "minimum": minimum,
        "maximum": maximum,
        "minimum_finite_row_sum": minimum_row_sum,
        "maximum_finite_row_sum": maximum_row_sum,
        "one_hot_valid_rows": valid_rows,
        "one_hot_valid_rate": valid_rows / population_n,
        "invalid_binary_rows": invalid_binary_rows,
        "invalid_sum_rows": invalid_sum_rows,
        "class_counts": _class_count_document(total_counts),
        "missing_classes": [int(index) for index in np.flatnonzero(total_counts == 0)],
        "class_counts_by_sample_role": {
            role: _class_count_document(by_sample_role[index])
            for index, role in enumerate(SAMPLE_ROLES)
        },
        "invalid_rows_by_sample_role": {
            role: int(invalid_by_sample_role[index]) for index, role in enumerate(SAMPLE_ROLES)
        },
        "missing_classes_by_sample_role": {
            role: [int(class_index) for class_index in np.flatnonzero(by_sample_role[index] == 0)]
            for index, role in enumerate(SAMPLE_ROLES)
        },
        "class_counts_by_city_role": {
            role: _class_count_document(by_city_role[index])
            for index, role in enumerate(CITY_ROLES)
        },
    }


def _quality_gate(
    datasets: Mapping[str, Mapping[str, Any]],
    geo: Mapping[str, Any],
) -> dict[str, Any]:
    critical: list[str] = []
    warnings: list[str] = []
    for sensor in ("sen1", "sen2"):
        profile = datasets[sensor]
        if profile["nonfinite_elements"]:
            critical.append(f"{sensor} contains non-finite pixel values")
        if profile["all_zero_rows"]:
            warnings.append(f"{sensor} contains all-zero rows")
        if profile["adjacent_exact_duplicate_rows"]:
            warnings.append(f"{sensor} contains adjacent exact duplicate rows")
        duplicate_screen = profile["sampled_exact_duplicate_screen"]
        if duplicate_screen["duplicate_rows_beyond_first"]:
            warnings.append(f"{sensor} sampled exact-duplicate screen found repeated rows")
        if any(band["standard_deviation"] == 0.0 for band in profile["per_band"]):
            warnings.append(f"{sensor} contains at least one constant band")
    labels = datasets["label"]
    if labels["nonfinite_elements"]:
        critical.append("label contains non-finite values")
    if labels["one_hot_valid_rows"] != labels["shape"][0]:
        critical.append("label contains rows that are not valid 17-class one-hot vectors")
    if labels["missing_classes"]:
        critical.append("complete training population is missing one or more declared classes")
    roles_with_missing = [
        role for role, missing in labels["missing_classes_by_sample_role"].items() if missing
    ]
    if roles_with_missing:
        warnings.append("some sealed sample roles do not contain every class")
    duplicate_geo = geo["geo_identity_duplicate_screen"]["duplicate_fingerprint_rows"]
    if duplicate_geo:
        warnings.append("normalized city/EPSG/TFW identity screen found repeated fingerprints")
    ready = not critical
    if not ready:
        status = "SOURCE_DATA_PREFLIGHT_BLOCKED"
    elif warnings:
        status = "SOURCE_DATA_PREFLIGHT_PASSED_WITH_WARNINGS"
    else:
        status = "SOURCE_DATA_PREFLIGHT_PASSED"
    return {
        "status": status,
        "ready_for_source_training": ready,
        "critical_findings": critical,
        "warnings": warnings,
        "range_interpretation": (
            "Empirical sensor ranges are reported but not judged against an undocumented cutoff."
        ),
    }


def audit_source_training_h5(
    training_h5: str | os.PathLike[str],
    population_manifest: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    duplicate_sample_rows: int = DEFAULT_DUPLICATE_SAMPLE_ROWS,
    h5_factory: H5Factory | None = None,
) -> dict[str, Any]:
    """Audit the official training container and write a create-only JSON receipt."""

    source = require_source_training_path(training_h5)
    destination = Path(output).expanduser().resolve()
    destination_receipt = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or destination_receipt.exists():
        raise IntegrityError(f"refusing to overwrite artifact/receipt pair for {destination}")
    config = _audit_configuration(
        chunk_rows=chunk_rows,
        duplicate_sample_rows=duplicate_sample_rows,
    )
    code_before = _code_identity()
    (
        manifest_path,
        manifest,
        split_counts,
        training_cities,
        target_cities,
        allocation,
    ) = _load_sealed_manifest(population_manifest)
    expected_rows = split_counts["training"]
    training_geo = source.with_name(TRAINING_GEO_BASENAME)
    if _path_tokens(training_geo) & _TARGET_SPLIT_TOKENS:
        raise IntegrityError("derived training geography path contains a target split token")
    factory = _default_h5_factory if h5_factory is None else h5_factory
    geo_index = VerifiedTrainingGeoIndex(
        manifest,
        training_geo,
        h5_factory=factory,
        expected_split_counts=split_counts,
        expected_training_cities=training_cities,
        expected_target_cities=target_cities,
        allocation_contract=allocation,
    )
    geo_alignment, sample_role_codes, city_role_codes = _build_geo_alignment(
        geo_index,
        expected_rows=expected_rows,
        manifest=manifest,
    )
    source_stat_before = _stat_signature(source)
    container_hash = file_sha256(source)
    with factory(source) as handle:
        keys = sorted(str(key) for key in handle.keys())
        if keys != list(EXPECTED_DATASETS):
            raise IntegrityError(
                f"training.h5 must contain exactly {list(EXPECTED_DATASETS)}, found {keys}"
            )
        datasets = {
            "sen1": _scan_pixels(
                handle["sen1"],
                name="sen1",
                population_n=expected_rows,
                chunk_rows=chunk_rows,
                duplicate_sample_rows=duplicate_sample_rows,
            ),
            "sen2": _scan_pixels(
                handle["sen2"],
                name="sen2",
                population_n=expected_rows,
                chunk_rows=chunk_rows,
                duplicate_sample_rows=duplicate_sample_rows,
            ),
            "label": _scan_labels(
                handle["label"],
                population_n=expected_rows,
                chunk_rows=chunk_rows,
                sample_role_codes=sample_role_codes,
                city_role_codes=city_role_codes,
            ),
        }
    if _stat_signature(source) != source_stat_before:
        raise IntegrityError("training.h5 changed while the preflight was running")
    code_after = _code_identity()
    if code_after != code_before:
        raise IntegrityError("preflight code changed while the audit was running")
    quality = _quality_gate(datasets, geo_alignment)
    manifest_receipt_path = manifest_path.with_name(manifest_path.name + ".receipt.json")
    manifest_receipt = strict_json_load(manifest_receipt_path)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": quality["status"],
        "scope": {
            "official_image_split_opened": "training",
            "opened_hdf5_files": [TRAINING_GEO_BASENAME, TRAINING_DATA_BASENAME],
            "opened_training_datasets": ["sen1", "sen2", "label"],
            "target_image_containers_opened": False,
            "target_outcome_arrays_opened": False,
            "target_outcome_arrays_counted": False,
            "target_outcome_arrays_hashed": False,
            "training_label_aggregation": (
                "class_counts_only_by_sealed_source_and_gate_development_roles"
            ),
        },
        "configuration": config,
        "configuration_sha256": stable_sha256(config),
        "code_identity": code_after,
        "population_manifest_identity": {
            "basename": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "file_sha256": file_sha256(manifest_path),
            "manifest_sha256": require_sha256(
                manifest.get("manifest_sha256"), field="manifest_sha256"
            ),
            "population_identity_sha256": require_sha256(
                manifest.get("population_identity_sha256"),
                field="population_identity_sha256",
            ),
            "receipt_artifact_sha256": require_sha256(
                manifest_receipt.get("artifact_sha256"), field="receipt.artifact_sha256"
            ),
        },
        "training_container_identity": {
            "basename": source.name,
            "bytes": source.stat().st_size,
            "sha256": container_hash,
        },
        "training_geo_alignment": geo_alignment,
        "datasets": datasets,
        "quality_gate": quality,
    }
    receipt = write_immutable_json_with_receipt(
        destination,
        document,
        receipt_schema="kbound_so2sat_source_data_preflight_receipt_v1",
    )
    return {
        "document": document,
        "artifact_receipt": receipt,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-h5", required=True, help="Exact path to official training.h5")
    parser.add_argument(
        "--population-manifest",
        required=True,
        help="Receipt-verified label-free population manifest",
    )
    parser.add_argument("--output", required=True, help="Create-only JSON audit artifact")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument(
        "--duplicate-sample-rows",
        type=int,
        default=DEFAULT_DUPLICATE_SAMPLE_ROWS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = audit_source_training_h5(
        args.training_h5,
        args.population_manifest,
        args.output,
        chunk_rows=args.chunk_rows,
        duplicate_sample_rows=args.duplicate_sample_rows,
    )
    document = result["document"]
    receipt = result["artifact_receipt"]
    print(
        f"{document['status']} artifact={receipt['artifact_sha256']} "
        f"container={document['training_container_identity']['sha256']}"
    )
    return 0 if document["quality_gate"]["ready_for_source_training"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
