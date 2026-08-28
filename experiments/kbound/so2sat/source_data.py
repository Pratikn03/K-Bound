"""Fail-closed source-only data path for So2Sat LCZ42 training.

Only two sample roles may expose labels here: ``source_train`` for fitting and
``source_monitor`` for checkpoint selection.  Role assignment is reconstructed
from the receipt-verified, label-free geographic manifest before ``training.h5``
is opened.  No API in this module accepts a validation or testing container.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from .integrity import (
    IntegrityError,
    file_sha256,
    ordered_records_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .model import INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES


TRAINING_DATA_BASENAME = "training.h5"
SOURCE_TRAIN_ROLE = "source_train"
SOURCE_MONITOR_ROLE = "source_monitor"
SOURCE_ROLES = frozenset({SOURCE_TRAIN_ROLE, SOURCE_MONITOR_ROLE})
KNOWN_TRAINING_ROLES = frozenset(
    {
        SOURCE_TRAIN_ROLE,
        SOURCE_MONITOR_ROLE,
        "gate_fit_probe",
        "gate_fit_evaluation",
        "gate_cal_probe",
        "gate_cal_evaluation",
    }
)
NORMALIZER_SCHEMA = "kbound_so2sat_source_band_normalizer_v1"
NORMALIZER_STATUS = "SEALED_SOURCE_TRAIN_ONLY"
SENTINEL2_BAND_ORDER = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)


@dataclass(frozen=True)
class SourceRow:
    """Label-access authority for one verified row in ``training.h5``."""

    row_index: int
    sample_id: str
    city_id: str
    spatial_block_id: str
    sample_role: str

    def __post_init__(self) -> None:
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int) or self.row_index < 0:
            raise IntegrityError("source row index must be a non-negative integer")
        if self.sample_role not in SOURCE_ROLES:
            raise IntegrityError("SourceRow may carry only source_train or source_monitor authority")
        if not self.sample_id.startswith("training:"):
            raise IntegrityError("source sample id must belong to the official training split")
        if any(not isinstance(value, str) or not value for value in (self.city_id, self.spatial_block_id)):
            raise IntegrityError("source row city/block identities must be nonempty strings")

    def commitment(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "sample_id": self.sample_id,
            "city_id": self.city_id,
            "spatial_block_id": self.spatial_block_id,
            "sample_role": self.sample_role,
        }


@dataclass(frozen=True)
class SourceRoleInventory:
    """The two source roles reproduced from a sealed geographic population."""

    population_identity_sha256: str
    population_manifest_sha256: str
    training_geo_sha256: str
    training_population_n: int
    source_train_rows: tuple[SourceRow, ...]
    source_monitor_rows: tuple[SourceRow, ...]
    source_rows_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.population_identity_sha256, field="population_identity_sha256")
        require_sha256(self.population_manifest_sha256, field="population_manifest_sha256")
        require_sha256(self.training_geo_sha256, field="training_geo_sha256")
        require_sha256(self.source_rows_sha256, field="source_rows_sha256")
        if (
            isinstance(self.training_population_n, bool)
            or not isinstance(self.training_population_n, int)
            or self.training_population_n < 1
        ):
            raise IntegrityError("training population must be a positive integer")
        if not self.source_train_rows or not self.source_monitor_rows:
            raise IntegrityError("sealed source_train and source_monitor roles must both be nonempty")
        for expected_role, rows in (
            (SOURCE_TRAIN_ROLE, self.source_train_rows),
            (SOURCE_MONITOR_ROLE, self.source_monitor_rows),
        ):
            indices = [row.row_index for row in rows]
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise IntegrityError(f"{expected_role} row indices must be unique and sorted")
            if any(row.sample_role != expected_role for row in rows):
                raise IntegrityError(f"{expected_role} inventory contains another role")
            if any(not 0 <= row.row_index < self.training_population_n for row in rows):
                raise IntegrityError(f"{expected_role} inventory contains an out-of-range row")
        overlap = {row.row_index for row in self.source_train_rows} & {
            row.row_index for row in self.source_monitor_rows
        }
        if overlap:
            raise IntegrityError(f"source_train/source_monitor overlap at rows {sorted(overlap)[:5]}")
        observed_hash = ordered_records_sha256(
            row.commitment()
            for row in (*self.source_train_rows, *self.source_monitor_rows)
        )
        if observed_hash != self.source_rows_sha256:
            raise IntegrityError("source role row commitment SHA-256 mismatch")

    @property
    def source_train_indices(self) -> tuple[int, ...]:
        return tuple(row.row_index for row in self.source_train_rows)

    @property
    def source_monitor_indices(self) -> tuple[int, ...]:
        return tuple(row.row_index for row in self.source_monitor_rows)

    def identity(self) -> dict[str, Any]:
        return {
            "population_identity_sha256": self.population_identity_sha256,
            "population_manifest_sha256": self.population_manifest_sha256,
            "training_geo_sha256": self.training_geo_sha256,
            "training_population_n": self.training_population_n,
            "source_train_n": len(self.source_train_rows),
            "source_monitor_n": len(self.source_monitor_rows),
            "source_rows_sha256": self.source_rows_sha256,
        }


def _source_row(record: Any) -> SourceRow | None:
    required = (
        "row_index",
        "sample_id",
        "city_id",
        "spatial_block_id",
        "sample_role",
        "official_split",
    )
    if any(not hasattr(record, field) for field in required):
        raise IntegrityError("verified training record lacks the required provenance fields")
    if record.official_split != "training":
        raise IntegrityError("source inventory received a non-training geographic record")
    if record.sample_role not in KNOWN_TRAINING_ROLES:
        raise IntegrityError(f"unknown role in sealed training population: {record.sample_role!r}")
    if isinstance(record.row_index, bool) or not isinstance(record.row_index, int):
        raise IntegrityError("verified training row index must be an integer")
    for field in ("sample_id", "city_id", "spatial_block_id"):
        value = getattr(record, field)
        if not isinstance(value, str) or not value:
            raise IntegrityError(f"verified training record has invalid {field}")
    if record.sample_role not in SOURCE_ROLES:
        return None
    return SourceRow(
        row_index=record.row_index,
        sample_id=record.sample_id,
        city_id=record.city_id,
        spatial_block_id=record.spatial_block_id,
        sample_role=record.sample_role,
    )


def build_source_role_inventory(
    geo_index: Any,
    population_manifest: Mapping[str, Any],
) -> SourceRoleInventory:
    """Reproduce every training role, retaining only the two source roles."""

    splits = population_manifest.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get("training"), Mapping):
        raise IntegrityError("population manifest lacks a training split")
    training = splits["training"]
    population_n = training.get("observed_samples")
    if isinstance(population_n, bool) or not isinstance(population_n, int) or population_n < 1:
        raise IntegrityError("population manifest has an invalid training row count")
    expected_counts = training.get("sample_role_counts")
    if not isinstance(expected_counts, Mapping) or set(expected_counts) != set(KNOWN_TRAINING_ROLES):
        raise IntegrityError("population manifest training role schema drift")
    manifest_population_identity = require_sha256(
        population_manifest.get("population_identity_sha256"),
        field="population_identity_sha256",
    )
    if geo_index.population_identity_sha256 != manifest_population_identity:
        raise IntegrityError("training geo index population identity differs from the manifest")

    role_counts: Counter[str] = Counter()
    source_train: list[SourceRow] = []
    source_monitor: list[SourceRow] = []
    records = (
        geo_index.iter_records()
        if callable(getattr(geo_index, "iter_records", None))
        else (geo_index.record(row_index) for row_index in range(population_n))
    )
    observed_rows = 0
    for row_index, record in enumerate(records):
        if row_index >= population_n:
            raise IntegrityError("verified training geo index exceeded the sealed population")
        row = _source_row(record)
        if record.row_index != row_index:
            raise IntegrityError("verified training geo index returned a different row index")
        role_counts[record.sample_role] += 1
        if row is not None and row.sample_role == SOURCE_TRAIN_ROLE:
            source_train.append(row)
        elif row is not None and row.sample_role == SOURCE_MONITOR_ROLE:
            source_monitor.append(row)
        observed_rows += 1
    if observed_rows != population_n:
        raise IntegrityError(
            f"verified training geo index returned {observed_rows} rows, expected {population_n}"
        )
    if dict(role_counts) != dict(expected_counts):
        raise IntegrityError(
            "reconstructed training role counts differ from the sealed population manifest"
        )
    geo = training.get("geo_artifact")
    if not isinstance(geo, Mapping):
        raise IntegrityError("population manifest lacks the training geo identity")
    rows = (*source_train, *source_monitor)
    return SourceRoleInventory(
        population_identity_sha256=manifest_population_identity,
        population_manifest_sha256=require_sha256(
            population_manifest.get("manifest_sha256"), field="manifest_sha256"
        ),
        training_geo_sha256=require_sha256(geo.get("sha256"), field="training_geo_sha256"),
        training_population_n=population_n,
        source_train_rows=tuple(source_train),
        source_monitor_rows=tuple(source_monitor),
        source_rows_sha256=ordered_records_sha256(row.commitment() for row in rows),
    )


def load_verified_source_inventory(
    population_manifest_path: str | os.PathLike[str],
    training_geo_path: str | os.PathLike[str],
    *,
    population_manifest_receipt_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], SourceRoleInventory]:
    """Load a receipt-verified manifest through the training-only geo index."""

    verify_artifact_receipt(
        population_manifest_path,
        population_manifest_receipt_path,
    )
    manifest = strict_json_load(population_manifest_path)
    if not isinstance(manifest, dict):
        raise IntegrityError("So2Sat population manifest must be a JSON mapping")
    # This import is intentionally local.  The source CLI depends only on the
    # training-only index and never constructs the target pixel loader.
    from .label_firewall import VerifiedTrainingGeoIndex

    geo_index = VerifiedTrainingGeoIndex(manifest, training_geo_path)
    return manifest, build_source_role_inventory(geo_index, manifest)


class SourceContainer(Protocol):
    """Minimal source-only pixel/label interface used by training and tests."""

    population_n: int
    identity_sha256: str
    identity: Mapping[str, Any]

    def read_pixels(self, row_indices: Sequence[int]) -> np.ndarray: ...

    def read_labeled(self, row_index: int) -> tuple[np.ndarray, np.ndarray]: ...

    def read_labeled_many(
        self, row_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]: ...


H5Factory = Callable[[Path], AbstractContextManager[Any]]


def _default_h5_factory(path: Path) -> AbstractContextManager[Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on execution environment
        raise RuntimeError("So2Sat source loading requires h5py") from exc
    return h5py.File(path, "r")


class H5SourceContainer:
    """Exact-index access to ``training.h5``; no other split is representable."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        expected_rows: int,
        h5_factory: H5Factory | None = None,
    ) -> None:
        source = Path(path).expanduser().resolve()
        if source.name != TRAINING_DATA_BASENAME:
            raise IntegrityError(
                f"source trainer accepts only {TRAINING_DATA_BASENAME!r}; found {source.name!r}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"missing source training container: {source}")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1:
            raise IntegrityError("expected source population must be a positive integer")
        self.path = source
        self.population_n = expected_rows
        self._factory = _default_h5_factory if h5_factory is None else h5_factory
        self._live_context: AbstractContextManager[Any] | None = None
        self._live_handle: Any | None = None
        self._live_pid: int | None = None
        with self._factory(self.path) as handle:
            # Exact literal keys only.  There is no caller-controlled dataset
            # name and no read of sen1 or of any non-source container.
            pixels = handle["sen2"]
            labels = handle["label"]
            pixel_shape = getattr(pixels, "shape", None)
            label_shape = getattr(labels, "shape", None)
            expected_pixel_shape = (expected_rows, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
            expected_label_shape = (expected_rows, NUM_CLASSES)
            if pixel_shape != expected_pixel_shape:
                raise IntegrityError(
                    f"training.h5/sen2 shape drift: expected {expected_pixel_shape}, found {pixel_shape}"
                )
            if label_shape != expected_label_shape:
                raise IntegrityError(
                    f"training.h5/label shape drift: expected {expected_label_shape}, found {label_shape}"
                )
            pixel_dtype = str(getattr(pixels, "dtype", "unknown"))
            label_dtype = str(getattr(labels, "dtype", "unknown"))
        identity = {
            "schema": "kbound_so2sat_source_container_identity_v1",
            "basename": source.name,
            "bytes": source.stat().st_size,
            "file_sha256": file_sha256(source),
            "sen2_shape": list(expected_pixel_shape),
            "sen2_dtype": pixel_dtype,
            "label_shape": list(expected_label_shape),
            "label_dtype": label_dtype,
            "accessible_official_split": "training",
            "target_split_paths": [],
        }
        self.identity = identity
        self.identity_sha256 = stable_sha256(identity)

    def _handle(self) -> Any:
        """Keep one read-only HDF5 handle per DataLoader process."""

        process_id = os.getpid()
        if self._live_handle is not None and self._live_pid == process_id:
            return self._live_handle
        self.close()
        context = self._factory(self.path)
        self._live_handle = context.__enter__()
        self._live_context = context
        self._live_pid = process_id
        return self._live_handle

    def close(self) -> None:
        if self._live_context is not None:
            self._live_context.__exit__(None, None, None)
        self._live_context = None
        self._live_handle = None
        self._live_pid = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is implementation-specific
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _indices(row_indices: Sequence[int], population_n: int) -> list[int]:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in row_indices
        ):
            raise IntegrityError("source row indices must be integers")
        indices = [int(value) for value in row_indices]
        if not indices:
            raise IntegrityError("source pixel read requires at least one row")
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise IntegrityError("source batch row indices must be unique and sorted")
        if any(not 0 <= value < population_n for value in indices):
            raise IntegrityError("source pixel read contains an out-of-range row")
        return indices

    def read_pixels(self, row_indices: Sequence[int]) -> np.ndarray:
        indices = self._indices(row_indices, self.population_n)
        with self._factory(self.path) as handle:
            values = np.asarray(handle["sen2"][indices], dtype=np.float32)
        expected = (len(indices), INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
        if values.shape != expected:
            raise IntegrityError(f"source pixel batch shape drift: {values.shape} != {expected}")
        return values

    def read_labeled(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise IntegrityError("source row index must be an integer")
        if not 0 <= row_index < self.population_n:
            raise IntegrityError("source labeled read contains an out-of-range row")
        handle = self._handle()
        pixels = np.asarray(handle["sen2"][row_index], dtype=np.float32)
        label = np.asarray(handle["label"][row_index])
        return pixels, label

    def read_labeled_many(
        self, row_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in row_indices
        ):
            raise IntegrityError("source row indices must be integers")
        original = [int(value) for value in row_indices]
        if not original or len(original) != len(set(original)):
            raise IntegrityError("source labeled batch rows must be nonempty and unique")
        if any(not 0 <= value < self.population_n for value in original):
            raise IntegrityError("source labeled batch contains an out-of-range row")
        sorted_indices = sorted(original)
        handle = self._handle()
        sorted_pixels = np.asarray(handle["sen2"][sorted_indices], dtype=np.float32)
        sorted_labels = np.asarray(handle["label"][sorted_indices])
        position = {row_index: index for index, row_index in enumerate(sorted_indices)}
        restore = [position[row_index] for row_index in original]
        return sorted_pixels[restore], sorted_labels[restore]


class ArraySourceContainer:
    """In-memory source container for deterministic synthetic smoke tests."""

    def __init__(self, pixels: np.ndarray, labels: np.ndarray, *, identity_tag: str) -> None:
        pixel_array = np.asarray(pixels, dtype=np.float32)
        label_array = np.asarray(labels)
        if pixel_array.ndim != 4 or pixel_array.shape[1:] != (
            INPUT_HEIGHT,
            INPUT_WIDTH,
            INPUT_CHANNELS,
        ):
            raise IntegrityError("synthetic source pixels must have shape N x 32 x 32 x 10")
        if label_array.shape != (pixel_array.shape[0], NUM_CLASSES):
            raise IntegrityError("synthetic source labels must have shape N x 17")
        if not identity_tag:
            raise IntegrityError("synthetic source identity tag must be nonempty")
        self._pixels = pixel_array.copy()
        self._labels = label_array.copy()
        self.population_n = int(pixel_array.shape[0])
        raw = hashlib.sha256()
        raw.update(self._pixels.tobytes(order="C"))
        raw.update(self._labels.tobytes(order="C"))
        self.identity = {
            "schema": "kbound_so2sat_synthetic_source_identity_v1",
            "identity_tag": identity_tag,
            "population_n": self.population_n,
            "content_sha256": raw.hexdigest(),
            "target_split_paths": [],
        }
        self.identity_sha256 = stable_sha256(self.identity)

    def read_pixels(self, row_indices: Sequence[int]) -> np.ndarray:
        indices = H5SourceContainer._indices(row_indices, self.population_n)
        return self._pixels[indices].copy()

    def read_labeled(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise IntegrityError("source row index must be an integer")
        if not 0 <= row_index < self.population_n:
            raise IntegrityError("source labeled read contains an out-of-range row")
        return self._pixels[row_index].copy(), self._labels[row_index].copy()

    def read_labeled_many(
        self, row_indices: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in row_indices
        ):
            raise IntegrityError("source row indices must be integers")
        indices = [int(value) for value in row_indices]
        if not indices or len(indices) != len(set(indices)):
            raise IntegrityError("source labeled batch rows must be nonempty and unique")
        if any(not 0 <= value < self.population_n for value in indices):
            raise IntegrityError("source labeled batch contains an out-of-range row")
        return self._pixels[indices].copy(), self._labels[indices].copy()


@dataclass(frozen=True)
class BandNormalizer:
    """Source-train-only per-band population mean and standard deviation."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    source_train_n: int
    source_train_pixel_n: int
    source_rows_sha256: str
    source_container_identity_sha256: str
    normalizer_sha256: str

    def __post_init__(self) -> None:
        if len(self.mean) != INPUT_CHANNELS or len(self.std) != INPUT_CHANNELS:
            raise IntegrityError("So2Sat normalizer requires exactly ten band statistics")
        if not all(np.isfinite(value) for value in (*self.mean, *self.std)):
            raise IntegrityError("So2Sat normalizer contains a non-finite statistic")
        if any(value <= 0.0 for value in self.std):
            raise IntegrityError("So2Sat normalizer standard deviations must be positive")
        if self.source_train_n < 1:
            raise IntegrityError("So2Sat normalizer source_train_n must be positive")
        expected_pixels = self.source_train_n * INPUT_HEIGHT * INPUT_WIDTH
        if self.source_train_pixel_n != expected_pixels:
            raise IntegrityError("So2Sat normalizer pixel count is inconsistent")
        require_sha256(self.source_rows_sha256, field="source_rows_sha256")
        require_sha256(
            self.source_container_identity_sha256,
            field="source_container_identity_sha256",
        )
        require_sha256(self.normalizer_sha256, field="normalizer_sha256")
        unsigned = self.document(include_hash=False)
        if stable_sha256(unsigned) != self.normalizer_sha256:
            raise IntegrityError("So2Sat normalizer SHA-256 mismatch")

    def document(self, *, include_hash: bool = True) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema": NORMALIZER_SCHEMA,
            "status": NORMALIZER_STATUS,
            "fit_role": SOURCE_TRAIN_ROLE,
            "excluded_roles": sorted(KNOWN_TRAINING_ROLES - {SOURCE_TRAIN_ROLE}),
            "method": "float64_parallel_welford_population_moments",
            "band_order": list(SENTINEL2_BAND_ORDER),
            "mean": list(self.mean),
            "std": list(self.std),
            "source_train_n": self.source_train_n,
            "source_train_pixel_n": self.source_train_pixel_n,
            "source_rows_sha256": self.source_rows_sha256,
            "source_container_identity_sha256": self.source_container_identity_sha256,
        }
        if include_hash:
            document["normalizer_sha256"] = self.normalizer_sha256
        return document

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "BandNormalizer":
        if document.get("schema") != NORMALIZER_SCHEMA or document.get("status") != NORMALIZER_STATUS:
            raise IntegrityError("unknown or unsealed So2Sat normalizer artifact")
        if document.get("fit_role") != SOURCE_TRAIN_ROLE:
            raise IntegrityError("So2Sat normalization was not fit on source_train")
        if document.get("excluded_roles") != sorted(KNOWN_TRAINING_ROLES - {SOURCE_TRAIN_ROLE}):
            raise IntegrityError("So2Sat normalizer excluded-role contract drift")
        if document.get("method") != "float64_parallel_welford_population_moments":
            raise IntegrityError("So2Sat normalizer method drift")
        if document.get("band_order") != list(SENTINEL2_BAND_ORDER):
            raise IntegrityError("So2Sat normalizer band order drift")
        mean = document.get("mean")
        std = document.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            raise IntegrityError("So2Sat normalizer mean/std must be arrays")
        source_train_n = document.get("source_train_n")
        source_train_pixel_n = document.get("source_train_pixel_n")
        if (
            isinstance(source_train_n, bool)
            or not isinstance(source_train_n, int)
            or isinstance(source_train_pixel_n, bool)
            or not isinstance(source_train_pixel_n, int)
        ):
            raise IntegrityError("So2Sat normalizer counts must be integers")
        return cls(
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
            source_train_n=source_train_n,
            source_train_pixel_n=source_train_pixel_n,
            source_rows_sha256=require_sha256(
                document.get("source_rows_sha256"), field="source_rows_sha256"
            ),
            source_container_identity_sha256=require_sha256(
                document.get("source_container_identity_sha256"),
                field="source_container_identity_sha256",
            ),
            normalizer_sha256=require_sha256(
                document.get("normalizer_sha256"), field="normalizer_sha256"
            ),
        )

    def apply(self, chw: torch.Tensor) -> torch.Tensor:
        if chw.shape != (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH):
            raise IntegrityError(f"normalizer expected 10x32x32 tensor, found {tuple(chw.shape)}")
        mean = torch.as_tensor(self.mean, dtype=chw.dtype, device=chw.device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=chw.dtype, device=chw.device)[:, None, None]
        return (chw - mean) / std


def fit_band_normalizer(
    container: SourceContainer,
    inventory: SourceRoleInventory,
    *,
    chunk_rows: int = 128,
) -> BandNormalizer:
    """Fit streaming ten-band moments from source_train pixels and no others."""

    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise IntegrityError("normalizer chunk_rows must be a positive integer")
    if container.population_n != inventory.training_population_n:
        raise IntegrityError("source container and geographic inventory populations differ")
    count = 0
    mean = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    m2 = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    indices = inventory.source_train_indices
    for start in range(0, len(indices), chunk_rows):
        batch_indices = indices[start : start + chunk_rows]
        pixels = np.asarray(container.read_pixels(batch_indices), dtype=np.float64)
        expected = (len(batch_indices), INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
        if pixels.shape != expected:
            raise IntegrityError(f"source normalizer pixel shape drift: {pixels.shape} != {expected}")
        if not np.isfinite(pixels).all():
            raise IntegrityError("source_train pixels contain NaN or Infinity")
        flat = pixels.reshape(-1, INPUT_CHANNELS)
        batch_n = int(flat.shape[0])
        batch_mean = flat.mean(axis=0, dtype=np.float64)
        centered = flat - batch_mean
        batch_m2 = np.square(centered, dtype=np.float64).sum(axis=0, dtype=np.float64)
        if count == 0:
            mean = batch_mean
            m2 = batch_m2
            count = batch_n
            continue
        delta = batch_mean - mean
        combined = count + batch_n
        mean = mean + delta * (batch_n / combined)
        m2 = m2 + batch_m2 + np.square(delta) * (count * batch_n / combined)
        count = combined
    expected_count = len(indices) * INPUT_HEIGHT * INPUT_WIDTH
    if count != expected_count:
        raise IntegrityError("source_train normalizer did not consume its exact sealed population")
    variance = m2 / count
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise IntegrityError("source_train contains a constant or invalid Sentinel-2 band")
    unsigned = {
        "schema": NORMALIZER_SCHEMA,
        "status": NORMALIZER_STATUS,
        "fit_role": SOURCE_TRAIN_ROLE,
        "excluded_roles": sorted(KNOWN_TRAINING_ROLES - {SOURCE_TRAIN_ROLE}),
        "method": "float64_parallel_welford_population_moments",
        "band_order": list(SENTINEL2_BAND_ORDER),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "source_train_n": len(indices),
        "source_train_pixel_n": count,
        "source_rows_sha256": inventory.source_rows_sha256,
        "source_container_identity_sha256": container.identity_sha256,
    }
    return BandNormalizer(
        mean=tuple(unsigned["mean"]),
        std=tuple(unsigned["std"]),
        source_train_n=len(indices),
        source_train_pixel_n=count,
        source_rows_sha256=inventory.source_rows_sha256,
        source_container_identity_sha256=container.identity_sha256,
        normalizer_sha256=stable_sha256(unsigned),
    )


def seal_band_normalizer(
    path: str | os.PathLike[str],
    normalizer: BandNormalizer,
) -> dict[str, Any]:
    """Create an immutable normalizer plus byte receipt."""

    return write_immutable_json_with_receipt(path, normalizer.document())


def load_sealed_band_normalizer(
    path: str | os.PathLike[str],
    *,
    receipt_path: str | os.PathLike[str] | None = None,
) -> BandNormalizer:
    verify_artifact_receipt(path, receipt_path)
    document = strict_json_load(path)
    if not isinstance(document, Mapping):
        raise IntegrityError("So2Sat normalizer artifact must be a JSON mapping")
    return BandNormalizer.from_document(document)


def _one_hot_class(label: np.ndarray) -> int:
    values = np.asarray(label, dtype=np.float64)
    if values.shape != (NUM_CLASSES,):
        raise IntegrityError(f"source label shape drift: {values.shape} != {(NUM_CLASSES,)}")
    if not np.isfinite(values).all():
        raise IntegrityError("source label contains NaN or Infinity")
    close_zero = np.isclose(values, 0.0, atol=1e-6, rtol=0.0)
    close_one = np.isclose(values, 1.0, atol=1e-6, rtol=0.0)
    if int(close_one.sum()) != 1 or not np.all(close_zero | close_one):
        raise IntegrityError("source label must be exactly one-hot over 17 classes")
    return int(np.flatnonzero(close_one)[0])


class So2SatSourceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset whose row list itself carries source label-access authority."""

    def __init__(
        self,
        container: SourceContainer,
        rows: Sequence[SourceRow],
        normalizer: BandNormalizer,
        *,
        expected_role: str,
        augmentation_seed: int | None = None,
    ) -> None:
        if expected_role not in SOURCE_ROLES:
            raise IntegrityError("source dataset role must be source_train or source_monitor")
        self.container = container
        self.rows = tuple(rows)
        self.normalizer = normalizer
        self.expected_role = expected_role
        self.augmentation_seed = augmentation_seed
        self.epoch = 0
        if not self.rows or any(row.sample_role != expected_role for row in self.rows):
            raise IntegrityError(f"dataset rows do not exclusively belong to {expected_role}")
        if expected_role == SOURCE_MONITOR_ROLE and augmentation_seed is not None:
            raise IntegrityError("source_monitor evaluation must not use random augmentation")

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise IntegrityError("dataset epoch must be a non-negative integer")
        self.epoch = epoch

    def set_augmentation_seed(self, seed: int) -> None:
        if self.expected_role != SOURCE_TRAIN_ROLE:
            raise IntegrityError("only source_train may receive an augmentation seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise IntegrityError("augmentation seed must be a non-negative integer")
        self.augmentation_seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def _augment(self, tensor: torch.Tensor, row_index: int) -> torch.Tensor:
        if self.augmentation_seed is None:
            return tensor
        payload = f"{self.augmentation_seed}\x00{self.epoch}\x00{row_index}".encode("ascii")
        choices = hashlib.sha256(payload).digest()
        tensor = torch.rot90(tensor, k=choices[0] % 4, dims=(1, 2))
        if choices[1] & 1:
            tensor = torch.flip(tensor, dims=(2,))
        if choices[1] & 2:
            tensor = torch.flip(tensor, dims=(1,))
        return tensor.contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        if row.sample_role != self.expected_role:
            raise IntegrityError("source dataset row authority changed after construction")
        pixels, label = self.container.read_labeled(row.row_index)
        array = np.asarray(pixels, dtype=np.float32)
        if array.shape != (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS):
            raise IntegrityError("source Sentinel-2 sample is not 32x32x10")
        if not np.isfinite(array).all():
            raise IntegrityError("source Sentinel-2 sample contains NaN or Infinity")
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = self.normalizer.apply(tensor)
        tensor = self._augment(tensor, row.row_index)
        target = torch.tensor(_one_hot_class(label), dtype=torch.long)
        return tensor, target, torch.tensor(row.row_index, dtype=torch.long)

    def __getitems__(
        self, indices: Sequence[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch HDF5 reads while preserving the sampler's deterministic order."""

        selected = [self.rows[int(index)] for index in indices]
        if not selected or any(row.sample_role != self.expected_role for row in selected):
            raise IntegrityError("source dataset batch contains an unauthorized role")
        row_indices = [row.row_index for row in selected]
        pixels, labels = self.container.read_labeled_many(row_indices)
        expected_pixels = (len(selected), INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
        if pixels.shape != expected_pixels or labels.shape != (len(selected), NUM_CLASSES):
            raise IntegrityError("source labeled batch shape drift")
        if not np.isfinite(pixels).all():
            raise IntegrityError("source Sentinel-2 batch contains NaN or Infinity")
        output: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for row, image, label in zip(selected, pixels, labels, strict=True):
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).contiguous()
            tensor = self.normalizer.apply(tensor)
            tensor = self._augment(tensor, row.row_index)
            output.append(
                (
                    tensor,
                    torch.tensor(_one_hot_class(label), dtype=torch.long),
                    torch.tensor(row.row_index, dtype=torch.long),
                )
            )
        return output


@dataclass(frozen=True)
class SourceDataBundle:
    inventory: SourceRoleInventory
    container: SourceContainer
    normalizer: BandNormalizer
    source_train: So2SatSourceDataset
    source_monitor: So2SatSourceDataset
    data_identity_sha256: str


def build_source_data_bundle(
    inventory: SourceRoleInventory,
    container: SourceContainer,
    normalizer: BandNormalizer,
    *,
    augmentation_seed: int,
) -> SourceDataBundle:
    if normalizer.source_rows_sha256 != inventory.source_rows_sha256:
        raise IntegrityError("normalizer was fit for a different source role inventory")
    if normalizer.source_container_identity_sha256 != container.identity_sha256:
        raise IntegrityError("normalizer was fit for a different source container")
    if normalizer.source_train_n != len(inventory.source_train_rows):
        raise IntegrityError("normalizer source_train count differs from the role inventory")
    identity = {
        "inventory": inventory.identity(),
        "source_container_identity": dict(container.identity),
        "source_container_identity_sha256": container.identity_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "label_access_roles": [SOURCE_TRAIN_ROLE, SOURCE_MONITOR_ROLE],
        "optimization_role": SOURCE_TRAIN_ROLE,
        "checkpoint_selection_role": SOURCE_MONITOR_ROLE,
        "target_split_paths": [],
    }
    return SourceDataBundle(
        inventory=inventory,
        container=container,
        normalizer=normalizer,
        source_train=So2SatSourceDataset(
            container,
            inventory.source_train_rows,
            normalizer,
            expected_role=SOURCE_TRAIN_ROLE,
            augmentation_seed=augmentation_seed,
        ),
        source_monitor=So2SatSourceDataset(
            container,
            inventory.source_monitor_rows,
            normalizer,
            expected_role=SOURCE_MONITOR_ROLE,
            augmentation_seed=None,
        ),
        data_identity_sha256=stable_sha256(identity),
    )


def synthetic_source_inventory(
    source_train_indices: Iterable[int],
    source_monitor_indices: Iterable[int],
    *,
    population_n: int,
    identity_tag: str = "synthetic_smoke_v1",
) -> SourceRoleInventory:
    """Construct an explicit fake inventory without weakening production reads."""

    train = tuple(
        SourceRow(index, f"training:{index:06d}", "synthetic", f"synthetic:{index}", SOURCE_TRAIN_ROLE)
        for index in sorted(source_train_indices)
    )
    monitor = tuple(
        SourceRow(
            index,
            f"training:{index:06d}",
            "synthetic",
            f"synthetic:{index}",
            SOURCE_MONITOR_ROLE,
        )
        for index in sorted(source_monitor_indices)
    )
    rows_hash = ordered_records_sha256(row.commitment() for row in (*train, *monitor))
    return SourceRoleInventory(
        population_identity_sha256=stable_sha256({"identity_tag": identity_tag, "kind": "population"}),
        population_manifest_sha256=stable_sha256({"identity_tag": identity_tag, "kind": "manifest"}),
        training_geo_sha256=stable_sha256({"identity_tag": identity_tag, "kind": "training_geo"}),
        training_population_n=population_n,
        source_train_rows=train,
        source_monitor_rows=monitor,
        source_rows_sha256=rows_hash,
    )
