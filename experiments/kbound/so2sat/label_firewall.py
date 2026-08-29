"""Target-pixel interfaces that cannot request an So2Sat outcome dataset.

The target HDF5 containers co-locate pixels and outcomes.  This module therefore
has no arbitrary dataset-name API: callers choose one of two declared sensor
modalities, and the implementation indexes only that fixed pixel dataset.  Safe
geographic metadata comes from the separately hashed ``*_geo.h5`` files.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .integrity import IntegrityError, LabelFirewallError, file_sha256, require_sha256
from .metadata_manifest import (
    DEFAULT_CITY_ALLOCATION,
    GEO_BASENAMES,
    SPLITS,
    TARGET_CITY_IDS,
    TRAINING_CITY_IDS,
    CityAllocationContract,
    GeoRecord,
    H5Factory,
    assign_training_city_roles,
    development_easting_thresholds,
    inspect_training_geography,
    iter_geo_records,
    read_geo_record,
    validate_population_manifest,
)
from .protocol import OFFICIAL_SPLIT_COUNTS

TARGET_SPLITS = ("validation", "testing")
PIXEL_DATASET_BY_MODALITY = {
    "sen1_8_band": ("sen1", (32, 32, 8)),
    "sen2_10_band": ("sen2", (32, 32, 10)),
}
TARGET_DATA_BASENAMES = {split: f"{split}.h5" for split in TARGET_SPLITS}


def _default_h5_factory(path: Path) -> AbstractContextManager[Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on research environment
        raise RuntimeError("So2Sat pixel loading requires h5py") from exc
    return h5py.File(path, "r")


@dataclass(frozen=True)
class PixelSample:
    """One live target sample; by design it has pixels and safe metadata only."""

    pixels: Any
    metadata: GeoRecord

    def safe_metadata(self) -> dict[str, Any]:
        return self.metadata.as_dict()


class VerifiedTrainingGeoIndex:
    """Training-only index that never receives a target geo or data path."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        training_geo_path: str | Path,
        *,
        h5_factory: H5Factory | None = None,
        expected_split_counts: Mapping[str, int] = OFFICIAL_SPLIT_COUNTS,
        expected_training_cities: frozenset[str] = TRAINING_CITY_IDS,
        expected_target_cities: frozenset[str] = TARGET_CITY_IDS,
        allocation_contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
    ) -> None:
        validate_population_manifest(
            manifest,
            expected_split_counts=expected_split_counts,
            expected_training_cities=expected_training_cities,
            expected_target_cities=expected_target_cities,
            allocation_contract=allocation_contract,
        )
        self._manifest = dict(manifest)
        self._uses_canonical_h5_factory = h5_factory is None
        self._training_path = Path(training_geo_path).expanduser().resolve()
        self._factory = h5_factory
        self._expected_counts = dict(expected_split_counts)
        if not self._training_path.is_file():
            raise FileNotFoundError(f"missing training geo file: {self._training_path}")
        identity = self._manifest["splits"]["training"]["geo_artifact"]
        if self._training_path.name != GEO_BASENAMES["training"]:
            raise IntegrityError(
                f"training geo file must be named {GEO_BASENAMES['training']!r}"
            )
        if self._training_path.stat().st_size != identity["bytes"]:
            raise IntegrityError("training geo file byte count changed")
        if file_sha256(self._training_path) != identity["sha256"]:
            raise IntegrityError("training geo file SHA-256 changed")
        observed_geography = inspect_training_geography(
            self._training_path,
            h5_factory=self._factory,
        )
        sealed_geography = self._manifest["cities"]["training_geography"]
        if observed_geography != sealed_geography:
            raise IntegrityError("training geography differs from the sealed metadata inventory")
        self._city_roles = assign_training_city_roles(
            observed_geography["city_counts"],
            observed_geography["city_distinct_block_easting_counts"],
            contract=allocation_contract,
        )
        self._easting_thresholds = development_easting_thresholds(
            self._city_roles,
            observed_geography,
            allocation_contract=allocation_contract,
        )
        if self._easting_thresholds != self._manifest["cities"]["development_easting_thresholds"]:
            raise IntegrityError("recomputed development easting thresholds differ from the seal")

    @property
    def population_identity_sha256(self) -> str:
        return require_sha256(
            self._manifest.get("population_identity_sha256"),
            field="population_identity_sha256",
        )

    @property
    def uses_canonical_h5_factory(self) -> bool:
        """True only when no injectable HDF5 factory was supplied."""

        return self._uses_canonical_h5_factory

    def record(self, row_index: int) -> GeoRecord:
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise IntegrityError("row_index must be an integer")
        if not 0 <= row_index < self._expected_counts["training"]:
            raise IntegrityError(
                f"row_index {row_index} is outside the sealed training population"
            )
        return read_geo_record(
            "training",
            self._training_path,
            row_index,
            city_roles=self._city_roles,
            development_easting_thresholds=self._easting_thresholds,
            h5_factory=self._factory,
        )

    def iter_records(self) -> Iterator[GeoRecord]:
        """Iterate the sealed training metadata with one safe HDF5 open."""

        return iter_geo_records(
            "training",
            self._training_path,
            city_roles=self._city_roles,
            development_easting_thresholds=self._easting_thresholds,
            h5_factory=self._factory,
        )


class VerifiedGeoIndex(VerifiedTrainingGeoIndex):
    """Content-bound access to all three label-free v4.2 geo populations."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        geo_paths: Mapping[str, str | Path],
        *,
        h5_factory: H5Factory | None = None,
        expected_split_counts: Mapping[str, int] = OFFICIAL_SPLIT_COUNTS,
        expected_training_cities: frozenset[str] = TRAINING_CITY_IDS,
        expected_target_cities: frozenset[str] = TARGET_CITY_IDS,
        allocation_contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
    ) -> None:
        if set(geo_paths) != set(SPLITS):
            raise IntegrityError(f"geo_paths must contain exactly {list(SPLITS)}")
        paths = {split: Path(geo_paths[split]).expanduser().resolve() for split in SPLITS}
        super().__init__(
            manifest,
            paths["training"],
            h5_factory=h5_factory,
            expected_split_counts=expected_split_counts,
            expected_training_cities=expected_training_cities,
            expected_target_cities=expected_target_cities,
            allocation_contract=allocation_contract,
        )
        self._paths = paths
        self._target_cities = set(expected_target_cities)
        for split in TARGET_SPLITS:
            path = self._paths[split]
            if not path.is_file():
                raise FileNotFoundError(f"missing {split} geo file: {path}")
            identity = self._manifest["splits"][split]["geo_artifact"]
            if path.name != GEO_BASENAMES[split]:
                raise IntegrityError(f"{split} geo file must be named {GEO_BASENAMES[split]!r}")
            if path.stat().st_size != identity["bytes"]:
                raise IntegrityError(f"{split} geo file byte count changed")
            if file_sha256(path) != identity["sha256"]:
                raise IntegrityError(f"{split} geo file SHA-256 changed")

    def record(self, split: str, row_index: int) -> GeoRecord:
        if split == "training":
            return super().record(row_index)
        if split not in TARGET_SPLITS:
            raise IntegrityError(f"unknown So2Sat split: {split!r}")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise IntegrityError("row_index must be an integer")
        if not 0 <= row_index < self._expected_counts[split]:
            raise IntegrityError(f"row_index {row_index} is outside the sealed {split} population")
        record = read_geo_record(
            split,
            self._paths[split],
            row_index,
            city_roles=self._city_roles,
            development_easting_thresholds=self._easting_thresholds,
            h5_factory=self._factory,
        )
        if record.city_id not in self._target_cities:
            raise IntegrityError(f"target metadata returned undeclared city {record.city_id!r}")
        return record

    def iter_records(self, split: str = "training") -> Iterator[GeoRecord]:
        """Iterate one verified split with a single safe geo-container open."""

        if split == "training":
            yield from super().iter_records()
            return
        if split not in TARGET_SPLITS:
            raise IntegrityError(f"unknown So2Sat split: {split!r}")
        for record in iter_geo_records(
            split,
            self._paths[split],
            city_roles=self._city_roles,
            development_easting_thresholds=self._easting_thresholds,
            h5_factory=self._factory,
        ):
            if record.city_id not in self._target_cities:
                raise IntegrityError(
                    f"target metadata returned undeclared city {record.city_id!r}"
                )
            yield record


class LabelFreeTargetLoader:
    """Load target pixels only after opaque container identities are verified."""

    def __init__(
        self,
        geo_index: VerifiedGeoIndex,
        data_paths: Mapping[str, str | Path],
        expected_data_identities: Mapping[str, Mapping[str, Any]],
        *,
        modality: str = "sen2_10_band",
        h5_factory: H5Factory | None = None,
        expected_split_counts: Mapping[str, int] = OFFICIAL_SPLIT_COUNTS,
    ) -> None:
        if modality not in PIXEL_DATASET_BY_MODALITY:
            raise LabelFirewallError(
                f"unsupported target modality {modality!r}; allowed={sorted(PIXEL_DATASET_BY_MODALITY)}"
            )
        if set(data_paths) != set(TARGET_SPLITS) or set(expected_data_identities) != set(TARGET_SPLITS):
            raise IntegrityError(f"target data paths/identities must contain exactly {list(TARGET_SPLITS)}")
        self._geo_index = geo_index
        self._paths = {
            split: Path(data_paths[split]).expanduser().resolve() for split in TARGET_SPLITS
        }
        self._identities = {split: dict(expected_data_identities[split]) for split in TARGET_SPLITS}
        self._modality = modality
        self._uses_canonical_h5_factory = h5_factory is None
        self._factory = _default_h5_factory if h5_factory is None else h5_factory
        self._expected_counts = {split: expected_split_counts[split] for split in TARGET_SPLITS}
        self._verified = False
        self._access_log: list[dict[str, Any]] = []
        for split, path in self._paths.items():
            if path.name != TARGET_DATA_BASENAMES[split]:
                raise IntegrityError(f"{split} target file must be named {TARGET_DATA_BASENAMES[split]!r}")
            identity = self._identities[split]
            if isinstance(identity.get("bytes"), bool) or not isinstance(identity.get("bytes"), int):
                raise IntegrityError(f"{split} target identity has an invalid byte count")
            if identity["bytes"] < 1:
                raise IntegrityError(f"{split} target identity has an empty byte count")
            require_sha256(identity.get("sha256"), field=f"{split}.sha256")

    @property
    def access_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._access_log)

    @property
    def uses_canonical_h5_factory(self) -> bool:
        """True only for the fixed production h5py path."""

        return self._uses_canonical_h5_factory

    def verify_containers(self) -> dict[str, Any]:
        """Hash raw container bytes without opening or deserializing HDF5 datasets."""

        rows = []
        for split in TARGET_SPLITS:
            path = self._paths[split]
            if not path.is_file():
                raise FileNotFoundError(f"missing target pixel container: {path}")
            expected = self._identities[split]
            if path.stat().st_size != expected["bytes"]:
                raise IntegrityError(f"{split} target container byte count changed")
            observed = file_sha256(path)
            if observed != expected["sha256"]:
                raise IntegrityError(f"{split} target container SHA-256 changed")
            rows.append(
                {
                    "split": split,
                    "basename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": observed,
                    "hdf5_datasets_deserialized": False,
                }
            )
        self._verified = True
        return {
            "schema": "kbound_so2sat_opaque_target_container_verification_v1",
            "population_identity_sha256": self._geo_index.population_identity_sha256,
            "containers": rows,
        }

    def read(self, split: str, row_index: int) -> PixelSample:
        return self.read_many(split, [row_index])[0]

    def _validate_split(self, split: str) -> tuple[str, tuple[int, ...]]:
        if not self._verified:
            raise IntegrityError("target containers must be hash-verified before pixel access")
        if split not in TARGET_SPLITS:
            raise LabelFirewallError(
                f"live target loader permits only {list(TARGET_SPLITS)}, found {split!r}"
            )
        return PIXEL_DATASET_BY_MODALITY[self._modality]

    def read_verified_many(
        self,
        split: str,
        records: Sequence[GeoRecord],
    ) -> list[PixelSample]:
        """Read a verified metadata batch with one fixed-key HDF5 open.

        The live runner obtains ``records`` from ``VerifiedGeoIndex.iter_records``;
        this method revalidates every safe field before using the row indices.
        It never accepts a dataset name from the caller.
        """

        dataset_name, trailing_shape = self._validate_split(split)
        if not records:
            raise IntegrityError("read_verified_many requires at least one metadata record")
        row_indices = [record.row_index for record in records]
        if (
            any(isinstance(index, bool) or not isinstance(index, int) for index in row_indices)
            or row_indices != sorted(set(row_indices))
            or any(not 0 <= index < self._expected_counts[split] for index in row_indices)
        ):
            raise IntegrityError("verified target metadata batch has invalid row indices")
        required_role = "target_probe" if split == "validation" else "target_evaluation"
        if any(
            record.official_split != split
            or record.city_role != "target"
            or record.sample_role != required_role
            for record in records
        ):
            raise LabelFirewallError(
                "safe metadata does not match the target probe/evaluation contract"
            )
        pixels_by_row: list[Any] = []
        with self._factory(self._paths[split]) as handle:
            # Deliberately do not enumerate this handle: the official container
            # co-locates outcomes.  Exactly one fixed sensor dataset is indexed.
            dataset = handle[dataset_name]
            shape = getattr(dataset, "shape", None)
            expected_shape = (self._expected_counts[split], *trailing_shape)
            if shape != expected_shape:
                raise IntegrityError(
                    f"{split}/{dataset_name} shape drift: expected {expected_shape}, found {shape}"
                )
            for row_index in row_indices:
                pixels_by_row.append(dataset[row_index])
        samples: list[PixelSample] = []
        for metadata, pixels in zip(records, pixels_by_row, strict=True):
            self._access_log.append(
                {
                    "split": split,
                    "row_index": metadata.row_index,
                    "dataset": dataset_name,
                    "target_outcome_dataset_accessed": False,
                }
            )
            samples.append(PixelSample(pixels=pixels, metadata=metadata))
        return samples

    def read_many(self, split: str, row_indices: Sequence[int]) -> list[PixelSample]:
        self._validate_split(split)
        if not row_indices:
            raise IntegrityError("read_many requires at least one row index")
        if len(set(row_indices)) != len(row_indices):
            raise IntegrityError("read_many refuses duplicate row indices")
        records = [self._geo_index.record(split, row_index) for row_index in row_indices]
        return self.read_verified_many(split, records)
