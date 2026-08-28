"""Source-only and data-quality tests for the So2Sat HDF5 preflight."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.metadata_manifest import (
    GEO_BASENAMES,
    CityAllocationContract,
    assign_sample_role,
    assign_training_city_roles,
    build_population_manifest,
    spatial_block_coordinates,
    spatial_block_id,
)
from experiments.kbound.so2sat.source_preflight import (
    SAMPLE_ROLES,
    audit_source_training_h5,
    require_source_training_path,
)


class _Dataset:
    def __init__(self, values: np.ndarray[Any, Any]) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.reads: list[Any] = []

    def __getitem__(self, index: Any) -> Any:
        self.reads.append(index)
        return self.values[index]


class _Handle:
    def __init__(self, datasets: Mapping[str, _Dataset]) -> None:
        self.datasets = dict(datasets)
        self.accessed: list[str] = []

    def __enter__(self) -> "_Handle":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def keys(self) -> list[str]:
        return list(self.datasets)

    def __getitem__(self, key: str) -> _Dataset:
        self.accessed.append(key)
        return self.datasets[key]


class _Factory:
    def __init__(self, handles: Mapping[Path, _Handle]) -> None:
        self.handles = {path.resolve(): handle for path, handle in handles.items()}
        self.opened: list[Path] = []

    def __call__(self, path: Path) -> _Handle:
        resolved = path.resolve()
        self.opened.append(resolved)
        if resolved not in self.handles:
            raise AssertionError(f"unexpected HDF5 open: {resolved}")
        return self.handles[resolved]


def _geo_handle(cities: list[str], tfw: list[list[float]]) -> _Handle:
    count = len(cities)
    return _Handle(
        {
            "city": _Dataset(
                np.asarray([city.encode("utf-8") for city in cities]).reshape(count, 1)
            ),
            "epsg": _Dataset(np.full((count, 1), 32632, dtype=np.int32)),
            "tfw": _Dataset(np.asarray(tfw, dtype=np.float64)),
        }
    )


def _candidate_tfw(easting: int, northing: int = 312) -> list[float]:
    return [
        10.0,
        0.0,
        0.0,
        -10.0,
        float(easting * 6_400),
        float(northing * 6_400 + 3_200),
    ]


def _select_source_tfw(city: str, role: str, count: int, cursor: int) -> list[list[float]]:
    monitor: list[list[float]] = []
    train: list[list[float]] = []
    for easting in range(cursor, cursor + 10_000):
        tfw = _candidate_tfw(easting)
        block_id = spatial_block_id(32632, tfw)
        _, block_easting, _ = spatial_block_coordinates(32632, tfw)
        assigned = assign_sample_role(
            split="training",
            city_role=role,
            city_id=city,
            block_id=block_id,
            block_easting=block_easting,
            development_easting_thresholds={},
        )
        (monitor if assigned == "source_monitor" else train).append(tfw)
        if monitor and len(train) >= count:
            break
    if not monitor or not train:
        raise AssertionError("test fixture could not find both source-monitor roles")
    if count == 1:
        return train[:1]
    return [monitor[0], *train[: count - 1]]


def _synthetic_inputs(
    tmp_path: Path,
    *,
    invalid_label: bool = False,
    extra_training_key: bool = False,
) -> tuple[Path, Path, Path, _Factory, dict[str, _Dataset]]:
    data_root = tmp_path / "source_data"
    data_root.mkdir(parents=True)
    city_counts = {"tiny": 1, "core": 12, "gatealpha": 9, "gatebeta": 8}
    allocation = CityAllocationContract(
        minimum_eligible_rows=2,
        expected_ineligible_city_count=1,
        source_fit_core_count=1,
        gate_fit_count=1,
        gate_cal_count=1,
    )
    roles = assign_training_city_roles(
        city_counts,
        {city: count for city, count in city_counts.items()},
        contract=allocation,
    )
    training_cities: list[str] = []
    training_tfw: list[list[float]] = []
    cursor = 1_000
    for city in sorted(city_counts):
        count = city_counts[city]
        role = roles[city]
        if role in {"source_fit_ineligible", "source_fit_core"}:
            tfw = _select_source_tfw(city, role, count, cursor)
        else:
            tfw = [_candidate_tfw(cursor + offset) for offset in range(count)]
        training_cities.extend([city] * count)
        training_tfw.extend(tfw)
        cursor += 20_000
    target_city = "targetcity"
    geo_paths = {
        split: data_root / GEO_BASENAMES[split]
        for split in ("training", "validation", "testing")
    }
    for split, path in geo_paths.items():
        path.write_bytes(f"synthetic-safe-geo-{split}".encode("ascii"))
    geo_handles = {
        geo_paths["training"]: _geo_handle(training_cities, training_tfw),
        geo_paths["validation"]: _geo_handle([target_city], [_candidate_tfw(91_000)]),
        geo_paths["testing"]: _geo_handle([target_city], [_candidate_tfw(92_000)]),
    }
    geo_factory = _Factory(geo_handles)
    row_count = sum(city_counts.values())
    manifest = build_population_manifest(
        geo_paths,
        h5_factory=geo_factory,
        expected_split_counts={"training": row_count, "validation": 1, "testing": 1},
        expected_training_cities=frozenset(city_counts),
        expected_target_cities=frozenset({target_city}),
        allocation_contract=allocation,
    )
    assert set(manifest["splits"]["training"]["sample_role_counts"]) == set(SAMPLE_ROLES)
    manifest_path = tmp_path / "population_manifest.json"
    write_immutable_json_with_receipt(manifest_path, manifest)

    training_path = data_root / "training.h5"
    training_path.write_bytes(b"synthetic-source-container-v1")
    sen1 = np.arange(row_count * 32 * 32 * 8, dtype=np.float32).reshape(row_count, 32, 32, 8)
    sen2 = (
        np.arange(row_count * 32 * 32 * 10, dtype=np.float32).reshape(row_count, 32, 32, 10)
        * np.float32(0.5)
        + np.float32(1.0)
    )
    labels = np.zeros((row_count, 17), dtype=np.float32)
    labels[np.arange(row_count), np.arange(row_count) % 17] = 1.0
    if invalid_label:
        labels[0] = 0.0
    datasets = {
        "sen1": _Dataset(sen1),
        "sen2": _Dataset(sen2),
        "label": _Dataset(labels),
    }
    if extra_training_key:
        datasets["outcome_copy"] = _Dataset(labels.copy())
    audit_factory = _Factory(
        {
            geo_paths["training"]: geo_handles[geo_paths["training"]],
            training_path: _Handle(datasets),
        }
    )
    return training_path, manifest_path, tmp_path / "source_preflight.json", audit_factory, datasets


def test_source_preflight_is_chunked_role_stratified_and_target_closed(tmp_path: Path) -> None:
    training, manifest, output, factory, datasets = _synthetic_inputs(tmp_path)
    result = audit_source_training_h5(
        training,
        manifest,
        output,
        chunk_rows=3,
        duplicate_sample_rows=7,
        h5_factory=factory,
    )
    document = result["document"]
    assert document["quality_gate"]["ready_for_source_training"] is True
    assert document["scope"]["target_image_containers_opened"] is False
    assert document["scope"]["target_outcome_arrays_opened"] is False
    assert document["training_geo_alignment"]["status"] == "VERIFIED"
    assert set(document["datasets"]["label"]["class_counts_by_sample_role"]) == set(SAMPLE_ROLES)
    assert document["datasets"]["label"]["one_hot_valid_rate"] == 1.0
    assert document["configuration_sha256"]
    assert document["code_identity"]["aggregate_sha256"]
    assert document["training_container_identity"]["sha256"]
    assert set(factory.opened) == {
        training.resolve(),
        training.with_name("training_geo.h5").resolve(),
    }
    for name in ("sen1", "sen2", "label"):
        reads = datasets[name].reads
        assert reads
        assert all(isinstance(read, slice) for read in reads)
        assert max(read.stop - read.start for read in reads) <= 3
    verify_artifact_receipt(
        output,
        receipt_schema="kbound_so2sat_source_data_preflight_receipt_v1",
    )


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        (Path("validation") / "training.h5", "target split token"),
        (Path("testing") / "training.h5", "target split token"),
        (Path("source") / "validation.h5", "accepts only 'training.h5'"),
        (Path("source") / "testing.h5", "accepts only 'training.h5'"),
    ],
)
def test_source_path_contract_explicitly_refuses_target_names_and_paths(
    tmp_path: Path,
    relative: Path,
    message: str,
) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"must-not-open")
    with pytest.raises(IntegrityError, match=message):
        require_source_training_path(path)


def test_invalid_one_hot_labels_create_an_immutable_blocking_receipt(tmp_path: Path) -> None:
    training, manifest, output, factory, _ = _synthetic_inputs(tmp_path, invalid_label=True)
    result = audit_source_training_h5(
        training,
        manifest,
        output,
        chunk_rows=5,
        h5_factory=factory,
    )
    assert result["document"]["status"] == "SOURCE_DATA_PREFLIGHT_BLOCKED"
    assert result["document"]["quality_gate"]["ready_for_source_training"] is False
    assert any(
        "one-hot" in finding
        for finding in result["document"]["quality_gate"]["critical_findings"]
    )
    with pytest.raises(IntegrityError, match="refusing to overwrite"):
        audit_source_training_h5(
            training,
            manifest,
            output,
            chunk_rows=5,
            h5_factory=factory,
        )


def test_source_preflight_rejects_training_schema_drift(tmp_path: Path) -> None:
    training, manifest, output, factory, _ = _synthetic_inputs(tmp_path, extra_training_key=True)
    with pytest.raises(IntegrityError, match="must contain exactly"):
        audit_source_training_h5(
            training,
            manifest,
            output,
            chunk_rows=8,
            h5_factory=factory,
        )
    assert not output.exists()
