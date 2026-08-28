"""Integrity tests for the prospective, target-label-blind So2Sat path."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    LabelFirewallError,
    file_sha256,
    stable_sha256,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.label_firewall import (
    LabelFreeTargetLoader,
    VerifiedGeoIndex,
    VerifiedTrainingGeoIndex,
)
from experiments.kbound.so2sat.metadata_manifest import (
    GEO_BASENAMES,
    SOURCE_MONITOR_BLOCK_SALT,
    CityAllocationContract,
    GeoRecord,
    assign_sample_role,
    assign_training_city_roles,
    build_population_manifest,
    development_easting_thresholds,
    iter_geo_records,
    normalize_city,
    normalize_epsg,
    spatial_block_id,
    validate_population_manifest,
)
from experiments.kbound.so2sat.protocol import (
    load_protocol,
    protocol_identity,
    verify_checked_in_protocol_receipt,
)


class _FakeDataset:
    def __init__(self, values: list[Any], *, shape: tuple[int, ...] | None = None) -> None:
        self.values = values
        self.shape = (len(values),) if shape is None else shape

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


class _FakeGeoHandle:
    def __init__(self, datasets: dict[str, _FakeDataset]) -> None:
        self.datasets = datasets
        self.accessed: list[str] = []

    def __enter__(self) -> "_FakeGeoHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def keys(self) -> list[str]:
        return list(self.datasets)

    def __getitem__(self, key: str) -> _FakeDataset:
        self.accessed.append(key)
        return self.datasets[key]


class _FakeTargetHandle:
    def __init__(self, split_count: int) -> None:
        self.accessed: list[str] = []
        self.datasets = {
            "sen2": _FakeDataset(
                [("pixels", index) for index in range(split_count)],
                shape=(split_count, 32, 32, 10),
            ),
            "label": _FakeDataset([[1, 0] for _ in range(split_count)], shape=(split_count, 17)),
        }

    def __enter__(self) -> "_FakeTargetHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def keys(self) -> list[str]:
        raise AssertionError("the target container must never be enumerated")

    def __getitem__(self, key: str) -> _FakeDataset:
        self.accessed.append(key)
        if key == "label":
            raise AssertionError("the target outcome array was accessed")
        return self.datasets[key]


class _Factory:
    def __init__(self, handles: dict[Path, Any]) -> None:
        self.handles = {path.resolve(): handle for path, handle in handles.items()}

    def __call__(self, path: Path) -> Any:
        return self.handles[path.resolve()]


def _geo_handle(cities: list[str], *, block_offset: int = 0) -> _FakeGeoHandle:
    epsg = [32632 for _ in cities]
    tfw = [
        [10.0, 0.0, 0.0, -10.0, float((index + block_offset) * 6_400), 2_000_000.0]
        for index in range(len(cities))
    ]
    return _FakeGeoHandle(
        {
            "city": _FakeDataset([city.encode("utf-8") for city in cities], shape=(len(cities), 1)),
            "epsg": _FakeDataset(epsg, shape=(len(cities), 1)),
            "tfw": _FakeDataset(tfw, shape=(len(cities), 6)),
        }
    )


def _mini_population(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    _Factory,
    dict[str, int],
    frozenset[str],
    frozenset[str],
    CityAllocationContract,
]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    city_counts = {
        "smallalpha": 10,
        "smallbeta": 12,
        "unsplittable": 50,
        "core": 100,
        "eligibleone": 80,
        "eligibletwo": 70,
        "eligiblethree": 60,
    }
    training_cities = frozenset(city_counts)
    target_cities = frozenset({"delta", "epsilon"})
    allocation = CityAllocationContract(
        minimum_eligible_rows=20,
        expected_ineligible_city_count=3,
        source_fit_core_count=1,
        gate_fit_count=1,
        gate_cal_count=2,
    )
    distinct_easting_counts = {
        city: 1 if city == "unsplittable" else count for city, count in city_counts.items()
    }
    allocated = assign_training_city_roles(
        city_counts,
        distinct_easting_counts,
        contract=allocation,
    )
    training_values: list[str] = []
    training_tfw: list[list[float]] = []
    cursor = 0
    for city in sorted(training_cities):
        count = city_counts[city]
        candidate_blocks = (
            [(cursor, 300 + index) for index in range(2_000)]
            if city == "unsplittable"
            else [(cursor + index, 312) for index in range(2_000)]
        )
        if allocated[city] in {"source_fit_ineligible", "source_fit_core"}:
            monitor = [
                block
                for block in candidate_blocks
                if assign_sample_role(
                    split="training",
                    city_role=allocated[city],
                    city_id=city,
                    block_id=f"32632:{block[0]}:{block[1]}",
                    block_easting=block[0],
                    development_easting_thresholds={},
                )
                == "source_monitor"
            ][0]
            train = [
                block
                for block in candidate_blocks
                if assign_sample_role(
                    split="training",
                    city_role=allocated[city],
                    city_id=city,
                    block_id=f"32632:{block[0]}:{block[1]}",
                    block_easting=block[0],
                    development_easting_thresholds={},
                )
                == "source_train"
            ]
            selected = [monitor, *train[: count - 1]]
        else:
            selected = candidate_blocks[:count]
        training_values.extend([city] * count)
        training_tfw.extend(
            [
                10.0,
                0.0,
                0.0,
                -10.0,
                float(easting * 6_400),
                float(northing * 6_400 + 3_200),
            ]
            for easting, northing in selected
        )
        cursor += 3_000
    handles: dict[Path, Any] = {}
    paths: dict[str, Path] = {}
    for split in ("training", "validation", "testing"):
        path = tmp_path / GEO_BASENAMES[split]
        path.write_bytes(f"safe-geographic-metadata-{split}".encode())
        paths[split] = path
    training_count = sum(city_counts.values())
    handles[paths["training"]] = _FakeGeoHandle(
        {
            "city": _FakeDataset([city.encode() for city in training_values], shape=(training_count, 1)),
            "epsg": _FakeDataset([32632] * training_count, shape=(training_count, 1)),
            "tfw": _FakeDataset(training_tfw, shape=(training_count, 6)),
        }
    )
    handles[paths["validation"]] = _geo_handle(sorted(target_cities))
    handles[paths["testing"]] = _geo_handle(sorted(target_cities), block_offset=10)
    expected_counts = {"training": training_count, "validation": 2, "testing": 2}
    factory = _Factory(handles)
    manifest = build_population_manifest(
        paths,
        h5_factory=factory,
        expected_split_counts=expected_counts,
        expected_training_cities=training_cities,
        expected_target_cities=target_cities,
        allocation_contract=allocation,
    )
    return (
        manifest,
        paths,
        factory,
        expected_counts,
        training_cities,
        target_cities,
        allocation,
    )


def test_protocol_and_checked_in_receipt_are_valid_but_disclose_pending_execution_seal() -> None:
    protocol = load_protocol()
    identity = protocol_identity()
    receipt = verify_checked_in_protocol_receipt()
    assert protocol["status"] == "STRUCTURAL_PROTOCOL_SEALED_EXECUTION_CONFIG_PENDING"
    assert protocol["target_label_firewall"]["live_objects_expose_labels"] is False
    assert protocol["roles"]["target"]["probe_split"] == "validation"
    assert protocol["roles"]["target"]["evaluation_split"] == "testing"
    assert protocol["dataset"]["split_counts"] == {
        "training": 352366,
        "validation": 24119,
        "testing": 24188,
    }
    assert protocol["roles"]["training_city_partition"]["role_counts"] == {
        "source_fit_ineligible": 9,
        "source_fit_core": 5,
        "gate_fit": 9,
        "gate_cal": 19,
    }
    assert receipt["artifact_sha256"] == identity["file_sha256"]
    assert receipt["canonical_document_sha256"] == identity["canonical_document_sha256"]


def test_city_and_spatial_partitions_are_order_invariant_and_label_free() -> None:
    counts = {"tiny": 3, "spatial": 10, "core": 30, "gateone": 20, "gatetwo": 10}
    easting_counts = {city: 1 if city == "spatial" else 2 for city in counts}
    allocation = CityAllocationContract(
        minimum_eligible_rows=5,
        expected_ineligible_city_count=2,
        source_fit_core_count=1,
        gate_fit_count=1,
        gate_cal_count=1,
    )
    first = assign_training_city_roles(counts, easting_counts, contract=allocation)
    second = assign_training_city_roles(
        dict(reversed(list(counts.items()))),
        dict(reversed(list(easting_counts.items()))),
        contract=allocation,
    )
    assert first == second
    assert Counter(first.values()) == Counter(allocation.role_counts())
    assert first["tiny"] == "source_fit_ineligible"
    assert first["spatial"] == "source_fit_ineligible"
    assert first["core"] == "source_fit_core"
    assert counts[next(city for city, role in first.items() if role == "gate_fit")] >= 5
    assert counts[next(city for city, role in first.items() if role == "gate_cal")] >= 5
    with pytest.raises(IntegrityError, match="same cities"):
        assign_training_city_roles(counts, {"tiny": 2}, contract=allocation)
    with pytest.raises(IntegrityError, match="at least two sorted distinct block eastings"):
        development_easting_thresholds(
            {"spatial": "gate_fit"},
            {
                "city_counts": {"spatial": 10},
                "city_distinct_block_eastings": {"spatial": [7]},
            },
            allocation_contract=allocation,
        )
    assert normalize_city(b"San Francisco") == "sanfrancisco"
    with pytest.raises(IntegrityError, match="EPSG must be an integer"):
        normalize_epsg(True)
    block = spatial_block_id(32610, [10.0, 0.0, 0.0, -10.0, 6400.0, 12800.0])
    assert block == "32610:1:1"
    development_role = assign_sample_role(
        split="training",
        city_role="gate_fit",
        city_id="alpha",
        block_id=block,
        block_easting=1,
        development_easting_thresholds={"alpha": 2},
    )
    monitor_role = assign_sample_role(
        split="training",
        city_role="source_fit_core",
        city_id="alpha",
        block_id=block,
        block_easting=1,
        development_easting_thresholds={},
    )
    assert development_role in {"gate_fit_probe", "gate_fit_evaluation"}
    assert monitor_role in {"source_train", "source_monitor"}
    assert stable_sha256("median-easting") != stable_sha256(SOURCE_MONITOR_BLOCK_SALT)


def test_metadata_manifest_covers_population_and_binds_deterministic_roles(tmp_path: Path) -> None:
    manifest, _, _, counts, training, target, roles = _mini_population(tmp_path)
    validate_population_manifest(
        manifest,
        expected_split_counts=counts,
        expected_training_cities=training,
        expected_target_cities=target,
        allocation_contract=roles,
    )
    assert manifest["access_contract"] == {
        "opened_files": ["training_geo.h5", "validation_geo.h5", "testing_geo.h5"],
        "allowed_hdf5_datasets": ["city", "epsg", "tfw"],
        "image_containers_opened": False,
        "target_outcome_arrays_opened": False,
        "target_outcome_arrays_counted": False,
        "target_outcome_arrays_hashed": False,
    }
    assert manifest["splits"]["validation"]["sample_role_counts"] == {"target_probe": 2}
    assert manifest["splits"]["testing"]["sample_role_counts"] == {"target_evaluation": 2}
    assert set(manifest["cities"]["training_roles"]) == {
        "source_fit_ineligible",
        "source_fit_core",
        "gate_fit",
        "gate_cal",
    }
    city_counts = manifest["cities"]["training_geography"]["city_counts"]
    easting_counts = manifest["cities"]["training_geography"]["city_distinct_block_easting_counts"]
    assert all(
        city_counts[city] < roles.minimum_eligible_rows or easting_counts[city] < 2
        for city in manifest["cities"]["training_roles"]["source_fit_ineligible"]
    )
    assert all(
        city_counts[city] >= roles.minimum_eligible_rows and easting_counts[city] >= 2
        for role in ("gate_fit", "gate_cal")
        for city in manifest["cities"]["training_roles"][role]
    )
    assert manifest["cities"]["source_fit_ineligible_reasons"]["unsplittable"] == [
        "fewer_than_two_distinct_block_eastings"
    ]
    tampered = copy.deepcopy(manifest)
    tampered["splits"]["testing"]["observed_samples"] -= 1
    with pytest.raises(IntegrityError, match="manifest hash mismatch"):
        validate_population_manifest(
            tampered,
            expected_split_counts=counts,
            expected_training_cities=training,
            expected_target_cities=target,
            allocation_contract=roles,
        )


def test_geo_reader_rejects_extra_outcome_dataset_without_indexing_it(tmp_path: Path) -> None:
    path = tmp_path / "validation_geo.h5"
    path.write_bytes(b"fake")
    handle = _geo_handle(["delta"])
    handle.datasets["label"] = _FakeDataset([[1, 0]], shape=(1, 17))
    with pytest.raises(IntegrityError, match="must contain exactly"):
        list(
            iter_geo_records(
                "validation",
                path,
                city_roles={},
                development_easting_thresholds={},
                h5_factory=_Factory({path: handle}),
            )
        )
    assert "label" not in handle.accessed


def test_training_only_index_needs_no_target_path_and_reconstructs_sealed_roles(
    tmp_path: Path,
) -> None:
    manifest, geo_paths, factory, counts, training, target, allocation = _mini_population(tmp_path)
    index = VerifiedTrainingGeoIndex(
        manifest,
        geo_paths["training"],
        h5_factory=factory,
        expected_split_counts=counts,
        expected_training_cities=training,
        expected_target_cities=target,
        allocation_contract=allocation,
    )
    records = list(index.iter_records())
    assert len(records) == counts["training"]
    roles = {record.sample_role for record in records}
    assert roles == {
        "source_train",
        "source_monitor",
        "gate_fit_probe",
        "gate_fit_evaluation",
        "gate_cal_probe",
        "gate_cal_evaluation",
    }
    thresholds = manifest["cities"]["development_easting_thresholds"]
    block_roles: dict[str, set[str]] = {}
    for record in records:
        block_roles.setdefault(record.spatial_block_id, set()).add(record.sample_role)
        if record.city_role in {"gate_fit", "gate_cal"}:
            side = record.sample_role.rsplit("_", 1)[-1]
            if side == "probe":
                assert record.spatial_block_easting < thresholds[record.city_id]
            else:
                assert record.spatial_block_easting >= thresholds[record.city_id]
    assert all(len(assigned) == 1 for assigned in block_roles.values())


def test_live_target_loader_never_enumerates_or_indexes_outcomes(tmp_path: Path) -> None:
    manifest, geo_paths, geo_factory, counts, training, target, roles = _mini_population(tmp_path / "geo")
    index = VerifiedGeoIndex(
        manifest,
        geo_paths,
        h5_factory=geo_factory,
        expected_split_counts=counts,
        expected_training_cities=training,
        expected_target_cities=target,
        allocation_contract=roles,
    )
    data_paths: dict[str, Path] = {}
    target_handles: dict[Path, _FakeTargetHandle] = {}
    identities: dict[str, dict[str, Any]] = {}
    for split in ("validation", "testing"):
        path = tmp_path / f"{split}.h5"
        path.write_bytes(f"opaque-target-container-{split}".encode())
        data_paths[split] = path
        target_handles[path] = _FakeTargetHandle(counts[split])
        identities[split] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    loader = LabelFreeTargetLoader(
        index,
        data_paths,
        identities,
        h5_factory=_Factory(target_handles),
        expected_split_counts=counts,
    )
    with pytest.raises(IntegrityError, match="hash-verified"):
        loader.read("validation", 0)
    receipt = loader.verify_containers()
    assert all(row["hdf5_datasets_deserialized"] is False for row in receipt["containers"])
    probe = loader.read("validation", 0)
    evaluation = loader.read("testing", 1)
    assert probe.pixels == ("pixels", 0)
    assert evaluation.pixels == ("pixels", 1)
    assert probe.metadata.sample_role == "target_probe"
    assert evaluation.metadata.sample_role == "target_evaluation"
    assert set(probe.safe_metadata()).isdisjoint({"label", "labels", "target", "y", "accuracy"})
    assert target_handles[data_paths["validation"]].accessed == ["sen2"]
    assert target_handles[data_paths["testing"]].accessed == ["sen2"]
    assert {row["dataset"] for row in loader.access_log} == {"sen2"}
    assert all(row["target_outcome_dataset_accessed"] is False for row in loader.access_log)
    with pytest.raises(LabelFirewallError, match="permits only"):
        loader.read("training", 0)
    with pytest.raises(LabelFirewallError, match="unsupported target modality"):
        LabelFreeTargetLoader(
            index,
            data_paths,
            identities,
            modality="label",
            h5_factory=_Factory(target_handles),
            expected_split_counts=counts,
        )


def test_geo_index_detects_byte_tampering_before_record_access(tmp_path: Path) -> None:
    manifest, paths, factory, counts, training, target, roles = _mini_population(tmp_path)
    paths["testing"].write_bytes(b"mutated-safe-metadata")
    with pytest.raises(IntegrityError, match="byte count changed|SHA-256 changed"):
        VerifiedGeoIndex(
            manifest,
            paths,
            h5_factory=factory,
            expected_split_counts=counts,
            expected_training_cities=training,
            expected_target_cities=target,
            allocation_contract=roles,
        )


def test_immutable_json_receipt_is_create_only_and_detects_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(destination, {"schema": "test", "value": 1})
    assert verify_artifact_receipt(destination) == receipt
    with pytest.raises(IntegrityError, match="refusing to overwrite"):
        write_immutable_json_with_receipt(destination, {"schema": "test", "value": 2})
    destination.chmod(0o644)
    destination.write_text('{"schema":"test","value":2}\n', encoding="utf-8")
    with pytest.raises(IntegrityError, match="byte count mismatch|file SHA-256 mismatch"):
        verify_artifact_receipt(destination)


def test_public_live_sample_type_has_no_outcome_field() -> None:
    record = GeoRecord(
        sample_id="testing:000000",
        official_split="testing",
        row_index=0,
        city_id="delta",
        epsg=32632,
        tfw=(10.0, 0.0, 0.0, -10.0, 0.0, 0.0),
        spatial_block_id="32632:0:-1",
        spatial_block_easting=0,
        spatial_block_northing=-1,
        city_role="target",
        sample_role="target_evaluation",
    )
    keys = set(record.as_dict())
    assert keys == {
        "sample_id",
        "official_split",
        "row_index",
        "city_id",
        "epsg",
        "tfw",
        "spatial_block_id",
        "spatial_block_easting",
        "spatial_block_northing",
        "city_role",
        "sample_role",
    }
