"""Synthetic contract tests for the label-free geo-index public API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from experiments.kbound.so2sat import label_firewall
from experiments.kbound.so2sat.integrity import IntegrityError
from experiments.kbound.so2sat.label_firewall import VerifiedGeoIndex
from experiments.kbound.so2sat.metadata_manifest import GeoRecord


def _synthetic_record(split: str, row_index: int) -> GeoRecord:
    return GeoRecord(
        sample_id=f"{split}-{row_index}",
        official_split=split,
        row_index=row_index,
        city_id="synthetic-target-city" if split != "training" else "synthetic-training-city",
        epsg=4326,
        tfw=(1.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        spatial_block_id="synthetic-block",
        spatial_block_easting=0,
        spatial_block_northing=0,
        city_role="synthetic-city-role",
        sample_role="synthetic-sample-role",
    )


def _call_record_with_arguments(index: VerifiedGeoIndex, arguments: tuple[object, ...]) -> GeoRecord:
    return cast(Callable[..., GeoRecord], index.record)(*arguments)


def _call_record_with_keywords(index: VerifiedGeoIndex, **keywords: object) -> GeoRecord:
    return cast(Callable[..., GeoRecord], index.record)(**keywords)


def _call_record_with_mixed_arguments(
    index: VerifiedGeoIndex,
    arguments: tuple[object, ...],
    **keywords: object,
) -> GeoRecord:
    return cast(Callable[..., GeoRecord], index.record)(*arguments, **keywords)


@pytest.fixture
def synthetic_geo_index(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]]:
    """Build an uninitialized index whose record reads remain entirely synthetic."""

    index = object.__new__(VerifiedGeoIndex)
    index._expected_counts = {"training": 2, "validation": 2, "testing": 2}
    index._training_path = Path("/synthetic/training_geo.h5")
    index._paths = {
        "training": Path("/synthetic/training_geo.h5"),
        "validation": Path("/synthetic/validation_geo.h5"),
        "testing": Path("/synthetic/testing_geo.h5"),
    }
    index._city_roles = {}
    index._easting_thresholds = {}
    index._factory = None
    index._target_cities = {"synthetic-target-city"}
    reads: list[tuple[str, Path, int]] = []

    def read_synthetic_geo_record(
        split: str,
        geo_path: str | Path,
        row_index: int,
        **_kwargs: Any,
    ) -> GeoRecord:
        reads.append((split, Path(geo_path), row_index))
        return _synthetic_record(split, row_index)

    monkeypatch.setattr(label_firewall, "read_geo_record", read_synthetic_geo_record)
    return index, reads


def test_record_accepts_a_single_training_index(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
) -> None:
    index, reads = synthetic_geo_index

    record = index.record(1)

    assert record == _synthetic_record("training", 1)
    assert reads == [("training", Path("/synthetic/training_geo.h5"), 1)]


def test_record_accepts_a_named_split_and_index(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
) -> None:
    index, reads = synthetic_geo_index

    record = index.record("validation", 1)

    assert record == _synthetic_record("validation", 1)
    assert reads == [("validation", Path("/synthetic/validation_geo.h5"), 1)]


def test_record_accepts_base_compatible_keyword_row_index(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
) -> None:
    index, reads = synthetic_geo_index

    record = index.record(row_index=1)

    assert record == _synthetic_record("training", 1)
    assert reads == [("training", Path("/synthetic/training_geo.h5"), 1)]


def test_record_accepts_named_split_keyword_form(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
) -> None:
    index, reads = synthetic_geo_index

    record = index.record(split="testing", row_index=1)

    assert record == _synthetic_record("testing", 1)
    assert reads == [("testing", Path("/synthetic/testing_geo.h5"), 1)]


@pytest.mark.parametrize(
    ("split", "path"),
    [
        ("training", Path("/synthetic/training_geo.h5")),
        ("validation", Path("/synthetic/validation_geo.h5")),
    ],
)
def test_record_accepts_positional_split_with_keyword_row_index(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    split: str,
    path: Path,
) -> None:
    index, reads = synthetic_geo_index

    record = index.record(split, row_index=1)

    assert record == _synthetic_record(split, 1)
    assert reads == [(split, path, 1)]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((), "row_index is required"),
        (("validation",), "row_index is required for a named So2Sat split"),
        ((None, 1), "record accepts either an index or a (split, index) pair"),
    ],
)
def test_record_rejects_missing_or_undeclared_argument_shapes(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    arguments: tuple[object, ...],
    error: str,
) -> None:
    index, reads = synthetic_geo_index

    with pytest.raises(IntegrityError, match=re.escape(error)):
        _call_record_with_arguments(index, arguments)

    assert reads == []


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((1, 0), "split must be text"),
        (("development", 0), "unknown So2Sat split"),
        (("validation", True), "row_index must be an integer"),
    ],
)
def test_record_rejects_invalid_split_or_index_values(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    arguments: tuple[object, ...],
    error: str,
) -> None:
    index, reads = synthetic_geo_index

    with pytest.raises(IntegrityError, match=re.escape(error)):
        _call_record_with_arguments(index, arguments)

    assert reads == []


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ((True,), "row_index must be an integer"),
        ((1.5,), "row_index must be an integer"),
        ((None,), "row_index must be an integer"),
        ((2,), "outside the sealed training population"),
        (("testing", 2), "outside the sealed testing population"),
    ],
)
def test_record_rejects_invalid_single_argument_or_bounds(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    arguments: tuple[object, ...],
    error: str,
) -> None:
    index, reads = synthetic_geo_index

    with pytest.raises(IntegrityError, match=re.escape(error)):
        _call_record_with_arguments(index, arguments)

    assert reads == []


@pytest.mark.parametrize(
    ("keywords", "error"),
    [
        ({"split": 1}, "split must be text"),
        ({"split": None, "row_index": 1}, "record accepts either an index or a (split, index) pair"),
        ({"split": "validation"}, "row_index is required for a named So2Sat split"),
        ({"row_index": None}, "row_index must be an integer"),
        ({"split": "testing", "row_index": None}, "row_index is required for a named So2Sat split"),
    ],
)
def test_record_rejects_invalid_keyword_shapes_or_values(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    keywords: dict[str, object],
    error: str,
) -> None:
    index, reads = synthetic_geo_index

    with pytest.raises(IntegrityError, match=re.escape(error)):
        _call_record_with_keywords(index, **keywords)

    assert reads == []


@pytest.mark.parametrize(
    ("arguments", "keywords", "error"),
    [
        ((1,), {"row_index": 1}, "record accepts either an index or a (split, index) pair"),
        (("validation",), {"split": "testing"}, "record accepts either an index or a (split, index) pair"),
        (("validation",), {"row_index": None}, "row_index is required for a named So2Sat split"),
    ],
)
def test_record_rejects_malformed_mixed_forms(
    synthetic_geo_index: tuple[VerifiedGeoIndex, list[tuple[str, Path, int]]],
    arguments: tuple[object, ...],
    keywords: dict[str, object],
    error: str,
) -> None:
    index, reads = synthetic_geo_index

    with pytest.raises(IntegrityError, match=re.escape(error)):
        _call_record_with_mixed_arguments(index, arguments, **keywords)

    assert reads == []
