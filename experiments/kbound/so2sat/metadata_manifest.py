#!/usr/bin/env python3
"""Build the So2Sat v4.2 population manifest from geolocation files only.

The official ``*_geo.h5`` files contain only ``city``, ``epsg``, and ``tfw``.
This module refuses any other HDF5 dataset and never opens ``training.h5``,
``validation.h5``, or ``testing.h5``.  The manifest binds the complete ordered
population, deterministic city roles, and spatial-block roles without reading
an outcome array.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .integrity import (
    IntegrityError,
    canonical_json_bytes,
    file_sha256,
    require_sha256,
    stable_sha256,
    write_immutable_json_with_receipt,
)
from .protocol import (
    GATE_CITY_SALT,
    MINIMUM_GATE_CITY_ROWS,
    OFFICIAL_GEO_KEYS,
    OFFICIAL_SPLIT_COUNTS,
    PROTOCOL_ID,
    protocol_identity,
    verify_checked_in_protocol_receipt,
)


SCHEMA = "kbound_so2sat_label_free_population_manifest_v1"
SPLITS = ("training", "validation", "testing")
GEO_BASENAMES = {split: f"{split}_geo.h5" for split in SPLITS}
TRAINING_CITY_IDS = frozenset(
    {
        "amsterdam",
        "beijing",
        "berlin",
        "bogota",
        "buenosaires",
        "cairo",
        "capetown",
        "caracas",
        "changsha",
        "chicago",
        "cologne",
        "dhaka",
        "dongying",
        "hongkong",
        "islamabad",
        "istanbul",
        "kyoto",
        "lima",
        "lisbon",
        "london",
        "losangeles",
        "madrid",
        "melbourne",
        "milan",
        "nanjing",
        "newyork",
        "orangitown",
        "paris",
        "philadelphia",
        "qingdao",
        "quezon",
        "riodejaneiro",
        "rome",
        "salvador",
        "saopaulo",
        "shanghai",
        "shenzhen",
        "tokyo",
        "vancouver",
        "washingtondc",
        "wuhan",
        "zurich",
    }
)
TARGET_CITY_IDS = frozenset(
    {
        "guangzhou",
        "jakarta",
        "moscow",
        "mumbai",
        "munich",
        "nairobi",
        "sanfrancisco",
        "santiago",
        "sydney",
        "tehran",
    }
)
SOURCE_MONITOR_BLOCK_SALT = "KBOUND_SO2SAT_SOURCE_MONITOR_BLOCK_ROLES_v1"
BLOCK_METRES = 6_400

H5Factory = Callable[[Path], AbstractContextManager[Any]]


@dataclass(frozen=True)
class CityAllocationContract:
    """Count-aware training-city allocation fixed before target outcomes."""

    minimum_eligible_rows: int = MINIMUM_GATE_CITY_ROWS
    expected_ineligible_city_count: int = 9
    source_fit_core_count: int = 5
    gate_fit_count: int = 9
    gate_cal_count: int = 19
    gate_salt: str = GATE_CITY_SALT

    def role_counts(self) -> dict[str, int]:
        return {
            "source_fit_ineligible": self.expected_ineligible_city_count,
            "source_fit_core": self.source_fit_core_count,
            "gate_fit": self.gate_fit_count,
            "gate_cal": self.gate_cal_count,
        }


DEFAULT_CITY_ALLOCATION = CityAllocationContract()


def _default_h5_factory(path: Path) -> AbstractContextManager[Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on research environment
        raise RuntimeError("So2Sat metadata preparation requires h5py") from exc
    return h5py.File(path, "r")


def _singleton(value: Any, *, field: str) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise IntegrityError(f"{field} must be scalar or singleton-shaped")
        value = value[0]
        if hasattr(value, "tolist"):
            value = value.tolist()
    if hasattr(value, "item"):
        value = value.item()
    return value


def normalize_city(value: Any) -> str:
    value = _singleton(value, field="city")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise IntegrityError("city contains invalid UTF-8") from exc
    if not isinstance(value, str):
        raise IntegrityError(f"city must be text, found {type(value).__name__}")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if not normalized or any(unicodedata.category(character).startswith("C") for character in normalized):
        raise IntegrityError(f"city contains empty/control text: {value!r}")
    city_id = "".join(character for character in normalized if character.isalnum())
    if not city_id:
        raise IntegrityError(f"city has no alphanumeric identity: {value!r}")
    return city_id


def normalize_epsg(value: Any) -> int:
    value = _singleton(value, field="epsg")
    if isinstance(value, bool):
        raise IntegrityError(f"EPSG must be an integer, found {value!r}")
    if not isinstance(value, int):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IntegrityError(f"EPSG must be an integer, found {value!r}") from exc
        if converted != value:
            raise IntegrityError(f"EPSG must be integral, found {value!r}")
        value = converted
    if value <= 0 or value > 999_999:
        raise IntegrityError(f"EPSG is outside the supported positive range: {value}")
    return value


def normalize_tfw(value: Any) -> tuple[float, float, float, float, float, float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise IntegrityError("TFW metadata must contain six world-file parameters")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError("TFW metadata must be numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise IntegrityError("TFW metadata contains NaN or Infinity")
    a, d, b, e, _, _ = result
    if not (0.0 < abs(a) <= 100.0 and 0.0 < abs(e) <= 100.0):
        raise IntegrityError(f"TFW pixel scales are implausible: A={a}, E={e}")
    if abs(b) > 1.0 or abs(d) > 1.0:
        raise IntegrityError(f"TFW rotation terms are implausible: B={b}, D={d}")
    return result


def spatial_block_coordinates(epsg: int, tfw: Sequence[float]) -> tuple[int, int, int]:
    """Return EPSG and the 6.4-km block containing the patch centre."""

    if len(tfw) != 6:
        raise IntegrityError("spatial block construction requires six TFW values")
    a, d, b, e, c, f = (float(value) for value in tfw)
    center_x = c + 16.0 * a + 16.0 * b
    center_y = f + 16.0 * d + 16.0 * e
    return int(epsg), math.floor(center_x / BLOCK_METRES), math.floor(center_y / BLOCK_METRES)


def spatial_block_id(epsg: int, tfw: Sequence[float]) -> str:
    """Return the stable string identity of one 6.4-km spatial block."""

    block_epsg, block_easting, block_northing = spatial_block_coordinates(epsg, tfw)
    return f"{block_epsg}:{block_easting}:{block_northing}"


def _hash_is_below_fraction(payload: str, fraction: float) -> bool:
    if not 0.0 < fraction < 1.0:
        raise IntegrityError("partition fraction must lie strictly between zero and one")
    rational = Fraction(str(float(fraction)))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")
    return value * rational.denominator < rational.numerator * (1 << 256)


def assign_training_city_roles(
    city_counts: Mapping[str, int],
    distinct_block_easting_counts: Mapping[str, int],
    *,
    contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
) -> dict[str, str]:
    if not city_counts:
        raise IntegrityError("training-city allocation requires nonempty metadata counts")
    if set(distinct_block_easting_counts) != set(city_counts):
        raise IntegrityError("city allocation easting counts must cover the same cities as row counts")
    for city, count in city_counts.items():
        if not isinstance(city, str) or not city:
            raise IntegrityError("training-city count keys must be nonempty normalized strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise IntegrityError(f"training city {city!r} has an invalid population count")
        easting_count = distinct_block_easting_counts[city]
        if isinstance(easting_count, bool) or not isinstance(easting_count, int) or easting_count < 1:
            raise IntegrityError(f"training city {city!r} has an invalid distinct-easting count")
    expected_total = sum(contract.role_counts().values())
    if len(city_counts) != expected_total:
        raise IntegrityError(
            f"city allocation expects {expected_total} cities, found {len(city_counts)}"
        )
    ineligible = sorted(
        city
        for city, count in city_counts.items()
        if count < contract.minimum_eligible_rows or distinct_block_easting_counts[city] < 2
    )
    if len(ineligible) != contract.expected_ineligible_city_count:
        raise IntegrityError(
            f"expected exactly {contract.expected_ineligible_city_count} cities failing row/spatial "
            f"eligibility, found {len(ineligible)}"
        )
    eligible = [city for city in city_counts if city not in set(ineligible)]
    core = sorted(eligible, key=lambda city: (-city_counts[city], city))[: contract.source_fit_core_count]
    gate_candidates = sorted(set(eligible) - set(core))
    expected_gate_candidates = contract.gate_fit_count + contract.gate_cal_count
    if len(gate_candidates) != expected_gate_candidates:
        raise IntegrityError(
            f"expected {expected_gate_candidates} eligible gate candidates, found {len(gate_candidates)}"
        )
    if any(
        city_counts[city] < contract.minimum_eligible_rows
        or distinct_block_easting_counts[city] < 2
        for city in gate_candidates
    ):
        raise IntegrityError("a gate candidate fails row-count or spatial-splittability eligibility")
    gate_ranked = sorted(
        gate_candidates,
        key=lambda city: (
            hashlib.sha256(f"{contract.gate_salt}\x00{city}".encode("utf-8")).hexdigest(),
            city,
        ),
    )
    gate_fit = gate_ranked[: contract.gate_fit_count]
    gate_cal = gate_ranked[contract.gate_fit_count :]
    assignments = {city: "source_fit_ineligible" for city in ineligible}
    assignments.update({city: "source_fit_core" for city in core})
    assignments.update({city: "gate_fit" for city in gate_fit})
    assignments.update({city: "gate_cal" for city in gate_cal})
    if Counter(assignments.values()) != Counter(contract.role_counts()):
        raise IntegrityError("count-aware city allocation failed its role-count contract")
    return assignments


def training_city_ineligibility_reasons(
    city_counts: Mapping[str, int],
    distinct_block_easting_counts: Mapping[str, int],
    *,
    contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
) -> dict[str, list[str]]:
    if set(city_counts) != set(distinct_block_easting_counts):
        raise IntegrityError("ineligibility inputs must cover identical city sets")
    reasons: dict[str, list[str]] = {}
    for city in sorted(city_counts):
        city_reasons = []
        if city_counts[city] < contract.minimum_eligible_rows:
            city_reasons.append(f"rows_below_minimum_{contract.minimum_eligible_rows}")
        if distinct_block_easting_counts[city] < 2:
            city_reasons.append("fewer_than_two_distinct_block_eastings")
        if city_reasons:
            reasons[city] = city_reasons
    return reasons


def assign_sample_role(
    *,
    split: str,
    city_role: str,
    city_id: str,
    block_id: str,
    block_easting: int,
    development_easting_thresholds: Mapping[str, int],
) -> str:
    if split == "validation":
        return "target_probe"
    if split == "testing":
        return "target_evaluation"
    if split != "training":
        raise IntegrityError(f"unknown So2Sat split: {split!r}")
    if city_role in {"source_fit_ineligible", "source_fit_core"}:
        payload = f"{SOURCE_MONITOR_BLOCK_SALT}\x00{city_id}\x00{block_id}"
        return "source_monitor" if _hash_is_below_fraction(payload, 0.10) else "source_train"
    if city_role in {"gate_fit", "gate_cal"}:
        if city_id not in development_easting_thresholds:
            raise IntegrityError(f"gate city {city_id!r} has no sealed median easting threshold")
        threshold = development_easting_thresholds[city_id]
        suffix = "probe" if block_easting < threshold else "evaluation"
        return f"{city_role}_{suffix}"
    raise IntegrityError(f"unknown training city role: {city_role!r}")


@dataclass(frozen=True)
class GeoRecord:
    """The complete metadata available to prospective loaders."""

    sample_id: str
    official_split: str
    row_index: int
    city_id: str
    epsg: int
    tfw: tuple[float, float, float, float, float, float]
    spatial_block_id: str
    spatial_block_easting: int
    spatial_block_northing: int
    city_role: str
    sample_role: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "official_split": self.official_split,
            "row_index": self.row_index,
            "city_id": self.city_id,
            "epsg": self.epsg,
            "tfw": list(self.tfw),
            "spatial_block_id": self.spatial_block_id,
            "spatial_block_easting": self.spatial_block_easting,
            "spatial_block_northing": self.spatial_block_northing,
            "city_role": self.city_role,
            "sample_role": self.sample_role,
        }


def _dataset_length(dataset: Any, *, field: str) -> int:
    shape = getattr(dataset, "shape", None)
    if not isinstance(shape, tuple) or not shape:
        raise IntegrityError(f"geo dataset {field!r} must expose a nonempty tuple shape")
    count = shape[0]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise IntegrityError(f"geo dataset {field!r} has an invalid leading dimension")
    return count


def iter_geo_records(
    split: str,
    geo_path: str | Path,
    *,
    city_roles: Mapping[str, str],
    development_easting_thresholds: Mapping[str, int],
    h5_factory: H5Factory | None = None,
) -> Iterator[GeoRecord]:
    """Yield safe metadata rows while refusing any non-geographic dataset."""

    if split not in SPLITS:
        raise IntegrityError(f"unknown So2Sat split: {split!r}")
    source = Path(geo_path)
    factory = _default_h5_factory if h5_factory is None else h5_factory
    with factory(source) as handle:
        keys = {str(key) for key in handle.keys()}
        if keys != set(OFFICIAL_GEO_KEYS):
            raise IntegrityError(
                f"{source.name} must contain exactly {list(OFFICIAL_GEO_KEYS)}, found {sorted(keys)}"
            )
        city_dataset = handle["city"]
        epsg_dataset = handle["epsg"]
        tfw_dataset = handle["tfw"]
        counts = {
            "city": _dataset_length(city_dataset, field="city"),
            "epsg": _dataset_length(epsg_dataset, field="epsg"),
            "tfw": _dataset_length(tfw_dataset, field="tfw"),
        }
        if len(set(counts.values())) != 1:
            raise IntegrityError(f"geo dataset leading dimensions do not align: {counts}")
        for row_index in range(counts["city"]):
            yield _build_geo_record(
                split=split,
                row_index=row_index,
                city_value=city_dataset[row_index],
                epsg_value=epsg_dataset[row_index],
                tfw_value=tfw_dataset[row_index],
                city_roles=city_roles,
                development_easting_thresholds=development_easting_thresholds,
            )


def _build_geo_record(
    *,
    split: str,
    row_index: int,
    city_value: Any,
    epsg_value: Any,
    tfw_value: Any,
    city_roles: Mapping[str, str],
    development_easting_thresholds: Mapping[str, int],
) -> GeoRecord:
    city_id = normalize_city(city_value)
    epsg = normalize_epsg(epsg_value)
    tfw = normalize_tfw(tfw_value)
    _, block_easting, block_northing = spatial_block_coordinates(epsg, tfw)
    block_id = f"{epsg}:{block_easting}:{block_northing}"
    if split == "training":
        if city_id not in city_roles:
            raise IntegrityError(f"training metadata contains undeclared city {city_id!r}")
        city_role = city_roles[city_id]
    elif split in {"validation", "testing"}:
        city_role = "target"
    else:
        raise IntegrityError(f"unknown So2Sat split: {split!r}")
    return GeoRecord(
        sample_id=f"{split}:{row_index:06d}",
        official_split=split,
        row_index=row_index,
        city_id=city_id,
        epsg=epsg,
        tfw=tfw,
        spatial_block_id=block_id,
        spatial_block_easting=block_easting,
        spatial_block_northing=block_northing,
        city_role=city_role,
        sample_role=assign_sample_role(
            split=split,
            city_role=city_role,
            city_id=city_id,
            block_id=block_id,
            block_easting=block_easting,
            development_easting_thresholds=development_easting_thresholds,
        ),
    )


def read_geo_record(
    split: str,
    geo_path: str | Path,
    row_index: int,
    *,
    city_roles: Mapping[str, str],
    development_easting_thresholds: Mapping[str, int],
    h5_factory: H5Factory | None = None,
) -> GeoRecord:
    """Read one safe metadata row from a verified v4.2 geolocation file."""

    if split not in SPLITS:
        raise IntegrityError(f"unknown So2Sat split: {split!r}")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise IntegrityError("row_index must be a non-negative integer")
    source = Path(geo_path)
    factory = _default_h5_factory if h5_factory is None else h5_factory
    with factory(source) as handle:
        keys = {str(key) for key in handle.keys()}
        if keys != set(OFFICIAL_GEO_KEYS):
            raise IntegrityError(
                f"{source.name} must contain exactly {list(OFFICIAL_GEO_KEYS)}, found {sorted(keys)}"
            )
        datasets = {field: handle[field] for field in OFFICIAL_GEO_KEYS}
        counts = {field: _dataset_length(dataset, field=field) for field, dataset in datasets.items()}
        if len(set(counts.values())) != 1:
            raise IntegrityError(f"geo dataset leading dimensions do not align: {counts}")
        if row_index >= counts["city"]:
            raise IntegrityError(f"row_index {row_index} is outside {split} population {counts['city']}")
        return _build_geo_record(
            split=split,
            row_index=row_index,
            city_value=datasets["city"][row_index],
            epsg_value=datasets["epsg"][row_index],
            tfw_value=datasets["tfw"][row_index],
            city_roles=city_roles,
            development_easting_thresholds=development_easting_thresholds,
        )


def inspect_training_geography(
    geo_path: str | Path,
    *,
    h5_factory: H5Factory | None = None,
) -> dict[str, Any]:
    """Count cities and collect distinct eastings using safe metadata only."""

    source = Path(geo_path)
    factory = _default_h5_factory if h5_factory is None else h5_factory
    city_counts: Counter[str] = Counter()
    city_eastings: dict[str, set[int]] = defaultdict(set)
    city_epsg: dict[str, set[int]] = defaultdict(set)
    with factory(source) as handle:
        keys = {str(key) for key in handle.keys()}
        if keys != set(OFFICIAL_GEO_KEYS):
            raise IntegrityError(
                f"{source.name} must contain exactly {list(OFFICIAL_GEO_KEYS)}, found {sorted(keys)}"
            )
        datasets = {field: handle[field] for field in OFFICIAL_GEO_KEYS}
        counts = {field: _dataset_length(dataset, field=field) for field, dataset in datasets.items()}
        if len(set(counts.values())) != 1:
            raise IntegrityError(f"geo dataset leading dimensions do not align: {counts}")
        for row_index in range(counts["city"]):
            city = normalize_city(datasets["city"][row_index])
            epsg = normalize_epsg(datasets["epsg"][row_index])
            tfw = normalize_tfw(datasets["tfw"][row_index])
            _, block_easting, _ = spatial_block_coordinates(epsg, tfw)
            city_counts[city] += 1
            city_eastings[city].add(block_easting)
            city_epsg[city].add(epsg)
    for city, values in city_epsg.items():
        if len(values) != 1:
            raise IntegrityError(
                f"training city {city!r} spans multiple EPSG systems, so easting is not directly comparable: {sorted(values)}"
            )
    return {
        "row_count": sum(city_counts.values()),
        "city_counts": dict(sorted(city_counts.items())),
        "city_distinct_block_eastings": {
            city: sorted(values) for city, values in sorted(city_eastings.items())
        },
        "city_distinct_block_easting_counts": {
            city: len(values) for city, values in sorted(city_eastings.items())
        },
        "city_epsg": {city: next(iter(values)) for city, values in sorted(city_epsg.items())},
    }


def development_easting_thresholds(
    city_roles: Mapping[str, str],
    training_geography: Mapping[str, Any],
    *,
    allocation_contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
) -> dict[str, int]:
    """Use the upper median distinct block easting as each gate city's cut."""

    city_counts = training_geography.get("city_counts")
    easting_rows = training_geography.get("city_distinct_block_eastings")
    if not isinstance(city_counts, Mapping) or not isinstance(easting_rows, Mapping):
        raise IntegrityError("training geography lacks city counts/eastings")
    output: dict[str, int] = {}
    for city, role in city_roles.items():
        if role not in {"gate_fit", "gate_cal"}:
            continue
        count = city_counts.get(city)
        if isinstance(count, bool) or not isinstance(count, int) or count < allocation_contract.minimum_eligible_rows:
            raise IntegrityError(
                f"gate city {city!r} has {count!r} rows; minimum is {allocation_contract.minimum_eligible_rows}"
            )
        eastings = easting_rows.get(city)
        if (
            not isinstance(eastings, list)
            or len(eastings) < 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in eastings)
            or eastings != sorted(set(eastings))
        ):
            raise IntegrityError(
                f"gate city {city!r} needs at least two sorted distinct block eastings"
            )
        threshold = eastings[len(eastings) // 2]
        if not any(value < threshold for value in eastings) or not any(value >= threshold for value in eastings):
            raise IntegrityError(f"gate city {city!r} median easting does not create two roles")
        output[city] = threshold
    expected = {
        city for city, role in city_roles.items() if role in {"gate_fit", "gate_cal"}
    }
    if set(output) != expected:
        raise IntegrityError("development easting thresholds do not cover every gate city")
    return dict(sorted(output.items()))


def _digest_update(digest: Any, row: Mapping[str, Any]) -> None:
    digest.update(canonical_json_bytes(dict(row)))
    digest.update(b"\n")


def _required_roles_for_city(city_role: str) -> set[str]:
    if city_role in {"source_fit_ineligible", "source_fit_core"}:
        return {"source_train", "source_monitor"}
    if city_role == "gate_fit":
        return {"gate_fit_probe", "gate_fit_evaluation"}
    if city_role == "gate_cal":
        return {"gate_cal_probe", "gate_cal_evaluation"}
    if city_role == "target":
        return {"target_probe", "target_evaluation"}
    raise IntegrityError(f"unknown city role {city_role!r}")


def build_population_manifest(
    geo_paths: Mapping[str, str | Path],
    *,
    h5_factory: H5Factory | None = None,
    expected_split_counts: Mapping[str, int] = OFFICIAL_SPLIT_COUNTS,
    expected_training_cities: frozenset[str] = TRAINING_CITY_IDS,
    expected_target_cities: frozenset[str] = TARGET_CITY_IDS,
    allocation_contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
    require_official_basenames: bool = True,
) -> dict[str, Any]:
    """Scan all safe geo rows and produce a compact, content-bound manifest."""

    if set(geo_paths) != set(SPLITS):
        raise IntegrityError(f"geo_paths must contain exactly {list(SPLITS)}")
    if set(expected_split_counts) != set(SPLITS):
        raise IntegrityError(f"expected_split_counts must contain exactly {list(SPLITS)}")
    if expected_training_cities & expected_target_cities:
        raise IntegrityError("source and target city sets must be disjoint")
    paths = {split: Path(geo_paths[split]).expanduser().resolve() for split in SPLITS}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} geolocation file: {path}")
        if require_official_basenames and path.name != GEO_BASENAMES[split]:
            raise IntegrityError(f"{split} geo file must be named {GEO_BASENAMES[split]!r}")

    training_geography = inspect_training_geography(
        paths["training"],
        h5_factory=h5_factory,
    )
    if training_geography["row_count"] != expected_split_counts["training"]:
        raise IntegrityError(
            "training geography preflight count mismatch: "
            f"expected {expected_split_counts['training']}, found {training_geography['row_count']}"
        )
    if set(training_geography["city_counts"]) != set(expected_training_cities):
        raise IntegrityError("training geography preflight city set mismatch")
    city_roles = assign_training_city_roles(
        training_geography["city_counts"],
        training_geography["city_distinct_block_easting_counts"],
        contract=allocation_contract,
    )
    ineligibility_reasons = training_city_ineligibility_reasons(
        training_geography["city_counts"],
        training_geography["city_distinct_block_easting_counts"],
        contract=allocation_contract,
    )
    easting_thresholds = development_easting_thresholds(
        city_roles,
        training_geography,
        allocation_contract=allocation_contract,
    )

    split_documents: dict[str, Any] = {}
    observed_city_sets: dict[str, set[str]] = {}
    target_city_roles: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        ordered_digest = hashlib.sha256()
        city_digests: dict[str, Any] = defaultdict(hashlib.sha256)
        city_counts: Counter[str] = Counter()
        city_role_counts: Counter[str] = Counter()
        sample_role_counts: Counter[str] = Counter()
        city_sample_roles: dict[str, set[str]] = defaultdict(set)
        observed_count = 0
        for record in iter_geo_records(
            split,
            paths[split],
            city_roles=city_roles,
            development_easting_thresholds=easting_thresholds,
            h5_factory=h5_factory,
        ):
            row = record.as_dict()
            _digest_update(ordered_digest, row)
            _digest_update(city_digests[record.city_id], row)
            observed_count += 1
            city_counts[record.city_id] += 1
            city_role_counts[record.city_role] += 1
            sample_role_counts[record.sample_role] += 1
            city_sample_roles[record.city_id].add(record.sample_role)
            if record.city_role == "target":
                target_city_roles[record.city_id].add(record.sample_role)
        expected_count = expected_split_counts[split]
        if observed_count != expected_count:
            raise IntegrityError(
                f"{split} population count mismatch: expected {expected_count}, found {observed_count}"
            )
        observed_cities = set(city_counts)
        expected_cities = expected_training_cities if split == "training" else expected_target_cities
        if observed_cities != set(expected_cities):
            raise IntegrityError(
                f"{split} city set mismatch; missing={sorted(set(expected_cities) - observed_cities)}, "
                f"extra={sorted(observed_cities - set(expected_cities))}"
            )
        observed_city_sets[split] = observed_cities
        if split == "training":
            if dict(sorted(city_counts.items())) != training_geography["city_counts"]:
                raise IntegrityError("training count preflight differs from manifest scan")
            for city, roles in city_sample_roles.items():
                required = _required_roles_for_city(city_roles[city])
                if city_roles[city] in {"source_fit_ineligible", "source_fit_core"}:
                    if not roles or not roles.issubset(required):
                        raise IntegrityError(
                            f"source city {city!r} has an invalid monitor partition: {sorted(roles)}"
                        )
                elif roles != required:
                    raise IntegrityError(
                        f"training city {city!r} lacks a complete block partition: {sorted(roles)} != {sorted(required)}"
                    )
        split_documents[split] = {
            "geo_artifact": {
                "basename": paths[split].name,
                "bytes": paths[split].stat().st_size,
                "sha256": file_sha256(paths[split]),
            },
            "expected_samples": expected_count,
            "observed_samples": observed_count,
            "ordered_metadata_sha256": ordered_digest.hexdigest(),
            "city_counts": dict(sorted(city_counts.items())),
            "city_metadata_sha256": {
                city: digest.hexdigest() for city, digest in sorted(city_digests.items())
            },
            "city_role_counts": dict(sorted(city_role_counts.items())),
            "sample_role_counts": dict(sorted(sample_role_counts.items())),
        }

    if observed_city_sets["validation"] != observed_city_sets["testing"]:
        raise IntegrityError("official validation/testing halves do not contain the same target cities")
    if observed_city_sets["training"] & observed_city_sets["validation"]:
        raise IntegrityError("training and target cities overlap")
    for city, roles in target_city_roles.items():
        required = _required_roles_for_city("target")
        if roles != required:
            raise IntegrityError(
                f"target city {city!r} is not represented in both official halves: {sorted(roles)}"
            )

    protocol = protocol_identity()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "LABEL_FREE_METADATA_POPULATION_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "protocol_identity": {
            "basename": Path(protocol["path"]).name,
            "bytes": protocol["bytes"],
            "file_sha256": protocol["file_sha256"],
            "canonical_document_sha256": protocol["canonical_document_sha256"],
        },
        "access_contract": {
            "opened_files": [GEO_BASENAMES[split] for split in SPLITS],
            "allowed_hdf5_datasets": list(OFFICIAL_GEO_KEYS),
            "image_containers_opened": False,
            "target_outcome_arrays_opened": False,
            "target_outcome_arrays_counted": False,
            "target_outcome_arrays_hashed": False,
        },
        "partition_contract": {
            "training_city_algorithm": "metadata_count_then_sha256",
            "minimum_eligible_rows": allocation_contract.minimum_eligible_rows,
            "minimum_eligible_distinct_block_eastings": 2,
            "expected_ineligible_city_count": allocation_contract.expected_ineligible_city_count,
            "source_fit_core_count": allocation_contract.source_fit_core_count,
            "source_fit_total_cities": (
                allocation_contract.expected_ineligible_city_count + allocation_contract.source_fit_core_count
            ),
            "gate_fit_count": allocation_contract.gate_fit_count,
            "gate_cal_count": allocation_contract.gate_cal_count,
            "gate_city_salt_sha256": stable_sha256(allocation_contract.gate_salt),
            "source_role_counts": allocation_contract.role_counts(),
            "development_assignment": "upper_median_distinct_block_easting_west_probe_east_evaluation",
            "source_monitor_block_salt_sha256": stable_sha256(SOURCE_MONITOR_BLOCK_SALT),
            "spatial_block_metres": BLOCK_METRES,
            "target_probe_split": "validation",
            "target_evaluation_split": "testing",
            "labels_used": False,
        },
        "cities": {
            "training": sorted(expected_training_cities),
            "target": sorted(expected_target_cities),
            "training_roles": {
                role: sorted(city for city, assigned in city_roles.items() if assigned == role)
                for role in allocation_contract.role_counts()
            },
            "training_geography": training_geography,
            "source_fit_ineligible_reasons": ineligibility_reasons,
            "development_easting_thresholds": easting_thresholds,
        },
        "splits": split_documents,
    }
    manifest["population_identity_sha256"] = stable_sha256(
        {
            split: {
                "ordered_metadata_sha256": split_documents[split]["ordered_metadata_sha256"],
                "geo_sha256": split_documents[split]["geo_artifact"]["sha256"],
            }
            for split in SPLITS
        }
    )
    manifest["manifest_sha256"] = stable_sha256(manifest)
    validate_population_manifest(
        manifest,
        expected_split_counts=expected_split_counts,
        expected_training_cities=expected_training_cities,
        expected_target_cities=expected_target_cities,
        allocation_contract=allocation_contract,
    )
    return manifest


def validate_population_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_split_counts: Mapping[str, int] = OFFICIAL_SPLIT_COUNTS,
    expected_training_cities: frozenset[str] = TRAINING_CITY_IDS,
    expected_target_cities: frozenset[str] = TARGET_CITY_IDS,
    allocation_contract: CityAllocationContract = DEFAULT_CITY_ALLOCATION,
) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("protocol_id") != PROTOCOL_ID:
        raise IntegrityError("unknown So2Sat population manifest schema/protocol")
    if manifest.get("status") != "LABEL_FREE_METADATA_POPULATION_VERIFIED":
        raise IntegrityError("So2Sat population manifest is not verified")
    claimed = require_sha256(manifest.get("manifest_sha256"), field="manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if stable_sha256(unsigned) != claimed:
        raise IntegrityError("So2Sat population manifest hash mismatch")
    current_protocol = protocol_identity()
    expected_protocol_identity = {
        "basename": Path(current_protocol["path"]).name,
        "bytes": current_protocol["bytes"],
        "file_sha256": current_protocol["file_sha256"],
        "canonical_document_sha256": current_protocol["canonical_document_sha256"],
    }
    if manifest.get("protocol_identity") != expected_protocol_identity:
        raise IntegrityError("population manifest protocol identity drift")
    access = manifest.get("access_contract")
    expected_access = {
        "opened_files": [GEO_BASENAMES[split] for split in SPLITS],
        "allowed_hdf5_datasets": list(OFFICIAL_GEO_KEYS),
        "image_containers_opened": False,
        "target_outcome_arrays_opened": False,
        "target_outcome_arrays_counted": False,
        "target_outcome_arrays_hashed": False,
    }
    if access != expected_access:
        raise IntegrityError("population manifest does not establish the label-free access contract")
    expected_partition = {
        "training_city_algorithm": "metadata_count_then_sha256",
        "minimum_eligible_rows": allocation_contract.minimum_eligible_rows,
        "minimum_eligible_distinct_block_eastings": 2,
        "expected_ineligible_city_count": allocation_contract.expected_ineligible_city_count,
        "source_fit_core_count": allocation_contract.source_fit_core_count,
        "source_fit_total_cities": (
            allocation_contract.expected_ineligible_city_count + allocation_contract.source_fit_core_count
        ),
        "gate_fit_count": allocation_contract.gate_fit_count,
        "gate_cal_count": allocation_contract.gate_cal_count,
        "gate_city_salt_sha256": stable_sha256(allocation_contract.gate_salt),
        "source_role_counts": allocation_contract.role_counts(),
        "development_assignment": "upper_median_distinct_block_easting_west_probe_east_evaluation",
        "source_monitor_block_salt_sha256": stable_sha256(SOURCE_MONITOR_BLOCK_SALT),
        "spatial_block_metres": BLOCK_METRES,
        "target_probe_split": "validation",
        "target_evaluation_split": "testing",
        "labels_used": False,
    }
    if manifest.get("partition_contract") != expected_partition:
        raise IntegrityError("population manifest partition contract drift")
    cities = manifest.get("cities")
    if not isinstance(cities, Mapping):
        raise IntegrityError("population manifest cities must be a mapping")
    if cities.get("training") != sorted(expected_training_cities):
        raise IntegrityError("population manifest training-city set drift")
    if cities.get("target") != sorted(expected_target_cities):
        raise IntegrityError("population manifest target-city set drift")
    training_geography = cities.get("training_geography")
    if not isinstance(training_geography, Mapping):
        raise IntegrityError("population manifest lacks training geography")
    training_counts = training_geography.get("city_counts")
    if not isinstance(training_counts, Mapping) or set(training_counts) != set(expected_training_cities):
        raise IntegrityError("population manifest training geography counts drift")
    if training_geography.get("row_count") != expected_split_counts["training"]:
        raise IntegrityError("population manifest training geography row count drift")
    if sum(training_counts.values()) != expected_split_counts["training"]:
        raise IntegrityError("population manifest training geography counts do not sum to the population")
    training_eastings = training_geography.get("city_distinct_block_eastings")
    training_easting_counts = training_geography.get("city_distinct_block_easting_counts")
    training_epsg = training_geography.get("city_epsg")
    if (
        not isinstance(training_eastings, Mapping)
        or set(training_eastings) != set(expected_training_cities)
        or not isinstance(training_easting_counts, Mapping)
        or set(training_easting_counts) != set(expected_training_cities)
        or not isinstance(training_epsg, Mapping)
        or set(training_epsg) != set(expected_training_cities)
    ):
        raise IntegrityError("population manifest training geography schema drift")
    for city in expected_training_cities:
        eastings = training_eastings[city]
        epsg = training_epsg[city]
        if (
            not isinstance(eastings, list)
            or not eastings
            or eastings != sorted(set(eastings))
            or any(isinstance(value, bool) or not isinstance(value, int) for value in eastings)
        ):
            raise IntegrityError(f"population manifest training eastings invalid for {city!r}")
        if training_easting_counts[city] != len(eastings):
            raise IntegrityError(f"population manifest training easting count invalid for {city!r}")
        if isinstance(epsg, bool) or not isinstance(epsg, int) or epsg <= 0:
            raise IntegrityError(f"population manifest training EPSG invalid for {city!r}")
    city_roles = assign_training_city_roles(
        training_counts,
        training_easting_counts,
        contract=allocation_contract,
    )
    expected_roles = {
        role: sorted(city for city, assigned in city_roles.items() if assigned == role)
        for role in allocation_contract.role_counts()
    }
    if cities.get("training_roles") != expected_roles:
        raise IntegrityError("population manifest deterministic city roles drift")
    expected_ineligibility_reasons = training_city_ineligibility_reasons(
        training_counts,
        training_easting_counts,
        contract=allocation_contract,
    )
    if cities.get("source_fit_ineligible_reasons") != expected_ineligibility_reasons:
        raise IntegrityError("population manifest ineligible-city reasons drift")
    expected_thresholds = development_easting_thresholds(
        city_roles,
        training_geography,
        allocation_contract=allocation_contract,
    )
    if cities.get("development_easting_thresholds") != expected_thresholds:
        raise IntegrityError("population manifest development easting thresholds drift")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise IntegrityError("population manifest split inventory drift")
    identity_payload: dict[str, Any] = {}
    for split in SPLITS:
        row = splits[split]
        if not isinstance(row, Mapping):
            raise IntegrityError(f"population manifest split {split!r} must be a mapping")
        if row.get("expected_samples") != expected_split_counts[split]:
            raise IntegrityError(f"population manifest expected {split} count drift")
        if row.get("observed_samples") != expected_split_counts[split]:
            raise IntegrityError(f"population manifest observed {split} count drift")
        geo = row.get("geo_artifact")
        if not isinstance(geo, Mapping) or geo.get("basename") != GEO_BASENAMES[split]:
            raise IntegrityError(f"population manifest {split} geo identity is invalid")
        if isinstance(geo.get("bytes"), bool) or not isinstance(geo.get("bytes"), int) or geo["bytes"] < 1:
            raise IntegrityError(f"population manifest {split} geo byte count is invalid")
        require_sha256(geo.get("sha256"), field=f"splits.{split}.geo_artifact.sha256")
        require_sha256(row.get("ordered_metadata_sha256"), field=f"splits.{split}.ordered_metadata_sha256")
        city_counts = row.get("city_counts")
        expected_city_set = expected_training_cities if split == "training" else expected_target_cities
        if not isinstance(city_counts, Mapping) or set(city_counts) != set(expected_city_set):
            raise IntegrityError(f"population manifest {split} city counts drift")
        if sum(city_counts.values()) != expected_split_counts[split]:
            raise IntegrityError(f"population manifest {split} city counts do not sum to the population")
        if split == "training" and dict(city_counts) != dict(training_counts):
            raise IntegrityError("population manifest training split/geography counts disagree")
        city_hashes = row.get("city_metadata_sha256")
        if not isinstance(city_hashes, Mapping) or set(city_hashes) != set(expected_city_set):
            raise IntegrityError(f"population manifest {split} city hashes drift")
        for city, digest in city_hashes.items():
            require_sha256(digest, field=f"splits.{split}.city_metadata_sha256.{city}")
        sample_roles = row.get("sample_role_counts")
        if not isinstance(sample_roles, Mapping) or sum(sample_roles.values()) != expected_split_counts[split]:
            raise IntegrityError(f"population manifest {split} sample roles do not cover the population")
        allowed_roles = (
            {
                "source_train",
                "source_monitor",
                "gate_fit_probe",
                "gate_fit_evaluation",
                "gate_cal_probe",
                "gate_cal_evaluation",
            }
            if split == "training"
            else {"target_probe"}
            if split == "validation"
            else {"target_evaluation"}
        )
        if set(sample_roles) != allowed_roles:
            raise IntegrityError(f"population manifest {split} sample-role schema drift")
        expected_city_role_names = (
            set(allocation_contract.role_counts()) if split == "training" else {"target"}
        )
        city_role_counts = row.get("city_role_counts")
        if (
            not isinstance(city_role_counts, Mapping)
            or set(city_role_counts) != expected_city_role_names
            or sum(city_role_counts.values()) != expected_split_counts[split]
        ):
            raise IntegrityError(f"population manifest {split} city-role counts drift")
        identity_payload[split] = {
            "ordered_metadata_sha256": row["ordered_metadata_sha256"],
            "geo_sha256": geo["sha256"],
        }
    if manifest.get("population_identity_sha256") != stable_sha256(identity_payload):
        raise IntegrityError("population identity SHA-256 mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_checked_in_protocol_receipt()
    manifest = build_population_manifest(
        {split: args.data_root / GEO_BASENAMES[split] for split in SPLITS}
    )
    receipt = write_immutable_json_with_receipt(args.output, manifest)
    print(
        f"So2Sat label-free population: PASS n={sum(OFFICIAL_SPLIT_COUNTS.values())} "
        f"manifest={manifest['manifest_sha256']} receipt={receipt['artifact_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
