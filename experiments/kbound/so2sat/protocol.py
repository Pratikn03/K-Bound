"""Schema and immutable identity checks for the prospective So2Sat protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .integrity import IntegrityError, file_sha256, stable_sha256, strict_json_load


SCHEMA = "kbound_so2sat_prospective_protocol_v1"
PROTOCOL_ID = "KBOUND_SO2SAT_LCZ42_PROSPECTIVE_CONFIRMATION_v1"
PROTOCOL_BASENAME = "prospective_protocol_v1.json"
PROTOCOL_RECEIPT_BASENAME = "prospective_protocol_v1.json.receipt.json"
OFFICIAL_SPLIT_COUNTS = {
    "training": 352_366,
    "validation": 24_119,
    "testing": 24_188,
}
OFFICIAL_GEO_KEYS = ("city", "epsg", "tfw")
SOURCE_CITY_ROLE_COUNTS = {
    "source_fit_ineligible": 9,
    "source_fit_core": 5,
    "gate_fit": 9,
    "gate_cal": 19,
}
MINIMUM_GATE_CITY_ROWS = 256
SOURCE_FIT_TOTAL_CITIES = 14
GATE_CITY_SALT = "KBOUND_SO2SAT_GATE_CITY_ROLES_v1"


def default_protocol_path() -> Path:
    return Path(__file__).resolve().with_name(PROTOCOL_BASENAME)


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    source = default_protocol_path() if path is None else Path(path)
    document = strict_json_load(source)
    if not isinstance(document, dict):
        raise IntegrityError(f"protocol {source} must contain a JSON mapping")
    validate_protocol(document)
    return document


def validate_protocol(document: Mapping[str, Any]) -> None:
    """Fail closed on any change to the structural prospective contract."""

    if document.get("schema") != SCHEMA or document.get("protocol_id") != PROTOCOL_ID:
        raise IntegrityError("unknown So2Sat prospective protocol schema or id")
    if document.get("status") != "STRUCTURAL_PROTOCOL_SEALED_EXECUTION_CONFIG_PENDING":
        raise IntegrityError("So2Sat protocol status must disclose the pending execution seal")

    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping):
        raise IntegrityError("protocol.dataset must be a mapping")
    expected_dataset = {
        "name": "So2Sat LCZ42",
        "release": "v4.2",
        "split_scenario": "culture-10",
        "class_count": 17,
        "patch_shape": [32, 32],
        "geo_keys": list(OFFICIAL_GEO_KEYS),
        "split_counts": OFFICIAL_SPLIT_COUNTS,
    }
    for field, expected in expected_dataset.items():
        if dataset.get(field) != expected:
            raise IntegrityError(f"protocol.dataset.{field} drift: {dataset.get(field)!r} != {expected!r}")

    roles = document.get("roles")
    if not isinstance(roles, Mapping):
        raise IntegrityError("protocol.roles must be a mapping")
    city_partition = roles.get("training_city_partition")
    if not isinstance(city_partition, Mapping):
        raise IntegrityError("protocol training-city partition is missing")
    expected_partition = {
        "unit": "normalized_city",
        "algorithm": "metadata_count_then_sha256",
        "count_source": "complete_training_geo_population",
        "eligibility_rule": "rows_at_least_256_and_at_least_2_distinct_6400m_block_eastings",
        "ineligible_city_rule": "any_city_failing_either_eligibility_condition_to_source_fit_ineligible",
        "expected_ineligible_city_count": 9,
        "core_city_rule": "five_largest_eligible_to_source_fit_core_count_descending_then_city_id",
        "gate_candidate_rule": "remaining_28_eligible_cities",
        "gate_hash_salt": GATE_CITY_SALT,
        "gate_hash_rule": "ascending_sha256_then_city_id_first_9_gate_fit_remaining_19_gate_cal",
        "role_counts": SOURCE_CITY_ROLE_COUNTS,
        "source_fit_total_cities": SOURCE_FIT_TOTAL_CITIES,
        "minimum_gate_city_rows": MINIMUM_GATE_CITY_ROWS,
        "minimum_gate_city_distinct_block_eastings": 2,
        "labels_used": False,
    }
    if dict(city_partition) != expected_partition:
        raise IntegrityError("protocol training-city partition drift")
    target = roles.get("target")
    expected_target = {
        "cities": "the_10_official_culture_10_cities",
        "action_unit": "city",
        "probe_split": "validation",
        "evaluation_split": "testing",
        "probe_labels_scored": False,
        "evaluation_labels_available_to_live_runner": False,
    }
    if target != expected_target:
        raise IntegrityError("protocol target role/firewall contract drift")

    partition = roles.get("within_development_city_partition")
    if not isinstance(partition, Mapping):
        raise IntegrityError("protocol development partition is missing")
    if (
        partition.get("unit") != "epsg_x_6400m_spatial_block"
        or partition.get("easting_threshold") != "upper_median_of_sorted_distinct_block_eastings_per_city"
        or partition.get("probe_rule") != "block_easting_strictly_west_of_threshold"
        or partition.get("evaluation_rule") != "block_easting_at_or_east_of_threshold"
        or partition.get("whole_block_atomicity") != "required"
        or partition.get("labels_used") is not False
    ):
        raise IntegrityError("protocol spatial-block partition drift")
    monitor = roles.get("source_monitor_partition")
    expected_monitor = {
        "unit": "epsg_x_6400m_spatial_block",
        "monitor_fraction": 0.10,
        "hash": "sha256_exact_rational_comparison",
        "salt": "KBOUND_SO2SAT_SOURCE_MONITOR_BLOCK_ROLES_v1",
        "labels_used": False,
    }
    if monitor != expected_monitor:
        raise IntegrityError("protocol source-monitor partition drift")

    firewall = document.get("target_label_firewall")
    expected_firewall = {
        "geo_reader_allowed_datasets": list(OFFICIAL_GEO_KEYS),
        "pixel_reader_allowed_datasets": ["sen1", "sen2"],
        "target_outcome_dataset_access": "forbidden_until_predictions_and_actions_are_sealed",
        "live_objects_expose_labels": False,
        "opaque_container_hashing_deserializes_hdf5_datasets": False,
        "scoring_implementation_in_this_package": False,
    }
    if firewall != expected_firewall:
        raise IntegrityError("protocol target-label firewall drift")

    execution = document.get("execution")
    if not isinstance(execution, Mapping):
        raise IntegrityError("protocol.execution must be a mapping")
    if (
        execution.get("model_seeds") != [0, 1, 2, 3, 4]
        or execution.get("modality") != "sen2_10_band"
        or execution.get("candidate_selection_data") != "gate_fit_cities_only"
        or execution.get("lock_before") != "gate_calibration_and_any_target_pixel_access"
        or execution.get("report_regardless_of_direction") is not True
    ):
        raise IntegrityError("protocol execution boundary drift")

    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        raise IntegrityError("protocol.inference must be a mapping")
    if (
        inference.get("cluster_unit") != "target_city"
        or inference.get("comparison_family") != ["kga_vs_always_adapt", "kga_vs_always_freeze"]
        or inference.get("multiplicity") != "holm"
        or inference.get("confidence_interval_sign") != "positive_favors_kga"
        or inference.get("report_all_10_target_cities") is not True
    ):
        raise IntegrityError("protocol inference contract drift")


def protocol_identity(path: str | Path | None = None) -> dict[str, Any]:
    source = default_protocol_path() if path is None else Path(path).expanduser().resolve()
    document = load_protocol(source)
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "file_sha256": file_sha256(source),
        "canonical_document_sha256": stable_sha256(document),
    }


def verify_checked_in_protocol_receipt(path: str | Path | None = None) -> dict[str, Any]:
    """Verify the portable checked-in receipt for the structural protocol."""

    source = default_protocol_path() if path is None else Path(path).expanduser().resolve()
    receipt_path = source.with_name(PROTOCOL_RECEIPT_BASENAME)
    receipt = strict_json_load(receipt_path)
    if not isinstance(receipt, Mapping):
        raise IntegrityError("checked-in protocol receipt must be a JSON mapping")
    expected = protocol_identity(source)
    expected_receipt = {
        "schema": "kbound_so2sat_protocol_receipt_v1",
        "artifact": PROTOCOL_BASENAME,
        "artifact_bytes": expected["bytes"],
        "artifact_sha256": expected["file_sha256"],
        "canonical_document_sha256": expected["canonical_document_sha256"],
    }
    if dict(receipt) != expected_receipt:
        raise IntegrityError("checked-in So2Sat protocol receipt mismatch")
    return dict(receipt)
