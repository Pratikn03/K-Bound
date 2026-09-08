"""Apply the approved, hash-bound painting quarantine before development scores.

This metadata-only transformation preserves the stopped V1 and every raw file.
It does not certify population sampling or independent environment claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.kbound.domainnet import pilot_data, source_data

AMENDMENT_SHA256 = "e13985b0fd46bca222dd63041b55b81059b3efd2e37de7d5562a82b24e196f66"
PRODUCTION_IDENTITIES = {
    "amendment_sha256": AMENDMENT_SHA256,
    "v1_stop_sha256": "164c276ab5db4c720964d54c4ea0f44455c8a32e3b36c1b88c2a5c815ee1db61",
    "v1_summary_sha256": "20987c8fe007454ae9fc8f87a3dfdae0a0a6ea37a90efad7bbe1cb023adc8223",
    "v1_inventory_sha256": "c9b0e2fcd87eadd9d17f27ca653d67f6c42e6269bbbc9e239375803f448badd4",
    "v1_public_manifest_sha256": "fecf268de9af5b6a5dd36ab93979f022209d153df7399258ef454cc56070a0ec",
    "v1_labels_sha256": "d8316f5680b0022b61ccdaea8f8731cf3c77aaa840c0ee156bd5e58c85dea7c5",
    "source_inventory_sha256": pilot_data.SOURCE_INVENTORY_SHA256,
    "archive_sha256": pilot_data.ARCHIVE_SHA256,
    "list_sha256": pilot_data.LIST_SHA256,
    "list_git_blob": pilot_data.LIST_GIT_BLOB,
}
PARENT_NAMES = {
    "inventory.json": "v1_inventory_sha256",
    "public_manifest.json": "v1_public_manifest_sha256",
    "labels.json": "v1_labels_sha256",
    "summary.json": "v1_summary_sha256",
}
PUBLIC_FIELDS = (
    "sample_id",
    "official_index",
    "image_sha256",
    "group_id",
    "width",
    "height",
    "partition",
    "cell",
    "window",
)


@dataclass(frozen=True)
class SyntheticIdentity:
    """Reduced-class library fixtures only; none may use a production identity."""

    amendment_sha256: str
    v1_stop_sha256: str
    v1_summary_sha256: str
    v1_inventory_sha256: str
    v1_public_manifest_sha256: str
    v1_labels_sha256: str
    source_inventory_sha256: str
    archive_sha256: str
    list_sha256: str
    list_git_blob: str
    class_count: int

    def __post_init__(self) -> None:
        if type(self.class_count) is not int or not 1 <= self.class_count < 126:
            raise ValueError("synthetic identities require a reduced integer class count")
        real = {*PRODUCTION_IDENTITIES.values(), pilot_data.DESIGN_SHA256}
        for name in PRODUCTION_IDENTITIES:
            value = getattr(self, name)
            length = 40 if name == "list_git_blob" else 64
            if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None or value in real:
                raise ValueError("synthetic identity is malformed or uses a production identity")


def _safe_path(path: Path, *, directory: bool = False, new: bool = False) -> Path:
    if ".." in path.parts:
        raise ValueError("unsafe path traversal")
    absolute = path.absolute()
    for part in (*reversed(absolute.parents), absolute):
        if part.is_symlink():
            raise ValueError(f"symlink path forbidden: {part}")
    if new:
        if absolute.exists():
            raise FileExistsError(path)
        if not absolute.parent.is_dir():
            raise ValueError("output parent must already be a directory")
    else:
        mode = absolute.stat().st_mode
        if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
            raise ValueError("input must be a regular file or the expected directory")
    return absolute


def _read_pinned(path: Path, expected: str) -> dict[str, Any]:
    path = _safe_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("input is not a regular file")
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value: str) -> Any:
        raise ValueError(f"nonfinite JSON: {value}")

    document = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(document, dict):
        raise ValueError("expected JSON object")
    return document


def _equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"inconsistent {description}")


def _conflicts(groups: list[dict[str, Any]]) -> dict[str, Any]:
    conflicts = [group for group in groups if len(group["labels"]) > 1]
    return {"groups": conflicts, "group_count": len(conflicts), "row_count": sum(g["multiplicity"] for g in conflicts)}


def _source_groups(source: dict[str, Any], class_count: int) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    for key, expected in {
        "schema": "kbound-domainnet-source-inventory/2",
        "domain": "clipart",
        "role": "source",
        "status": "SOURCE_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK",
    }.items():
        _equal(source.get(key), expected, "source authority")
    _equal(source.get("identity", {}).get("class_count"), class_count, "source class count")
    classes: dict[str, int] = {}
    reverse: dict[int, str] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    paths = set()
    for row in source["rows"]:
        path = source_data.canonical_path(row["path"])
        label, group = row["label"], row["group_id"]
        if len(path.split("/")) != 3 or path in paths or type(label) is not int or not 0 <= label < class_count:
            raise ValueError("invalid source member")
        if not isinstance(group, str) or re.fullmatch("[0-9a-f]{64}", group) is None:
            raise ValueError("invalid source decoded identity")
        name = path.split("/")[1]
        if classes.get(name, label) != label or reverse.get(label, name) != name:
            raise ValueError("inconsistent source class mapping")
        classes[name], reverse[label] = label, name
        paths.add(path)
        groups.setdefault(group, []).append(row)
    _equal(set(reverse), set(range(class_count)), "source class coverage")
    return classes, groups


def _validate_rows(rows: list[dict[str, Any]], classes: dict[str, int]) -> None:
    paths = set()
    for index, row in enumerate(rows):
        path = pilot_data.canonical_painting_path(row["path"])
        label = row["label"]
        if path in paths or type(label) is not int or classes.get(path.split("/")[1]) != label:
            raise ValueError("invalid painting class mapping or duplicate path")
        _equal(row["official_index"], index, "official row order")
        _equal(
            row["sample_id"],
            hashlib.sha256(("kbound-painting-pilot-v1:sample:" + path).encode()).hexdigest(),
            "sample alias",
        )
        for key in ("group_id", "image_sha256"):
            if not isinstance(row[key], str) or re.fullmatch("[0-9a-f]{64}", row[key]) is None:
                raise ValueError("invalid painting content identity")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("width", "height")):
            raise ValueError("invalid painting dimensions")
        paths.add(path)


def _derive(
    v1_dir: Path,
    source_inventory: Path,
    design: Path,
    amendment: Path,
    v1_stop: Path,
    synthetic_identity: SyntheticIdentity | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if synthetic_identity is not None:
        if type(synthetic_identity) is not SyntheticIdentity:
            raise ValueError("explicit SyntheticIdentity required")
        synthetic_identity.__post_init__()
    pins: dict[str, Any] = (
        asdict(synthetic_identity) if synthetic_identity is not None else {**PRODUCTION_IDENTITIES, "class_count": 126}
    )
    identity = {
        key: pins[key]
        for key in ("archive_sha256", "list_sha256", "list_git_blob", "source_inventory_sha256", "class_count")
    }
    parents = {key: value for key, value in pins.items() if key.startswith("v1_") or key == "source_inventory_sha256"}
    v1_dir = _safe_path(v1_dir, directory=True)
    design_document = _read_pinned(design, pilot_data.DESIGN_SHA256)
    amendment_document = _read_pinned(amendment, pins["amendment_sha256"])
    stop = _read_pinned(v1_stop, pins["v1_stop_sha256"])
    summary = _read_pinned(v1_dir / "summary.json", pins["v1_summary_sha256"])
    expected_outputs = {name: pins[key] for name, key in PARENT_NAMES.items() if name != "summary.json"}
    _equal(summary.get("output_sha256"), expected_outputs, "parent output hashes")
    for name, key in PARENT_NAMES.items():
        stop_key = "summary_sha256" if name == "summary.json" else name.removesuffix(".json") + "_sha256"
        _equal(stop.get("preparation", {}).get(stop_key), pins[key], "STOP parent identities")
    for key in ("v1_stop_sha256", "v1_summary_sha256", "v1_inventory_sha256", "source_inventory_sha256"):
        _equal(amendment_document.get(key), pins[key], "amendment parent identities")
    _equal(amendment_document.get("base_design_sha256"), pilot_data.DESIGN_SHA256, "amendment base design")
    if synthetic_identity is not None:
        canonical = _read_pinned(
            Path(__file__).resolve().parents[3] / "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_AMENDMENT_v2.json",
            AMENDMENT_SHA256,
        )
        adjustable = {
            "v1_stop_sha256",
            "v1_summary_sha256",
            "v1_inventory_sha256",
            "source_inventory_sha256",
            "expected_quarantined_groups",
            "expected_quarantined_painting_rows",
            "expected_original_painting_rows",
            "expected_remaining_painting_rows",
        }
        _equal(
            {k: v for k, v in amendment_document.items() if k not in adjustable},
            {k: v for k, v in canonical.items() if k not in adjustable},
            "amendment rule and unchanged settings",
        )
    source = _read_pinned(source_inventory, pins["source_inventory_sha256"])
    classes, source_groups = _source_groups(source, pins["class_count"])
    parent = _read_pinned(v1_dir / "inventory.json", pins["v1_inventory_sha256"])
    scope = "SYNTHETIC_TEST" if synthetic_identity is not None else "PAINTING_DEVELOPMENT_ONLY"
    for key, expected in {
        "schema": "kbound-painting-pilot-inventory/1",
        "status": "STOP_SOURCE_OVERLAP",
        "domain": "painting",
        "role": "development",
        "identity": identity,
        "class_map": classes,
        "eligible_for_pilot": False,
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "design": {"sha256": pilot_data.DESIGN_SHA256, "document": design_document},
        "partition_salt": pilot_data.PARTITION_SALT,
        "group_order_salt": pilot_data.ORDER_SALT,
        "custody_scope": design_document["custody_scope"],
    }.items():
        _equal(parent.get(key), expected, "V1 inventory authority")
    rows = parent["rows"]
    _validate_rows(rows, classes)
    # Only row dictionaries are copied: never deep-copy the giant parent inventory.
    original_rows = [dict(row) for row in rows]
    original_packing, original_counts, original_groups = pilot_data._packing(original_rows)
    for actual, expected, description in (
        (rows, original_rows, "V1 row placements"),
        (parent["packing"], original_packing, "V1 packing"),
        (parent["counts"], original_counts, "V1 counts"),
        (parent["groups"], original_groups, "V1 group inventory"),
        (parent["conflicts"], _conflicts(original_groups), "V1 conflicts"),
    ):
        _equal(actual, expected, description)
    overlaps = [
        {
            "group_id": group["group_id"],
            "painting_members": group["members"],
            "source_members": source_groups[group["group_id"]],
        }
        for group in original_groups
        if group["group_id"] in source_groups
    ]
    overlap_counts = {
        "group_count": len(overlaps),
        "painting_row_count": sum(len(g["painting_members"]) for g in overlaps),
        "source_row_count": sum(len(g["source_members"]) for g in overlaps),
    }
    if not overlaps:
        raise ValueError("V1 STOP must disclose source overlap")
    _equal(parent["source_overlap"], {"groups": overlaps, **overlap_counts}, "V1 source overlap membership")
    expected_public = {
        "schema": "kbound-painting-pilot-public/1",
        "status": "STOP_SOURCE_OVERLAP",
        "eligible_for_pilot": False,
        "execution_scope": scope,
        "design_sha256": pilot_data.DESIGN_SHA256,
        "packing": original_packing,
        "rows": [{k: r[k] for k in PUBLIC_FIELDS} for r in rows],
    }
    _equal(
        _read_pinned(v1_dir / "public_manifest.json", pins["v1_public_manifest_sha256"]),
        expected_public,
        "V1 public manifest",
    )
    _equal(
        _read_pinned(v1_dir / "labels.json", pins["v1_labels_sha256"]),
        {r["sample_id"]: r["label"] for r in rows},
        "V1 scorer labels",
    )
    expected_summary = {
        "schema": "kbound-painting-pilot-summary/1",
        "status": "STOP_SOURCE_OVERLAP",
        "eligible_for_pilot": False,
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "identity": identity,
        "design_sha256": pilot_data.DESIGN_SHA256,
        "counts": original_counts,
        "row_count": len(rows),
        "group_count": len(original_groups),
        "conflict_group_count": parent["conflicts"]["group_count"],
        "source_overlap_group_count": len(overlaps),
        "output_sha256": expected_outputs,
    }
    _equal(summary, expected_summary, "V1 summary")
    excluded = [row for row in rows if row["group_id"] in source_groups]
    retained = [dict(row) for row in rows if row["group_id"] not in source_groups]
    for key, expected in {
        "schema": "kbound-domainnet-development-stop/1",
        "status": "STOP_SOURCE_OVERLAP",
        "stage": "PRE_MODEL_DEVELOPMENT_INPUT_VALIDATION",
        "scientific_outcome": "NOT_TESTED",
        "design_sha256": pilot_data.DESIGN_SHA256,
        "preparation_cli_exit_code": 2,
        "rows": len(rows),
        "decoded_content_groups": len(original_groups),
        "within_painting_cross_label_groups": parent["conflicts"]["group_count"],
        "within_painting_cross_label_rows": parent["conflicts"]["row_count"],
        "cross_source_duplicate_groups": len(overlaps),
        "overlapping_painting_rows": len(excluded),
        "overlapping_clipart_rows": overlap_counts["source_row_count"],
        "affected_painting_rows_by_partition": {
            p: sum(r["partition"] == p for r in excluded) for p in pilot_data.PARTITIONS
        },
        "complete_cells_before_stop": {p: original_counts[p]["complete_cells"] for p in pilot_data.PARTITIONS},
        "unscored_tail_rows": sum(c["tail_rows"] for c in original_counts.values()),
        "minimum_geometry_passes": all(
            original_counts[p]["complete_cells"] >= pilot_data.MINIMUM_CELLS[p] for p in pilot_data.PARTITIONS
        ),
        "eligible_for_pilot": False,
        "eligible_for_confirmatory": False,
        "source_pilot_models_completed": 0,
        "full_training_launched": False,
        "candidate_or_gate_outcomes_scored": False,
    }.items():
        _equal(stop.get(key), expected, "immutable V1 STOP")
    for key, expected in {
        "expected_quarantined_groups": len(overlaps),
        "expected_quarantined_painting_rows": len(excluded),
        "expected_original_painting_rows": len(rows),
        "expected_remaining_painting_rows": len(retained),
    }.items():
        if type(amendment_document.get(key)) is not int:
            raise ValueError("amendment exclusion count must be an integer")
        _equal(amendment_document[key], expected, "approved exact quarantine count")
    packing, counts, groups = pilot_data._packing(retained)
    conflicts = _conflicts(groups)
    eligible = all(counts[p]["complete_cells"] >= pilot_data.MINIMUM_CELLS[p] for p in pilot_data.PARTITIONS)
    status = "DEV_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK" if eligible else "INCONCLUSIVE_DATA_GEOMETRY"
    common = {
        "status": status,
        "eligible_for_pilot": eligible,
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "parents": parents,
    }
    inventory = {
        **common,
        "schema": "kbound-painting-pilot-inventory/2",
        "domain": "painting",
        "role": "development",
        "identity": identity,
        "design": parent["design"],
        "amendment": {"sha256": pins["amendment_sha256"], "document": amendment_document},
        "class_map": classes,
        "rows": retained,
        "groups": groups,
        "packing": packing,
        "counts": counts,
        "partition_salt": pilot_data.PARTITION_SALT,
        "group_order_salt": pilot_data.ORDER_SALT,
        "conflicts": conflicts,
        "source_overlap": {"groups": [], "group_count": 0, "painting_row_count": 0, "source_row_count": 0},
        "custody_scope": design_document["custody_scope"],
    }
    public = {
        **common,
        "schema": "kbound-painting-pilot-public/2",
        "design_sha256": pilot_data.DESIGN_SHA256,
        "amendment_sha256": pins["amendment_sha256"],
        "packing": packing,
        "rows": [{k: r[k] for k in PUBLIC_FIELDS} for r in retained],
    }
    quarantine = {
        "schema": "kbound-painting-pilot-quarantine/2",
        "status": "SOURCE_OVERLAP_QUARANTINED_NOT_SCORED",
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "parents": parents,
        "design_sha256": pilot_data.DESIGN_SHA256,
        "amendment_sha256": pins["amendment_sha256"],
        "rule": amendment_document["quarantine_rule"],
        "rows": excluded,
        "groups": overlaps,
        "row_count": len(excluded),
        "group_count": len(overlaps),
        "source_row_count": overlap_counts["source_row_count"],
        "original_v1_stop_preserved": True,
        "raw_files_deleted_or_modified": False,
    }
    summary_v2 = {
        **common,
        "schema": "kbound-painting-pilot-summary/2",
        "identity": identity,
        "design_sha256": pilot_data.DESIGN_SHA256,
        "amendment_sha256": pins["amendment_sha256"],
        "counts": counts,
        "row_count": len(retained),
        "group_count": len(groups),
        "original_row_count": len(rows),
        "original_group_count": len(original_groups),
        "quarantined_row_count": len(excluded),
        "quarantined_group_count": len(overlaps),
        "quarantined_source_row_count": overlap_counts["source_row_count"],
        "conflict_group_count": conflicts["group_count"],
        "conflict_row_count": conflicts["row_count"],
        "source_overlap_group_count": 0,
        "source_overlap_row_count": 0,
        "geometry_status": "MINIMUM_COMPLETE_CELLS_MET" if eligible else "INCONCLUSIVE_DATA_GEOMETRY",
        "original_v1_stop_preserved": True,
        "raw_files_deleted_or_modified": False,
    }
    documents = {
        "inventory.json": inventory,
        "public_manifest.json": public,
        "labels.json": {
            "schema": "kbound-painting-pilot-labels/2",
            "labels": {r["sample_id"]: r["label"] for r in retained},
        },
        "quarantine.json": quarantine,
    }
    return documents, summary_v2


def prepare_painting_v2(
    v1_dir: Path,
    source_inventory: Path,
    design: Path,
    amendment: Path,
    v1_stop: Path,
    output_dir: Path,
    *,
    synthetic_identity: SyntheticIdentity | None = None,
) -> dict[str, Any]:
    """Validate pinned parents, repack only retained groups, publish summary last."""
    output_dir = _safe_path(output_dir, new=True)
    documents, summary = _derive(v1_dir, source_inventory, design, amendment, v1_stop, synthetic_identity)
    _safe_path(output_dir, new=True)
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    hashes = {}
    for name, document in documents.items():
        source_data.write_new_json(output_dir / name, document)
        hashes[name] = source_data.file_hash(output_dir / name)
    summary["output_sha256"] = hashes
    source_data.write_new_json(output_dir / "summary.json", summary)
    return summary


def verify_prepared_painting_v2(
    prepared_dir: Path,
    v1_dir: Path,
    source_inventory: Path,
    design: Path,
    amendment: Path,
    v1_stop: Path,
    *,
    expected_summary_sha256: str,
    synthetic_identity: SyntheticIdentity | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only replay of the exact pinned-parent derivation, including quarantine."""
    prepared_dir = _safe_path(prepared_dir, directory=True)
    summary = _read_pinned(prepared_dir / "summary.json", expected_summary_sha256)
    documents, expected = _derive(v1_dir, source_inventory, design, amendment, v1_stop, synthetic_identity)
    hashes = {
        name: hashlib.sha256(source_data.json_bytes(document)).hexdigest() for name, document in documents.items()
    }
    expected["output_sha256"] = hashes
    _equal(summary, expected, "V2 summary and complete parent replay")
    for name, digest in hashes.items():
        _read_pinned(prepared_dir / name, digest)
    return documents["inventory.json"], documents["public_manifest.json"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1-dir", "source-inventory", "design", "amendment", "v1-stop", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    summary = prepare_painting_v2(
        args.v1_dir, args.source_inventory, args.design, args.amendment, args.v1_stop, args.output_dir
    )
    return 0 if summary["eligible_for_pilot"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
