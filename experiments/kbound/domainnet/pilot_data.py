"""Prepare only the identity-pinned painting development pilot, without extraction.

The inventory belongs to the ingestion broker; labels.json belongs to the scorer.
Public sample aliases are not cryptographic confidentiality or OS custody claims.
Exact RGB groups do not establish creator or near-duplicate independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import struct
import unicodedata
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.kbound.domainnet import source_data

DESIGN_SHA256 = "37ea2956edf627f01f13116c845fca0ab13c3095463bd690420fd3be005a2a31"
ARCHIVE_SHA256 = "fa47e6d405503ea0286cabd767f176bd30e988b63ddd7b9db16cf030c9770015"
LIST_SHA256 = "eafd460aa2fcc401b179ec3ca30f2519bf5977f94479be8d4def9cfeeba995e0"
LIST_GIT_BLOB = "3d6be286e06ea232c62552e7f6f1c64e19da1311"
SOURCE_INVENTORY_SHA256 = "5fffb4773d3447c02a25b231e5567a2bf2d905dd9756bc3fad9a86ee8f4d1a26"
PARTITION_SALT = "kbound-painting-development-pilot-v1:partition:20260905:"
ORDER_SALT = "kbound-painting-development-pilot-v1:order:20260905:"
PARTITIONS = ("DEV_fit", "DEV_radius", "DEV_check")
WINDOWS = ("U", "V", "E")
WINDOW_MINIMA = {"U": 64, "V": 64, "E": 128}
MINIMUM_CELLS = {"DEV_fit": 40, "DEV_radius": 10, "DEV_check": 20}


@dataclass(frozen=True)
class SyntheticIdentity:
    """Library-only small-class fixtures; cannot select any real input identity."""

    archive_sha256: str
    list_sha256: str
    list_git_blob: str
    source_inventory_sha256: str
    class_count: int

    def __post_init__(self) -> None:
        if type(self.class_count) is not int or not 1 <= self.class_count < 126:
            raise ValueError("synthetic identities require a reduced integer class count")
        for name, real, length in (
            ("archive_sha256", ARCHIVE_SHA256, 64),
            ("list_sha256", LIST_SHA256, 64),
            ("list_git_blob", LIST_GIT_BLOB, 40),
            ("source_inventory_sha256", SOURCE_INVENTORY_SHA256, 64),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None or value == real:
                raise ValueError("synthetic identity is malformed or uses a real identity")


def partition_for_rank(rank: int) -> str:
    """Apply the reviewed literal integer boundaries without floating point."""
    if type(rank) is not int or not 0 <= rank < 2**256:
        raise ValueError("partition rank outside SHA256 range")
    if rank < (3 * 2**256) // 5:
        return "DEV_fit"
    return "DEV_radius" if rank < (4 * 2**256) // 5 else "DEV_check"


def partition_for_group(group_id: str) -> str:
    return partition_for_rank(int(hashlib.sha256((PARTITION_SALT + group_id).encode()).hexdigest(), 16))


def pack_ordered_groups(groups: list[list[str]], minima: dict[str, int] | None = None) -> dict[str, Any]:
    """Pure greedy whole-group packing; incomplete final cells stay unscored.

    Explicit minima support hand-checkable pure examples. Preparation always uses
    the reviewed production minima, including when identities are synthetic.
    """
    minima = WINDOW_MINIMA if minima is None else minima
    if set(minima) != set(WINDOWS) or any(type(v) is not int or v <= 0 for v in minima.values()):
        raise ValueError("window minima must be positive U/V/E integers")
    cells: list[dict[str, list[str]]] = []
    pending: dict[str, list[str]] = {w: [] for w in WINDOWS}
    stage = 0
    for members in groups:
        if not members:
            raise ValueError("empty decoded group")
        window = WINDOWS[stage]
        pending[window].extend(members)
        if len(pending[window]) >= minima[window]:
            stage += 1
            if stage == len(WINDOWS):
                cells.append(pending)
                pending = {w: [] for w in WINDOWS}
                stage = 0
    return {"cells": cells, "tail": pending}


def canonical_painting_path(name: str, *, directory: bool = False) -> str:
    candidate = name[:-1] if directory and name.endswith("/") else name
    parts = candidate.split("/")
    if (
        not candidate
        or parts[0] != "painting"
        or (len(parts) not in {1, 2} if directory else len(parts) != 3)
        or "\\" in candidate
        or ":" in candidate
        or any(part in {"", ".", ".."} or part.endswith((".", " ")) for part in parts)
        or unicodedata.normalize("NFC", candidate) != candidate
        or any(unicodedata.category(char).startswith("C") or char.isspace() for char in candidate)
    ):
        raise ValueError(f"noncanonical painting path: {name!r}")
    return candidate


def _record_aliases(path: str, aliases: dict[str, str]) -> None:
    parts = path.split("/")
    for end in range(1, len(parts) + 1):
        prefix = "/".join(parts[:end])
        alias = prefix.casefold()
        if alias in aliases and aliases[alias] != prefix:
            raise ValueError("case-aliased path or class directory")
        aliases[alias] = prefix


def validate_painting_members(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > source_data.MAX_MEMBERS or sum(i.file_size for i in infos) > source_data.MAX_TOTAL_BYTES:
        raise ValueError("archive exceeds declared resource bounds")
    names: set[str] = set()
    files: set[str] = set()
    aliases: dict[str, str] = {}
    for info in infos:
        path = canonical_painting_path(info.filename, directory=info.is_dir())
        _record_aliases(path, aliases)
        kind = stat.S_IFMT(info.external_attr >> 16)
        allowed = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
        if (
            path in names
            or info.orig_filename != info.filename
            or kind not in allowed
            or info.flag_bits & (1 | 64 | 8192)
            or not 0 <= info.file_size <= source_data.MAX_IMAGE_BYTES
            or (info.is_dir() and info.file_size != 0)
        ):
            raise ValueError(f"invalid or duplicate ZIP member: {info.filename!r}")
        names.add(path)
        if not info.is_dir():
            files.add(path)
    return files


def parse_painting_list(raw: bytes, identity: dict[str, Any]) -> tuple[list[tuple[str, int]], dict[str, int]]:
    if hashlib.sha256(raw).hexdigest() != identity["list_sha256"]:
        raise ValueError("painting list SHA256 mismatch")
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != identity["list_git_blob"]:
        raise ValueError("painting list Git blob mismatch")
    lines = raw.decode("utf-8", errors="strict").split("\n")
    if lines[-1] == "":
        lines.pop()
    rows = []
    names: set[str] = set()
    aliases: dict[str, str] = {}
    class_map: dict[str, int] = {}
    label_classes: dict[int, str] = {}
    for line in lines:
        matched = re.fullmatch(r"([^\s]+)[ \t]+(0|[1-9][0-9]*)", line)
        if matched is None:
            raise ValueError("painting list requires canonical path and integer label")
        path = canonical_painting_path(matched[1])
        _record_aliases(path, aliases)
        label = int(matched[2])
        name = path.split("/")[1]
        if path in names or not 0 <= label < identity["class_count"]:
            raise ValueError("duplicate painting path or out of range label")
        if (name in class_map and class_map[name] != label) or (
            label in label_classes and label_classes[label] != name
        ):
            raise ValueError("painting class mapping is inconsistent")
        names.add(path)
        class_map[name] = label
        label_classes[label] = name
        rows.append((path, label))
    if set(label_classes) != set(range(identity["class_count"])):
        raise ValueError("painting labels do not cover the exact required class range")
    return rows, class_map


def _source_reference(path: Path, identity: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity["source_inventory_sha256"]:
        raise ValueError("source inventory SHA256 mismatch")
    inventory = json.loads(raw)
    if (
        inventory.get("schema") != "kbound-domainnet-source-inventory/2"
        or inventory.get("domain") != "clipart"
        or inventory.get("role") != "source"
        or inventory.get("status") != "SOURCE_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK"
        or inventory.get("identity", {}).get("class_count") != identity["class_count"]
    ):
        raise ValueError("invalid source inventory authority")
    class_map: dict[str, int] = {}
    label_classes: dict[int, str] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in inventory["rows"]:
        source_path = source_data.canonical_path(row["path"])
        if len(source_path.split("/")) != 3:
            raise ValueError("invalid source inventory path")
        name, label, group = source_path.split("/")[1], row["label"], row["group_id"]
        if (
            type(label) is not int
            or not 0 <= label < identity["class_count"]
            or re.fullmatch("[0-9a-f]{64}", group) is None
        ):
            raise ValueError("invalid source inventory label or group")
        if (name in class_map and class_map[name] != label) or (
            label in label_classes and label_classes[label] != name
        ):
            raise ValueError("inconsistent source class mapping")
        class_map[name], label_classes[label] = label, name
        groups.setdefault(group, []).append(row)
    if set(label_classes) != set(range(identity["class_count"])):
        raise ValueError("source class map lacks required classes")
    return class_map, groups


def _packing(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    by_id = {row["sample_id"]: row for row in rows}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    packing: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for partition in PARTITIONS:
        order = sorted(
            (group for group in grouped if partition_for_group(group) == partition),
            key=lambda group: (hashlib.sha256((ORDER_SALT + group).encode()).hexdigest(), group),
        )
        packed = pack_ordered_groups([[r["sample_id"] for r in grouped[group]] for group in order])
        packing[partition] = packed
        for cell_index, cell in enumerate(packed["cells"]):
            for window in WINDOWS:
                for sample_id in cell[window]:
                    by_id[sample_id].update(partition=partition, cell=cell_index, window=window)
        for window in WINDOWS:
            for sample_id in packed["tail"][window]:
                by_id[sample_id].update(partition=partition, cell=None, window="TAIL")
        cell_rows = [{w: len(cell[w]) for w in WINDOWS} for cell in packed["cells"]]
        scored_rows = sum(sum(cell.values()) for cell in cell_rows)
        tail_counts = {w: len(packed["tail"][w]) for w in WINDOWS}
        counts[partition] = {
            "complete_cells": len(cell_rows),
            "cell_rows": cell_rows,
            "scored_rows": scored_rows,
            "tail_rows": sum(tail_counts.values()),
            "tail_window_rows": tail_counts,
            "total_rows": scored_rows + sum(tail_counts.values()),
            "decoded_groups": len(order),
        }
    groups = []
    for group, members in sorted(grouped.items()):
        labels = sorted({r["label"] for r in members})
        groups.append(
            {
                "group_id": group,
                "members": members,
                "multiplicity": len(members),
                "labels": labels,
                "label_multiplicity": {str(label): sum(r["label"] == label for r in members) for label in labels},
                "encoded_sha256s": sorted({r["image_sha256"] for r in members}),
            }
        )
    return packing, counts, groups


def prepare_painting(
    archive: Path,
    painting_list: Path,
    source_inventory: Path,
    design: Path,
    output_dir: Path,
    *,
    synthetic_identity: SyntheticIdentity | None = None,
) -> dict[str, Any]:
    """Validate every listed image, freeze geometry, then publish the summary last."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if synthetic_identity is not None and type(synthetic_identity) is not SyntheticIdentity:
        raise ValueError("explicit SyntheticIdentity required")
    if synthetic_identity is not None:
        synthetic_identity.__post_init__()
    identity = (
        asdict(synthetic_identity)
        if synthetic_identity is not None
        else {
            "archive_sha256": ARCHIVE_SHA256,
            "list_sha256": LIST_SHA256,
            "list_git_blob": LIST_GIT_BLOB,
            "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "class_count": 126,
        }
    )
    design_raw = design.read_bytes()
    if hashlib.sha256(design_raw).hexdigest() != DESIGN_SHA256:
        raise ValueError("reviewed pilot design SHA256 mismatch")
    design_document = json.loads(design_raw)
    source_classes, source_groups = _source_reference(source_inventory, identity)
    listed, class_map = parse_painting_list(painting_list.read_bytes(), identity)
    if class_map != source_classes:
        raise ValueError("painting class map differs from hash-bound source class map")
    rows: list[dict[str, Any]] = []
    with archive.open("rb") as stream:
        if source_data.stream_hash(stream) != identity["archive_sha256"]:
            raise ValueError("painting archive SHA256 mismatch")
        with zipfile.ZipFile(stream) as zipped:
            members = validate_painting_members(zipped)
            if any(path not in members for path, _ in listed):
                raise ValueError("listed painting image missing from archive")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for index, (path, label) in enumerate(listed):
                    raw = zipped.read(path)  # Complete read checks the member CRC; never extracts files.
                    rgb = source_data.decode_rgb(raw)
                    try:
                        group = hashlib.sha256(struct.pack(">QQ", *rgb.size) + rgb.tobytes()).hexdigest()
                        rows.append(
                            {
                                "sample_id": hashlib.sha256(
                                    ("kbound-painting-pilot-v1:sample:" + path).encode()
                                ).hexdigest(),
                                "official_index": index,
                                "path": path,
                                "label": label,
                                "image_sha256": hashlib.sha256(raw).hexdigest(),
                                "group_id": group,
                                "width": rgb.width,
                                "height": rgb.height,
                            }
                        )
                    finally:
                        rgb.close()
    packing, counts, groups = _packing(rows)
    conflict_groups = [group for group in groups if len(group["labels"]) > 1]
    overlaps = [
        {"group_id": g["group_id"], "painting_members": g["members"], "source_members": source_groups[g["group_id"]]}
        for g in groups
        if g["group_id"] in source_groups
    ]
    eligible = not overlaps and all(counts[p]["complete_cells"] >= MINIMUM_CELLS[p] for p in PARTITIONS)
    status = (
        "STOP_SOURCE_OVERLAP"
        if overlaps
        else ("DEV_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK" if eligible else "INCONCLUSIVE_DATA_GEOMETRY")
    )
    scope = "SYNTHETIC_TEST" if synthetic_identity is not None else "PAINTING_DEVELOPMENT_ONLY"
    inventory = {
        "schema": "kbound-painting-pilot-inventory/1",
        "status": status,
        "eligible_for_pilot": eligible,
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "domain": "painting",
        "role": "development",
        "identity": identity,
        "design": {"sha256": DESIGN_SHA256, "document": design_document},
        "class_map": class_map,
        "rows": rows,
        "groups": groups,
        "packing": packing,
        "counts": counts,
        "partition_salt": PARTITION_SALT,
        "group_order_salt": ORDER_SALT,
        "conflicts": {
            "groups": conflict_groups,
            "group_count": len(conflict_groups),
            "row_count": sum(g["multiplicity"] for g in conflict_groups),
        },
        "source_overlap": {
            "groups": overlaps,
            "group_count": len(overlaps),
            "painting_row_count": sum(len(g["painting_members"]) for g in overlaps),
            "source_row_count": sum(len(g["source_members"]) for g in overlaps),
        },
        "custody_scope": design_document["custody_scope"],
    }
    public = {
        "schema": "kbound-painting-pilot-public/1",
        "status": status,
        "eligible_for_pilot": eligible,
        "execution_scope": scope,
        "design_sha256": DESIGN_SHA256,
        "packing": packing,
        "rows": [{k: v for k, v in row.items() if k not in {"path", "label"}} for row in rows],
    }
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, value in (
        ("inventory.json", inventory),
        ("public_manifest.json", public),
        ("labels.json", {r["sample_id"]: r["label"] for r in rows}),
    ):
        source_data.write_new_json(output_dir / name, value)
    summary = {
        "schema": "kbound-painting-pilot-summary/1",
        "status": status,
        "eligible_for_pilot": eligible,
        "eligible_for_confirmatory": False,
        "execution_scope": scope,
        "identity": identity,
        "design_sha256": DESIGN_SHA256,
        "counts": counts,
        "row_count": len(rows),
        "group_count": len(groups),
        "conflict_group_count": len(conflict_groups),
        "source_overlap_group_count": len(overlaps),
        "output_sha256": {
            name: source_data.file_hash(output_dir / name)
            for name in ("inventory.json", "public_manifest.json", "labels.json")
        },
    }
    source_data.write_new_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--painting-list", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = prepare_painting(args.archive, args.painting_list, args.source_inventory, args.design, args.output_dir)
    return 0 if summary["eligible_for_pilot"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
