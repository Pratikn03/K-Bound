"""Validate the pinned clipart source without extracting other archive members.

Exact decoded-content groups do not establish near-duplicate or creator independence.
The production CLI has no identity override. Explicit library identities support tests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import struct
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

SPLIT_SALT = "kbound-domainnet-source-v1:decoded-group:20260905:"
SUBSET_COMMIT = "ee04dddc846eec12eff1cbf2b0fa746945c2b173"
MAX_MEMBERS = 100_000
MAX_IMAGE_BYTES = 50 * 1024**2
MAX_TOTAL_BYTES = 20 * 1024**3
SOURCE_POLICY_V2_SHA256 = "007f2bfb12e8d74f99332d28f675006c493813119275fe47e4f6631d20663e53"


@dataclass(frozen=True)
class SourceIdentity:
    archive_sha256: str = "f8f248a3e8bbc9868c523ae24e6601b4c2ad102afc60c0efeb4f3255df22354d"
    list_sha256: str = "8f241bd8400c2e1676c8ff96a63b8917e488cc318cbcc4e97883d15567d03d92"
    list_git_blob: str = "a1aeea574d7a93b2bf54b05e3c61a8f6b6be6953"
    class_count: int = 126


def stream_hash(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_hash(stream)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    # Retain the private staging inode for inspection on both success and failure.
    # Linking is atomic and cannot replace caller files, unlike rename on POSIX.
    staged = path.with_name("." + path.name + ".staged")
    with staged.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    staged.chmod(0o444)
    os.link(staged, path)


def strict_json(path: Path) -> Any:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def load_source_policy(policy_path: Path | None, expected_policy_sha256: str | None) -> dict[str, Any] | None:
    """Only the exact approved source-informed V2 policy can select the V2 branch."""
    if (policy_path is None) != (expected_policy_sha256 is None):
        raise ValueError("source policy path and externally expected SHA256 must be supplied together")
    if policy_path is None:
        return None
    if expected_policy_sha256 != SOURCE_POLICY_V2_SHA256:
        raise ValueError("source policy hash is not the approved V2 authority")
    raw = policy_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_POLICY_V2_SHA256:
        raise ValueError("source policy SHA256 mismatch")
    # Exact byte identity pins every field, including amendment context and rationale.
    policy = json.loads(raw)
    if policy["schema"] != "kbound-domainnet-source-policy/2" or policy["split_salt"] != SPLIT_SALT:
        raise ValueError("source policy schema/salt mismatch")
    return {"sha256": SOURCE_POLICY_V2_SHA256, "document": policy}


def conflict_manifest(rows: list[dict[str, Any]], class_count: int) -> dict[str, Any]:
    """Canonical full cross-label content-group disclosure; no row modifications."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    conflicts = []
    for group_id, members in sorted(grouped.items()):
        labels = sorted({row["label"] for row in members})
        if len(labels) > 1:
            conflicts.append(
                {
                    "group_id": group_id,
                    "split": members[0]["split"],
                    "labels": labels,
                    "encoded_sha256s": sorted({row["image_sha256"] for row in members}),
                    "multiplicity": len(members),
                    "label_multiplicity": {
                        str(label): sum(row["label"] == label for row in members) for label in labels
                    },
                    "members": members,
                }
            )

    def counts(groups: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "groups": len(groups),
            "rows": sum(group["multiplicity"] for group in groups),
            "per_class": {
                str(label): {
                    "groups": sum(label in group["labels"] for group in groups),
                    "rows": sum(group["label_multiplicity"].get(str(label), 0) for group in groups),
                }
                for label in range(class_count)
            },
        }

    aggregate = counts(conflicts)
    aggregate["per_split"] = {
        split: counts([group for group in conflicts if group["split"] == split])
        for split in ("source_fit", "source_monitor")
    }
    return {"groups": conflicts, "counts": aggregate}


def canonical_path(name: str, *, directory: bool = False) -> str:
    candidate = name[:-1] if directory and name.endswith("/") else name
    parts = candidate.split("/")
    if (
        not candidate
        or "\\" in candidate
        or "\x00" in candidate
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0] != "clipart"
        or unicodedata.normalize("NFC", candidate) != candidate
        or any(ord(char) < 32 for char in candidate)
    ):
        raise ValueError(f"noncanonical source path: {name!r}")
    return candidate


def validate_members(archive: zipfile.ZipFile) -> set[str]:
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS or sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
        raise ValueError("archive exceeds declared resource bounds")
    names: set[str] = set()
    aliases: set[str] = set()
    for info in infos:
        path = canonical_path(info.filename, directory=info.is_dir())
        alias = path.casefold()
        kind = stat.S_IFMT(info.external_attr >> 16)
        allowed = {0, stat.S_IFDIR} if info.is_dir() else {0, stat.S_IFREG}
        if (
            alias in aliases
            or info.orig_filename != info.filename
            or kind not in allowed
            or info.flag_bits & 1
            or info.file_size > MAX_IMAGE_BYTES
            or (info.is_dir() and info.file_size != 0)
        ):
            raise ValueError(f"invalid or aliased ZIP member: {info.filename!r}")
        aliases.add(alias)
        if not info.is_dir():
            names.add(path)
    return names


def parse_source_list(raw: bytes, identity: SourceIdentity) -> list[tuple[str, int]]:
    if hashlib.sha256(raw).hexdigest() != identity.list_sha256:
        raise ValueError("source list SHA256 mismatch")
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != identity.list_git_blob:
        raise ValueError("source list Git blob mismatch")
    rows = []
    aliases = set()
    class_labels: dict[str, int] = {}
    label_classes: dict[int, str] = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) != 2 or re.fullmatch(r"0|[1-9][0-9]*", fields[1]) is None:
            raise ValueError("source list requires path and canonical integer label")
        path = canonical_path(fields[0])
        if len(path.split("/")) != 3:
            raise ValueError("source path must contain domain/class/image")
        label = int(fields[1])
        class_name = path.split("/")[1]
        if path.casefold() in aliases or not 0 <= label < identity.class_count:
            raise ValueError("duplicate path or out of range label")
        if class_name in class_labels and class_labels[class_name] != label:
            raise ValueError("class has inconsistent labels")
        if label in label_classes and label_classes[label] != class_name:
            raise ValueError("label has inconsistent class names")
        aliases.add(path.casefold())
        class_labels[class_name] = label
        label_classes[label] = class_name
        rows.append((path, label))
    if set(label_classes) != set(range(identity.class_count)):
        raise ValueError("source labels do not cover exact required class range")
    if len({name.casefold() for name in class_labels}) != len(class_labels):
        raise ValueError("aliased class names")
    return sorted(rows)


def decode_rgb(raw: bytes) -> Image.Image:
    # PIL warnings propagate; callers can and should run with -W error.
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
        rgb.load()
    return rgb


def build_inventory(
    archive_path: Path,
    raw_list: bytes,
    identity: SourceIdentity,
    *,
    policy_path: Path | None = None,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    policy = load_source_policy(policy_path, expected_policy_sha256)
    rows = []
    groups: dict[str, int] = {}
    with archive_path.open("rb") as handle:
        actual_hash = stream_hash(handle)
        if actual_hash != identity.archive_sha256:
            raise ValueError("source archive SHA256 mismatch")
        listed = parse_source_list(raw_list, identity)
        with zipfile.ZipFile(handle) as archive:
            members = validate_members(archive)
            if any(path not in members for path, _ in listed):
                raise ValueError("listed source image missing from archive")
            for path, label in listed:
                raw = archive.read(path)  # ZIP CRC checked by read; no extraction.
                rgb = decode_rgb(raw)
                group = hashlib.sha256(struct.pack(">QQ", *rgb.size) + rgb.tobytes()).hexdigest()
                if policy is None and group in groups and groups[group] != label:
                    raise ValueError("decoded-content duplicate has conflicting labels")
                groups[group] = label
                rank = int(hashlib.sha256((SPLIT_SALT + group).encode()).hexdigest(), 16)
                split = "source_monitor" if rank < (1 << 256) // 10 else "source_fit"
                rows.append(
                    {
                        "path": path,
                        "label": label,
                        "split": split,
                        "group_id": group,
                        "image_sha256": hashlib.sha256(raw).hexdigest(),
                        "width": rgb.width,
                        "height": rgb.height,
                    }
                )
    counts = {split: sum(row["split"] == split for row in rows) for split in ("source_fit", "source_monitor")}
    if not all(counts.values()) or {row["label"] for row in rows if row["split"] == "source_fit"} != set(
        range(identity.class_count)
    ):
        raise ValueError("empty split or source-fit lacks a required class")
    representation = {
        split: {
            str(label): sum(row["split"] == split and row["label"] == label for row in rows)
            for label in range(identity.class_count)
        }
        for split in counts
    }
    inventory = {
        "schema": "kbound-domainnet-source-inventory/1",
        "dataset": "DomainNet-126",
        "domain": "clipart",
        "role": "source",
        "status": "SOURCE_ONLY_PREPARED_NOT_SCIENTIFIC_LOCK",
        "subset_commit": SUBSET_COMMIT,
        "identity": asdict(identity),
        "split_salt": SPLIT_SALT,
        "split_rule": "sha256(salt + group hex) < floor(2^256 / 10) => source_monitor",
        "grouping_scope": "exact decoded RGB bytes and dimensions within listed source only; no near-duplicate or creator-independence audit",
        "bounds": {"max_members": MAX_MEMBERS, "max_image_bytes": MAX_IMAGE_BYTES, "max_total_bytes": MAX_TOTAL_BYTES},
        "counts": counts,
        "class_representation": representation,
        "decoded_groups": len(groups),
        "rows": rows,
    }
    if policy is not None:
        inventory["schema"] = "kbound-domainnet-source-inventory/2"
        inventory["source_policy"] = policy
        inventory["annotation_status"] = "OFFICIAL_ANNOTATIONS_PRESERVED_WITH_COMPLETE_CONFLICT_DISCLOSURE"
        inventory["conflict_manifest"] = conflict_manifest(rows, identity.class_count)
    return inventory


def prepare_source(
    archive: Path,
    source_list: Path,
    output_dir: Path,
    *,
    identity: SourceIdentity = SourceIdentity(),
    policy_path: Path | None = None,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    load_source_policy(policy_path, expected_policy_sha256)
    raw = source_list.read_bytes()
    inventory = build_inventory(
        archive, raw, identity, policy_path=policy_path, expected_policy_sha256=expected_policy_sha256
    )
    output_dir.mkdir(parents=False, exist_ok=False)
    # Success is published last. Partial I/O failures never leave a success receipt.
    with (output_dir / "source-list.txt").open("xb") as stream:
        stream.write(raw)
    (output_dir / "source-list.txt").chmod(0o444)
    write_new_json(output_dir / "inventory.json", inventory)
    summary = {key: value for key, value in inventory.items() if key != "rows"}
    summary["inventory_sha256"] = file_hash(output_dir / "inventory.json")
    write_new_json(output_dir / "summary.json", summary)
    return summary


def load_verified_source(
    inventory_dir: Path,
    archive: Path,
    *,
    identity: SourceIdentity = SourceIdentity(),
    policy_path: Path | None = None,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    policy = load_source_policy(policy_path, expected_policy_sha256)
    inventory = strict_json(inventory_dir / "inventory.json")
    expected_schema = "kbound-domainnet-source-inventory/1" if policy is None else "kbound-domainnet-source-inventory/2"
    if inventory.get("schema") != expected_schema or inventory.get("source_policy") != policy:
        raise ValueError("source inventory version/policy mismatch")
    summary = strict_json(inventory_dir / "summary.json")
    if summary.get("inventory_sha256") != file_hash(inventory_dir / "inventory.json"):
        raise ValueError("inventory SHA256 mismatch")
    rebuilt = build_inventory(
        archive,
        (inventory_dir / "source-list.txt").read_bytes(),
        identity,
        policy_path=policy_path,
        expected_policy_sha256=expected_policy_sha256,
    )
    expected_summary = {key: value for key, value in rebuilt.items() if key != "rows"}
    expected_summary["inventory_sha256"] = file_hash(inventory_dir / "inventory.json")
    if inventory != rebuilt or summary != expected_summary:
        raise ValueError("prepared source inventory differs from verified source reconstruction")
    return rebuilt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path)
    parser.add_argument("--expected-policy-sha256")
    args = parser.parse_args(argv)
    prepare_source(
        args.archive,
        args.source_list,
        args.output_dir,
        policy_path=args.source_policy,
        expected_policy_sha256=args.expected_policy_sha256,
    )


if __name__ == "__main__":
    main()
