#!/usr/bin/env python3
"""Validate static bindings for the reference-source-then-KGA campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
POLICY = "published_reference_source_then_kga"
DATASET_NAMES = frozenset(
    {
        "cifar10c",
        "cifar101",
        "imagenetc",
        "imagenetr",
        "camelyon17",
        "rxrx1",
        "pacs",
        "iwildcam",
        "officehome",
    }
)
_SHA256_CHARS = frozenset("0123456789abcdef")
_HASH_CHUNK_BYTES = 1024 * 1024
_RXRX1_CHECKPOINT_COUNT = 5


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_https_url(value: object) -> bool:
    if not _is_nonempty_string(value):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and bool(parsed.netloc)
            and parsed.hostname is not None
            and not any(character.isspace() for character in parsed.netloc)
        )
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path_key(path: str) -> str:
    return os.path.normpath(path)


def _check_artifact(
    artifact: object,
    location: str,
    blockers: list[str],
    *,
    expected_role: str | None = None,
) -> str | None:
    if not isinstance(artifact, dict):
        blockers.append(f"{location} must be an artifact object")
        return None

    role = artifact.get("role")
    if not _is_nonempty_string(role):
        blockers.append(f"{location} artifact role must be a nonempty string")
    elif expected_role is not None and role != expected_role:
        blockers.append(f"{location} role must be {expected_role}")

    path_value = artifact.get("path")
    if not _is_nonempty_string(path_value):
        blockers.append(f"{location} artifact path must be a nonempty absolute string")
        path = None
    else:
        path = Path(path_value)
        if not path.is_absolute():
            blockers.append(f"{location} artifact path must be absolute: {path_value}")
            path = None

    expected_sha256 = artifact.get("sha256")
    sha256_valid = (
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in _SHA256_CHARS for character in expected_sha256)
    )
    if not sha256_valid:
        blockers.append(f"{location} sha256 must be 64 lowercase hexadecimal characters")

    if path is None:
        return None

    try:
        if path.is_symlink():
            blockers.append(f"{location} artifact path must not be a symlink: {path}")
            return _artifact_path_key(path_value)
        if not path.is_file():
            blockers.append(f"{location} artifact path is not a regular file: {path}")
            return _artifact_path_key(path_value)
        if not os.access(path, os.R_OK):
            blockers.append(f"{location} artifact file is not readable: {path}")
            return _artifact_path_key(path_value)
        if sha256_valid:
            try:
                actual_sha256 = _sha256_file(path)
            except OSError as exc:
                blockers.append(f"{location} artifact file is not readable: {path}: {exc}")
            else:
                if actual_sha256 != expected_sha256:
                    blockers.append(f"{location} artifact SHA-256 mismatch: {path}")
    except (OSError, ValueError) as exc:
        blockers.append(f"{location} artifact path cannot be inspected: {path}: {exc}")

    return _artifact_path_key(path_value)


def _check_output_root(value: object, blockers: list[str]) -> None:
    if not _is_nonempty_string(value):
        blockers.append("output_root must be a nonempty absolute string")
        return

    path = Path(value)
    if not path.is_absolute():
        blockers.append("output_root must be absolute")
        return

    try:
        symlink_part = next((part for part in (path, *path.parents) if part.is_symlink()), None)
        if symlink_part is not None:
            blockers.append(f"output_root has a symlink ancestor: {symlink_part}")
        if os.path.lexists(path):
            blockers.append(f"output_root already exists: {path}")

        parent = path.parent
        if not parent.is_dir():
            blockers.append(f"output_root parent must be an existing directory: {parent}")
        elif not os.access(parent, os.W_OK):
            blockers.append(f"output_root parent is not writable: {parent}")
    except (OSError, ValueError) as exc:
        blockers.append(f"output_root cannot be inspected: {path}: {exc}")


def _check_reference(value: object, location: str, blockers: list[str]) -> None:
    if not isinstance(value, dict):
        blockers.append(f"{location}.reference must be an object")
        return

    url = value.get("url")
    if not _is_https_url(url):
        blockers.append(f"{location}.reference.url must be a nonempty HTTPS URL")

    revision = value.get("revision")
    if not _is_nonempty_string(revision):
        blockers.append(f"{location}.reference.revision must be a nonempty string")


def _check_row(row: object, index: int, blockers: list[str]) -> str | None:
    location = f"datasets[{index}]"
    if not isinstance(row, dict):
        blockers.append(f"{location} must be an object")
        return None

    name = row.get("name")
    if not _is_nonempty_string(name):
        blockers.append(f"{location}.name must be a nonempty string")
        row_location = location
        valid_name = None
    else:
        row_location = f"dataset {name}"
        valid_name = name
        if name not in DATASET_NAMES:
            blockers.append(f"unknown dataset: {name}")

    declared_blockers = row.get("blockers")
    if not isinstance(declared_blockers, list):
        blockers.append(f"{row_location}.blockers must be a list of nonempty strings")
    else:
        for blocker_index, blocker in enumerate(declared_blockers):
            if not _is_nonempty_string(blocker):
                blockers.append(
                    f"{row_location}.blockers[{blocker_index}] must be a nonempty string"
                )
            else:
                blockers.append(f"{row_location} blocker: {blocker}")

    _check_reference(row.get("reference"), row_location, blockers)

    source_ready = row.get("source_ready")
    if type(source_ready) is not bool:
        blockers.append(f"{row_location}.source_ready must be boolean true")
    elif not source_ready:
        blockers.append(f"{row_location}.source_ready must be true")

    seen_paths: set[str] = set()
    roles: set[str] = set()
    checkpoint_count = 0
    checkpoint_sha256s: list[str] = []
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(f"{row_location}.artifacts must be a nonempty list")
    else:
        for artifact_index, artifact in enumerate(artifacts):
            artifact_location = f"{row_location}.artifacts[{artifact_index}]"
            path_key = _check_artifact(artifact, artifact_location, blockers)
            if isinstance(artifact, dict) and _is_nonempty_string(artifact.get("role")):
                roles.add(artifact["role"])
                sha256 = artifact.get("sha256")
                if artifact["role"] == "checkpoint":
                    checkpoint_count += 1
                    if isinstance(sha256, str):
                        checkpoint_sha256s.append(sha256)
            if path_key is not None:
                if path_key in seen_paths:
                    blockers.append(f"{row_location} has duplicate artifact path: {path_key}")
                seen_paths.add(path_key)

    for required_role in ("checkpoint", "data_manifest"):
        if required_role not in roles:
            blockers.append(f"{row_location} missing artifact role: {required_role}")

    if name == "rxrx1":
        if checkpoint_count != _RXRX1_CHECKPOINT_COUNT:
            blockers.append(
                f"{row_location} must have exactly five checkpoint artifacts"
            )
        if len(checkpoint_sha256s) != len(set(checkpoint_sha256s)):
            blockers.append(
                f"{row_location} checkpoint SHA-256 identities must be distinct"
            )

    receipt_path = _check_artifact(
        row.get("kga_partition_receipt"),
        f"{row_location}.kga_partition_receipt",
        blockers,
        expected_role="kga_partition_receipt",
    )
    if receipt_path is not None and receipt_path in seen_paths:
        blockers.append(f"{row_location} has duplicate artifact path: {receipt_path}")

    return valid_name


def check_contract(document: dict) -> list[str]:
    """Return blockers for a static contract; an empty list means statically ready."""

    blockers: list[str] = []
    if not isinstance(document, dict):
        return ["manifest must be a JSON object"]

    schema_version = document.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        blockers.append("schema_version must be integer 1")

    policy = document.get("policy")
    if not isinstance(policy, str) or policy != POLICY:
        blockers.append(f"policy must be exactly {POLICY}")

    _check_output_root(document.get("output_root"), blockers)

    datasets = document.get("datasets")
    if not isinstance(datasets, list):
        blockers.append("datasets must be a list containing exactly nine dataset objects")
        return blockers

    if len(datasets) != len(DATASET_NAMES):
        blockers.append("datasets must contain exactly nine rows")

    seen_names: set[str] = set()
    for index, row in enumerate(datasets):
        name = _check_row(row, index, blockers)
        if name is None:
            continue
        if name in seen_names:
            blockers.append(f"duplicate dataset name: {name}")
        seen_names.add(name)

    for missing_name in sorted(DATASET_NAMES - seen_names):
        blockers.append(f"missing dataset: {missing_name}")

    return blockers


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_manifest(path: Path) -> tuple[object | None, list[str]]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [f"manifest could not be read: {path}: {exc}"]

    try:
        document = json.loads(
            content,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return None, [f"manifest could not be parsed: {exc}"]
    return document, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)

    document, blockers = _read_manifest(args.manifest)
    if not blockers:
        blockers = check_contract(document)

    ready = not blockers
    print(json.dumps({"ready": ready, "blockers": blockers}))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
