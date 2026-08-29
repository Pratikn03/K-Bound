#!/usr/bin/env python3
"""Build a label-free, full-population manifest for CCT-20 ``trans_test``.

This module deliberately does *not* call ``json.load`` on the held-out target
annotation file.  Its restricted reader syntactically skips only the official
leading ``info``/``categories`` envelope values without deserializing them,
parses ``images`` to its closing bracket, and stops.  It never deserializes,
iterates over, counts, or hashes the target ``annotations`` field.

Every image declared in ``images`` must have a unique id and path, resolve
inside the supplied image root, decode completely, and contribute its bytes to
the content manifest.  There is no retry, replacement, glob fallback, or sample
substitution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from PIL import Image, UnidentifiedImageError

try:
    from .integrity import IntegrityError, atomic_json_dump, stable_sha256
except ImportError:  # pragma: no cover - direct script execution
    from integrity import IntegrityError, atomic_json_dump, stable_sha256


SCHEMA = "kbound_cct20_label_free_target_manifest_v1"
TARGET_ANNOTATIONS_BASENAME = "trans_test_annotations.json"
LOCKED_TARGET_IMAGE_COUNT = 23_275
LOCKED_TARGET_LOCATIONS = frozenset({0, 7, 28, 40, 46, 78, 100, 105, 130})
LOCKED_TARGET_ID_SET_SHA256 = (
    "39c97c54586c0a9aa96b2c23d08e6db442a2274c6f274514d58035245f6cea3b"
)
REQUIRED_STREAM_FIELDS = frozenset(
    {"id", "file_name", "location", "date_captured", "seq_id", "frame_num"}
)
SAFE_IMAGE_FIELDS = frozenset(
    {
        "id",
        "file_name",
        "location",
        "datetime",
        "date_captured",
        "seq_id",
        "frame_num",
        "seq_num_frames",
        "width",
        "height",
        "rights_holder",
        "corrupt",
    }
)
FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "annotation",
        "annotations",
        "category",
        "categories",
        "category_id",
        "class",
        "class_id",
        "label",
        "labels",
        "target",
        "y",
    }
)
SAFE_SKIPPED_TOP_LEVEL_FIELDS = frozenset({"info", "categories", "licenses"})


class _CharStream:
    """A small incremental character stream used by the restricted reader."""

    def __init__(self, handle: TextIO, chunk_size: int = 64 * 1024) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        if self.eof:
            return False
        self.buffer = self.handle.read(self.chunk_size)
        self.position = 0
        if not self.buffer:
            self.eof = True
            return False
        return True

    def peek(self) -> str:
        return self.buffer[self.position] if self._fill() else ""

    def read(self) -> str:
        if not self._fill():
            return ""
        character = self.buffer[self.position]
        self.position += 1
        return character

    def skip_whitespace(self) -> None:
        while self.peek() and self.peek().isspace():
            self.read()


def _expect(stream: _CharStream, expected: str, *, context: str) -> None:
    actual = stream.read()
    if actual != expected:
        rendered = "end of file" if not actual else repr(actual)
        raise IntegrityError(f"expected {expected!r} {context}, found {rendered}")


def _read_json_string(stream: _CharStream, *, context: str) -> str:
    _expect(stream, '"', context=context)
    token = ['"']
    escaped = False
    while True:
        character = stream.read()
        if not character:
            raise IntegrityError(f"unterminated JSON string {context}")
        token.append(character)
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            break
    try:
        value = json.loads("".join(token))
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"invalid JSON string {context}: {exc}") from exc
    if not isinstance(value, str):  # defensive; json string always returns str
        raise IntegrityError(f"expected a string {context}")
    return value


def _read_compound_value(stream: _CharStream, *, context: str) -> str:
    """Read one JSON object/array without consuming the following delimiter."""

    opening = stream.peek()
    if opening not in "[{":
        raise IntegrityError(f"{context} must be a JSON object")
    closers = {"[": "]", "{": "}"}
    stack: list[str] = []
    token: list[str] = []
    in_string = False
    escaped = False
    while True:
        character = stream.read()
        if not character:
            raise IntegrityError(f"unterminated JSON value {context}")
        token.append(character)
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in closers:
            stack.append(closers[character])
        elif character in "]}":
            if not stack or character != stack.pop():
                raise IntegrityError(f"unbalanced JSON value {context}")
            if not stack:
                return "".join(token)


def iter_target_image_metadata(
    annotation_path: str | os.PathLike[str],
    *,
    _skipped_fields_audit: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield only target ``images`` metadata and stop at its array boundary.

    The official COCO envelope places ``info`` and ``categories`` before
    ``images``.  Those allowlisted values are balanced syntactically but never
    passed to ``json.loads`` or retained.  Encountering ``annotations`` before
    ``images`` fails immediately without consuming its value.
    """

    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"target annotation envelope is missing: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        stream = _CharStream(handle)
        stream.skip_whitespace()
        _expect(stream, "{", context="at the start of the target envelope")
        stream.skip_whitespace()
        while True:
            key = _read_json_string(stream, context="for a top-level key")
            stream.skip_whitespace()
            _expect(stream, ":", context=f"after top-level key {key!r}")
            stream.skip_whitespace()
            if key == "images":
                break
            if key == "annotations":
                raise IntegrityError(
                    "target envelope places 'annotations' before 'images'; refusing "
                    "to consume any target annotation value"
                )
            if key not in SAFE_SKIPPED_TOP_LEVEL_FIELDS:
                raise IntegrityError(
                    f"unapproved top-level target field before images: {key!r}"
                )
            _read_compound_value(stream, context=f"while skipping top-level {key!r}")
            if _skipped_fields_audit is not None:
                _skipped_fields_audit.append(key)
            stream.skip_whitespace()
            _expect(stream, ",", context=f"after skipped top-level field {key!r}")
            stream.skip_whitespace()
        _expect(stream, "[", context="at the start of the target images array")
        stream.skip_whitespace()
        if stream.peek() == "]":
            stream.read()
            return
        index = 0
        while True:
            stream.skip_whitespace()
            text = _read_compound_value(stream, context=f"for images[{index}]")
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid target images[{index}]: {exc}") from exc
            if not isinstance(row, dict):
                raise IntegrityError(f"target images[{index}] must be an object")
            yield row
            index += 1
            stream.skip_whitespace()
            delimiter = stream.read()
            if delimiter == "]":
                return
            if delimiter != ",":
                rendered = "end of file" if not delimiter else repr(delimiter)
                raise IntegrityError(
                    f"expected ',' or ']' after target images[{index - 1}], found {rendered}"
                )


def _normalize_image_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    unknown = sorted(set(row) - SAFE_IMAGE_FIELDS)
    forbidden = sorted(set(row) & FORBIDDEN_LABEL_FIELDS)
    if forbidden:
        raise IntegrityError(
            f"target images[{index}] contains label-bearing fields: {forbidden}"
        )
    if unknown:
        raise IntegrityError(
            f"target images[{index}] contains unapproved metadata fields: {unknown}; "
            "review and explicitly add only demonstrably label-free fields"
        )
    missing = sorted(REQUIRED_STREAM_FIELDS - set(row))
    if missing:
        raise IntegrityError(
            f"target images[{index}] is missing required label-free stream fields: {missing}"
        )
    image_id = row["id"]
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise IntegrityError(f"target images[{index}].id must be an integer or string")
    if isinstance(image_id, str) and not image_id.strip():
        raise IntegrityError(f"target images[{index}].id must not be an empty string")
    file_name = row["file_name"]
    if not isinstance(file_name, str) or not file_name.strip():
        raise IntegrityError(f"target images[{index}].file_name must be a non-empty string")
    location = row["location"]
    if isinstance(location, bool) or not isinstance(location, (int, str)):
        raise IntegrityError(
            f"target images[{index}].location must be an integer or string"
        )
    seq_id = row["seq_id"]
    if not isinstance(seq_id, str) or not seq_id.strip():
        raise IntegrityError(f"target images[{index}].seq_id must be a non-empty string")
    date_captured = row["date_captured"]
    if not isinstance(date_captured, str) or not date_captured.strip():
        raise IntegrityError(
            f"target images[{index}].date_captured must be a non-empty string"
        )
    frame_num = row["frame_num"]
    if isinstance(frame_num, bool) or not isinstance(frame_num, int) or frame_num <= 0:
        raise IntegrityError(
            f"target images[{index}].frame_num must be a positive integer"
        )
    if "seq_num_frames" in row:
        sequence_length = row["seq_num_frames"]
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length <= 0
            or frame_num > sequence_length
        ):
            raise IntegrityError(
                f"target images[{index}] has an invalid frame_num/seq_num_frames pair"
            )
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            raise IntegrityError(
                f"target images[{index}].{key} must be scalar label-free metadata"
            )
    # Normalize the declared relative path here as well as at initial build so
    # validation rejects a recomputed manifest containing path traversal.
    normalized = dict(row)
    normalized["file_name"] = _relative_image_path(file_name).as_posix()
    return normalized


def _relative_image_path(file_name: str) -> Path:
    posix = PurePosixPath(file_name.replace("\\", "/"))
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise IntegrityError(f"unsafe target image path: {file_name!r}")
    return Path(*posix.parts)


def _read_and_validate_image(path: Path) -> tuple[int, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot read declared target image {path}: {exc}") from exc
    if not payload:
        raise IntegrityError(f"declared target image is empty: {path}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise IntegrityError(f"declared target image does not decode completely: {path}: {exc}") from exc
    return len(payload), hashlib.sha256(payload).hexdigest()


def _resolve_declared_image(image_root: Path, file_name: str) -> Path:
    root = image_root.expanduser().resolve(strict=True)
    relative = _relative_image_path(file_name)
    try:
        candidate = (root / relative).resolve(strict=True)
    except FileNotFoundError as exc:
        raise IntegrityError(f"declared target image is missing: {relative.as_posix()}") from exc
    if root != candidate and root not in candidate.parents:
        raise IntegrityError(f"declared target image escapes image root: {file_name!r}")
    if not candidate.is_file():
        raise IntegrityError(f"declared target image is not a regular file: {candidate}")
    return candidate


def _id_key(value: int | str) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_label_free_target_manifest(
    annotation_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
    *,
    expected_count: int,
    expected_id_set_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and bind every declared target image without opening labels."""

    annotation = Path(annotation_path)
    if annotation.name != TARGET_ANNOTATIONS_BASENAME:
        raise IntegrityError(
            "prospective target preparation requires an official envelope named "
            f"{TARGET_ANNOTATIONS_BASENAME!r}; found {annotation.name!r}"
        )
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise IntegrityError("expected_count must be a positive integer fixed before preparation")
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"target image root is missing: {root}")

    metadata_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    ids: set[str] = set()
    file_names: set[str] = set()
    skipped_fields: list[str] = []
    for index, raw in enumerate(
        iter_target_image_metadata(annotation, _skipped_fields_audit=skipped_fields)
    ):
        row = _normalize_image_row(raw, index=index)
        key = _id_key(row["id"])
        if key in ids:
            raise IntegrityError(f"duplicate target image id at images[{index}]: {row['id']!r}")
        ids.add(key)
        normalized_name = _relative_image_path(row["file_name"]).as_posix()
        if normalized_name in file_names:
            raise IntegrityError(
                f"duplicate target image path at images[{index}]: {normalized_name!r}"
            )
        file_names.add(normalized_name)
        row["file_name"] = normalized_name
        resolved = _resolve_declared_image(root, normalized_name)
        byte_count, content_sha256 = _read_and_validate_image(resolved)
        metadata_rows.append(row)
        samples.append(
            {
                "id": row["id"],
                "file_name": normalized_name,
                "image_bytes": byte_count,
                "image_sha256": content_sha256,
            }
        )

    actual_count = len(metadata_rows)
    if actual_count != expected_count:
        raise IntegrityError(
            f"target population count mismatch: expected exactly {expected_count}, found {actual_count}"
        )
    id_set_sha256 = stable_sha256(sorted(ids))
    if expected_id_set_sha256 is not None and id_set_sha256 != expected_id_set_sha256:
        raise IntegrityError(
            "target id-set hash mismatch: "
            f"expected {expected_id_set_sha256}, found {id_set_sha256}"
        )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "LABEL_FREE_POPULATION_VERIFIED",
        "target_role": "trans_test",
        "target_annotation_envelope_basename": annotation.name,
        "target_annotations_access_contract": {
            "parsed_top_level_fields": ["images"],
            "skipped_without_deserialization": skipped_fields,
            "annotations_field_parsed": False,
            "annotations_field_counted": False,
            "annotations_field_hashed": False,
            "reader_stops_at_images_array_end": True,
            "annotations_before_images_allowed": False,
        },
        "coverage": {
            "expected_images": expected_count,
            "declared_images": actual_count,
            "unique_ids": len(ids),
            "unique_file_names": len(file_names),
            "resolved_regular_files": len(samples),
            "fully_decoded_images": len(samples),
            "coverage_fraction": 1.0,
            "sample_substitution": False,
        },
        "identity": {
            "id_set_sha256": id_set_sha256,
            "ordered_images_metadata_sha256": stable_sha256(metadata_rows),
            "ordered_image_content_manifest_sha256": stable_sha256(samples),
            "population_sha256": stable_sha256(
                {
                    "metadata": metadata_rows,
                    "content": samples,
                }
            ),
        },
        "images": metadata_rows,
        "samples": samples,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    return manifest


def validate_label_free_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise IntegrityError(f"unexpected target-manifest schema: {manifest.get('schema')!r}")
    claimed_hash = manifest.get("manifest_sha256")
    if not isinstance(claimed_hash, str):
        raise IntegrityError("target manifest is missing manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if stable_sha256(body) != claimed_hash:
        raise IntegrityError("target manifest hash does not match its contents")
    contract = manifest.get("target_annotations_access_contract", {})
    if contract.get("parsed_top_level_fields") != ["images"]:
        raise IntegrityError("target manifest does not prove images-only parsing")
    if any(contract.get(field) is not False for field in (
        "annotations_field_parsed",
        "annotations_field_counted",
        "annotations_field_hashed",
    )):
        raise IntegrityError("target manifest claims access to target annotations")
    skipped = contract.get("skipped_without_deserialization")
    if (
        not isinstance(skipped, list)
        or len(skipped) != len(set(skipped))
        or any(field not in SAFE_SKIPPED_TOP_LEVEL_FIELDS for field in skipped)
    ):
        raise IntegrityError("target manifest has an invalid skipped-field audit")
    if contract.get("reader_stops_at_images_array_end") is not True:
        raise IntegrityError("target manifest does not require stopping after images")
    if contract.get("annotations_before_images_allowed") is not False:
        raise IntegrityError("target manifest permits annotations before images")
    coverage = manifest.get("coverage", {})
    counts = [
        coverage.get("expected_images"),
        coverage.get("declared_images"),
        coverage.get("unique_ids"),
        coverage.get("unique_file_names"),
        coverage.get("resolved_regular_files"),
        coverage.get("fully_decoded_images"),
    ]
    if not counts or any(value != counts[0] for value in counts):
        raise IntegrityError(f"target manifest does not have exact full coverage: {counts}")
    if coverage.get("coverage_fraction") != 1.0 or coverage.get("sample_substitution") is not False:
        raise IntegrityError("target manifest permits incomplete coverage or sample substitution")
    images = manifest.get("images")
    samples = manifest.get("samples")
    if not isinstance(images, list) or not isinstance(samples, list):
        raise IntegrityError("target manifest images and samples must be arrays")
    if len(images) != counts[0] or len(samples) != counts[0]:
        raise IntegrityError("target manifest arrays do not match the declared population count")
    normalized_images = [
        _normalize_image_row(row, index=index) if isinstance(row, dict) else None
        for index, row in enumerate(images)
    ]
    if any(row is None for row in normalized_images):
        raise IntegrityError("target manifest images must contain only objects")
    id_keys = [_id_key(row["id"]) for row in normalized_images if row is not None]
    file_names = [row["file_name"] for row in normalized_images if row is not None]
    if len(set(id_keys)) != len(id_keys) or len(set(file_names)) != len(file_names):
        raise IntegrityError("target manifest contains duplicate ids or file names")
    normalized_samples: list[dict[str, Any]] = []
    for index, row in enumerate(samples):
        if not isinstance(row, dict) or set(row) != {
            "id",
            "file_name",
            "image_bytes",
            "image_sha256",
        }:
            raise IntegrityError(f"target samples[{index}] has an invalid schema")
        if row["id"] != normalized_images[index]["id"] or row["file_name"] != file_names[index]:
            raise IntegrityError(f"target samples[{index}] does not align with images[{index}]")
        if (
            isinstance(row["image_bytes"], bool)
            or not isinstance(row["image_bytes"], int)
            or row["image_bytes"] <= 0
        ):
            raise IntegrityError(f"target samples[{index}].image_bytes must be positive")
        digest = row["image_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise IntegrityError(f"target samples[{index}].image_sha256 is invalid")
        normalized_samples.append(dict(row))
    expected_identity = {
        "id_set_sha256": stable_sha256(sorted(id_keys)),
        "ordered_images_metadata_sha256": stable_sha256(normalized_images),
        "ordered_image_content_manifest_sha256": stable_sha256(normalized_samples),
        "population_sha256": stable_sha256(
            {"metadata": normalized_images, "content": normalized_samples}
        ),
    }
    if manifest.get("identity") != expected_identity:
        raise IntegrityError("target manifest internal population hashes do not match its records")


def validate_locked_target_population(manifest: dict[str, Any]) -> None:
    """Enforce the immutable CCT-20 population contract used by production."""

    validate_label_free_manifest(manifest)
    coverage = manifest["coverage"]
    if coverage["expected_images"] != LOCKED_TARGET_IMAGE_COUNT:
        raise IntegrityError(
            "prospective target count differs from the locked 23,275-image population"
        )
    identity = manifest["identity"]
    if identity["id_set_sha256"] != LOCKED_TARGET_ID_SET_SHA256:
        raise IntegrityError("prospective target id set differs from the locked population")
    locations = {row["location"] for row in manifest["images"]}
    if locations != LOCKED_TARGET_LOCATIONS:
        raise IntegrityError(
            "prospective target locations differ from the locked nine-location population: "
            f"{sorted(locations, key=str)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-annotation-envelope", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=LOCKED_TARGET_IMAGE_COUNT,
        help="must equal the sealed CCT-20 target population (23,275)",
    )
    parser.add_argument(
        "--expected-id-set-sha256",
        default=LOCKED_TARGET_ID_SET_SHA256,
        help="must equal the sealed CCT-20 target id-set digest",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_count != LOCKED_TARGET_IMAGE_COUNT:
        raise IntegrityError(
            f"--expected-count must equal locked value {LOCKED_TARGET_IMAGE_COUNT}"
        )
    if args.expected_id_set_sha256 != LOCKED_TARGET_ID_SET_SHA256:
        raise IntegrityError("--expected-id-set-sha256 differs from the locked target id set")
    manifest = build_label_free_target_manifest(
        args.target_annotation_envelope,
        args.image_root,
        expected_count=args.expected_count,
        expected_id_set_sha256=args.expected_id_set_sha256,
    )
    validate_locked_target_population(manifest)
    if args.output.exists():
        raise IntegrityError(f"refusing to overwrite existing prospective manifest: {args.output}")
    atomic_json_dump(args.output, manifest)
    print(
        f"target manifest: PASS n={manifest['coverage']['declared_images']} "
        f"sha256={manifest['manifest_sha256']} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
