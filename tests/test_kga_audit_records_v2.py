"""Behavioral tests for local byte identity; these records provide no custody."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

PAYLOAD = {"z": "café", "a": [True, None, 1.5]}
PAYLOAD_BYTES = b'{"a":[true,null,1.5],"z":"caf\xc3\xa9"}'
PAYLOAD_HASH = "98030d52386ee4c7486dfc04c79d8471607ffdca23e2e9863ffcc79c3035c7b8"
ENVELOPE_BYTES = (
    b'{"payload":{"a":[true,null,1.5],"z":"caf\xc3\xa9"},'
    b'"payload_sha256":"98030d52386ee4c7486dfc04c79d8471607ffdca23e2e9863ffcc79c3035c7b8",'
    b'"record_type":"REVIEW","schema_version":2}'
)
ENVELOPE_HASH = "e7192598da89a260bd02fb46e3ece5bb31807a21a7644828f401650c7d0d4f70"


@pytest.fixture
def records() -> ModuleType:
    return importlib.import_module("kga.audit_records")


def test_canonical_bytes_are_literal_utf8_sorted_and_compact(records: ModuleType) -> None:
    assert records.canonical_json_bytes(PAYLOAD) == PAYLOAD_BYTES


def test_writer_binds_exact_envelope_bytes_and_reader_returns_payload(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    identity = records.write_record(path, PAYLOAD, record_type="REVIEW")
    assert path.read_bytes() == ENVELOPE_BYTES
    assert identity == ENVELOPE_HASH
    assert identity != PAYLOAD_HASH
    assert records.read_record(path, expected_sha256=identity, record_type="REVIEW") == PAYLOAD


@pytest.mark.parametrize("record_type", ["PROTOCOL", "ACTION", "OUTCOME", "REVIEW"])
def test_recognized_record_types_round_trip(records: ModuleType, tmp_path: Path, record_type: str) -> None:
    path = tmp_path / "record.json"
    identity = records.write_record(path, {}, record_type=record_type)
    assert records.read_record(path, expected_sha256=identity, record_type=record_type) == {}


@pytest.mark.parametrize(
    "value",
    [
        (1,),
        {"nested": {1: "bad key"}},
        {"x": [float("nan")]},
        [float("inf")],
        [float("-inf")],
        {"x": object()},
        {"x": b"bytes"},
        {"x": {1, 2}},
        {"x": "\ud800"},
    ],
)
def test_canonical_json_rejects_nested_non_json_values(records: ModuleType, value: object) -> None:
    with pytest.raises(ValueError):
        records.canonical_json_bytes(value)


def test_canonical_json_rejects_subclasses_cycles_and_excessive_depth(records: ModuleType) -> None:
    class IntSubclass(int):
        pass

    cycle: list[object] = []
    cycle.append(cycle)
    deep: object = None
    for _ in range(2000):
        deep = [deep]
    for value in [IntSubclass(1), cycle, deep]:
        with pytest.raises(ValueError):
            records.canonical_json_bytes(value)


@pytest.mark.parametrize("record_type", ["", "review", "LOCKED", None, [], 1])
def test_invalid_record_type_cannot_create_files(records: ModuleType, tmp_path: Path, record_type: object) -> None:
    path = tmp_path / "record.json"
    with pytest.raises(ValueError):
        records.write_record(path, {}, record_type=record_type)
    assert not path.exists()


@pytest.mark.parametrize("payload", [[], None, 1, {"x": float("nan")}, {"x": "\ud800"}])
def test_invalid_payload_is_rejected_before_creation(records: ModuleType, tmp_path: Path, payload: object) -> None:
    path = tmp_path / "record.json"
    with pytest.raises(ValueError):
        records.write_record(path, payload, record_type="REVIEW")
    assert not path.exists()


def test_two_writers_cannot_overwrite_same_destination(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"

    def write_once(n: int) -> str | None:
        try:
            return cast(str, records.write_record(path, {"n": n}, record_type="ACTION"))
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write_once, [1, 2]))
    assert sum(result is not None for result in results) == 1
    identity = next(result for result in results if result is not None)
    assert records.read_record(path, expected_sha256=identity, record_type="ACTION") in [{"n": 1}, {"n": 2}]


def test_existing_arbitrary_bytes_survive_all_write_errors(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    original = b"existing bytes, not even valid JSON\x00"
    path.write_bytes(original)
    for payload in [{}, {"invalid": float("nan")}]:
        with pytest.raises((ValueError, FileExistsError)):
            records.write_record(path, payload, record_type="REVIEW")
        assert path.read_bytes() == original


@pytest.mark.parametrize("dangling", [False, True])
def test_output_symlink_never_modifies_target(records: ModuleType, tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "target.json"
    if not dangling:
        target.write_bytes(b"preserve")
    link = tmp_path / "record.json"
    link.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        records.write_record(link, {}, record_type="REVIEW")
    assert link.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert target.read_bytes() == b"preserve"


def test_input_symlink_is_rejected_even_with_matching_identity(records: ModuleType, tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(ENVELOPE_BYTES)
    link = tmp_path / "record.json"
    link.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        records.read_record(link, expected_sha256=ENVELOPE_HASH, record_type="REVIEW")
    assert target.read_bytes() == ENVELOPE_BYTES


def test_parent_symlink_is_rejected_without_touching_target(records: ModuleType, tmp_path: Path) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    (directory / "record.json").write_bytes(ENVELOPE_BYTES)
    link = tmp_path / "alias"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        records.write_record(link / "new.json", {}, record_type="REVIEW")
    with pytest.raises((ValueError, OSError)):
        records.read_record(link / "record.json", expected_sha256=ENVELOPE_HASH, record_type="REVIEW")
    assert not (directory / "new.json").exists()
    assert (directory / "record.json").read_bytes() == ENVELOPE_BYTES


def test_directory_cannot_be_a_record(records: ModuleType, tmp_path: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        records.write_record(tmp_path, {}, record_type="REVIEW")
    with pytest.raises((ValueError, OSError)):
        records.read_record(tmp_path, expected_sha256=ENVELOPE_HASH, record_type="REVIEW")


@pytest.mark.parametrize("expected_hash", ["", "0" * 63, "g" * 64, "0" * 65, ENVELOPE_HASH.upper(), None, []])
def test_external_identity_must_be_a_valid_lowercase_sha256(
    records: ModuleType, tmp_path: Path, expected_hash: object
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(ENVELOPE_BYTES)
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=expected_hash, record_type="REVIEW")


def test_external_identity_is_mandatory(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(ENVELOPE_BYTES)
    with pytest.raises(TypeError):
        records.read_record(path, record_type="REVIEW")


def test_wrong_identity_and_tampering_are_rejected(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(ENVELOPE_BYTES)
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256="0" * 64, record_type="REVIEW")
    path.write_bytes(ENVELOPE_BYTES.replace(b"true", b"false"))
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=ENVELOPE_HASH, record_type="REVIEW")


def test_wrong_requested_type_is_rejected(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(ENVELOPE_BYTES)
    requested_types: list[object] = ["ACTION", "unknown", [], None]
    for requested in requested_types:
        with pytest.raises(ValueError):
            records.read_record(path, expected_sha256=ENVELOPE_HASH, record_type=requested)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b"{",
        b"[]",
        b"null",
        b"{}",
        ENVELOPE_BYTES.replace(b'"schema_version":2', b'"schema_version":true'),
        ENVELOPE_BYTES.replace(b'"schema_version":2', b'"schema_version":2.0'),
        ENVELOPE_BYTES.replace(b'"schema_version":2', b'"schema_version":1'),
        ENVELOPE_BYTES.replace(b'"schema_version":2', b'"schema_version":2,"extra":0'),
        ENVELOPE_BYTES.replace(b',"schema_version":2', b""),
        ENVELOPE_BYTES.replace(b'"record_type":"REVIEW"', b'"record_type":"LOCKED"'),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b"[]"),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b'{"x":NaN}'),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b'{"x":Infinity}'),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b'{"x":1e999}'),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b'{"x":"\\ud800"}'),
        ENVELOPE_BYTES.replace(PAYLOAD_HASH.encode(), b"0" * 64),
        ENVELOPE_BYTES.replace(b'"schema_version":2', b'"schema_version":1,"schema_version":2'),
        ENVELOPE_BYTES.replace(PAYLOAD_BYTES, b'{"a":1,"a":2}'),
        b" " + ENVELOPE_BYTES,
        b"[" * 2000 + b"0" + b"]" * 2000,
    ],
)
def test_malformed_envelope_rejected_even_when_external_hash_matches(
    records: ModuleType, tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=hashlib.sha256(raw).hexdigest(), record_type="REVIEW")
    assert path.read_bytes() == raw


def test_nested_duplicate_keys_with_otherwise_valid_envelope_are_rejected(records: ModuleType, tmp_path: Path) -> None:
    # The payload digest matches the last-key-wins interpretation, so only
    # duplicate rejection prevents acceptance.
    payload_hash = hashlib.sha256(b'{"x":{"a":2}}').hexdigest()
    raw = (
        b'{"payload":{"x":{"a":1,"a":2}},"payload_sha256":"'
        + payload_hash.encode()
        + b'","record_type":"REVIEW","schema_version":2}'
    )
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=hashlib.sha256(raw).hexdigest(), record_type="REVIEW")


def test_payload_hash_cannot_substitute_for_external_envelope_identity(records: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_bytes(ENVELOPE_BYTES)
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=json.loads(ENVELOPE_BYTES)["payload_sha256"], record_type="REVIEW")


def test_interrupted_write_preserves_partial_file_and_cannot_be_retried_over_it(
    records: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.json"
    real_write = os.write

    def interrupt_write(descriptor: int, data: bytes) -> int:
        real_write(descriptor, data[:10])
        raise OSError("simulated interrupted storage write")

    monkeypatch.setattr(os, "write", interrupt_write)
    with pytest.raises(OSError):
        records.write_record(path, PAYLOAD, record_type="REVIEW")
    assert path.read_bytes() == ENVELOPE_BYTES[:10]
    with pytest.raises(FileExistsError):
        records.write_record(path, PAYLOAD, record_type="REVIEW")
    with pytest.raises(ValueError):
        records.read_record(path, expected_sha256=ENVELOPE_HASH, record_type="REVIEW")


def test_short_writes_still_produce_complete_bytes(
    records: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.json"
    real_write = os.write

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[:7])

    monkeypatch.setattr(os, "write", short_write)
    assert records.write_record(path, PAYLOAD, record_type="REVIEW") == ENVELOPE_HASH
    assert path.read_bytes() == ENVELOPE_BYTES


def test_zero_length_write_fails_and_leaves_empty_file(
    records: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.json"
    monkeypatch.setattr(os, "write", lambda descriptor, data: 0)
    with pytest.raises(OSError):
        records.write_record(path, PAYLOAD, record_type="REVIEW")
    assert path.exists()
    assert path.read_bytes() == b""


def test_fsync_failure_returns_no_identity_and_keeps_written_bytes(
    records: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.json"

    def failed_fsync(descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", failed_fsync)
    with pytest.raises(OSError):
        records.write_record(path, PAYLOAD, record_type="REVIEW")
    assert path.read_bytes() == ENVELOPE_BYTES
