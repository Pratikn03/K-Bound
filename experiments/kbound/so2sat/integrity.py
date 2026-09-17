"""Strict hashing and immutable-artifact helpers for the So2Sat campaign."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_RECEIPT_SCHEMA_V1 = "kbound_so2sat_artifact_receipt_v1"
ARTIFACT_RECEIPT_SCHEMA_V2 = "kbound_so2sat_artifact_receipt_v2"
DEFAULT_ARTIFACT_RECEIPT_SCHEMA = ARTIFACT_RECEIPT_SCHEMA_V2
_PRIVATE_QUARANTINE_PREFIX = ".kbound-quarantine-"
_PRIVATE_QUARANTINE_TOKEN_HEX_LENGTH = 32
_DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME = ".kbound-publication-incomplete.json"
_DIRECTORY_PUBLICATION_COMPLETE_BASENAME = ".kbound-publication-complete.json"
_DIRECTORY_PUBLICATION_RETAINED_INCOMPLETE_BASENAME = "publication-incomplete.json"
_DIRECTORY_PUBLICATION_FAILED_PAYLOAD_BASENAME = "failed-payload"
_DIRECTORY_PUBLICATION_FAILED_COMPLETE_BASENAME = "failed-complete.json"
_DIRECTORY_PUBLICATION_STATE_SCHEMA = "kbound_directory_publication_state_v1"

_RECEIPT_COMMON_FIELDS = frozenset(
    {
        "schema",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_document_sha256",
    }
)


class IntegrityError(ValueError):
    """Raised when prospective data or an artifact violates its contract."""


class LabelFirewallError(IntegrityError):
    """Raised when code attempts to cross the target-label firewall."""


class _UnownedPublicationError(IntegrityError):
    """Raised when a final name may contain an inode the writer does not own."""


class _AtomicRenameSourceMissing(IntegrityError):
    """Raised when an atomic capture source vanished before the syscall."""


@dataclass
class DirectoryPublicationReservation:
    """Pinned capability for one create-only directory publication."""

    publication_path: Path
    holder_path: Path
    staging_path: Path
    _parent_descriptor: int
    _holder_descriptor: int
    _staging_descriptor: int
    _parent_identity: tuple[int, int, int]
    _holder_identity: tuple[int, int, int]
    _staging_identity: tuple[int, int, int]
    _closed: bool = False

    def __truediv__(self, child: str | os.PathLike[str]) -> Path:
        return self.staging_path / child

    def exists(self) -> bool:
        return self.staging_path.exists()

    def _descriptors(self) -> tuple[int, int, int]:
        if self._closed:
            raise IntegrityError("directory publication reservation is already consumed")
        return self._parent_descriptor, self._holder_descriptor, self._staging_descriptor

    def close(self) -> None:
        if self._closed:
            return
        os.close(self._staging_descriptor)
        os.close(self._holder_descriptor)
        os.close(self._parent_descriptor)
        self._closed = True


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic ASCII JSON bytes, rejecting NaN and Infinity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ordered_records_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash a record stream without materializing the population in memory."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(dict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _decode_strict_json_bytes(payload: bytes, *, source: str | os.PathLike[str]) -> Any:
    def reject_constant(token: str) -> None:
        raise IntegrityError(f"non-standard JSON constant {token!r} in {source}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise IntegrityError(f"duplicate JSON key {key!r} in {source}")
            document[key] = value
        return document

    def reject_nonfinite_numbers(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise IntegrityError(f"non-finite JSON number in {source}")
        if isinstance(value, Mapping):
            for member in value.values():
                reject_nonfinite_numbers(member)
        elif isinstance(value, list):
            for member in value:
                reject_nonfinite_numbers(member)

    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"artifact is not strict UTF-8 JSON: {source}") from exc
    reject_nonfinite_numbers(document)
    return document


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    source = Path(path)
    return _decode_strict_json_bytes(source.read_bytes(), source=source)


def _require_no_follow_primitives() -> None:
    required_constants = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_constants):
        raise IntegrityError("required directory-FD no-follow primitives are unavailable")
    if (
        os.open not in os.supports_dir_fd
        or os.link not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.link not in os.supports_follow_symlinks
        or os.stat not in os.supports_follow_symlinks
    ):
        raise IntegrityError("required directory-FD no-follow primitives are unavailable")


def _snapshot_from_stat(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
    )


def _absolute_without_resolving(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def _open_directory_chain_no_follow(path: Path, *, create: bool) -> int:
    """Open an absolute directory through pinned, no-follow components."""

    if not path.is_absolute():  # pragma: no cover - internal contract
        raise IntegrityError("receipt output parent must be absolute")
    _require_no_follow_primitives()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    descriptor = os.open(path.anchor, directory_flags)
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise IntegrityError(f"cannot create receipt output parent component: {current}") from exc
            flags = directory_flags | os.O_NOFOLLOW
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise IntegrityError(f"receipt output parent contains a symlink or non-directory: {current}") from exc
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):  # pragma: no cover - O_DIRECTORY guard
                os.close(child)
                raise IntegrityError(f"receipt output parent is not a directory: {current}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_from_directory(
    parent_descriptor: int,
    basename: str,
    display_path: Path,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one regular leaf once through an already-pinned parent directory."""

    _require_portable_basename(basename, field="artifact basename")
    try:
        before_path = _snapshot_from_stat(os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False))
    except OSError as exc:
        raise IntegrityError(f"required artifact is missing or inaccessible: {display_path}") from exc
    if stat.S_ISLNK(before_path[2]):
        raise IntegrityError(f"artifact path is a symlink: {display_path}")
    if not stat.S_ISREG(before_path[2]):
        raise IntegrityError(f"artifact is not a regular file: {display_path}")
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise IntegrityError(f"cannot securely open artifact: {display_path}") from exc
    try:
        before_fd = _snapshot_from_stat(os.fstat(descriptor))
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after_fd = _snapshot_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    try:
        after_path = _snapshot_from_stat(os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False))
    except OSError as exc:
        raise IntegrityError(f"artifact changed while being read: {display_path}") from exc
    if not (before_path == before_fd == after_fd == after_path):
        raise IntegrityError(f"artifact changed while being read: {display_path}")
    payload = b"".join(chunks)
    if len(payload) != before_fd[3]:
        raise IntegrityError(f"artifact byte count changed while being read: {display_path}")
    return payload, before_fd


def _read_regular_file_once_no_follow(
    path: str | os.PathLike[str],
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    absolute = _absolute_without_resolving(path)
    parent_descriptor = _open_directory_chain_no_follow(absolute.parent, create=False)
    try:
        return _read_regular_file_from_directory(parent_descriptor, absolute.name, absolute)
    finally:
        os.close(parent_descriptor)


def read_secure_json_mapping_pair_once(
    first_path: str | os.PathLike[str],
    second_path: str | os.PathLike[str],
) -> tuple[tuple[dict[str, Any], bytes], tuple[dict[str, Any], bytes]]:
    """Read two mappings without reopening either path, pinning a shared parent."""

    first = _absolute_without_resolving(first_path)
    second = _absolute_without_resolving(second_path)
    if first.parent != second.parent:
        raise IntegrityError("artifact and receipt must have the exact same lexical parent")
    parent_descriptor = _open_directory_chain_no_follow(first.parent, create=False)
    try:
        _require_publication_eligible_private_transaction_residues_at(parent_descriptor)
        transaction_basename = _pair_transaction_basename(first.name)
        if _entry_exists_at(parent_descriptor, transaction_basename):
            raise IntegrityError("artifact/receipt publication is incomplete or still in progress")
        first_payload, _ = _read_regular_file_from_directory(
            parent_descriptor,
            first.name,
            first,
        )
        second_payload, _ = _read_regular_file_from_directory(
            parent_descriptor,
            second.name,
            second,
        )
        if _entry_exists_at(parent_descriptor, transaction_basename):
            raise IntegrityError("artifact/receipt publication changed during the secure read")
        _require_publication_eligible_private_transaction_residues_at(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    first_document = _decode_strict_json_bytes(first_payload, source=first)
    second_document = _decode_strict_json_bytes(second_payload, source=second)
    if not isinstance(first_document, Mapping) or not isinstance(second_document, Mapping):
        raise IntegrityError("secure JSON pair members must both be mappings")
    return (dict(first_document), first_payload), (dict(second_document), second_payload)


def _entry_exists_at(parent_descriptor: int, basename: str) -> bool:
    _require_portable_basename(basename, field="artifact basename")
    try:
        os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _pair_transaction_basename(artifact_basename: str) -> str:
    basename = _require_portable_basename(artifact_basename, field="artifact basename")
    identity = hashlib.sha256(basename.encode("utf-8")).hexdigest()[:32]
    return f".kbound-pair-{identity}.transaction"


def _entry_snapshot_at(
    parent_descriptor: int,
    basename: str,
) -> tuple[int, int, int, int, int]:
    return _snapshot_from_stat(os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False))


def is_retained_private_transaction_residue_name(value: str) -> bool:
    """Recognize only the random namespace emitted by secure transaction retirement."""

    if not value.startswith(_PRIVATE_QUARANTINE_PREFIX):
        return False
    token = value.removeprefix(_PRIVATE_QUARANTINE_PREFIX)
    return len(token) == _PRIVATE_QUARANTINE_TOKEN_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in token
    )


def _validate_retained_private_transaction_residue_at(
    parent_descriptor: int,
    basename: str,
) -> None:
    """Accept only an empty private workspace as publication-eligible residue.

    A nonempty rollback workspace records a failed transaction.  Its captured
    bytes and optional provenance remain available for private diagnostics, but
    same-UID metadata cannot authenticate those bytes for publication or later
    consumption.
    """

    if not is_retained_private_transaction_residue_name(basename):
        raise IntegrityError("invalid private transaction residue name")
    residue_descriptor: int | None = None
    try:
        try:
            residue_descriptor = os.open(
                basename,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            retained = _snapshot_from_stat(os.fstat(residue_descriptor))
            observed = _entry_snapshot_at(parent_descriptor, basename)
            if retained != observed or stat.S_IMODE(retained[2]) != 0o700:
                raise IntegrityError("invalid private transaction residue identity or mode")
            entries = set(os.listdir(residue_descriptor))
            if entries:
                raise IntegrityError("nonempty private transaction residue records a failed transaction")
            final_retained = _snapshot_from_stat(os.fstat(residue_descriptor))
            final_observed = _entry_snapshot_at(parent_descriptor, basename)
            if final_retained != retained or final_observed != retained:
                raise IntegrityError("private transaction residue changed during inspection")
        except IntegrityError:
            raise
        except OSError as exc:
            raise IntegrityError("invalid private transaction residue") from exc
    finally:
        if residue_descriptor is not None:
            os.close(residue_descriptor)


def validate_retained_private_transaction_residue(
    path: str | os.PathLike[str],
) -> None:
    """Fail closed unless ``path`` is one pinned, empty private workspace."""

    residue = _absolute_without_resolving(path)
    parent_descriptor = _open_directory_chain_no_follow(residue.parent, create=False)
    try:
        _validate_retained_private_transaction_residue_at(
            parent_descriptor,
            residue.name,
        )
    finally:
        os.close(parent_descriptor)


def _require_publication_eligible_private_transaction_residues_at(
    parent_descriptor: int,
) -> None:
    """Reject every nonempty or malformed rollback residue in a pinned directory."""

    try:
        names = os.listdir(parent_descriptor)
    except OSError as exc:
        raise IntegrityError("cannot inspect private transaction residue namespace") from exc
    for name in names:
        if is_retained_private_transaction_residue_name(name):
            _validate_retained_private_transaction_residue_at(parent_descriptor, name)


def _normalize_required_regular_file_inventory(
    values: Iterable[str],
    *,
    _allow_publication_state_names: bool = False,
) -> frozenset[str]:
    """Validate an exact portable relative-file inventory."""

    if isinstance(values, (str, bytes)):
        raise IntegrityError("publication inventory must be an iterable of relative files")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise IntegrityError("publication inventory must be iterable") from exc
    if not items:
        raise IntegrityError("publication requires a nonempty regular-file inventory")
    normalized: set[str] = set()
    for value in items:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise IntegrityError("publication inventory contains an invalid relative file")
        components = value.split("/")
        if any(component in {"", ".", ".."} for component in components):
            raise IntegrityError("publication inventory contains an invalid relative file")
        for component in components:
            _require_portable_basename(component, field="publication inventory component")
            if is_retained_private_transaction_residue_name(component):
                raise IntegrityError("publication inventory overlaps the private residue namespace")
            if not _allow_publication_state_names and component in {
                _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
                _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            }:
                raise IntegrityError("publication inventory overlaps the completion-state namespace")
        normalized.add("/".join(components))
    if len(normalized) != len(items):
        raise IntegrityError("publication inventory contains duplicate relative files")
    return frozenset(normalized)


def _directory_publication_state_payload(
    required: frozenset[str],
    *,
    state: str,
) -> bytes:
    """Bind a publication state marker to one exact caller-declared inventory."""

    document = {
        "schema": _DIRECTORY_PUBLICATION_STATE_SCHEMA,
        "state": state,
        "required_regular_file_count": len(required),
        "required_regular_files_sha256": hashlib.sha256(canonical_json_bytes(sorted(required))).hexdigest(),
    }
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _require_publication_state_payload_at(
    root_descriptor: int,
    basename: str,
    expected_payload: bytes,
) -> None:
    payload, _ = _read_regular_file_from_directory(
        root_descriptor,
        basename,
        Path(basename),
    )
    if payload != expected_payload:
        raise IntegrityError("directory publication completion state is not bound to its exact inventory")


def _verify_pinned_directory_regular_file_inventory(
    root_descriptor: int,
    required_relative_regular_files: Iterable[str],
    *,
    _allow_publication_state_names: bool = False,
) -> None:
    """Verify an exact tree through directory FDs without trusting a root path."""

    required = _normalize_required_regular_file_inventory(
        required_relative_regular_files,
        _allow_publication_state_names=_allow_publication_state_names,
    )
    expected_directories: set[str] = set()
    for relative_file in required:
        components = relative_file.split("/")
        expected_directories.update("/".join(components[:depth]) for depth in range(1, len(components)))
    observed_files: set[str] = set()

    def walk(directory_descriptor: int, prefix: str) -> None:
        directory_before = _snapshot_from_stat(os.fstat(directory_descriptor))
        if not stat.S_ISDIR(directory_before[2]):
            raise IntegrityError("publication inventory root is not a directory")
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            raise IntegrityError("cannot enumerate pinned publication inventory") from exc
        for name in names:
            _require_portable_basename(name, field="publication inventory entry")
            relative_name = f"{prefix}/{name}" if prefix else name
            try:
                entry_snapshot = _entry_snapshot_at(directory_descriptor, name)
            except OSError as exc:
                raise IntegrityError("publication inventory changed during inspection") from exc
            if is_retained_private_transaction_residue_name(name):
                _validate_retained_private_transaction_residue_at(
                    directory_descriptor,
                    name,
                )
                continue
            if stat.S_ISDIR(entry_snapshot[2]):
                if relative_name not in expected_directories:
                    raise IntegrityError(f"publication inventory has unexpected directory: {relative_name}")
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_descriptor,
                    )
                    child_retained = _snapshot_from_stat(os.fstat(child_descriptor))
                    if child_retained != entry_snapshot:
                        raise IntegrityError("publication inventory directory identity changed")
                    walk(child_descriptor, relative_name)
                    if (
                        _snapshot_from_stat(os.fstat(child_descriptor)) != child_retained
                        or _entry_snapshot_at(directory_descriptor, name) != child_retained
                    ):
                        raise IntegrityError("publication inventory directory changed during inspection")
                except IntegrityError:
                    raise
                except OSError as exc:
                    raise IntegrityError("publication inventory directory is not stable and no-follow") from exc
                finally:
                    if child_descriptor is not None:
                        os.close(child_descriptor)
                continue
            if not stat.S_ISREG(entry_snapshot[2]):
                raise IntegrityError(f"publication inventory has non-regular entry: {relative_name}")
            if relative_name not in required:
                raise IntegrityError(f"publication inventory has unexpected regular file: {relative_name}")
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                retained_file = _snapshot_from_stat(os.fstat(file_descriptor))
                if (
                    retained_file != entry_snapshot
                    or not stat.S_ISREG(retained_file[2])
                    or _entry_snapshot_at(directory_descriptor, name) != retained_file
                ):
                    raise IntegrityError("publication inventory regular-file identity changed")
            except IntegrityError:
                raise
            except OSError as exc:
                raise IntegrityError("publication inventory file is not stable and no-follow") from exc
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)
            observed_files.add(relative_name)
        if _snapshot_from_stat(os.fstat(directory_descriptor)) != directory_before:
            raise IntegrityError("publication inventory changed during inspection")

    walk(root_descriptor, "")
    missing = sorted(required - observed_files)
    if missing:
        raise IntegrityError("publication inventory is incomplete; missing: " + ", ".join(missing))


def _open_matching_regular_file_at(
    parent_descriptor: int,
    basename: str,
    expected_snapshot: tuple[int, int, int, int, int],
) -> int:
    """Retain an FD capability for the exact regular inode just created."""

    _require_portable_basename(basename, field="artifact basename")
    try:
        descriptor = os.open(
            basename,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise IntegrityError("created artifact vanished before its capability was retained") from exc
    observed_fd = _snapshot_from_stat(os.fstat(descriptor))
    try:
        observed_path = _entry_snapshot_at(parent_descriptor, basename)
    except FileNotFoundError:
        os.close(descriptor)
        raise IntegrityError("created artifact identity changed before capability retention") from None
    if observed_fd != expected_snapshot or observed_path != expected_snapshot or not stat.S_ISREG(observed_fd[2]):
        os.close(descriptor)
        raise IntegrityError("created artifact identity changed before capability retention")
    return descriptor


def _unlink_matching_entry_at(
    parent_descriptor: int,
    basename: str,
    expected_snapshot: tuple[int, int, int, int, int] | None,
    *,
    expected_descriptor: int | None = None,
) -> bool:
    """Remove only an owned inode after atomically capturing the public name.

    A pathname ``stat`` followed by ``unlink`` is not a conditional deletion:
    another process can substitute the directory entry between those calls.
    Move the entry into a fresh mode-0700 quarantine directory first and
    compare the captured inode with the retained capability.  Never issue a
    later pathname deletion for ``captured``: a same-UID process could replace
    that name after authentication.  The securely captured entry therefore
    remains in the private quarantine.  A substituted entry is restored and
    preserved.

    The return value reports only whether the requested entry was absent or
    securely removed from the requested public name.  The quarantine and its
    captured regular file are intentionally retained: portable POSIX has no
    inode-conditional ``unlink`` or ``rmdir`` for these mutable names, so later
    cleanup would reintroduce the substitution races this helper prevents.
    """

    if expected_snapshot is None:
        return True
    try:
        observed = _entry_snapshot_at(parent_descriptor, basename)
    except FileNotFoundError:
        return True
    if observed != expected_snapshot:
        return False
    if expected_descriptor is not None and _snapshot_from_stat(os.fstat(expected_descriptor)) != expected_snapshot:
        return False

    quarantine_basename: str | None = None
    quarantine_descriptor: int | None = None
    for _ in range(32):
        candidate = f"{_PRIVATE_QUARANTINE_PREFIX}{secrets.token_hex(16)}"
        try:
            os.mkdir(candidate, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        quarantine_basename = candidate
        quarantine_descriptor = os.open(
            candidate,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        break
    if quarantine_basename is None or quarantine_descriptor is None:
        raise IntegrityError("could not reserve a private rollback quarantine")

    captured_basename = "captured"
    try:
        try:
            _native_rename_entry_create_only_at(
                parent_descriptor,
                basename,
                quarantine_descriptor,
                captured_basename,
            )
        except _AtomicRenameSourceMissing:
            return False
        captured = _entry_snapshot_at(quarantine_descriptor, captured_basename)
        if captured != expected_snapshot or (
            expected_descriptor is not None and _snapshot_from_stat(os.fstat(expected_descriptor)) != expected_snapshot
        ):
            try:
                _native_rename_entry_create_only_at(
                    quarantine_descriptor,
                    captured_basename,
                    parent_descriptor,
                    basename,
                )
            except IntegrityError:
                # The unowned entry remains quarantined.  Never delete it merely
                # to make cleanup look successful.
                return False
            return False
        return True
    finally:
        os.close(quarantine_descriptor)
        # Deliberately retain the private workspace, even when it is empty.
        # There is no portable syscall that removes a shared directory entry
        # only if it still names the inode held by ``quarantine_descriptor``.


def _move_staged_entry_create_only_at(
    source_parent_descriptor: int,
    source_basename: str,
    destination_parent_descriptor: int,
    destination_basename: str,
    display_path: Path,
    *,
    source_descriptor: int,
    expected_source_snapshot: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    """Move one descriptor-retained stage into a create-only final name."""

    _require_portable_basename(source_basename, field="staged artifact basename")
    _require_portable_basename(destination_basename, field="artifact basename")
    retained = _snapshot_from_stat(os.fstat(source_descriptor))
    try:
        current_source = _entry_snapshot_at(source_parent_descriptor, source_basename)
    except FileNotFoundError as exc:
        raise IntegrityError(f"staged artifact identity changed before publication: {display_path}") from exc
    if retained != expected_source_snapshot or current_source != expected_source_snapshot:
        raise IntegrityError(f"staged artifact identity changed before publication: {display_path}")
    try:
        _native_rename_entry_create_only_at(
            source_parent_descriptor,
            source_basename,
            destination_parent_descriptor,
            destination_basename,
        )
    except _AtomicRenameSourceMissing as exc:
        raise IntegrityError(f"staged artifact identity changed before publication: {display_path}") from exc
    except IntegrityError as exc:
        raise IntegrityError(f"cannot publish immutable artifact without replacement: {display_path}") from exc
    try:
        published = _entry_snapshot_at(destination_parent_descriptor, destination_basename)
    except FileNotFoundError as exc:
        raise _UnownedPublicationError(
            f"published artifact identity changed at the syscall boundary: {display_path}"
        ) from exc
    retained_after = _snapshot_from_stat(os.fstat(source_descriptor))
    if published != expected_source_snapshot or retained_after != expected_source_snapshot:
        raise _UnownedPublicationError(f"published artifact identity changed at the syscall boundary: {display_path}")
    return published


def _quarantine_transaction_directory_at(
    parent_descriptor: int,
    transaction_basename: str,
    transaction_descriptor: int,
    *,
    require_empty: bool,
) -> str:
    """Move a pinned pair workspace to private residue without deleting a shared name."""

    retained = _snapshot_from_stat(os.fstat(transaction_descriptor))
    try:
        observed = _entry_snapshot_at(parent_descriptor, transaction_basename)
    except FileNotFoundError as exc:
        raise IntegrityError("artifact/receipt transaction workspace vanished") from exc
    if retained != observed or not stat.S_ISDIR(retained[2]):
        raise IntegrityError("artifact/receipt transaction workspace identity changed")
    if require_empty and os.listdir(transaction_descriptor):
        raise IntegrityError("completed artifact/receipt transaction workspace is not empty")
    for _ in range(32):
        quarantine_basename = f"{_PRIVATE_QUARANTINE_PREFIX}{secrets.token_hex(16)}"
        try:
            _native_rename_entry_create_only_at(
                parent_descriptor,
                transaction_basename,
                parent_descriptor,
                quarantine_basename,
            )
        except IntegrityError as exc:
            if "destination already exists" in str(exc):
                continue
            raise
        final_retained = _snapshot_from_stat(os.fstat(transaction_descriptor))
        final_observed = _entry_snapshot_at(parent_descriptor, quarantine_basename)
        if final_retained != retained or final_observed != retained:
            raise IntegrityError("artifact/receipt transaction workspace changed during quarantine")
        if require_empty and os.listdir(transaction_descriptor):
            raise IntegrityError("completed artifact/receipt transaction workspace gained residue")
        return quarantine_basename
    raise IntegrityError("could not reserve private artifact/receipt transaction residue")


def _native_rename_entry_create_only_at(
    source_parent_descriptor: int,
    source_basename: str,
    destination_parent_descriptor: int,
    destination_basename: str,
) -> None:
    """Atomically rename one entry between pinned parents without replacement."""

    source = os.fsencode(_require_portable_basename(source_basename, field="staging directory basename"))
    destination = os.fsencode(_require_portable_basename(destination_basename, field="publication directory basename"))
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            rename_no_replace = libc.renameatx_np
        except AttributeError as exc:  # pragma: no cover - platform contract
            raise IntegrityError("atomic create-only directory rename is unavailable") from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            source_parent_descriptor,
            source,
            destination_parent_descriptor,
            destination,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:  # pragma: no cover - platform contract
            raise IntegrityError("atomic create-only directory rename is unavailable") from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            source_parent_descriptor,
            source,
            destination_parent_descriptor,
            destination,
            0x00000001,  # RENAME_NOREPLACE
        )
    else:  # pragma: no cover - fail-closed platform guard
        raise IntegrityError("atomic create-only directory rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.ENOENT:
            raise _AtomicRenameSourceMissing("atomic create-only rename source is missing")
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise IntegrityError("create-only publication destination already exists")
        raise IntegrityError(f"atomic create-only publication failed: {os.strerror(error_number)}")


def _native_rename_directory_create_only_at(
    parent_descriptor: int,
    source_basename: str,
    destination_basename: str,
) -> None:
    """Atomically rename one sibling directory without replacement."""

    _native_rename_entry_create_only_at(
        parent_descriptor,
        source_basename,
        parent_descriptor,
        destination_basename,
    )


def reserve_create_only_directory_publication(
    path: str | os.PathLike[str],
) -> DirectoryPublicationReservation:
    """Reserve a pinned private holder and inner staging-directory capability.

    Once a holder name has been created in the shared parent, failure cleanup
    closes every live descriptor but retains that private workspace.  Removing
    the holder by pathname would be vulnerable to directory-entry substitution.
    """

    publication = _absolute_without_resolving(path)
    _require_portable_basename(publication.name, field="publication directory basename")
    parent_descriptor = _open_directory_chain_no_follow(publication.parent, create=True)
    holder_descriptor: int | None = None
    staging_descriptor: int | None = None
    holder_basename: str | None = None
    try:
        if _entry_exists_at(parent_descriptor, publication.name):
            raise IntegrityError("create-only publication destination already exists")
        identity = hashlib.sha256(publication.name.encode("utf-8")).hexdigest()[:16]
        for _ in range(32):
            candidate = f".kbound-{identity}-{secrets.token_hex(8)}.reservation"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            holder_basename = candidate
            break
        if holder_basename is None:
            raise IntegrityError("could not reserve a unique v2 staging holder")
        holder_descriptor = os.open(
            holder_basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        holder_snapshot = _snapshot_from_stat(os.fstat(holder_descriptor))
        if not stat.S_ISDIR(holder_snapshot[2]):  # pragma: no cover - O_DIRECTORY guard
            raise IntegrityError("reserved v2 holder is not a directory")
        staging_basename = "payload"
        os.mkdir(staging_basename, mode=0o700, dir_fd=holder_descriptor)
        staging_descriptor = os.open(
            staging_basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=holder_descriptor,
        )
        staging_snapshot = _snapshot_from_stat(os.fstat(staging_descriptor))
        if not stat.S_ISDIR(staging_snapshot[2]):  # pragma: no cover - O_DIRECTORY guard
            raise IntegrityError("reserved v2 staging entry is not a directory")
        os.fsync(holder_descriptor)
        os.fsync(parent_descriptor)
        holder_path = publication.with_name(holder_basename)
        return DirectoryPublicationReservation(
            publication_path=publication,
            holder_path=holder_path,
            staging_path=holder_path / staging_basename,
            _parent_descriptor=parent_descriptor,
            _holder_descriptor=holder_descriptor,
            _staging_descriptor=staging_descriptor,
            _parent_identity=_snapshot_from_stat(os.fstat(parent_descriptor))[:3],
            _holder_identity=holder_snapshot[:3],
            _staging_identity=staging_snapshot[:3],
        )
    except BaseException:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if holder_descriptor is not None:
            os.close(holder_descriptor)
        os.close(parent_descriptor)
        raise


def publish_directory_create_only(
    reservation: DirectoryPublicationReservation,
    publication_path: str | os.PathLike[str] | None = None,
    *,
    required_relative_regular_files: Iterable[str],
) -> Path:
    """Publish exactly the directory retained by an opaque live reservation.

    The pinned payload is checked before and after the create-only rename.  It
    remains explicitly INCOMPLETE until that post-rename walk succeeds; only
    then is a pre-created, descriptor-retained completion marker moved into the
    payload and the final exact inventory checked.  A consumer must call
    :func:`verify_complete_directory_publication` rather than treating an
    internal document's status field as publication completion.
    """

    if not isinstance(reservation, DirectoryPublicationReservation):
        raise IntegrityError("directory publication requires its live reservation capability")
    publication = (
        reservation.publication_path if publication_path is None else _absolute_without_resolving(publication_path)
    )
    if publication != reservation.publication_path:
        raise IntegrityError("publication path differs from its reservation capability")
    _require_portable_basename(publication.name, field="publication directory basename")
    required = _normalize_required_regular_file_inventory(required_relative_regular_files)
    incomplete_payload = _directory_publication_state_payload(required, state="INCOMPLETE")
    complete_payload = _directory_publication_state_payload(
        required,
        state="COMPLETE_AFTER_POST_RENAME_DESCRIPTOR_REVALIDATION",
    )
    parent_descriptor, holder_descriptor, staging_descriptor = reservation._descriptors()
    incomplete_descriptor: int | None = None
    complete_descriptor: int | None = None
    incomplete_snapshot: tuple[int, int, int, int, int] | None = None
    complete_snapshot: tuple[int, int, int, int, int] | None = None
    directory_published = False
    incomplete_retained = False
    complete_published = False
    try:
        if _snapshot_from_stat(os.fstat(parent_descriptor))[:3] != reservation._parent_identity:
            raise IntegrityError("publication parent capability identity changed")
        if _snapshot_from_stat(os.fstat(holder_descriptor))[:3] != reservation._holder_identity:
            raise IntegrityError("publication holder capability identity changed")
        if _snapshot_from_stat(os.fstat(staging_descriptor))[:3] != reservation._staging_identity:
            raise IntegrityError("reserved staging directory capability identity changed")
        try:
            current_staging = _entry_snapshot_at(holder_descriptor, "payload")
        except FileNotFoundError as exc:
            raise IntegrityError("reserved staging directory is missing or substituted") from exc
        if current_staging[:3] != reservation._staging_identity:
            raise IntegrityError("reserved staging directory identity was substituted")
        _exclusive_write_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            reservation.staging_path / _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            incomplete_payload,
        )
        incomplete_snapshot = _entry_snapshot_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
        )
        incomplete_descriptor = _open_matching_regular_file_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            incomplete_snapshot,
        )
        _exclusive_write_at(
            holder_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            reservation.holder_path / _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            complete_payload,
        )
        complete_snapshot = _entry_snapshot_at(
            holder_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
        )
        complete_descriptor = _open_matching_regular_file_at(
            holder_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            complete_snapshot,
        )
        _verify_pinned_directory_regular_file_inventory(
            staging_descriptor,
            required | {_DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME},
            _allow_publication_state_names=True,
        )
        _require_publication_state_payload_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            incomplete_payload,
        )
        try:
            current_staging = _entry_snapshot_at(holder_descriptor, "payload")
        except FileNotFoundError as exc:
            raise IntegrityError("reserved staging directory vanished after inventory verification") from exc
        if current_staging[:3] != reservation._staging_identity:
            raise IntegrityError("reserved staging directory changed after inventory verification")
        if _entry_exists_at(parent_descriptor, publication.name):
            raise IntegrityError("create-only publication destination already exists")
        _native_rename_entry_create_only_at(
            holder_descriptor,
            "payload",
            parent_descriptor,
            publication.name,
        )
        directory_published = True
        published_snapshot = _entry_snapshot_at(parent_descriptor, publication.name)
        if published_snapshot[:3] != reservation._staging_identity:
            raise IntegrityError("published directory identity differs from its reservation capability")
        _verify_pinned_directory_regular_file_inventory(
            staging_descriptor,
            required | {_DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME},
            _allow_publication_state_names=True,
        )
        _require_publication_state_payload_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            incomplete_payload,
        )
        if (
            incomplete_snapshot is None
            or complete_snapshot is None
            or incomplete_descriptor is None
            or complete_descriptor is None
            or _snapshot_from_stat(os.fstat(incomplete_descriptor)) != incomplete_snapshot
            or _snapshot_from_stat(os.fstat(complete_descriptor)) != complete_snapshot
        ):  # pragma: no cover - defensive descriptor narrowing
            raise IntegrityError("directory publication state capability changed")
        _native_rename_entry_create_only_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
            holder_descriptor,
            _DIRECTORY_PUBLICATION_RETAINED_INCOMPLETE_BASENAME,
        )
        incomplete_retained = True
        _native_rename_entry_create_only_at(
            holder_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            staging_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
        )
        complete_published = True
        if (
            _entry_snapshot_at(
                holder_descriptor,
                _DIRECTORY_PUBLICATION_RETAINED_INCOMPLETE_BASENAME,
            )
            != incomplete_snapshot
            or _entry_snapshot_at(
                staging_descriptor,
                _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            )
            != complete_snapshot
        ):
            raise IntegrityError("directory publication state identity changed during completion")
        os.fsync(staging_descriptor)
        os.fsync(parent_descriptor)
        _verify_pinned_directory_regular_file_inventory(
            staging_descriptor,
            required | {_DIRECTORY_PUBLICATION_COMPLETE_BASENAME},
            _allow_publication_state_names=True,
        )
        _require_publication_state_payload_at(
            staging_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            complete_payload,
        )
        # Retain the holder and its authenticated INCOMPLETE marker.  Removing
        # either shared pathname would reintroduce an unconditional deletion
        # race against a same-UID process.
    except BaseException:
        if directory_published:
            if complete_published:
                try:
                    _native_rename_entry_create_only_at(
                        staging_descriptor,
                        _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
                        holder_descriptor,
                        _DIRECTORY_PUBLICATION_FAILED_COMPLETE_BASENAME,
                    )
                    complete_published = False
                except IntegrityError:
                    pass
            if incomplete_retained and not complete_published:
                try:
                    _native_rename_entry_create_only_at(
                        holder_descriptor,
                        _DIRECTORY_PUBLICATION_RETAINED_INCOMPLETE_BASENAME,
                        staging_descriptor,
                        _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME,
                    )
                    incomplete_retained = False
                except IntegrityError:
                    pass
            try:
                _native_rename_entry_create_only_at(
                    parent_descriptor,
                    publication.name,
                    holder_descriptor,
                    _DIRECTORY_PUBLICATION_FAILED_PAYLOAD_BASENAME,
                )
                directory_published = False
            except IntegrityError:
                # If quarantine loses a same-UID race, the descriptor-rooted
                # failure is still surfaced and no success is returned.  The
                # INCOMPLETE marker remains whenever it could be restored.
                pass
        raise
    finally:
        for descriptor in (complete_descriptor, incomplete_descriptor):
            if descriptor is not None:
                os.close(descriptor)
        reservation.close()
    return publication


def verify_complete_directory_publication(
    path: str | os.PathLike[str],
    *,
    required_relative_regular_files: Iterable[str],
) -> Path:
    """Consume only an exact tree carrying its inventory-bound COMPLETE state.

    This check pins both the lexical parent and publication root, rejects any
    INCOMPLETE state, and walks the caller-declared inventory through directory
    descriptors.  It authenticates the publication transaction boundary; as
    with all portable same-UID POSIX paths, it cannot make the tree immutable
    against later mutation by another process with the same authority.
    """

    publication = _absolute_without_resolving(path)
    _require_portable_basename(publication.name, field="publication directory basename")
    required = _normalize_required_regular_file_inventory(required_relative_regular_files)
    expected_complete_payload = _directory_publication_state_payload(
        required,
        state="COMPLETE_AFTER_POST_RENAME_DESCRIPTOR_REVALIDATION",
    )
    parent_descriptor = _open_directory_chain_no_follow(publication.parent, create=False)
    root_descriptor: int | None = None
    try:
        try:
            before_path = _entry_snapshot_at(parent_descriptor, publication.name)
            root_descriptor = os.open(
                publication.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise IntegrityError("directory publication is missing or inaccessible") from exc
        before_fd = _snapshot_from_stat(os.fstat(root_descriptor))
        if before_path != before_fd or not stat.S_ISDIR(before_fd[2]):
            raise IntegrityError("directory publication root identity changed")
        if _entry_exists_at(root_descriptor, _DIRECTORY_PUBLICATION_INCOMPLETE_BASENAME):
            raise IntegrityError("directory publication remains explicitly incomplete")
        _verify_pinned_directory_regular_file_inventory(
            root_descriptor,
            required | {_DIRECTORY_PUBLICATION_COMPLETE_BASENAME},
            _allow_publication_state_names=True,
        )
        _require_publication_state_payload_at(
            root_descriptor,
            _DIRECTORY_PUBLICATION_COMPLETE_BASENAME,
            expected_complete_payload,
        )
        if (
            _snapshot_from_stat(os.fstat(root_descriptor)) != before_fd
            or _entry_snapshot_at(parent_descriptor, publication.name) != before_fd
        ):
            raise IntegrityError("directory publication changed during completion verification")
    except IntegrityError:
        raise
    except OSError as exc:
        raise IntegrityError("directory publication changed during completion verification") from exc
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(parent_descriptor)
    return publication


def _exclusive_write_at(
    parent_descriptor: int,
    basename: str,
    display_path: Path,
    data: bytes,
) -> None:
    _require_portable_basename(basename, field="artifact basename")
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(basename, flags, 0o444, dir_fd=parent_descriptor)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written < 1:  # pragma: no cover - defensive OS contract
                raise OSError("immutable artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise IntegrityError(f"refusing to overwrite immutable artifact: {display_path}") from exc
    except OSError as exc:
        try:
            current_snapshot = _snapshot_from_stat(os.fstat(descriptor)) if descriptor is not None else None
            _unlink_matching_entry_at(
                parent_descriptor,
                basename,
                current_snapshot,
                expected_descriptor=descriptor,
            )
        except OSError:
            pass
        raise IntegrityError(f"cannot securely create immutable artifact: {display_path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _exclusive_write(path: Path, data: bytes) -> None:
    """Compatibility wrapper for one create-only write through a pinned parent."""

    parent_descriptor = _open_directory_chain_no_follow(path.parent, create=True)
    try:
        _exclusive_write_at(parent_descriptor, path.name, path, data)
    finally:
        os.close(parent_descriptor)


def _require_portable_basename(value: Any, *, field: str) -> str:
    """Require one literal filename, not a path or traversal expression."""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise IntegrityError(f"{field} must be one exact portable artifact basename")
    return value


def _validate_receipt_schema_name(value: Any) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise IntegrityError("receipt_schema must be one non-empty token")
    return value


def _receipt_document(
    destination: Path,
    document: Mapping[str, Any],
    *,
    artifact_payload: bytes,
    receipt_schema: str,
) -> dict[str, Any]:
    """Build a receipt from the exact bytes passed to the create-only writer."""

    schema = _validate_receipt_schema_name(receipt_schema)
    basename = _require_portable_basename(destination.name, field="artifact basename")
    receipt = {
        "schema": schema,
        "artifact_bytes": len(artifact_payload),
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "canonical_document_sha256": stable_sha256(dict(document)),
    }
    if schema == ARTIFACT_RECEIPT_SCHEMA_V1:
        # Legacy v1 remains available only through an explicit schema request.
        # Its absolute path is deliberately verified exactly for compatibility.
        receipt["artifact_path"] = str(destination)
    else:
        receipt["artifact_basename"] = basename
    return receipt


def write_immutable_json_with_receipt(
    path: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    receipt_schema: str = DEFAULT_ARTIFACT_RECEIPT_SCHEMA,
) -> dict[str, Any]:
    """Write a create-only JSON artifact and portable create-only receipt.

    New generic receipts default to v2 and bind the exact artifact basename,
    making a verified artifact/receipt pair relocatable as a unit.  Passing the
    legacy generic v1 schema explicitly retains the original absolute-path
    representation for controlled compatibility tests and migrations.  Other
    explicit custom schemas use the portable basename representation.
    """

    destination = _absolute_without_resolving(path)
    schema = _validate_receipt_schema_name(receipt_schema)
    _require_portable_basename(destination.name, field="artifact basename")
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    payload = (
        json.dumps(
            dict(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    receipt = _receipt_document(
        destination,
        document,
        artifact_payload=payload,
        receipt_schema=schema,
    )
    if receipt["artifact_bytes"] != len(payload):  # pragma: no cover - defensive race guard
        raise IntegrityError("artifact changed while its receipt was being constructed")
    receipt_payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    parent_descriptor = _open_directory_chain_no_follow(destination.parent, create=True)
    transaction_basename = _pair_transaction_basename(destination.name)
    token = secrets.token_hex(16)
    artifact_stage_basename = f".kbound-artifact-{token}.tmp"
    receipt_stage_basename = f".kbound-receipt-{token}.tmp"
    transaction_descriptor: int | None = None
    transaction_quarantined = False
    artifact_stage_snapshot: tuple[int, int, int, int, int] | None = None
    receipt_stage_snapshot: tuple[int, int, int, int, int] | None = None
    artifact_published_snapshot: tuple[int, int, int, int, int] | None = None
    receipt_published_snapshot: tuple[int, int, int, int, int] | None = None
    artifact_stage_descriptor: int | None = None
    receipt_stage_descriptor: int | None = None
    try:
        if _entry_exists_at(parent_descriptor, destination.name) or _entry_exists_at(
            parent_descriptor,
            receipt_path.name,
        ):
            raise IntegrityError(f"refusing to overwrite artifact/receipt pair for {destination}")
        try:
            os.mkdir(transaction_basename, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise IntegrityError("artifact/receipt publication is incomplete or still in progress") from exc
        transaction_descriptor = os.open(
            transaction_basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        transaction_snapshot = _snapshot_from_stat(os.fstat(transaction_descriptor))
        if transaction_snapshot != _entry_snapshot_at(parent_descriptor, transaction_basename):
            raise IntegrityError("artifact/receipt transaction workspace identity changed")
        os.fsync(parent_descriptor)
        if _entry_exists_at(parent_descriptor, destination.name) or _entry_exists_at(
            parent_descriptor,
            receipt_path.name,
        ):
            raise IntegrityError(f"artifact/receipt destination raced with publication for {destination}")
        _exclusive_write_at(
            transaction_descriptor,
            artifact_stage_basename,
            destination,
            payload,
        )
        artifact_stage_snapshot = _entry_snapshot_at(transaction_descriptor, artifact_stage_basename)
        artifact_stage_descriptor = _open_matching_regular_file_at(
            transaction_descriptor,
            artifact_stage_basename,
            artifact_stage_snapshot,
        )
        _exclusive_write_at(
            transaction_descriptor,
            receipt_stage_basename,
            receipt_path,
            receipt_payload,
        )
        receipt_stage_snapshot = _entry_snapshot_at(transaction_descriptor, receipt_stage_basename)
        receipt_stage_descriptor = _open_matching_regular_file_at(
            transaction_descriptor,
            receipt_stage_basename,
            receipt_stage_snapshot,
        )
        receipt_published_snapshot = _move_staged_entry_create_only_at(
            transaction_descriptor,
            receipt_stage_basename,
            parent_descriptor,
            receipt_path.name,
            receipt_path,
            source_descriptor=receipt_stage_descriptor,
            expected_source_snapshot=receipt_stage_snapshot,
        )
        receipt_stage_snapshot = None
        artifact_published_snapshot = _move_staged_entry_create_only_at(
            transaction_descriptor,
            artifact_stage_basename,
            parent_descriptor,
            destination.name,
            destination,
            source_descriptor=artifact_stage_descriptor,
            expected_source_snapshot=artifact_stage_snapshot,
        )
        artifact_stage_snapshot = None
        os.fsync(parent_descriptor)
        _quarantine_transaction_directory_at(
            parent_descriptor,
            transaction_basename,
            transaction_descriptor,
            require_empty=True,
        )
        transaction_quarantined = True
        os.fsync(parent_descriptor)
    except BaseException as exc:
        finals_clean = not isinstance(exc, _UnownedPublicationError)
        finals_clean = (
            _unlink_matching_entry_at(
                parent_descriptor,
                destination.name,
                artifact_published_snapshot,
                expected_descriptor=artifact_stage_descriptor,
            )
            and finals_clean
        )
        finals_clean = (
            _unlink_matching_entry_at(
                parent_descriptor,
                receipt_path.name,
                receipt_published_snapshot,
                expected_descriptor=receipt_stage_descriptor,
            )
            and finals_clean
        )
        if finals_clean and transaction_descriptor is not None and not transaction_quarantined:
            try:
                _quarantine_transaction_directory_at(
                    parent_descriptor,
                    transaction_basename,
                    transaction_descriptor,
                    require_empty=False,
                )
                transaction_quarantined = True
            except IntegrityError:
                pass
        try:
            os.fsync(parent_descriptor)
        except OSError:
            pass
        raise
    finally:
        for descriptor in (
            receipt_stage_descriptor,
            artifact_stage_descriptor,
            transaction_descriptor,
        ):
            if descriptor is not None:
                os.close(descriptor)
        os.close(parent_descriptor)
    return receipt


def load_verified_json_mapping_with_receipt(
    artifact_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str] | None = None,
    *,
    receipt_schema: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Securely consume a mapping and its portable, legacy, or custom receipt.

    With no schema argument, only the two generic schemas are accepted.  A
    custom schema must be requested explicitly and uses the portable basename
    layout.  Legacy v1 verification remains path-exact: moving only one member
    of a v1 artifact/receipt pair is rejected rather than silently rebased.
    """

    artifact = _absolute_without_resolving(artifact_path)
    artifact_basename = _require_portable_basename(artifact.name, field="artifact basename")
    receipt_file = (
        _absolute_without_resolving(receipt_path)
        if receipt_path is not None
        else artifact.with_name(artifact.name + ".receipt.json")
    )
    (document, artifact_payload), (receipt, _) = read_secure_json_mapping_pair_once(
        artifact,
        receipt_file,
    )
    observed_schema = receipt.get("schema")
    if not isinstance(observed_schema, str):
        raise IntegrityError("unknown So2Sat receipt schema")
    if receipt_schema is None:
        accepted_schemas = {ARTIFACT_RECEIPT_SCHEMA_V1, ARTIFACT_RECEIPT_SCHEMA_V2}
    else:
        accepted_schemas = {_validate_receipt_schema_name(receipt_schema)}
    if observed_schema not in accepted_schemas:
        raise IntegrityError("unknown So2Sat receipt schema")
    if observed_schema == ARTIFACT_RECEIPT_SCHEMA_V1:
        expected_fields = _RECEIPT_COMMON_FIELDS | {"artifact_path"}
        if set(receipt) != expected_fields:
            raise IntegrityError("legacy receipt has unknown or missing fields")
        claimed_path = receipt.get("artifact_path")
        if not isinstance(claimed_path, str) or not Path(claimed_path).is_absolute():
            raise IntegrityError("legacy receipt artifact_path must be absolute")
        if Path(claimed_path).name != artifact_basename:
            raise IntegrityError("legacy receipt artifact basename mismatch")
        if claimed_path != str(artifact):
            raise IntegrityError("legacy receipt artifact_path mismatch")
    else:
        expected_fields = _RECEIPT_COMMON_FIELDS | {"artifact_basename"}
        if set(receipt) != expected_fields:
            raise IntegrityError("portable receipt has unknown or missing fields")
        claimed_basename = _require_portable_basename(
            receipt.get("artifact_basename"),
            field="receipt artifact_basename",
        )
        if claimed_basename != artifact_basename:
            raise IntegrityError("portable receipt artifact_basename mismatch")
    artifact_bytes = receipt.get("artifact_bytes")
    if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int) or artifact_bytes < 1:
        raise IntegrityError("receipt artifact_bytes must be a positive integer")
    if artifact_bytes != len(artifact_payload):
        raise IntegrityError("receipt byte count mismatch")
    artifact_sha256 = require_sha256(receipt.get("artifact_sha256"), field="artifact_sha256")
    if artifact_sha256 != hashlib.sha256(artifact_payload).hexdigest():
        raise IntegrityError("receipt file SHA-256 mismatch")
    canonical_sha256 = require_sha256(
        receipt.get("canonical_document_sha256"),
        field="canonical_document_sha256",
    )
    if canonical_sha256 != stable_sha256(dict(document)):
        raise IntegrityError("receipt canonical document SHA-256 mismatch")
    return dict(document), dict(receipt)


def verify_artifact_receipt(
    artifact_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str] | None = None,
    *,
    receipt_schema: str | None = None,
) -> dict[str, Any]:
    """Verify a receipt through the same read-once path used by consumers."""

    _, receipt = load_verified_json_mapping_with_receipt(
        artifact_path,
        receipt_path,
        receipt_schema=receipt_schema,
    )
    return receipt
