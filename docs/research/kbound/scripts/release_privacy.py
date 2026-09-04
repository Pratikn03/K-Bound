#!/usr/bin/env python3
"""Fail-closed privacy and deterministic-archive primitives for K-Bound releases."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import zipfile
import zlib
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MAX_NESTED_DEPTH = 3
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


class PrivacyError(RuntimeError):
    """A release member is unsafe, ambiguous, or not anonymously publishable."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_public_member_path(value: str) -> str:
    """Return one canonical relative POSIX path or fail closed."""

    if not value or value == "." or "\x00" in value or "\\" in value:
        raise PrivacyError(f"unsafe public member path: {value!r}")
    lowered = value.lower()
    if value.startswith(("/", "~", "$HOME/", "${HOME}/")) or lowered.startswith(("file:", "%userprofile%/")):
        raise PrivacyError(f"unsafe public member path: {value!r}")
    if re.match(r"^[A-Za-z]:", value) or value.startswith("//"):
        raise PrivacyError(f"unsafe public member path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != value:
        raise PrivacyError(f"unsafe public member path: {value!r}")
    if any(part in {"", "."} for part in parsed.parts):
        raise PrivacyError(f"unsafe public member path: {value!r}")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PrivacyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise PrivacyError(f"non-finite JSON number: {value}")


def strict_json_loads(payload: bytes) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except PrivacyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivacyError(f"invalid JSON: {exc}") from exc


def _forbidden_member_reason(relative: str) -> str | None:
    parts = PurePosixPath(relative).parts
    name = parts[-1]
    if ".git" in parts:
        return "forbidden member (.git)"
    if name == ".DS_Store" or name.startswith("._") or name.startswith("~$"):
        return "forbidden member (temporary/editor metadata)"
    lowered = relative.lower()
    if lowered in {
        "word/comments.xml",
        "word/commentsextended.xml",
        "word/commentids.xml",
        "word/people.xml",
    }:
        return "forbidden OOXML comment metadata member"
    return None


_IDENTITY_RE = re.compile(r"(?:Pratik\s+Niroula|Pratikn03|pratik_n|mnsu\.edu)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])")
_PRIVATE_POSIX_RE = re.compile(r"/(?:Users|home|Volumes)/[^\s\x00<>\"']+")
_ABSOLUTE_POSIX_RE = re.compile(
    r"(?<![:/A-Za-z0-9])/(?:private|tmp|var|data|mnt|opt|etc|srv|workspace)"
    r"(?:/[A-Za-z0-9._~+@=-]+)+",
    re.IGNORECASE,
)
_FILE_URI_RE = re.compile(r"\bfile:(?:[/\\]+|[A-Za-z]:[/\\])[^\s\x00<>\"']*", re.IGNORECASE)
_HOME_ALIAS_RE = re.compile(r"(?:\$HOME|\$\{HOME\}|%USERPROFILE%)[/\\]", re.IGNORECASE)
_WINDOWS_PRIVATE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\(?:Users|home|Volumes|private|tmp|var|data|"
    r"work|Documents|Desktop|Downloads)(?:\\[^\s\x00<>\"']+)+|"
    r"\\\\[^\\\s]+\\[^\\\s]+)",
    re.IGNORECASE,
)
_TRACK_CHANGE_RE = re.compile(r"<(?:w:)?(?:ins|del)\b|\bw:author\s*=", re.IGNORECASE)
_SAFE_SHEBANG_RE = re.compile(
    r"\A#!(?:/usr/bin/env[ \t]+(?:python3?|bash|sh)|"
    r"/usr/bin/python3?|/bin/(?:bash|sh))(?:[ \t]+[^\r\n]*)?(?:\r?\n|$)"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_METADATA_CHUNKS = frozenset({b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"})
_PNG_SAFE_CHUNKS = frozenset(
    {
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"cHRM",
        b"gAMA",
        b"sBIT",
        b"sRGB",
        b"bKGD",
        b"hIST",
        b"tRNS",
        b"pHYs",
        b"acTL",
        b"fcTL",
        b"fdAT",
    }
)


def sanitize_png_metadata(payload: bytes) -> bytes:
    """Validate a PNG and remove chunks that can carry private metadata."""

    if not payload.startswith(_PNG_SIGNATURE):
        raise PrivacyError("malformed PNG signature")
    output = bytearray(_PNG_SIGNATURE)
    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    saw_iend = False
    saw_ihdr = False
    saw_idat = False
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise PrivacyError("truncated PNG chunk")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        end = offset + 12 + length
        if length > MAX_MEMBER_BYTES or end > len(payload):
            raise PrivacyError("oversized or truncated PNG chunk")
        chunk_type = payload[offset + 4 : offset + 8]
        if len(chunk_type) != 4 or not all(
            ord("A") <= value <= ord("Z") or ord("a") <= value <= ord("z") for value in chunk_type
        ):
            raise PrivacyError("malformed PNG chunk type")
        chunk_data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        observed_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise PrivacyError("PNG chunk CRC mismatch")
        if chunk_index == 0 and chunk_type != b"IHDR":
            raise PrivacyError("PNG does not begin with IHDR")
        if chunk_type == b"IHDR":
            if saw_ihdr or length != 13:
                raise PrivacyError("malformed or duplicate PNG IHDR")
            saw_ihdr = True
        if chunk_type == b"IDAT":
            saw_idat = True
        if saw_iend:
            raise PrivacyError("PNG contains data after IEND")
        if chunk_type == b"IEND":
            if length != 0:
                raise PrivacyError("malformed PNG IEND")
            saw_iend = True
        if chunk_type not in _PNG_SAFE_CHUNKS and chunk_type not in _PNG_METADATA_CHUNKS:
            # An uppercase first byte denotes a critical chunk. Unknown critical
            # semantics cannot be scrubbed without risking a misleading image.
            if not (chunk_type[0] & 0x20):
                raise PrivacyError(f"unknown critical PNG chunk: {chunk_type!r}")
        elif chunk_type in _PNG_SAFE_CHUNKS:
            output.extend(payload[offset:end])
        offset = end
        chunk_index += 1
    if not saw_ihdr or not saw_idat or not saw_iend:
        raise PrivacyError("PNG lacks a required IHDR, IDAT, or IEND chunk")
    return bytes(output)


def _scan_text(relative: str, text: str) -> None:
    if _IDENTITY_RE.search(text):
        raise PrivacyError(f"identity disclosure in {relative}")
    if _EMAIL_RE.search(text):
        raise PrivacyError(f"email disclosure in {relative}")
    if _PRIVATE_POSIX_RE.search(text):
        raise PrivacyError(f"private POSIX path disclosure in {relative}")
    if _FILE_URI_RE.search(text):
        raise PrivacyError(f"file URI disclosure in {relative}")
    if _HOME_ALIAS_RE.search(text):
        raise PrivacyError(f"home alias disclosure in {relative}")
    # A standard interpreter shebang is portable executable metadata, not a
    # source-machine data location.  Exempt only that first line; absolute
    # paths anywhere else remain release-blocking.
    absolute_path_text = _SAFE_SHEBANG_RE.sub("", text, count=1)
    if _ABSOLUTE_POSIX_RE.search(absolute_path_text):
        raise PrivacyError(f"absolute POSIX path disclosure in {relative}")
    if _WINDOWS_PRIVATE_RE.search(text):
        raise PrivacyError(f"private Windows/UNC path disclosure in {relative}")
    if _TRACK_CHANGE_RE.search(text):
        raise PrivacyError(f"tracked-change or author metadata in {relative}")


def _scan_zip(payload: bytes, *, parent: str, depth: int) -> None:
    if depth > MAX_NESTED_DEPTH:
        raise PrivacyError(f"nested archive depth exceeds {MAX_NESTED_DEPTH}: {parent}")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (OSError, zipfile.BadZipFile) as exc:
        raise PrivacyError(f"malformed ZIP/OOXML archive in {parent}: {exc}") from exc
    with archive:
        if archive.comment:
            raise PrivacyError(f"ZIP archive comment metadata in {parent}")
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if not names:
            raise PrivacyError(f"empty nested archive: {parent}")
        if len(names) != len(set(names)):
            raise PrivacyError(f"duplicate nested archive member in {parent}")
        total = 0
        for info in infos:
            relative = validate_public_member_path(info.filename.rstrip("/"))
            if info.comment or info.extra:
                raise PrivacyError(f"ZIP member metadata in {parent}!{relative}")
            if info.flag_bits & 0x1:
                raise PrivacyError(f"encrypted archive member: {parent}!{relative}")
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise PrivacyError(f"oversized archive member: {parent}!{relative}")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise PrivacyError(f"archive expands beyond safety limit: {parent}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PrivacyError(f"symlink archive member: {parent}!{relative}")
            try:
                member_payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PrivacyError(f"cannot read archive member {parent}!{relative}: {exc}") from exc
            scan_member(
                f"{parent}!{relative}",
                member_payload,
                _depth=depth,
                _validate_outer=False,
            )
        if infos[0].header_offset != 0:
            raise PrivacyError(f"noncanonical nested ZIP structure: {parent}")
        with io.BytesIO(payload) as handle:
            _verify_local_zip_headers(handle, infos, central_offset=archive.start_dir)
        if len(payload) < 22 or payload[-22:-18] != b"PK\x05\x06" or payload[-2:] != b"\x00\x00":
            raise PrivacyError(f"noncanonical nested ZIP structure: {parent}")


def scan_member(
    relative: str,
    payload: bytes,
    *,
    _depth: int = 0,
    _validate_outer: bool = True,
) -> None:
    """Reject identity, path, VCS, editor, OOXML, and nested-archive leaks."""

    outer = relative.split("!", 1)[0] if not _validate_outer else relative
    member = relative.rsplit("!", 1)[-1]
    if _validate_outer:
        validate_public_member_path(relative)
    else:
        validate_public_member_path(member)
    reason = _forbidden_member_reason(member)
    if reason:
        raise PrivacyError(f"{reason}: {relative}")
    if len(payload) > MAX_MEMBER_BYTES:
        raise PrivacyError(f"oversized release member: {relative}")
    if payload.startswith(_PNG_SIGNATURE):
        if sanitize_png_metadata(payload) != payload:
            raise PrivacyError(f"PNG metadata requires scrubbing: {relative}")
        return
    if payload.startswith(b"%PDF-"):
        raise PrivacyError(f"PDF requires type-specific privacy verification: {relative}")
    # ZIP permits an executable/preamble before its first local header, so a
    # first-four-byte signature check is not a safe archive detector.
    if b"PK" in payload and zipfile.is_zipfile(io.BytesIO(payload)):
        _scan_zip(payload, parent=relative, depth=_depth + 1)
        return
    if member.lower().endswith(".json"):
        strict_json_loads(payload)
    decoded: list[str] = [payload.decode("utf-8", errors="ignore")]
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")) or (payload and payload.count(b"\x00") * 4 > len(payload)):
        try:
            decoded.append(payload.decode("utf-16"))
        except UnicodeDecodeError as exc:
            raise PrivacyError(f"malformed UTF-16 text in {relative}: {exc}") from exc
    for text in decoded:
        _scan_text(relative, text)
    # Avoid an unused-variable false positive while retaining the explicit
    # distinction between the outer release path and nested member path.
    del outer


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits |= 0x800
    return info


def _verify_local_zip_headers(
    handle: BinaryIO,
    infos: list[zipfile.ZipInfo],
    *,
    central_offset: int,
) -> None:
    """Reject metadata or hidden gaps present only in ZIP local headers."""

    fixed_dos_time = 0
    fixed_dos_date = 1 | (1 << 5)
    for index, info in enumerate(infos):
        handle.seek(info.header_offset)
        raw_header = handle.read(30)
        if len(raw_header) != 30:
            raise PrivacyError(f"truncated ZIP local header: {info.filename}")
        (
            signature,
            extract_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
        ) = struct.unpack("<4s5H3I2H", raw_header)
        if signature != b"PK\x03\x04":
            raise PrivacyError(f"invalid ZIP local header: {info.filename}")
        raw_name = handle.read(name_size)
        encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            local_name = raw_name.decode(encoding)
        except UnicodeDecodeError as exc:
            raise PrivacyError(f"malformed ZIP local filename: {info.filename}") from exc
        if extra_size:
            raise PrivacyError(f"noncanonical ZIP local-header metadata: {info.filename}")
        if (
            local_name != info.filename
            or extract_version != info.extract_version
            or flags != info.flag_bits
            or flags & ~0x800
            or compression != info.compress_type
            or modified_time != fixed_dos_time
            or modified_date != fixed_dos_date
            or crc != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
        ):
            raise PrivacyError(f"noncanonical ZIP local header: {info.filename}")
        payload_end = info.header_offset + 30 + name_size + compressed_size
        next_offset = infos[index + 1].header_offset if index + 1 < len(infos) else central_offset
        if payload_end != next_offset:
            raise PrivacyError("anonymous archive has non-canonical ZIP structure")


def _normalize_verified_pdf_members(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise PrivacyError("verified PDF allowlist must be a collection of member paths")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise PrivacyError("verified PDF allowlist contains a non-string member path")
        canonical = validate_public_member_path(value)
        if PurePosixPath(canonical).suffix.lower() != ".pdf":
            raise PrivacyError(f"verified PDF allowlist contains a non-PDF member: {canonical}")
        normalized.add(canonical)
    return frozenset(normalized)


def build_deterministic_zip(
    path: Path,
    entries: Mapping[str, bytes],
    *,
    verified_pdf_members: Collection[str] = (),
) -> None:
    """Validate all entries first, then atomically publish deterministic bytes."""

    if path.exists() or path.is_symlink():
        raise PrivacyError(f"refusing to overwrite anonymous archive: {path}")
    parent_parts = path.parent.absolute().parts
    if any(Path(*parent_parts[:index]).is_symlink() for index in range(1, len(parent_parts) + 1)):
        raise PrivacyError(f"anonymous archive output parent is a symlink: {path.parent}")
    verified_pdfs = _normalize_verified_pdf_members(verified_pdf_members)
    normalized: dict[str, bytes] = {}
    for relative, payload in entries.items():
        canonical = validate_public_member_path(relative)
        if canonical in normalized:
            raise PrivacyError(f"duplicate release member: {canonical}")
        if not isinstance(payload, bytes):
            raise PrivacyError(f"release payload is not bytes: {canonical}")
        if canonical in verified_pdfs:
            if not payload.startswith(b"%PDF-"):
                raise PrivacyError(f"verified PDF member is not a PDF: {canonical}")
            if len(payload) > MAX_MEMBER_BYTES:
                raise PrivacyError(f"oversized release member: {canonical}")
        else:
            scan_member(canonical, payload)
        normalized[canonical] = payload
    if not normalized:
        raise PrivacyError("anonymous archive would be empty")
    unknown_verified_pdfs = verified_pdfs.difference(normalized)
    if unknown_verified_pdfs:
        raise PrivacyError(
            "verified PDF allowlist names missing archive members: " + ", ".join(sorted(unknown_verified_pdfs))
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for relative in sorted(normalized):
                archive.writestr(
                    _zip_info(relative),
                    normalized[relative],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        verify_anonymous_zip(temporary, verified_pdf_members=verified_pdfs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_anonymous_zip(
    path: Path,
    *,
    verified_pdf_members: Collection[str] = (),
) -> dict[str, Any]:
    verified_pdfs = _normalize_verified_pdf_members(verified_pdf_members)
    if path.is_symlink() or not path.is_file():
        raise PrivacyError(f"anonymous archive is missing or a symlink: {path}")
    archive_bytes = path.stat().st_size
    if archive_bytes > MAX_TOTAL_BYTES:
        raise PrivacyError("anonymous archive exceeds the compressed-size safety limit")
    if archive_bytes < 22:
        raise PrivacyError("anonymous archive is malformed: truncated ZIP")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise PrivacyError(f"anonymous archive is malformed: {exc}") from exc
    with archive:
        if archive.comment:
            raise PrivacyError("noncanonical ZIP archive comment metadata")
        with path.open("rb") as handle:
            handle.seek(-22, os.SEEK_END)
            end_record = handle.read(22)
        if len(end_record) != 22 or end_record[:4] != b"PK\x05\x06" or end_record[-2:] != b"\x00\x00":
            raise PrivacyError("anonymous archive has non-canonical ZIP structure")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise PrivacyError("anonymous archive members are empty, duplicated, or non-canonical")
        if infos[0].header_offset != 0:
            raise PrivacyError("anonymous archive has non-canonical ZIP structure")
        unknown_verified_pdfs = verified_pdfs.difference(names)
        if unknown_verified_pdfs:
            raise PrivacyError(
                "verified PDF allowlist names missing archive members: " + ", ".join(sorted(unknown_verified_pdfs))
            )
        expanded_bytes = 0
        for info in infos:
            if info.file_size > MAX_MEMBER_BYTES:
                raise PrivacyError(f"oversized ZIP member: {info.filename}")
            expanded_bytes += info.file_size
            if expanded_bytes > MAX_TOTAL_BYTES:
                raise PrivacyError("anonymous archive expands beyond safety limit")
        for info in infos:
            validate_public_member_path(info.filename)
            if info.comment or info.extra:
                raise PrivacyError(f"noncanonical ZIP member metadata: {info.filename}")
            if info.date_time != FIXED_ZIP_TIME:
                raise PrivacyError(f"nondeterministic ZIP timestamp: {info.filename}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise PrivacyError(f"noncanonical ZIP compression: {info.filename}")
            mode = info.external_attr >> 16
            if info.create_system != 3 or mode != (stat.S_IFREG | 0o644):
                raise PrivacyError(f"noncanonical ZIP mode: {info.filename}")
            if info.flag_bits & 0x1:
                raise PrivacyError(f"encrypted ZIP member: {info.filename}")
        with path.open("rb") as handle:
            _verify_local_zip_headers(handle, infos, central_offset=archive.start_dir)
        for info in infos:
            payload = archive.read(info)
            if info.filename in verified_pdfs:
                if not payload.startswith(b"%PDF-"):
                    raise PrivacyError(f"verified PDF member is not a PDF: {info.filename}")
            else:
                scan_member(info.filename, payload)
    return {
        "schema": "kbound-anonymous-archive-verification-v1",
        "status": "PASS",
        "member_count": len(infos),
        "archive_bytes": archive_bytes,
        "archive_sha256": file_sha256(path),
    }
