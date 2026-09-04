#!/usr/bin/env python3
"""Build a deterministic, source-bound anonymous supplementary archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from docs.research.kbound.scripts import build_release_source_seal as source_seal
from docs.research.kbound.scripts import verify_python_environment
from docs.research.kbound.scripts.build_cct20_public_bundle import verify_public_bundle
from docs.research.kbound.scripts.release_privacy import (
    FIXED_ZIP_TIME,
    MAX_MEMBER_BYTES,
    MAX_TOTAL_BYTES,
    PrivacyError,
    build_deterministic_zip,
    sanitize_png_metadata,
    scan_member,
    strict_json_loads,
    validate_public_member_path,
    verify_anonymous_zip,
)
from docs.research.kbound.scripts.verify_release_checksums import (
    REQUIRED_RELEASE_PATHS,
    verify_checksum_file,
)

SCHEMA = "kbound-anonymous-supplement-v2"
DEFAULT_SOURCE_SEAL = Path("docs/research/kbound/audits/release_source_seal_2026_08_29.json")
DEFAULT_CHECKSUMS = Path("docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt")
DEFAULT_PUBLIC_BUNDLE = Path("docs/research/kbound/release/cct20_public_evidence_bundle.zip")
DEFAULT_OUTPUT = Path("docs/research/kbound/release/kbound_anonymous_supplement.zip")
PORTABLE_PDF_TOOL_SHA256 = {
    "pdfdetach": "d11c62315753939947bbfff4a41068bda8fdc1b2dd360ee70977f82c87413d16",
    "pdfinfo": "e30dc1a286803e8909b2344b55647d120d78627679a7f7ed7870e1824a9ce870",
    "pdftotext": "4633f0c9053159fc19951e53e75807b8f2fe33bcb7ecb6a105d5da3d286c3c2b",
}
PORTABLE_POPPLER_LIBRARY_SHA256 = "7dabb3e94a0eb93449e890b98bfac4a6bb223b6901a0ebcf13e734e053a61930"
_TMLR_DRIVER = "docs/research/kbound/kbound_tmlr.tex"
_TMLR_PDF = "docs/research/kbound/kbound_tmlr.pdf"
_FORBIDDEN_NAMED_INPUTS = frozenset(
    {
        "CITATION.cff",
        "README.md",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_submission.tex",
    }
)
_TEX_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_TEX_GRAPHIC_RE = re.compile(r"\\(?:kbgraphics|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}")
_TEX_PACKAGE_RE = re.compile(r"\\usepackage(?:\[[^]]*\])?\{([^}]+)\}")
_LITERAL_TEX_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+")
_ALLOWED_SUFFIXES = frozenset(
    {
        ".bib",
        ".csv",
        ".docx",
        ".json",
        ".md",
        ".pdf",
        ".png",
        ".sha256",
        ".sty",
        ".tex",
        ".txt",
        ".yaml",
        ".yml",
        ".zip",
    }
)
_TRACKED_CHANGE = re.compile(rb"<(?:[A-Za-z0-9_]+:)?(?:ins|del)\b", re.IGNORECASE)
_COMMENT_MARKER = frozenset({"commentRangeStart", "commentRangeEnd", "commentReference"})
_REVISION_MARKERS = frozenset(
    {
        "cellDel",
        "cellIns",
        "cellMerge",
        "customXmlDelRangeEnd",
        "customXmlDelRangeStart",
        "customXmlInsRangeEnd",
        "customXmlInsRangeStart",
        "customXmlMoveFromRangeEnd",
        "customXmlMoveFromRangeStart",
        "customXmlMoveToRangeEnd",
        "customXmlMoveToRangeStart",
        "del",
        "ins",
        "moveFrom",
        "moveFromRangeEnd",
        "moveFromRangeStart",
        "moveTo",
        "moveToRangeEnd",
        "moveToRangeStart",
        "numberingChange",
        "pPrChange",
        "rPrChange",
        "sectPrChange",
        "tblPrChange",
        "tcPrChange",
        "trPrChange",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "normalization",
        "published_bytes",
        "published_path",
        "published_sha256",
        "source_bytes",
        "source_path",
        "source_sha256",
    }
)
_ARTIFACT_NORMALIZATIONS = frozenset(
    {
        "byte-identical-copy",
        "deterministic-ooxml-metadata-scrub",
        "validated-png-metadata-scrub",
        "verified-pdf-byte-identical-copy",
    }
)
_LOWER_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_ALLOWED_PDFINFO_FIELDS = frozenset(
    {
        "Title",
        "Subject",
        "Keywords",
        "Author",
        "Creator",
        "Producer",
        "CreationDate",
        "ModDate",
        "Custom Metadata",
        "Metadata Stream",
        "Tagged",
        "UserProperties",
        "Suspects",
        "Form",
        "JavaScript",
        "Pages",
        "Encrypted",
        "Page size",
        "Page rot",
        "File size",
        "Optimized",
        "PDF version",
        "PDF subtype",
        "PDF subtype part",
        "PDF subtype conformance",
    }
)
_ALLOWED_XMP_NAMESPACES = frozenset(
    {
        "adobe:ns:meta/",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://purl.org/dc/elements/1.1/",
        "http://ns.adobe.com/xap/1.0/",
        "http://ns.adobe.com/pdf/1.3/",
        "http://ns.adobe.com/photoshop/1.0/",
    }
)
_FORBIDDEN_XMP_TEXT_FIELDS = frozenset({"creator", "subject", "Producer", "CreatorTool", "AuthorsPosition"})
_TEX_AUTHOR_RE = re.compile(r"\\author\s*\{([^{}]*)\}", re.DOTALL)
_TEX_IDENTITY_MACRO_RE = re.compile(
    r"\\(?:email|affiliation|institute|address)\s*\{([^{}]*)\}", re.DOTALL | re.IGNORECASE
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _valid_size(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _LOWER_HEX_64_RE.fullmatch(value) is not None


def _validate_artifact_ledger(value: object) -> list[dict[str, Any]]:
    """Validate the complete, canonical published-artifact ledger."""

    if not isinstance(value, list) or not value:
        raise PrivacyError("anonymous artifact inventory must be a non-empty list")
    validated: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _ARTIFACT_FIELDS:
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}")
        published_path = row.get("published_path")
        try:
            relative = validate_public_member_path(published_path if isinstance(published_path, str) else "")
        except PrivacyError as exc:
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}: {exc}") from exc
        if not relative.startswith("artifacts/"):
            raise PrivacyError(f"anonymous artifact is outside artifacts/: {relative}")
        source_path = row.get("source_path")
        try:
            source_relative = validate_public_member_path(source_path if isinstance(source_path, str) else "")
        except PrivacyError as exc:
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}: {exc}") from exc
        if relative != "artifacts/" + source_relative:
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}")
        if (
            row.get("normalization") not in _ARTIFACT_NORMALIZATIONS
            or not _valid_size(row.get("published_bytes"))
            or not _valid_size(row.get("source_bytes"))
            or not _valid_sha256(row.get("published_sha256"))
            or not _valid_sha256(row.get("source_sha256"))
        ):
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}")
        if row["normalization"] in {
            "byte-identical-copy",
            "verified-pdf-byte-identical-copy",
        } and (row["published_bytes"] != row["source_bytes"] or row["published_sha256"] != row["source_sha256"]):
            raise PrivacyError(f"anonymous artifact provenance is malformed at row {index}")
        paths.append(relative)
        validated.append(row)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PrivacyError("anonymous artifact paths are duplicated or non-canonical")
    return validated


def _parse_checksum_payload(payload: bytes) -> dict[str, str]:
    """Parse one canonical, complete K-Bound release checksum receipt."""

    if len(payload) > MAX_MEMBER_BYTES:
        raise PrivacyError("release checksum receipt exceeds the safety limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrivacyError(f"release checksum receipt is not UTF-8: {exc}") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise PrivacyError("release checksum receipt is not canonical text")
    entries: dict[str, str] = {}
    order: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise PrivacyError(f"malformed release checksum line {line_number}")
        digest, raw_relative = match.groups()
        try:
            relative = validate_public_member_path(raw_relative)
        except PrivacyError as exc:
            raise PrivacyError(f"unsafe release checksum path on line {line_number}: {raw_relative}") from exc
        if relative in entries:
            raise PrivacyError(f"duplicate release checksum entry: {relative}")
        entries[relative] = digest
        order.append(relative)
    missing = [relative for relative in REQUIRED_RELEASE_PATHS if relative not in entries]
    if missing:
        raise PrivacyError("required checksum entries are missing: " + ", ".join(missing))
    required = list(REQUIRED_RELEASE_PATHS)
    extras = sorted(set(order) - set(REQUIRED_RELEASE_PATHS))
    if order != required + extras:
        raise PrivacyError("release checksum entries are not in canonical inventory order")
    return entries


def _relative_control_path(root: Path, path: Path, *, label: str) -> str:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise PrivacyError(f"{label} is outside the release repository: {path}") from exc
    validated = validate_public_member_path(relative)
    if not isinstance(validated, str):
        raise PrivacyError(f"{label} did not resolve to a public member path: {path}")
    return validated


def _source_authorities(seal: dict[str, object]) -> dict[str, tuple[int, str]]:
    rows = seal.get("artifacts")
    if not isinstance(rows, list):
        raise PrivacyError("release source seal artifact ledger is malformed")
    result: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PrivacyError("release source seal artifact ledger is malformed")
        path = str(row.get("path", ""))
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _valid_sha256(digest)
        ):
            raise PrivacyError(f"release source seal authority is malformed: {path}")
        result[path] = (size, digest)
    return result


def _require_source_binding(
    *,
    relative: str,
    size: int,
    digest: str,
    checksums: dict[str, str],
    sealed: dict[str, tuple[int, str]],
) -> None:
    authorities = 0
    expected_checksum = checksums.get(relative)
    if expected_checksum is not None:
        authorities += 1
        if expected_checksum != digest:
            raise PrivacyError(f"anonymous artifact source authority mismatch: {relative}")
    expected_seal = sealed.get(relative)
    if expected_seal is not None:
        authorities += 1
        if expected_seal != (size, digest):
            raise PrivacyError(f"anonymous artifact source authority mismatch: {relative}")
    if authorities == 0:
        raise PrivacyError(f"anonymous artifact has no source authority: {relative}")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def _remove_named_elements(root: ET.Element, names: frozenset[str]) -> None:
    for parent in root.iter():
        for child in list(parent):
            if _local_name(child.tag) in names:
                parent.remove(child)


def _deterministic_zip_bytes(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True
    ) as archive:
        for relative in sorted(entries):
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.flag_bits |= 0x800
            archive.writestr(info, entries[relative], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def _normalize_xml(relative: str, payload: bytes) -> bytes | None:
    lowered = relative.lower()
    if lowered in {"word/comments.xml", "word/commentsextended.xml", "word/commentids.xml", "word/people.xml"}:
        return None
    if lowered.startswith("docprops/") and lowered not in {
        "docprops/core.xml",
        "docprops/app.xml",
    }:
        return None
    if _TRACKED_CHANGE.search(payload):
        raise PrivacyError(f"tracked-change markup in {relative}")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise PrivacyError(f"malformed OOXML XML member {relative}: {exc}") from exc
    for element in root.iter():
        local_tag = _local_name(element.tag)
        if local_tag in _REVISION_MARKERS or any(
            _local_name(attribute) in {"author", "initials"} for attribute in element.attrib
        ):
            raise PrivacyError(f"revision metadata in {relative}")
    if lowered == "docprops/core.xml":
        for element in root.iter():
            local = _local_name(element.tag)
            if local in {"creator", "lastModifiedBy"}:
                element.text = "Anonymous"
            elif local not in {"coreProperties", "created", "modified", "revision", "version"}:
                element.text = None
    elif lowered == "docprops/app.xml":
        for element in root.iter():
            local = _local_name(element.tag)
            if local == "Application":
                element.text = "Anonymous"
            elif local == "AppVersion":
                element.text = "0"
            elif local != "Properties":
                element.text = None
    elif lowered.endswith(".rels"):
        for child in list(root):
            target = str(child.attrib.get("Target", "")).lower()
            relation_type = str(child.attrib.get("Type", "")).lower()
            if (
                "comment" in target
                or "comment" in relation_type
                or "people" in relation_type
                or "custom-properties" in relation_type
                or "docprops/custom" in target
            ):
                root.remove(child)
    elif lowered == "[content_types].xml":
        for child in list(root):
            part_name = str(child.attrib.get("PartName", "")).lower()
            content_type = str(child.attrib.get("ContentType", "")).lower()
            if "comment" in part_name or "custom-properties" in content_type or "/docprops/custom" in part_name:
                root.remove(child)
    else:
        _remove_named_elements(root, _COMMENT_MARKER)
    return bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def sanitize_ooxml(relative: str, payload: bytes) -> bytes:
    """Remove known identity metadata/comments and reject tracked changes."""

    try:
        source = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise PrivacyError(f"malformed OOXML archive in {relative}: {exc}") from exc
    entries: dict[str, bytes] = {}
    with source:
        infos = source.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise PrivacyError(f"duplicate OOXML member in {relative}")
        total = 0
        for info in infos:
            name = validate_public_member_path(info.filename.rstrip("/"))
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                raise PrivacyError(f"oversized OOXML member: {name}")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise PrivacyError(f"OOXML archive expands beyond safety limit: {relative}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise PrivacyError(f"symlink OOXML member: {name}")
            if info.flag_bits & 0x1:
                raise PrivacyError(f"encrypted OOXML member: {name}")
            try:
                raw = source.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PrivacyError(f"cannot read OOXML member {name}: {exc}") from exc
            if name.lower().endswith((".xml", ".rels")):
                normalized = _normalize_xml(name, raw)
                if normalized is None:
                    continue
                raw = normalized
            elif name.lower().startswith("docprops/"):
                continue
            entries[name] = raw
    result = _deterministic_zip_bytes(entries)
    scan_member(relative, result)
    return result


def _validate_publication_title(value: str, *, label: str) -> None:
    normalized = " ".join(value.split()).casefold()
    if normalized in {"", "anonymous", "anonymous manuscript", "anonymous authors"}:
        return
    if normalized == "k-bound" or normalized.startswith("k-bound:"):
        return
    raise PrivacyError(f"non-publication title metadata in {label}")


def _validate_pdf_info(relative: str, metadata: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in fields:
            raise PrivacyError(f"duplicate PDF metadata field {key}: {relative}")
        if key not in _ALLOWED_PDFINFO_FIELDS:
            raise PrivacyError(f"unknown PDF metadata field {key}: {relative}")
        fields[key] = value.strip()

    author = fields.get("Author", "")
    if author.casefold() not in {"", "anonymous", "anonymous authors"}:
        raise PrivacyError(f"identity disclosure in PDF author metadata: {relative}")
    _validate_publication_title(fields.get("Title", ""), label=f"PDF {relative}")
    for key in ("Subject", "Keywords"):
        if fields.get(key, ""):
            raise PrivacyError(f"identity-bearing PDF {key} metadata must be empty: {relative}")
    creator = fields.get("Creator", "")
    if creator not in {"", "LaTeX with hyperref"}:
        raise PrivacyError(f"identity-bearing PDF Creator metadata is not an allowed tool: {relative}")
    producer = fields.get("Producer", "")
    if producer and re.fullmatch(r"pdfTeX-\d+(?:\.\d+)+", producer) is None:
        raise PrivacyError(f"identity-bearing PDF Producer metadata is not an allowed tool: {relative}")
    return fields


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") and "}" in tag else ""


def _validate_xmp(relative: str, payload: str) -> None:
    if not payload.strip():
        return
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise PrivacyError(f"malformed XMP metadata in {relative}: {exc}") from exc
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in root.iter():
        namespace = _namespace(element.tag)
        if namespace not in _ALLOWED_XMP_NAMESPACES:
            raise PrivacyError(f"unknown namespaced XMP metadata in {relative}")
        for attribute, value in element.attrib.items():
            local_attribute = _local_name(attribute)
            if value.strip() and not (local_attribute == "about" and value.strip() == "" or local_attribute == "lang"):
                raise PrivacyError(f"unapproved XMP attribute metadata in {relative}")
        text = (element.text or "").strip()
        if not text:
            continue
        local = _local_name(element.tag)
        if local in _FORBIDDEN_XMP_TEXT_FIELDS:
            raise PrivacyError(f"identity-bearing XMP {local} metadata in {relative}")
        ancestors: list[str] = []
        parent = parents.get(element)
        while parent is not None:
            ancestors.append(_local_name(parent.tag))
            parent = parents.get(parent)
        if local == "li" and "title" in ancestors:
            _validate_publication_title(text, label=f"XMP title for {relative}")
        elif local == "format" and text == "application/pdf":
            continue
        elif local in {"CreateDate", "ModifyDate", "MetadataDate"} and re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+", text):
            continue
        elif local == "PDFVersion" and re.fullmatch(r"\d+(?:\.\d+)+", text):
            continue
        else:
            raise PrivacyError(f"unapproved non-empty XMP metadata in {relative}")


def _validate_tex_anonymity(relative: str, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrivacyError(f"anonymous TeX source is not UTF-8: {relative}") from exc
    for match in _TEX_AUTHOR_RE.finditer(text):
        normalized = " ".join(match.group(1).replace("\\\\", " ").split()).casefold()
        if normalized not in {
            "anonymous",
            "anonymous authors",
            "anonymous authors paper under double-blind review",
        }:
            raise PrivacyError(f"identity-bearing TeX author metadata in {relative}")
    if any(match.group(1).strip() for match in _TEX_IDENTITY_MACRO_RE.finditer(text)):
        raise PrivacyError(f"identity-bearing TeX author metadata in {relative}")


def _verify_pinned_portable_file(
    path_text: str | None,
    *,
    expected_sha256: str,
    label: str,
    executable: bool,
) -> Path:
    if path_text is None:
        raise PrivacyError(f"{label} must name a hash-bound absolute path")
    unresolved = Path(path_text).expanduser()
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise PrivacyError(f"{label} is not a resident file") from exc
    if (
        not unresolved.is_absolute()
        or unresolved.is_symlink()
        or unresolved != resolved
        or not resolved.is_file()
        or (executable and not os.access(resolved, os.X_OK))
    ):
        raise PrivacyError(f"{label} must name an absolute non-symlink{' executable' if executable else ''} realpath")
    before = _path_identity(resolved)
    observed = _sha(resolved.read_bytes())
    if _path_identity(resolved) != before:
        raise PrivacyError(f"{label} changed while its digest was verified")
    if observed != expected_sha256:
        raise PrivacyError(f"{label} digest does not match the pinned portable Poppler bytes")
    return resolved


def verify_portable_pdf_toolchain() -> None:
    """Bind Linux portable PDF inspection to the reviewed Poppler package bytes."""

    for name, expected_sha256 in sorted(PORTABLE_PDF_TOOL_SHA256.items()):
        _verify_pinned_portable_file(
            os.environ.get(f"KBOUND_TOOL_{name.upper()}"),
            expected_sha256=expected_sha256,
            label=f"KBOUND_TOOL_{name.upper()}",
            executable=True,
        )
    _verify_pinned_portable_file(
        os.environ.get("KBOUND_PORTABLE_POPPLER_LIBRARY"),
        expected_sha256=PORTABLE_POPPLER_LIBRARY_SHA256,
        label="KBOUND_PORTABLE_POPPLER_LIBRARY",
        executable=False,
    )


def verify_pdf_anonymity(relative: str, payload: bytes) -> None:
    """Require Poppler metadata and extracted-text checks for every PDF."""

    if not payload.startswith(b"%PDF-"):
        raise PrivacyError(f"malformed PDF signature: {relative}")

    def required_tool(name: str) -> str:
        override_name = "KBOUND_TOOL_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
        override = os.environ.get(override_name)
        candidate = override if override else shutil.which(name)
        if candidate is None:
            raise PrivacyError(f"{name} is required for anonymous PDF verification")
        if not override:
            return candidate
        unresolved = Path(override).expanduser()
        try:
            resolved = unresolved.resolve(strict=True)
        except OSError as exc:
            raise PrivacyError(f"{override_name} is not a resident executable") from exc
        if (
            not unresolved.is_absolute()
            or unresolved.is_symlink()
            or unresolved != resolved
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
        ):
            raise PrivacyError(f"{override_name} must name an absolute non-symlink executable realpath")
        return str(resolved)

    pdfinfo = required_tool("pdfinfo")
    pdftotext = required_tool("pdftotext")
    pdfdetach = required_tool("pdfdetach")
    inspection_environment: dict[str, str] | None = None
    portable_library = os.environ.get("KBOUND_PORTABLE_POPPLER_LIBRARY")
    if portable_library is not None:
        # Revalidate immediately before each PDF and force the loader to the
        # exact reviewed libpoppler object instead of a mutable system copy.
        verify_portable_pdf_toolchain()
        inspection_environment = os.environ.copy()
        inspection_environment["LD_LIBRARY_PATH"] = str(Path(portable_library).parent)
    with tempfile.TemporaryDirectory(prefix="kbound-pdf-privacy-") as temporary:
        source = Path(temporary) / "document.pdf"
        source.write_bytes(payload)
        try:
            metadata = subprocess.run(
                [pdfinfo, str(source)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env=inspection_environment,
            ).stdout
            xmp_metadata = subprocess.run(
                [pdfinfo, "-meta", str(source)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env=inspection_environment,
            ).stdout
            extracted = subprocess.run(
                [pdftotext, str(source), "-"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env=inspection_environment,
            ).stdout
            attachments = subprocess.run(
                [pdfdetach, "-list", str(source)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                env=inspection_environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise PrivacyError(f"cannot inspect anonymous PDF {relative}: {exc}") from exc
    fields = _validate_pdf_info(relative, metadata)
    _validate_xmp(relative, xmp_metadata)
    if fields.get("Encrypted", "").lower() != "no":
        raise PrivacyError(f"encrypted PDF is not anonymously inspectable: {relative}")
    if fields.get("JavaScript", "").lower() != "no":
        raise PrivacyError(f"PDF contains JavaScript: {relative}")
    if fields.get("Form", "").lower() != "none":
        raise PrivacyError(f"PDF contains an interactive form: {relative}")
    if attachments.strip() != "0 embedded files":
        raise PrivacyError(f"PDF contains an embedded file: {relative}")
    scan_member(relative + ".pdfinfo.txt", metadata.encode("utf-8"))
    scan_member(relative + ".xmp.txt", xmp_metadata.encode("utf-8"))
    scan_member(relative + ".pdftotext.txt", extracted.encode("utf-8"))


def _tex_reference(root: Path, reference: str, *, suffix: str) -> str | None:
    latex_root = root / "docs/research/kbound"
    candidate = latex_root / reference
    if not Path(reference).suffix:
        candidate = candidate.with_suffix(suffix)
    if not candidate.is_file():
        return None
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise PrivacyError(f"TMLR dependency escapes the repository: {reference}") from exc


def _tex_graphic_reference(root: Path, reference: str) -> str | None:
    if Path(reference).suffix:
        return _tex_reference(root, reference, suffix="")
    matches = [
        match for suffix in (".pdf", ".png") if (match := _tex_reference(root, reference, suffix=suffix)) is not None
    ]
    if len(matches) > 1:
        raise PrivacyError(f"live TMLR graphic reference is ambiguous: {reference}")
    return matches[0] if matches else None


def validate_anonymous_inventory(paths: list[str] | tuple[str, ...]) -> list[str]:
    canonical: list[str] = []
    for relative in paths:
        value = validate_public_member_path(relative)
        if value in _FORBIDDEN_NAMED_INPUTS:
            raise PrivacyError(f"forbidden named-release input in anonymous package: {value}")
        canonical.append(value)
    if len(canonical) != len(set(canonical)):
        raise PrivacyError("anonymous package input inventory contains duplicates")
    return sorted(canonical)


def _live_tex(text: str) -> str:
    """Remove TeX comments while preserving escaped percent characters."""

    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        cutoff = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cutoff = index
                break
        suffix = "\n" if line.endswith("\n") and cutoff < len(line) else ""
        lines.append(line[:cutoff].rstrip("\r\n") + suffix)
    return "".join(lines)


def anonymous_tmlr_inventory(root: Path) -> list[str]:
    """Derive the exact live TeX, generated-table, style, bibliography and figure closure."""

    root = root.resolve()
    pending = [_TMLR_DRIVER]
    selected = {_TMLR_PDF, "docs/research/kbound/paper/vendor/tmlr/LICENSE"}
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        source = _safe_input(root, relative)
        text = _live_tex(source.read_text(encoding="utf-8"))
        selected.add(relative)
        for reference in _TEX_INPUT_RE.findall(text):
            dependency = _tex_reference(root, reference, suffix=".tex")
            if dependency is None:
                raise PrivacyError(f"live TMLR input is missing: {reference}")
            pending.append(dependency)
        for reference in _TEX_PACKAGE_RE.findall(text):
            dependency = _tex_reference(root, reference, suffix=".sty")
            if dependency is not None:
                selected.add(dependency)
        for reference in _TEX_GRAPHIC_RE.findall(text):
            if _LITERAL_TEX_PATH_RE.fullmatch(reference) is None:
                continue
            dependency = _tex_graphic_reference(root, reference)
            if dependency is None:
                raise PrivacyError(f"live TMLR graphic is missing: {reference}")
            selected.add(dependency)
    # The maintained expanded references are inline TeX, but retain the source
    # bibliography requested by the release checklist as review provenance.
    selected.add("docs/research/kbound/paper/references/refs.bib")
    return validate_anonymous_inventory(tuple(selected))


def _replace_docx_document(payload: bytes, document_xml: str) -> bytes:
    """Test-fixture convenience: replace the document member deterministically."""

    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        entries = {
            item.filename: source.read(item)
            for item in source.infolist()
            if not item.is_dir() and item.filename != "word/document.xml"
        }
    entries["word/document.xml"] = document_xml.encode("utf-8")
    return _deterministic_zip_bytes(entries)


def prepare_anonymous_payload(relative: str, payload: bytes) -> tuple[bytes, str]:
    """Classify, scrub where defined, and scan one requested package member."""

    # Check the public name independently before type classification. This
    # guarantees VCS/editor metadata is reported as forbidden, never disguised
    # as an unsupported extension or silently skipped.
    scan_member(relative, b"")
    suffix = PurePosixPath(relative).suffix.lower()
    is_tmlr_license = relative == "docs/research/kbound/paper/vendor/tmlr/LICENSE"
    if suffix not in _ALLOWED_SUFFIXES and not is_tmlr_license:
        raise PrivacyError(f"unclassified supplementary input type: {relative}")
    if suffix == ".docx":
        return sanitize_ooxml(relative, payload), "deterministic-ooxml-metadata-scrub"
    if suffix == ".pdf":
        verify_pdf_anonymity(relative, payload)
        return payload, "verified-pdf-byte-identical-copy"
    if suffix == ".png":
        published = sanitize_png_metadata(payload)
        scan_member(relative, published)
        return published, "validated-png-metadata-scrub"
    if suffix == ".tex":
        _validate_tex_anonymity(relative, payload)
    scan_member(relative, payload)
    return payload, "byte-identical-copy"


def _safe_input(root: Path, relative: str) -> Path:
    validate_public_member_path(relative)
    parts = PurePosixPath(relative).parts
    candidate = root.joinpath(*parts)
    if any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise PrivacyError(f"supplementary input traverses a symlink: {relative}")
    if not candidate.is_file():
        raise PrivacyError(f"supplementary input is missing: {relative}")
    return candidate


def _require_regular_unlinked(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    if any(Path(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise PrivacyError(f"{label} traverses a symlink: {path}")
    if not absolute.is_file():
        raise PrivacyError(f"{label} is missing: {path}")


def _path_identity(path: Path) -> tuple[int, int, int, int]:
    state = path.stat(follow_symlinks=False)
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)


def _has_identity(path: Path, expected: tuple[int, int, int, int]) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and _path_identity(path) == expected
    except OSError:
        return False


def _stream_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_verified_candidate(
    candidate: Path,
    output: Path,
    *,
    prior_identity: tuple[int, int, int, int] | None,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    """Publish a verified archive without a candidate or destination race."""

    candidate_identity = _path_identity(candidate)
    candidate_sha256 = _stream_file_sha256(candidate)
    if _path_identity(candidate) != candidate_identity or (candidate_identity[2], candidate_sha256) != (
        expected_bytes,
        expected_sha256,
    ):
        raise PrivacyError("anonymous archive candidate changed after it was verified")
    displaced: Path | None = None
    displaced_is_verified = False
    published = False
    try:
        if prior_identity is not None:
            descriptor, displaced_text = tempfile.mkstemp(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".verified-existing",
            )
            os.close(descriptor)
            displaced = Path(displaced_text)
            displaced.unlink()
            os.replace(output, displaced)
            if not _has_identity(displaced, prior_identity):
                if not output.exists() and not output.is_symlink():
                    os.replace(displaced, output)
                    displaced = None
                raise PrivacyError("anonymous supplement changed before verified replacement")
            displaced_is_verified = True

        try:
            os.link(candidate, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise PrivacyError("anonymous supplement output appeared during atomic publication") from exc
        if (
            not _has_identity(output, candidate_identity)
            or _stream_file_sha256(output) != expected_sha256
            or not _has_identity(output, candidate_identity)
        ):
            raise PrivacyError("published anonymous archive does not retain the verified candidate identity")
        published = True
        if displaced is not None:
            displaced.unlink()
            displaced = None
    except OSError as exc:
        raise PrivacyError(f"cannot atomically publish verified anonymous archive: {exc}") from exc
    finally:
        if not published and _has_identity(output, candidate_identity):
            output.unlink()
        if not published and displaced_is_verified and displaced is not None:
            if not output.exists() and not output.is_symlink():
                try:
                    os.link(displaced, output, follow_symlinks=False)
                except OSError:
                    pass
            if prior_identity is not None and _has_identity(output, prior_identity):
                displaced.unlink(missing_ok=True)


def build_anonymous_supplement(
    *,
    root: Path,
    include_paths: list[str] | tuple[str, ...],
    source_seal_path: Path,
    checksum_path: Path,
    public_bundle_path: Path,
    output_path: Path,
    require_git_checkout: bool = True,
    replace_verified: bool = False,
) -> dict[str, Any]:
    """Build only from an explicit, completely classified input inventory."""

    output_exists = output_path.exists() or output_path.is_symlink()
    if output_exists and not replace_verified:
        raise PrivacyError(f"anonymous supplement output already exists: {output_path}")
    if output_path.is_symlink():
        raise PrivacyError(f"refusing to replace a symlinked anonymous supplement: {output_path}")
    prior_receipt: dict[str, Any] | None = None
    prior_identity: tuple[int, int, int, int] | None = None
    if output_path.exists():
        prior_identity = _path_identity(output_path)
        prior_receipt = verify_anonymous_supplement(output_path)
        if _path_identity(output_path) != prior_identity:
            raise PrivacyError("anonymous supplement changed while existing output was verified")

    root = root.resolve()
    inventory = validate_anonymous_inventory(include_paths)
    for relative in inventory:
        # Reject unsafe public names before interpreting repository-control
        # directories (for example an attempted .git/config input) as state.
        scan_member(relative, b"")
    _require_regular_unlinked(source_seal_path, label="release source seal")
    _require_regular_unlinked(checksum_path, label="release checksum file")
    _require_regular_unlinked(public_bundle_path, label="CCT-20 public bundle")
    seal_source = source_seal_path.read_bytes()
    seal = strict_json_loads(seal_source)
    try:
        seal = source_seal.validate_seal_document(seal)
    except (PrivacyError, ValueError) as exc:
        raise PrivacyError(f"invalid release source seal: {exc}") from exc
    if seal_source != source_seal._seal_bytes(seal):
        raise PrivacyError("release source seal is not canonical JSON")
    git_marker = root / ".git"
    if git_marker.exists() or git_marker.is_symlink():
        try:
            source_seal.validate_seal(root, source_seal_path)
        except (OSError, PrivacyError, subprocess.SubprocessError, ValueError) as exc:
            raise PrivacyError(f"release source seal does not validate against repository: {exc}") from exc
    elif require_git_checkout:
        raise PrivacyError("anonymous release must be built from a validated Git checkout")
    commit = seal.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise PrivacyError("release source seal lacks a full immutable source commit")
    checksum_payload = checksum_path.read_bytes()
    checksum_entries = _parse_checksum_payload(checksum_payload)
    try:
        verify_checksum_file(
            checksum_path,
            root=root,
            required_paths=REQUIRED_RELEASE_PATHS,
        )
    except (OSError, ValueError) as exc:
        raise PrivacyError(f"release checksum verification failed: {exc}") from exc
    sealed_sources = _source_authorities(seal)
    seal_relative = _relative_control_path(root, source_seal_path, label="release source seal")
    public_relative = _relative_control_path(root, public_bundle_path, label="CCT-20 public bundle")
    if checksum_entries.get(seal_relative) != _sha(seal_source):
        raise PrivacyError("release source seal is not bound by the checksum receipt")
    verify_public_bundle(public_bundle_path)
    public_payload = public_bundle_path.read_bytes()
    if checksum_entries.get(public_relative) != _sha(public_payload):
        raise PrivacyError("CCT-20 public bundle is not bound by the checksum receipt")

    entries: dict[str, bytes] = {}
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative in inventory:
        canonical = validate_public_member_path(relative)
        if canonical in seen:
            raise PrivacyError(f"duplicate supplementary input: {canonical}")
        seen.add(canonical)
        source = _safe_input(root, canonical)
        source_payload = source.read_bytes()
        source_digest = _sha(source_payload)
        _require_source_binding(
            relative=canonical,
            size=len(source_payload),
            digest=source_digest,
            checksums=checksum_entries,
            sealed=sealed_sources,
        )
        published, normalization = prepare_anonymous_payload(canonical, source_payload)
        published_path = "artifacts/" + canonical
        entries[published_path] = published
        artifacts.append(
            {
                "normalization": normalization,
                "published_bytes": len(published),
                "published_path": published_path,
                "published_sha256": _sha(published),
                "source_bytes": len(source_payload),
                "source_path": canonical,
                "source_sha256": source_digest,
            }
        )
    if not artifacts:
        raise PrivacyError("anonymous supplementary input inventory is empty")

    canonical_seal = source_seal._seal_bytes(seal)
    entries.update(
        {
            "receipts/KBOUND_RELEASE_SHA256SUMS.txt": checksum_payload,
            "receipts/release_source_seal.json": canonical_seal,
        }
    )
    commitments = {
        "cct20_public_evidence_bundle": {
            "bytes": len(public_payload),
            "sha256": _sha(public_payload),
            "source_path": public_relative,
        },
        "release_checksums": {"bytes": len(checksum_payload), "sha256": _sha(checksum_payload)},
        "release_source_seal": {
            "published_bytes": len(canonical_seal),
            "published_sha256": _sha(canonical_seal),
            "source_bytes": len(seal_source),
            "source_commit": commit,
            "source_path": seal_relative,
            "source_sha256": _sha(seal_source),
        },
    }
    manifest = {
        "artifacts": sorted(artifacts, key=lambda row: row["published_path"]),
        "commitments": commitments,
        "schema": SCHEMA,
    }
    entries["manifest.json"] = _canonical_json(manifest)
    verified_pdf_paths = {
        str(row["published_path"]) for row in artifacts if row["normalization"] == "verified-pdf-byte-identical-copy"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, candidate_text = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".candidate",
    )
    os.close(descriptor)
    candidate = Path(candidate_text)
    candidate.unlink()
    try:
        build_deterministic_zip(
            candidate,
            entries,
            verified_pdf_members=verified_pdf_paths,
        )
        receipt = verify_anonymous_supplement(candidate)
        if prior_receipt is not None:
            if prior_identity is None or _path_identity(output_path) != prior_identity:
                raise PrivacyError("anonymous supplement changed before verified replacement")
            if receipt["archive_sha256"] == prior_receipt["archive_sha256"]:
                return prior_receipt
        elif output_path.exists() or output_path.is_symlink():
            raise PrivacyError("anonymous supplement output appeared during publication")
        _publish_verified_candidate(
            candidate,
            output_path,
            prior_identity=prior_identity,
            expected_bytes=int(receipt["archive_bytes"]),
            expected_sha256=str(receipt["archive_sha256"]),
        )
        return receipt
    finally:
        candidate.unlink(missing_ok=True)


def _declared_verified_pdf_members(path: Path) -> frozenset[str]:
    """Read only the bounded manifest needed to select type-verified PDFs."""

    if path.is_symlink() or not path.is_file():
        raise PrivacyError(f"anonymous archive is missing or a symlink: {path}")
    if path.stat().st_size > MAX_TOTAL_BYTES:
        raise PrivacyError("anonymous archive exceeds the compressed-size safety limit")
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("manifest.json")
            if info.file_size > _MAX_MANIFEST_BYTES:
                raise PrivacyError("anonymous-supplement manifest exceeds the safety limit")
            with archive.open(info) as handle:
                manifest_payload = handle.read(_MAX_MANIFEST_BYTES + 1)
    except (KeyError, NotImplementedError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PrivacyError(f"cannot read anonymous-supplement manifest safely: {exc}") from exc
    if len(manifest_payload) > _MAX_MANIFEST_BYTES:
        raise PrivacyError("anonymous-supplement manifest exceeds the safety limit")
    manifest = strict_json_loads(manifest_payload)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise PrivacyError("unsupported anonymous-supplement schema")
    if set(manifest) != {"artifacts", "commitments", "schema"}:
        raise PrivacyError("anonymous-supplement manifest schema is malformed")
    verified: set[str] = set()
    for row in _validate_artifact_ledger(manifest.get("artifacts")):
        if row["normalization"] != "verified-pdf-byte-identical-copy":
            continue
        published = str(row["published_path"])
        source = str(row["source_path"])
        if PurePosixPath(published).suffix.lower() != ".pdf" or PurePosixPath(source).suffix.lower() != ".pdf":
            raise PrivacyError("verified-PDF normalization is bound to a non-PDF artifact")
        verified.add(published)
    return frozenset(verified)


def verify_anonymous_supplement(path: Path) -> dict[str, Any]:
    verified_pdf_paths = _declared_verified_pdf_members(path)
    raw_receipt = verify_anonymous_zip(path, verified_pdf_members=verified_pdf_paths)
    if not isinstance(raw_receipt, dict):
        raise PrivacyError("anonymous archive verifier returned a malformed receipt")
    receipt = cast(dict[str, Any], raw_receipt)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "manifest.json" not in names:
            raise PrivacyError("anonymous supplement has no manifest")
        manifest_payload = archive.read("manifest.json")
        manifest = strict_json_loads(manifest_payload)
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise PrivacyError("unsupported anonymous-supplement schema")
        if set(manifest) != {"artifacts", "commitments", "schema"}:
            raise PrivacyError("anonymous-supplement manifest schema is malformed")
        if manifest_payload != _canonical_json(manifest):
            raise PrivacyError("anonymous-supplement manifest is not canonical JSON")
        expected = {
            "manifest.json",
            "receipts/KBOUND_RELEASE_SHA256SUMS.txt",
            "receipts/release_source_seal.json",
        }
        embedded_checksum_payload = archive.read("receipts/KBOUND_RELEASE_SHA256SUMS.txt")
        checksum_entries = _parse_checksum_payload(embedded_checksum_payload)
        embedded_seal_payload = archive.read("receipts/release_source_seal.json")
        embedded_seal = strict_json_loads(embedded_seal_payload)
        try:
            embedded_seal = source_seal.validate_seal_document(embedded_seal)
        except (PrivacyError, ValueError) as exc:
            raise PrivacyError(f"anonymous package contains an invalid source seal: {exc}") from exc
        if embedded_seal_payload != source_seal._seal_bytes(embedded_seal):
            raise PrivacyError("anonymous package source seal is not canonical JSON")
        sealed_sources = _source_authorities(embedded_seal)
        for row in _validate_artifact_ledger(manifest.get("artifacts")):
            relative = validate_public_member_path(str(row.get("published_path", "")))
            payload = archive.read(relative) if relative in names else None
            if (
                payload is None
                or len(payload) != row.get("published_bytes")
                or _sha(payload) != row.get("published_sha256")
            ):
                raise PrivacyError(f"anonymous artifact commitment mismatch: {relative}")
            source_relative = str(row["source_path"])
            validate_anonymous_inventory([source_relative])
            _require_source_binding(
                relative=source_relative,
                size=int(row["source_bytes"]),
                digest=str(row["source_sha256"]),
                checksums=checksum_entries,
                sealed=sealed_sources,
            )
            if PurePosixPath(source_relative).suffix.lower() == ".pdf":
                verify_pdf_anonymity(source_relative, payload)
            expected.add(relative)
        commitments = manifest.get("commitments")
        if not isinstance(commitments, dict) or set(commitments) != {
            "cct20_public_evidence_bundle",
            "release_checksums",
            "release_source_seal",
        }:
            raise PrivacyError("anonymous package commitments are malformed")
        bound_paths = {
            "release_checksums": "receipts/KBOUND_RELEASE_SHA256SUMS.txt",
            "release_source_seal": "receipts/release_source_seal.json",
        }
        for label, relative in bound_paths.items():
            record = commitments.get(label)
            payload = archive.read(relative)
            if not isinstance(record, dict):
                raise PrivacyError(f"missing anonymous package commitment: {label}")
            expected_fields = (
                {"bytes", "sha256"}
                if label == "release_checksums"
                else {
                    "published_bytes",
                    "published_sha256",
                    "source_bytes",
                    "source_commit",
                    "source_path",
                    "source_sha256",
                }
            )
            if set(record) != expected_fields:
                raise PrivacyError(f"malformed anonymous package commitment: {label}")
            wanted_bytes = record.get("published_bytes", record.get("bytes"))
            wanted_sha = record.get("published_sha256", record.get("sha256"))
            if len(payload) != wanted_bytes or _sha(payload) != wanted_sha:
                raise PrivacyError(f"anonymous package commitment mismatch: {label}")
        public_record = commitments.get("cct20_public_evidence_bundle")
        if not isinstance(public_record, dict):
            raise PrivacyError("missing CCT-20 public-bundle commitment")
        public_bytes = public_record.get("bytes")
        public_sha = public_record.get("sha256")
        public_source = public_record.get("source_path")
        if (
            set(public_record) != {"bytes", "sha256", "source_path"}
            or isinstance(public_bytes, bool)
            or not isinstance(public_bytes, int)
            or public_bytes < 1
            or not isinstance(public_sha, str)
            or len(public_sha) != 64
            or any(character not in "0123456789abcdef" for character in public_sha)
        ):
            raise PrivacyError("malformed CCT-20 public-bundle commitment")
        if not isinstance(public_source, str) or checksum_entries.get(public_source) != public_sha:
            raise PrivacyError("CCT-20 public-bundle checksum authority mismatch")
        seal_record = commitments["release_source_seal"]
        if (
            set(seal_record)
            != {
                "published_bytes",
                "published_sha256",
                "source_bytes",
                "source_commit",
                "source_path",
                "source_sha256",
            }
            or seal_record.get("source_commit") != embedded_seal.get("source_commit")
            or seal_record.get("source_bytes") != len(embedded_seal_payload)
            or seal_record.get("source_sha256") != _sha(embedded_seal_payload)
        ):
            raise PrivacyError("anonymous package source-seal provenance commitment mismatch")
        seal_source = seal_record.get("source_path")
        if not isinstance(seal_source, str) or checksum_entries.get(seal_source) != _sha(embedded_seal_payload):
            raise PrivacyError("anonymous package source seal checksum authority mismatch")
        if set(names) != expected:
            raise PrivacyError("anonymous supplement contains an unmanifested or missing member")
    receipt["schema"] = "kbound-anonymous-supplement-verification-v2"
    return receipt


def _stable_control_bytes(path: Path, *, label: str) -> bytes:
    _require_regular_unlinked(path, label=label)
    before = _path_identity(path)
    payload = path.read_bytes()
    if _path_identity(path) != before:
        raise PrivacyError(f"{label} changed while being read")
    return payload


def verify_anonymous_release_bindings(
    path: Path,
    *,
    source_seal_path: Path,
    checksum_path: Path,
    public_bundle_path: Path,
) -> dict[str, Any]:
    """Cross-bind a verified archive to the external portable release controls."""

    _require_regular_unlinked(path, label="anonymous archive")
    archive_identity = _path_identity(path)
    receipt = verify_anonymous_supplement(path)
    if path.is_symlink() or not path.is_file() or _path_identity(path) != archive_identity:
        raise PrivacyError("anonymous archive changed after it was semantically verified")
    checksum_payload = _stable_control_bytes(checksum_path, label="external release checksum")
    source_seal_payload = _stable_control_bytes(source_seal_path, label="external release source seal")
    public_receipt = verify_public_bundle(public_bundle_path)
    public_payload = _stable_control_bytes(public_bundle_path, label="external CCT-20 public bundle")
    try:
        with path.open("rb") as archive_handle, tempfile.TemporaryFile() as stable_archive:
            descriptor_state = os.fstat(archive_handle.fileno())
            descriptor_identity = (
                descriptor_state.st_dev,
                descriptor_state.st_ino,
                descriptor_state.st_size,
                descriptor_state.st_mtime_ns,
            )
            if descriptor_identity != archive_identity:
                raise PrivacyError("anonymous archive changed before external bindings were read")
            digest = hashlib.sha256()
            copied_bytes = 0
            for chunk in iter(lambda: archive_handle.read(1024 * 1024), b""):
                copied_bytes += len(chunk)
                if copied_bytes > MAX_TOTAL_BYTES:
                    raise PrivacyError("anonymous archive changed beyond the verified size limit")
                digest.update(chunk)
                stable_archive.write(chunk)
            if copied_bytes != receipt.get("archive_bytes") or digest.hexdigest() != receipt.get("archive_sha256"):
                raise PrivacyError("anonymous archive bytes changed after semantic verification")
            stable_archive.seek(0)
            with zipfile.ZipFile(stable_archive) as archive:
                if archive.read("receipts/KBOUND_RELEASE_SHA256SUMS.txt") != checksum_payload:
                    raise PrivacyError("anonymous package does not embed the external release checksum bytes")
                if archive.read("receipts/release_source_seal.json") != source_seal_payload:
                    raise PrivacyError("anonymous package does not embed the external release source seal bytes")
                manifest = strict_json_loads(archive.read("manifest.json"))
            final_descriptor_state = os.fstat(archive_handle.fileno())
            final_descriptor_identity = (
                final_descriptor_state.st_dev,
                final_descriptor_state.st_ino,
                final_descriptor_state.st_size,
                final_descriptor_state.st_mtime_ns,
            )
            if final_descriptor_identity != archive_identity:
                raise PrivacyError("anonymous archive changed while external bindings were read")
    except (KeyError, NotImplementedError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PrivacyError(f"cannot read verified anonymous archive bindings: {exc}") from exc
    if path.is_symlink() or not path.is_file() or _path_identity(path) != archive_identity:
        raise PrivacyError("anonymous archive changed while external bindings were verified")
    if not isinstance(manifest, dict):  # pragma: no cover - enforced by the archive verifier
        raise PrivacyError("anonymous package manifest is malformed")
    commitments = manifest.get("commitments")
    public = commitments.get("cct20_public_evidence_bundle") if isinstance(commitments, dict) else None
    if (
        not isinstance(public, dict)
        or public.get("bytes") != len(public_payload)
        or public.get("sha256") != _sha(public_payload)
        or public_receipt.get("archive_sha256") != public.get("sha256")
    ):
        raise PrivacyError("anonymous package is not bound to the external CCT-20 public bundle")
    return receipt


def verify_release_python_content() -> None:
    """Require the sealed release Python content before archive processing."""

    root = Path(__file__).resolve().parents[4]
    verify_python_environment.verify_exact_content_profile(
        root / "requirements-release-macos-arm64.lock.txt",
        root / "docs/research/kbound/release_python_environment_macos_arm64.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--source-seal", type=Path, default=DEFAULT_SOURCE_SEAL)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--public-bundle", type=Path, default=DEFAULT_PUBLIC_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true")
    checks.add_argument(
        "--portable-check",
        action="store_true",
        help="verify archive semantics without enforcing the source build runtime",
    )
    parser.add_argument("--replace-verified", action="store_true")
    args = parser.parse_args()
    if not args.portable_check:
        verify_release_python_content()
    if args.portable_check:
        verify_portable_pdf_toolchain()
        receipt = verify_anonymous_release_bindings(
            args.output,
            source_seal_path=args.source_seal,
            checksum_path=args.checksums,
            public_bundle_path=args.public_bundle,
        )
    elif args.check:
        receipt = verify_anonymous_supplement(args.output)
    else:
        receipt = build_anonymous_supplement(
            root=args.root,
            include_paths=args.include or anonymous_tmlr_inventory(args.root),
            source_seal_path=args.source_seal,
            checksum_path=args.checksums,
            public_bundle_path=args.public_bundle,
            output_path=args.output,
            replace_verified=args.replace_verified,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
