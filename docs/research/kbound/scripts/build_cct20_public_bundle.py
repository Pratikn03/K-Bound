#!/usr/bin/env python3
"""Build and independently verify a portable CCT-20 evidence archive.

The canonical release manifest retains the exact hashes of the machine on which
the experiment was sealed.  This builder never rewrites those commitments.  It
adds portable, content-addressed copies for redistributable objects and explicit
hash-only descriptors for raw archives and model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from docs.research.kbound.scripts import verify_python_environment
from docs.research.kbound.scripts.release_privacy import (
    FIXED_ZIP_TIME,
    MAX_MEMBER_BYTES,
    MAX_TOTAL_BYTES,
    PrivacyError,
    file_sha256,
    scan_member,
    strict_json_loads,
    validate_public_member_path,
)

SCHEMA = "kbound-cct20-public-evidence-bundle-v2"
VERIFICATION_SCHEMA = "kbound-cct20-public-evidence-verification-v2"
DESCRIPTOR_MODE_VERIFY = "verify-sources"
DESCRIPTOR_MODE_OFFLINE = "offline-commitments"
DESCRIPTOR_MODES = frozenset({DESCRIPTOR_MODE_VERIFY, DESCRIPTOR_MODE_OFFLINE})
MAX_PUBLIC_UNCOMPRESSED_BYTES = MAX_TOTAL_BYTES
MAX_PUBLIC_MEMBERS = 100_000
MAX_SOURCE_MANIFEST_BYTES = 64 * 1024 * 1024
DEFAULT_RELEASE = Path("docs/research/kbound/paper/generated/cct20_release_manifest.json")
DEFAULT_OUTPUT = Path("docs/research/kbound/release/cct20_public_evidence_bundle.zip")
README_BYTES = (
    b"# CCT-20 public evidence bundle\n\n"
    b"Objects are content-addressed. Source SHA-256 commitments are preserved; "
    b"normalized portable copies are explicitly labeled in manifest.json. Raw "
    b"archives and checkpoints are represented only by hash-bound descriptors.\n"
)
PORTABLE_NORMALIZATION = "canonical JSON; private locations replaced by portable references or descriptors"
OBJECT_NORMALIZATIONS = frozenset(
    {
        "byte-identical-copy",
        "canonical-json-with-portable-path-descriptors",
        "canonical-json-from-yaml-with-portable-path-descriptors",
    }
)
_NONREDISTRIBUTABLE_SUFFIXES = (
    ".ckpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".tgz",
)
_REDISTRIBUTABLE_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sha256",
        ".tex",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_FILE_LOCATION_RE = re.compile(r"\bfile:(?:[/\\]+|[A-Za-z]:[/\\])[^\s\x00<>\"']*", re.IGNORECASE)
_HOME_LOCATION_RE = re.compile(r"(?:\$HOME|\$\{HOME\}|%USERPROFILE%)[/\\][^\s\x00<>\"']*", re.IGNORECASE)
_WINDOWS_LOCATION_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\x00<>\"']+|\\\\[^\\\s]+\\[^\\\s]+)")
_ABSOLUTE_LOCATION_RE = re.compile(r"(?<![:/A-Za-z0-9])/(?!/)[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)+")
_BINARY_SIGNATURES = (
    b"7z\xbc\xaf\x27\x1c",
    b"BZh",
    b"GIF87a",
    b"GIF89a",
    b"\x1f\x8b",
    b"\x7fELF",
    b"\x89HDF\r\n\x1a\n",
    b"\x89PNG\r\n\x1a\n",
    b"\x93NUMPY",
    b"\xff\xd8\xff",
    b"PAR1",
    b"PK\x03\x04",
    b"Rar!\x1a\x07",
    b"SQLite format 3\x00",
    b"\xfd7zXZ\x00",
    b"%PDF-",
)
_KNOWN_SYSTEM_ALIASES = {
    Path("/var"): Path("/private/var"),
    Path("/tmp"): Path("/private/tmp"),
    Path("/etc"): Path("/private/etc"),
}


def verify_release_python_content() -> None:
    """Require portable locked-distribution content before bundle semantics."""

    root = Path(__file__).resolve().parents[4]
    verify_python_environment.verify_exact_content_profile(
        root / "requirements-release-macos-arm64.lock.txt",
        root / "docs/research/kbound/release_python_environment_macos_arm64_v2.json",
    )


def _zip_info(relative: str, size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits |= 0x800
    info.file_size = size
    return info


def _verify_public_zip_container(path: Path) -> list[zipfile.ZipInfo]:
    """Verify ZIP safety/canonical metadata while retaining one member at a time."""

    if path.is_symlink() or not path.is_file():
        raise PrivacyError(f"public bundle is missing or a symlink: {path}")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PrivacyError(f"public bundle is malformed: {exc}") from exc
    with archive:
        if archive.comment:
            raise PrivacyError("noncanonical public-bundle ZIP comment metadata")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if not names or names != sorted(names) or len(names) != len(set(names)):
            raise PrivacyError("public-bundle members are empty, duplicated, or non-canonical")
        if len(infos) > MAX_PUBLIC_MEMBERS:
            raise PrivacyError("public bundle exceeds the member-count safety limit")
        total = 0
        for info in infos:
            relative = validate_public_member_path(info.filename)
            if info.comment or info.extra:
                raise PrivacyError(f"noncanonical ZIP member metadata: {relative}")
            if info.date_time != FIXED_ZIP_TIME:
                raise PrivacyError(f"nondeterministic ZIP timestamp: {relative}")
            if info.compress_type != zipfile.ZIP_DEFLATED:
                raise PrivacyError(f"noncanonical ZIP compression: {relative}")
            mode = info.external_attr >> 16
            if info.create_system != 3 or mode != (stat.S_IFREG | 0o644):
                raise PrivacyError(f"noncanonical ZIP mode: {relative}")
            if info.flag_bits & 0x1:
                raise PrivacyError(f"encrypted ZIP member: {relative}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise PrivacyError(f"oversized public-bundle member: {relative}")
            total += info.file_size
            if total > MAX_PUBLIC_UNCOMPRESSED_BYTES:
                raise PrivacyError("public bundle exceeds the aggregate uncompressed safety limit")
            try:
                payload = archive.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PrivacyError(f"cannot read public-bundle member: {relative}") from exc
            scan_member(relative, payload)
        return infos


def _output_parent_is_safe(parent: Path) -> bool:
    for link in _symlink_components(parent.absolute()):
        if not _is_known_system_alias(link):
            return False
    return True


def _build_public_zip(path: Path, entries: dict[str, Path]) -> None:
    """Stream validated staged files into an atomically published canonical ZIP."""

    if path.exists() or path.is_symlink():
        raise PrivacyError(f"public bundle output already exists: {path}")
    if len(entries) > MAX_PUBLIC_MEMBERS:
        raise PrivacyError("public bundle exceeds the member-count safety limit")
    normalized: dict[str, tuple[Path, int]] = {}
    total = 0
    for relative, staged in entries.items():
        canonical = validate_public_member_path(relative)
        if canonical in normalized:
            raise PrivacyError(f"duplicate public-bundle member: {canonical}")
        if staged.is_symlink() or not staged.is_file():
            raise PrivacyError(f"staged public-bundle member is missing: {canonical}")
        size = staged.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise PrivacyError(f"oversized public-bundle member: {canonical}")
        total += size
        if total > MAX_PUBLIC_UNCOMPRESSED_BYTES:
            raise PrivacyError("public bundle exceeds the aggregate uncompressed safety limit")
        payload = staged.read_bytes()
        scan_member(canonical, payload)
        normalized[canonical] = (staged, size)
    if not normalized:
        raise PrivacyError("public bundle would be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    if not _output_parent_is_safe(path.parent):
        raise PrivacyError(f"public-bundle output parent contains an unsafe symlink: {path.parent}")
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_text)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for relative in sorted(normalized):
                staged, size = normalized[relative]
                info = _zip_info(relative, size)
                with (
                    staged.open("rb") as source,
                    archive.open(
                        info,
                        mode="w",
                        force_zip64=size >= zipfile.ZIP64_LIMIT,
                    ) as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        _verify_public_zip_container(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PrivacyError(f"evidence document is not canonical JSON data: {exc}") from exc
    return encoded.encode("utf-8") + b"\n"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.nodes.ScalarNode):
            raise PrivacyError("YAML evidence mapping keys must be scalar")
        # JSON object keys are strings. Preserve the scalar spelling so numeric
        # camera/checkpoint IDs remain unambiguous in the portable form.
        key = key_node.value
        if key in result:
            raise PrivacyError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _validate_json_tree(value: Any, *, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PrivacyError("non-finite YAML number")
        return
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in seen:
            raise PrivacyError("recursive YAML aliases are not publishable")
        seen.add(identity)
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise PrivacyError("YAML evidence mapping keys must be strings")
            for item in value.values():
                _validate_json_tree(item, seen=seen)
        else:
            for item in value:
                _validate_json_tree(item, seen=seen)
        seen.remove(identity)
        return
    raise PrivacyError(f"unsupported YAML evidence value: {type(value).__name__}")


def _strict_yaml_loads(payload: bytes) -> Any:
    try:
        document = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except PrivacyError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PrivacyError(f"invalid YAML evidence: {exc}") from exc
    _validate_json_tree(document)
    return document


def _is_private_location(value: str) -> bool:
    lowered = value.lower()
    return (
        value.startswith(("/", "~", "\\\\"))
        or lowered.startswith("file:")
        or value.startswith(("$HOME/", "${HOME}/"))
        or lowered.startswith("%userprofile%/")
        or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
        or ".." in Path(value.replace("\\", "/")).parts
    )


def _is_nonredistributable(role: str, source: str) -> bool:
    label = role.lower()
    lowered = source.lower()
    return (
        "raw_archive" in label
        or "raw.archive" in label
        or lowered.endswith(_NONREDISTRIBUTABLE_SUFFIXES)
        or Path(source).suffix.lower() not in _REDISTRIBUTABLE_SUFFIXES
    )


def _redact_machine_locations(value: str) -> str:
    redacted = _FILE_LOCATION_RE.sub("[private-source-location-redacted]", value)
    redacted = _HOME_LOCATION_RE.sub("[private-source-location-redacted]", redacted)
    redacted = _WINDOWS_LOCATION_RE.sub("[private-source-location-redacted]", redacted)
    return _ABSOLUTE_LOCATION_RE.sub("[private-source-location-redacted]", redacted)


def _identity_field(record: dict[str, Any], path_key: str, suffix: str) -> Any:
    stem = "" if path_key == "path" else path_key[: -len("path")]
    candidates = [stem + suffix]
    if not stem:
        candidates.extend((f"file_{suffix}", f"artifact_{suffix}"))
    for key in candidates:
        if key in record:
            return record[key]
    return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _symlink_components(path: Path) -> list[Path]:
    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    links: list[Path] = []
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            links.append(current)
    return links


def _is_known_system_alias(link: Path) -> bool:
    expected = _KNOWN_SYSTEM_ALIASES.get(link)
    return expected is not None and link.resolve(strict=False) == expected


def _resolve_trusted_source(
    candidate: Path,
    *,
    trusted_roots: tuple[Path, ...],
    label: str,
    allow_leaf_symlink: bool,
) -> Path:
    """Resolve only OS aliases or symlinks that stay inside a trusted root."""

    absolute = candidate.absolute()
    links = _symlink_components(absolute)
    if links and links[-1] == absolute and not allow_leaf_symlink:
        raise PrivacyError(f"{label} is a symlink: {candidate}")
    resolved = absolute.resolve(strict=False)
    roots = tuple(root.absolute().resolve(strict=False) for root in trusted_roots)
    for link in links:
        if _is_known_system_alias(link):
            continue
        if not any(_is_relative_to(resolved, root) for root in roots):
            raise PrivacyError(f"{label} traverses a symlink outside a trusted root: {candidate}")
    return resolved


def _safe_source(
    path_text: str,
    *,
    base: Path,
    root: Path | None = None,
    allow_missing: bool = False,
) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [base / candidate]
        if root is not None and root != base:
            candidates.append(root / candidate)
    existing = [item for item in candidates if item.exists() or item.is_symlink()]
    if len(existing) > 1:
        resolved = {item.resolve() for item in existing}
        if len(resolved) != 1:
            raise PrivacyError(f"ambiguous relative evidence path: {path_text!r}")
    candidate = existing[0] if existing else candidates[0]
    trusted_roots = (base,) if root is None else (base, root)
    candidate = _resolve_trusted_source(
        candidate,
        trusted_roots=trusted_roots,
        label="source evidence path",
        allow_leaf_symlink=True,
    )
    if candidate.exists() and not candidate.is_file():
        raise PrivacyError(f"redistributable evidence object is missing: {path_text!r}")
    if not candidate.exists() and not allow_missing:
        raise PrivacyError(f"redistributable evidence object is missing: {path_text!r}")
    return candidate


def _source_payload_requires_descriptor(
    path_text: str,
    *,
    base: Path,
    state: _BuildState,
) -> bool:
    """Detect opaque data hidden behind an otherwise redistributable suffix."""

    candidate = _safe_source(
        path_text,
        base=base,
        root=state.root,
        allow_missing=state.descriptor_mode == DESCRIPTOR_MODE_OFFLINE,
    )
    if not candidate.exists():
        return False
    before = _stat_identity(candidate)
    with candidate.open("rb") as handle:
        prefix = handle.read(8192)
    if _stat_identity(candidate) != before:
        raise PrivacyError(f"source evidence changed while being classified: {candidate.name}")
    if any(prefix.startswith(signature) for signature in _BINARY_SIGNATURES):
        return True
    if b"\x00" in prefix:
        return True
    try:
        prefix.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


@dataclass
class _BuildState:
    base: Path
    root: Path | None = None
    staging: Path | None = None
    entries: dict[str, Path] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
    descriptors: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[tuple[str, str], tuple[str, int, str]] = field(default_factory=dict)
    source_cache: dict[Path, tuple[tuple[int, int, int, int], Path, str]] = field(default_factory=dict)
    descriptor_hash_cache: dict[Path, tuple[tuple[int, int, int, int], str]] = field(default_factory=dict)
    active_sources: dict[Path, str] = field(default_factory=dict)
    descriptor_mode: str = DESCRIPTOR_MODE_VERIFY


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _has_identity(path: Path, expected: tuple[int, int, int, int]) -> bool:
    try:
        return not path.is_symlink() and path.is_file() and _stat_identity(path) == expected
    except OSError:
        return False


def _publish_verified_candidate(
    candidate: Path,
    output: Path,
    *,
    prior_identity: tuple[int, int, int, int] | None,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    """Publish without an identity-check/overwrite race at the destination."""

    candidate_identity = _stat_identity(candidate)
    candidate_sha256 = _stream_sha256(candidate)
    if _stat_identity(candidate) != candidate_identity or (candidate_identity[2], candidate_sha256) != (
        expected_bytes,
        expected_sha256,
    ):
        raise PrivacyError("public bundle candidate changed after its relocated copy was verified")
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
                raise PrivacyError("public bundle changed before verified replacement")
            displaced_is_verified = True

        try:
            os.link(candidate, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise PrivacyError("public bundle output appeared during atomic publication") from exc
        if (
            not _has_identity(output, candidate_identity)
            or _stream_sha256(output) != expected_sha256
            or not _has_identity(output, candidate_identity)
        ):
            raise PrivacyError("published public bundle does not retain the verified candidate identity")
        published = True
        if displaced is not None:
            displaced.unlink()
            displaced = None
    except OSError as exc:
        raise PrivacyError(f"cannot atomically publish verified public bundle: {exc}") from exc
    finally:
        if not published and _has_identity(output, candidate_identity):
            output.unlink()
        if not published and displaced_is_verified and displaced is not None:
            if not output.exists() and not output.is_symlink():
                try:
                    os.link(displaced, output, follow_symlinks=False)
                except OSError:
                    pass
        if displaced is not None and prior_identity is not None and _has_identity(output, prior_identity):
            displaced.unlink(missing_ok=True)


def _require_staging(state: _BuildState) -> Path:
    if state.staging is None:
        raise PrivacyError("internal public-bundle staging directory is unavailable")
    return state.staging


def _write_staged_payload(
    payload: bytes,
    *,
    state: _BuildState,
    namespace: str,
    identifier: str,
) -> Path:
    root = _require_staging(state) / namespace
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identifier
    if destination.exists():
        if destination.stat().st_size != len(payload) or _stream_sha256(destination) != _sha(payload):
            raise PrivacyError("staged content-address collision")
    else:
        destination.write_bytes(payload)
    return destination


def _stage_entry(relative: str, payload: bytes, *, state: _BuildState) -> Path:
    canonical = validate_public_member_path(relative)
    scan_member(canonical, payload)
    destination = _write_staged_payload(
        payload,
        state=state,
        namespace="entries",
        identifier=_sha(canonical.encode("utf-8")),
    )
    prior = state.entries.get(canonical)
    if prior is not None and (prior.stat().st_size != len(payload) or _stream_sha256(prior) != _sha(payload)):
        raise PrivacyError("content-address collision while staging public bundle")
    state.entries[canonical] = destination
    return destination


def _read_stable_source(path: Path, *, state: _BuildState) -> tuple[bytes, str]:
    """Read a redistributable source once and reject concurrent mutation."""

    identity = _stat_identity(path)
    cached = state.source_cache.get(path.absolute())
    if cached is not None and cached[0] == identity:
        if _stat_identity(path) != identity:
            raise PrivacyError(f"source evidence changed while being read: {path.name}")
        return cached[1].read_bytes(), cached[2]
    if identity[2] > MAX_MEMBER_BYTES:
        raise PrivacyError(f"redistributable evidence object is oversized: {path.name}")
    payload = path.read_bytes()
    after = _stat_identity(path)
    if identity != after:
        raise PrivacyError(f"source evidence changed while being read: {path.name}")
    digest = _sha(payload)
    raw_stage = _write_staged_payload(
        payload,
        state=state,
        namespace="source-cache",
        identifier=digest,
    )
    state.source_cache[path.absolute()] = (after, raw_stage, digest)
    return payload, digest


def _hash_stable_source(path: Path, *, state: _BuildState) -> tuple[int, str]:
    """Stream and verify a descriptor source at most once per stable identity."""

    key = path.absolute()
    identity = _stat_identity(path)
    cached = state.descriptor_hash_cache.get(key)
    if cached is not None and cached[0] == identity:
        if _stat_identity(path) != identity:
            raise PrivacyError(f"source evidence changed while being verified: {path.name}")
        return identity[2], cached[1]
    observed_sha = file_sha256(path)
    after = _stat_identity(path)
    if identity != after:
        raise PrivacyError(f"source evidence changed while being verified: {path.name}")
    state.descriptor_hash_cache[key] = (after, observed_sha)
    return after[2], observed_sha


def _normalize_structured_artifact(
    payload: bytes,
    *,
    role: str,
    state: _BuildState,
    base: Path,
    kind: str,
) -> bytes:
    document = strict_json_loads(payload) if kind == "json" else _strict_yaml_loads(payload)
    normalized = _normalize(document, role=role, state=state, base=base)
    return _canonical_json(normalized)


def _publish_object(
    path_text: str,
    *,
    role: str,
    claimed_sha: str | None,
    claimed_bytes: int | None,
    state: _BuildState,
    base: Path,
) -> tuple[str, int, str]:
    source = _safe_source(path_text, base=base, root=state.root)
    source_key = source.absolute()
    if source_key in state.active_sources:
        raise PrivacyError(f"recursive evidence reference: {source.name}")
    payload, observed_sha = _read_stable_source(source, state=state)
    if claimed_sha is not None and claimed_sha != observed_sha:
        raise PrivacyError(f"source SHA-256 commitment mismatch for {role}")
    if claimed_bytes is not None and claimed_bytes != len(payload):
        raise PrivacyError(f"source byte commitment mismatch for {role}")
    cache_key = (observed_sha, role)
    if cache_key in state.cache:
        return state.cache[cache_key]

    state.active_sources[source_key] = role
    try:
        normalization = "byte-identical-copy"
        published = payload
        suffix = source.suffix.lower()
        if suffix == ".json":
            published = _normalize_structured_artifact(payload, role=role, state=state, base=source.parent, kind="json")
            normalization = "canonical-json-with-portable-path-descriptors"
        elif suffix in {".yaml", ".yml"}:
            published = _normalize_structured_artifact(payload, role=role, state=state, base=source.parent, kind="yaml")
            normalization = "canonical-json-from-yaml-with-portable-path-descriptors"
    finally:
        del state.active_sources[source_key]
    published_sha = _sha(published)
    bundle_path = f"objects/{published_sha}"
    try:
        scan_member(bundle_path, published)
    except PrivacyError as exc:
        raise PrivacyError(f"{exc}; source evidence role: {role}") from exc
    prior = state.entries.get(bundle_path)
    if prior is not None and (prior.stat().st_size != len(published) or _stream_sha256(prior) != published_sha):
        raise PrivacyError("content-address collision while constructing public bundle")
    _stage_entry(bundle_path, published, state=state)
    state.objects.append(
        {
            "bundle_path": bundle_path,
            "bytes": len(published),
            "content_role": role,
            "normalization": normalization,
            "published_sha256": published_sha,
            "source_bytes": len(payload),
            "source_sha256": observed_sha,
        }
    )
    result = (bundle_path, len(published), published_sha)
    state.cache[cache_key] = result
    return result


def _descriptor(
    *,
    role: str,
    claimed_sha: Any,
    claimed_bytes: Any,
    state: _BuildState,
    path_text: str | None = None,
    base: Path | None = None,
) -> dict[str, Any]:
    if claimed_sha is None:
        raise PrivacyError(f"non-redistributable object is not hash-bound: {role}")
    if not _is_sha256(claimed_sha):
        raise PrivacyError(f"non-redistributable object has invalid SHA-256: {role}")
    valid_claimed_bytes = not isinstance(claimed_bytes, bool) and isinstance(claimed_bytes, int) and claimed_bytes >= 0
    if claimed_bytes is not None and not valid_claimed_bytes:
        raise PrivacyError(f"non-redistributable object lacks a byte commitment: {role}")
    if path_text is not None:
        candidate = _safe_source(
            path_text,
            base=base or state.base,
            root=state.root,
            allow_missing=state.descriptor_mode == DESCRIPTOR_MODE_OFFLINE,
        )
        if candidate.exists():
            if not candidate.is_file():
                raise PrivacyError(f"non-redistributable source is not a file: {role}")
            observed_bytes, observed_sha = _hash_stable_source(candidate, state=state)
            if claimed_bytes is None:
                claimed_bytes = observed_bytes
                valid_claimed_bytes = True
            if observed_bytes != claimed_bytes or observed_sha != claimed_sha:
                raise PrivacyError(f"non-redistributable source commitment mismatch: {role}")
    if not valid_claimed_bytes:
        raise PrivacyError(f"non-redistributable object lacks a byte commitment: {role}")
    value = {
        "bytes": claimed_bytes,
        "content_role": role,
        "descriptor_only": True,
        "sha256": claimed_sha,
    }
    if value not in state.descriptors:
        state.descriptors.append(value)
    return value


def _normalize(value: Any, *, role: str, state: _BuildState, base: Path | None = None) -> Any:
    if base is None:
        base = state.base
    if isinstance(value, str):
        return _redact_machine_locations(value)
    if isinstance(value, list):
        return [_normalize(item, role=f"{role}[{index}]", state=state, base=base) for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    path_keys = [key for key, item in value.items() if key == "path" or key.endswith("_path") if isinstance(item, str)]
    for key, item in value.items():
        child_role = f"{role}.{key}" if role else key
        if key not in path_keys:
            result[key] = _normalize(item, role=child_role, state=state, base=base)

    for path_key in path_keys:
        path_text = value[path_key]
        object_role = role or path_key
        claimed_sha = _identity_field(value, path_key, "sha256")
        claimed_bytes = _identity_field(value, path_key, "bytes")
        nonredistributable = _is_nonredistributable(object_role, path_text)
        if not nonredistributable:
            nonredistributable = _source_payload_requires_descriptor(
                path_text,
                base=base,
                state=state,
            )
        # A path with an opaque suffix or a content commitment denotes an
        # evidence object, even when its source-machine location is private.
        # An uncommitted extensionless path field is location metadata only.
        if nonredistributable and (claimed_sha is not None or Path(path_text).suffix):
            descriptor = _descriptor(
                role=object_role,
                claimed_sha=claimed_sha,
                claimed_bytes=claimed_bytes,
                state=state,
                path_text=path_text,
                base=base,
            )
            result[path_key + "_descriptor"] = descriptor
            continue
        if claimed_sha is None and _is_private_location(path_text):
            result[path_key] = "private-source-location-redacted"
            result[path_key + "_normalization"] = "location-only; no content claim"
            continue
        if nonredistributable:
            descriptor = _descriptor(
                role=object_role,
                claimed_sha=claimed_sha,
                claimed_bytes=claimed_bytes,
                state=state,
                path_text=path_text,
                base=base,
            )
            result[path_key + "_descriptor"] = descriptor
            continue
        source = _safe_source(path_text, base=base, root=state.root)
        source_key = source.absolute()
        if source_key in state.active_sources:
            payload, observed_sha = _read_stable_source(source, state=state)
            if isinstance(claimed_sha, str) and claimed_sha != observed_sha:
                raise PrivacyError(f"recursive source SHA-256 commitment mismatch for {object_role}")
            if isinstance(claimed_bytes, int) and not isinstance(claimed_bytes, bool) and claimed_bytes != len(payload):
                raise PrivacyError(f"recursive source byte commitment mismatch for {object_role}")
            result[path_key + "_source_reference"] = {
                "bytes": len(payload),
                "content_role": state.active_sources[source_key],
                "normalization": "recursive-source-reference",
                "source_sha256": observed_sha,
            }
            continue
        bundle_path, published_bytes, published_sha = _publish_object(
            path_text,
            role=object_role,
            claimed_sha=claimed_sha if isinstance(claimed_sha, str) else None,
            claimed_bytes=claimed_bytes if isinstance(claimed_bytes, int) else None,
            state=state,
            base=base,
        )
        result[path_key] = bundle_path
        result[path_key + "_published_bytes"] = published_bytes
        result[path_key + "_published_sha256"] = published_sha
    return result


def _assemble_public_bundle(
    *,
    source_path: Path,
    source_bytes: bytes,
    source_document: dict[str, Any],
    output_path: Path,
    descriptor_mode: str,
    prior_receipt: dict[str, Any] | None,
    prior_identity: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[4]
    source_root = code_root if source_path.is_relative_to(code_root) else source_path.parent
    with tempfile.TemporaryDirectory(prefix="cct20-public-stage-") as staging_text:
        state = _BuildState(
            base=source_path.parent,
            root=source_root,
            staging=Path(staging_text),
            descriptor_mode=descriptor_mode,
        )
        portable = _normalize(
            source_document,
            role="",
            state=state,
            base=source_path.parent,
        )
        portable_bytes = _canonical_json(portable)
        portable_path = f"objects/{_sha(portable_bytes)}"
        _stage_entry(portable_path, portable_bytes, state=state)
        manifest = {
            "descriptor_source_verification": descriptor_mode,
            "nonredistributable_objects": sorted(
                state.descriptors,
                key=lambda row: (row["content_role"], row["sha256"]),
            ),
            "objects": sorted(
                state.objects,
                key=lambda row: (row["bundle_path"], row["content_role"]),
            ),
            "portable_release": {
                "bytes": len(portable_bytes),
                "bundle_path": portable_path,
                "normalization": PORTABLE_NORMALIZATION,
                "published_sha256": _sha(portable_bytes),
            },
            "schema": SCHEMA,
            "source_release": {
                "bytes": len(source_bytes),
                "sha256": _sha(source_bytes),
            },
        }
        _stage_entry("README.md", README_BYTES, state=state)
        _stage_entry("manifest.json", _canonical_json(manifest), state=state)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, candidate_text = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".candidate",
        )
        os.close(descriptor)
        candidate_output = Path(candidate_text)
        candidate_output.unlink()
        _build_public_zip(candidate_output, state.entries)

        # Verification occurs from a fresh directory and consumes no source path.
        try:
            with tempfile.TemporaryDirectory(prefix="cct20-public-verify-") as temporary:
                relocated = Path(temporary) / "bundle.zip"
                shutil.copyfile(candidate_output, relocated)
                receipt = verify_public_bundle(
                    relocated,
                    allow_offline_descriptors=(descriptor_mode == DESCRIPTOR_MODE_OFFLINE),
                )
            if prior_identity is not None:
                if (
                    output_path.is_symlink()
                    or not output_path.is_file()
                    or _stat_identity(output_path) != prior_identity
                ):
                    raise PrivacyError("public bundle changed before verified replacement")
            elif output_path.exists() or output_path.is_symlink():
                raise PrivacyError("public bundle output appeared during publication")
            if prior_receipt is not None and receipt["archive_sha256"] == prior_receipt["archive_sha256"]:
                return prior_receipt
            _publish_verified_candidate(
                candidate_output,
                output_path,
                prior_identity=prior_identity,
                expected_bytes=int(receipt["archive_bytes"]),
                expected_sha256=str(receipt["archive_sha256"]),
            )
            return receipt
        finally:
            candidate_output.unlink(missing_ok=True)


def build_public_bundle(
    release_manifest_path: Path,
    output_path: Path,
    *,
    descriptor_mode: str = DESCRIPTOR_MODE_VERIFY,
    replace_verified: bool = False,
) -> dict[str, Any]:
    """Build an immutable ZIP without exposing any source-machine location."""

    if descriptor_mode not in DESCRIPTOR_MODES:
        raise PrivacyError(f"unsupported descriptor mode: {descriptor_mode!r}")
    if (output_path.exists() or output_path.is_symlink()) and not replace_verified:
        raise PrivacyError(f"public bundle output already exists: {output_path}")
    if output_path.is_symlink():
        raise PrivacyError(f"refusing to replace a symlinked public bundle: {output_path}")
    prior_receipt: dict[str, Any] | None = None
    prior_identity: tuple[int, int, int, int] | None = None
    if output_path.exists():
        prior_identity = _stat_identity(output_path)
        prior_receipt = verify_public_bundle(
            output_path,
            allow_offline_descriptors=True,
        )
        if output_path.is_symlink() or not output_path.is_file() or _stat_identity(output_path) != prior_identity:
            raise PrivacyError("public bundle changed while existing output was verified")

    raw_source = release_manifest_path.absolute()
    source_path = _resolve_trusted_source(
        raw_source,
        trusted_roots=(raw_source.parent,),
        label="CCT-20 source manifest",
        allow_leaf_symlink=False,
    )
    if not source_path.is_file():
        raise PrivacyError(f"CCT-20 source manifest is missing: {release_manifest_path}")
    if source_path.stat().st_size > MAX_SOURCE_MANIFEST_BYTES:
        raise PrivacyError("CCT-20 source manifest exceeds the safety limit")
    source_identity = _stat_identity(source_path)
    source_bytes = source_path.read_bytes()
    if _stat_identity(source_path) != source_identity:
        raise PrivacyError("CCT-20 source manifest changed while being read")
    source_document = strict_json_loads(source_bytes)
    if not isinstance(source_document, dict):
        raise PrivacyError("CCT-20 release manifest must be a JSON object")
    return _assemble_public_bundle(
        source_path=source_path,
        source_bytes=source_bytes,
        source_document=source_document,
        output_path=output_path,
        descriptor_mode=descriptor_mode,
        prior_receipt=prior_receipt,
        prior_identity=prior_identity,
    )


def _verify_portable_graph(
    value: Any,
    *,
    role: str,
    objects: list[dict[str, Any]],
    descriptors: list[dict[str, Any]],
) -> tuple[set[int], set[int]]:
    object_refs: set[int] = set()
    descriptor_refs: set[int] = set()
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_objects, child_descriptors = _verify_portable_graph(
                item,
                role=f"{role}[{index}]",
                objects=objects,
                descriptors=descriptors,
            )
            object_refs.update(child_objects)
            descriptor_refs.update(child_descriptors)
        return object_refs, descriptor_refs
    if not isinstance(value, dict):
        return object_refs, descriptor_refs

    for key in value:
        if key in {"path_published_bytes", "path_published_sha256"} or key.endswith(
            ("_path_published_bytes", "_path_published_sha256")
        ):
            path_key = key.rsplit("_published_", 1)[0]
            if path_key not in value:
                raise PrivacyError(f"orphan portable path commitment: {role}.{key}")
        elif key == "path_normalization" or key.endswith("_path_normalization"):
            path_key = key[: -len("_normalization")]
            if value.get(path_key) != "private-source-location-redacted":
                raise PrivacyError(f"orphan portable path normalization: {role}.{key}")

    for key, item in value.items():
        child_role = f"{role}.{key}" if role else key
        if key == "path_source_reference" or key.endswith("_path_source_reference"):
            if (
                not isinstance(item, dict)
                or set(item) != {"bytes", "content_role", "normalization", "source_sha256"}
                or item.get("normalization") != "recursive-source-reference"
                or not _is_count(item.get("bytes"))
                or not _is_sha256(item.get("source_sha256"))
            ):
                raise PrivacyError(f"malformed recursive source reference: {child_role}")
            matches = [
                index
                for index, row in enumerate(objects)
                if row.get("content_role") == item.get("content_role")
                and row.get("source_sha256") == item.get("source_sha256")
                and row.get("source_bytes") == item.get("bytes")
            ]
            if len(matches) != 1:
                raise PrivacyError(f"unbound recursive source reference: {child_role}")
            object_refs.add(matches[0])
            continue
        if key == "path_descriptor" or key.endswith("_path_descriptor"):
            path_key = key[: -len("_descriptor")]
            expected_role = role or path_key
            matches = [
                index
                for index, descriptor in enumerate(descriptors)
                if item == descriptor and descriptor.get("content_role") == expected_role
            ]
            if len(matches) != 1:
                raise PrivacyError(f"unbound portable descriptor: {child_role}")
            descriptor_refs.add(matches[0])
            continue
        if (key == "path" or key.endswith("_path")) and isinstance(item, str):
            if item == "private-source-location-redacted":
                if value.get(key + "_normalization") != "location-only; no content claim":
                    raise PrivacyError(f"unlabeled redacted portable path: {child_role}")
                continue
            matches = [
                index
                for index, row in enumerate(objects)
                if row.get("bundle_path") == item and row.get("content_role") == (role or key)
            ]
            if len(matches) != 1:
                raise PrivacyError(f"unbound portable path reference: {child_role}")
            row = objects[matches[0]]
            claimed_bytes = _identity_field(value, key, "bytes")
            claimed_sha = _identity_field(value, key, "sha256")
            if (
                value.get(key + "_published_bytes") != row.get("bytes")
                or value.get(key + "_published_sha256") != row.get("published_sha256")
                or (claimed_bytes is not None and claimed_bytes != row.get("source_bytes"))
                or (claimed_sha is not None and claimed_sha != row.get("source_sha256"))
            ):
                raise PrivacyError(f"portable path commitment mismatch: {child_role}")
            object_refs.add(matches[0])
            continue
        child_objects, child_descriptors = _verify_portable_graph(
            item,
            role=child_role,
            objects=objects,
            descriptors=descriptors,
        )
        object_refs.update(child_objects)
        descriptor_refs.update(child_descriptors)
    return object_refs, descriptor_refs


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_exact_fields(
    value: object,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PrivacyError(f"public-bundle {label} schema fields are malformed")
    return value


def _validated_manifest_ledgers(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if set(manifest) != {
        "schema",
        "descriptor_source_verification",
        "source_release",
        "portable_release",
        "objects",
        "nonredistributable_objects",
    }:
        raise PrivacyError("public-bundle manifest schema fields are malformed")
    if manifest["descriptor_source_verification"] not in DESCRIPTOR_MODES:
        raise PrivacyError("public-bundle descriptor source-verification mode is malformed")

    source = _require_exact_fields(
        manifest["source_release"],
        frozenset({"bytes", "sha256"}),
        label="source_release",
    )
    if not _is_count(source["bytes"]) or not _is_sha256(source["sha256"]):
        raise PrivacyError("public-bundle source_release commitment is malformed")

    portable = _require_exact_fields(
        manifest["portable_release"],
        frozenset({"bytes", "bundle_path", "normalization", "published_sha256"}),
        label="portable_release",
    )
    if (
        not _is_count(portable["bytes"])
        or not _is_sha256(portable["published_sha256"])
        or portable["normalization"] != PORTABLE_NORMALIZATION
        or portable["bundle_path"] != f"objects/{portable['published_sha256']}"
    ):
        raise PrivacyError("public-bundle portable_release is not content-addressed")
    validate_public_member_path(portable["bundle_path"])

    raw_rows = manifest["objects"]
    raw_descriptors = manifest["nonredistributable_objects"]
    if not isinstance(raw_rows, list) or not isinstance(raw_descriptors, list):
        raise PrivacyError("public-bundle manifest inventories are malformed")
    rows: list[dict[str, Any]] = []
    object_fields = frozenset(
        {
            "bundle_path",
            "bytes",
            "content_role",
            "normalization",
            "published_sha256",
            "source_bytes",
            "source_sha256",
        }
    )
    for raw in raw_rows:
        row = _require_exact_fields(raw, object_fields, label="object ledger row")
        if (
            not isinstance(row["content_role"], str)
            or not row["content_role"]
            or row["normalization"] not in OBJECT_NORMALIZATIONS
            or not _is_count(row["bytes"])
            or not _is_count(row["source_bytes"])
            or not _is_sha256(row["published_sha256"])
            or not _is_sha256(row["source_sha256"])
            or row["bundle_path"] != f"objects/{row['published_sha256']}"
        ):
            raise PrivacyError("public-bundle object ledger row is malformed")
        if row["normalization"] == "byte-identical-copy" and (
            row["bytes"] != row["source_bytes"] or row["published_sha256"] != row["source_sha256"]
        ):
            raise PrivacyError("public-bundle byte-identical object ledger is inconsistent")
        validate_public_member_path(row["bundle_path"])
        rows.append(row)
    sorted_rows = sorted(rows, key=lambda row: (row["bundle_path"], row["content_role"]))
    if rows != sorted_rows or len({(row["bundle_path"], row["content_role"]) for row in rows}) != len(rows):
        raise PrivacyError("public-bundle object ledger is unsorted or duplicated")

    descriptors: list[dict[str, Any]] = []
    descriptor_fields = frozenset({"bytes", "content_role", "descriptor_only", "sha256"})
    for raw in raw_descriptors:
        descriptor = _require_exact_fields(
            raw,
            descriptor_fields,
            label="nonredistributable descriptor",
        )
        if (
            descriptor["descriptor_only"] is not True
            or not isinstance(descriptor["content_role"], str)
            or not descriptor["content_role"]
            or not _is_count(descriptor["bytes"])
            or not _is_sha256(descriptor["sha256"])
        ):
            raise PrivacyError("public-bundle nonredistributable descriptor is malformed")
        descriptors.append(descriptor)
    sorted_descriptors = sorted(
        descriptors,
        key=lambda row: (row["content_role"], row["sha256"]),
    )
    if descriptors != sorted_descriptors or len({(row["content_role"], row["sha256"]) for row in descriptors}) != len(
        descriptors
    ):
        raise PrivacyError("public-bundle nonredistributable descriptor ledger is unsorted or duplicated")
    return rows, portable, descriptors


def verify_public_bundle(
    path: Path,
    *,
    allow_offline_descriptors: bool = False,
    source_release_path: Path | None = None,
) -> dict[str, Any]:
    """Verify canonical bytes and the complete object graph using the ZIP alone."""

    source_release_identity: tuple[int, int, int, int] | None = None
    source_release_resolved: Path | None = None
    source_release_commitment: tuple[int, str] | None = None
    if source_release_path is not None:
        raw_source = source_release_path.absolute()
        source_release_resolved = _resolve_trusted_source(
            raw_source,
            trusted_roots=(raw_source.parent,),
            label="CCT-20 source manifest",
            allow_leaf_symlink=False,
        )
        if not source_release_resolved.is_file():
            raise PrivacyError(f"CCT-20 source manifest is missing: {source_release_path}")
        if source_release_resolved.stat().st_size > MAX_SOURCE_MANIFEST_BYTES:
            raise PrivacyError("CCT-20 source manifest exceeds the safety limit")
        source_release_identity = _stat_identity(source_release_resolved)
        source_release_payload = source_release_resolved.read_bytes()
        if _stat_identity(source_release_resolved) != source_release_identity:
            raise PrivacyError("CCT-20 source manifest changed while being read")
        source_release_commitment = (len(source_release_payload), _sha(source_release_payload))

    if path.is_symlink() or not path.is_file():
        raise PrivacyError(f"public bundle is missing or a symlink: {path}")
    archive_identity = _stat_identity(path)
    _verify_public_zip_container(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        name_set = set(names)
        if "manifest.json" not in names:
            raise PrivacyError("public bundle has no manifest.json")
        manifest_bytes = archive.read("manifest.json")
        manifest = strict_json_loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise PrivacyError("unsupported public-bundle manifest schema")
        if manifest_bytes != _canonical_json(manifest):
            raise PrivacyError("public-bundle manifest is not canonical JSON")
        rows, portable, descriptors = _validated_manifest_ledgers(manifest)
        if source_release_commitment is not None:
            source_release = manifest["source_release"]
            if (source_release["bytes"], source_release["sha256"]) != source_release_commitment:
                raise PrivacyError("public bundle does not match the current CCT-20 source release")
        descriptor_mode = manifest["descriptor_source_verification"]
        if descriptor_mode == DESCRIPTOR_MODE_OFFLINE and not allow_offline_descriptors:
            raise PrivacyError(
                "public bundle uses offline descriptor commitments; strict source verification was not performed"
            )
        if archive.read("README.md") != README_BYTES:
            raise PrivacyError("public-bundle README bytes are not canonical")
        expected = {"manifest.json", "README.md", str(portable.get("bundle_path", ""))}
        verified_objects: dict[str, tuple[int, str]] = {}
        for row in rows:
            relative = validate_public_member_path(str(row.get("bundle_path", "")))
            if relative not in name_set:
                raise PrivacyError(f"public-bundle object hash mismatch: {relative}")
            observed = verified_objects.get(relative)
            if observed is None:
                payload = archive.read(relative)
                observed = (len(payload), _sha(payload))
                verified_objects[relative] = observed
            if observed != (row.get("bytes"), row.get("published_sha256")):
                raise PrivacyError(f"public-bundle object hash mismatch: {relative}")
            expected.add(relative)
        portable_path = validate_public_member_path(str(portable.get("bundle_path", "")))
        portable_payload = archive.read(portable_path) if portable_path in name_set else None
        if (
            portable_payload is None
            or len(portable_payload) != portable.get("bytes")
            or _sha(portable_payload) != portable.get("published_sha256")
        ):
            raise PrivacyError("portable release manifest hash mismatch")
        portable_document = strict_json_loads(portable_payload)
        if not isinstance(portable_document, dict):
            raise PrivacyError("portable release manifest is not a JSON object")
        if portable_payload != _canonical_json(portable_document):
            raise PrivacyError("portable release manifest is not canonical JSON")
        object_refs, descriptor_refs = _verify_portable_graph(
            portable_document,
            role="",
            objects=rows,
            descriptors=descriptors,
        )
        pending = list(object_refs)
        visited: set[int] = set()
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            row = rows[index]
            if not str(row.get("normalization", "")).startswith("canonical-json"):
                continue
            nested_payload = archive.read(str(row["bundle_path"]))
            nested_document = strict_json_loads(nested_payload)
            if nested_payload != _canonical_json(nested_document):
                raise PrivacyError("normalized evidence object is not canonical JSON")
            child_objects, child_descriptors = _verify_portable_graph(
                nested_document,
                role=str(row.get("content_role", "")),
                objects=rows,
                descriptors=descriptors,
            )
            descriptor_refs.update(child_descriptors)
            unseen = child_objects - object_refs
            object_refs.update(child_objects)
            pending.extend(unseen)
        if object_refs != set(range(len(rows))):
            raise PrivacyError("public-bundle ledger contains an unreferenced object")
        if descriptor_refs != set(range(len(descriptors))):
            raise PrivacyError("public-bundle ledger contains an unreferenced descriptor")
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or descriptor.get("descriptor_only") is not True:
                raise PrivacyError("non-redistributable descriptor is malformed")
            _descriptor(
                role=str(descriptor.get("content_role", "")),
                claimed_sha=descriptor.get("sha256"),
                claimed_bytes=descriptor.get("bytes"),
                state=_BuildState(base=Path(".")),
            )
        if set(names) != expected:
            raise PrivacyError("public bundle contains an unmanifested or missing member")
    if _stat_identity(path) != archive_identity:
        raise PrivacyError("public bundle changed while being verified")
    archive_sha = _stream_sha256(path)
    if _stat_identity(path) != archive_identity:
        raise PrivacyError("public bundle changed while being hashed")
    if (
        source_release_resolved is not None
        and source_release_identity is not None
        and (
            source_release_resolved.is_symlink()
            or not source_release_resolved.is_file()
            or _stat_identity(source_release_resolved) != source_release_identity
        )
    ):
        raise PrivacyError("CCT-20 source manifest changed during public-bundle verification")
    return {
        "archive_bytes": archive_identity[2],
        "archive_sha256": archive_sha,
        "object_count": len(rows),
        "descriptor_source_verification": descriptor_mode,
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true")
    checks.add_argument(
        "--portable-check",
        action="store_true",
        help="verify archive semantics without enforcing the source build runtime",
    )
    parser.add_argument(
        "--descriptor-mode",
        choices=sorted(DESCRIPTOR_MODES),
        default=DESCRIPTOR_MODE_VERIFY,
        help="strictly verify descriptor sources, or explicitly build from commitments offline",
    )
    parser.add_argument("--replace-verified", action="store_true")
    parser.add_argument("--allow-offline-descriptors", action="store_true")
    args = parser.parse_args()
    if args.portable_check and args.allow_offline_descriptors:
        parser.error("--portable-check preserves strict descriptor provenance labels")
    if not args.portable_check:
        verify_release_python_content()
    if args.portable_check:
        receipt = verify_public_bundle(args.output, source_release_path=args.release_manifest)
    elif args.check:
        receipt = verify_public_bundle(
            args.output,
            allow_offline_descriptors=args.allow_offline_descriptors,
            source_release_path=args.release_manifest,
        )
    else:
        receipt = build_public_bundle(
            args.release_manifest,
            args.output,
            descriptor_mode=args.descriptor_mode,
            replace_verified=args.replace_verified,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
