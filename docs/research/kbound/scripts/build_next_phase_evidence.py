#!/usr/bin/env python3
"""Freeze and package explicitly selected next-phase study evidence unchanged.

The committed manifest is source-sealed; its hashes bind ignored evidence bytes
transitively. This is an internal reviewer archive, not an anonymous publication
or a scientific-completeness certificate. No target data are discovered outside
the explicitly named study directories. Original attempts and source snapshots
are retained byte for byte. Generate the manifest only after writers stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "kbound-next-phase-evidence-v1"
MANIFEST_MEMBER = "NEXT_PHASE_EVIDENCE_MANIFEST.json"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
SCOPE = "integrity_only_not_scientific_verification"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("evidence path must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise ValueError(f"unsafe evidence path: {value!r}")
    return value


def _study(value: object) -> str:
    try:
        relative = _relative(value)
    except ValueError as exc:
        raise ValueError(f"unsafe study directory: {value!r}") from exc
    path = PurePosixPath(relative)
    if len(path.parts) != 3 or path.parts[:2] != ("output", "next_phase"):
        raise ValueError(f"study must be one explicit output/next_phase child: {relative}")
    return relative


def _unlinked(path: Path) -> None:
    absolute = path.absolute()
    if any(parent.is_symlink() for parent in (absolute, *absolute.parents)):
        raise ValueError(f"evidence path contains a symlink: {path}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> bytes:
    _unlinked(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError(f"evidence must be a bounded regular file: {path}")
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"evidence grew beyond the file limit: {path}")
    return data


def _paths(repo: Path, studies: list[str]) -> list[str]:
    found: list[str] = []
    for relative in studies:
        root = repo / _study(relative)
        _unlinked(root)
        if not root.is_dir():
            raise ValueError(f"study directory is missing: {relative}")
        for parent, directories, filenames in os.walk(root, followlinks=False):
            for name in directories:
                _unlinked(Path(parent) / name)
            for name in filenames:
                path = Path(parent) / name
                _unlinked(path)
                if not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError(f"study contains a non-regular file: {path}")
                found.append(path.relative_to(repo).as_posix())
    return sorted(found)


def validate_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("scope") != SCOPE:
        raise ValueError("unsupported evidence manifest schema or scope")
    studies = payload.get("study_directories")
    if not isinstance(studies, list) or not studies or studies != sorted(set(studies)):
        raise ValueError("study directories must be nonempty and canonical")
    for study in studies:
        _study(study)
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evidence artifacts must be nonempty")
    paths: list[str] = []
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("malformed evidence artifact")
        relative = _relative(row["path"])
        if not any(relative.startswith(study + "/") for study in studies):
            raise ValueError(f"artifact path outside selected studies: {relative}")
        size, digest = row["bytes"], row["sha256"]
        if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
            raise ValueError("invalid evidence byte count")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid evidence SHA-256")
        paths.append(relative)
        total += size
    if paths != sorted(set(paths)):
        raise ValueError("artifact paths must be canonical and unique")
    if total > MAX_TOTAL_BYTES or payload.get("total_bytes") != total:
        raise ValueError("invalid evidence total size")
    authorities = payload.get("authority_paths")
    if not isinstance(authorities, list) or not authorities or authorities != sorted(set(authorities)):
        raise ValueError("authority paths must be nonempty and canonical")
    if not set(authorities) <= set(paths):
        raise ValueError("authority must be a manifested file")
    if any(not any(path.startswith(study + "/") for path in authorities) for study in studies):
        raise ValueError("every study needs an explicitly selected final authority")
    if payload.get("artifacts_sha256") != _sha(_canonical(rows)):
        raise ValueError("invalid aggregate evidence hash")
    return payload


def build_manifest(repo: Path, studies: list[str], authorities: list[str]) -> dict[str, Any]:
    studies = sorted({_study(value) for value in studies})
    rows: list[dict[str, Any]] = []
    for relative in _paths(repo, studies):
        data = _read(repo / relative)
        rows.append({"path": relative, "bytes": len(data), "sha256": _sha(data)})
    payload = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "distribution": "internal_reviewer_archive_with_original_provenance_paths",
        "study_directories": studies,
        "authority_paths": sorted(set(authorities)),
        "artifacts": rows,
        "artifacts_sha256": _sha(_canonical(rows)),
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    return validate_manifest(payload)


def _publish(temporary: Path, output: Path) -> None:
    # Hard-link publication is create-only and atomic on the release APFS volume.
    # Unsupported filesystems fail visibly; existing outputs are never replaced.
    _unlinked(output)
    os.link(temporary, output)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    validate_manifest(payload)
    _unlinked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evidence-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
            _publish(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read(path)
    payload = validate_manifest(json.loads(raw))
    if _canonical(payload) != raw:
        raise ValueError("evidence manifest bytes are not canonical")
    return payload, raw


def _check_worktree(repo: Path, manifest: dict[str, Any]) -> None:
    current = build_manifest(repo, manifest["study_directories"], manifest["authority_paths"])
    if current != manifest:
        raise ValueError("evidence drift since manifest freeze")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build_archive(repo: Path, manifest_path: Path, output: Path) -> None:
    manifest, raw = _load(manifest_path)
    _check_worktree(repo, manifest)
    _unlinked(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".evidence-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(_zip_info(MANIFEST_MEMBER), raw)
            for row in manifest["artifacts"]:
                data = _read(repo / row["path"])
                if len(data) != row["bytes"] or _sha(data) != row["sha256"]:
                    raise ValueError(f"evidence drift during packaging: {row['path']}")
                archive.writestr(_zip_info(row["path"]), data)
        verify_archive(manifest_path, temporary)
        _check_worktree(repo, manifest)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(manifest_path: Path, archive_path: Path) -> None:
    manifest, raw = _load(manifest_path)
    _unlinked(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        expected = [MANIFEST_MEMBER, *(row["path"] for row in manifest["artifacts"])]
        if names != expected or len(names) != len(set(names)):
            raise ValueError("archive members differ from the exact evidence inventory")
        if archive.getinfo(MANIFEST_MEMBER).file_size != len(raw) or archive.read(MANIFEST_MEMBER) != raw:
            raise ValueError("archive manifest differs from the source-bound manifest")
        for row in manifest["artifacts"]:
            info = archive.getinfo(row["path"])
            if info.file_size != row["bytes"] or info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"archive byte count differs: {row['path']}")
            if _sha(archive.read(info)) != row["sha256"]:
                raise ValueError(f"archive hash differs: {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--study-directory", action="append", default=[])
    parser.add_argument("--authority", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not args.archive or args.study_directory or args.authority:
            parser.error("--check requires --archive and no creation arguments")
        verify_archive(args.manifest, args.archive)
        print("next-phase evidence archive: PASS (byte integrity only)")
    elif args.study_directory:
        if args.archive:
            parser.error("freeze the manifest before creating the archive")
        write_manifest(args.manifest, build_manifest(args.repo, args.study_directory, args.authority))
        print("next-phase evidence manifest frozen; commit before the source seal")
    elif args.archive:
        if args.authority:
            parser.error("--authority is only valid with --study-directory")
        build_archive(args.repo, args.manifest, args.archive)
        print("next-phase evidence archive created and verified")
    else:
        parser.error("provide study directories to freeze, or --archive to package")


if __name__ == "__main__":
    main()
