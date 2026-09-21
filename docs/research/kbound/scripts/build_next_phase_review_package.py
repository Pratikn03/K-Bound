#!/usr/bin/env python3
"""Build a create-only, independently verifiable internal reviewer package.

The source archive is the declared release inventory, not full Git history.
Only named source-seal/checksum members are read; protected data are never
discovered. Hashes establish byte integrity, not authorship or scientific truth.
QA records assert human page inspection; this program cannot perform it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

evidence = importlib.import_module("docs.research.kbound.scripts.build_next_phase_evidence")
seals = importlib.import_module("docs.research.kbound.scripts.build_release_source_seal")
checksums = importlib.import_module("docs.research.kbound.scripts.verify_release_checksums")
verification = importlib.import_module("docs.research.kbound.scripts.run_repository_verification")

SCHEMA = "kbound-next-phase-review-package-v1"
MANIFEST_MEMBER = "REVIEW_PACKAGE_MANIFEST.json"
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("archive path must be a string")
    parsed = PurePosixPath(value)
    if not value or value.startswith(("/", "~")) or ".." in parsed.parts or "\\" in value or str(parsed) != value:
        raise ValueError(f"unsafe archive path: {value!r}")
    if any(ord(c) < 32 for c in value):
        raise ValueError("unsafe control character in archive path")
    return value


def _unlinked(path: Path) -> None:
    if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError(f"symlink not permitted: {path}")


def _read(path: Path) -> bytes:
    _unlinked(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MEMBER_BYTES:
        raise ValueError(f"not a bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > MAX_MEMBER_BYTES:
        raise ValueError(f"file grew beyond bound: {path}")
    return raw


def _repo_path(repo: Path, path: Path) -> str:
    _unlinked(path)
    try:
        return _relative(path.absolute().relative_to(repo.absolute()).as_posix())
    except ValueError as exc:
        raise ValueError(f"metadata/document must reside under the repository: {path}") from exc


def _rows(members: dict[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": len(data), "sha256": _sha(data)} for name, data in sorted(members.items())]


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            _relative(name)
            archive.writestr(evidence._zip_info(name), data)
    return buffer.getvalue()


def _zip_members(raw: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        total = 0
        for info in infos:
            _relative(info.filename)
            mode = info.external_attr >> 16
            if not stat.S_ISREG(mode) or info.is_dir() or info.flag_bits & 1:
                raise ValueError("archive contains nonregular or encrypted member")
            if info.date_time != evidence.FIXED_TIME:
                raise ValueError("archive timestamp is not canonical")
            total += info.file_size
            if info.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("archive exceeds byte limits")
        return {info.filename: archive.read(info) for info in infos}


def _checksum_entries(raw: bytes, required: tuple[str, ...]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("malformed checksum line")
        digest, relative = match.groups()
        _relative(relative)
        if relative in entries:
            raise ValueError("duplicate checksum member")
        entries[relative] = digest
    if tuple(entries) != required:
        raise ValueError("checksum inventory is not the exact canonical release inventory")
    return entries


def _check_verification(raw: bytes, seal: dict[str, Any]) -> None:
    payload = json.loads(raw)
    required = [gate.name for gate in verification.release_gate_plan(python=sys.executable)]
    if (
        payload.get("schema") != verification.SCHEMA
        or payload.get("execution_status") != "PASS"
        or payload.get("executed_gates") != required
        or payload.get("execution_scope") == "pytest_only"
        or payload.get("unexpected_skips") != []
        or payload.get("unexpected_warnings") != []
        or payload.get("inventory_sha256") != verification.inventory_digest(payload)
    ):
        raise ValueError("verification is not a complete all-gates PASS receipt")
    if any(payload.get(field) != seal[field] for field in ("source_commit", "source_tree")):
        raise ValueError("verification/source seal commit or tree mismatch")
    verification.verify_pytest_process_group_records(payload["pytest_paths"], payload["pytest_process_groups"])
    for field in ("immutable_hook_configuration", "private_path_scan", "validator_execution"):
        if payload.get(field, {}).get("status") != "PASS":
            raise ValueError(f"verification subreceipt is incomplete: {field}")
    validators = payload["validator_execution"]
    if validators.get("source_commit") != seal["source_commit"] or validators.get("results") != [
        {"path": p, "status": "PASS"} for p in payload["validator_paths"]
    ]:
        raise ValueError("standalone validator coverage is incomplete")
    execution = payload.get("pytest_execution", {})
    expected_groups = [group["group_id"] for group in payload["pytest_process_groups"]]
    if (
        execution.get("status") != "COMPLETE_EXACTLY_ONCE"
        or execution.get("executed_module_count") != len(payload["pytest_paths"])
        or execution.get("executed_group_count") != len(expected_groups)
        or execution.get("executed_group_ids") != expected_groups
    ):
        raise ValueError("verification pytest execution is incomplete")


def _check_qa(raw: bytes, documents: dict[str, bytes]) -> None:
    payload = json.loads(raw)
    if payload.get("status") != "PASS" or not isinstance(payload.get("documents"), list):
        raise ValueError("QA must contain a completed PASS and document inspection records")
    seen: set[str] = set()
    for row in payload["documents"]:
        path = _relative(row["path"])
        if path in seen or path not in documents:
            raise ValueError("QA document inventory differs from supplied documents")
        seen.add(path)
        data = documents[path]
        pages = row.get("pages")
        if (
            row.get("sha256") != _sha(data)
            or row.get("bytes") != len(data)
            or type(pages) is not int
            or pages < 1
            or row.get("inspected_pages") != list(range(1, pages + 1))
            or any(type(page) is not int for page in row["inspected_pages"])
        ):
            raise ValueError(f"QA does not bind complete page inspection: {path}")
    if seen != set(documents):
        raise ValueError("QA omits a supplied document")


def _verify_content(members: dict[str, bytes], manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA or manifest.get("scope") != "declared_release_inventory_not_full_git_history":
        raise ValueError("unsupported reviewer manifest")
    if manifest.get("artifacts") != _rows({k: v for k, v in members.items() if k != MANIFEST_MEMBER}):
        raise ValueError("outer archive inventory or member hash mismatch")
    bindings = manifest["bindings"]
    expected_outer = {
        "SOURCE_SEAL.json",
        "RELEASE_SHA256SUMS.txt",
        "EVIDENCE_MANIFEST.json",
        "NEXT_PHASE_EVIDENCE.zip",
        "VERIFICATION.json",
        "DOCUMENT_QA.json",
        "RELEASE_SOURCE.zip",
        MANIFEST_MEMBER,
    }
    if bindings.get("post_checksums") is not None:
        expected_outer.add("POST_CHECKSUM_SHA256SUMS.txt")
    for role in ("documents", "reports"):
        for name, relative in bindings[role].items():
            _relative(relative)
            if not name.startswith(role + "/") or len(PurePosixPath(_relative(name)).parts) != 2:
                raise ValueError("invalid named deliverable binding")
            expected_outer.add(name)
    if set(members) != expected_outer:
        raise ValueError("outer archive contains unbound members")
    seal = seals.validate_seal_document(json.loads(members["SOURCE_SEAL.json"]))
    if manifest.get("source_commit") != seal["source_commit"] or manifest.get("source_tree") != seal["source_tree"]:
        raise ValueError("outer/source seal identity mismatch")
    sealed = {row["path"]: row for row in seal["artifacts"]}
    if bindings["evidence_manifest"] not in sealed:
        raise ValueError("evidence manifest is not source-sealed")
    sums = _checksum_entries(members["RELEASE_SHA256SUMS.txt"], checksums.REQUIRED_RELEASE_PATHS)
    if bindings["seal"] not in sums or bindings["verification"] not in sums:
        raise ValueError("seal and final verification must be canonical checksummed payloads")
    expected = set(sealed) | set(sums) | {bindings["checksums"]}
    metadata = {
        bindings["seal"]: "SOURCE_SEAL.json",
        bindings["checksums"]: "RELEASE_SHA256SUMS.txt",
        bindings["evidence_manifest"]: "EVIDENCE_MANIFEST.json",
        bindings["verification"]: "VERIFICATION.json",
    }
    if len(metadata) != 4:
        raise ValueError("metadata bindings must have distinct paths")
    if bindings.get("post_checksums") is not None:
        post = _checksum_entries(members["POST_CHECKSUM_SHA256SUMS.txt"], checksums.POST_CHECKSUM_REQUIRED_PATHS)
        if set(sums) & set(post):
            raise ValueError("overlapping checksum inventories")
        sums.update(post)
        expected.update(post)
        expected.add(bindings["post_checksums"])
        metadata[bindings["post_checksums"]] = "POST_CHECKSUM_SHA256SUMS.txt"
    source = _zip_members(members["RELEASE_SOURCE.zip"])
    if set(source) != expected:
        raise ValueError("source archive differs from seal/checksum inventory union")
    for name, row in sealed.items():
        data = source[name]
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
        if len(data) != row["bytes"] or _sha(data) != row["sha256"] or blob != row["git_blob"]:
            raise ValueError(f"source seal hash/blob mismatch: {name}")
    for name, digest in sums.items():
        if _sha(source[name]) != digest:
            raise ValueError(f"archived canonical checksum mismatch: {name}")
    for relative, member in metadata.items():
        if source.get(relative) != members[member]:
            raise ValueError(f"outer/source metadata crossbinding mismatch: {relative}")
    evidence_manifest = evidence.validate_manifest(json.loads(members["EVIDENCE_MANIFEST.json"]))
    evidence_members = _zip_members(members["NEXT_PHASE_EVIDENCE.zip"])
    expected_evidence = {row["path"] for row in evidence_manifest["artifacts"]} | {evidence.MANIFEST_MEMBER}
    if set(evidence_members) != expected_evidence:
        raise ValueError("evidence archive inventory differs")
    if evidence_members[evidence.MANIFEST_MEMBER] != members["EVIDENCE_MANIFEST.json"]:
        raise ValueError("archived evidence manifest differs")
    for row in evidence_manifest["artifacts"]:
        data = evidence_members[row["path"]]
        if len(data) != row["bytes"] or _sha(data) != row["sha256"]:
            raise ValueError(f"archived evidence hash mismatch: {row['path']}")
    _check_verification(members["VERIFICATION.json"], seal)
    documents = {relative: members[member] for member, relative in bindings["documents"].items()}
    if len(documents) != len(bindings["documents"]) or not documents:
        raise ValueError("document paths must be nonempty and unique")
    if {PurePosixPath(path).suffix.lower() for path in documents} != {".pdf", ".docx"}:
        raise ValueError("reviewer package must contain inspected PDF and DOCX documents")
    for relative, data in documents.items():
        if relative in source and source[relative] != data:
            raise ValueError("outer/source document crossbinding mismatch")
    _check_qa(members["DOCUMENT_QA.json"], documents)


def verify_archive_bytes(raw: bytes) -> dict[str, Any]:
    members = _zip_members(raw)
    manifest: dict[str, Any] = json.loads(members[MANIFEST_MEMBER])
    if not isinstance(manifest, dict):
        raise ValueError("outer manifest must be a JSON object")
    if _canonical(manifest) != members[MANIFEST_MEMBER]:
        raise ValueError("outer manifest is not canonical")
    _verify_content(members, manifest)
    return manifest


def build_package(
    *,
    repo: Path,
    seal: Path,
    checksums_path: Path,
    evidence_manifest: Path,
    evidence_archive: Path,
    verification_path: Path,
    qa: Path,
    documents: dict[str, Path],
    reports: dict[str, Path],
    output_directory: Path,
    post_checksums: Path | None = None,
) -> None:
    repo, output_directory = repo.absolute(), output_directory.absolute()
    outer_zip = output_directory.with_name(output_directory.name + ".zip")
    receipt_path = output_directory.with_name(output_directory.name + ".zip.receipt.json")
    for path in (output_directory, outer_zip, receipt_path):
        _unlinked(path)
        if path.exists():
            raise FileExistsError(path)
    bindings: dict[str, Any] = {
        "seal": _repo_path(repo, seal),
        "checksums": _repo_path(repo, checksums_path),
        "evidence_manifest": _repo_path(repo, evidence_manifest),
        "verification": _repo_path(repo, verification_path),
        "post_checksums": _repo_path(repo, post_checksums) if post_checksums else None,
        "documents": {},
        "reports": {},
    }
    members = {
        "SOURCE_SEAL.json": _read(seal),
        "RELEASE_SHA256SUMS.txt": _read(checksums_path),
        "EVIDENCE_MANIFEST.json": _read(evidence_manifest),
        "NEXT_PHASE_EVIDENCE.zip": _read(evidence_archive),
        "VERIFICATION.json": _read(verification_path),
        "DOCUMENT_QA.json": _read(qa),
    }
    seal_payload = seals.validate_seal_document(json.loads(members["SOURCE_SEAL.json"]))
    source_paths = {row["path"] for row in seal_payload["artifacts"]}
    source_paths.update(_checksum_entries(members["RELEASE_SHA256SUMS.txt"], checksums.REQUIRED_RELEASE_PATHS))
    source_paths.add(bindings["checksums"])
    if post_checksums:
        members["POST_CHECKSUM_SHA256SUMS.txt"] = _read(post_checksums)
        source_paths.update(
            _checksum_entries(members["POST_CHECKSUM_SHA256SUMS.txt"], checksums.POST_CHECKSUM_REQUIRED_PATHS)
        )
        source_paths.add(bindings["post_checksums"])
    source = {name: _read(repo / _relative(name)) for name in sorted(source_paths)}
    members["RELEASE_SOURCE.zip"] = _zip_bytes(source)
    for role, values in (("documents", documents), ("reports", reports)):
        for name, path in values.items():
            name = _relative(name)
            if "/" in name:
                raise ValueError("deliverable names must be single filenames")
            member = role + "/" + name
            members[member] = _read(path)
            bindings[role][member] = _repo_path(repo, path)
    manifest = {
        "schema": SCHEMA,
        "scope": "declared_release_inventory_not_full_git_history",
        "distribution": "internal_reviewer_archive_with_original_provenance_paths",
        "source_commit": seal_payload["source_commit"],
        "source_tree": seal_payload["source_tree"],
        "bindings": bindings,
        "artifacts": _rows(members),
    }
    members[MANIFEST_MEMBER] = _canonical(manifest)
    raw = _zip_bytes(members)
    verify_archive_bytes(raw)
    receipt = {
        "schema": "kbound-next-phase-review-package-receipt-v1",
        "status": "PASS",
        "bytes": len(raw),
        "sha256": _sha(raw),
        "manifest_sha256": _sha(members[MANIFEST_MEMBER]),
        "source_commit": seal_payload["source_commit"],
        "scope": "byte_integrity_and_receipt_crossbindings",
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    # All validation precedes publication. Atomic create-only file publication
    # never overwrites an earlier release; a process interruption can leave a
    # visibly incomplete package, which verify_package rejects.
    with tempfile.TemporaryDirectory(prefix=".review-package-", dir=output_directory.parent) as temporary:
        stage = Path(temporary)
        for name, data in members.items():
            target = stage / "directory" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for name, data in (("archive", raw), ("receipt", _canonical(receipt))):
            with (stage / name).open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        output_directory.mkdir()
        for child in (stage / "directory").iterdir():
            shutil.move(str(child), output_directory / child.name)
        os.link(stage / "archive", outer_zip)
        os.link(stage / "receipt", receipt_path)
    verify_package(output_directory)


def verify_package(output_directory: Path) -> None:
    raw = _read(output_directory.with_name(output_directory.name + ".zip"))
    receipt = json.loads(_read(output_directory.with_name(output_directory.name + ".zip.receipt.json")))
    manifest = verify_archive_bytes(raw)
    if (
        receipt.get("schema") != "kbound-next-phase-review-package-receipt-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("bytes") != len(raw)
        or receipt.get("sha256") != _sha(raw)
        or receipt.get("manifest_sha256") != _sha(_canonical(manifest))
        or receipt.get("source_commit") != manifest["source_commit"]
    ):
        raise ValueError("external ZIP receipt mismatch")
    # Only the created reviewer directory is inventoried, never the repository.
    _unlinked(output_directory)
    actual: dict[str, bytes] = {}
    for parent, directories, filenames in os.walk(output_directory, followlinks=False):
        for name in directories:
            _unlinked(Path(parent) / name)
        for name in filenames:
            path = Path(parent) / name
            actual[path.relative_to(output_directory).as_posix()] = _read(path)
    if actual != _zip_members(raw):
        raise ValueError("reviewer directory and final ZIP differ")


def _named(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not path or name in result:
            raise ValueError("deliverable must have unique name=path")
        result[name] = Path(path).absolute()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    for name in ("seal", "checksums", "post-checksums", "evidence-manifest", "evidence-archive", "verification", "qa"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--document", action="append", default=[])
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify_package(args.output_directory)
    else:
        if any(
            getattr(args, name) is None
            for name in ("seal", "checksums", "evidence_manifest", "evidence_archive", "verification", "qa")
        ):
            parser.error("creation requires seal, checksums, evidence manifest/archive, verification, and QA")
        build_package(
            repo=args.repo,
            seal=args.seal,
            checksums_path=args.checksums,
            post_checksums=args.post_checksums,
            evidence_manifest=args.evidence_manifest,
            evidence_archive=args.evidence_archive,
            verification_path=args.verification,
            qa=args.qa,
            documents=_named(args.document),
            reports=_named(args.report),
            output_directory=args.output_directory,
        )
    print("reviewer package: PASS (byte integrity, source/evidence bindings, all-gates and QA receipts)")


if __name__ == "__main__":
    main()
