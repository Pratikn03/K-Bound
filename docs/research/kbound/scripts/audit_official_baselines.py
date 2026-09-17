#!/usr/bin/env python3
"""Source hashing and fail-closed verification of existing native artifacts.

The CLI verifies an existing schema-3 audit, physical evidence, and independent
witness signature; it never creates a benchmark or witness attestation. It writes
a separate verification report, preserving its input audit and historical logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cached_data",
    "log",
    "raw_logs",
    "public",
}
_EXCLUDED_SUFFIXES = {
    ".ckpt", ".npy", ".npz", ".pkl", ".pt", ".pth",
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str | None:
    """Hash source files while excluding model/data payloads and UI assets.

    A directory named ``dataset`` can contain the native preprocessing and
    sampling implementation (notably in POEM); it must not be excluded wholesale.
    Image bytes are authenticated separately, never by this source-tree digest.
    """

    if not root.is_dir():
        return None
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if path.suffix.lower() in _EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _repository_relative(path: Path, repo: Path) -> str:
    resolved_repo = repo.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_repo).as_posix()
    except ValueError as exc:
        raise ValueError(
            "audited path is outside the repository-root binding: "
            f"{resolved_path.name}"
        ) from exc


def repository_relative(path: Path, repo: Path) -> str:
    """Return a repository-relative POSIX path for native-run receipts.

    The native runner imports this helper as part of its provenance contract.
    Keep the checked implementation private so existing audit callers remain
    unchanged, while exposing a stable, typed entry point for the runner.
    """

    return _repository_relative(path, repo)


def native_logs(root: Path, *, repo: Path, method: str | None = None) -> dict[str, Any]:
    """Hash captured native logs and fail closed on missing/unreadable files."""

    del method  # compatibility argument; semantic checks live in verify_method_artifacts
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    hashes: dict[str, str] = {}
    unavailable: list[str] = []
    empty: list[str] = []
    failure_markers: list[str] = []
    for path in files:
        relative = _repository_relative(path, repo)
        try:
            hashes[relative] = sha256_file(path)
            size = path.stat().st_size
        except OSError:
            unavailable.append(relative)
            continue
        if size == 0:
            empty.append(relative)
        if path.suffix.lower() in {".txt", ".log", ".json", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                unavailable.append(relative)
                hashes.pop(relative, None)
                continue
            if (
                "Traceback (most recent call last)" in text
                or "RuntimeError:" in text
                or "FATAL:" in text
            ):
                failure_markers.append(relative)
    return {
        "count": len(files),
        "sha256": hashes,
        "unavailable": sorted(set(unavailable)),
        "empty": sorted(set(empty)),
        "failure_markers": sorted(failure_markers),
        "successful": bool(files) and not unavailable and not empty and not failure_markers,
    }


def _checked_path(path: Path, repo: Path, *, regular_file: bool = False) -> Path:
    """Keep audit reads/writes inside the selected repository, without symlinks."""
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(repo)
    except ValueError as exc:
        raise ValueError("audit path is outside the repository") from exc
    current = repo
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"audit path contains a symlink: {relative}")
    if regular_file and not path.is_file():
        raise ValueError(f"required audit input missing: {relative}")
    return path


def verify_method_artifacts(
    *, repo: Path, out_dir: Path, audit_path: Path, method: str,
    stream: Path, environment: Path, toolchain: Path,
) -> dict[str, Any]:
    """Revalidate existing evidence; never manufacture an official attestation.

    This is an artifact-verification command, not a native benchmark executor or
    a converter. The schema-3 audit and its witness must already exist.
    """
    from official_baseline_provenance import load_json_strict, validate_promotable_audit

    result: dict[str, Any] = {"official_label_allowed": False, "blockers": []}
    try:
        paths = {
            "audit": audit_path,
            "decisions": out_dir / f"{method}_decisions.json",
            "locked_stream": stream,
            "environment_receipt": environment,
            "toolchain_receipt": toolchain,
        }
        initial: dict[Path, str] = {}
        for name, path in paths.items():
            checked = _checked_path(path, repo, regular_file=True)
            initial[checked] = sha256_file(checked)
            result.setdefault("input_sha256", {})[name] = initial[checked]
        audit = load_json_strict(audit_path)
        if not isinstance(audit, dict):
            raise ValueError("official audit must be an object")
        record = audit.get("methods", {}).get(method)
        if not isinstance(record, dict):
            raise ValueError(f"official audit has no record for {method}")
        source = _checked_path(repo / "external" / f"{method}_official", repo)
        current_source_hash = tree_hash(source)
        if current_source_hash is None or record.get("source_tree_sha256") != current_source_hash:
            raise ValueError("current native source tree does not match the audited execution")
        logs = record.get("native_logs", {})
        recorded_logs = logs.get("sha256") if isinstance(logs, dict) else None
        if not isinstance(recorded_logs, dict) or not recorded_logs:
            raise ValueError("native artifact hashes are absent")
        control_fields = {
            "native_invocation.json": "invocation",
            "native_runner_receipt.json": "runner_receipt",
            "native_completion.json": "completion",
            "native_execution_attestation.json": "execution_attestation",
        }
        physical_controls: dict[str, Path] = {}
        for relative, digest in recorded_logs.items():
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("native artifact path must be repository-relative")
            path = _checked_path(repo / relative, repo, regular_file=True)
            if not path.is_relative_to(out_dir):
                raise ValueError("native artifact is outside the selected run output")
            actual = sha256_file(path)
            if actual != digest:
                raise ValueError(f"native artifact content changed: {relative}")
            initial[path] = actual
            if path.name in control_fields:
                field = control_fields[path.name]
                if field in physical_controls or load_json_strict(path) != logs.get(field):
                    raise ValueError(f"physical native {field} disagrees with authenticated audit record")
                physical_controls[field] = path
        if set(physical_controls) != set(control_fields.values()):
            raise ValueError("physical native invocation, receipt, completion, or attestation is missing")
        if (initial[physical_controls["invocation"]] != logs["runner_receipt"].get("invocation_sha256")
                or initial[physical_controls["runner_receipt"]] != logs["completion"].get("runner_receipt_sha256")):
            raise ValueError("physical native control bytes do not match the signed execution chain")
        payload = load_json_strict(paths["decisions"])
        if not isinstance(payload, dict) or payload.get("method") != method:
            raise ValueError("converted decision method identity is missing or mismatched")
        if payload.get("provenance_audit_sha256") != initial[audit_path]:
            raise ValueError("converted decisions are bound to a different provenance audit")
        decisions = payload.get("decisions")
        if (not isinstance(decisions, dict) or not decisions
                or any(not key or value not in {"adapt", "freeze", "abstain"}
                       for key, value in decisions.items())):
            raise ValueError("converted decisions are malformed")
        stream_payload = load_json_strict(stream)
        records = stream_payload.get("records") if isinstance(stream_payload, dict) else None
        if not isinstance(records, list) or not records:
            raise ValueError("locked stream records are absent")
        conditions = [item.get("condition") if isinstance(item, dict) else None for item in records]
        if (any(not isinstance(item, str) or not item for item in conditions)
                or len(set(conditions)) != len(conditions) or set(conditions) != set(decisions)):
            raise ValueError("converted decisions do not match the exact unique locked conditions")
        bindings = {
            "locked_stream_sha256": initial[stream],
            "environment_receipt_sha256": initial[environment],
            "toolchain_receipt_sha256": initial[toolchain],
        }
        if any(payload.get(key) != value for key, value in bindings.items()):
            raise ValueError("converted decisions do not bind the supplied stream/runtime receipts")
        verified_audit_hash = validate_promotable_audit(
            audit_path, method=method, decisions=decisions,
            source_log_sha256=payload.get("source_log_sha256"), **bindings,
        )
        if verified_audit_hash != initial[audit_path]:
            raise ValueError("official audit changed during verification")
        if any(sha256_file(path) != digest for path, digest in initial.items()):
            raise ValueError("evidence changed during verification")
        if tree_hash(source) != current_source_hash:
            raise ValueError("native source changed during verification")
        result.update(official_label_allowed=True, decision_count=len(decisions),
                      source_tree_sha256=current_source_hash)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        result["blockers"].append(str(exc))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify existing official AETTA/POEM evidence; does not run or attest a benchmark.",
        allow_abbrev=False,
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="existing schema-3 OFFICIAL_BASELINE_AUDIT.json")
    parser.add_argument("--output", type=Path, help="fresh verification report, never the source audit")
    parser.add_argument("--stream", type=Path)
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--toolchain-receipt", type=Path)
    parser.add_argument("--method", choices=["aetta", "poem"], action="append")
    parser.add_argument("--require-promotable", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        if not repo.is_dir():
            raise ValueError("audited repository is missing")
        out_dir = _checked_path(args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir, repo)
        def selected(value: Path | None, default: str) -> Path:
            path = value if value is not None else out_dir / default
            return _checked_path(path if path.is_absolute() else repo / path, repo)
        output = selected(args.output, "OFFICIAL_BASELINE_VERIFICATION.json")
        if output.exists():
            raise ValueError("verification output already exists; choose a fresh --output")
        audit = selected(args.audit, "OFFICIAL_BASELINE_AUDIT.json")
        if output == audit:
            raise ValueError("verification output cannot replace the input audit")
        methods = {
            method: verify_method_artifacts(
                repo=repo, out_dir=out_dir, audit_path=audit, method=method,
                stream=selected(args.stream, "locked_stream.json"),
                environment=selected(args.environment_receipt, "environment_receipt.json"),
                toolchain=selected(args.toolchain_receipt, "toolchain_receipt.json"),
            ) for method in sorted(set(args.method or ["aetta", "poem"]))
        }
        allowed = all(record["official_label_allowed"] for record in methods.values())
        report = {
            "schema": "kbound-official-baseline-verification-v1",
            "status": "PASS" if allowed else "OPEN", "official_label_allowed": allowed,
            "scope": "Existing AETTA/POEM artifacts only; not TTA/ALine or common-panel completeness.",
            "native_execution_launched": False, "benchmark_complete": False,
            "methods": methods,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 2 if args.require_promotable and not allowed else 0
    except (OSError, ValueError) as exc:
        print(f"official baseline audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
