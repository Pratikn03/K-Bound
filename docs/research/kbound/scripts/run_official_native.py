#!/usr/bin/env python3
"""Execute an official baseline command and emit sealed native-run evidence.

This runner records a locally consistent trace of stdout, stderr, and a zero
exit status.  It intentionally cannot make an official claim: promotion also
requires a separate attestation from the pinned independent witness key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_official_baselines import repository_relative, sha256_file, tree_hash  # noqa: E402
from official_baseline_provenance import NATIVE_TRACE_CONTROL_FILENAMES  # noqa: E402

COMPLETION_SCHEMA = "kbound-official-native-completion-v3"
INVOCATION_SCHEMA = "kbound-official-native-invocation-v1"
RUNNER_RECEIPT_SCHEMA = "kbound-official-native-runner-receipt-v2"
PRODUCER_RELATIVE_PATH = "docs/research/kbound/scripts/run_official_native.py"
CONTROL_FILENAMES = NATIVE_TRACE_CONTROL_FILENAMES
METHOD_SOURCES = {"aetta": Path("AETTA"), "poem": Path("external/poem")}
METHOD_OUTPUTS = {"aetta": "aetta_native", "poem": "poem_imagenetc"}


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _destination(repo: Path, out_dir: Path, method: str) -> Path:
    target = (out_dir / METHOD_OUTPUTS[method]).absolute()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError("native evidence output must be inside the audited repository") from exc
    if target.exists() or target.is_symlink():
        raise ValueError(f"native evidence output already exists: {target}")
    return target


def _artifact_hashes(root: Path, *, published_root: Path, repo: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in CONTROL_FILENAMES:
            continue
        rows[repository_relative(published_root / path.relative_to(root), repo)] = sha256_file(path)
    return rows


def execute_native_method(*, repo: Path, out_dir: Path, method: str, command: Sequence[str]) -> tuple[int, Path]:
    """Run one command in the method's official source tree and capture evidence."""

    if method not in METHOD_SOURCES:
        raise ValueError(f"unsupported official native method: {method}")
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("official native command must be a non-empty argv vector")
    repository = repo.resolve()
    source = repository / METHOD_SOURCES[method]
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"official native source is missing or a symlink: {METHOD_SOURCES[method]}")
    producer = Path(__file__).resolve()
    if not producer.is_file() or producer.is_symlink():
        raise ValueError("maintained native runner source is unavailable")
    source_tree_sha256 = tree_hash(source)
    if source_tree_sha256 is None:
        raise ValueError("official native source cannot be hashed")
    target = _destination(repository, out_dir, method)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    returncode = 127
    try:
        stdout = temporary / "native_stdout.log"
        stderr = temporary / "native_stderr.log"
        with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
            try:
                completed = subprocess.run(
                    list(command), cwd=source, stdout=stdout_handle, stderr=stderr_handle, check=False
                )
                returncode = completed.returncode
            except OSError as exc:
                stderr_handle.write(f"runner execution error: {exc}\n".encode("utf-8", errors="replace"))
        invocation = {
            "command": list(command),
            "method": method,
            "producer": {"path": PRODUCER_RELATIVE_PATH, "sha256": sha256_file(producer)},
            "schema": INVOCATION_SCHEMA,
            "source": {
                "path": METHOD_SOURCES[method].as_posix(),
                "tree_sha256": source_tree_sha256,
            },
            "working_directory": repository_relative(source, repository),
        }
        invocation_bytes = canonical_json_bytes(invocation)
        (temporary / "native_invocation.json").write_bytes(invocation_bytes)
        # Receipt eligibility binds the source that was actually launched.  A
        # zero-exit command that rewrites tracked source cannot rewrite the
        # provenance it later asks the auditor to trust.
        if returncode == 0 and tree_hash(source) == source_tree_sha256:
            artifacts = _artifact_hashes(temporary, published_root=target, repo=repository)
            runner_receipt = {
                "artifacts_sha256": artifacts,
                "invocation_sha256": _sha256_bytes(invocation_bytes),
                "method": method,
                "producer": invocation["producer"],
                "returncode": 0,
                "schema": RUNNER_RECEIPT_SCHEMA,
                "status": "zero-exit-recorded",
            }
            runner_bytes = canonical_json_bytes(runner_receipt)
            (temporary / "native_runner_receipt.json").write_bytes(runner_bytes)
            completion = {
                "exit_status": 0,
                "log_sha256": artifacts,
                "method": method,
                "runner_receipt_sha256": _sha256_bytes(runner_bytes),
                "schema": COMPLETION_SCHEMA,
                "status": "runner-zero-exit-recorded",
            }
            _write_canonical(temporary / "native_completion.json", completion)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return returncode, target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_SOURCES), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command argv after --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        returncode, output = execute_native_method(
            repo=args.repo, out_dir=args.out_dir, method=args.method, command=command
        )
    except (OSError, ValueError) as exc:
        print(f"official native runner failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"method": args.method, "output": str(output), "returncode": returncode}, sort_keys=True))
    return returncode if returncode != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
