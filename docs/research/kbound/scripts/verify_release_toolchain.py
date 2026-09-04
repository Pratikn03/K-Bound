#!/usr/bin/env python3
"""Verify the non-Python K-Bound publication toolchain against exact bytes.

The public lock contains logical command names, first-line version identities,
and SHA-256 digests, never machine-local paths.  Operators may point a command
at its pinned executable with ``KBOUND_TOOL_<NAME>``; otherwise it is resolved
from ``PATH``.  The emitted receipt likewise omits private paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zlib
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "kbound-release-toolchain-lock-v1"
RECEIPT_SCHEMA = "kbound-release-toolchain-receipt-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
PERL_SCRIPT_TOOLS = frozenset({"latexmk", "latexpand"})
CANONICAL_TOOL_NAMES = frozenset(
    {
        "latexmk",
        "latexpand",
        "pandoc",
        "perl",
        "pdfdetach",
        "pdfinfo",
        "pdflatex",
        "pdftoppm",
        "pdftotext",
        "soffice",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def runtime_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-amd64"
    return f"{system}-{machine}"


def _strict_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {token}")),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read toolchain lock {path}: {exc}") from exc
    if raw != canonical_json_bytes(value):
        raise ValueError(f"toolchain lock is not canonical JSON: {path}")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def load_profile(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    document = _strict_json(resolved)
    if not isinstance(document, dict) or set(document) != {
        "platform",
        "python_zlib",
        "schema",
        "tools",
    }:
        raise ValueError("toolchain lock must contain only the declared top-level fields")
    if document.get("schema") != SCHEMA:
        raise ValueError("toolchain lock has an unknown schema")
    if not isinstance(document.get("platform"), str) or not document["platform"]:
        raise ValueError("toolchain lock platform must be non-empty")
    zlib_identity = document.get("python_zlib")
    if (
        not isinstance(zlib_identity, dict)
        or set(zlib_identity) != {"compile", "runtime"}
        or not all(
            isinstance(zlib_identity.get(field), str) and zlib_identity[field] for field in ("compile", "runtime")
        )
    ):
        raise ValueError("toolchain lock has an invalid Python zlib identity")
    tools = document.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise ValueError("toolchain lock must declare at least one tool")
    for name, record in tools.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError("toolchain lock contains an invalid logical tool name")
        if not isinstance(record, dict) or set(record) != {
            "command",
            "sha256",
            "version_args",
            "version_line",
            "version_returncode",
        }:
            raise ValueError(f"toolchain lock entry {name} has an invalid schema")
        command = record.get("command")
        if not isinstance(command, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", command):
            raise ValueError(f"toolchain lock entry {name} has an invalid command")
        if not isinstance(record.get("sha256"), str) or _SHA256_RE.fullmatch(record["sha256"]) is None:
            raise ValueError(f"toolchain lock entry {name} has an invalid SHA-256")
        args = record.get("version_args")
        if (
            not isinstance(args, list)
            or not args
            or not all(isinstance(value, str) and "\x00" not in value for value in args)
        ):
            raise ValueError(f"toolchain lock entry {name} has invalid version arguments")
        version_line = record.get("version_line")
        if (
            not isinstance(version_line, str)
            or not version_line
            or any(character in version_line for character in "\r\n\x00")
        ):
            raise ValueError(f"toolchain lock entry {name} has an invalid version line")
        returncode = record.get("version_returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int) or not 0 <= returncode <= 255:
            raise ValueError(f"toolchain lock entry {name} has an invalid version return code")
    return document


def require_canonical_tool_profile(profile: Mapping[str, Any]) -> None:
    """Reject profiles that cannot authorize a public release receipt."""

    tools = profile.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != CANONICAL_TOOL_NAMES:
        raise ValueError("release toolchain profile must contain exactly the canonical ten tools")


def _tool_override_name(name: str) -> str:
    return "KBOUND_TOOL_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _resolve_tool(name: str, command: str) -> Path:
    override = os.environ.get(_tool_override_name(name))
    candidate = override if override else shutil.which(command)
    if not candidate:
        raise ValueError(f"required tool {name} ({command}) is unavailable")
    unresolved = Path(candidate).expanduser()
    if override and (not unresolved.is_absolute() or unresolved.is_symlink()):
        raise ValueError(f"required tool override {_tool_override_name(name)} must be an absolute non-symlink realpath")
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"required tool {name} cannot be resolved: {exc}") from exc
    if override and (unresolved != resolved):
        raise ValueError(f"required tool override {_tool_override_name(name)} must be an absolute non-symlink realpath")
    if any(character in str(resolved) for character in "\r\n\t\x00"):
        raise ValueError(f"required tool {name} resolves to a path with control characters")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"required tool {name} does not resolve to a regular executable")
    if not os.access(resolved, os.X_OK):
        raise ValueError(f"required tool {name} is not executable")
    return resolved


def _version_probe(command: Sequence[str], args: Sequence[str], *, name: str) -> tuple[str, int]:
    try:
        completed = subprocess.run(
            [*command, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"required tool {name} version probe failed: {exc}") from exc
    lines = [line.strip() for line in (completed.stdout + completed.stderr).splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"required tool {name} emitted no version identity")
    return lines[0], completed.returncode


def build_receipt(
    profile: Mapping[str, Any],
    *,
    profile_path: Path,
    runtime_platform: str,
    zlib_compile: str,
    zlib_runtime: str,
    resolved_tools: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    profile_digest = sha256_file(profile_path)
    expected_platform = profile.get("platform")
    if runtime_platform != expected_platform:
        raise ValueError(f"toolchain platform mismatch: expected {expected_platform}, got {runtime_platform}")
    expected_zlib = profile.get("python_zlib")
    if not isinstance(expected_zlib, Mapping):
        raise ValueError("toolchain lock has no Python zlib identity")
    if zlib_compile != expected_zlib.get("compile"):
        raise ValueError(f"zlib compile version mismatch: expected {expected_zlib.get('compile')}, got {zlib_compile}")
    if zlib_runtime != expected_zlib.get("runtime"):
        raise ValueError(f"zlib runtime version mismatch: expected {expected_zlib.get('runtime')}, got {zlib_runtime}")
    tools = profile.get("tools")
    if not isinstance(tools, Mapping) or not tools:
        raise ValueError("toolchain lock has no tools")
    if resolved_tools is not None:
        resolved_tools.clear()
    executables: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for name in sorted(tools):
        record = tools[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"toolchain lock entry {name} is invalid")
        executable = _resolve_tool(name, str(record.get("command", "")))
        digest = sha256_file(executable)
        if digest != record.get("sha256"):
            raise ValueError(f"required tool {name} SHA-256 mismatch: expected {record.get('sha256')}, got {digest}")
        executables[name] = executable
        digests[name] = digest
    if PERL_SCRIPT_TOOLS & executables.keys() and "perl" not in executables:
        raise ValueError("toolchain lock must bind perl for latexmk and latexpand")

    observed: dict[str, dict[str, str | int]] = {}
    for name in sorted(tools):
        record = tools[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"toolchain lock entry {name} is invalid")
        executable = executables[name]
        command = [str(executables["perl"]), str(executable)] if name in PERL_SCRIPT_TOOLS else [str(executable)]
        version, version_returncode = _version_probe(
            command,
            list(record.get("version_args", [])),
            name=name,
        )
        if version_returncode != record.get("version_returncode"):
            raise ValueError(
                f"required tool {name} version return code mismatch: "
                f"expected {record.get('version_returncode')}, got {version_returncode}"
            )
        if version != record.get("version_line"):
            raise ValueError(
                f"required tool {name} version mismatch: expected {record.get('version_line')!r}, got {version!r}"
            )
        post_probe_digest = sha256_file(executable)
        if post_probe_digest != digests[name]:
            raise ValueError(f"required tool {name} changed during verification")
        observed[name] = {
            "command": str(record["command"]),
            "sha256": digests[name],
            "version_line": version,
            "version_returncode": version_returncode,
        }
        if resolved_tools is not None:
            resolved_tools[_tool_override_name(name)] = str(executable)
    if sha256_file(profile_path) != profile_digest:
        raise ValueError("toolchain lock changed during verification")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "verified",
        "profile": {
            "path": profile_path.name,
            "sha256": profile_digest,
        },
        "runtime": {
            "platform": runtime_platform,
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "zlib_compile": zlib_compile,
            "zlib_runtime": zlib_runtime,
        },
        "tools": observed,
    }


def write_receipt_atomic(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink receipt destination: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(canonical_json_bytes(dict(receipt)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_resolved_tools_atomic(path: Path, resolved_tools: Mapping[str, str]) -> None:
    """Write private verified command paths for the release driver to import.

    Unlike the publication receipt, this short-lived file contains local paths.
    It is therefore owner-readable only and is consumed without ``eval``.
    """

    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink resolved-tools destination: {path}")
    lines: list[str] = []
    for variable, value in sorted(resolved_tools.items()):
        if re.fullmatch(r"KBOUND_TOOL_[A-Z0-9_]+", variable) is None:
            raise ValueError(f"invalid resolved-tool environment name: {variable}")
        tool = Path(value)
        try:
            real_tool = tool.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"invalid resolved-tool realpath for {variable}") from exc
        if (
            not tool.is_absolute()
            or tool.is_symlink()
            or tool != real_tool
            or not tool.is_file()
            or not os.access(tool, os.X_OK)
            or any(character in value for character in "\r\n\t\x00")
        ):
            raise ValueError(f"invalid resolved-tool realpath for {variable}")
        lines.append(f"{variable}\t{value}\n")
    if not lines:
        raise ValueError("resolved-tool environment is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = None
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def clear_receipt_output(path: Path) -> None:
    """Remove a stale public receipt before attempting a fresh verification."""

    if path.is_symlink():
        raise ValueError(f"refusing to remove symlink receipt destination: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"receipt destination is not a regular file: {path}")
        path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resolved-tools-output",
        type=Path,
        help="private, ephemeral TSV of verified KBOUND_TOOL_* realpaths",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output is not None:
            clear_receipt_output(args.output)
        profile = load_profile(args.profile)
        require_canonical_tool_profile(profile)
        resolved_tools: dict[str, str] = {}
        receipt = build_receipt(
            profile,
            profile_path=args.profile,
            runtime_platform=runtime_platform(),
            zlib_compile=zlib.ZLIB_VERSION,
            zlib_runtime=zlib.ZLIB_RUNTIME_VERSION,
            resolved_tools=resolved_tools,
        )
        if args.output is None:
            sys.stdout.buffer.write(canonical_json_bytes(receipt))
        else:
            write_receipt_atomic(args.output, receipt)
        if args.resolved_tools_output is not None:
            write_resolved_tools_atomic(args.resolved_tools_output, resolved_tools)
    except (OSError, ValueError) as exc:
        print(f"release toolchain verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
