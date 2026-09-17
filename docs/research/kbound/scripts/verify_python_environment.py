#!/usr/bin/env python3
"""Verify an installed Python environment against a K-Bound hash lock.

The verifier is intentionally independent of pip and uv.  It validates the
lock grammar, checks the running Python/platform target, compares every
installed distribution, and emits a canonical JSON receipt bound to the lock
bytes by SHA-256.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform as platform_module
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

_HEADER_FIELDS = {
    "python": "Python",
    "platform": "Platform",
    "exclude_newer": "Exclude-Newer",
    "resolver": "Resolver",
    "only_binary": "Only-Binary",
}
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)")
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;,\\]+)(.*)$")
_BOOTSTRAP_DISTRIBUTIONS: tuple[str, ...] = ("pip", "setuptools", "wheel")
_CONTENT_PROFILE_SCHEMA = "kbound-python-environment-content-v1"


class EnvironmentLock(NamedTuple):
    path: Path
    python: str
    platform: str
    exclude_newer: str
    resolver: str
    only_binary: str
    packages: dict[str, str]
    hashes: dict[str, tuple[str, ...]]


def normalize_name(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_values(text: str, path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, label in _HEADER_FIELDS.items():
        matches = re.findall(rf"^# {re.escape(label)}:\s*(\S.*?)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"{path}: expected exactly one '# {label}:' header")
        values[key] = matches[0]
    if values["exclude_newer"] != "2026-09-01T00:00:00Z":
        raise ValueError(f"{path}: Exclude-Newer must be 2026-09-01T00:00:00Z")
    if values["only_binary"] != ":all:":
        raise ValueError(f"{path}: Only-Binary must be :all:")
    if not re.fullmatch(r"uv \d+\.\d+\.\d+(?:\s+\([^)]*\))?", values["resolver"]):
        raise ValueError(f"{path}: Resolver must identify an exact uv version")
    return values


def _logical_requirements(text: str, path: Path) -> list[str]:
    requirements: list[str] = []
    pending: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continued = stripped.endswith("\\")
        token = stripped[:-1].rstrip() if continued else stripped
        if token.startswith("-") and not pending:
            raise ValueError(f"{path}:{number}: lock option/include lines are forbidden")
        pending.append(token)
        if not continued:
            requirements.append(" ".join(pending))
            pending = []
    if pending:
        raise ValueError(f"{path}: truncated line continuation")
    return requirements


def load_lock(path: str | Path) -> EnvironmentLock:
    """Parse and validate one platform-specific, hash-complete lock."""
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read lock {resolved}: {exc}") from exc
    values = _header_values(text, resolved)
    packages: dict[str, str] = {}
    hashes: dict[str, tuple[str, ...]] = {}
    for requirement in _logical_requirements(text, resolved):
        if " @ " in requirement or re.search(r"(?:https?|file|git\+)://", requirement):
            raise ValueError(f"{resolved}: direct URL requirement is forbidden: {requirement!r}")
        match = _PIN_RE.fullmatch(requirement)
        if match is None:
            raise ValueError(f"{resolved}: requirement must be exactly pinned with ==: {requirement!r}")
        raw_name, version, remainder = match.groups()
        name = normalize_name(raw_name)
        if name in packages:
            raise ValueError(f"{resolved}: duplicate normalized package name: {name}")
        found_hashes = tuple(_HASH_RE.findall(remainder))
        if not found_hashes:
            raise ValueError(f"{resolved}: {name} has no complete SHA-256 hash")
        without_hashes = _HASH_RE.sub("", remainder).strip()
        if without_hashes:
            raise ValueError(f"{resolved}: unsupported or mutable requirement syntax: {requirement!r}")
        if len(found_hashes) != len(set(found_hashes)):
            raise ValueError(f"{resolved}: {name} repeats a SHA-256 hash")
        packages[name] = version
        hashes[name] = found_hashes
    if not packages:
        raise ValueError(f"{resolved}: lock contains no requirements")
    return EnvironmentLock(
        path=resolved,
        python=values["python"],
        platform=values["platform"],
        exclude_newer=values["exclude_newer"],
        resolver=values["resolver"],
        only_binary=values["only_binary"],
        packages=dict(sorted(packages.items())),
        hashes=dict(sorted(hashes.items())),
    )


def runtime_platform() -> str:
    system = platform_module.system().lower()
    machine = platform_module.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-amd64"
    return f"{system}-{machine}"


def runtime_python_version() -> str:
    return ".".join(str(value) for value in sys.version_info[:3])


def verify_runtime(lock: EnvironmentLock, *, python_version: str, platform: str) -> None:
    wanted_python = lock.python.split(".")
    actual_python = python_version.split(".")
    if actual_python[: len(wanted_python)] != wanted_python:
        raise ValueError(f"Python target mismatch: lock requires {lock.python}, runtime is {python_version}")
    if platform != lock.platform:
        raise ValueError(f"platform target mismatch: lock requires {lock.platform}, runtime is {platform}")


def installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = normalize_name(raw_name)
        version = distribution.version
        previous = installed.setdefault(name, version)
        if previous != version:
            raise ValueError(f"installed distribution {name} has conflicting versions: {previous}, {version}")
    return dict(sorted(installed.items()))


def _sha256_urlsafe(path: Path) -> str:
    raw = bytes.fromhex(sha256_file(path))
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _resident_regular_file(path: Path, *, prefix: Path, label: str) -> tuple[Path, str]:
    """Resolve one installed file without accepting a symlink or prefix escape."""

    prefix_absolute = Path(os.path.abspath(prefix))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(prefix_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the running Python prefix: {path}") from exc
    cursor = prefix_absolute
    for component in relative.parts:
        cursor = cursor / component
        try:
            mode = os.lstat(cursor).st_mode
        except OSError as exc:
            raise ValueError(f"cannot stat installed file {label}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"installed file {label} traverses a symlink: {cursor}")
    if not stat.S_ISREG(os.stat(path_absolute).st_mode):
        raise ValueError(f"installed file {label} is not a regular file")
    try:
        path_absolute.resolve(strict=True).relative_to(prefix_absolute.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f"installed file {label} escapes the running Python prefix") from exc
    return path_absolute, relative.as_posix()


def verify_distribution_contents(
    distributions: Sequence[importlib.metadata.Distribution],
    *,
    expected: Mapping[str, str],
    prefix: Path,
) -> dict[str, dict[str, object]]:
    """Verify installed wheel RECORD rows and return hashable content evidence.

    Every expected distribution must have one identity, one RECORD, and a
    SHA-256/size entry for every non-RECORD file. The resulting tree digest also
    includes the current RECORD bytes, so a checked-in content profile detects
    changes to either installed payloads or their metadata.
    """

    normalized_expected = {normalize_name(name): version for name, version in expected.items()}
    selected: dict[str, importlib.metadata.Distribution] = {}
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = normalize_name(raw_name)
        if name not in normalized_expected:
            continue
        if name in selected:
            raise ValueError(f"duplicate installed distribution identity: {name}")
        selected[name] = distribution

    missing = sorted(set(normalized_expected) - set(selected))
    if missing:
        raise ValueError("missing installed distribution content: " + ", ".join(missing))

    evidence: dict[str, dict[str, object]] = {}
    for name in sorted(selected):
        distribution = selected[name]
        if distribution.version != normalized_expected[name]:
            raise ValueError(
                f"installed distribution {name} content version mismatch: "
                f"expected {normalized_expected[name]} got {distribution.version}"
            )
        entries = distribution.files
        if not entries:
            raise ValueError(f"installed distribution {name} has no RECORD inventory")
        seen_paths: set[str] = set()
        record_count = 0
        rows: list[tuple[str, int, str]] = []
        for entry in entries:
            label = f"{name}:{entry}"
            located, relative = _resident_regular_file(
                Path(str(distribution.locate_file(entry))), prefix=prefix, label=label
            )
            if relative in seen_paths:
                raise ValueError(f"installed distribution {name} repeats RECORD path: {relative}")
            seen_paths.add(relative)
            size = located.stat().st_size
            is_record = relative.endswith(".dist-info/RECORD") and entry.hash is None and entry.size is None
            if is_record:
                record_count += 1
                digest = sha256_file(located)
            else:
                if entry.hash is None or entry.hash.mode != "sha256":
                    raise ValueError(f"installed file {label} is missing a sha256 RECORD hash")
                if entry.size is None or isinstance(entry.size, bool) or entry.size < 0:
                    raise ValueError(f"installed file {label} is missing a valid RECORD size")
                if size != entry.size:
                    raise ValueError(f"RECORD size mismatch for {label}: expected {entry.size}, got {size}")
                actual_encoded = _sha256_urlsafe(located)
                if actual_encoded != entry.hash.value:
                    raise ValueError(f"RECORD hash mismatch for {label}")
                digest = sha256_file(located)
            rows.append((relative, size, digest))
        if record_count != 1:
            raise ValueError(f"installed distribution {name} must contain exactly one RECORD; got {record_count}")
        tree = hashlib.sha256()
        for relative, size, digest in sorted(rows):
            encoded = relative.encode("utf-8")
            tree.update(len(encoded).to_bytes(8, "big"))
            tree.update(encoded)
            tree.update(size.to_bytes(8, "big"))
            tree.update(bytes.fromhex(digest))
        evidence[name] = {
            "version": distribution.version,
            "file_count": len(rows),
            "tree_sha256": tree.hexdigest(),
        }
    return evidence


def build_content_profile(
    *,
    lock_sha256: str,
    python_version: str,
    platform: str,
    evidence: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    distributions = {name: dict(value) for name, value in sorted(evidence.items())}
    aggregate = hashlib.sha256(
        json.dumps(distributions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": _CONTENT_PROFILE_SCHEMA,
        "lock_sha256": lock_sha256,
        "runtime": {"python": python_version, "platform": platform},
        "aggregate_sha256": aggregate,
        "distributions": distributions,
    }


def load_json_strict(path: Path) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {path}: {value}")

    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read content profile {path}: {exc}") from exc


def verify_content_profile(path: Path, computed: Mapping[str, object]) -> dict[str, str]:
    stored = load_json_strict(path)
    if stored != computed:
        raise ValueError(f"installed Python content does not match sealed profile: {path}")
    aggregate = computed.get("aggregate_sha256")
    if not isinstance(aggregate, str) or not re.fullmatch(r"[0-9a-f]{64}", aggregate):
        raise ValueError(f"invalid aggregate digest in content profile: {path}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "aggregate_sha256": aggregate,
    }


def verify_exact_content_profile(lock_path: Path, content_profile_path: Path) -> dict[str, str]:
    """Verify the running interpreter and every locked distribution byte.

    Release artifact CLIs call this directly rather than relying on a parent
    runbook to have performed the same check.  It deliberately shares the
    receipt verifier's complete runtime, distribution, RECORD, and sealed
    content-profile validation path.
    """

    lock = load_lock(lock_path)
    python_version = runtime_python_version()
    platform = runtime_platform()
    distribution_objects = list(importlib.metadata.distributions())
    installed = installed_distributions()
    verify_runtime(lock, python_version=python_version, platform=platform)
    verify_installed(lock.packages, installed, allowed_extra=_BOOTSTRAP_DISTRIBUTIONS)
    evidence = verify_distribution_contents(
        distribution_objects,
        expected=lock.packages,
        prefix=Path(sys.prefix),
    )
    computed_profile = build_content_profile(
        lock_sha256=sha256_file(lock.path),
        python_version=python_version,
        platform=platform,
        evidence=evidence,
    )
    return verify_content_profile(content_profile_path, computed_profile)


def verify_installed(
    expected: Mapping[str, str],
    installed: Mapping[str, str],
    *,
    allowed_extra: Sequence[str] = (),
) -> None:
    normalized_expected = {normalize_name(name): version for name, version in expected.items()}
    normalized_installed = {normalize_name(name): version for name, version in installed.items()}
    missing = sorted(set(normalized_expected) - set(normalized_installed))
    wrong = sorted(
        name
        for name in set(normalized_expected) & set(normalized_installed)
        if normalized_expected[name] != normalized_installed[name]
    )
    allowed = {normalize_name(name) for name in allowed_extra}
    unexpected = sorted(set(normalized_installed) - set(normalized_expected) - allowed)
    failures: list[str] = []
    if missing:
        failures.append("missing distributions: " + ", ".join(missing))
    if wrong:
        failures.append(
            "version mismatch: "
            + ", ".join(
                f"{name} expected {normalized_expected[name]} got {normalized_installed[name]}" for name in wrong
            )
        )
    if unexpected:
        failures.append("unexpected distributions: " + ", ".join(unexpected))
    if failures:
        raise ValueError("; ".join(failures))


def build_receipt(
    lock: EnvironmentLock,
    *,
    python_version: str,
    platform: str,
    installed: Mapping[str, str],
    content_profile: Mapping[str, str],
) -> dict[str, object]:
    verify_runtime(lock, python_version=python_version, platform=platform)
    verify_installed(lock.packages, installed)
    return {
        "schema_version": 2,
        "status": "verified",
        "lock": {
            "path": lock.path.name,
            "sha256": sha256_file(lock.path),
            "exclude_newer": lock.exclude_newer,
            "resolver": lock.resolver,
            "only_binary": lock.only_binary,
        },
        "runtime": {"python": python_version, "platform": platform},
        "content_profile": dict(content_profile),
        "distributions": dict(sorted((normalize_name(name), version) for name, version in installed.items())),
    }


def build_installed_structure_receipt(
    lock: EnvironmentLock,
    *,
    python_version: str,
    platform: str,
    installed: Mapping[str, str],
) -> dict[str, object]:
    """Describe a lock/version check without claiming installed-byte identity."""

    verify_runtime(lock, python_version=python_version, platform=platform)
    verify_installed(lock.packages, installed)
    return {
        "schema_version": 2,
        "status": "verified",
        "verification_scope": "runtime-and-installed-distribution-versions",
        "lock": {
            "path": lock.path.name,
            "sha256": sha256_file(lock.path),
            "exclude_newer": lock.exclude_newer,
            "resolver": lock.resolver,
            "only_binary": lock.only_binary,
        },
        "runtime": {"python": python_version, "platform": platform},
        "distributions": dict(sorted((normalize_name(name), version) for name, version in installed.items())),
    }


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_receipt_atomic(path: Path, receipt: Mapping[str, object]) -> None:
    """Publish a verified receipt atomically without following a target symlink."""

    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink receipt destination: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(canonical_json(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def clear_receipt_output(path: Path) -> None:
    """Remove stale public verification output before a fresh attempt."""

    if path.is_symlink():
        raise ValueError(f"refusing to remove symlink receipt destination: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"receipt destination is not a regular file: {path}")
        path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    verification = parser.add_mutually_exclusive_group(required=True)
    verification.add_argument("--content-profile", type=Path)
    verification.add_argument(
        "--installed-structure-only",
        action="store_true",
        help=(
            "verify the locked runtime and exact installed version set without "
            "claiming installed distribution-byte identity"
        ),
    )
    parser.add_argument(
        "--initialize-content-profile",
        action="store_true",
        help="write a new profile only when the destination does not yet exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output:
            clear_receipt_output(args.output)
        lock = load_lock(args.lock)
        python_version = runtime_python_version()
        platform = runtime_platform()
        installed = installed_distributions()
        verify_runtime(lock, python_version=python_version, platform=platform)
        verify_installed(lock.packages, installed, allowed_extra=_BOOTSTRAP_DISTRIBUTIONS)
        receipt_installed = {name: installed[name] for name in lock.packages}
        if args.installed_structure_only:
            if args.initialize_content_profile:  # pragma: no cover - guarded by argparse usage
                raise ValueError("--initialize-content-profile requires --content-profile")
            receipt = build_installed_structure_receipt(
                lock,
                python_version=python_version,
                platform=platform,
                installed=receipt_installed,
            )
        else:
            distribution_objects = list(importlib.metadata.distributions())
            evidence = verify_distribution_contents(
                distribution_objects,
                expected=lock.packages,
                prefix=Path(sys.prefix),
            )
            computed_profile = build_content_profile(
                lock_sha256=sha256_file(lock.path),
                python_version=python_version,
                platform=platform,
                evidence=evidence,
            )
            if args.initialize_content_profile:
                if args.content_profile.exists() or args.content_profile.is_symlink():
                    raise ValueError(f"refusing to overwrite existing content profile: {args.content_profile}")
                write_receipt_atomic(args.content_profile, computed_profile)
            profile_binding = verify_content_profile(args.content_profile, computed_profile)
            receipt = build_receipt(
                lock,
                python_version=python_version,
                platform=platform,
                installed=receipt_installed,
                content_profile=profile_binding,
            )
    except (OSError, ValueError) as exc:
        print(f"environment verification failed: {exc}", file=sys.stderr)
        return 1
    rendered = canonical_json(receipt)
    if args.output:
        try:
            write_receipt_atomic(args.output, receipt)
        except (OSError, ValueError) as exc:
            print(f"environment verification failed: {exc}", file=sys.stderr)
            return 1
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
