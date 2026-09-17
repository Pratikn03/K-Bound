r"""Single source of truth for maintained K-Bound manuscript inputs.

Release checks start from every maintained driver and follow each live
``\input``/``\include`` closure.  This prevents a generated table or nested
section from bypassing claim validation while still excluding archived drafts
that are not compiled.  The same live-LaTeX projection removes comments and
``\iffalse`` audit blocks before both dependency discovery and claim scanning.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath

__all__ = [
    "ACTIVE_DRIVER_RELATIVE_PATHS",
    "SourceBoundaryError",
    "active_source_paths",
    "live_latex",
    "probe_public_file",
    "read_public_bytes",
    "read_public_text",
    "read_unlinked_bytes",
    "read_unlinked_text",
    "require_unlinked_file",
    "validate_source_reference",
    "validated_root",
]


ACTIVE_DRIVER_RELATIVE_PATHS = (
    "docs/research/kbound/kbound_submission.tex",
    "docs/research/kbound/kbound_tmlr.tex",
    "docs/research/kbound/kbound_full.tex",
    "docs/research/kbound/kbound_short_main.tex",
    "docs/research/kbound/kbound_short_supplement.tex",
)

_PROTECTED_AUTHORITY_PREFIXES = (
    "experiments/kbound/so2sat",
    "experiments/kbound/data/so2sat",
    "experiments/kbound/results/so2sat",
    "docs/research/kbound/audits/so2sat",
    "docs/research/kbound/release/so2sat",
    "audits/so2sat",
    "release/so2sat",
)


class SourceBoundaryError(RuntimeError):
    """A manuscript input is unsafe, redirected, or outside the public closure."""


def _is_protected_authority(value: str) -> bool:
    lowered = value.casefold().rstrip("/")
    return any(lowered.startswith(prefix) for prefix in _PROTECTED_AUTHORITY_PREFIXES)


def validate_source_reference(value: str, *, label: str = "manuscript dependency") -> str:
    """Return one canonical public relative path before any filesystem I/O."""

    if not value or value != value.strip() or "\x00" in value or "\\" in value:
        raise SourceBoundaryError(f"unsafe {label}: {value!r}")
    lowered = value.casefold()
    if value.startswith(("/", "~", "$HOME/", "${HOME}/", "//")) or lowered.startswith(("file:", "%userprofile%/")):
        raise SourceBoundaryError(f"unsafe absolute {label}: {value!r}")
    if re.match(r"^[A-Za-z]:", value):
        raise SourceBoundaryError(f"unsafe absolute {label}: {value!r}")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != value
    ):
        raise SourceBoundaryError(f"unsafe non-canonical {label}: {value!r}")
    if _is_protected_authority(value):
        raise SourceBoundaryError(f"protected So2Sat authority is outside the publication closure: {value}")
    return value


def _absolute_lexical_path(path: str | Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise SourceBoundaryError(f"unsafe {label}: {raw!r}")
    normalized = os.path.normpath(raw)
    if normalized != raw or any(part == ".." for part in Path(raw).parts):
        raise SourceBoundaryError(f"unsafe non-canonical {label}: {raw!r}")
    absolute = Path(raw) if Path(raw).is_absolute() else Path.cwd() / raw
    rendered = absolute.as_posix().casefold().rstrip("/") + "/"
    for marker in _PROTECTED_AUTHORITY_PREFIXES:
        if "/" + marker in rendered:
            raise SourceBoundaryError(f"protected So2Sat authority is outside the publication closure: {raw}")
    return absolute


def _lstat_components(path: Path, *, final_kind: str, label: str, missing_ok: bool = False) -> bool:
    """Check every component without following a symlink."""

    if not path.is_absolute():  # pragma: no cover - guarded by caller
        raise SourceBoundaryError(f"{label} is not absolute: {path}")
    parts = path.parts
    for index in range(1, len(parts) + 1):
        component = Path(*parts[:index])
        try:
            state = os.lstat(component)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise SourceBoundaryError(f"{label} is missing: {path}") from None
        except OSError as exc:
            raise SourceBoundaryError(f"cannot inspect {label} without following links: {path}") from exc
        if stat.S_ISLNK(state.st_mode):
            raise SourceBoundaryError(f"{label} traverses a symlink: {component}")
        is_final = index == len(parts)
        if not is_final and not stat.S_ISDIR(state.st_mode):
            raise SourceBoundaryError(f"{label} parent is not a directory: {component}")
        if is_final and final_kind == "directory" and not stat.S_ISDIR(state.st_mode):
            raise SourceBoundaryError(f"{label} is not a directory: {path}")
        if is_final and final_kind == "file" and not stat.S_ISREG(state.st_mode):
            raise SourceBoundaryError(f"{label} is not a regular file: {path}")
    return True


def validated_root(root: str | Path) -> Path:
    """Return an absolute repository root whose entire path is unredirected."""

    absolute = _absolute_lexical_path(root, label="publication root")
    _lstat_components(absolute, final_kind="directory", label="publication root")
    return absolute


def probe_public_file(root: str | Path, relative: str, *, label: str = "manuscript dependency") -> Path | None:
    """Probe one already-public relative file without following any link."""

    canonical = validate_source_reference(relative, label=label)
    absolute_root = validated_root(root)
    candidate = absolute_root.joinpath(*PurePosixPath(canonical).parts)
    if not _lstat_components(candidate, final_kind="file", label=label, missing_ok=True):
        return None
    return candidate


def _read_regular_fd(path: Path, *, label: str) -> bytes:
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_descriptor: int | None = None
    try:
        try:
            directory_descriptor = os.open("/", directory_flags)
            for component in path.parent.parts[1:]:
                next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            descriptor = os.open(path.name, file_flags, dir_fd=directory_descriptor)
        except OSError:
            descriptor = os.open(str(path), file_flags)
    except OSError as exc:
        raise SourceBoundaryError(f"cannot open {label} without following links: {path}") from exc
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceBoundaryError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity or len(payload) != after.st_size:
            raise SourceBoundaryError(f"{label} changed while being read: {path}")
        return payload
    finally:
        os.close(descriptor)


def read_public_bytes(root: str | Path, relative: str, *, label: str = "manuscript dependency") -> bytes:
    """Read a validated public input through a no-follow descriptor."""

    path = probe_public_file(root, relative, label=label)
    if path is None:
        raise SourceBoundaryError(f"{label} is missing: {relative}")
    return _read_regular_fd(path, label=label)


def read_public_text(
    root: str | Path,
    relative: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    label: str = "manuscript dependency",
) -> str:
    return read_public_bytes(root, relative, label=label).decode(encoding, errors=errors)


def read_unlinked_bytes(path: str | Path, *, label: str = "publication input") -> bytes:
    """Read an absolute or cwd-relative regular file without link traversal."""

    absolute = _absolute_lexical_path(path, label=label)
    _lstat_components(absolute, final_kind="file", label=label)
    return _read_regular_fd(absolute, label=label)


def require_unlinked_file(path: str | Path, *, label: str = "publication input") -> Path:
    """Validate a regular file path without opening or following it."""

    absolute = _absolute_lexical_path(path, label=label)
    _lstat_components(absolute, final_kind="file", label=label)
    return absolute


def read_unlinked_text(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    label: str = "publication input",
) -> str:
    return read_unlinked_bytes(path, label=label).decode(encoding, errors=errors)


def active_source_paths(repo_root: str | Path) -> tuple[Path, ...]:
    """Return the ordered live TeX dependency closure under ``repo_root``.

    Literal inputs are resolved first relative to the including file, then the
    K-Bound manuscript root, then the repository root.  An unresolved literal
    is retained as a missing path so release callers can fail closed.
    """
    root = validated_root(repo_root)
    kbound_prefix = PurePosixPath("docs/research/kbound")
    ordered: list[Path] = []
    seen: set[str] = set()
    input_re = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
    input_command_re = re.compile(r"\\(?:input|include)\b")
    graphic_re = re.compile(r"\\(?:kbgraphics|includegraphics)(?:\[[^]]*\])?\s*\{([^{}]+)\}")
    package_re = re.compile(r"\\(?:usepackage|RequirePackage)(?:\[[^]]*\])?\s*\{([^{}]+)\}")
    bibliography_re = re.compile(r"\\(?:bibliography|addbibresource)(?:\[[^]]*\])?\s*\{([^{}]+)\}")
    class_re = re.compile(r"\\documentclass(?:\[[^]]*\])?\s*\{([^{}]+)\}")

    def candidates(including: str, token: str, *, suffixes: tuple[str, ...]) -> tuple[str, ...]:
        canonical = validate_source_reference(token.strip(), label="TeX input")
        raw = PurePosixPath(canonical)
        variants = (raw,) if raw.suffix else tuple(raw.with_suffix(suffix) for suffix in suffixes) + (raw,)
        bases = (PurePosixPath(including).parent, kbound_prefix, PurePosixPath())
        values: list[str] = []
        for base in bases:
            for variant in variants:
                relative = (base / variant).as_posix()
                validate_source_reference(relative, label="TeX input")
                if relative not in values:
                    values.append(relative)
        return tuple(values)

    def visit(relative: str) -> None:
        canonical = validate_source_reference(relative, label="manuscript source")
        if canonical in seen:
            return
        seen.add(canonical)
        path = probe_public_file(root, canonical, label="manuscript source")
        ordered.append(root / canonical)
        if path is None:
            return
        text = live_latex(read_public_text(root, canonical, errors="ignore", label="manuscript source"))
        input_matches = tuple(input_re.finditer(text))
        input_starts = {match.start() for match in input_matches}
        for command in input_command_re.finditer(text):
            if command.start() not in input_starts:
                raise SourceBoundaryError("unsupported unbraced TeX input is outside the public dependency closure")
        for match in input_matches:
            token = match.group(1).strip()
            if not token or "\\" in token or "#" in token:
                raise SourceBoundaryError(f"dynamic TeX input is outside the public dependency closure: {token!r}")
            options = candidates(canonical, token, suffixes=(".tex",))
            target = next(
                (
                    candidate
                    for candidate in options
                    if probe_public_file(root, candidate, label="TeX input") is not None
                ),
                options[0],
            )
            visit(target)
        for match in graphic_re.finditer(text):
            token = match.group(1).strip()
            if not token or "\\" in token:
                raise SourceBoundaryError(f"dynamic TeX graphic is outside the public dependency closure: {token!r}")
            if "#" in token:
                continue
            options = candidates(canonical, token, suffixes=(".pdf", ".png"))
            target = next(
                (
                    candidate
                    for candidate in options
                    if probe_public_file(root, candidate, label="TeX graphic") is not None
                ),
                options[0],
            )
            if target not in seen:
                seen.add(target)
                ordered.append(root / target)
        for match in package_re.finditer(text):
            for token in (value.strip() for value in match.group(1).split(",")):
                options = candidates(canonical, token, suffixes=(".sty",))
                target = next(
                    (
                        candidate
                        for candidate in options
                        if probe_public_file(root, candidate, label="TeX package") is not None
                    ),
                    options[0],
                )
                if probe_public_file(root, target, label="TeX package") is not None:
                    visit(target)
                elif "/" in token or PurePosixPath(token).suffix:
                    visit(target)
        for match in class_re.finditer(text):
            token = match.group(1).strip()
            options = candidates(canonical, token, suffixes=(".cls",))
            target = next(
                (
                    candidate
                    for candidate in options
                    if probe_public_file(root, candidate, label="TeX document class") is not None
                ),
                options[0],
            )
            if probe_public_file(root, target, label="TeX document class") is not None:
                visit(target)
            elif "/" in token or PurePosixPath(token).suffix:
                visit(target)
        for match in bibliography_re.finditer(text):
            for token in (value.strip() for value in match.group(1).split(",")):
                options = candidates(canonical, token, suffixes=(".bib",))
                target = next(
                    (
                        candidate
                        for candidate in options
                        if probe_public_file(root, candidate, label="TeX bibliography") is not None
                    ),
                    options[0],
                )
                if target not in seen:
                    seen.add(target)
                    ordered.append(root / target)

    for relative in ACTIVE_DRIVER_RELATIVE_PATHS:
        visit(relative)
    return tuple(ordered)


def live_latex(text: str) -> str:
    """Remove disabled blocks and comments before checking assertion text."""
    text = re.sub(r"\\iffalse.*?\\fi", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        lines.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    return "\n".join(lines)
