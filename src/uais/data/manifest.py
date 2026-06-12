"""Data-integrity manifest for the K-Bound reproduction inputs (stdlib only).

A *manifest* is a JSON record of the SHA-256 hash, size, and modification time of
every file under a set of key data roots, so a reproducer can verify that the
inputs feeding the K-Bound experiments are bit-for-bit the ones the results were
produced from.  The default root is the 123-task ELARA-U score archive
(``experiments/elara_u/score_archive``) that
``src/scripts/kbound/knowability_experiment.py`` and
``mixed_regime_experiment.py`` consume.

This module uses only the Python standard library (``hashlib``/``json``/
``pathlib``) -- no numpy, no third-party deps -- so integrity can be checked in a
minimal environment.

Manifest schema (``data/MANIFEST.json``)
----------------------------------------
``{``
``  "version":   "<schema version string>",``
``  "generated": "<UTC ISO-8601 timestamp>",``
``  "roots":     ["<repo-relative root>", ...],``
``  "entries": [``
``    {"path": "<repo-relative path>", "sha256": "<hex>",``
``     "size_bytes": <int>, "mtime": "<UTC ISO-8601 timestamp>"},``
``    ...``
``  ]``
``}``

Entries are sorted by ``path`` for a stable, diffable manifest.

CLI
---
    python -m uais.data.manifest build   [--roots R [R ...]] [--output PATH]
    python -m uais.data.manifest verify  [--manifest PATH]

``build`` (re)writes the manifest with current hashes; ``verify`` re-hashes the
listed files and reports any missing, extra, or changed entries.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

#: Manifest schema version (bump on incompatible format changes).
MANIFEST_VERSION = "kbound-data-manifest/1"

#: Default data roots to hash (repo-relative).  Skipped gracefully if absent.
DEFAULT_ROOTS: list[str] = ["experiments/elara_u/score_archive"]

#: Default manifest location (repo-relative).
DEFAULT_MANIFEST = os.path.join("data", "MANIFEST.json")

#: Chunk size for streaming hashes (1 MiB) -- bounds memory on large files.
_CHUNK = 1 << 20


def repo_root() -> Path:
    """Return the repository root, four levels above this file.

    ``src/uais/data/manifest.py`` -> repo root is ``parents[3]``.
    """
    return Path(__file__).resolve().parents[3]


def _utcnow_iso() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _mtime_iso(path: Path) -> str:
    """File modification time as a UTC ISO-8601 string."""
    ts = path.stat().st_mtime
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lower-case hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    """Return all regular files under ``root`` (recursively), sorted.

    Hidden AppleDouble sidecar files (``._*``) and ``.DS_Store`` are skipped so
    the manifest is portable across filesystems.
    """
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name == ".DS_Store" or name.startswith("._"):
            continue
        out.append(p)
    return out


def _rel(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` using forward slashes."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def build_manifest(
    roots: Sequence[str] | None = None,
    output: str | None = None,
    *,
    base: Path | None = None,
) -> dict[str, object]:
    """Build a data-integrity manifest and write it to ``output`` as JSON.

    Parameters
    ----------
    roots : sequence of str, optional
        Repo-relative (or absolute) data roots to hash.  Missing roots are
        skipped gracefully (recorded in ``roots`` but contributing no entries),
        so the build never fails just because an optional archive is absent.
        Defaults to :data:`DEFAULT_ROOTS`.
    output : str, optional
        Repo-relative (or absolute) path for the manifest JSON.  Defaults to
        :data:`DEFAULT_MANIFEST` (``data/MANIFEST.json``).
    base : pathlib.Path, optional
        Base directory that paths are recorded relative to (defaults to the
        repository root).  Mainly for testing.

    Returns
    -------
    dict
        The manifest dictionary that was written.

    Notes
    -----
    If no listed root exists (e.g. the score archive is a local-only artifact),
    the manifest is still written with ``entries: []`` so that ``data/
    MANIFEST.json`` always exists with a valid version stamp; the lead can re-run
    ``build`` centrally once the real archive is present to populate hashes.
    """
    base = base or repo_root()
    roots = list(roots) if roots is not None else list(DEFAULT_ROOTS)
    out_path = Path(output) if output is not None else Path(DEFAULT_MANIFEST)
    if not out_path.is_absolute():
        out_path = base / out_path

    entries: list[dict[str, object]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = base / root_path
        for f in _iter_files(root_path):
            entries.append(
                {
                    "path": _rel(f, base),
                    "sha256": sha256_file(f),
                    "size_bytes": int(f.stat().st_size),
                    "mtime": _mtime_iso(f),
                }
            )
    entries.sort(key=lambda e: e["path"])  # stable, diffable order

    manifest: dict[str, object] = {
        "version": MANIFEST_VERSION,
        "generated": _utcnow_iso(),
        "roots": [str(r) for r in roots],
        "entries": entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return manifest


def verify_manifest(path: str | None = None, *, base: Path | None = None) -> dict[str, object]:
    """Verify on-disk files against a manifest and return a structured report.

    Re-hashes every entry's file and compares the digest (and size) to the
    recorded value, and flags entries whose files are missing.

    Parameters
    ----------
    path : str, optional
        Manifest path (repo-relative or absolute).  Defaults to
        :data:`DEFAULT_MANIFEST`.
    base : pathlib.Path, optional
        Base directory the entry paths are resolved against (defaults to the
        repository root).

    Returns
    -------
    dict
        Report with keys:

        * ``ok`` (bool) -- ``True`` iff nothing is missing or changed;
        * ``manifest`` (str) -- the manifest path checked;
        * ``version`` (str) -- the manifest's schema version;
        * ``n_entries`` (int) -- number of entries checked;
        * ``n_ok`` (int) -- number of entries that matched;
        * ``missing`` (list of str) -- entry paths whose files are absent;
        * ``changed`` (list of dict) -- entries whose hash/size differs, each
          ``{"path", "reason"}``;
        * ``empty`` (bool) -- ``True`` if the manifest has no entries.

    Raises
    ------
    FileNotFoundError
        If the manifest file itself does not exist.
    """
    base = base or repo_root()
    man_path = Path(path) if path is not None else Path(DEFAULT_MANIFEST)
    if not man_path.is_absolute():
        man_path = base / man_path
    if not man_path.exists():
        raise FileNotFoundError(f"Manifest not found: {man_path}")

    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    entries = manifest.get("entries", [])
    missing: list[str] = []
    changed: list[dict[str, str]] = []
    n_ok = 0

    for entry in entries:
        rel = str(entry["path"])
        fpath = base / rel
        if not fpath.exists():
            missing.append(rel)
            continue
        actual_size = int(fpath.stat().st_size)
        if actual_size != int(entry.get("size_bytes", -1)):
            changed.append({"path": rel, "reason": "size mismatch"})
            continue
        actual_hash = sha256_file(fpath)
        if actual_hash != str(entry.get("sha256", "")):
            changed.append({"path": rel, "reason": "sha256 mismatch"})
            continue
        n_ok += 1

    ok = not missing and not changed
    return {
        "ok": ok,
        "manifest": str(man_path),
        "version": str(manifest.get("version", "")),
        "n_entries": len(entries),
        "n_ok": n_ok,
        "missing": missing,
        "changed": changed,
        "empty": len(entries) == 0,
    }


def _cmd_build(args: argparse.Namespace) -> int:
    """Handle the ``build`` subcommand."""
    manifest = build_manifest(roots=args.roots, output=args.output)
    n = len(manifest["entries"])  # type: ignore[arg-type]
    out = args.output or DEFAULT_MANIFEST
    print(f"[manifest] wrote {out} with {n} entr{'y' if n == 1 else 'ies'}")
    print(f"[manifest] version={manifest['version']} roots={manifest['roots']}")
    if n == 0:
        print(
            "[manifest] note: no files hashed (data roots absent). "
            "Re-run `build` once the score archive is present to populate hashes."
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Handle the ``verify`` subcommand."""
    try:
        report = verify_manifest(path=args.manifest)
    except FileNotFoundError as exc:
        print(f"[manifest] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    if report["empty"]:
        print(
            "[manifest] note: manifest has no entries (nothing to verify yet).",
            file=sys.stderr,
        )
        return 0
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m uais.data.manifest``."""
    parser = argparse.ArgumentParser(
        prog="uais.data.manifest",
        description="Build/verify the K-Bound data-integrity manifest (data/MANIFEST.json).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Hash the data roots and write the manifest.")
    p_build.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help=f"Data roots to hash (repo-relative). Default: {DEFAULT_ROOTS}.",
    )
    p_build.add_argument(
        "--output",
        default=None,
        help=f"Manifest output path. Default: {DEFAULT_MANIFEST}.",
    )
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="Re-hash files and report drift.")
    p_verify.add_argument(
        "--manifest",
        default=None,
        help=f"Manifest path to verify. Default: {DEFAULT_MANIFEST}.",
    )
    p_verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch the chosen subcommand.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
