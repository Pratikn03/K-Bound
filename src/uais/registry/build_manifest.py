"""Generate ``models/MANIFEST.json`` by hashing every model artifact.

This module walks the ``models/`` directory for serialized model artifacts
(``*.pkl``, ``*.pt``, ``*.joblib``), computes a SHA-256 digest and size for each,
and writes a JSON manifest that :class:`uais.registry.ModelRegistry` consumes for
integrity verification.

Run it from the repository root with ``src`` on the path::

    PYTHONPATH=src python -m uais.registry.build_manifest

Or point it at an explicit repo root / output location::

    PYTHONPATH=src python -m uais.registry.build_manifest \\
        --repo-root /path/to/repo --output models/MANIFEST.json

The generated manifest schema is::

    {
        "version": 1,
        "generated": "<ISO-8601 UTC timestamp>",
        "artifacts": [
            {
                "name": "<derived stable name>",
                "path": "<path relative to repo root, POSIX separators>",
                "sha256": "<hex digest>",
                "size_bytes": <int>,
                "mtime": "<ISO-8601 UTC timestamp of file mtime>"
            },
            ...
        ]
    }

Only the standard library is required.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_registry import DEFAULT_MANIFEST_PATH, PENDING_SHA256, sha256_file

__all__ = [
    "MANIFEST_VERSION",
    "ARTIFACT_SUFFIXES",
    "derive_name",
    "discover_artifacts",
    "build_manifest",
    "write_manifest",
    "main",
]

# Manifest schema version (bump when the structure changes incompatibly).
MANIFEST_VERSION = 1

# Artifact file extensions considered model binaries.
ARTIFACT_SUFFIXES = (".pkl", ".pt", ".joblib")


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: Path) -> str:
    """Return a file's modification time as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def derive_name(relative_path: Path) -> str:
    """Derive a stable, human-readable artifact name from its relative path.

    The name is built from the path components beneath ``models/`` so that
    artifacts in different domains never collide. For example::

        models/fraud/supervised/fraud_model.pkl   -> "fraud__supervised__fraud_model"
        models/behavior/behavior_lof.pkl          -> "behavior__behavior_lof"
        models/fusion/fusion_meta_model.pkl       -> "fusion__fusion_meta_model"

    Args:
        relative_path: Artifact path relative to the repository root. The leading
            ``models`` component, if present, is dropped from the derived name.

    Returns:
        A double-underscore-joined name without the file extension.
    """
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[0] == "models":
        parts = parts[1:]
    return "__".join(parts) if parts else relative_path.stem


def discover_artifacts(models_dir: Path) -> list[Path]:
    """Return sorted absolute paths of all artifact files under ``models_dir``.

    Args:
        models_dir: Directory to walk recursively.

    Returns:
        Sorted list of absolute artifact paths. Empty if the directory does not
        exist or contains no matching files.
    """
    if not models_dir.exists():
        return []
    found: list[Path] = []
    for suffix in ARTIFACT_SUFFIXES:
        found.extend(models_dir.rglob(f"*{suffix}"))
    # Deduplicate (rglob patterns are disjoint here, but be defensive) and sort.
    unique = sorted({p.resolve() for p in found if p.is_file()})
    return unique


def build_manifest(repo_root: Path, compute_hashes: bool = True) -> dict[str, Any]:
    """Build the manifest dictionary for all artifacts under ``<repo_root>/models``.

    Args:
        repo_root: Repository root containing the ``models/`` directory.
        compute_hashes: When ``True`` (default) compute the real SHA-256 of each
            artifact. When ``False`` the ``sha256`` field is set to the
            ``"PENDING"`` sentinel so the manifest structure can be written
            without reading large binaries.

    Returns:
        A JSON-serialisable manifest dictionary.
    """
    repo_root = Path(repo_root).resolve()
    models_dir = repo_root / "models"
    artifacts: list[dict[str, Any]] = []

    for artifact_path in discover_artifacts(models_dir):
        relative = artifact_path.relative_to(repo_root)
        entry: dict[str, Any] = {
            "name": derive_name(relative),
            "path": relative.as_posix(),
            "sha256": sha256_file(artifact_path) if compute_hashes else PENDING_SHA256,
            "size_bytes": artifact_path.stat().st_size,
            "mtime": _mtime_iso(artifact_path),
        }
        artifacts.append(entry)

    return {
        "version": MANIFEST_VERSION,
        "generated": _utc_now_iso(),
        "artifacts": artifacts,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """Write a manifest dict to ``output_path`` as pretty-printed JSON.

    Args:
        manifest: The manifest dictionary (see :func:`build_manifest`).
        output_path: Destination file path. Parent directories are created.

    Returns:
        The path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate models/MANIFEST.json by hashing model artifacts.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing the models/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output manifest path (default: <repo-root>/{DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Write the 'PENDING' sentinel instead of computing real SHA-256 digests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build and write the model manifest.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = _parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    output_path: Path = args.output or (repo_root / DEFAULT_MANIFEST_PATH)

    manifest = build_manifest(repo_root, compute_hashes=not args.no_hashes)
    write_manifest(manifest, output_path)

    n = len(manifest["artifacts"])
    mode = "PENDING (no hashes)" if args.no_hashes else "with SHA-256 digests"
    print(f"Wrote manifest {mode}: {output_path} ({n} artifact(s))")
    for entry in manifest["artifacts"]:
        print(f"  - {entry['name']}: {entry['path']} [{entry['sha256'][:16]}...]")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
