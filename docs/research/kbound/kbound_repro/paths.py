"""kbound_repro.paths -- repository-root discovery and external data roots.

Replaces hard-coded absolute paths (``/Users/pratik_n/...``, ``/Volumes/T9/...``)
in executable code with three portable mechanisms, in priority order:

1. an explicit CLI argument passed by the caller,
2. an environment variable with a documented default, or
3. ``pathlib`` resolution relative to the discovered repository root.

External dataset / results locations are accepted through environment variables
so a clean checkout on any machine can point at wherever the (git-excluded)
heavy data actually lives::

    KBOUND_DATA_ROOT        parent of datasets (default: <repo>/experiments/kbound/data)
    KBOUND_RESULTS_ROOT     parent of result artifacts (default: <repo>/experiments/kbound/results)
    KBOUND_IMAGENETR_ROOT   ImageNet-R images (default: <KBOUND_DATA_ROOT>/imagenet-r)
    KBOUND_PACS_ROOT        PACS images (default: <KBOUND_DATA_ROOT>/PACS)

Nothing here reads or writes the datasets; it only *resolves and validates* the
paths, failing with an actionable message when required data is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "RepoRootNotFound",
    "DataRootError",
    "find_repo_root",
    "repo_relative",
    "data_root",
    "results_root",
    "imagenetr_root",
    "pacs_root",
    "require_dir",
]

# Markers that identify the repository root when walking up the tree.
_ROOT_MARKERS = ("pyproject.toml", ".git", "AGENTS.md")


class RepoRootNotFound(RuntimeError):
    """Raised when the repository root cannot be discovered."""


class DataRootError(FileNotFoundError):
    """Raised when a required external data/results root is missing."""


def find_repo_root(start: str | os.PathLike | None = None) -> Path:
    """Discover the repository root by walking upward from ``start``.

    ``start`` defaults to this file's location, so discovery works no matter
    what the process's current working directory is (fixing runbooks that only
    worked from one absolute checkout).
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and any((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    raise RepoRootNotFound(
        "could not locate the repository root (no pyproject.toml/.git/AGENTS.md found "
        f"walking up from {here}). Pass an explicit path or run inside the repo."
    )


def repo_relative(*parts: str, start: str | os.PathLike | None = None) -> Path:
    """Return ``<repo_root>/<parts...>`` using repository-root discovery."""
    return find_repo_root(start).joinpath(*parts)


def _resolve(env_name: str, default: Path) -> Path:
    override = os.environ.get(env_name)
    return Path(override).expanduser().resolve() if override else default


def data_root() -> Path:
    """External dataset root (``$KBOUND_DATA_ROOT`` or repo default)."""
    return _resolve("KBOUND_DATA_ROOT", find_repo_root() / "experiments" / "kbound" / "data")


def results_root() -> Path:
    """External results root (``$KBOUND_RESULTS_ROOT`` or repo default)."""
    return _resolve("KBOUND_RESULTS_ROOT", find_repo_root() / "experiments" / "kbound" / "results")


def imagenetr_root() -> Path:
    """ImageNet-R image root (``$KBOUND_IMAGENETR_ROOT`` or under the data root)."""
    return _resolve("KBOUND_IMAGENETR_ROOT", data_root() / "imagenet-r")


def pacs_root() -> Path:
    """PACS image root (``$KBOUND_PACS_ROOT`` or under the data root)."""
    return _resolve("KBOUND_PACS_ROOT", data_root() / "PACS")


def require_dir(path: str | os.PathLike, *, what: str, env_var: str | None = None) -> Path:
    """Return ``path`` if it exists and is a directory; else raise actionably.

    ``what`` names the resource and ``env_var`` (if given) tells the user which
    environment variable to set -- so a missing dataset fails with guidance
    rather than a bare ``FileNotFoundError`` deep in a data loader.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        return p.resolve()
    hint = f" Set {env_var} to its location." if env_var else ""
    raise DataRootError(
        f"required {what} not found at {p}.{hint} "
        "Datasets are intentionally git-excluded; see docs/research/kbound/REPRODUCE.md "
        "and scripts/download_data.py to obtain them."
    )
