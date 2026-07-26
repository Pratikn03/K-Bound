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

The single external volume (defect D8)
--------------------------------------
Everything above resolves *inside* the repository by default.  A few artifacts
genuinely cannot: the raw dataset volume, the model-checkpoint store and the
scratch/cache directories were, on the author's machine, ``/Volumes/T9/uav`` and
``/Users/pratik_n/...``.  ``EXTERNAL_STORAGE_POLICY.md`` bans those strings from
tracked code, so they route through **one** documented variable::

    KBOUND_EXTERNAL_ROOT    the external volume holding git-excluded heavy data

with a fixed, documented layout underneath it::

    $KBOUND_EXTERNAL_ROOT/datasets/wilds        WILDS (Camelyon17, iWildCam, RxRx1)
    $KBOUND_EXTERNAL_ROOT/imagenetc_local       ImageNet-C tars / extracted groups
    $KBOUND_EXTERNAL_ROOT/kbound_rxrx1_ckpt     RxRx1 ERM checkpoints
    $KBOUND_EXTERNAL_ROOT/kbound_rxrx1_data     RxRx1 raw data
    $KBOUND_EXTERNAL_ROOT/kbound_rxrx1_results  RxRx1 run outputs
    $KBOUND_EXTERNAL_ROOT/kbound_inr_results    ImageNet-R run outputs
    $KBOUND_EXTERNAL_ROOT/tmp                   scratch (TMPDIR)
    $KBOUND_EXTERNAL_ROOT/torch_cache           TORCH_HOME

:func:`external_root` **raises** when the variable is unset.  It deliberately
does not fall back to ``$HOME``: silently writing a multi-gigabyte dataset into
somebody's home directory is worse than refusing to start.

A second variable, ``KBOUND_PYTHON``, names the interpreter for the runbook shell
scripts (they used to hard-code ``/Users/pratik_n/.venv_wilds/bin/python`` and
``/opt/anaconda3/envs/aetta/bin/python``).  It defaults to ``python3``, so it
needs no error branch.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "RepoRootNotFound",
    "DataRootError",
    "ExternalRootUnset",
    "EXTERNAL_ROOT_ENV",
    "EXTERNAL_LAYOUT",
    "find_repo_root",
    "repo_relative",
    "data_root",
    "results_root",
    "imagenetr_root",
    "pacs_root",
    "external_root",
    "external_path",
    "interpreter",
    "require_dir",
]

#: The one environment variable that points outside the repository (defect D8).
EXTERNAL_ROOT_ENV = "KBOUND_EXTERNAL_ROOT"

#: Documented layout under ``$KBOUND_EXTERNAL_ROOT``.  Keys are the names used by
#: the runners; values are the relative sub-paths.
EXTERNAL_LAYOUT = {
    "wilds": "datasets/wilds",
    "imagenetc": "imagenetc_local",
    "rxrx1_ckpt": "kbound_rxrx1_ckpt",
    "rxrx1_data": "kbound_rxrx1_data",
    "rxrx1_results": "kbound_rxrx1_results",
    "imagenetr_results": "kbound_inr_results",
    "scratch": "tmp",
    "torch_cache": "torch_cache",
}

# Markers that identify the repository root when walking up the tree.
_ROOT_MARKERS = ("pyproject.toml", ".git", "AGENTS.md")


class RepoRootNotFound(RuntimeError):
    """Raised when the repository root cannot be discovered."""


class DataRootError(FileNotFoundError):
    """Raised when a required external data/results root is missing."""


class ExternalRootUnset(RuntimeError):
    """Raised when ``$KBOUND_EXTERNAL_ROOT`` is needed but not set.

    Deliberately an error rather than a default.  The paths this replaces were
    one author's external SSD and home directory; guessing a replacement would
    either fail confusingly later or write gigabytes somewhere unexpected.
    """


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


def external_root() -> Path:
    """The one external volume, from ``$KBOUND_EXTERNAL_ROOT``. Never defaulted.

    Raises
    ------
    ExternalRootUnset
        If the variable is unset or empty, with the documented layout in the
        message so the caller knows what to point it at.
    """
    value = os.environ.get(EXTERNAL_ROOT_ENV, "").strip()
    if not value:
        layout = "\n".join(f"    ${EXTERNAL_ROOT_ENV}/{sub:<24} ({key})" for key, sub in EXTERNAL_LAYOUT.items())
        raise ExternalRootUnset(
            f"{EXTERNAL_ROOT_ENV} is not set. This run needs data that is deliberately "
            "not in the git release (raw datasets, checkpoints, caches). Point "
            f"{EXTERNAL_ROOT_ENV} at the volume that holds them; the expected layout is:\n"
            f"{layout}\n"
            "See docs/research/kbound/EXTERNAL_STORAGE_POLICY.md and DATA.md for how to "
            "obtain each one. There is no default: this used to be one author's external "
            "SSD, and silently substituting $HOME would write gigabytes somewhere you did "
            "not choose."
        )
    return Path(value).expanduser().resolve()


def external_path(key: str, *parts: str) -> Path:
    """Resolve ``EXTERNAL_LAYOUT[key]`` (plus ``parts``) under the external root."""
    if key not in EXTERNAL_LAYOUT:
        raise KeyError(f"unknown external location {key!r}; known: {sorted(EXTERNAL_LAYOUT)}")
    return external_root().joinpath(EXTERNAL_LAYOUT[key], *parts)


def interpreter() -> str:
    """Interpreter for spawned runs (``$KBOUND_PYTHON``, default ``python3``).

    Replaces the hard-coded ``/Users/pratik_n/.venv_wilds/bin/python`` and
    ``/opt/anaconda3/envs/aetta/bin/python`` in the runbooks.  Unlike
    :func:`external_root` this has a sane default, so it does not raise.
    """
    return os.environ.get("KBOUND_PYTHON", "").strip() or "python3"


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
