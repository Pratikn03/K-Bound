"""kbound_repro.storage -- repository storage policy + guardrails.

Separates the repository into:

* **tracked publication evidence** -- compact authoritative JSON, checksums,
  small generated tables/figures, protocol files, reproduction commands;
* **external storage** -- datasets, model checkpoints, large raw predictions,
  full logs, caches -- which must NOT enter ordinary commits.

Provides pure, testable helpers used by ``check_repo.py`` and
``release_candidate.sh``:

* :func:`path_class`             classify a path into a storage class;
* :func:`scan_forbidden_paths`   flag files that belong in external storage;
* :func:`scan_large_files`       flag oversized files (with an allowlist);
* :func:`scan_absolute_paths`    flag executable files hard-coding machine paths.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

__all__ = [
    "DEFAULT_SIZE_THRESHOLD",
    "FIGURE_ALLOWLIST_SUFFIXES",
    "path_class",
    "scan_forbidden_paths",
    "scan_large_files",
    "scan_absolute_paths",
]

DEFAULT_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5 MB

# Intentional binary publication artifacts allowed above the size threshold.
FIGURE_ALLOWLIST_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg"})

# Path classes -> ordered (regex, class) rules. First match wins.
# Ordering rationale:
#   1. macOS junk;
#   2. HARD-exclusion directories (venv / external repo / caches) override
#      extensions -- a .py inside .venv is a virtualenv file, not source;
#   3. binary payload extensions (npy/pt/pkl/archives/log);
#   4. text evidence / source extensions -- so a tracked result JSON inside a
#      dataset-named dir (e.g. .../wilds/result.json) is EVIDENCE, not a dataset
#      (prevents the commit guard from flagging authoritative artifacts);
#   5. SOFT dataset-directory heuristics for extension-less payloads.
_CLASS_RULES: list[tuple[str, str]] = [
    (r"(^|/)\._|(^|/)\.DS_Store$", "macos_junk"),
    (r"(^|/)\.venv(-macos)?/|(^|/)venv/", "virtualenv"),
    (r"(^|/)external/|(^|/)DomainBed/", "external_repo"),
    (r"(^|/)AETTA/cached_data/|(^|/)\.torch_cache/|(^|/)__pycache__/|(^|/)\.pytest_cache/|(^|/)\.ruff_cache/", "cache"),
    (r"\.(pt|pth|ckpt)$", "checkpoint"),
    (r"\.pkl$", "cache"),
    (r"\.(npy|npz)$", "dataset"),
    (r"\.(zip|tar|tar\.gz|tgz|tar\.bz2|bz2|xz|7z|rar)$", "archive"),
    (r"\.log$", "raw_log"),
    (r"\.(json|csv|md|yaml|yml|sha256|txt)$", "tracked_evidence"),
    (r"\.(py|sh|tex|toml|cfg|lean)$", "tracked_source"),
    (r"(^|/)AETTA/dataset/|(^|/)data/(raw|interim|processed)/|(^|/)datasets?/", "dataset"),
    (r"(^|/)CIFAR-\d+-C/|imagenet-[cr]/|(^|/)PACS/|(^|/)wilds/", "dataset"),
    (r"(^|/)checkpoints?/", "checkpoint"),
    (r"(^|/)logs?/", "raw_log"),
]


def path_class(path: str) -> str:
    """Classify ``path`` into one storage class (first matching rule wins)."""
    p = str(path)
    for pattern, cls in _CLASS_RULES:
        if re.search(pattern, p):
            return cls
    return "unknown"


#: Classes that must never be committed in an ordinary commit.
_EXTERNAL_CLASSES = frozenset(
    {"cache", "dataset", "checkpoint", "raw_log", "archive", "virtualenv", "external_repo", "macos_junk"}
)


def scan_forbidden_paths(paths: Iterable[str]) -> list[dict]:
    """Return records for paths whose class belongs in external storage."""
    out = []
    for p in paths:
        cls = path_class(p)
        if cls in _EXTERNAL_CLASSES:
            out.append({"path": str(p), "class": cls})
    return out


def scan_large_files(
    files: Iterable[str],
    *,
    root: str | Path = ".",
    threshold: int = DEFAULT_SIZE_THRESHOLD,
    allowlist_suffixes: Iterable[str] = FIGURE_ALLOWLIST_SUFFIXES,
) -> list[dict]:
    """Flag files larger than ``threshold`` unless they are allowlisted figures.

    Missing files are skipped (a staged path may not exist on disk in a dry run).
    """
    root = Path(root)
    allow = {s.lower() for s in allowlist_suffixes}
    flagged = []
    for f in files:
        fp = (root / f) if not Path(f).is_absolute() else Path(f)
        try:
            size = fp.stat().st_size
        except OSError:
            continue
        if size > threshold and fp.suffix.lower() not in allow:
            flagged.append({"path": str(f), "size": size, "suffix": fp.suffix})
    return flagged


# Executable machine-specific path patterns (Phase 2 portability guard).
_ABS_PATTERNS = [
    r"/Users/pratik_n",
    r"/Volumes/T9",
]
_ABS_RE = re.compile("|".join(_ABS_PATTERNS))
_EXEC_SUFFIXES = frozenset({".py", ".sh", ".toml", ".cfg", ".yaml", ".yml"})

#: The guard's own detector/doc files legitimately contain the forbidden path
#: strings (as patterns and documentation). They are not path *dependencies*.
SELF_ALLOWLIST = frozenset({
    "docs/research/kbound/kbound_repro/storage.py",
    "docs/research/kbound/kbound_repro/paths.py",
    "docs/research/kbound/kbound_repro/check_repo.py",
})


def scan_absolute_paths(
    files: Iterable[str],
    *,
    root: str | Path = ".",
    provenance_allowlist: Iterable[str] = (),
) -> list[dict]:
    """Flag executable files that hard-code ``/Users/pratik_n`` or ``/Volumes/T9``.

    ``provenance_allowlist`` names files where such a string is an immutable
    historical *provenance* field (non-executable metadata) and is allowed.
    """
    root = Path(root)
    allow = set(provenance_allowlist)
    flagged = []
    for f in files:
        if f in allow:
            continue
        fp = (root / f) if not Path(f).is_absolute() else Path(f)
        if fp.suffix.lower() not in _EXEC_SUFFIXES:
            continue
        try:
            text = fp.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = _ABS_RE.search(line)
            if m:
                flagged.append({"path": str(f), "line": i, "match": m.group(0),
                                "text": line.strip()[:120]})
    return flagged
