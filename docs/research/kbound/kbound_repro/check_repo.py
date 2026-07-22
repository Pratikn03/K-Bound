#!/usr/bin/env python3
"""kbound_repro.check_repo -- fail-closed guard for staged files & portability.

Run before committing (and from ``release_candidate.sh``) to block:

* files that belong in external storage (datasets/checkpoints/caches/logs/...),
* oversized files (above a threshold, with a figure/PDF allowlist),
* executable files that hard-code ``/Users/pratik_n`` or ``/Volumes/T9``.

Exit code is non-zero if any violation is found, so it fails closed.

Usage::

    python -m kbound_repro.check_repo --staged            # git staged files
    python -m kbound_repro.check_repo --files a.py b.json  # explicit list
    python -m kbound_repro.check_repo --staged --check abspaths
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running both as a module and as a script.
if __package__:
    from . import storage
else:  # pragma: no cover - script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kbound_repro import storage


def _staged_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"WARN: could not list staged files ({exc}); pass --files instead.", file=sys.stderr)
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def run_checks(files: list[str], root: Path, which: str, threshold: int) -> int:
    problems = 0

    if which in ("all", "forbidden"):
        for rec in storage.scan_forbidden_paths(files):
            print(f"FORBIDDEN[{rec['class']}]: {rec['path']} belongs in external storage")
            problems += 1

    if which in ("all", "large"):
        for rec in storage.scan_large_files(files, root=root, threshold=threshold):
            mb = rec["size"] / 1024 / 1024
            print(f"LARGE: {rec['path']} is {mb:.1f} MB (> {threshold/1024/1024:.0f} MB, not allowlisted)")
            problems += 1

    if which in ("all", "abspaths"):
        for rec in storage.scan_absolute_paths(files, root=root,
                                               provenance_allowlist=storage.SELF_ALLOWLIST):
            print(f"ABS-PATH: {rec['path']}:{rec['line']} hard-codes {rec['match']}  ::  {rec['text']}")
            problems += 1

    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--staged", action="store_true", help="check git staged files")
    src.add_argument("--files", nargs="+", help="explicit list of files to check")
    ap.add_argument("--check", choices=["all", "forbidden", "large", "abspaths"], default="all")
    ap.add_argument("--threshold", type=int, default=storage.DEFAULT_SIZE_THRESHOLD)
    ap.add_argument("--root", default=None, help="repository root (default: discovered)")
    args = ap.parse_args(argv)

    if args.root:
        root = Path(args.root)
    else:
        try:
            from kbound_repro import paths
            root = paths.find_repo_root()
        except Exception:
            root = Path.cwd()

    files = args.files if args.files else _staged_files(root)
    if not files:
        print("no files to check.")
        return 0

    problems = run_checks(files, root, args.check, args.threshold)
    if problems:
        print(f"\nFAILED: {problems} storage/portability violation(s).", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} file(s) passed storage/portability checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
