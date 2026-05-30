#!/usr/bin/env python3
"""Poll audit JSON mtimes and re-run elara_research_snapshot when they change."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def repo_root_from(start: Path) -> Path:
    start = start.resolve()
    for parent in [start, *start.parents]:
        marker = parent / "elara_master_c" / "audits" / "checklist_progress.json"
        if marker.is_file():
            return parent
    raise SystemExit("Could not find repo root (checklist_progress.json missing)")


def watch_paths(root: Path) -> list[Path]:
    audits = root / "elara_master_c" / "audits"
    return [
        audits / "checklist_progress.json",
        audits / "confirmatory_statistics_report.json",
        audits / "python_file_catalog.json",
        root / "research_lock" / "FLAGSHIP_DEV_PROTOCOL_v1.yaml",
    ]


def max_mtime(paths: list[Path]) -> float:
    mt = 0.0
    for p in paths:
        if p.is_file():
            mt = max(mt, p.stat().st_mtime)
    return mt


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch audits and refresh research dashboard snapshot")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval seconds")
    parser.add_argument("--binary", type=Path, default=None, help="Path to elara_research_snapshot")
    args = parser.parse_args()

    root = args.repo_root or repo_root_from(Path(__file__).resolve())
    binary = args.binary or (root / "research_dashboard" / "build" / "elara_research_snapshot")
    if not binary.is_file():
        print(f"Missing aggregator binary: {binary}\nRun research_dashboard/build.sh first.", file=sys.stderr)
        return 1

    paths = watch_paths(root)
    last = max_mtime(paths)
    print(f"Watching {len(paths)} files under {root} (interval={args.interval}s)")
    print(f"Aggregator: {binary}")

    while True:
        time.sleep(args.interval)
        current = max_mtime(paths)
        if current <= last:
            continue
        last = current
        print("\n--- audit change detected, regenerating snapshot ---")
        proc = subprocess.run(
            [str(binary), "--repo-root", str(root)],
            cwd=root,
        )
        if proc.returncode != 0:
            print(f"Aggregator exited {proc.returncode}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
