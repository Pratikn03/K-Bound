#!/usr/bin/env python3
"""Orchestrate Scenario C Phase 2 (M2 v2 MulSen) + Phase 3 (M3/M4/tier evidence)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", action="store_true", help="Download/seal/run MulSen M2 v2")
    parser.add_argument("--phase3", action="store_true", help="M3 confirmatory + M4 audit + stats")
    parser.add_argument("--download-mulsen", action="store_true")
    parser.add_argument("--skip-mulsen-confirmatory", action="store_true")
    parser.add_argument("--skip-m3-confirmatory", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if not args.phase2 and not args.phase3:
        args.phase2 = args.phase3 = True

    root = _repo_root()
    py = sys.executable
    log: dict = {"started_utc": datetime.now(timezone.utc).isoformat(), "steps": []}
    rc = 0

    def step(name: str, cmd: list[str]) -> None:
        nonlocal rc
        code = _run(cmd, root, name)
        log["steps"].append({"name": name, "exit_code": code})
        rc = rc or code

    if args.phase2:
        if args.download_mulsen:
            step("download_mulsen", [py, "src/scripts/scenario_c/download_m2_external_mulsen.py"])
        step(
            "seal_mulsen",
            [
                py,
                "src/scripts/scenario_c/seal_m2_external_mulsen.py",
                *(["--skip-download"] if not args.download_mulsen else []),
            ],
        )
        if not args.skip_mulsen_confirmatory:
            step(
                "m2_v2_confirmatory",
                [
                    py,
                    "src/scripts/scenario_c/run_m2_external_v2_confirmatory.py",
                    "--seeds",
                    *[str(s) for s in args.seeds],
                    "--skip-stats",
                ],
            )

    if args.phase3:
        if not args.skip_m3_confirmatory:
            step(
                "m3_confirmatory",
                [
                    py,
                    "src/scripts/scenario_c/run_m3_healthcare_confirmatory.py",
                    "--seeds",
                    *[str(s) for s in args.seeds],
                    "--skip-stats",
                ],
            )
        step("m4_audit", [py, "src/scripts/scenario_c/run_m4_temporal_monitoring_audit.py"])
        step("confirmatory_stats", [py, "src/scripts/scenario_c/confirmatory_statistics.py", "--write-report"])
        step("checklist", [py, "src/scripts/scenario_c/audit_checklist_progress.py"])
        step("snapshot", [str(root / "research_dashboard/build/elara_research_snapshot"), "--repo-root", str(root)])

    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["exit_code"] = rc
    out = root / "elara_master_c/audits/phase23_flagship_execution_log.json"
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Log -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
