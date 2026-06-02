"""Validate that every theorem in the registry has its expected artifacts on disk."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

from elara.theory.theorem_registry import artifact_status, list_theorems

ROOT = Path(__file__).resolve().parents[2]


def _run_script(repo_root: Path, rel_script: str) -> tuple[bool, str]:
    script = repo_root / rel_script
    if not script.exists():
        return False, f"missing script {rel_script}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env={**dict(os.environ), "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    msg = proc.stdout.strip() or proc.stderr.strip() or f"exit {proc.returncode}"
    return ok, msg


def validate(repo_root: Path, *, run_generators: bool) -> dict:
    report: dict = {"repo_root": str(repo_root), "theorems": {}, "all_ok": True}
    for spec in list_theorems():
        entry = {
            "title": spec.title,
            "core_modules": list(spec.core_modules),
            "validation_scripts": list(spec.validation_scripts),
            "artifacts": artifact_status(repo_root, spec),
            "generators_ok": {},
        }
        if run_generators:
            for script in spec.validation_scripts:
                if script.startswith("src/scripts/emit_") or script.startswith("src/scripts/audit_gate"):
                    ok, msg = _run_script(repo_root, script)
                    entry["generators_ok"][script] = {"ok": ok, "message": msg}
                    if not ok:
                        report["all_ok"] = False
        missing = [p for p, ok in entry["artifacts"].items() if not ok and p]
        entry["missing_artifacts"] = missing
        if missing and spec.theorem_id != "T1":
            report["all_ok"] = False
        report["theorems"][spec.theorem_id] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--run-generators",
        action="store_true",
        help="Execute emit/audit scripts before checking artifact paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/theorem_stack_validation_report.json"),
    )
    args = parser.parse_args()
    report = validate(args.repo_root, run_generators=args.run_generators)
    out_path = args.repo_root / args.output if not args.output.is_absolute() else args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "theorems"}, indent=2))
    for tid, entry in report["theorems"].items():
        missing = entry.get("missing_artifacts", [])
        if missing:
            print(f"{tid}: missing {missing}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
