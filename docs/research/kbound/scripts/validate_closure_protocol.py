#!/usr/bin/env python3
"""Validate and optionally hash the prospective K-Bound closure protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kga.experiment_contract import (  # noqa: E402
    ContractError,
    load_protocol,
    protocol_sha256,
    validate_protocol,
)

DEFAULT_PROTOCOL = ROOT / "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml"


def _git_sha() -> str:
    cp = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return cp.stdout.strip() if cp.returncode == 0 else "UNAVAILABLE"


def _report(document: dict, path: Path, *, require_sealed: bool) -> dict:
    errors = validate_protocol(document, require_sealed=require_sealed)
    primary = document.get("primary_natural_track") or {}
    compatibility = document.get("launcher_compatibility") or {}
    return {
        "schema_version": 1,
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_path": str(path.relative_to(ROOT)),
        "protocol_id": document.get("protocol_id"),
        "protocol_status": document.get("status"),
        "protocol_sha256": protocol_sha256(document),
        "git_sha": _git_sha(),
        "require_sealed": require_sealed,
        "valid": not errors,
        "errors": errors,
        "primary_dataset": primary.get("dataset"),
        "primary_provenance_status": primary.get("provenance_status"),
        "launcher_compatibility": compatibility,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--require-sealed", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--write-lock",
        type=Path,
        help="write a protocol lock JSON; allowed only when --require-sealed passes",
    )
    args = parser.parse_args()

    try:
        document = load_protocol(args.protocol)
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = _report(document, args.protocol.resolve(), require_sealed=args.require_sealed)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")

    if args.write_lock:
        if not args.require_sealed:
            print("ERROR: --write-lock requires --require-sealed", file=sys.stderr)
            return 2
        if not report["valid"]:
            print("ERROR: refusing to lock an invalid or unsealed protocol", file=sys.stderr)
            return 2
        if args.write_lock.exists():
            print(f"ERROR: refusing to overwrite existing lock {args.write_lock}", file=sys.stderr)
            return 2
        lock = {
            "schema_version": 1,
            "protocol_id": report["protocol_id"],
            "protocol_sha256": report["protocol_sha256"],
            "git_sha": report["git_sha"],
            "sealed_utc": report["checked_utc"],
            "protocol_path": report["protocol_path"],
        }
        args.write_lock.parent.mkdir(parents=True, exist_ok=True)
        args.write_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"WROTE {args.write_lock}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
