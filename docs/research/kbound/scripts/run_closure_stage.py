#!/usr/bin/env python3
"""Run allowlisted structured commands from a sealed closure protocol."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kga.experiment_contract import (  # noqa: E402
    ContractError,
    assert_valid,
    load_protocol,
    protocol_sha256,
    validate_protocol,
)

DEFAULT_PROTOCOL = ROOT / "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml"


def _expand(value: str) -> str:
    return os.path.expandvars(value.replace("{repo}", str(ROOT)))


def _verify_lock(document: dict, lock_path: Path) -> None:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read protocol lock {lock_path}: {exc}") from exc
    expected = protocol_sha256(document)
    if lock.get("protocol_sha256") != expected:
        raise ContractError("protocol hash does not match the sealed lock; resealing after test access is forbidden")
    if lock.get("protocol_id") != document.get("protocol_id"):
        raise ContractError("protocol ID does not match the sealed lock")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run commands; default is dry-run")
    args = parser.parse_args()

    try:
        document = load_protocol(args.protocol)
        assert_valid(validate_protocol(document, require_sealed=True))
        _verify_lock(document, args.lock)
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    commands = document["execution"][args.stage]
    for index, spec in enumerate(commands, start=1):
        argv = [_expand(part) for part in spec["argv"]]
        cwd = Path(_expand(spec.get("cwd", "{repo}"))).resolve()
        try:
            cwd.relative_to(ROOT)
        except ValueError:
            print(f"ERROR: command cwd escapes repository: {cwd}", file=sys.stderr)
            return 2
        print(f"[{index}/{len(commands)}] {spec['name']}")
        print(f"  cwd: {cwd}")
        print(f"  argv: {shlex.join(argv)}")
        if args.execute:
            subprocess.run(argv, cwd=cwd, check=True)

    if not args.execute:
        print("DRY RUN ONLY. Re-run through the closure wrapper with KBOUND_EXECUTE=1 to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
