#!/usr/bin/env python3
"""Retired synchronization entrypoint; no evidence or optional runtime access."""
import sys

RETIREMENT_MESSAGE = (
    "Retired synchronization helper: use the audited release route: "
    "bash docs/research/kbound/runbooks/release_candidate.sh all. "
    "This command starts no experiment and changes no results."
)


def run_sync():
    raise RuntimeError(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    print(RETIREMENT_MESSAGE, file=sys.stderr)
    raise SystemExit(2)
