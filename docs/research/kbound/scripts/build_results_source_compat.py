#!/usr/bin/env python3
"""Generate the legacy results_source.json view from the table manifest.

This file remains only for older tests/tools.  New claim authority is
RESULT_MANIFEST.json; paper table values live in paper/generated/kbound_result_manifest.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "docs/research/kbound/paper/generated/kbound_result_manifest.json"
CLAIMS = ROOT / "docs/research/kbound/RESULT_MANIFEST.json"
OUT = ROOT / "docs/research/kbound/results_source.json"


def main() -> None:
    table = json.loads(TABLE.read_text())
    payload = {
        "_README": (
            "GENERATED COMPATIBILITY VIEW. Natural-shift promoted values are OUT-OF-FOLD; "
            "superseded in-sample-radius wins are not authoritative. Wording/status authority "
            "is claim_ledger.json; promoted claim backing is RESULT_MANIFEST.json; table values "
            "are generated from paper/generated/kbound_result_manifest.json."
        ),
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_claim_manifest": CLAIMS.relative_to(ROOT).as_posix(),
        "canonical_table_manifest": TABLE.relative_to(ROOT).as_posix(),
        "alpha": table.get("alpha"),
        "tracks": table.get("tracks", {}),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
