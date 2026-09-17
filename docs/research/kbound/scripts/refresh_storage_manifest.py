#!/usr/bin/env python3
"""Refresh hashes for the small set of declared mutable release authorities.

This utility only updates byte counts and SHA-256 fields in the storage
manifest.  It never changes status, historical seals, or scientific values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "docs/research/kbound/STORAGE_MANIFEST.json"

# These are the four current release authorities named by the manifest's
# generation policy.  Historical result rows and sealed evidence are excluded.
REFRESHABLE_AUTHORITIES = frozenset(
    {
        "docs/research/kbound/claim_ledger.json",
        "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
        "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("artifacts"), list):
        raise ValueError("storage manifest must contain an artifacts list")
    return value


def direct_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("artifacts", []):
        if not isinstance(row, dict):
            continue
        location = row.get("expected_location")
        if isinstance(location, str) and not location.startswith("$"):
            if location in rows:
                raise ValueError(f"duplicate storage authority location: {location}")
            rows[location] = row
    return rows


def refresh(manifest: dict[str, Any]) -> None:
    rows = direct_rows(manifest)
    for location in REFRESHABLE_AUTHORITIES:
        path = ROOT / location
        if not path.is_file():
            raise FileNotFoundError(location)
        row = rows.get(location)
        if row is None:
            raise KeyError(f"refreshable authority is not declared: {location}")
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = sha256(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-public-only", action="store_true",
                        help="refresh only the four declared public authorities; never follow other entries")
    parser.parse_args(argv)
    manifest = load_manifest(MANIFEST)
    refresh(manifest)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"refreshed {len(REFRESHABLE_AUTHORITIES)} declared authorities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
