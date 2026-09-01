#!/usr/bin/env python3
"""Refresh and validate the bounded, mutable part of STORAGE_MANIFEST.json.

The storage manifest contains two different kinds of hashes:

* immutable historical evidence/checkpoint seals, which this script must never
  rewrite; and
* four tracked release authorities that are deterministically regenerated during
  a release refresh.

Only the latter set is refreshable.  This prevents a broad "hash whatever is on
disk" command from blessing an accidental change to historical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "docs/research/kbound/STORAGE_MANIFEST.json"
REFRESHABLE_AUTHORITIES = {
    "docs/research/kbound/claim_ledger.json",
    "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
    "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value.get("schema_version") != "kbound-storage-manifest-v1":
        raise ValueError("unsupported storage-manifest schema")
    if not isinstance(value.get("artifacts"), list):
        raise ValueError("storage manifest must contain an artifacts list")
    return value


def direct_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest["artifacts"]:
        location = row.get("expected_location")
        if not isinstance(location, str) or location.startswith("$"):
            continue
        if location in rows:
            raise ValueError(f"duplicate direct storage row: {location}")
        rows[location] = row
    missing = sorted(REFRESHABLE_AUTHORITIES - rows.keys())
    if missing:
        raise ValueError(f"storage manifest is missing refreshable authorities: {missing}")
    return rows


def refresh(manifest: dict[str, Any]) -> None:
    rows = direct_rows(manifest)
    for location in sorted(REFRESHABLE_AUTHORITIES):
        path = ROOT / location
        if not path.is_file():
            raise FileNotFoundError(f"refreshable authority is missing: {location}")
        row = rows[location]
        if row.get("tracked") is not True:
            raise ValueError(f"refreshable authority is not marked tracked: {location}")
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = sha256(path)
    rows[
        "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
    ]["scope"] = (
        "Retrospective six-corruption-family current-policy sensitivity; ordinary intervals "
        "are unadjusted. Retrospective Holm adjustment over the six prospectively named "
        "contrasts is non-confirmatory and no candidate passes both baselines."
    )
    rows[
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
    ]["drift_note"] = (
        "The current canonical authority is regenerated from archived panel inputs with "
        "current generator/source identities. Updating those identities changes the artifact "
        "hash even if panel measurements agree. Current-policy six-family sensitivity remains "
        "a separately hashed artifact synchronized into the paper manifests. Historical "
        "provenance-audit expectations and verdicts are retained; this refreshed row is not "
        "itself a numerical-invariance test or a new experimental result."
    )
    manifest["generated_by"] = (
        "mechanically refreshed by docs/research/kbound/scripts/"
        "refresh_storage_manifest.py from the four declared tracked release authorities; "
        "historical evidence seals are never rewritten; this remains a working-copy snapshot, "
        "not a clean-commit or final outer-checksum attestation"
    )


def validate(manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    rows = direct_rows(manifest)
    for location, row in rows.items():
        expected_hash = row.get("sha256")
        if expected_hash is None:
            continue
        path = ROOT / location
        if not path.is_file():
            problems.append(f"direct storage artifact is missing: {location}")
            continue
        if path.stat().st_size != row.get("size_bytes"):
            problems.append(f"direct storage artifact byte count is stale: {location}")
        if sha256(path) != expected_hash:
            problems.append(f"direct storage artifact SHA-256 is stale: {location}")

    sealed = manifest.get("sealed_evidence_checksums")
    if not isinstance(sealed, dict):
        problems.append("sealed_evidence_checksums must be an object")
        return problems
    counts = {"present": 0, "absent": 0}
    for location, row in sealed.items():
        status = str(row.get("status", "")).lower()
        if status not in counts:
            problems.append(f"sealed evidence has invalid status: {location}")
            continue
        counts[status] += 1
        path = ROOT / location
        if status == "absent":
            if path.exists():
                problems.append(f"sealed evidence is unexpectedly present: {location}")
            continue
        if not path.is_file():
            problems.append(f"sealed evidence is missing: {location}")
            continue
        if path.stat().st_size != row.get("size_bytes"):
            problems.append(f"sealed evidence byte count is stale: {location}")
        if sha256(path) != row.get("sha256"):
            problems.append(f"sealed evidence SHA-256 is stale: {location}")

    summary = manifest.get("sealed_evidence_summary") or {}
    observed = {
        "files": len(sealed),
        "present": counts["present"],
        "absent": counts["absent"],
    }
    for field, value in observed.items():
        if summary.get(field) != value:
            problems.append(
                f"sealed evidence summary {field} is stale: "
                f"expected {summary.get(field)!r}, observed {value!r}"
            )

    for row in manifest.get("unsealed_present_artifacts", []):
        location = row.get("path")
        path = ROOT / str(location)
        if row.get("status") != "present_unsealed":
            problems.append(f"unsealed artifact has invalid status: {location}")
            continue
        if not path.is_file():
            problems.append(f"unsealed artifact is missing: {location}")
            continue
        if path.stat().st_size != row.get("current_bytes"):
            problems.append(f"unsealed artifact byte count is stale: {location}")
        if sha256(path) != row.get("current_sha256"):
            problems.append(f"unsealed artifact SHA-256 is stale: {location}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="refresh only the four declared mutable authority rows before validating",
    )
    args = parser.parse_args()
    manifest = load_manifest()
    if args.write:
        refresh(manifest)
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    problems = validate(manifest)
    if problems:
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(
        "storage manifest: PASS "
        f"({len(REFRESHABLE_AUTHORITIES)} refreshable authorities; historical seals unchanged)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
