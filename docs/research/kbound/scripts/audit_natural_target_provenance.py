#!/usr/bin/env python3
"""Inventory whether candidate natural target environments were already opened.

An existing label-bearing result artifact is sufficient to mark a benchmark as
opened.  Absence of a matching file is never treated as proof that a target is
unopened; such a target remains UNKNOWN until independently verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TRACKS: dict[str, dict[str, Any]] = {
    "officehome": {
        "environments": ["Art", "Clipart", "Product", "Real_World"],
        "patterns": ["**/*officehome*/**/*.json", "**/officehome_splits.json"],
    },
    "iwildcam": {
        "environments": ["id_val", "val", "test", "camera/location groups"],
        "patterns": ["**/*iwildcam*/**/*.json", "iwildcam_kbound_RESULTS.json"],
    },
    "camelyon17": {
        "environments": ["id_val", "val", "test", "hospital groups"],
        "patterns": ["**/*camelyon*/**/*.json"],
    },
    "rxrx1": {
        "environments": ["id_test", "val", "test", "experiment groups"],
        "patterns": ["**/*rxrx1*/**/*.json", "rxrx1_internal_backup/*.csv"],
    },
    "pacs": {
        "environments": ["art_painting", "cartoon", "photo", "sketch"],
        "patterns": ["**/*pacs*/**/*.json", "pacs_seed*.json", "per_cell/pacs_*.json"],
    },
    "imagenet_r": {
        "environments": ["ImageNet-R evaluation set"],
        "patterns": ["**/*imagenetr*/**/*.json", "**/*imagenet_r*/**/*.json"],
    },
    "cifar10_1": {
        "environments": ["CIFAR-10.1 v6 evaluation set"],
        "patterns": ["**/*cifar101*/**/*.json"],
    },
    "fmow": {
        "environments": ["WILDS val/test regions and years"],
        "patterns": ["fmow_protocol_L_*.log", "**/*fmow*/**/*.json"],
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_artifacts(results: Path, patterns: list[str]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path for path in results.glob(pattern) if path.is_file())
    return sorted(matches)


def audit(results: Path) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    for name, spec in TRACKS.items():
        artifacts = matching_artifacts(results, spec["patterns"])
        if artifacts:
            status = "OPENED_BEFORE_PROSPECTIVE_CLOSURE"
            reason = "label-bearing metrics or decisions were already archived"
        else:
            status = "UNKNOWN_REQUIRES_EXTERNAL_VERIFICATION"
            reason = "no local match is not proof of unopened provenance"
        tracks[name] = {
            "status": status,
            "environments": spec["environments"],
            "reason": reason,
            "artifact_count": len(artifacts),
            "evidence": [
                {
                    "path": str(path.relative_to(results.parent.parent.parent)),
                    "sha256": sha256_file(path),
                }
                for path in artifacts[:25]
            ],
            "evidence_truncated": len(artifacts) > 25,
        }
    unopened = [name for name, row in tracks.items() if row["status"] == "UNOPENED_VERIFIED"]
    return {
        "schema_version": 1,
        "audit_rule": (
            "Existing label-bearing output marks a target opened; lack of local output does not "
            "establish unopened provenance."
        ),
        "tracks": tracks,
        "verified_unopened_tracks": unopened,
        "prospective_natural_track_available": bool(unopened),
        "verdict": (
            "eligible_unopened_target_found" if unopened else "no_verified_unopened_target_found"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "experiments/kbound/results",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-unopened", action="store_true")
    args = parser.parse_args()
    output = args.output or args.results / "natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json"
    payload = audit(args.results.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Natural target provenance: {payload['verdict']}")
    for name, row in payload["tracks"].items():
        print(f"  {name}: {row['status']} ({row['artifact_count']} artifacts)")
    print(f"wrote {output}")
    return 0 if payload["prospective_natural_track_available"] or not args.require_unopened else 2


if __name__ == "__main__":
    raise SystemExit(main())
