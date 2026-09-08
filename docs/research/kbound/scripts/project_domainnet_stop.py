#!/usr/bin/env python3
"""Create a path-only portable companion to the pinned pre-model DomainNet STOP.

Reads only the supplied original STOP, never datasets, linked receipts or model
outcomes. Historical evidence and production identities remain unchanged.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

ORIGINAL_SHA256 = "164c276ab5db4c720964d54c4ea0f44455c8a32e3b36c1b88c2a5c815ee1db61"
ORIGINAL_BYTES = 4244
LOCATIONS = (
    ("preparation", "directory", "private-workspace/painting-v1-preparation"),
    ("independent_verification", "receipt", "private-artifact/painting-v1-independent-verification"),
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _private_path(value):
    return isinstance(value, str) and bool(
        value.startswith(("/", "~/", "\\\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or re.search(r"(?:^|\s)/(?:Volumes|Users|home|private|tmp)/", value)
    )


def _reject_private(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if _private_path(key):
                raise ValueError("Unmapped private path in dictionary key")
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)
    elif _private_path(value):
        raise ValueError("Unmapped private path in value")


def redact_payload(document):
    """Relabel exactly two declared locators, preserving every other JSON value."""
    if not isinstance(document, dict):
        raise ValueError("Expected historical STOP object")
    payload = copy.deepcopy(document)
    redactions = []
    for parent, key, label in LOCATIONS:
        container = payload.get(parent)
        if not isinstance(container, dict) or not _private_path(container.get(key)):
            raise ValueError("Missing or nonpath historical location: " + parent + "/" + key)
        value = container[key]
        container[key] = label
        redactions.append({
            "kind": "string_value",
            "json_pointer": "/" + parent + "/" + key,
            "replacement_value": label,
            "redacted_utf8_sha256": _sha(value.encode("utf-8")),
            "reason": "Private machine location relabeled; linked-artifact hashes are unchanged.",
        })
    _reject_private(payload)
    return payload, redactions


def project_bytes(raw):
    if len(raw) != ORIGINAL_BYTES or _sha(raw) != ORIGINAL_SHA256:
        raise ValueError("Original bytes do not match the approved historical STOP")
    payload, redactions = redact_payload(json.loads(raw))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return {
        "schema": "kbound-portable-domainnet-development-stop/1",
        "projection_version": 1,
        "activity": "PATH_ONLY_PROJECTION_NO_NEW_DATA_OR_MODEL_EXECUTION",
        "original_stop": {
            "logical_path": "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_v1_STOP.json",
            "sha256": ORIGINAL_SHA256,
            "bytes": ORIGINAL_BYTES,
            "original_bytes_included": False,
        },
        "historical_stop": payload,
        "redactions": redactions,
        "redaction_count": len(redactions),
        "projected_payload_sha256": _sha(canonical),
        "preservation": "All original fields and values are preserved except the two explicitly listed private location values. Status, stage, scientific outcome, counts, booleans, hashes, historical assertions and limitations are unchanged.",
        "interpretation": "This companion preserves a historical PRE_MODEL STOP_SOURCE_OVERLAP with scientific outcome NOT_TESTED. It is not a new data check, model run, scientific result or production protocol authority. Private-workspace/private-artifact labels are redacted locators, not delivered paths. Original and linked-artifact hashes identify bytes without establishing public availability or historical target nonaccess.",
        "new_data_or_model_execution_performed": False,
        "original_stop_modified": False,
    }


def render(projected):
    return (json.dumps(projected, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    projected = render(project_bytes(args.source.read_bytes()))
    with args.output.open("xb") as handle:
        handle.write(projected)


if __name__ == "__main__":
    main()
