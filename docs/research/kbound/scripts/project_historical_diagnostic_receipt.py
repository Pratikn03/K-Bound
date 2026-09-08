#!/usr/bin/env python3
"""Create one path-only companion for the pinned historical diagnostic receipt.

This reads only the supplied receipt. No diagnostics are recomputed, no linked
artifact is opened, and no K-Bound confirmation is performed. Output is created
exclusively; the original receipt and its historical assertions are preserved.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

ORIGINAL_SHA256 = "3e53221320370bb6e62f7e1955e2ecdb37822a2d9613f28960b8cf09d44a27c8"
ORIGINAL_BYTES = 1122
LOGICAL_PATH = "experiments/audit/historical_diagnostic_recovery_receipt.json"
REPLACEMENT = "private-workspace/recovered-historical-checkout"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


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
                raise ValueError("Unreviewed private path in dictionary key")
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)
    elif _private_path(value):
        raise ValueError("Unreviewed private path in value")


def redact_payload(document):
    if not isinstance(document, dict) or not _private_path(document.get("source_checkout")):
        raise ValueError("Missing or malformed historical source checkout")
    payload = copy.deepcopy(document)
    original_path = payload["source_checkout"]
    payload["source_checkout"] = REPLACEMENT
    _reject_private(payload)
    redactions = [{
        "kind": "string_value",
        "json_pointer": "/source_checkout",
        "replacement_value": REPLACEMENT,
        "redacted_utf8_sha256": sha(original_path.encode("utf-8")),
        "reason": "Private checkout path relabeled; no claim that the checkout is distributed.",
    }]
    return payload, redactions


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite JSON constant in recovery receipt")


def project_bytes(raw):
    if len(raw) != ORIGINAL_BYTES or sha(raw) != ORIGINAL_SHA256:
        raise ValueError("Original bytes do not match the approved historical receipt")
    document = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    payload, redactions = redact_payload(document)
    return {
        "schema": "portable-historical-diagnostic-recovery-receipt-v1",
        "projection_version": 1,
        "activity": "PATH_ONLY_PROJECTION_NO_METRIC_RECOMPUTATION_OR_KBOUND_CONFIRMATION",
        "original_receipt": {"logical_path": LOGICAL_PATH, "sha256": ORIGINAL_SHA256,
                             "bytes": ORIGINAL_BYTES, "original_bytes_included": False},
        "historical_receipt": payload,
        "redactions": redactions,
        "redaction_count": len(redactions),
        "projected_payload_sha256": sha(canonical(payload)),
        "preservation": "All original fields and values are preserved except the explicitly listed private source_checkout location. Artifact hashes, byte counts, blank-field assertions, historical classification, scope and negative claims are unchanged.",
        "interpretation": "This companion preserves recovered historical assertions; it does not recompute or endorse diagnostics. The redacted workspace locator does not identify a delivered directory. The CSV and label-semantics artifacts retain their original exact bytes, including blank fields and historical NaN tokens.",
        "original_receipt_modified": False,
        "metric_recomputation_performed": False,
        "kbound_confirmation": False,
    }


def render(projected):
    return (json.dumps(projected, indent=2, sort_keys=True, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    projected = render(project_bytes(args.source.read_bytes()))
    with args.output.open("xb") as output:
        output.write(projected)


if __name__ == "__main__":
    main()
