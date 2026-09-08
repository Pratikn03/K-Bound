#!/usr/bin/env python3
"""One-time projection of four approved opened records into portable inputs.

Source hashes bind exact opened records; extraction never validates their
counts against original labels. Only the explicit sufficient statistics leave
the source objects. This tool cannot select a new outcome subset.
"""

import argparse
import json
from pathlib import Path

from empirical_macros import canonical, compute, digest, parse

SOURCES = {
    "entropy": (
        "domainnet_entropy_development/result.json",
        "a48d8011e26fe3991342cb4118e33dafa36134dd4ae6b3b32431c6facf3619b9",
    ),
    "bridge": (
        "domainnet_adacontrast_development/result.json",
        "741605359bde9f79cc616f34d6d70e2c2c6c4110fe75da226e8837060378946c",
    ),
    "officehome": (
        "officehome_opened_archive/test_seed234.json",
        "a184f085bb19af055eefe3a3b5209eb2e82047c1c8f82860a70c0a358e2f31da",
    ),
    "smoke": (
        "officehome_mechanics_smoke/completion.json",
        "2e0cfcbaf7561a3e76ee1bf98e1d9bd1085784b404e22bfb982aa87b1c786258",
    ),
}


def select_domainnet(data):
    fields = ("cell_id", "n_images", "frozen_correct", "candidate_correct", "action")
    return [{k: row[k] for k in fields} for row in data["cells"]]


def select_officehome(data):
    names = data["metadata"]["evidence_names"]
    if names.count("update_norm") != 1:
        raise ValueError("Ambiguous update_norm coordinate")
    index = names.index("update_norm")
    rows = []
    for r in data["records"]:
        if r["candidate"] != "sar_online_aggressive" or r["metric"] != "accuracy" or r["split"] != "test":
            raise ValueError("Unexpected historical candidate, metric or split")
        identity = "|".join(str(r[k]) for k in ("domain", "comp", "regime", "seed", "candidate", "metric", "split"))
        rows.append({"cell_id": identity, "a0": r["a0"], "a_adapted": r["a_adapted"], "update_norm": r["Z"][index]})
    return rows


def extract(paths, output):
    payloads = {}
    bindings = {}
    for role, (source_id, expected_sha) in SOURCES.items():
        raw = Path(paths[role]).read_bytes()
        if digest(raw) != expected_sha:
            raise ValueError("Input is not the approved source identity: " + role)
        data = parse(raw)
        rows = (
            select_domainnet(data)
            if role in ("entropy", "bridge")
            else select_officehome(data)
            if role == "officehome"
            else [{"cell_id": "whole-run", "wall_seconds": data["wall_seconds"]}]
        )
        for row in rows:
            row["content_id"] = digest(canonical(row))
        payload = (
            json.dumps(
                {"schema": "kbound-empirical-macro-extract-v1", "role": role, "rows": rows},
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        payloads[role + ".json"] = payload
        bindings[role] = {
            "path": role + ".json",
            "sha256": digest(payload.encode()),
            "source_id": source_id,
            "original_sha256": expected_sha,
            "row_count": len(rows),
            "ordered_cell_ids": [r["cell_id"] for r in rows],
        }
    manifest = {
        "schema": "kbound-empirical-macro-bundle-v1",
        "extraction_version": 1,
        "records": bindings,
        "mapping": {
            "domainnet": "cells[*]: cell_id,n_images,frozen_correct,candidate_correct,action; original order; all 24 cells per record.",
            "officehome": "records[*]: identity=(domain,comp,regime,seed,candidate,metric,split),a0,a_adapted,Z[index of metadata.evidence_names update_norm]; original order; all 54 existing rows.",
            "smoke": "completion.wall_seconds only; whole-run timing, not per-arm latency.",
            "numeric_representation": "JSON numbers round-trip to the same Python binary64 values used by the prior validator; no tolerance or numeric tuning. Content IDs hash canonical projected JSON, not original lexical number tokens.",
        },
        "arithmetic": {
            "weighting": "equal-cell, not image-weighted",
            "sum": "math.fsum in source record order",
            "delta": "(candidate_correct-frozen_correct)/n_images",
            "gate_regret_pp": "100 * mean(max(delta,0) - delta*(action==ADAPT))",
            "adapt_regret_pp": "100 * mean(max(-delta,0))",
            "zero_rule_regret": "mean(max(a_adapted-a0,0) - (a_adapted-a0)*(update_norm==0.0))",
            "rounding": "DomainNet 6 decimals; OfficeHome regret 8; whole-run seconds 2",
        },
        "evidence_scope": "Opened development panels and retrospective archive; mechanics smoke only. No new outcome selection, confirmation, population protection, or KGA evaluation implied by smoke.",
        "trust_limits": "Original hashes establish source identity, not original count correctness. Extract hashes/content IDs establish projection integrity. Manifest hash must be pinned by caller from a trusted receipt. No raw labels/images/predictions/checkpoints are distributed.",
    }
    payloads["manifest.json"] = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("x", encoding="ascii") as handle:
            handle.write(payload)
    manifest_hash = digest(payloads["manifest.json"].encode())
    values = compute(output, manifest_hash)
    receipt = {
        "schema": "kbound-six-macro-projection-receipt-v1",
        "manifest_sha256": manifest_hash,
        "macro_values": values,
        "scope": "Projection and arithmetic verification only; original count correctness not independently established by checksum.",
        "logical_bundle": "bundle",
        "generator": "empirical_macros.py",
        "extractor": "extract_macro_bundle.py",
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in SOURCES:
        parser.add_argument("--" + role, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = extract({role: getattr(args, role) for role in SOURCES}, args.output)
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
