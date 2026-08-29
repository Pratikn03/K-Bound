#!/usr/bin/env python3
"""Convert official POEM / AETTA outputs into the per-condition decision JSON that
scripts/official_baselines_headtohead.py ingests via --decisions.

Strict no-fabrication policy: every condition in the locked stream must be present in the
official logs, or this converter exits with an error listing the missing conditions.

Pre-registered decision mappings (CAMERA_READY_RUNBOOK.md, Item 11):
  POEM : condition -> "freeze" if the official protector's martingale fired on that
         condition's batch sequence, else "adapt".  (No abstain; decisive rate 1.0.)
  AETTA: condition -> "adapt" iff official dropout est_acc(adapted) > est_acc(frozen).

Input --logs formats:
  poem : JSON {condition: {"fired": bool}}  or  {condition: "adapt"|"freeze"}
  aetta: directory of the vendored repo's eval_results logs, or a JSON
         {condition: {"est_acc_adapted": float, "est_acc_frozen": float}}
Output: JSON {condition: "adapt"|"freeze"|"abstain"} over the exact locked stream order.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def load_stream_conditions(stream_path: str) -> list[str]:
    with open(stream_path) as f:
        recs = json.load(f)["records"]
    conditions = [r.get("condition", "") for r in recs]
    if not conditions or any(not condition for condition in conditions):
        sys.exit("[convert] stream contains no conditions or an empty condition identifier")
    if len(conditions) != len(set(conditions)):
        sys.exit("[convert] stream condition identifiers are not unique")
    return conditions


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_poem(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for cond, v in raw.items():
        if isinstance(v, str):
            out[cond] = v
        elif isinstance(v, dict) and "fired" in v:
            out[cond] = "freeze" if v["fired"] else "adapt"
        else:
            sys.exit(f"[convert] unrecognized POEM record for condition {cond!r}: {v!r}")
    return out


def parse_aetta(path: str) -> dict:
    if os.path.isdir(path):
        sys.exit(
            "[convert] AETTA eval_results directory parsing depends on the vendored repo's "
            "log layout for your run config. Export a JSON "
            '{condition: {"est_acc_adapted": x, "est_acc_frozen": y}} from those logs '
            "(see AETTA/print_est.py) and pass it to --logs instead."
        )
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for cond, v in raw.items():
        if isinstance(v, str):
            out[cond] = v
        else:
            out[cond] = "adapt" if float(v["est_acc_adapted"]) > float(v["est_acc_frozen"]) else "freeze"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["poem", "aetta"], required=True)
    ap.add_argument("--logs", required=True, help="official output (JSON file or AETTA log dir)")
    ap.add_argument("--stream", required=True, help="locked per_condition_cifar10c_*_seed0.json")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--provenance-audit",
        help="OFFICIAL_BASELINE_AUDIT.json; required when --require-official-label is set",
    )
    ap.add_argument("--require-official-label", action="store_true")
    args = ap.parse_args()

    conditions = load_stream_conditions(args.stream)
    decisions = parse_poem(args.logs) if args.method == "poem" else parse_aetta(args.logs)

    missing = [c for c in conditions if c not in decisions]
    if missing:
        sys.exit(
            f"[convert] REFUSING to write: {len(missing)}/{len(conditions)} locked conditions "
            f"missing from official logs (first 5: {missing[:5]}). No fabrication."
        )
    bad = {c: d for c, d in decisions.items() if d not in ("adapt", "freeze", "abstain")}
    if bad:
        sys.exit(f"[convert] invalid decision values: {list(bad.items())[:5]}")

    extra = sorted(set(decisions) - set(conditions))
    if extra:
        sys.exit(
            f"[convert] REFUSING to write: {len(extra)} conditions are not in the locked stream "
            f"(first 5: {extra[:5]})."
        )

    official_allowed = False
    audit_sha = None
    if args.provenance_audit:
        with open(args.provenance_audit) as handle:
            audit = json.load(handle)
        audit_sha = sha256_file(args.provenance_audit)
        official_allowed = bool(
            audit.get("methods", {}).get(args.method, {}).get("official_label_allowed", False)
        )
    if args.require_official_label and not official_allowed:
        sys.exit(
            f"[convert] {args.method} provenance gate is not promotable; retain the "
            "protocol-matched-port label"
        )

    ordered = {c: decisions[c] for c in conditions}
    output = {
        "schema_version": 2,
        "method": args.method,
        "label": (
            "official_implementation_under_protocol_adapter"
            if official_allowed
            else "external_protocol_adapter_unverified"
        ),
        "conversion_rule": (
            "protector fired => freeze; otherwise adapt"
            if args.method == "poem"
            else "estimated adapted accuracy > estimated frozen accuracy => adapt; otherwise freeze"
        ),
        "source_log_sha256": sha256_file(args.logs),
        "locked_stream_sha256": sha256_file(args.stream),
        "provenance_audit_sha256": audit_sha,
        "official_label_allowed": official_allowed,
        "converted_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": ordered,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    n_adapt = sum(1 for d in ordered.values() if d == "adapt")
    print(f"[convert] wrote {args.out}: {len(ordered)} conditions, adapt-rate {n_adapt/len(ordered):.3f}")


if __name__ == "__main__":
    main()
