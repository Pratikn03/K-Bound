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
  aetta: exported JSON {condition: {"est_acc_adapted": number, "est_acc_frozen": number}}.
         Both estimates must be finite numeric scalars in the same units; native
         percentage and fraction inputs preserve the same comparison rule. Direct
         directory parsing is unsupported and never silently approximated.
Output: schema-3 wrapper containing exact-stream decisions and current-byte bindings.
--stage emits explicitly unverified decisions; official promotion requires the
separate strict audit plus current environment/toolchain receipts. Historical
schema-2 audits cannot authorize official labels.
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from official_decision_artifact import (
    SCHEMA_VERSION, OFFICIAL_LABEL, UNVERIFIED_LABEL, STAGED, VALIDATED,
    atomic_json, convert_native_decisions, stream_conditions, validate_official_decisions,
)


def load_stream_conditions(stream_path: str) -> list[str]:
    return stream_conditions(stream_path)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_poem(path: str) -> dict:
    return convert_native_decisions('poem', path)


def parse_aetta(path: str) -> dict:
    if os.path.isdir(path):
        sys.exit(
            "[convert] AETTA eval_results directory parsing depends on the vendored repo's "
            "log layout for your run config. Export a JSON "
            '{condition: {"est_acc_adapted": x, "est_acc_frozen": y}} from those logs '
            "(see AETTA/print_est.py) and pass it to --logs instead."
        )
    return convert_native_decisions('aetta', path)


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
    ap.add_argument("--environment-receipt")
    ap.add_argument("--toolchain-receipt")
    ap.add_argument("--stage", action="store_true", help="write explicitly unverified staged decisions; no promotion")
    args = ap.parse_args()
    if args.stage and (args.provenance_audit or args.require_official_label):
        ap.error('--stage cannot be combined with provenance/promotion arguments')

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
    bindings = {}
    if args.provenance_audit:
        try:
            bindings = validate_official_decisions(
                audit=args.provenance_audit, method=args.method, decisions=decisions,
                source_log=args.logs, locked_stream=args.stream,
                environment_receipt=args.environment_receipt, toolchain_receipt=args.toolchain_receipt,
            )
        except (OSError, ValueError) as exc:
            sys.exit(f'[convert] official provenance rejected: {exc}')
        audit_sha = bindings['provenance_audit_sha256']
        official_allowed = True
    if args.require_official_label and not official_allowed:
        sys.exit(
            f"[convert] {args.method} provenance gate is not promotable; retain the "
            "protocol-matched-port label"
        )

    ordered = {c: decisions[c] for c in conditions}
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": VALIDATED if official_allowed else STAGED,
        "method": args.method,
        "label": (
            OFFICIAL_LABEL
            if official_allowed
            else UNVERIFIED_LABEL
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
        **bindings,
    }
    atomic_json(args.out, output)
    n_adapt = sum(1 for d in ordered.values() if d == "adapt")
    print(f"[convert] wrote {args.out}: {len(ordered)} conditions, adapt-rate {n_adapt/len(ordered):.3f}")


if __name__ == "__main__":
    main()
