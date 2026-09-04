#!/usr/bin/env python3
"""Fail-closed bindings for promoting external baseline decisions as official."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = 3
EXPECTED_CONDITION_COUNT = 432
OFFICIAL_LABEL = "official_implementation_under_protocol_adapter"
NATIVE_EXECUTION_ATTESTATION_SCHEMA = "kbound-official-native-attestation-v1"
NATIVE_EXECUTION_WITNESS_KEY_ID = "kbound-official-native-witness-2026-09-02"
# This public key identifies the independent release witness.  The private key
# is deliberately not present in this repository or in the native runner.
NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64 = "9ZCUlrdMPsm7ZJ3YC8b2IbqAMxuHR1QIXC376YwyknM="
NATIVE_EXECUTION_ATTESTATION_FIELDS = frozenset(
    {"method", "runner_receipt_sha256", "schema", "signature_b64", "witness_key_id"}
)
NATIVE_TRACE_CONTROL_FILENAMES = frozenset(
    {
        "native_completion.json",
        "native_invocation.json",
        "native_runner_receipt.json",
        "native_execution_attestation.json",
    }
)
_NATIVE_COMPLETION_STATUSES = {
    "kbound-official-native-completion-v2": "runner-zero-exit-verified",
    "kbound-official-native-completion-v3": "runner-zero-exit-recorded",
}
_NATIVE_RUNNER_RECEIPT_STATUSES = {
    "kbound-official-native-runner-receipt-v1": "zero-exit-verified",
    "kbound-official-native-runner-receipt-v2": "zero-exit-recorded",
}
REQUIRED_CHECKS = {
    "aetta": frozenset(
        {
            "source_present",
            "upstream_commit_recorded",
            "license_present",
            "environment_lock_present",
            "native_logs_successful",
            "native_execution_attested",
            "converted_decisions_complete",
            "locked_stream_bound",
            "environment_receipt_bound",
            "toolchain_receipt_bound",
        }
    ),
    "poem": frozenset(
        {
            "source_present",
            "upstream_remote_recorded",
            "commit_recorded",
            "source_clean",
            "root_license_present",
            "environment_lock_present",
            "native_logs_successful",
            "native_execution_attested",
            "converted_decisions_complete",
            "locked_stream_bound",
            "environment_receipt_bound",
            "toolchain_receipt_bound",
        }
    ),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n"
    ).hexdigest()


def is_native_trace_control_file(path: str | Path) -> bool:
    """Whether a native-trace file is metadata rather than bound output."""

    return Path(path).name in NATIVE_TRACE_CONTROL_FILENAMES


def verify_native_execution_attestation(
    attestation: object,
    *,
    method: str,
    runner_receipt_sha256: str,
) -> bool:
    """Verify an independent witness signature over one runner receipt.

    The local runner's hashes only establish an internally consistent trace.
    Promotion requires this separate Ed25519 signature from the pinned witness
    key, whose private component is intentionally unavailable to the runner.
    """

    if (
        not isinstance(attestation, dict)
        or set(attestation) != NATIVE_EXECUTION_ATTESTATION_FIELDS
        or attestation.get("schema") != NATIVE_EXECUTION_ATTESTATION_SCHEMA
        or attestation.get("witness_key_id") != NATIVE_EXECUTION_WITNESS_KEY_ID
        or attestation.get("method") != method
        or attestation.get("runner_receipt_sha256") != runner_receipt_sha256
        or not isinstance(attestation.get("signature_b64"), str)
    ):
        return False
    try:
        signature = base64.b64decode(attestation["signature_b64"], validate=True)
        if base64.b64encode(signature).decode("ascii") != attestation["signature_b64"]:
            return False
        public_key = base64.b64decode(NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64, validate=True)
        if len(signature) != 64 or len(public_key) != 32:
            return False
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except (ImportError, ValueError, binascii.Error):
        return False
    signed_claim = {key: value for key, value in attestation.items() if key != "signature_b64"}
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            json.dumps(signed_claim, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n",
        )
    except (InvalidSignature, ValueError):
        return False
    return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path: str | Path) -> object:
    source = Path(path)

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number in {source}: {value}")

    try:
        with source.open(encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read provenance JSON {source}: {exc}") from exc


def decisions_sha256(decisions: Mapping[str, object]) -> str:
    rendered = json.dumps(decisions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"official provenance {label} must be a SHA-256 digest")
    return value


def _schema_status_matches(record: Mapping[str, object], expected: Mapping[str, str]) -> bool:
    schema = record.get("schema")
    return isinstance(schema, str) and expected.get(schema) == record.get("status")


def validate_promotable_audit(
    audit_path: str | Path,
    *,
    method: str,
    decisions: Mapping[str, object],
    source_log_sha256: str,
    locked_stream_sha256: str,
    environment_receipt_sha256: str,
    toolchain_receipt_sha256: str,
) -> str:
    """Validate a separate audit against all bytes used by one conversion."""

    if method not in REQUIRED_CHECKS:
        raise ValueError(f"unsupported official baseline method: {method}")
    audit = load_json_strict(audit_path)
    if not isinstance(audit, dict) or audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("official provenance audit must use schema_version 3")
    methods = audit.get("methods")
    if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
        raise ValueError(f"official provenance audit has no method record for {method}")
    method_record = methods[method]
    if method_record.get("method") != method:
        raise ValueError("official provenance audit method identity mismatch")
    if method_record.get("official_label_allowed") is not True:
        raise ValueError("official provenance audit is not promotable")
    checks = method_record.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("official provenance audit checks are missing")
    missing_checks = sorted(REQUIRED_CHECKS[method] - set(checks))
    if missing_checks:
        raise ValueError("official provenance audit omits checks: " + ", ".join(missing_checks))
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    if failed_checks:
        raise ValueError("official provenance audit has failed checks: " + ", ".join(failed_checks))

    if len(decisions) != EXPECTED_CONDITION_COUNT:
        raise ValueError(f"official provenance requires {EXPECTED_CONDITION_COUNT} decisions; got {len(decisions)}")
    expected_decisions_sha = decisions_sha256(decisions)
    if method_record.get("decision_count") != EXPECTED_CONDITION_COUNT:
        raise ValueError("official provenance decision count mismatch")
    if method_record.get("decision_payload_sha256") != expected_decisions_sha:
        raise ValueError("official provenance decision payload hash mismatch")

    binding = audit.get("promotion_binding")
    if not isinstance(binding, dict):
        raise ValueError("official provenance promotion binding is missing")
    expected_bindings = {
        "locked_stream_sha256": locked_stream_sha256,
        "environment_receipt_sha256": environment_receipt_sha256,
        "toolchain_receipt_sha256": toolchain_receipt_sha256,
        "condition_count": EXPECTED_CONDITION_COUNT,
    }
    for key, expected in expected_bindings.items():
        if binding.get(key) != expected:
            raise ValueError(f"official provenance {key} binding mismatch")
    for key in (
        "locked_stream_sha256",
        "environment_receipt_sha256",
        "toolchain_receipt_sha256",
    ):
        _require_sha(binding[key], key)

    logs = method_record.get("native_logs")
    if not isinstance(logs, dict) or logs.get("successful") is not True:
        raise ValueError("official provenance native logs are not successful")
    if logs.get("unavailable") not in ([], None) or logs.get("failure_markers") not in ([], None):
        raise ValueError("official provenance native logs are incomplete or failed")
    log_hashes = logs.get("sha256")
    if not isinstance(log_hashes, dict) or source_log_sha256 not in log_hashes.values():
        raise ValueError("official provenance does not bind the converted source log")
    _require_sha(source_log_sha256, "source log")
    completion = logs.get("completion")
    runner_receipt = logs.get("runner_receipt")
    execution_attestation = logs.get("execution_attestation")
    expected_receipt_fields = {
        "artifacts_sha256",
        "invocation_sha256",
        "method",
        "producer",
        "returncode",
        "schema",
        "status",
    }
    if (
        logs.get("completion_verified") is not True
        or logs.get("runner_receipt_verified") is not True
        or logs.get("execution_attested") is not True
        or not isinstance(completion, dict)
        or not isinstance(runner_receipt, dict)
        or set(completion) != {"exit_status", "log_sha256", "method", "runner_receipt_sha256", "schema", "status"}
        or not _schema_status_matches(completion, _NATIVE_COMPLETION_STATUSES)
        or completion.get("method") != method
        or completion.get("exit_status") != 0
        or isinstance(completion.get("exit_status"), bool)
        or not isinstance(completion.get("log_sha256"), dict)
        or completion["log_sha256"]
        != {key: value for key, value in log_hashes.items() if not is_native_trace_control_file(key)}
        or set(runner_receipt) != expected_receipt_fields
        or not _schema_status_matches(runner_receipt, _NATIVE_RUNNER_RECEIPT_STATUSES)
        or runner_receipt.get("method") != method
        or runner_receipt.get("returncode") != 0
        or isinstance(runner_receipt.get("returncode"), bool)
        or runner_receipt.get("artifacts_sha256") != completion["log_sha256"]
        or not isinstance(runner_receipt.get("producer"), dict)
        or set(runner_receipt["producer"]) != {"path", "sha256"}
        or runner_receipt["producer"].get("path") != "docs/research/kbound/scripts/run_official_native.py"
        or not isinstance(runner_receipt["producer"].get("sha256"), str)
        or _SHA256_RE.fullmatch(runner_receipt["producer"]["sha256"]) is None
        or not isinstance(runner_receipt.get("invocation_sha256"), str)
        or _SHA256_RE.fullmatch(runner_receipt["invocation_sha256"]) is None
        or completion.get("runner_receipt_sha256") != _canonical_json_sha256(runner_receipt)
        or not verify_native_execution_attestation(
            execution_attestation,
            method=method,
            runner_receipt_sha256=_canonical_json_sha256(runner_receipt),
        )
    ):
        raise ValueError("official provenance native execution attestation is invalid")

    if method == "aetta":
        _require_sha(method_record.get("source_tree_sha256"), "AETTA source tree")
        upstream = method_record.get("upstream_commit")
        if not isinstance(upstream, str) or re.fullmatch(r"[0-9a-f]{40}", upstream) is None:
            raise ValueError("official provenance AETTA upstream commit is missing")
    else:
        upstream = method_record.get("upstream_commit")
        if not isinstance(upstream, str) or re.fullmatch(r"[0-9a-f]{40}", upstream) is None:
            raise ValueError("official provenance POEM upstream commit is missing")
        if not isinstance(method_record.get("upstream_remote"), str):
            raise ValueError("official provenance POEM upstream remote is missing")

    return sha256_file(audit_path)
