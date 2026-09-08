"""Shared strict I/O and current-byte bindings for official decision artifacts.

This is an integration layer, not an alternative provenance validator or trust
root. Scientific protocol compatibility and real execution attestation remain
the responsibility of the existing strict audit contract.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from official_baseline_provenance import (
    OFFICIAL_LABEL, load_json_strict, sha256_file, validate_promotable_audit,
)

SCHEMA_VERSION = 3
UNVERIFIED_LABEL = 'external_protocol_adapter_unverified'
STAGED = 'STAGED_UNVERIFIED'
VALIDATED = 'VALIDATED_OFFICIAL'


def stream_conditions(path):
    raw = load_json_strict(path)
    if not isinstance(raw, dict) or not isinstance(raw.get('records'), list):
        raise ValueError('locked stream must contain records')
    conditions = [r.get('condition') if isinstance(r, dict) else None for r in raw['records']]
    if (not conditions or any(not isinstance(c, str) or not c for c in conditions)
            or len(set(conditions)) != len(conditions)):
        raise ValueError('locked stream requires nonempty unique string conditions')
    return conditions


def validate_decisions(decisions, conditions=None):
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError('decisions must be a nonempty object')
    if any(not isinstance(c, str) or not c for c in decisions):
        raise ValueError('decision conditions must be nonempty strings')
    if any(type(a) is not str or a not in ('adapt', 'freeze', 'abstain') for a in decisions.values()):
        raise ValueError('invalid decision action')
    if conditions is not None and set(decisions) != set(conditions):
        missing = sorted(set(conditions) - set(decisions))
        extra = sorted(set(decisions) - set(conditions))
        raise ValueError(f'decisions/locked stream mismatch: missing={missing[:5]}, extra={extra[:5]}')
    return decisions


def convert_native_decisions(method, path):
    """Recompute the fixed conversion rule from bound native JSON, never a verdict."""
    raw = load_json_strict(path)
    if not isinstance(raw, dict) or method not in ('aetta', 'poem'):
        raise ValueError('native decisions require a supported method and JSON object')
    decisions = {}
    for condition, value in raw.items():
        if isinstance(value, str):
            decisions[condition] = value
        elif method == 'poem' and isinstance(value, dict) and type(value.get('fired')) is bool:
            decisions[condition] = 'freeze' if value['fired'] else 'adapt'
        elif method == 'aetta' and isinstance(value, dict):
            try:
                adapted, frozen = value['est_acc_adapted'], value['est_acc_frozen']
            except KeyError as exc:
                raise ValueError('AETTA native record requires both accuracy estimates') from exc
            if any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in (adapted, frozen)):
                raise ValueError('AETTA accuracy estimates must be finite numeric scalars')
            decisions[condition] = 'adapt' if adapted > frozen else 'freeze'
        else:
            raise ValueError(f'invalid {method} native record for {condition!r}')
    return validate_decisions(decisions)


def current_bindings(*, source_log, locked_stream, environment_receipt, toolchain_receipt):
    paths = dict(source_log_sha256=source_log, locked_stream_sha256=locked_stream,
                 environment_receipt_sha256=environment_receipt, toolchain_receipt_sha256=toolchain_receipt)
    if any(p is None for p in paths.values()):
        raise ValueError('official provenance requires current log, stream, environment and toolchain files')
    # Runtime receipts are JSON contracts, not arbitrary files with an attractive hash.
    for path in (environment_receipt, toolchain_receipt):
        if not isinstance(load_json_strict(path), dict):
            raise ValueError('official provenance runtime receipt must be a JSON object')
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_official_decisions(*, audit, method, decisions, source_log, locked_stream,
                                environment_receipt, toolchain_receipt):
    if audit is None:
        raise ValueError('official provenance audit is required')
    bindings = current_bindings(source_log=source_log, locked_stream=locked_stream,
                                environment_receipt=environment_receipt, toolchain_receipt=toolchain_receipt)
    validate_decisions(decisions, stream_conditions(locked_stream))
    audit_hash = validate_promotable_audit(audit, method=method, decisions=decisions, **bindings)
    return dict(bindings, provenance_audit_sha256=audit_hash)


def atomic_json(path, value):
    """Publish only after complete serialization; validation happens before this call."""
    target = Path(path)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=target.parent,
                                         prefix='.' + target.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
