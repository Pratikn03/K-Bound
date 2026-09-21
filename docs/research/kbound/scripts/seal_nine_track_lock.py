#!/usr/bin/env python3
"""Read-only historical verifier; never creates or promotes a nine-track seal.

Use --verify to authenticate archived receipts against their pinned Git source
and check the historical file inventory. A PASS concerns historical byte
identity only, not current-policy validity or publication eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE_COMMIT = "660d893caede49c3b7daa8c18e43bb6cbbce5480"
ARCHIVE = Path("docs/research/kbound/archive/superseded_empirical_authorities_2026-09-02")
ARCHIVE_MANIFEST = ROOT / ARCHIVE / "MANIFEST.json"
RETIRED_MANIFEST = ROOT / ARCHIVE / "RETIRED_SURFACES_MANIFEST.json"
SEAL_REL = "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json"
SHA_REL = "experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.sha256"
YAML_REL = "research_lock/NINE_TRACK_LOCK_SEAL_v1.yaml"
SEAL_JSON = ROOT / ARCHIVE / "retired_tree" / SEAL_REL
SEAL_SHA = ROOT / ARCHIVE / "retired_tree" / SHA_REL
LOCK_YAML = ROOT / ARCHIVE / "retired_tree" / YAML_REL

# These inventories' 64 records were compared to SOURCE_COMMIT before repair.
# Pinning also rejects removed entries and tandem archive/receipt replacement.
MANIFEST_SHA = "4d561ed33399643413cf7666e64a9a6f936b8933229761ee1f4380d1bffef4be"
RETIRED_SHA = "e07d8444d1f7b0689c0fdc6e694586a6afcb0f6ae038623ff414f49771e63e8f"
MAX_FILE_BYTES = 16 * 1024 * 1024


def _read_regular(path: Path) -> bytes:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink in historical evidence path: {parent}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError(f"not a bounded regular historical evidence file: {path}")
    with path.open('rb') as handle:
        payload = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(payload) != before.st_size:
        raise ValueError(f"historical evidence changed during read: {path}")
    return payload


def _baseline(relative: str) -> bytes:
    result = subprocess.run(
        ['git', '--no-replace-objects', 'show', f'{SOURCE_COMMIT}:{relative}'],
        cwd=ROOT, capture_output=True, timeout=15,
        env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'},
    )
    if result.returncode:
        raise ValueError(f"pinned historical baseline unavailable: {relative}")
    return result.stdout


def _match(payload: bytes, metadata: dict, label: str) -> None:
    if (type(metadata.get('bytes')) is not int or metadata['bytes'] != len(payload)
            or metadata.get('sha256') != hashlib.sha256(payload).hexdigest()):
        raise ValueError(f"historical hash/size mismatch: {label}")


def verify_historical_seal() -> tuple[dict, list[str]]:
    """Verify all identities, returning errors without writes or fallback seals."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from docs.research.kbound.scripts.release_privacy import (
        PrivacyError,
        strict_json_loads,
        validate_public_member_path,
    )

    seal = {}
    try:
        routes = {}
        for manifest_path, branch, expected_hash in (
            (ARCHIVE_MANIFEST, 'tree', MANIFEST_SHA),
            (RETIRED_MANIFEST, 'retired_tree', RETIRED_SHA),
        ):
            raw = _read_regular(manifest_path)
            manifest = strict_json_loads(raw)
            if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
                raise ValueError('invalid historical manifest schema')
            if manifest.get('source_commit') != SOURCE_COMMIT:
                raise ValueError('historical manifest baseline mismatch')
            records = manifest.get('records')
            if not isinstance(records, list) or not records:
                raise ValueError('missing historical manifest records')
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError('invalid historical manifest record')
                relative = validate_public_member_path(record['original_path'])
                try:
                    archive_path = validate_public_member_path(record['archive_path'])
                except PrivacyError as exc:
                    raise ValueError(f'noncanonical archive_path: {exc}') from exc
                expected_path = (ARCHIVE / branch / relative).as_posix()
                if archive_path != expected_path or relative in routes:
                    raise ValueError('noncanonical or duplicated archive_path')
                routes[relative] = record
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f'historical manifest differs from pinned baseline: {manifest_path.name}')

        # Never trust a checksum and its replaceable sidecar as sole authority.
        for relative, record in routes.items():
            payload = _read_regular(ROOT / record['archive_path'])
            _match(payload, record, f'retired receipt {relative}')
            if payload != _baseline(relative):
                raise ValueError(f'retired receipt differs from pinned baseline: {relative}')

        bound = {}
        for relative, path, label in (
            (SEAL_REL, SEAL_JSON, 'retired seal'),
            (SHA_REL, SEAL_SHA, 'checksum sidecar'),
            (YAML_REL, LOCK_YAML, 'YAML receipt'),
        ):
            payload = _read_regular(path)
            _match(payload, routes[relative], f'{label} baseline')
            bound[relative] = payload
        digest = hashlib.sha256(bound[SEAL_REL]).hexdigest()
        if bound[SHA_REL] != f'{digest}  LOCK_SEAL.json\n'.encode('ascii'):
            raise ValueError('malformed historical checksum sidecar')
        seal = strict_json_loads(bound[SEAL_REL])
        if not isinstance(seal, dict) or seal.get('seal_id') != 'NINE_TRACK_LOCK_SEAL_v1':
            raise ValueError('invalid historical seal schema')
        for name, track in seal['tracks'].items():
            for relative, metadata in track['files'].items():
                validate_public_member_path(relative)
                routed = routes[relative]['archive_path'] if relative in routes else relative
                _match(_read_regular(ROOT / routed), metadata, f'{name}: {relative}')
        return seal, []
    except (OSError, ValueError, TypeError, KeyError, PrivacyError, subprocess.SubprocessError) as exc:
        return seal, [f'historical verification failed: {exc}']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        parser.error('read-only historical verifier: --verify is required; seal creation is retired')
    seal, errors = verify_historical_seal()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"PASS: historical byte verification only, {len(seal['tracks'])} tracks; "
          'not current-policy authority or release promotion')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
