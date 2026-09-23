#!/usr/bin/env python3
"""Check claim references and byte identities; do not adjudicate scientific truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
REGISTER = "docs/research/kbound/journal_revision/claim_evidence_register.json"
STATUSES = {"supported", "conditional", "descriptive", "unestablished"}


def read_reference(root: Path, value: str) -> bytes:
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value or str(p) != value:
        raise ValueError(f"unsafe reference: {value!r}")
    path = root / value
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"unsafe symlink reference: {value}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
        raise ValueError(f"unsafe file reference: {value}")
    return path.read_bytes()


def validate_register(root: Path, payload: dict) -> list[str]:
    errors = []
    if payload.get("schema") != "kbound-journal-claim-register-v1":
        errors.append("unsupported register schema")
    rows = payload.get("claims", [])
    if not isinstance(rows, list) or not rows:
        return errors + ["claims must be a nonempty list"]
    seen = set()
    for row in rows:
        ident = row.get("id", "")
        if not ident or ident in seen:
            errors.append(f"missing or duplicate claim id: {ident}")
        seen.add(ident)
        if row.get("status") not in STATUSES:
            errors.append(f"{ident}: invalid scientific status")
        for field in ("claim", "assumptions", "metric", "unit", "evidence_type", "justified_wording"):
            if not row.get(field):
                errors.append(f"{ident}: missing {field}")
        for kind in ("locations", "evidence"):
            if not row.get(kind):
                errors.append(f"{ident}: missing {kind}")
            for reference in row.get(kind, []):
                try:
                    raw = read_reference(root, reference["path"])
                    if kind == "locations":
                        anchor = reference.get("anchor")
                        if not anchor or anchor not in raw.decode("utf-8"):
                            errors.append(f"{ident}: anchor missing: {reference['path']}")
                    else:
                        expected = reference.get("sha256", "")
                        if not re.fullmatch(r"[0-9a-f]{64}", expected) or hashlib.sha256(raw).hexdigest() != expected:
                            errors.append(f"{ident}: evidence hash mismatch: {reference['path']}")
                except (OSError, ValueError, KeyError, UnicodeError) as exc:
                    errors.append(f"{ident}: {exc}")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--register", default=REGISTER)
    args = parser.parse_args(argv)
    payload = json.loads(read_reference(args.repo, args.register))
    errors = validate_register(args.repo, payload)
    print(
        json.dumps(
            {
                "status": "FAIL" if errors else "PASS",
                "claims": len(payload.get("claims", [])),
                "scope": "references and byte identities, not scientific authentication",
                "errors": errors,
            },
            indent=2,
        )
    )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
