"""Phase 2.2C — if FAMILY_D_PARTITION_MANIFEST_v2.json exists, it must have no placeholders.

In Phase 2.2C the manifest is correctly NOT created (archive SHA256 deferred to a future
hash-only download pass). The test guards against silent placeholder-laden manifest
creation in any future task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "docs" / "research" / "phase2" / "FAMILY_D_PARTITION_MANIFEST_v2.json"

FORBIDDEN_TOKENS = ("TBD", "TO_BE_FILLED", "TO_BE_RECORDED", "placeholder",
                    "unknown hash", "planned later", "DEFERRED")


def test_manifest_absent_or_no_placeholders():
    if not MAN.exists():
        # Correct state in Phase 2.2C — manifest is intentionally withheld
        return
    t = MAN.read_text()
    for tok in FORBIDDEN_TOKENS:
        assert tok not in t, (
            f"FAMILY_D_PARTITION_MANIFEST_v2.json contains forbidden placeholder {tok!r}"
        )
    # Also verify required structural fields exist
    j = json.loads(t)
    assert j.get("test_evaluation_executed") is False


def test_blocked_report_documents_partition_manifest_blocker():
    p = ROOT / "docs" / "research" / "phase2" / "PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md"
    t = p.read_text()
    assert "FAMILY_D_PARTITION_MANIFEST_v2.json" in t
    assert "archive_sha256" in t or "archive SHA256" in t
