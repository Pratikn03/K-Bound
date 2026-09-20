"""Retirement must not replace divergent local history with older archive bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("docs/research/kbound/archive/preserved_worktree_2026-09-20")
EXPECTED = {
    "docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md": (11763, "921e1ad3c29eba4c1fb1d94b50e59a47fc9bf36410c97a45fe2d5236177619f4"),
    "docs/research/kbound/THEORY_100_PERCENT_CLOSURE_PLAN.md": (3238, "082c736fe521b9a8f0cefe86180984466157644db4fbb760715448b67aab06f5"),
    "docs/research/kbound/kbound_short.docx": (434768, "3c6df6190f9ed7cb860916367046a2f4739c37227037dec949ea095f147a487d"),
    "docs/research/kbound/manuscript/README.md": (1497, "ade12a91b7bb881189d8ef76e21ef949e6321eba18d0c812007af99db9abbbb3"),
    "docs/research/kbound/scripts/kbound_tour.py": (11785, "545c120007bddc4bcdb7d079d520b24e0ba757b205da353eae1165d571edaed0"),
    "docs/research/kbound/scripts/reproduce_headlines.py": (4758, "b2ed58dbe0027effd5b8e7269fcf0a157895e433a402d8ef8257066a59af4034"),
    "docs/research/kbound/scripts/reproduce_submission.sh": (13141, "32510962ceb6968d6444902c8789ed1b78005f10486ff418c40c56216499f1c1"),
}
HISTORICAL_MANIFESTS = {
    "docs/research/kbound/archive/legacy_publication_surfaces_2026-09-02/MANIFEST.json": "dab69ca51372c34da32ab5a85e4b007d0d171664a9e595d17c752c2770117664",
    "docs/research/kbound/archive/stale_publication_builds_2026-09-02/MANIFEST.json": "1373f2884aaf05ed6d17bf2e594b9c163306661c376953d1e3c99688e7870cb3",
}


def test_retirement_preserves_both_local_and_older_archived_payloads() -> None:
    manifest_path = ROOT / ARCHIVE / "MANIFEST.json"
    assert manifest_path.is_file(), "retired local originals need their own preservation manifest"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema"] == "kbound_preserved_worktree_archive_v1"
    assert manifest["release_authority"] is False
    assert manifest["record_count"] == 7
    assert manifest["total_bytes"] == 480950
    rows = manifest["records"]
    assert [r["original_path"] for r in rows] == sorted(EXPECTED)
    archived_paths = set()
    for row in rows:
        original = row["original_path"]
        assert not (ROOT / original).exists(), original
        expected_path = ARCHIVE / "tree" / original
        assert row["archive_path"] == expected_path.as_posix()
        archived = ROOT / expected_path
        assert not any(p.is_symlink() for p in [archived, *archived.parents])
        data = archived.read_bytes()
        identity = len(data), hashlib.sha256(data).hexdigest()
        assert identity == EXPECTED[original]
        assert identity == (row["bytes"], row["sha256"])
        older = (ROOT / row["historical_archive_path"]).read_bytes()
        assert hashlib.sha256(older).hexdigest() == row["historical_archive_sha256"]
        assert row["differs_from_older_archive"] is (data != older)
        archived_paths.add(expected_path.as_posix())
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / ARCHIVE / "tree").rglob("*") if p.is_file()} == archived_paths
    for relative, expected in HISTORICAL_MANIFESTS.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
