from __future__ import annotations

from pathlib import Path

from docs.research.kbound.kbound_repro import release_checks


def test_promoted_discovery_excludes_archived_and_superseded_drafts() -> None:
    root = Path(__file__).resolve().parents[1]
    discovered = {Path(path).resolve() for path in release_checks._discover_promoted_tex(root)}
    assert (root / "docs/research/kbound/kbound_submission.tex").resolve() in discovered
    assert (root / "docs/research/kbound/kbound_short_body.tex").resolve() not in discovered
    assert (
        root / "docs/research/kbound/archive/paper_drafts_2026-07-15/kbound_short_2col.tex"
    ).resolve() not in discovered
