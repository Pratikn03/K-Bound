"""Phase 2.2C — Eyecandies must remain untouched at outcome level."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_eyecandies_in_experiment_outcome_artifacts():
    """Eyecandies must not appear in any prediction archive index or statistics CSV."""
    forbidden_roots = (
        ROOT / "experiments" / "phase2" / "statistics",
        ROOT / "experiments" / "phase2" / "predictions",
        ROOT / "experiments" / "phase2" / "mechanism",
        ROOT / "experiments" / "phase2" / "certification",
    )
    for r in forbidden_roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("._"):
                continue
            try:
                t = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, Exception):
                continue
            assert "eyecandies" not in t.lower(), (
                f"Eyecandies reference found in outcome artifact {p}"
            )


def test_no_local_eyecandies_data():
    """A local data/raw/eyecandies directory must not exist before the hash-only pass."""
    p = ROOT / "data" / "raw" / "eyecandies"
    assert not p.exists(), (
        f"{p} exists — Eyecandies dataset appears to have been downloaded; "
        f"this test must be revised in the Phase 2.2D hash-only pass"
    )
