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


def test_local_eyecandies_data_is_hash_only_archive():
    """If local Eyecandies data exists, it must be the hash-only archive pass
    (Phase 2.2D): tar archives in _archives/ with a SHA256 file. NO extracted
    sample images, NO computed metrics, NO trained model output."""
    p = ROOT / "data" / "raw" / "eyecandies"
    if not p.exists():
        return  # acceptable pre-Phase-2.2D state
    archives_dir = p / "_archives"
    sha_file = ROOT / "experiments" / "phase2" / "family_d" / "eyecandies_archive_sha256.txt"
    assert archives_dir.exists(), (
        "data/raw/eyecandies/ exists but no _archives/ subdir — unexpected state"
    )
    assert sha_file.exists(), (
        "_archives/ exists but eyecandies_archive_sha256.txt missing — "
        "hash-only pass must accompany any download"
    )
    # Forbidden: any extracted sample directory or anomaly-detection output.
    for item in p.iterdir():
        if item.name == "_archives" or item.name.startswith("."):
            continue
        # Allow nothing else for now (no extracted Eyecandies trees).
        assert False, (
            f"Unexpected entry in data/raw/eyecandies/: {item.name}. "
            "Phase 2.2D is a hash-only pass; no extraction or model output."
        )
