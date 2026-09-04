"""Contracts for the shared K-Bound release identity include."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "docs/research/kbound/scripts/generate_release_identity.py"
AUTHORITY = ROOT / "docs/research/kbound/paper/release/current_release.json"


def test_generator_emits_named_and_anonymous_blocks_without_identity_leak(tmp_path: Path) -> None:
    output = tmp_path / "identity.tex"
    commit = "0123456789ab"
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--authority",
            str(AUTHORITY),
            "--output",
            str(output),
            "--source-snapshot-commit",
            commit,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = output.read_text(encoding="ascii")
    assert "KBOUND-2026-09-03-R1" in rendered
    assert commit in rendered
    assert "Named compact main paper" in rendered
    assert "Anonymous integrated TMLR review manuscript" in rendered
    anonymous = rendered.split(r"\newcommand{\KBoundAnonymousReleaseBlock}", 1)[1]
    assert "Source snapshot commit" not in anonymous
    for forbidden in ("Pratik", "Niroula", "Mankato", "mnsu.edu", "/Users/"):
        assert forbidden not in anonymous


def test_authority_defines_exactly_four_roles() -> None:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    assert set(authority["documents"]) == {
        "short_main",
        "short_supplement",
        "tmlr",
        "full_report",
    }
    assert sum(document["anonymous"] for document in authority["documents"].values()) == 1
