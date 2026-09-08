"""Contracts for the shared K-Bound release identity include."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def _revision_authority() -> dict:
    historical = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    documents = historical["documents"]
    documents["main_with_appendix"] = {
        "role": "Main paper with integrated appendices",
        "anonymous": False,
        "output_filename": "kbound_main_with_appendix.pdf",
    }
    return {
        "schema": "kbound_manuscript_revision_v1",
        "revision_id": "KBOUND-REVISION-2026-09-08",
        "revision_date": "September 8, 2026",
        "base_commit": "0123456789ab",
        "canonical_panel_sha256": historical["canonical_panel_sha256"],
        "historical_evidence_manifest_sha256": historical["source_manifest_sha256"],
        "documents": documents,
    }


def _generate_revision(tmp_path: Path, revision: dict) -> tuple[subprocess.CompletedProcess, Path]:
    authority = tmp_path / "revision.json"
    authority.write_text(json.dumps(revision), encoding="utf-8")
    output = tmp_path / "identity.tex"
    output.write_text("previous identity must survive rejection", encoding="ascii")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--authority", str(authority), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result, output


def test_revision_generation_distinguishes_edits_from_historical_evidence(tmp_path: Path) -> None:
    result, output = _generate_revision(tmp_path, _revision_authority())
    assert result.returncode == 0, result.stdout + result.stderr
    text = output.read_text(encoding="ascii")
    assert r"\newcommand{\KBoundReleaseID}{KBOUND-REVISION-2026-09-08}" in text
    assert r"\newcommand{\KBoundRoleMainWithAppendix}{Main paper with integrated appendices}" in text
    named, anonymous = text.split(r"\newcommand{\KBoundAnonymousReleaseBlock}", 1)
    assert "Revision date:" in named
    assert "Historical evidence-manifest SHA-256:" in named
    assert "Base commit (with subsequent manuscript edits):" in named
    assert "Source closure:" not in text
    assert "Source snapshot commit:" not in text
    assert "KBoundSourceSnapshotCommit" not in anonymous
    assert "Historical evidence-manifest SHA-256:" in anonymous
    assert "KBoundRoleTMLR" in anonymous


@pytest.mark.parametrize("mutation", ["missing_role", "anonymous_named_role", "bad_hash", "bad_commit"])
def test_revision_rejects_invalid_authority_without_replacing_output(tmp_path: Path, mutation: str) -> None:
    revision = _revision_authority()
    if mutation == "missing_role":
        del revision["documents"]["main_with_appendix"]
    elif mutation == "anonymous_named_role":
        revision["documents"]["main_with_appendix"]["anonymous"] = True
    elif mutation == "bad_hash":
        revision["historical_evidence_manifest_sha256"] = "not-a-hash"
    else:
        revision["base_commit"] = "not-a-commit"
    result, output = _generate_revision(tmp_path, revision)
    assert result.returncode != 0
    assert output.read_text(encoding="ascii") == "previous identity must survive rejection"
