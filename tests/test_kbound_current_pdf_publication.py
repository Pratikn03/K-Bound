"""Contracts for publishing one unambiguous current K-Bound PDF set."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import render_pdf_pages

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "docs/research/kbound/scripts/publish_current_pdfs.py"
BUILD_SCRIPT = ROOT / "docs/research/kbound/scripts/build_pdfs.sh"
RELEASE_RUNBOOK = ROOT / "docs/research/kbound/runbooks/release_candidate.sh"

SOURCE_TO_CURRENT = {
    "kbound_full_report.pdf": "kbound_full_report.pdf",
    "kbound_short_main.pdf": "kbound_short_main.pdf",
    "kbound_short_supplement.pdf": "kbound_short_supplement.pdf",
    "kbound_tmlr.pdf": "kbound_tmlr.pdf",
}


def _fake_pdf(name: str) -> bytes:
    return b"%PDF-1.7\n" + name.encode("ascii") + b"\n%%EOF\n"


def test_publisher_replaces_only_four_roles_and_preserves_other_artifacts(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "paper"
    release = tmp_path / "release" / "current"
    output = tmp_path / "output" / "pdf"
    paper.mkdir()
    release.mkdir(parents=True)
    output.mkdir(parents=True)

    payloads: dict[str, bytes] = {}
    for source_name, current_name in SOURCE_TO_CURRENT.items():
        payload = _fake_pdf(source_name)
        (paper / source_name).write_bytes(payload)
        payloads[current_name] = payload

    preserved_names = (
        "kbound_full_report_99pp.pdf",
        "kbound_short_main_21pp.pdf",
        "kbound_short_supplement_25pp.pdf",
        "kbound_combined_current.pdf",
        "kbound_combined_44page_current.pdf",
        "kbound_combined_43page_preserved.pdf",
        "kbound_short_final_draft.pdf",
        "kbound_short_final_draft_revised.pdf",
        "kbound_tmlr_45page_current.pdf",
        "kbound_tmlr_revised.pdf",
        "kbound_main_with_appendix.pdf",
    )
    for destination in (release, output):
        for preserved in preserved_names:
            (destination / preserved).write_bytes(_fake_pdf(preserved))
        for current_name in SOURCE_TO_CURRENT.values():
            (destination / current_name).write_bytes(_fake_pdf("previous-role"))
        (destination / "KBOUND_CURRENT_SHA256SUMS.txt").write_text("previous seal\n")
        (destination / "KBOUND_FOUR_ARTIFACT_SHA256SUMS.txt").write_text(
            "obsolete\n", encoding="utf-8"
        )
        (destination / "KBOUND_PUBLICATION_DRAFT_SHA256SUMS.txt").write_text(
            "obsolete\n", encoding="utf-8"
        )
    unrelated_release = release / "README.pdf"
    unrelated_release.write_bytes(_fake_pdf("README.pdf"))
    unrelated_output = output / "another_project.pdf"
    unrelated_output.write_bytes(_fake_pdf("another_project.pdf"))

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLISHER),
            "--paper-dir",
            str(paper),
            "--release-dir",
            str(release),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    expected = [
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}"
        for name in sorted(payloads)
    ]
    for destination in (release, output):
        current_paths = {path.name for path in destination.glob("kbound*.pdf")}
        assert current_paths == set(SOURCE_TO_CURRENT.values()) | set(preserved_names)
        for preserved in preserved_names:
            assert (destination / preserved).read_bytes() == _fake_pdf(preserved)
        for name, payload in payloads.items():
            assert (destination / name).read_bytes() == payload
        checksum = destination / "KBOUND_CURRENT_SHA256SUMS.txt"
        assert checksum.read_text(encoding="utf-8").splitlines() == expected
        assert (destination / "KBOUND_FOUR_ARTIFACT_SHA256SUMS.txt").read_bytes() == b"obsolete\n"
        assert (destination / "KBOUND_PUBLICATION_DRAFT_SHA256SUMS.txt").read_bytes() == b"obsolete\n"

    assert unrelated_release.read_bytes() == _fake_pdf("README.pdf")
    assert unrelated_output.read_bytes() == _fake_pdf("another_project.pdf")
    assert (release / "KBOUND_CURRENT_SHA256SUMS.txt").read_bytes() == (
        output / "KBOUND_CURRENT_SHA256SUMS.txt"
    ).read_bytes()


@pytest.mark.parametrize("link_kind", (
    "source_pdf", "broken_source_pdf", "paper_directory",
    "release_directory", "output_directory", "output_parent",
    "release_pdf", "output_pdf", "release_checksum", "output_checksum",
    "broken_output_pdf", "lexical_parent_before_dotdot",
))
def test_cli_rejects_symlink_paths_without_changing_either_surface(
    tmp_path: Path, link_kind: str,
) -> None:
    paper = tmp_path / "paper"
    release = tmp_path / "release"
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    for directory in (paper, release, output, outside):
        directory.mkdir()
    for source_name in SOURCE_TO_CURRENT:
        (paper / source_name).write_bytes(_fake_pdf(source_name))
    for directory in (release, output, outside):
        for current_name in (*SOURCE_TO_CURRENT.values(), "KBOUND_CURRENT_SHA256SUMS.txt"):
            (directory / current_name).write_bytes(b"preserved existing bytes")

    if link_kind in ("source_pdf", "broken_source_pdf"):
        selected = paper / "kbound_full_report.pdf"
        selected.unlink()
        target = outside / ("missing.pdf" if link_kind.startswith("broken") else "source.pdf")
        if not link_kind.startswith("broken"):
            target.write_bytes(_fake_pdf("external source"))
        selected.symlink_to(target)
    elif link_kind == "paper_directory":
        alias = tmp_path / "paper-link"
        alias.symlink_to(paper, target_is_directory=True)
        paper = alias
    elif link_kind in ("release_directory", "output_directory", "output_parent"):
        alias = tmp_path / "destination-link"
        alias.symlink_to(outside, target_is_directory=True)
        if link_kind == "release_directory":
            release = alias
        else:
            output = alias / "nested" if link_kind == "output_parent" else alias
    elif link_kind == "lexical_parent_before_dotdot":
        alias = tmp_path / "parent-link"
        alias.symlink_to(outside, target_is_directory=True)
        output = alias / ".." / "output"
    else:
        destination = release if link_kind.startswith("release") else output
        name = "KBOUND_CURRENT_SHA256SUMS.txt" if link_kind.endswith("checksum") else "kbound_tmlr.pdf"
        selected = destination / name
        selected.unlink()
        selected.symlink_to(outside / ("missing.pdf" if link_kind.startswith("broken") else name))

    def inventory():
        return {
            str(path.relative_to(tmp_path)): (
                ("symlink", str(path.readlink())) if path.is_symlink()
                else ("file", path.read_bytes()) if path.is_file()
                else ("directory", None)
            )
            for path in tmp_path.rglob("*")
        }
    before = inventory()
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), "--paper-dir", str(paper),
         "--release-dir", str(release), "--output-dir", str(output)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0, "symlinked publication inputs must be rejected"
    assert "symlink paths are not allowed" in result.stderr.lower()
    assert inventory() == before


def test_cli_retains_absolute_publication_paths_for_relative_arguments(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    for name in SOURCE_TO_CURRENT:
        (paper / name).write_bytes(_fake_pdf(name))
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), "--paper-dir", "paper",
         "--release-dir", "release", "--output-dir", "output"],
        cwd=tmp_path, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {
        str(tmp_path / directory / name)
        for directory in ("release", "output")
        for name in (*SOURCE_TO_CURRENT.values(), "KBOUND_CURRENT_SHA256SUMS.txt")
    }
    assert set(result.stdout.splitlines()) == expected


@pytest.mark.parametrize("surface", ("release", "output"))
@pytest.mark.parametrize("target_name", (*SOURCE_TO_CURRENT.values(), "KBOUND_CURRENT_SHA256SUMS.txt"))
def test_owned_target_directory_collision_rejects_before_any_replacement(
    tmp_path: Path, surface: str, target_name: str,
) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    for name in SOURCE_TO_CURRENT:
        (paper / name).write_bytes(_fake_pdf(name))
    for directory_name in ("release", "output"):
        directory = tmp_path / directory_name
        directory.mkdir()
        for name in (*SOURCE_TO_CURRENT.values(), "KBOUND_CURRENT_SHA256SUMS.txt"):
            (directory / name).write_bytes(b"existing role or seal")
        (directory / "kbound_main_with_appendix.pdf").write_bytes(b"preserved other role")
    collision = tmp_path / surface / target_name
    collision.unlink()
    collision.mkdir()
    (collision / "keep.txt").write_bytes(b"preserved directory contents")

    def inventory():
        return {
            str(path.relative_to(tmp_path)): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")
        }
    before = inventory()
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), "--paper-dir", str(paper),
         "--release-dir", str(tmp_path / "release"), "--output-dir", str(tmp_path / "output")],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert inventory() == before
    assert "not a regular file" in result.stderr.lower()


def test_collision_does_not_create_the_other_publication_directory(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    for name in SOURCE_TO_CURRENT:
        (paper / name).write_bytes(_fake_pdf(name))
    output = tmp_path / "output"
    output.mkdir()
    (output / "KBOUND_CURRENT_SHA256SUMS.txt").mkdir()
    release = tmp_path / "not-created"

    def inventory():
        return {
            path.name: ("file", path.read_bytes()) if path.is_file() else ("directory", None)
            for path in output.iterdir()
        }
    before = inventory()
    result = subprocess.run(
        [sys.executable, str(PUBLISHER), "--paper-dir", str(paper),
         "--release-dir", str(release), "--output-dir", str(output)],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not release.exists()
    assert inventory() == before


def test_all_driver_build_publishes_the_stable_current_set() -> None:
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "scripts/publish_current_pdfs.py" in build
    assert 'BUILD_LONG_TMLR" == "1"' in build
    assert 'BUILD_SHORT_MAIN" == "1"' in build
    assert 'BUILD_SHORT_SUPPLEMENT" == "1"' in build
    assert 'BUILD_FULL_REPORT" == "1"' in build
    assert "Publishing one stable current PDF set" in build
    assert '--release-dir "$ROOT/release/current"' in build
    assert '--output-dir "$REPO/output/pdf"' in build


def test_release_pdf_mode_builds_every_maintained_driver_before_publication() -> None:
    runbook = RELEASE_RUNBOOK.read_text(encoding="utf-8")
    pdf_step = runbook.split("step_pdf() {", 1)[1].split("\n}", 1)[0]
    for flag in (
        "BUILD_LONG_TMLR=1",
        "BUILD_SHORT_MAIN=1",
        "BUILD_SHORT_SUPPLEMENT=1",
        "BUILD_FULL_REPORT=1",
    ):
        assert flag in pdf_step


def test_visual_release_check_covers_exactly_four_current_pdf_roles() -> None:
    assert render_pdf_pages.CURRENT_PDF_NAMES == (
        "kbound_short_main.pdf",
        "kbound_short_supplement.pdf",
        "kbound_tmlr.pdf",
        "kbound_full_report.pdf",
    )


def test_publisher_leaves_existing_outputs_untouched_when_source_set_is_incomplete(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "paper"
    release = tmp_path / "release" / "current"
    output = tmp_path / "output" / "pdf"
    paper.mkdir()
    release.mkdir(parents=True)
    output.mkdir(parents=True)

    for source_name in tuple(SOURCE_TO_CURRENT)[:-1]:
        (paper / source_name).write_bytes(_fake_pdf(source_name))
    legacy_release = release / "kbound_preserved_until_replacement.pdf"
    legacy_output = output / "kbound_preserved_until_replacement.pdf"
    legacy_payload = _fake_pdf(legacy_release.name)
    legacy_release.write_bytes(legacy_payload)
    legacy_output.write_bytes(legacy_payload)

    result = subprocess.run(
        [
            sys.executable,
            str(PUBLISHER),
            "--paper-dir",
            str(paper),
            "--release-dir",
            str(release),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert legacy_release.read_bytes() == legacy_payload
    assert legacy_output.read_bytes() == legacy_payload
    for destination in (release, output):
        assert not set(SOURCE_TO_CURRENT.values()).intersection(
            path.name for path in destination.iterdir()
        )
