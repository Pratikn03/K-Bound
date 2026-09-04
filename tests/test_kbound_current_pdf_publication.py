"""Contracts for publishing one unambiguous current K-Bound PDF set."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

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


def test_publisher_replaces_superseded_outputs_with_four_identical_release_roles(
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

    for destination in (release, output):
        for obsolete in (
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
        ):
            (destination / obsolete).write_bytes(_fake_pdf(obsolete))
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
        assert current_paths == set(SOURCE_TO_CURRENT.values())
        for name, payload in payloads.items():
            assert (destination / name).read_bytes() == payload
        checksum = destination / "KBOUND_CURRENT_SHA256SUMS.txt"
        assert checksum.read_text(encoding="utf-8").splitlines() == expected
        assert not (destination / "KBOUND_FOUR_ARTIFACT_SHA256SUMS.txt").exists()
        assert not (destination / "KBOUND_PUBLICATION_DRAFT_SHA256SUMS.txt").exists()

    assert unrelated_release.is_file()
    assert unrelated_output.is_file()
    assert (release / "KBOUND_CURRENT_SHA256SUMS.txt").read_bytes() == (
        output / "KBOUND_CURRENT_SHA256SUMS.txt"
    ).read_bytes()


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
