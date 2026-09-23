"""Rendered contracts for the independently compiled K-Bound paper pair."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "docs/research/kbound"
LATEXMK = shutil.which("latexmk") or "/opt/homebrew/bin/latexmk"
PDFINFO = shutil.which("pdfinfo")
_BUNDLED_PDFTOTEXT = (
    Path(PDFINFO).parents[2] / "native/poppler/poppler/bin/pdftotext" if PDFINFO else Path("/missing")
)
PDFTOTEXT = shutil.which("pdftotext") or str(_BUNDLED_PDFTOTEXT)


def _tool_available(path: str | None) -> bool:
    return bool(path and Path(path).is_file())


pytestmark = pytest.mark.skipif(
    not (_tool_available(LATEXMK) and _tool_available(PDFINFO) and _tool_available(PDFTOTEXT)),
    reason="the rendered publication contract needs latexmk, pdfinfo, and pdftotext",
)


@pytest.fixture(scope="module")
def split_builds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Compile both real standalone drivers into isolated output directories."""
    root = tmp_path_factory.mktemp("kbound-split-builds")
    outputs: dict[str, Path] = {}
    for stem in ("kbound_submission", "kbound_short_main", "kbound_short_supplement"):
        outdir = root / stem
        outdir.mkdir()
        result = subprocess.run(
            [
                LATEXMK,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-outdir={outdir}",
                f"-auxdir={outdir}",
                f"{stem}.tex",
            ],
            cwd=PAPER,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[stem] = outdir
    return outputs


def _page_count(pdf: Path) -> int:
    result = subprocess.run(
        [PDFINFO, str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    assert match, result.stdout
    return int(match.group(1))


def _pdf_text(pdf: Path) -> str:
    result = subprocess.run(
        [PDFTOTEXT, str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return " ".join(result.stdout.replace("’", "'").split())


def _assert_clean_reference_log(log: str) -> None:
    forbidden = (
        "There were undefined references",
        "Label(s) may have changed",
        "multiply defined",
        "Citation `",
        "Reference `",
        "No file ",
    )
    assert not any(marker in log for marker in forbidden), log


def test_main_is_a_20_to_28_page_paper_without_supplement_labels(split_builds: dict[str, Path]) -> None:
    """Catches a supplement leak, missing sections, or a page count outside the submission target."""
    outdir = split_builds["kbound_short_main"]
    assert 20 <= _page_count(outdir / "kbound_short_main.pdf") <= 28

    aux = (outdir / "kbound_short_main.aux").read_text(encoding="utf-8")
    log = (outdir / "kbound_short_main.log").read_text(encoding="utf-8")
    assert r"\newlabel{sec:compact-conclusion}{{10}" in aux
    assert r"\newlabel{app:" not in aux
    _assert_clean_reference_log(log)


def test_supplement_compiles_independently_as_appendices_a_through_q(
    split_builds: dict[str, Path],
) -> None:
    """Catches missing appendices and any dependency on the main paper's auxiliary file."""
    outdir = split_builds["kbound_short_supplement"]
    aux = (outdir / "kbound_short_supplement.aux").read_text(encoding="utf-8")
    log = (outdir / "kbound_short_supplement.log").read_text(encoding="utf-8")

    appendix_labels = (
        "app:population-transfer",
        "app:auxiliary-results",
        "app:compact-ledger",
        "app:compact-contracts",
        "app:compact-actions",
        "app:compact-inference",
        "app:compact-provenance",
        "app:compact-implementation",
        "app:compact-evidence",
        "app:compact-deployment",
        "app:compact-confirmation",
        "app:theorem-dependencies",
        "app:compact-formal",
        "sec:compact-repro",
        "app:nextphase-calibration",
        "app:nextphase-paired-proof",
        "app:comparison-provenance",
    )
    for letter, label in zip("ABCDEFGHIJKLMNOPQ", appendix_labels, strict=True):
        assert rf"\newlabel{{{label}}}{{{{{letter}}}" in aux
    _assert_clean_reference_log(log)


def test_default_and_standalone_reference_prose_is_grammatical(
    split_builds: dict[str, Path],
) -> None:
    """Catches wrong object prefixes and sentence-case fallbacks in rendered prose."""
    combined = _pdf_text(split_builds["kbound_submission"] / "kbound_submission.pdf")
    main = _pdf_text(split_builds["kbound_short_main"] / "kbound_short_main.pdf")
    supplement = _pdf_text(split_builds["kbound_short_supplement"] / "kbound_short_supplement.pdf")

    assert re.search(r"Corruption-family sensitivity breakdowns appear in Table \d+\.", combined)
    assert not re.search(r"family-sensitivity table in Appendix \d+", combined)
    # A two-column float can separate this sentence's halves in pdftotext's
    # reading order. Require the exact active source sentence and both rendered
    # halves, preserving its grammar without requiring extraction adjacency.
    body = re.sub(
        r"(?<!\\)%[^\n]*", "",
        (PAPER / "kbound_submission_body.tex").read_text(encoding="utf-8"),
    )
    assert (
        r"Corruption-family sensitivity breakdowns appear in "
        r"\KBSuppRef{tab:compact-current-family}{Table}{the supplementary family-sensitivity table}."
    ) in " ".join(body.split())
    assert "Corruption-family sensitivity breakdowns" in main
    assert "appear in the supplementary family-sensitivity table." in main
    assert (
        "to satisfy the main paper's coordinatewise sign-flip invariance condition."
        in supplement
    )
