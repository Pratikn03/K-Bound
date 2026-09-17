"""Scientific narrative and manuscript-closure guards; no datasets or training."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal
from docs.research.kbound.scripts import verify_release_checksums as checksums

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/research/kbound"


def live(path: Path) -> str:
    text = re.sub(r"\\iffalse.*?\\fi", "", path.read_text(encoding="utf-8"), flags=re.S)
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def words(path: Path) -> str:
    return " ".join(live(path).split())


def test_abstract_is_concise_and_keeps_empirical_and_population_targets_separate():
    abstract = words(PAPER / "kbound_abstract_core.tex")
    assert 180 <= len(abstract.split()) <= 230
    for required in (
        "empirical companion, not an implementation",
        "observed evaluation-cell benefit",
        "labeled historical outcomes",
        "SAR favors always-adapt",
        "no ADAPT decisions",
        "do not establish interval coverage or population-risk protection",
    ):
        assert required in abstract
    assert "Holm" not in abstract
    assert r"\hashtext" not in abstract


def test_main_story_has_ten_scientific_sections_and_three_contributions():
    body = live(PAPER / "kbound_submission_body.tex")
    sections = re.findall(r"^\\section\{([^}]+)\}", body, re.M)
    assert sections == [
        "Introduction", "Related Work", "Setup, Notation, and Claim Levels",
        "Population Partial-Identification Frontier",
        "What Label-Free Evidence Cannot Learn",
        "KGA: An Empirical Benefit-Interval Gate",
        "Primary Experimental Protocols", "Primary Results",
        "Limitations and Broader Impact", "Conclusion",
    ]
    assert body.count("Our primary research question is:") == 1
    contributions = body.split(r"\subsection{Three Scientific Contributions}", 1)[1]
    contributions = contributions.split(r"\end{enumerate}", 1)[0]
    assert contributions.count(r"\item") == 3
    assert body.index(r"\label{tab:compact-bridge}") < body.index(r"\section{Population Partial-Identification")
    assert body.index(r"\label{eq:cct-safe-utility}") < body.index(r"\section{Primary Results}")


@pytest.mark.parametrize("name", ["kbound_submission.tex", "kbound_tmlr.tex"])
def test_each_driver_places_references_before_supplement(name):
    driver = live(PAPER / name)
    body = driver.index(r"\input{kbound_submission_body}")
    bibliography = driver.index(r"\input{paper/references_kbound_expanded}")
    supplement = driver.index(r"\input{kbound_submission_supplement}")
    assert body < bibliography < supplement
    assert r"\appendix" not in live(PAPER / "kbound_submission_body.tex")
    assert live(PAPER / "kbound_submission_supplement.tex").lstrip().startswith(r"\appendix")


def test_tmlr_remains_anonymous_without_removing_named_companion():
    anonymous = live(PAPER / "kbound_tmlr.tex")
    named = live(PAPER / "kbound_submission.tex")
    assert r"\anontrue" in anonymous
    assert r"\author{Anonymous authors" in anonymous
    assert "pdfauthor={}" in anonymous
    assert r"\anonfalse" in named
    assert "Pratik Niroula" in named


def test_audits_move_to_supplement_without_discarding_adverse_records():
    body = live(PAPER / "kbound_submission_body.tex")
    supplement = live(PAPER / "kbound_submission_supplement.tex")
    for token in (r"\SourceManifestSHA", r"\CCTInferenceSHA", r"\appendix", "literal " + r"\texttt{Infinity}"):
        assert token not in body
    assert supplement.count(r"\SourceManifestSHA") >= 2
    for token in (
        "five-checkpoint", "invalid", "withheld", "PACS", "ImageNet-R",
        "no target natural-shift", "Historical Protocol-Matched POEM and AETTA",
        "142 declarations", "--full-foundations",
    ):
        assert token in supplement
    assert r"\input{paper/generated/kbound_primary_accuracy_table.tex}" in body
    assert r"\input{paper/generated/kbound_auxiliary_accuracy_table.tex}" in supplement
    assert r"\input{paper/generated/kbound_auxiliary_balanced_accuracy_table.tex}" in supplement
    assert "canonical_panel_table}" not in body + supplement


def test_coverage_to_action_is_a_proposition_not_a_new_calibration_theorem():
    source = live(PAPER / "paper/sections/theory_certificate.tex")
    assert r"\begin{proposition}[Coverage-to-action implication]" in source
    assert r"\begin{theorem}" not in source
    assert r"\label{thm:certificate}" in source  # Stable reference identifier, not display type.
    prose = live(PAPER / "kbound_submission_body.tex") + live(PAPER / "kbound_submission_supplement.tex")
    assert r"Theorem~\ref{thm:certificate}" not in prose
    assert r"Thm.~\ref{thm:certificate}" not in prose
    assert "Marginal false-adapt control (theorem)" not in prose


def test_interval_audit_cannot_be_promoted_to_fresh_group_validation():
    body = words(PAPER / "kbound_submission_body.tex")
    supplement = words(PAPER / "kbound_submission_supplement.tex")
    for token in ("rank-constrained", "not independent validation", "one false FREEZE among 359",
                  "only two FREEZE decisions", "conditional false-freeze frequency is undefined"):
        assert token in body
    assert "pooled LOO inclusion remains rank-constrained" in supplement
    assert "No fitting, tuning, new target access, or training is performed" in supplement
    assert "empirical order-statistic calibration" in body
    assert "exact-rank" not in body
    assert "Commitment rate" in body
    assert r"\mathrm{Cov}_{\mathrm{cell}}" in body


def test_fallback_semantics_and_development_labels_are_explicit():
    body = words(PAPER / "kbound_submission_body.tex")
    assert "label-free at deployment, not during development and calibration" in body
    assert "must be logged as ABSTAIN, not as a certified FREEZE" in body
    assert "not those additional interventions" in body
    assert "requires improvement over both fixed policies" in body


def test_new_inputs_and_outputs_are_in_the_release_inventory():
    all_sources = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    expected_sources = {
        "docs/research/kbound/kbound_submission_supplement.tex",
        "docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py",
        "tests/test_kbound_interval_diagnostics.py",
        "tests/test_kbound_metric_display_tables.py",
        "tests/test_kbound_narrative_revision.py",
        "tests/test_kbound_pdf_build_isolation.py",
    }
    expected_outputs = {
        f"docs/research/kbound/paper/generated/{name}"
        for name in (
            "current_policy_interval_diagnostics.json", "current_policy_interval_diagnostics.tex",
            "current_policy_interval_diagnostics_groups.tex", "kbound_primary_accuracy_table.tex",
            "kbound_auxiliary_accuracy_table.tex", "kbound_auxiliary_balanced_accuracy_table.tex",
            "cct20_safe_utility_display.tex",
        )
    }
    assert expected_sources <= all_sources
    assert expected_outputs <= seal.GENERATED_OUTPUT_ALLOWLIST
    assert expected_outputs <= set(checksums.REQUIRED_RELEASE_PATHS)
    assert not expected_outputs & all_sources


def test_build_checks_diagnostics_and_full_generation_refreshes_explicitly():
    build = (PAPER / "scripts/build_pdfs.sh").read_text()
    runbook = (PAPER / "runbooks/release_candidate.sh").read_text()
    assert "scripts/build_current_policy_interval_diagnostics.py --check" in build
    assert '"$KB/scripts/build_current_policy_interval_diagnostics.py" --refresh-existing' in runbook
    for test_name in ("test_kbound_interval_diagnostics", "test_kbound_metric_display_tables", "test_kbound_narrative_revision", "test_kbound_pdf_build_isolation"):
        assert f"tests/{test_name}.py" in runbook
