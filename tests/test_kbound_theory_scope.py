"""Exact-rational sanity checks and wording guards for the scoped paper theory.

These finite examples are regression checks, not a replacement for the proofs.
They do not load model outputs, labels, checkpoints, or benchmark data.
"""

from fractions import Fraction
from pathlib import Path
import re

import pytest

from tests.kbound_tex_helpers import live_tex

PAPER = Path(__file__).resolve().parents[1] / "docs/research/kbound"
HALF = Fraction(1, 2)
MARGINS = [Fraction(i, 20) for i in range(-10, 11)]
BUDGETS = [Fraction(i, 20) for i in range(21)]


def live(path: Path) -> str:
    return live_tex(path.read_text(encoding="utf-8"))


def test_general_audit_floor_states_the_kernel_checked_premises():
    body = " ".join(live(PAPER / "kbound_submission_body.tex").split())
    section = body.split(r"\subsection{Audit Setup and Evidence Fibres}", 1)[1].split(
        r"\paragraph{When calibration labels do not shrink the target fibre.}", 1
    )[0]
    assert "measurable real-valued audit" in section
    assert "the same distribution in every world" in section
    assert "bounded attainable residuals" in section
    assert "common joint law of $(W,U)$" in section
    assert "same law for $W$ and therefore" not in section
    assert r"0\le\beta\le\tfrac12" in section


def test_supplement_formal_inventory_and_scope_match_the_current_audit():
    supplement = live(PAPER / "kbound_submission_supplement.tex")
    section = supplement.split(r"\section{Formalization and Artifact Scope}", 1)[1].split(
        r"\section{Reproducibility and Release Procedures}", 1
    )[0]
    include = r"\input{paper/sections/proof_traceability}"
    assert [line.strip() for line in section.splitlines()].count(include) == 1
    assert re.findall(r"\\input\s*\{([^{}]+)\}", section) == ["paper/sections/proof_traceability"]
    section = " ".join(section.split())
    trace = " ".join(live(PAPER / "paper/sections/proof_traceability.tex").split())
    assert "303 registered declarations: 65 retained core declarations and 238 additional results" in trace
    assert "142 declarations" not in section + trace
    assert r"\paragraph{Audit floor.}" in trace
    audit_floor = trace.split(r"\paragraph{Audit floor.}", 1)[1].split(r"\paragraph", 1)[0]
    assert "possibly unattained supremum of a nonempty bounded residual fibre under a common observation law" in audit_floor
    assert "does not supply the reported experiments with a population sampling-error radius" in trace
    assert r"\texttt{--full-foundations} gate therefore remains failing for the sixth layer" in section
    assert "it does not claim that every sentence has a one-to-one Lean theorem" in section
    assert "These are different inventories, not a percentage of the paper proved." in trace
    assert "unchanged Lean sources do not make every revised sentence kernel-checked." in trace
    assert "the entire paper is kernel-checked" not in (section + trace).casefold()


def test_live_excludes_disabled_and_commented_scientific_claims(tmp_path):
    source = tmp_path / "scope.tex"
    source.write_text(
        "maintained scope\n\\iffalse\ndisabled scope\n\\fi\n% commented scope\n",
        encoding="utf-8",
    )

    assert " ".join(live(source).split()) == "maintained scope"


@pytest.fixture
def formal_scope_sources(tmp_path, monkeypatch):
    """Minimal live prose; mutation expectations do not depend on manuscript data."""
    supplement = tmp_path / "kbound_submission_supplement.tex"
    trace = tmp_path / "paper/sections/proof_traceability.tex"
    trace.parent.mkdir(parents=True)
    supplement.write_text(
        "\\section{Formalization and Artifact Scope}\n"
        "it does not claim that every sentence has a one-to-one Lean theorem.\n"
        "\\input{paper/sections/proof_traceability}\n"
        "\\texttt{--full-foundations} gate therefore remains failing for the sixth layer.\n"
        "\\section{Reproducibility and Release Procedures}\n", encoding="utf-8",
    )
    trace.write_text(
        "303 registered declarations: 65 retained core declarations and 238 additional results.\n"
        "These are different inventories, not a percentage of the paper proved.\n"
        "\\paragraph{Audit floor.}\n"
        "possibly unattained supremum of a nonempty bounded residual fibre under a common observation law.\n"
        "\\paragraph{Coverage and decision.}\n"
        "does not supply the reported experiments with a population sampling-error radius.\n"
        "unchanged Lean sources do not make every revised sentence kernel-checked.\n", encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "PAPER", tmp_path)
    return supplement, trace


def test_formal_scope_accepts_the_live_shared_traceability(formal_scope_sources):
    test_supplement_formal_inventory_and_scope_match_the_current_audit()


@pytest.mark.parametrize("mutation", [
    "removed_include", "commented_include", "disabled_include", "duplicate_include", "outside_include",
    "total_count", "core_count", "additional_count", "common_law", "unattained_supremum",
    "deleted_sampling_limit", "positive_sampling_claim", "hidden_sampling_limit", "hidden_traceability",
    "sixth_layer_pass", "hidden_blanket_limit", "live_blanket_claim",
])
def test_formal_scope_rejects_missing_live_bindings_and_scope_mutations(formal_scope_sources, mutation):
    supplement, trace = formal_scope_sources
    outer, shared = supplement.read_text(), trace.read_text()
    include = r"\input{paper/sections/proof_traceability}"
    sampling = "does not supply the reported experiments with a population sampling-error radius."
    if mutation == "removed_include":
        outer = outer.replace(include, "")
    elif mutation == "commented_include":
        outer = outer.replace(include, "% " + include)
    elif mutation == "disabled_include":
        outer = outer.replace(include, "\\iffalse\n" + include + "\n\\fi")
    elif mutation == "duplicate_include":
        outer = outer.replace(include, include + "\n" + include)
    elif mutation == "outside_include":
        outer = outer.replace(include, "") + include + "\n"
    elif mutation in {"total_count", "core_count", "additional_count"}:
        old, new = {"total_count": ("303", "304"), "core_count": ("65", "64"),
                    "additional_count": ("238", "237")}[mutation]
        shared = shared.replace(old, new)
    elif mutation == "common_law":
        shared = shared.replace("common observation law", "unrelated observation laws")
    elif mutation == "unattained_supremum":
        shared = shared.replace("possibly unattained supremum", "attained maximum")
    elif mutation == "deleted_sampling_limit":
        shared = shared.replace(sampling, "")
    elif mutation == "positive_sampling_claim":
        shared = shared.replace(sampling, sampling.replace("does not supply", "does supply"))
    elif mutation == "hidden_sampling_limit":
        shared = shared.replace(sampling, "% " + sampling)
    elif mutation == "hidden_traceability":
        shared = "\\iffalse\n" + shared + "\\fi\n"
    elif mutation == "sixth_layer_pass":
        outer = outer.replace("remains failing for the sixth layer", "passes every layer")
    elif mutation == "hidden_blanket_limit":
        outer = outer.replace("it does not claim", "% it does not claim")
    else:
        shared += "The entire paper is kernel-checked.\n"
    supplement.write_text(outer)
    trace.write_text(shared)
    with pytest.raises(AssertionError):
        test_supplement_formal_inventory_and_scope_match_the_current_audit()


def test_hidden_blanket_claims_do_not_become_live_prose(formal_scope_sources):
    _, trace = formal_scope_sources
    trace.write_text(trace.read_text() + "% The entire paper is kernel-checked.\n"
                     "\\iffalse\nThe entire paper is kernel-checked.\n\\fi\n")
    test_supplement_formal_inventory_and_scope_match_the_current_audit()


@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("budget", BUDGETS)
def test_feasible_identified_interval_frontier_and_audit_floor(margin, budget):
    """Check 441 feasible (M, beta) pairs, including endpoints and beta > 1/2."""
    residual_lo = max(-budget, -HALF - margin)
    residual_hi = min(budget, HALF - margin)
    assert residual_lo <= residual_hi

    normalized_benefit_lo = margin + residual_lo
    normalized_benefit_hi = margin + residual_hi
    assert normalized_benefit_lo == max(-HALF, margin - budget)
    assert normalized_benefit_hi == min(HALF, margin + budget)
    assert (normalized_benefit_lo > 0) == (margin > budget)
    assert (normalized_benefit_hi < 0) == (margin < -budget)

    # Constant target-correctness kernels attain both interval endpoints.
    for normalized_benefit in (normalized_benefit_lo, normalized_benefit_hi):
        correctness = HALF + normalized_benefit
        residual = correctness - HALF - margin
        assert 0 <= correctness <= 1
        assert abs(residual) <= budget

    fibre_radius = max(abs(residual_lo), abs(residual_hi))
    assert fibre_radius == min(budget, HALF + abs(margin))
    if budget <= HALF:
        assert fibre_radius == budget

    if abs(margin) < budget:
        delta = min(budget - abs(margin), HALF) / 2
        for direction in (-1, 1):
            correctness = HALF + direction * delta
            residual = direction * delta - margin
            assert 0 <= correctness <= 1
            assert abs(residual) < budget
            assert margin + residual == direction * delta
    elif abs(margin) == budget:
        assert residual_lo <= -margin <= residual_hi
        assert normalized_benefit_lo <= 0 <= normalized_benefit_hi


def test_source_calibration_does_not_remove_disagreement_residual():
    """The same source/target law can have gamma=1/2 after conditioning on D."""
    labels = (1, 0)
    frozen = (0, 1)
    adapted = (1, 1)
    weights = (HALF, HALF)
    scores = (HALF, HALF)
    adapted_correct = tuple(int(p == y) for p, y in zip(adapted, labels))
    assert sum(w * c for w, c in zip(weights, adapted_correct)) == HALF
    assert sum(w * s for w, s in zip(weights, scores)) == HALF
    disagreement = tuple(a != f for a, f in zip(adapted, frozen))
    mass = sum(w for w, d in zip(weights, disagreement) if d)
    margin = sum(w * s for w, s, d in zip(weights, scores, disagreement) if d) / mass - HALF
    residual = sum(w * (c - s) for w, c, s, d in zip(weights, adapted_correct, scores, disagreement) if d) / mass
    frozen_risk = sum(w * int(p != y) for w, p, y in zip(weights, frozen, labels))
    adapted_risk = sum(w * int(p != y) for w, p, y in zip(weights, adapted, labels))
    assert (mass, margin, residual) == (HALF, 0, HALF)
    assert frozen_risk - adapted_risk == 2 * mass * (margin + residual) == HALF


def test_manuscript_keeps_the_refined_scope_explicit():
    body = live(PAPER / "kbound_submission_body.tex")
    supplement = live(PAPER / "kbound_submission_supplement.tex")
    core = live(PAPER / "paper/sections/theory_core_main.tex")
    abstract = live(PAPER / "kbound_abstract_core.tex")
    for stale in (
        "declared calibration-drift class",
        "target calibration drift that labels would reveal",
        "Exact Minimax Label-Free Budget",
        "declared drift class supports",
    ):
        assert stale not in body + supplement + core + abstract
    assert "For any feasible margin" in core
    assert r"M\in[-\tfrac12,\tfrac12]" in core
    assert "fixed measurable candidate-correctness score" in body
    assert "This is weaker than uniform strict-sign" in body
    fibre = body.split(r"\subsection{Audit Setup and Evidence Fibres}", 1)[1].split(
        r"\paragraph{Declared budget and abstention radius.}", 1
    )[0]
    fibre = " ".join(fibre.split())
    assert r"\mathcal C(z)=\{P\in\mathcal C:\Law_P(W)=z\}" in fibre
    assert "oracle fibre benchmark" in fibre
    assert "not a general finite-batch estimator" in fibre
    assert "Repeated deployment needs a separate guarantee." in body
    assert "predeclared policy does not by" in body
    assert "Why a calibration residual need not be drift." in supplement


def test_closest_novelty_comparisons_remain_cited():
    body = live(PAPER / "kbound_submission_body.tex")
    for key in (
        "bendavid2010impossibility",
        "lamaakal2026drifttoaction",
        "schirmer2025monitoring",
        "steinhardt2016",
        "angelopoulos2025ltt",
    ):
        assert key in body
    assert "does not introduce label-free impossibility" in body
    controllers = body.split(r"\subsection{Monitoring and Adaptation Controllers}", 1)[1].split(
        r"\subsection{Partial Identification and Robust Decisions}", 1
    )[0]
    assert "it makes no anytime-valid risk claim" in " ".join(controllers.split())
