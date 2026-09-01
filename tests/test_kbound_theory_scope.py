"""Exact-rational sanity checks and wording guards for the scoped paper theory.

These finite examples are regression checks, not a replacement for the proofs.
They do not load model outputs, labels, checkpoints, or benchmark data.
"""

from fractions import Fraction
from pathlib import Path

import pytest


PAPER = Path(__file__).resolve().parents[1] / "docs/research/kbound"
HALF = Fraction(1, 2)
MARGINS = [Fraction(i, 20) for i in range(-10, 11)]
BUDGETS = [Fraction(i, 20) for i in range(21)]


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
    residual = sum(
        w * (c - s)
        for w, c, s, d in zip(weights, adapted_correct, scores, disagreement)
        if d
    ) / mass
    frozen_risk = sum(w * int(p != y) for w, p, y in zip(weights, frozen, labels))
    adapted_risk = sum(w * int(p != y) for w, p, y in zip(weights, adapted, labels))
    assert (mass, margin, residual) == (HALF, 0, HALF)
    assert frozen_risk - adapted_risk == 2 * mass * (margin + residual) == HALF


def test_manuscript_keeps_the_refined_scope_explicit():
    body = (PAPER / "kbound_submission_body.tex").read_text()
    supplement = (PAPER / "kbound_submission_supplement.tex").read_text()
    core = (PAPER / "paper/sections/theory_core_main.tex").read_text()
    abstract = (PAPER / "kbound_abstract_core.tex").read_text()
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
    assert "oracle benchmark for the fixed evidence-law fibre" in body
    assert "Repeated deployment needs a separate guarantee." in body
    assert "predeclared policy does not by" in body
    assert "Why a calibration residual need not be drift." in supplement


def test_closest_novelty_comparisons_remain_cited():
    body = (PAPER / "kbound_submission_body.tex").read_text()
    for key in (
        "bendavid2010impossibility",
        "lamaakal2026drifttoaction",
        "schirmer2025monitoring",
        "steinhardt2016",
        "angelopoulos2025ltt",
    ):
        assert key in body
    assert "does not introduce label-free impossibility" in body
    assert "does not establish an anytime-valid risk guarantee" in " ".join(body.split())
