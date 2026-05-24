"""Phase 2.1 — Family-D v2 must not list any dataset that has been
inspected in Family A or referenced by the Family-A registry."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"
REGISTRY_V2 = PHASE2 / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
ELIG = PHASE2 / "FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md"
HYP_V2 = PHASE2 / "FAMILY_D_HYPOTHESES_v2.csv"


def _family_a_benchmarks() -> set[str]:
    with REGISTRY_V2.open() as f:
        rows = list(csv.DictReader(f))
    return {r["benchmark"] for r in rows if r["analysis_family"] == "A"}


def test_eligibility_review_explicitly_excludes_visa():
    """VisA is registry-locked into Family A, so it must be marked
    INELIGIBLE_FOR_FAMILY_D in the eligibility review."""
    t = ELIG.read_text()
    assert "VisA" in t, "eligibility review must address VisA"
    # Look at the VisA paragraph: it must include the INELIGIBLE_FOR_FAMILY_D marker
    visa_idx = t.find("VisA")
    visa_block = t[visa_idx : visa_idx + 800]
    assert "INELIGIBLE_FOR_FAMILY_D" in visa_block, (
        "VisA must be marked INELIGIBLE_FOR_FAMILY_D in the eligibility review"
    )


def test_v2_hypotheses_do_not_use_any_family_a_benchmark():
    """If a v2 hypotheses CSV exists, no row may reference a Family-A benchmark."""
    if not HYP_V2.exists():
        # v2 is V2_DESIGN_PENDING — absence of HYP_V2 is correct
        return
    family_a = _family_a_benchmarks()
    with HYP_V2.open() as f:
        for row in csv.DictReader(f):
            bench = row.get("dataset", "")
            assert bench not in family_a, (
                f"v2 hypothesis {row.get('hypothesis_id')!r} references "
                f"Family-A benchmark {bench!r}"
            )


def test_v2_hypotheses_do_not_reference_visa_anywhere():
    """Even if a candidate v2 hypotheses file exists, VisA must not appear."""
    if not HYP_V2.exists():
        return
    t = HYP_V2.read_text()
    assert "VisA" not in t, "VisA must not appear anywhere in v2 hypotheses"
