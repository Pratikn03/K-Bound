"""Phase 1.1 — bundle of small consistency tests.

Covers:
  - test_real3d_single_policy_valid_row
  - test_no_promoted_canonical_degenerate_metrics
  - test_no_canonical_pr_auc_headline_figure (via mvtec3d cleanup tables only)
  - test_no_unsw_generalization_overclaim
  - test_master_table_validation_frozen_language
  - test_master_table_no_test_oracle_labels
  - test_family_a_not_called_confirmatory
  - test_no_causal_overclaim_in_manuscripts
  - test_model_response_sensitivity_title_present
  - test_pdf_text_no_deployment_grade_polarity
  - test_primary_path_never_flips_for_reporting
  - test_thesis_contains_audited_policy_section
  - test_thesis_abstract_matches_claim_boundary
  - test_reported_baselines_are_defined
  - test_phase1_1_no_stale_generated_tables
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path("docs/research/PAPER_DRAFT_v1.tex")
THESIS = Path("docs/research/THESIS_CHAPTER_v1.tex")
RUNNER = Path("src/scripts/run_breakthrough_experiment.py")
TABLES_DIR = Path("docs/research/tables")


# --- Real3D single policy-valid row ---
def test_real3d_single_policy_valid_row():
    paper = PAPER.read_text()
    # No "no longer the negative cell" claim
    assert "no longer the negative cell" not in paper, (
        "paper still contains forbidden Real3D 'no longer the negative cell' claim"
    )
    # No "FPFH+depth" stale label
    assert "FPFH+depth" not in paper
    # No "strongest non-router" framing
    assert "strongest non-router" not in paper


# --- Canonical promoted-metrics policy ---
def test_no_promoted_canonical_degenerate_metrics():
    canonical_files = [
        TABLES_DIR / "mvtec3d_patchcore_clean_ci_results.tex",
        TABLES_DIR / "mvtec3d_patchcore_calibration_cda.tex",
        TABLES_DIR / "mvtec3d_clean_ci_results.tex",
        TABLES_DIR / "mvtec3d_calibration_cda.tex",
    ]
    for f in canonical_files:
        if not f.exists():
            continue
        text = f.read_text()
        assert "0.7835" not in text, (
            f"canonical table {f.name} still contains 0.7835 — Phase 1.1 canonical cleanup not applied"
        )


def test_no_canonical_pr_auc_headline_figure():
    assert (TABLES_DIR / "mvtec3d_patchcore_clean_ci_results.tex").exists()
    txt = (TABLES_DIR / "mvtec3d_patchcore_clean_ci_results.tex").read_text()
    assert "ROC-AUC mean" in txt and "ROC-AUC 95\\% CI" in txt
    # Strip explanatory '%' comment lines before checking for forbidden tokens in the data rows.
    body = "\n".join(line for line in txt.splitlines() if not line.lstrip().startswith("%"))
    assert "PR-AUC" not in body
    assert "Brier" not in body
    assert r"\textbf{ECE}" not in body


# --- UNSW overclaim ---
def test_no_unsw_generalization_overclaim():
    for path in (PAPER, THESIS):
        t = path.read_text()
        for forbidden in ("prove the cross-benchmark",
                          "beats every non-ELARA",
                          "without losing the cross-domain generalization"):
            assert forbidden not in t, (
                f"{path} still contains forbidden UNSW phrase: {forbidden!r}"
            )


# --- Master table language ---
def test_master_table_validation_frozen_language():
    t = (TABLES_DIR / "milestone2_cross_benchmark.tex").read_text().lower()
    # Accept either "validation-frozen" or the abbreviated "val-frozen" used in the
    # rendered column headers.
    assert ("validation-frozen" in t) or ("val-frozen" in t), (
        "master table missing validation-frozen language in column headers"
    )


def test_master_table_no_test_oracle_labels():
    t = (TABLES_DIR / "milestone2_cross_benchmark.tex").read_text()
    for forbidden in ("Best non-router", "best non-router",
                      "MAX(router", "max(router"):
        assert forbidden not in t, f"master table contains forbidden header: {forbidden!r}"
    # Also check the ablation + milestone1 tables
    for tname in ("rga_plus_ablation.tex", "mvtec3d_milestone1_comparison.tex"):
        f = TABLES_DIR / tname
        if not f.exists():
            continue
        tx = f.read_text()
        for forbidden in ("Best non-router", "best non-router"):
            assert forbidden not in tx, f"{tname} still contains forbidden header: {forbidden!r}"


# --- Family A naming ---
def test_family_a_not_called_confirmatory():
    for path in (PAPER, THESIS):
        t = path.read_text()
        assert "Family A confirmatory" not in t


# --- Causal language gone from results ---
def test_no_causal_overclaim_in_manuscripts():
    for path in (PAPER, THESIS):
        t = path.read_text()
        for forbidden in ("Causal Reliability Attribution",
                          "Causal Inference for Reliability",
                          "Structural Causal Model",
                          "interventional ATE"):
            assert forbidden not in t, f"{path} contains forbidden causal phrase: {forbidden!r}"


# --- Model-response sensitivity title present ---
def test_model_response_sensitivity_title_present():
    paper = PAPER.read_text()
    assert "Model-Response Sensitivity to Per-Domain Reliability" in paper, (
        "paper missing Model-Response Sensitivity section title"
    )


# --- Polarity diagnostic only ---
def test_pdf_text_no_deployment_grade_polarity():
    for path in (PAPER, THESIS):
        t = path.read_text()
        assert "deployment-grade sanity check" not in t


def test_primary_path_never_flips_for_reporting():
    t = RUNNER.read_text()
    forbidden_patterns = [
        re.compile(r"^\s*static_probs\s*=\s*1\.0\s*-\s*static_probs", re.MULTILINE),
        re.compile(r"^\s*craf_probs\s*=\s*1\.0\s*-\s*craf_probs", re.MULTILINE),
    ]
    for pat in forbidden_patterns:
        assert pat.search(t) is None, "runner still applies a polarity flip to primary path"


# --- Thesis audited-policy subsection ---
def test_thesis_contains_audited_policy_section():
    t = THESIS.read_text()
    assert "Locked Audited-Reanalysis Policy and Future Replication Boundary" in t


def test_thesis_abstract_matches_claim_boundary():
    t = THESIS.read_text()
    abstract = t[:5000]
    # The thesis abstract must describe mixed audited outcomes; must include the primary deltas
    assert "+0.0506" in abstract and "+0.0319" in abstract
    # The thesis abstract must reference the UNSW practically-very-small effect
    assert "+0.0003" in abstract
    # And must NOT contain broad-cross-domain superiority claims.
    forbidden = ["beats every", "universally superior", "production-ready",
                 "broad cross-domain superiority", "leaderboard-leading"]
    for f in forbidden:
        assert f not in abstract


# --- Baselines defined ---
def test_reported_baselines_are_defined():
    # If EATA / SAR appear in any final results table, they must be defined in BOTH manuscripts.
    abl = (TABLES_DIR / "rga_plus_ablation.tex").read_text()
    if "EATA" in abl or "SAR" in abl:
        for path in (PAPER, THESIS):
            t = path.read_text()
            assert "EATA" in t, f"{path} references EATA in tables but does not define it"
            assert ("SAR" in t and "yang2023sar" in t), (
                f"{path} references SAR but does not define it or cite Yang et al. 2023"
            )


# --- No stale generated tables ---
def test_phase1_1_no_stale_generated_tables():
    for tname in ("rga_plus_ablation.tex", "milestone2_cross_benchmark.tex",
                  "mvtec3d_milestone1_comparison.tex"):
        f = TABLES_DIR / tname
        if not f.exists():
            continue
        t = f.read_text()
        assert "Best non-router" not in t and "best non-router" not in t
