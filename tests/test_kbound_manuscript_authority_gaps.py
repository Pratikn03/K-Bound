"""Regression checks for the maintained K-Bound authority boundary.

These checks intentionally inspect only the active manuscript, claim, formal, and
generated-manifest surfaces.  They do not load empirical target artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KBOUND = ROOT / "docs/research/kbound"
SYNC_PATH = ROOT / "scripts/sync_reconciled_panels.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("authority_gap_sync", SYNC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_definitions_keep_population_and_measured_cell_targets_distinct() -> None:
    manifest = json.loads((KBOUND / "paper/generated/kbound_result_manifest.json").read_text(encoding="utf-8"))
    definitions = manifest["definitions"]

    assert definitions["population_benefit"] == "Delta = R_T(f_0) - R_T(f_a)"
    assert definitions["measured_cell_benefit"].startswith("Delta^cell = S(f_a; E) - S(f_0; E)")
    assert "adapt-minus-frozen" in definitions["measured_cell_benefit"]
    assert "# measured cells" in definitions["false_adapt_unconditional"]
    assert "empirical fraction is not population risk" in definitions["false_adapt_unconditional"]
    assert "same declared scalar target" in definitions["population_certificate_target"]


def test_synchronizer_writes_the_authority_boundary_without_a_fabricated_timestamp() -> None:
    sync = _load_sync_module()
    table: dict[str, Any] = {}

    sync._set_manifest_authority_metadata(table)

    assert table["generated_for"] == (
        "all five maintained drivers: kbound_submission.tex, kbound_tmlr.tex, "
        "kbound_short_main.tex, kbound_short_supplement.tex, and kbound_full_report.tex"
    )
    assert table["generated_at"] is None
    assert "not recorded" in table["generation_identity"]
    definitions = table["definitions"]
    assert definitions["population_benefit"] == "Delta = R_T(f_0) - R_T(f_a)"


def test_certificate_claim_names_a_fixed_scalar_target_and_joint_unconditional_event() -> None:
    ledger = json.loads((KBOUND / "claim_ledger.json").read_text(encoding="utf-8"))
    claim = next(row for row in ledger["claims"] if row["claim_id"] == "KB-CLAIM-003")

    assert "evidence-synchronization date" in ledger["generation_identity"]
    assert "not an artifact-creation timestamp" in ledger["generation_identity"]
    assert "fixed declared scalar target B" in claim["claim_text"]
    assert "P(g=ADAPT and B<=0)" in claim["claim_text"]
    assert "P(B<=0 | g=ADAPT)" in claim["claim_text"]
    assert claim["allowed_wording"] == "P(g=ADAPT and B<=0) <= alpha under marginal coverage for B"
    assumptions = " ".join(claim["assumptions"])
    assert "common probability space" in assumptions
    assert "possibly infinite radius" in assumptions
    assert "exchangeability of the actual scores" in assumptions


def test_claim_manifest_points_to_the_currently_named_cifar_tables() -> None:
    manifest = (KBOUND / "KBOUND_SHORT_CLAIM_MANIFEST.md").read_text(encoding="utf-8")
    body = (KBOUND / "kbound_submission_body.tex").read_text(encoding="utf-8")
    supplement = (KBOUND / "kbound_submission_supplement.tex").read_text(encoding="utf-8")
    body_table_labels = re.findall(
        r"\\begin\{table\*?\}.*?\\label\{([^}]+)\}.*?\\end\{table\*?\}",
        body,
        flags=re.DOTALL,
    )
    supplement_table_labels = re.findall(
        r"\\begin\{table\*?\}.*?\\label\{([^}]+)\}.*?\\end\{table\*?\}",
        supplement,
        flags=re.DOTALL,
    )

    assert "Generation time: not recorded" in manifest
    assert "Date: 2026-08-29" not in manifest
    assert "tab:cifar-primary" in body_table_labels
    # The 45-page TMLR revision keeps the primary action/utility table in the
    # main results and moves the full interval and family diagnostics to
    # Appendix F.  Guard the semantic labels and their document roles rather
    # than a brittle ordinal table position.
    assert "tab:cifar-interval-diagnostics" not in body_table_labels
    assert "tab:cifar-interval-diagnostics" in supplement_table_labels
    assert "tab:cifar-family-sensitivity" in supplement_table_labels
    assert "`tab:cifar-primary`" in manifest
    assert "`tab:cifar-interval-diagnostics`" in manifest
    assert "`tab:cifar-family-sensitivity`" in manifest
    assert "(Table 5)" not in manifest
    assert "(Table 6)" not in manifest
    assert "Tables 4-5" not in manifest
    certificate_row = next(line for line in manifest.splitlines() if line.startswith("| Marginal FA_u certificate |"))
    assert "B_hat-B" in certificate_row
    assert "Delta_hat-Delta" not in certificate_row


def test_formal_comments_point_to_active_theory_sources() -> None:
    certificate = (KBOUND / "formal/KBound/Certificate.lean").read_text(encoding="utf-8")
    gate = (KBOUND / "formal/KBound/Gate.lean").read_text(encoding="utf-8")
    frontier = (KBOUND / "formal/KBound/Frontier.lean").read_text(encoding="utf-8")

    assert "paper/sections/theory_certificate.tex" in certificate
    assert "manuscript/chapters/" not in certificate
    assert "kbound_submission_body.tex" in gate
    assert "Section `sec:compact-kga`" in gate
    assert "not separately stated as an active paper theorem" in gate
    assert "Lemma `thm:gate`" not in gate
    assert "main_theory_5.tex" not in gate
    assert "paper/sections/theory_core_main.tex" in frontier
    assert "main_theory_5.tex" not in frontier


def test_live_body_uses_plain_candidate_tta_transductive_disclosure() -> None:
    body = (KBOUND / "kbound_submission_body.tex").read_text(encoding="utf-8")

    assert "candidate TTA is transductive" in body
    assert "unlabeled evaluation inputs" in body
    assert "evaluation-batch BatchNorm statistics" in body


def test_reader_facing_surfaces_name_regret_in_two_baseline_comparisons() -> None:
    surfaces = (
        KBOUND / "kbound_submission_body.tex",
        KBOUND / "kbound_submission_supplement.tex",
        KBOUND / "KBOUND_SHORT_CLAIM_MANIFEST.md",
        KBOUND / "KBOUND_SHORT_RESULT_AUDIT.md",
        KBOUND / "paper/generated/kbound_result_manifest.json",
        KBOUND / "results_source.json",
    )
    for path in surfaces:
        text = path.read_text(encoding="utf-8").lower()
        assert "beats both" not in text, path
        assert "beats-both" not in text, path
