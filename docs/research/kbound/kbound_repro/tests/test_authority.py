"""Authority-chain tests (Phase 7): claim matrix, forbidden wording, consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kbound_repro import authority as A  # noqa: E402
from kbound_repro import manuscript_sources as MS  # noqa: E402

KBOUND = Path(__file__).resolve().parents[2]
LEDGER = A.load_ledger(KBOUND / "claim_ledger.json")
MANIFEST = json.loads((KBOUND / "RESULT_MANIFEST.json").read_text())

WITHDRAWN = {"KB-CLAIM-004", "KB-CLAIM-012", "KB-CLAIM-022", "KB-CLAIM-023", "KB-CLAIM-050"}


def test_matrix_marks_withdrawn():
    rows = {r["claim_id"]: r for r in A.build_claim_matrix(LEDGER)}
    for cid in WITHDRAWN:
        assert rows[cid]["status"] == "withdrawn"
        assert rows[cid]["promoted"] is False


def test_matrix_marks_withheld_as_not_promoted():
    row = {r["claim_id"]: r for r in A.build_claim_matrix(LEDGER, MANIFEST)}["KB-CLAIM-021"]
    assert row["status"] == "withheld"
    assert row["promoted"] is False
    assert row["in_manifest"] is False


def test_forbidden_scan_catches_withdrawn_variants():
    samples = {
        "KB-CLAIM-022": "Our method beats both always-adapt and always-freeze on Camelyon17.",
        "KB-CLAIM-023": "We observe a 13x reduction in regret on the mixed stream.",
        "KB-CLAIM-012": "A jackknife+ finite-sample guarantee holds for the grid.",
        "KB-CLAIM-050": "KGA yields universal accuracy improvement across datasets.",
        "KB-CLAIM-003": "The certificate is guaranteed safe in the wild.",
    }
    for cid, text in samples.items():
        hits = A.scan_text_for_forbidden(text, LEDGER)
        assert any(h["claim_id"] == cid for h in hits), f"missed forbidden variant for {cid}"


def test_clean_promoted_text_has_no_withdrawn_hits():
    clean = (
        "On the CIFAR-10-C stress grid KGA beats both fixed policies under Protocol A. "
        "On Camelyon17 we report genuine OOD no-harm. The certificate bounds FA_u under "
        "the stated assumptions; risk alignment is assumed."
    )
    assert A.scan_text_for_forbidden(clean, LEDGER, only_withdrawn=True) == []


def test_camelyon_no_harm_wording_is_allowed_but_beats_both_is_not():
    assert A.scan_text_for_forbidden("Camelyon17 genuine OOD no-harm.", LEDGER, only_withdrawn=True) == []
    bad = A.scan_text_for_forbidden("Camelyon17 beats both fixed policies.", LEDGER, only_withdrawn=True)
    assert any(h["claim_id"] == "KB-CLAIM-022" for h in bad)


def test_curated_withheld_scan_catches_iwildcam_promotion_but_allows_disclosure():
    bad = A.scan_text_for_unreleased_curated(
        "The promoted iWildCam result ties always-freeze with zero adapt decisions.", LEDGER
    )
    assert any(hit["claim_id"] == "KB-CLAIM-021" for hit in bad)
    safe = A.scan_text_for_unreleased_curated(
        "The iWildCam numerical and action row is withheld pending an official-metric rerun.",
        LEDGER,
    )
    assert safe == []


@pytest.mark.parametrize(
    "leak",
    [
        r"iWildCam H--v2 & 72 & \multicolumn{5}{c}{withheld} \\",
        (
            "iWildCam is withheld. A diagnostic recomputation gives regrets "
            "0.005/0.075/0.005 and actions 0/4/68."
        ),
        r"iWildCam H v2 & \iWN & \iWAdaptCount & \iWFreezeCount & \iWAbstainCount \\",
    ],
)
def test_curated_withheld_scan_rejects_iwildcam_values_even_with_disclaimer(leak):
    hits = A.scan_text_for_unreleased_curated(leak, LEDGER)
    assert any(hit["claim_id"] == "KB-CLAIM-021" for hit in hits)


def test_maintained_source_inventory_includes_submission_and_excludes_archives():
    relative = {
        path.relative_to(KBOUND.parents[2]).as_posix()
        for path in MS.active_source_paths(KBOUND.parents[2])
    }
    assert "docs/research/kbound/kbound_submission_body.tex" in relative
    assert "docs/research/kbound/kbound_tmlr.tex" in relative
    assert "docs/research/kbound/paper/generated/kbound_numbers.tex" in relative
    assert "experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex" in relative
    assert not any("/archive/" in path for path in relative)
    assert all(path.is_file() for path in MS.active_source_paths(KBOUND.parents[2]))


def test_live_latex_removes_disabled_and_commented_assertions():
    text = "kept\n\\iffalse\nforbidden claim\n\\fi\n% forbidden comment\nalso kept"
    assert MS.live_latex(text) == "kept\n\n\nalso kept"


def _mini_manifest(claim_ids):
    return {
        "schema_version": "kbound-result-manifest-v1",
        "results": [
            {"claim_id": c, "dataset": "d", "protocol": "p", "status": "supported",
             "source_artifact": "a.json"}
            for c in claim_ids
        ],
    }


def test_claim_matrix_backing_requires_exact_status_agreement():
    manifest = _mini_manifest(["KB-CLAIM-010"])
    good = {row["claim_id"]: row for row in A.build_claim_matrix(LEDGER, manifest)}
    assert good["KB-CLAIM-010"]["backing_ok"] is True
    manifest["results"][0]["status"] = "diagnostic"
    bad = {row["claim_id"]: row for row in A.build_claim_matrix(LEDGER, manifest)}
    assert bad["KB-CLAIM-010"]["manifest_status_agrees"] is False
    assert bad["KB-CLAIM-010"]["backing_ok"] is False


def test_consistency_flags_unbacked_supported_empirical():
    # Manifest backs only 010; other supported empirical claims must be flagged
    # unless long-paper-only.
    manifest = _mini_manifest(["KB-CLAIM-010"])
    problems = A.consistency_problems(LEDGER, manifest)
    assert any("KB-CLAIM-011" in p for p in problems)
    # Marking the rest long-paper-only clears them:
    others = ["KB-CLAIM-011", "KB-CLAIM-020", "KB-CLAIM-021", "KB-CLAIM-024",
              "KB-CLAIM-026", "KB-CLAIM-027"]
    problems2 = A.consistency_problems(LEDGER, manifest, long_paper_only=others)
    assert not any(c in " ".join(problems2) for c in others)


def test_manifest_may_not_carry_withdrawn_claim():
    manifest = _mini_manifest(["KB-CLAIM-022"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("withdrawn in ledger but present" in p for p in problems)


def test_manifest_may_not_carry_withheld_claim_under_a_disguised_status():
    manifest = _mini_manifest(["KB-CLAIM-021"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("withheld in ledger but present" in p for p in problems)


def test_manifest_status_must_match_ledger_status():
    # KB-CLAIM-020 is descriptive in the ledger; disguising it as supported
    # would silently promote the claim without this cross-authority check.
    manifest = _mini_manifest(["KB-CLAIM-020"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("manifest status 'supported' disagrees with ledger status 'descriptive'" in p
               for p in problems)


def test_manifest_rejects_duplicate_claim_ids():
    manifest = _mini_manifest(["KB-CLAIM-010", "KB-CLAIM-010"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("duplicate claim_id KB-CLAIM-010" in p for p in problems)


def test_consistency_rejects_duplicate_ledger_claim_ids_without_schema_gate():
    ledger = {**LEDGER, "claims": [*LEDGER["claims"], dict(LEDGER["claims"][0])]}
    problems = A.consistency_problems(ledger, MANIFEST)
    assert any("claim ledger contains duplicate claim IDs" in p for p in problems)


def test_numerical_manifest_may_reference_only_empirical_claims():
    manifest = _mini_manifest(["KB-CLAIM-001"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("non-empirical ledger claim type 'theorem'" in p for p in problems)


def test_manifest_unknown_claim_flagged():
    manifest = _mini_manifest(["KB-CLAIM-999"])
    problems = A.consistency_problems(LEDGER, manifest, long_paper_only=[
        c["claim_id"] for c in LEDGER["claims"] if c["claim_type"] == "empirical"
    ])
    assert any("unknown claim" in p for p in problems)


def test_disclaimers_are_not_flagged():
    # A paper is allowed to state what it does NOT claim.
    disclaimers = [
        "K-Bound does not claim universal improvement across datasets.",
        "The finite-sample jackknife+ guarantee is not claimed for the stress grid.",
        "FA_c is reported descriptively; FA_c <= alpha is not claimed.",
    ]
    for text in disclaimers:
        assert A.scan_text_for_forbidden(text, LEDGER, only_withdrawn=True) == [], text


def test_loo_jackknife_method_name_is_not_forbidden_but_assertive_guarantee_is():
    # "LOO jackknife q_0.9" is a legitimate calibration method name.
    assert A.scan_text_for_forbidden("Calibration uses LOO jackknife q_0.9 per cell.", LEDGER) == []
    # An assertive jackknife+ guarantee is forbidden (KB-CLAIM-012 withdrawn).
    hits = A.scan_text_for_forbidden("A jackknife+ finite-sample guarantee holds on the grid.", LEDGER)
    assert any(h["claim_id"] == "KB-CLAIM-012" for h in hits)


def test_real_promoted_manuscripts_have_no_withdrawn_wording():
    import glob
    tex = glob.glob(str(KBOUND / "kbound_short*.tex"))
    if not tex:
        pytest.skip("promoted manuscript sources not present in this checkout")
    for t in tex:
        hits = A.scan_text_for_forbidden(Path(t).read_text(errors="ignore"), LEDGER, only_withdrawn=True)
        assert hits == [], f"{t}: {hits}"


def test_detect_disagreements_aggregates_manuscript_and_consistency():
    manifest = _mini_manifest([c["claim_id"] for c in LEDGER["claims"]
                               if c["claim_type"] == "empirical" and c["status"] in A.PROMOTED_STATUSES])
    # A manuscript that (wrongly) resurrects the withdrawn Camelyon beats-both line.
    bad_manu = {"kbound_short.tex": "Camelyon17 beats both fixed policies (13x)."}
    problems = A.detect_disagreements(LEDGER, manifest=manifest, manuscript_texts=bad_manu)
    assert any("KB-CLAIM-022" in p for p in problems)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
