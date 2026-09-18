from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KBOUND_ROOT = ROOT / "docs/research/kbound"
ARCHIVE_RELATIVE = Path("docs/research/kbound/archive/legacy_publication_surfaces_2026-09-02")
ARCHIVE = ROOT / ARCHIVE_RELATIVE
MANIFEST = ARCHIVE / "MANIFEST.json"
BASE_COMMIT = "660d893caede49c3b7daa8c18e43bb6cbbce5480"
SOURCE_SEAL_BUILDER = KBOUND_ROOT / "scripts/build_release_source_seal.py"
ANONYMOUS_BUILDER = KBOUND_ROOT / "scripts/build_anonymous_supplement.py"

EXPECTED_BY_CATEGORY = {
    "book_narrative": (
        "docs/research/kbound/manuscript/README.md",
        "docs/research/kbound/manuscript/main.tex",
        "docs/research/kbound/manuscript/frontmatter.tex",
        "docs/research/kbound/manuscript/references.tex",
        "docs/research/kbound/manuscript/extended_theory_includes.tex",
        "docs/research/kbound/manuscript/chapters/appA_reproducibility.tex",
        "docs/research/kbound/manuscript/chapters/appB_tables.tex",
        "docs/research/kbound/manuscript/chapters/appC_notation.tex",
        "docs/research/kbound/manuscript/chapters/ch01_introduction.tex",
        "docs/research/kbound/manuscript/chapters/ch02_related.tex",
        "docs/research/kbound/manuscript/chapters/ch03_formulation.tex",
        "docs/research/kbound/manuscript/chapters/ch04_theory_impossibility.tex",
        "docs/research/kbound/manuscript/chapters/ch05_theory_certificates.tex",
        "docs/research/kbound/manuscript/chapters/ch06_kga.tex",
        "docs/research/kbound/manuscript/chapters/ch07_experiments_anomaly.tex",
        "docs/research/kbound/manuscript/chapters/ch08_experiments_tta.tex",
        "docs/research/kbound/manuscript/chapters/ch09_elara.tex",
        "docs/research/kbound/manuscript/chapters/ch10_discussion.tex",
    ),
    "historical_docs": (
        "GAP_AUDIT.md",
        "INTEGRITY_FIXES.md",
        "docs/research/kbound/ADOPTION_TRACK.md",
        "docs/research/kbound/ALL_ARTIFACTS_FOLDIN.md",
        "docs/research/kbound/ASSUMPTION_CONTRACT_CHANGESET.md",
        "docs/research/kbound/CODE_AUDIT_REPORT_2026-07-05.md",
        "docs/research/kbound/COMPARISON_FAMILY.md",
        "docs/research/kbound/EVIDENCE_MATRIX.md",
        "docs/research/kbound/FIGURES_GUIDE.md",
        "docs/research/kbound/FOLDIN_KICKOFF.md",
        "docs/research/kbound/KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md",
        "docs/research/kbound/KBOUND_EMPIRICAL_RECOVERY_AUDIT_2026-08-13.md",
        "docs/research/kbound/KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md",
        "docs/research/kbound/KBOUND_REMAINING_TODOS.md",
        "docs/research/kbound/KBOUND_RESULT_AUDIT.md",
        "docs/research/kbound/KBOUND_SHORT_DRAFT_CHANGELOG.md",
        "docs/research/kbound/KBOUND_SHORT_REMAINING_WORK.md",
        "docs/research/kbound/KBOUND_TABLE4_NATURAL_SHIFT_RECONCILIATION_2026-08-27.md",
        "docs/research/kbound/KBOUND_THEORY_AUDIT.md",
        "docs/research/kbound/PHASE2_THEOREM_AUDIT.md",
        "docs/research/kbound/PHASE7_INTEGRATION_AUDIT.md",
        "docs/research/kbound/PIPELINE_VS_PDF_AUDIT.md",
        "docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md",
        "docs/research/kbound/RELEASE_10X_TRACK.md",
        "docs/research/kbound/RELEASE_NOTES_v0.1.0.md",
        "docs/research/kbound/REPO_LEVEL80_PLAN.md",
        "docs/research/kbound/REPRO_HARDENING_REPORT.md",
        "docs/research/kbound/REVIEWER_RESPONSE_ASSUMPTIONS.md",
        "docs/research/kbound/REVIEW_SHORT_PAPER_2026-07-04.md",
        "docs/research/kbound/SUBMISSION_LEDGER.md",
        "docs/research/kbound/THEORY_100_PERCENT_CLOSURE_PLAN.md",
        "docs/research/kbound/THEORY_AUDIT_senior_review.md",
        "docs/research/kbound/THEORY_TO_CODE_MAP.md",
        "docs/research/kbound/VENUE_BENCHMARK_2026-06.md",
        "docs/research/kbound/formal/LEAN_COVERAGE_UPGRADE_PLAN.md",
        "docs/research/kbound/reports/CODE_AUDIT_UAV.md",
        "docs/research/kbound/reports/INTEGRATION_GAP_AUDIT.md",
        "docs/research/kbound/reports/LOCAL_TODO_CLOSURE_2026-07-08.md",
        "docs/research/kbound/reports/MONOREPO_ENGINEERING.md",
        "docs/research/kbound/reports/THEORY_AUDIT_FULL.md",
        "docs/research/kbound/reports/kbound_claim_consistency_audit.md",
        "docs/research/kbound/reports/reproducibility_release_report.md",
    ),
    "legacy_tex": (
        "docs/research/kbound/kbound.tex",
        "docs/research/kbound/kbound_short.tex",
        "docs/research/kbound/kbound_short_body.tex",
        "docs/research/kbound/kbound_short_appendix.tex",
        "docs/research/kbound/kbound_frontier_appendix.tex",
        "docs/research/kbound/kbound_abstract_disclosures.tex",
        "docs/research/kbound/paper/kbound.tex",
        "docs/research/kbound/paper/proofs_appendix.tex",
        "docs/research/kbound/paper/appendix_cifar10c_cells.tex",
        "docs/research/kbound/paper/references_kbound_expanded_full.tex",
        "docs/research/kbound/paper/_bibcheck.tex",
    ),
    "obsolete_scripts": (
        "docs/research/kbound/build_submission.py",
        "docs/research/kbound/scrub_submission.py",
        "docs/research/kbound/scripts/00_verify_environment.py",
        "docs/research/kbound/scripts/01_build_manifests.py",
        "docs/research/kbound/scripts/02_verify_results.py",
        "docs/research/kbound/scripts/03_make_tables.py",
        "docs/research/kbound/scripts/04_make_figures.py",
        "docs/research/kbound/scripts/99_reproduce_kbound.py",
        "docs/research/kbound/scripts/build_results_source.py",
        "docs/research/kbound/scripts/code_audit_uav.py",
        "docs/research/kbound/scripts/kbound_tour.py",
        "docs/research/kbound/scripts/kbound_tour.sh",
        "docs/research/kbound/scripts/make_appendix_frontier_figs.py",
        "docs/research/kbound/scripts/regime_map.py",
        "docs/research/kbound/scripts/reproduce_headlines.py",
        "docs/research/kbound/scripts/reproduce_submission.sh",
        "scripts/migrate_repo_name_to_kbound.sh",
    ),
    "panel_narrative": (
        "docs/research/kbound/panel_review_2026-07-25/FINAL_STATUS.md",
        "docs/research/kbound/panel_review_2026-07-25/FIXES_APPLIED.md",
        "docs/research/kbound/panel_review_2026-07-25/NUMBERS_PACK.md",
        "docs/research/kbound/panel_review_2026-07-25/RESTRUCTURE_COMPLETE.md",
        "docs/research/kbound/panel_review_2026-07-25/RESTRUCTURE_PLAN.md",
        "docs/research/kbound/panel_review_2026-07-25/changed_files.txt",
        "docs/research/kbound/panel_review_2026-07-25/kbound_fix_queue.html",
        "docs/research/kbound/panel_review_2026-07-25/report_docs.md",
        "docs/research/kbound/panel_review_2026-07-25/report_library.md",
        "docs/research/kbound/panel_review_2026-07-25/report_rewrite_short.md",
        "docs/research/kbound/panel_review_2026-07-25/report_scripts.md",
        "docs/research/kbound/panel_review_2026-07-25/review_1_mathematician.md",
        "docs/research/kbound/panel_review_2026-07-25/review_2_ai-engineer.md",
        "docs/research/kbound/panel_review_2026-07-25/review_3_scientist.md",
        "docs/research/kbound/panel_review_2026-07-25/review_4_repro-auditor.md",
        "docs/research/kbound/panel_review_2026-07-25/review_5_area-chair.md",
        "docs/research/kbound/panel_review_2026-07-25/review_6_overall.md",
    ),
    "stale_snapshots": (
        "docs/research/kbound/RELEASE_MANIFEST.json",
        "docs/research/kbound/REPRO_INVENTORY.json",
    ),
}

EXPECTED_PATHS = tuple(sorted(path for paths in EXPECTED_BY_CATEGORY.values() for path in paths))
WORKTREE_DIVERGED_FROM_BASE = {
    "docs/research/kbound/scripts/kbound_tour.py",
    "docs/research/kbound/scripts/reproduce_headlines.py",
    "docs/research/kbound/scripts/reproduce_submission.sh",
}

EXPECTED_ACTIVE_CLOSURE = (
    "docs/research/kbound/kbound_submission.tex",
    "docs/research/kbound/paper/generated/kbound_numbers.tex",
    "docs/research/kbound/paper/generated/cct20_numbers.tex",
    "docs/research/kbound/paper/generated/so2sat_numbers.tex",
    "docs/research/kbound/paper/figure_fallback.tex",
    "docs/research/kbound/kbound_abstract.tex",
    "docs/research/kbound/kbound_abstract_core.tex",
    "docs/research/kbound/kbound_submission_body.tex",
    "docs/research/kbound/paper/sections/theory_core_main.tex",
    "docs/research/kbound/paper/sections/theory_certificate.tex",
    "docs/research/kbound/paper/generated/kbound_primary_accuracy_table.tex",
    "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.tex",
    "docs/research/kbound/paper/generated/cct20_safe_utility_display.tex",
    "docs/research/kbound/figures/fig_decision_flow.pdf",
    "docs/research/kbound/figures/fig_frontier_schematic.pdf",
    "docs/research/kbound/figures/fig_certificate.pdf",
    "docs/research/kbound/paper/references_kbound_expanded.tex",
    "docs/research/kbound/kbound_submission_supplement.tex",
    "docs/research/kbound/paper/generated/kbound_auxiliary_accuracy_table.tex",
    "docs/research/kbound/paper/generated/kbound_auxiliary_balanced_accuracy_table.tex",
    "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex",
    "docs/research/kbound/paper/generated/current_policy_interval_diagnostics_groups.tex",
    "docs/research/kbound/paper/generated/cct20_primary_table_display.tex",
    "docs/research/kbound/paper/generated/cct20_location_effects_display.tex",
    "docs/research/kbound/kbound_tmlr.tex",
    "docs/research/kbound/paper/float_params.tex",
    "docs/research/kbound/paper/vendor/tmlr/tmlr.sty",
    "docs/research/kbound/kbound_full.tex",
    "docs/research/kbound/kbound_full_supplement.tex",
    "docs/research/kbound/kbound_short_main.tex",
    "docs/research/kbound/kbound_short_supplement.tex",
)


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_publication_archive_is_exact_and_byte_preserved() -> None:
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["schema_version"] == 1
    assert manifest["base_commit"] == BASE_COMMIT
    assert manifest["policy"] == "legacy_publication_surfaces_not_current_authority"
    assert manifest["record_count"] == 107
    assert manifest["total_bytes"] == 2_128_389
    assert manifest_text == json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    records = manifest["records"]
    assert [record["original_path"] for record in records] == list(EXPECTED_PATHS)
    assert len({record["original_path"] for record in records}) == len(records)
    assert sum(record["bytes"] for record in records) == manifest["total_bytes"]

    divergences = set()
    for record in records:
        original_path = record["original_path"]
        archive_path = (ARCHIVE_RELATIVE / "retired_tree" / original_path).as_posix()
        assert record["archive_path"] == archive_path
        assert record["category"] in EXPECTED_BY_CATEGORY
        assert original_path in EXPECTED_BY_CATEGORY[record["category"]]
        assert record["authority"] is False
        assert record["active_manuscript_source"] is False
        assert record["release_entry_point"] is False
        assert isinstance(record["replacement"], str) and record["replacement"]
        assert (ROOT / record["replacement"]).exists()
        assert not (ROOT / original_path).exists()

        archived = ROOT / archive_path
        payload = archived.read_bytes()
        assert record["bytes"] == len(payload)
        assert record["sha256"] == _sha256(payload)

        base_payload = _git_bytes(original_path)
        assert record["base_commit_bytes"] == len(base_payload)
        assert record["base_commit_sha256"] == _sha256(base_payload)
        diverged = record["sha256"] != record["base_commit_sha256"]
        assert record["working_tree_diverged_from_base"] is diverged
        if diverged:
            divergences.add(original_path)

    assert divergences == WORKTREE_DIVERGED_FROM_BASE

    archived_files = {
        path.relative_to(ROOT).as_posix() for path in (ARCHIVE / "retired_tree").rglob("*") if path.is_file()
    }
    assert archived_files == {record["archive_path"] for record in records}


def test_all_maintained_drivers_define_an_archive_free_tex_closure() -> None:
    sys.path.insert(0, str(KBOUND_ROOT))
    try:
        from kbound_repro.manuscript_sources import (
            ACTIVE_DRIVER_RELATIVE_PATHS,
            active_source_paths,
            live_latex,
        )
    finally:
        sys.path.pop(0)

    assert ACTIVE_DRIVER_RELATIVE_PATHS == (
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_full.tex",
        "docs/research/kbound/kbound_short_main.tex",
        "docs/research/kbound/kbound_short_supplement.tex",
    )
    for relative_path in ACTIVE_DRIVER_RELATIVE_PATHS:
        assert "\\documentclass" in live_latex((ROOT / relative_path).read_text(errors="ignore"))
    actual = tuple(path.relative_to(ROOT).as_posix() for path in active_source_paths(ROOT))
    assert actual == EXPECTED_ACTIVE_CLOSURE
    assert not set(actual).intersection(EXPECTED_PATHS)
    assert all("/archive/" not in path for path in actual)


def test_release_and_anonymous_source_inventories_exclude_the_legacy_archive() -> None:
    source_seal = _load_module("legacy_archive_source_seal", SOURCE_SEAL_BUILDER)
    anonymous = _load_module("legacy_archive_anonymous", ANONYMOUS_BUILDER)

    inventories = {
        "source seal": {path for _category, path in source_seal._inventory(ROOT, "HEAD")},
        "anonymous bundle": set(anonymous.anonymous_tmlr_inventory(ROOT)),
    }
    for name, paths in inventories.items():
        assert not set(EXPECTED_PATHS).intersection(paths), name
        assert all("/archive/" not in path for path in paths), name

    for paths in inventories.values():
        assert "docs/research/kbound/kbound_tmlr.tex" in paths
    assert "docs/research/kbound/kbound_submission.tex" in inventories["source seal"]


def test_unique_formal_protocol_and_raw_evidence_surfaces_remain_live() -> None:
    retained = (
        "docs/research/kbound/manuscript/theory_spine/theory_beta_estimable.tex",
        "docs/research/kbound/panel_review_2026-07-25/NUMBERS_PACK.json",
        "docs/research/kbound/panel_review_2026-07-25/recompute/11_build_numbers_pack.py",
        "docs/research/kbound/panel_review_2026-07-25/theory_beta_estimable.tex",
        "docs/research/kbound/panel_review_2026-07-25/THEORY_BETA_ESTIMABLE.md",
        "docs/research/kbound/runbooks/release_candidate.sh",
        "docs/research/kbound/scripts/build_anonymous_supplement.py",
        "docs/research/kbound/scripts/verify_release_toolchain.py",
        "research_lock/KBOUND_6_DATASET_PANEL_v2.yaml",
        "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
    )
    missing = [path for path in retained if not (ROOT / path).is_file()]
    assert not missing, f"retained authority/evidence unexpectedly missing: {missing}"
