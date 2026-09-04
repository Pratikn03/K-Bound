#!/usr/bin/env python3
"""Build or validate the final K-Bound release source/evidence seal.

The seal records a clean, explicitly supplied source commit and tree, then
hashes the maintained manuscript, release code, runtime packages, selected
validation sources, pinned formal sources, and immutable receipt-bound
authorities from that commit. Protected natural-shift material requires a
separate explicitly authorized seal. Generated canonical
manifests and built documents are instead bound by the final outer checksum
file; they may be dirty after the release gate, but every allowed path is
enumerated below. The seal, built documents, and outer checksum are deliberately
not members of this source inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

verify_python_environment = importlib.import_module("docs.research.kbound.scripts.verify_python_environment")

DEFAULT_OUTPUT = ROOT / "docs/research/kbound/audits/release_source_seal_2026_08_29.json"
SCHEMA = "kbound-release-source-seal-v1"

# The four role PDFs and their local checksum are the citation-facing release.
# The identity JSON and generated TeX are finalized only after the source commit
# is known, so they are outer-checksummed outputs rather than source-seal inputs.
POST_SOURCE_RELEASE_OUTPUT_ALLOWLIST = frozenset(
    {
        "docs/research/kbound/release/current/kbound_short_main.pdf",
        "docs/research/kbound/release/current/kbound_short_supplement.pdf",
        "docs/research/kbound/release/current/kbound_tmlr.pdf",
        "docs/research/kbound/release/current/kbound_full_report.pdf",
        "docs/research/kbound/release/current/KBOUND_CURRENT_SHA256SUMS.txt",
        "docs/research/kbound/paper/generated/current_release_identity.tex",
        "docs/research/kbound/paper/release/current_release.json",
        "docs/research/kbound/CURRENT_RELEASE.md",
        "docs/research/kbound/paper/reports/KBOUND_SOURCE_OUTPUT_MAP.md",
        "docs/research/kbound/paper/reports/TMLR_ANONYMITY_AUDIT.md",
        "docs/research/kbound/paper/reports/KBOUND_CURRENT_RELEASE_REPAIR_REPORT.md",
        "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound_short.pdf",
        "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound.pdf",
        "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound_submission.pdf",
        "docs/research/kbound/archive/superseded_do_not_cite/kbound_short_SUPERSEDED.pdf",
        "docs/research/kbound/archive/superseded_do_not_cite/kbound_SUPERSEDED.pdf",
        "docs/research/kbound/archive/superseded_do_not_cite/kbound_submission_SUPERSEDED.pdf",
    }
)

# These files are local build products or byte-identical convenience mirrors.
# They are deliberately permitted after the source freeze but never enter the
# authoritative outer checksum inventory. In particular, the combined PDF and
# DOCX are comparison artifacts, not a fifth release role.
NONRELEASE_BUILD_OUTPUT_ALLOWLIST = frozenset(
    {
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/kbound_short_main.pdf",
        "docs/research/kbound/kbound_short_supplement.pdf",
        "docs/research/kbound/kbound_full_report.pdf",
        "output/pdf/kbound_short_main.pdf",
        "output/pdf/kbound_short_supplement.pdf",
        "output/pdf/kbound_tmlr.pdf",
        "output/pdf/kbound_full_report.pdf",
        "output/pdf/KBOUND_CURRENT_SHA256SUMS.txt",
    }
)


def verify_release_python_content() -> None:
    """Require the sealed release interpreter before source-seal semantics."""

    verify_python_environment.verify_exact_content_profile(
        ROOT / "requirements-release-macos-arm64.lock.txt",
        ROOT / "docs/research/kbound/release_python_environment_macos_arm64.json",
    )


# These are the only working-tree changes permitted when the seal is emitted
# after the clean-HEAD release gate.  This is intentionally an exact-path list.
GENERATED_OUTPUT_ALLOWLIST = frozenset(
    {
        *POST_SOURCE_RELEASE_OUTPUT_ALLOWLIST,
        *NONRELEASE_BUILD_OUTPUT_ALLOWLIST,
        "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt",
        "docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt",
        "docs/research/kbound/RESULT_MANIFEST.json",
        "docs/research/kbound/STORAGE_MANIFEST.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/artifact.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/audit_summary.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv",
        "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
        "docs/research/kbound/audits/repository_test_inventory.json",
        "docs/research/kbound/audits/formal_foundations_2026_08_31.json",
        "docs/research/kbound/claim_ledger.json",
        "docs/research/kbound/dashboard/data/snapshot.json",
        "docs/research/kbound/figures/fig_decision_value_frontier.png",
        "docs/research/kbound/figures/fig_phase_diagram.png",
        "docs/research/kbound/release/cct20_public_evidence_bundle.zip",
        "docs/research/kbound/release/kbound_anonymous_supplement.zip",
        "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.json",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.tex",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics_groups.tex",
        "docs/research/kbound/paper/generated/kbound_primary_accuracy_table.tex",
        "docs/research/kbound/paper/generated/kbound_auxiliary_accuracy_table.tex",
        "docs/research/kbound/paper/generated/kbound_auxiliary_balanced_accuracy_table.tex",
        "docs/research/kbound/paper/generated/cct20_safe_utility_display.tex",
        "docs/research/kbound/paper/generated/cct20_release_manifest.json",
        "docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json",
        "docs/research/kbound/paper/generated/cct20_numbers.tex",
        "docs/research/kbound/paper/generated/cct20_primary_table.tex",
        "docs/research/kbound/paper/generated/cct20_location_effects.tex",
        "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json",
        "docs/research/kbound/paper/generated/kbound_numbers.tex",
        "docs/research/kbound/paper/generated/kbound_result_manifest.json",
        "docs/research/kbound/paper/generated/uniform_verdicts.json",
        "docs/research/kbound/results_source.json",
        "experiments/kbound/frontier_sweep_v1/decision_value_results.json",
        "experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json",
        "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json",
        "experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json",
        "experiments/kbound/results/reconciled_panels_v1/CANONICAL_PANEL_RESULTS.md",
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex",
        "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
        "experiments/kbound/results/reconciled_panels_v1/source_manifest.json",
    }
)

EXPLICIT_FILES: dict[str, tuple[str, ...]] = {
    "formal_source": (
        "docs/research/kbound/formal/KBound.lean",
        "docs/research/kbound/formal/README.md",
        "docs/research/kbound/formal/build.sh",
        "docs/research/kbound/formal/formal_audit.py",
        "docs/research/kbound/formal/lakefile.lean",
        "docs/research/kbound/formal/lake-manifest.json",
        "docs/research/kbound/formal/lean-toolchain",
    ),
    "release_validation": (
        "docs/research/kbound/edge/conftest.py",
        "tests/conftest.py",
        "tests/test_build_docx_pipeline.py",
        "tests/test_anonymous_supplement.py",
        "tests/test_cct20_manuscript_claim_validation.py",
        "tests/test_cct20_public_bundle.py",
        "tests/test_cct20_release_builder.py",
        "tests/test_certificate_drift_guard.py",
        "tests/test_exact_confirmation_pipeline.py",
        "tests/test_independent_checkpoint_audit.py",
        "tests/test_kbound_bibliography.py",
        "tests/test_kbound_current_policy_bindings.py",
        "tests/test_kbound_current_pdf_publication.py",
        "tests/test_kbound_dashboard_metadata.py",
        "tests/test_kbound_interval_diagnostics.py",
        "tests/test_kbound_metric_display_tables.py",
        "tests/test_kbound_narrative_revision.py",
        "tests/test_kbound_pdf_build_isolation.py",
        "tests/test_kbound_pdf_visual_verification.py",
        "tests/test_kbound_estimand_inference_wording.py",
        "tests/test_kbound_formal_audit.py",
        "tests/test_kbound_theory_scope.py",
        "tests/test_kga_api_routes.py",
        "tests/test_kga_benefit_estimator.py",
        "tests/test_kga_canonical_rule.py",
        "tests/test_kga_experiment_contract.py",
        "tests/test_kga_frontier_api.py",
        "tests/test_kga_masked_inputs.py",
        "tests/test_kga_package.py",
        "tests/test_kga_routing.py",
        "tests/test_kga_unavailable_api.py",
        "tests/test_kga_unavailable_runtime.py",
        "tests/test_manuscript_claim_consistency.py",
        "tests/test_natural_target_provenance.py",
        "tests/test_official_baseline_provenance.py",
        "tests/test_pacs_replay_artifact.py",
        "tests/test_reconcile_no_implicit_cleanup.py",
        "tests/test_reconciled_panels.py",
        "tests/test_release_checksum_verifier.py",
        "tests/test_release_privacy.py",
        "tests/test_release_source_seal.py",
        "tests/test_tmlr_anonymity_audit.py",
        "tests/test_repository_verification_runner.py",
    ),
    "paper_source": (
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_short_main.tex",
        "docs/research/kbound/kbound_short_supplement.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_full_report.tex",
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_submission_supplement.tex",
        "docs/research/kbound/kbound_abstract.tex",
        "docs/research/kbound/kbound_abstract_core.tex",
        "docs/research/kbound/kbound_full_report_extensions.tex",
        "docs/research/kbound/paper/figures/decision_flow.tex",
        "docs/research/kbound/paper/figure_fallback.tex",
        "docs/research/kbound/paper/float_params.tex",
        "docs/research/kbound/paper/full_report/evidence_protocol_atlas.tex",
        "docs/research/kbound/paper/full_report/formal_reproducibility_atlas.tex",
        "docs/research/kbound/paper/full_report/theory_exposition.tex",
        "docs/research/kbound/paper/generated/cct20_reporting_numbers.tex",
        "docs/research/kbound/paper/references_kbound_expanded.tex",
        "docs/research/kbound/paper/references_kbound_context_archive.tex",
        "docs/research/kbound/paper/references/refs.bib",
        "docs/research/kbound/paper/sections/theory_certificate.tex",
        "docs/research/kbound/paper/sections/theory_core_main.tex",
        "docs/research/kbound/paper/generated/cct20_primary_table_display.tex",
        "docs/research/kbound/paper/generated/cct20_location_effects_display.tex",
        "docs/research/kbound/paper/vendor/tmlr/LICENSE",
        "docs/research/kbound/paper/vendor/tmlr/README.md",
        "docs/research/kbound/paper/vendor/tmlr/tmlr.sty",
        "docs/research/kbound/figures/fig_certificate.png",
        "docs/research/kbound/figures/fig_decision_flow.png",
        "docs/research/kbound/figures/fig_frontier_schematic.png",
    ),
    "release_code": (
        "scripts/reconcile_result_panels.py",
        "scripts/sync_reconciled_panels.py",
        "src/scripts/validate_manuscript_claims.py",
        "docs/research/kbound/runbooks/release_candidate.sh",
        "docs/research/kbound/runbooks/convert_official_logs_to_decisions.py",
        "docs/research/kbound/runbooks/run_item11_official_baselines.sh",
        "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py",
        "docs/research/kbound/scripts/audit_current_kbound_release.py",
        "docs/research/kbound/scripts/audit_tmlr_anonymity.py",
        "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py",
        "docs/research/kbound/scripts/audit_natural_target_provenance.py",
        "docs/research/kbound/scripts/audit_official_baselines.py",
        "docs/research/kbound/scripts/build_cct20_release.py",
        "docs/research/kbound/scripts/build_cct20_public_bundle.py",
        "docs/research/kbound/scripts/build_anonymous_supplement.py",
        "docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py",
        "docs/research/kbound/scripts/compare_pdf_renders.py",
        "docs/research/kbound/scripts/empirical_closure.py",
        "docs/research/kbound/scripts/build_dashboard_snapshot.py",
        "docs/research/kbound/scripts/build_docx.py",
        "docs/research/kbound/scripts/build_empirical_data_quality_report_artifact.py",
        "docs/research/kbound/scripts/build_pdfs.sh",
        "docs/research/kbound/scripts/build_release_source_seal.py",
        "docs/research/kbound/scripts/build_result_manifest.py",
        "docs/research/kbound/scripts/build_results_source_compat.py",
        "docs/research/kbound/scripts/generate_release_identity.py",
        "docs/research/kbound/scripts/make_tables.py",
        "docs/research/kbound/scripts/official_baseline_provenance.py",
        "docs/research/kbound/scripts/official_baselines_headtohead.py",
        "docs/research/kbound/scripts/make_submission_figures.py",
        "docs/research/kbound/scripts/plot_canonical_decision_frontier.py",
        "docs/research/kbound/scripts/plot_conceptual_regime_geometry.py",
        "docs/research/kbound/scripts/plot_kga_interval_rule.py",
        "docs/research/kbound/scripts/publish_current_pdfs.py",
        "docs/research/kbound/scripts/refresh_storage_manifest.py",
        "docs/research/kbound/scripts/release_privacy.py",
        "docs/research/kbound/scripts/render_pdf_pages.py",
        "docs/research/kbound/scripts/run_official_native.py",
        "docs/research/kbound/scripts/run_repository_verification.py",
        "docs/research/kbound/scripts/run_frontier_kga_bridge.py",
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
        "docs/research/kbound/scripts/verify_release_checksums.py",
        "docs/research/kbound/scripts/verify_pdf_structure.py",
        "docs/research/kbound/scripts/verify_python_environment.py",
        "docs/research/kbound/scripts/verify_release_toolchain.py",
    ),
    "deployment_api": (
        "deploy/api/main.py",
        "deploy/api/auth.py",
        "deploy/api/envutil.py",
        "deploy/api/rate_limit.py",
        "deploy/api/model_governance.py",
        "deploy/api/monitoring.py",
        "deploy/api/scope_guard.py",
        "deploy/api/kga_routes.py",
        "deploy/api/kga_service.py",
    ),
    "configuration": (
        ".dockerignore",
        ".pre-commit-config.yaml",
        ".python-version",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "docs/research/kbound/README.md",
        "docs/research/kbound/DOCS_INDEX.md",
        "docs/research/kbound/KBOUND_SHORT_CLAIM_MANIFEST.md",
        "docs/research/kbound/KBOUND_SHORT_RESULT_AUDIT.md",
        "docs/research/kbound/CIFAR10C_SAR_QUARANTINE.md",
        "docs/research/kbound/G8_EXACTRANK_REGEN.md",
        "docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md",
        "docs/research/kbound/RELATED_WORK_POSITIONING.md",
        "docs/research/kbound/REPRODUCE.md",
        "docs/research/kbound/RELEASE_CHECKLIST.md",
        "docs/research/kbound/paper/generated/empirical_audit/claim_matrix.md",
        "docs/research/kbound/paper/RELEASE_TABLE_CROSSWALK.md",
        "docs/research/kbound/kbound_pkg/README.md",
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements-api.txt",
        "requirements-api-py311-linux.lock.txt",
        "requirements-ci-py312-linux.lock.txt",
        "requirements-ci.txt",
        "requirements-dev.txt",
        "requirements-eyecandies-legacy.txt",
        "requirements-optional.txt",
        "requirements.txt",
        "requirements.lock.txt",
        "requirements-research.txt",
        "requirements-research-ci-overrides.txt",
        "requirements-research-ci.lock.txt",
        "requirements-release.txt",
        "requirements-release-macos-arm64.lock.txt",
        "docs/research/kbound/release_python_environment_macos_arm64.json",
        "docs/research/kbound/release_toolchain_macos_arm64.json",
        "requirements-paper.txt",
        "requirements-paper.lock.txt",
        "docs/research/kbound/kbound_pkg/pyproject.toml",
        "docs/research/kbound/kbound_pkg/LICENSE",
        "docs/research/kbound/dashboard/package.json",
        "docs/research/kbound/dashboard/package-lock.json",
        "docs/research/kbound/dashboard/tsconfig.json",
        "docs/research/multiclass_vector_capacity/formal/lake-manifest.json",
        "docs/research/multiclass_vector_capacity/formal/lean-toolchain",
        ".github/workflows/ci.yml",
        ".github/workflows/kbound-ci.yml",
    ),
    "verified_release_environment": (
        "docs/research/kbound/audits/python_environment_2026_09_02.json",
        "docs/research/kbound/audits/release_toolchain_2026_09_02.json",
    ),
    "immutable_release_locks": (
        "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml",
        "research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json",
    ),
}


# These are audited source-only subtrees, selected from the pinned Git tree.
# Never walk datasets, historical experiment results, Lake caches, distributions,
# or generated reports to construct the source inventory.
SOURCE_PREFIX_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("formal_source", "docs/research/kbound/formal/KBound/", (".lean",)),
    ("formal_source", "docs/research/multiclass_vector_capacity/formal/", (".lean", ".py")),
    ("repro_package", "docs/research/kbound/kbound_repro/", (".py",)),
    ("packaged_kbound", "docs/research/kbound/kbound_pkg/kbound/", (".py",)),
    ("assumption_audit_package", "docs/research/kbound/kbound_pkg/assumption_audit/", (".py",)),
    ("edge_runtime", "docs/research/kbound/edge/src/kbound_edge/", (".py",)),
    ("release_validation", "docs/research/kbound/kbound_pkg/tests/", (".py",)),
    ("release_validation", "docs/research/kbound/tests/", (".py",)),
    ("release_validation", "docs/research/kbound/edge/tests/", (".py",)),
    ("release_validation", "tests/", (".py",)),
    ("release_validation", "experiments/kbound/theory_validation/", (".py",)),
    ("release_validation", "docs/research/kbound/theory_v2/", (".py",)),
    ("release_validation", "docs/research/kbound/audit_validation/", (".py",)),
    ("release_validation", "docs/research/kbound/ttc_extension/", (".py",)),
    ("dashboard_source", "docs/research/kbound/dashboard/src/", (".ts",)),
    ("dashboard_source", "docs/research/kbound/dashboard/js/", (".js", ".map")),
    ("dashboard_source", "docs/research/kbound/dashboard/css/", (".css",)),
)
EXCLUDED_SOURCE_PARTS = frozenset({".lake", "build", "dist", "__pycache__"})
DEFAULT_PROTECTED_COMPONENT = "so2sat"
DEFAULT_PROTECTED_GIT_EXCLUDES = (
    ":(exclude,icase,glob)**/*so2sat*",
    ":(exclude,icase,glob)**/*so2sat*/**",
)


def _git(*args: str, repo: Path = ROOT) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _git_bytes(*args: str, repo: Path = ROOT) -> bytes:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return completed.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seal_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_json_loads(payload: bytes) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def _validate_relative_path(value: str) -> str:
    parsed = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or value.startswith(("/", "~"))
        or parsed.is_absolute()
        or ".." in parsed.parts
        or str(parsed) != value
    ):
        raise ValueError(f"unsafe release source seal path: {value!r}")
    return value


def _full_hex(value: object, *, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        if field == "source_commit":
            raise ValueError("sealed source_commit must be a full immutable commit ID")
        raise ValueError(f"release source seal {field} is not a full lowercase hex ID")
    return value


def validate_seal_document(payload: object) -> dict[str, object]:
    """Validate all source-seal invariants that do not require a Git checkout."""

    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported release-source-seal schema")
    _full_hex(payload.get("source_commit"), field="source_commit", length=40)
    _full_hex(payload.get("source_tree"), field="source_tree", length=40)
    rows = payload.get("artifacts")
    if not isinstance(rows, list):
        raise ValueError("release source seal artifacts must be a list")
    paths: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"release source seal artifact {index} is not an object")
        path = _validate_relative_path(str(row.get("path", "")))
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise ValueError(f"release source seal artifact {index} lacks a category")
        _full_hex(row.get("git_blob"), field=f"artifacts[{index}].git_blob", length=40)
        _full_hex(row.get("sha256"), field=f"artifacts[{index}].sha256", length=64)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"release source seal artifact {index} has invalid bytes")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("release source seal artifact paths are duplicated or non-canonical")
    digest_input = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if payload.get("artifacts_sha256") != _sha256(digest_input):
        raise ValueError("release source/evidence aggregate hash is invalid")
    if payload.get("sealed_artifact_count") != len(rows):
        raise ValueError("release source/evidence artifact count is invalid")
    if not isinstance(payload.get("exclusions"), list) or not all(
        isinstance(item, str) for item in payload["exclusions"]
    ):
        raise ValueError("release source seal exclusions are malformed")
    return payload


def _git_blob_oid(data: bytes, object_format: str) -> str:
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _dirty_paths(repo: Path) -> set[str]:
    raw = _git_bytes(
        "--no-optional-locks",
        "-c",
        "core.preloadindex=false",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.quotePath=false",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        ".",
        *DEFAULT_PROTECTED_GIT_EXCLUDES,
        repo=repo,
    )
    paths: set[str] = set()
    records = raw.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError(f"unparseable git status record: {record!r}")
        status = record[:2].decode("ascii")
        paths.add(record[3:].decode("utf-8"))
        if "R" in status or "C" in status:
            if index >= len(records) or not records[index]:
                raise ValueError("rename/copy status record is missing its source path")
            paths.add(records[index].decode("utf-8"))
            index += 1
    return paths


def _source_tree_pathspecs() -> tuple[str, ...]:
    """Return positive-only Git tree scopes for the portable source seal.

    ``git ls-tree`` does not implement exclusion pathspec magic.  A repository-
    wide listing followed by a Python filter would still enumerate protected
    natural-shift artifact names.  Positive source roots avoid that boundary.
    """

    paths = {
        path
        for values in EXPLICIT_FILES.values()
        for path in values
        if DEFAULT_PROTECTED_COMPONENT not in path.casefold()
    }
    paths.update(
        prefix
        for _category, prefix, _suffixes in SOURCE_PREFIX_RULES
        if DEFAULT_PROTECTED_COMPONENT not in prefix.casefold()
    )
    paths.add("kga/")
    return tuple(sorted(paths))


def _tree_blobs(
    repo: Path,
    commit: str,
    *,
    pathspecs: tuple[str, ...] | None = None,
) -> dict[str, str]:
    # Do not request ``ls-tree -l`` sizes here. Git may need to read every blob
    # to obtain them, which can hydrate unrelated macOS/iCloud dataless objects.
    # Tree entries already contain every path and blob object ID needed to select
    # the maintained inventory; byte counts are derived only for selected blobs
    # after their contents are read below.
    selected = _source_tree_pathspecs() if pathspecs is None else pathspecs
    if not selected or any(path in {"", "."} for path in selected):
        raise ValueError("release source tree requires non-root positive pathspecs")
    raw = _git_bytes("ls-tree", "-r", "-z", commit, "--", *selected, repo=repo)
    blobs: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        _mode, kind, oid = metadata.decode("ascii").split()
        if kind != "blob":
            continue
        path = encoded_path.decode("utf-8")
        blobs[path] = oid
    return blobs


def _inventory(repo: Path, commit: str) -> list[tuple[str, str]]:
    rows = [(category, path) for category, paths in EXPLICIT_FILES.items() for path in paths]
    tracked = _tree_blobs(repo, commit)
    for path in sorted(tracked):
        # Natural-shift gate/target material is intentionally outside the
        # portable default release authority.  Do not even select similarly
        # named source/test inputs until an explicitly authorized workflow is
        # introduced and reviewed separately.
        if DEFAULT_PROTECTED_COMPONENT in path.casefold():
            continue
        if EXCLUDED_SOURCE_PARTS.intersection(Path(path).parts):
            continue
        if path.startswith("kga/") and "__pycache__" not in path and not path.endswith(".pyc"):
            rows.append(("root_kga_package", path))
        for category, prefix, suffixes in SOURCE_PREFIX_RULES:
            if path.startswith(prefix) and path.endswith(suffixes):
                rows.append((category, path))
    seen: dict[str, str] = {}
    unique: list[tuple[str, str]] = []
    for category, path in sorted(rows, key=lambda item: (item[1], item[0])):
        previous = seen.get(path)
        if previous is not None and previous != category:
            raise ValueError(f"release seal path appears in multiple categories: {path}")
        if previous is not None:
            continue
        seen[path] = category
        unique.append((category, path))
    return unique


def _artifact_rows(repo: Path, commit: str) -> list[dict[str, object]]:
    blobs = _tree_blobs(repo, commit)
    object_format = _git("rev-parse", "--show-object-format", repo=repo)
    rows: list[dict[str, object]] = []
    for category, relative in _inventory(repo, commit):
        if relative not in blobs:
            raise FileNotFoundError(f"required release-seal input is not tracked at {commit}: {relative}")
        blob_oid = blobs[relative]
        current = repo / relative
        parts = Path(relative).parts
        linked = any(repo.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1))
        if linked or not current.is_file():
            raise FileNotFoundError(f"sealed maintained path is missing or a symlink: {relative}")
        data = current.read_bytes()
        if _git_blob_oid(data, object_format) != blob_oid:
            raise ValueError(f"checked-out bytes do not match source commit blob: {relative}")
        blob_bytes = len(data)
        rows.append(
            {
                "path": relative,
                "category": category,
                "git_blob": blob_oid,
                "bytes": blob_bytes,
                "sha256": _sha256(data),
            }
        )
    return rows


def build_payload(repo: Path, source_commit: str, *, require_clean: bool = False) -> dict[str, object]:
    resolved = _git("rev-parse", "--verify", f"{source_commit}^{{commit}}", repo=repo)
    head = _git("rev-parse", "--verify", "HEAD", repo=repo)
    if resolved != head:
        raise ValueError(f"source commit must equal HEAD: source={resolved}, HEAD={head}")
    dirty = _dirty_paths(repo)
    if require_clean and dirty:
        raise ValueError("release must start from a completely clean working tree: " + ", ".join(sorted(dirty)))
    maintained = {path for _, path in _inventory(repo, resolved)}
    dirty_maintained = sorted(dirty & maintained)
    if dirty_maintained:
        raise ValueError("maintained release-source paths are dirty: " + ", ".join(dirty_maintained))
    unexpected = sorted(dirty - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError("working tree is dirty outside the generated-output allowlist: " + ", ".join(unexpected))
    source_tree = _git("rev-parse", "--verify", f"{resolved}^{{tree}}", repo=repo)
    artifacts = _artifact_rows(repo, resolved)
    if _git("rev-parse", "--verify", "HEAD", repo=repo) != resolved:
        raise ValueError("HEAD changed during the release source check")
    digest_input = json.dumps(artifacts, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": SCHEMA,
        "source_commit": resolved,
        "source_tree": source_tree,
        "working_tree_gate": "clean outside exact generated-output allowlist",
        "sealed_artifact_count": len(artifacts),
        "artifacts_sha256": _sha256(digest_input),
        "artifacts": artifacts,
        "exclusions": [
            "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
            "release-generated canonical manifests and presentation assets (bound by the outer checksum file)",
            "four current role PDFs, their local checksum, and finalized release identity (bound by the outer checksum file)",
            "nonrelease build PDFs, combined DOCX, and convenience mirrors (excluded from the citation-facing checksum inventory)",
            "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt",
        ],
    }


def _validate_artifact_head(repo: Path, source_commit: str) -> None:
    """Permit source S or a descendant that commits generated artifacts only.

    The source seal continues to name S after the reviewed artifact commit R.
    A different source branch or committed net source changes are not that release.
    ``--no-renames`` avoids reading unrelated file contents for rename detection.
    """
    head = _git("rev-parse", "--verify", "HEAD", repo=repo)
    if head == source_commit:
        return
    try:
        _git("merge-base", "--is-ancestor", source_commit, head, repo=repo)
    except subprocess.CalledProcessError as exc:
        raise ValueError("sealed source commit is not a verified ancestor of HEAD") from exc
    raw = _git_bytes(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        "--no-renames",
        source_commit,
        head,
        "--",
        ".",
        *DEFAULT_PROTECTED_GIT_EXCLUDES,
        repo=repo,
    )
    changed = {path.decode("utf-8") for path in raw.split(b"\0") if path}
    unexpected = sorted(changed - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "committed changes since the sealed source are not generated outputs: " + ", ".join(unexpected)
        )


def _reject_symlink_path(path: Path, *, role: str) -> None:
    parts = path.absolute().parts
    if any(Path(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise ValueError(f"refusing to use {role} through a symlink: {path}")


def validate_seal(repo: Path, path: Path) -> dict[str, object]:
    payload = validate_seal_structure(path)
    commit = str(payload.get("source_commit", ""))
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}", repo=repo)
    if commit != resolved:
        raise ValueError("sealed source_commit must be a full immutable commit ID")
    _validate_artifact_head(repo, commit)
    expected_tree = _git("rev-parse", "--verify", f"{commit}^{{tree}}", repo=repo)
    if payload.get("source_tree") != expected_tree:
        raise ValueError("release source tree does not match the sealed commit")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or rows != _artifact_rows(repo, commit):
        raise ValueError("release source/evidence artifact inventory or hashes have drifted")
    dirty = _dirty_paths(repo)
    maintained = {str(row["path"]) for row in rows}
    dirty_maintained = sorted(dirty & maintained)
    if dirty_maintained:
        raise ValueError("maintained release-source paths are dirty: " + ", ".join(dirty_maintained))
    unexpected = sorted(dirty - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError("working tree is dirty outside the generated-output allowlist: " + ", ".join(unexpected))
    for row in rows:
        relative = str(row["path"])
        current = repo / relative
        if not current.is_file():
            raise FileNotFoundError(f"sealed maintained path is missing: {relative}")
        data = current.read_bytes()
        if len(data) != row["bytes"] or _sha256(data) != row["sha256"]:
            raise ValueError(f"sealed maintained path differs from source commit: {relative}")
    return payload


def validate_seal_structure(path: Path) -> dict[str, object]:
    """Validate canonical seal bytes without inspecting Git or the build runtime."""

    _reject_symlink_path(path, role="release source seal")
    raw = path.read_bytes()
    payload = _strict_json_loads(raw)
    payload = validate_seal_document(payload)
    if raw != _seal_bytes(payload):
        raise ValueError("release source seal is not canonical JSON")
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    validate_seal_document(payload)
    _reject_symlink_path(path, role="release source seal output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_seal_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", help="clean source commit; required when writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate an existing seal")
    mode.add_argument(
        "--check-portable",
        action="store_true",
        help="validate an existing seal against this checkout without enforcing the build runtime",
    )
    mode.add_argument(
        "--check-structure",
        action="store_true",
        help="validate only canonical source-seal structure; do not inspect Git or the build runtime",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="require a completely clean pinned HEAD before the release starts; write nothing",
    )
    mode.add_argument(
        "--check-source",
        action="store_true",
        help="require the same pinned HEAD and unchanged sources between release phases; write nothing",
    )
    args = parser.parse_args()
    if not (args.check_portable or args.check_structure):
        verify_release_python_content()
    # Preserve the caller's path components so the writer/validator can detect
    # a symlink instead of resolving it away before the safety check.
    output = args.output.absolute()
    if args.check_structure:
        payload = validate_seal_structure(output)
        print(f"release source seal structure: PASS ({payload['sealed_artifact_count']} files)")
        return 0
    if args.preflight or args.check_source:
        if not args.source_commit:
            parser.error("--source-commit is required for release source checks")
        payload = build_payload(ROOT, args.source_commit, require_clean=args.preflight)
        stage = "clean start" if args.preflight else "pinned sources"
        print(f"release source checkout: PASS ({stage}; {payload['sealed_artifact_count']} files)")
        return 0
    if args.check or args.check_portable:
        payload = validate_seal(ROOT, output)
        if args.source_commit:
            expected = _git("rev-parse", "--verify", f"{args.source_commit}^{{commit}}", repo=ROOT)
            if payload["source_commit"] != expected:
                raise ValueError("existing seal references a different source commit")
        qualifier = "portable checkout-bound" if args.check_portable else "release-runtime-bound"
        print(f"release source seal: PASS ({qualifier}; {payload['sealed_artifact_count']} files)")
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required when writing the seal")
    payload = build_payload(ROOT, args.source_commit)
    _write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
