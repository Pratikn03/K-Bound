#!/usr/bin/env python3
"""Build or validate the final K-Bound release source/evidence seal.

The seal records a clean, explicitly supplied source commit and tree, then
hashes the maintained manuscript, release code, runtime packages, selected
validation sources, pinned formal sources, So2Sat protocol code, and immutable
receipt-bound authorities from that commit. Generated canonical
manifests and built documents are instead bound by the final outer checksum
file; they may be dirty after the release gate, but every allowed path is
enumerated below. The seal, built documents, and outer checksum are deliberately
not members of this source inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "docs/research/kbound/audits/release_source_seal_2026_08_29.json"
SCHEMA = "kbound-release-source-seal-v1"

# These are the only working-tree changes permitted when the seal is emitted
# after the clean-HEAD release gate.  This is intentionally an exact-path list.
GENERATED_OUTPUT_ALLOWLIST = frozenset(
    {
        "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt",
        "docs/research/kbound/RESULT_MANIFEST.json",
        "docs/research/kbound/STORAGE_MANIFEST.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/artifact.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/audit_summary.json",
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv",
        "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
        "docs/research/kbound/audits/formal_foundations_2026_08_31.json",
        "docs/research/kbound/claim_ledger.json",
        "docs/research/kbound/dashboard/data/snapshot.json",
        "docs/research/kbound/figures/fig_decision_value_frontier.png",
        "docs/research/kbound/figures/fig_phase_diagram.png",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.json",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.tex",
        "docs/research/kbound/paper/generated/current_policy_interval_diagnostics_groups.tex",
        "docs/research/kbound/paper/generated/kbound_primary_accuracy_table.tex",
        "docs/research/kbound/paper/generated/kbound_auxiliary_accuracy_table.tex",
        "docs/research/kbound/paper/generated/kbound_auxiliary_balanced_accuracy_table.tex",
        "docs/research/kbound/paper/generated/cct20_safe_utility_display.tex",
        "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json",
        "docs/research/kbound/paper/generated/kbound_numbers.tex",
        "docs/research/kbound/paper/generated/kbound_result_manifest.json",
        "docs/research/kbound/paper/generated/so2sat_numbers.tex",
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
        "tests/test_cct20_manuscript_claim_validation.py",
        "tests/test_cct20_release_builder.py",
        "tests/test_certificate_drift_guard.py",
        "tests/test_exact_confirmation_pipeline.py",
        "tests/test_independent_checkpoint_audit.py",
        "tests/test_kbound_bibliography.py",
        "tests/test_kbound_current_policy_bindings.py",
        "tests/test_kbound_dashboard_metadata.py",
        "tests/test_kbound_interval_diagnostics.py",
        "tests/test_kbound_metric_display_tables.py",
        "tests/test_kbound_narrative_revision.py",
        "tests/test_kbound_pdf_build_isolation.py",
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
        "tests/test_release_source_seal.py",
        "tests/test_so2sat_numbers_builder.py",
        "tests/test_so2sat_prospective_protocol.py",
        "tests/test_so2sat_target_boundary.py",
    ),
    "paper_source": (
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_submission_supplement.tex",
        "docs/research/kbound/kbound_abstract.tex",
        "docs/research/kbound/kbound_abstract_core.tex",
        "docs/research/kbound/kbound_abstract_disclosures.tex",
        "docs/research/kbound/paper/figure_fallback.tex",
        "docs/research/kbound/paper/float_params.tex",
        "docs/research/kbound/paper/references_kbound_expanded.tex",
        "docs/research/kbound/paper/references_kbound_context_archive.tex",
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
        "docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py",
        "docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py",
        "docs/research/kbound/scripts/audit_natural_target_provenance.py",
        "docs/research/kbound/scripts/audit_official_baselines.py",
        "docs/research/kbound/scripts/build_cct20_release.py",
        "docs/research/kbound/scripts/build_current_policy_interval_diagnostics.py",
        "docs/research/kbound/scripts/build_dashboard_snapshot.py",
        "docs/research/kbound/scripts/build_docx.py",
        "docs/research/kbound/scripts/build_empirical_data_quality_report_artifact.py",
        "docs/research/kbound/scripts/build_pdfs.sh",
        "docs/research/kbound/scripts/build_release_source_seal.py",
        "docs/research/kbound/scripts/build_result_manifest.py",
        "docs/research/kbound/scripts/build_results_source_compat.py",
        "docs/research/kbound/scripts/build_so2sat_numbers.py",
        "docs/research/kbound/scripts/make_tables.py",
        "docs/research/kbound/scripts/make_submission_figures.py",
        "docs/research/kbound/scripts/plot_canonical_decision_frontier.py",
        "docs/research/kbound/scripts/plot_conceptual_regime_geometry.py",
        "docs/research/kbound/scripts/plot_kga_interval_rule.py",
        "docs/research/kbound/scripts/refresh_storage_manifest.py",
        "docs/research/kbound/scripts/render_pdf_pages.py",
        "docs/research/kbound/scripts/run_frontier_kga_bridge.py",
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
        "docs/research/kbound/scripts/verify_release_checksums.py",
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
        "README.md",
        "docs/research/kbound/README.md",
        "docs/research/kbound/DOCS_INDEX.md",
        "docs/research/kbound/kbound_pkg/README.md",
        "CITATION.cff",
        "LICENSE",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements-api.txt",
        "requirements-dev.txt",
        "requirements-eyecandies-legacy.txt",
        "requirements-optional.txt",
        "requirements.txt",
        "requirements.lock.txt",
        "requirements-research.txt",
        "requirements-research-ci-overrides.txt",
        "requirements-research-ci.lock.txt",
        "requirements-paper.txt",
        "requirements-paper.lock.txt",
        "docs/research/kbound/kbound_pkg/pyproject.toml",
        "docs/research/kbound/kbound_pkg/LICENSE",
        ".github/workflows/ci.yml",
        ".github/workflows/kbound-ci.yml",
    ),
    "immutable_release_locks": (
        "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml",
        "research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json",
    ),
    "separate_natural_shift_authorities": (
        "docs/research/kbound/paper/generated/cct20_release_manifest.json",
        "docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json.receipt.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json",
        "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json",
        "experiments/kbound/so2sat/prospective_protocol_v1.json",
        "experiments/kbound/so2sat/prospective_protocol_v1.json.receipt.json",
        "experiments/kbound/so2sat/target_boundary_amendment_v1_1.json",
        "experiments/kbound/so2sat/target_boundary_amendment_v1_1.json.receipt.json",
        "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml",
        "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml.sha256",
        "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json",
        "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json.receipt.json",
    ),
}


# These are audited source-only subtrees, selected from the pinned Git tree.
# Never walk datasets, historical experiment results, Lake caches, distributions,
# or generated reports to construct the source inventory.
SOURCE_PREFIX_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("formal_source", "docs/research/kbound/formal/KBound/", (".lean",)),
    ("repro_package", "docs/research/kbound/kbound_repro/", (".py",)),
    ("packaged_kbound", "docs/research/kbound/kbound_pkg/kbound/", (".py",)),
    ("assumption_audit_package", "docs/research/kbound/kbound_pkg/assumption_audit/", (".py",)),
    ("edge_runtime", "docs/research/kbound/edge/src/kbound_edge/", (".py",)),
    ("release_validation", "docs/research/kbound/kbound_pkg/tests/", (".py",)),
    ("release_validation", "docs/research/kbound/tests/", (".py",)),
    ("release_validation", "docs/research/kbound/edge/tests/", (".py",)),
)
EXCLUDED_SOURCE_PARTS = frozenset({".lake", "build", "dist", "__pycache__"})


def _git(*args: str, repo: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_bytes(*args: str, repo: Path = ROOT) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    )
    return completed.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _tree_blobs(repo: Path, commit: str) -> dict[str, str]:
    # Do not request ``ls-tree -l`` sizes here. Git may need to read every blob
    # to obtain them, which can hydrate unrelated macOS/iCloud dataless objects.
    # Tree entries already contain every path and blob object ID needed to select
    # the maintained inventory; byte counts are derived only for selected blobs
    # after their contents are read below.
    raw = _git_bytes("ls-tree", "-r", "-z", commit, repo=repo)
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
        if EXCLUDED_SOURCE_PARTS.intersection(Path(path).parts):
            continue
        if path.startswith("kga/") and "__pycache__" not in path and not path.endswith(".pyc"):
            rows.append(("root_kga_package", path))
        if path.startswith("experiments/kbound/so2sat/") and path.endswith(".py"):
            rows.append(("so2sat_code", path))
        for category, prefix, suffixes in SOURCE_PREFIX_RULES:
            if path.startswith(prefix) and path.endswith(suffixes):
                rows.append((category, path))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for category, path in sorted(rows, key=lambda item: (item[1], item[0])):
        if path in seen:
            raise ValueError(f"release seal path appears in multiple categories: {path}")
        seen.add(path)
        unique.append((category, path))
    return unique


def _artifact_rows(repo: Path, commit: str) -> list[dict[str, object]]:
    blobs = _tree_blobs(repo, commit)
    object_format = _git("rev-parse", "--show-object-format", repo=repo)
    rows: list[dict[str, object]] = []
    for category, relative in _inventory(repo, commit):
        if relative not in blobs:
            raise FileNotFoundError(
                f"required release-seal input is not tracked at {commit}: {relative}"
            )
        blob_oid = blobs[relative]
        current = repo / relative
        parts = Path(relative).parts
        linked = any(
            repo.joinpath(*parts[:index]).is_symlink()
            for index in range(1, len(parts) + 1)
        )
        if linked or not current.is_file():
            raise FileNotFoundError(f"sealed maintained path is missing or a symlink: {relative}")
        data = current.read_bytes()
        if _git_blob_oid(data, object_format) != blob_oid:
            raise ValueError(
                f"checked-out bytes do not match source commit blob: {relative}"
            )
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


def build_payload(
    repo: Path, source_commit: str, *, require_clean: bool = False
) -> dict[str, object]:
    resolved = _git("rev-parse", "--verify", f"{source_commit}^{{commit}}", repo=repo)
    head = _git("rev-parse", "--verify", "HEAD", repo=repo)
    if resolved != head:
        raise ValueError(f"source commit must equal HEAD: source={resolved}, HEAD={head}")
    dirty = _dirty_paths(repo)
    if require_clean and dirty:
        raise ValueError(
            "release must start from a completely clean working tree: " + ", ".join(sorted(dirty))
        )
    maintained = {path for _, path in _inventory(repo, resolved)}
    dirty_maintained = sorted(dirty & maintained)
    if dirty_maintained:
        raise ValueError(
            "maintained release-source paths are dirty: " + ", ".join(dirty_maintained)
        )
    unexpected = sorted(dirty - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "working tree is dirty outside the generated-output allowlist: "
            + ", ".join(unexpected)
        )
    source_tree = _git("rev-parse", "--verify", f"{resolved}^{{tree}}", repo=repo)
    artifacts = _artifact_rows(repo, resolved)
    if _git("rev-parse", "--verify", "HEAD", repo=repo) != resolved:
        raise ValueError("HEAD changed during the release source check")
    digest_input = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
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
            "built PDF and DOCX files (bound by the outer checksum file)",
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
        "diff-tree", "--no-commit-id", "--name-only", "-r", "-z", "--no-renames",
        source_commit, head, repo=repo,
    )
    changed = {path.decode("utf-8") for path in raw.split(b"\0") if path}
    unexpected = sorted(changed - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "committed changes since the sealed source are not generated outputs: "
            + ", ".join(unexpected)
        )


def validate_seal(repo: Path, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported release-source-seal schema")
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
        raise ValueError(
            "maintained release-source paths are dirty: " + ", ".join(dirty_maintained)
        )
    unexpected = sorted(dirty - GENERATED_OUTPUT_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "working tree is dirty outside the generated-output allowlist: "
            + ", ".join(unexpected)
        )
    for row in rows:
        relative = str(row["path"])
        current = repo / relative
        if not current.is_file():
            raise FileNotFoundError(f"sealed maintained path is missing: {relative}")
        data = current.read_bytes()
        if len(data) != row["bytes"] or _sha256(data) != row["sha256"]:
            raise ValueError(f"sealed maintained path differs from source commit: {relative}")
    digest_input = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if payload.get("artifacts_sha256") != _sha256(digest_input):
        raise ValueError("release source/evidence aggregate hash is invalid")
    if payload.get("sealed_artifact_count") != len(rows):
        raise ValueError("release source/evidence artifact count is invalid")
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", help="clean source commit; required when writing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate an existing seal")
    mode.add_argument(
        "--preflight", action="store_true",
        help="require a completely clean pinned HEAD before the release starts; write nothing",
    )
    mode.add_argument(
        "--check-source", action="store_true",
        help="require the same pinned HEAD and unchanged sources between release phases; write nothing",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.preflight or args.check_source:
        if not args.source_commit:
            parser.error("--source-commit is required for release source checks")
        payload = build_payload(ROOT, args.source_commit, require_clean=args.preflight)
        stage = "clean start" if args.preflight else "pinned sources"
        print(f"release source checkout: PASS ({stage}; {payload['sealed_artifact_count']} files)")
        return 0
    if args.check:
        payload = validate_seal(ROOT, output)
        if args.source_commit:
            expected = _git(
                "rev-parse", "--verify", f"{args.source_commit}^{{commit}}", repo=ROOT
            )
            if payload["source_commit"] != expected:
                raise ValueError("existing seal references a different source commit")
        print(f"release source seal: PASS ({payload['sealed_artifact_count']} files)")
        return 0
    if not args.source_commit:
        parser.error("--source-commit is required when writing the seal")
    payload = build_payload(ROOT, args.source_commit)
    _write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
