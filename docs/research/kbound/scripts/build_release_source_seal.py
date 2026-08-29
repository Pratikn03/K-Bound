#!/usr/bin/env python3
"""Build or validate the final K-Bound release source/evidence seal.

The seal records a clean, explicitly supplied source commit and tree, then
hashes the maintained manuscript, release code, package, So2Sat protocol code,
and immutable receipt-bound authorities from that commit. Generated canonical
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
        "docs/research/kbound/claim_ledger.json",
        "docs/research/kbound/dashboard/data/snapshot.json",
        "docs/research/kbound/figures/fig_decision_value_frontier.png",
        "docs/research/kbound/figures/fig_phase_diagram.png",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex",
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
    "paper_source": (
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_abstract.tex",
        "docs/research/kbound/kbound_abstract_core.tex",
        "docs/research/kbound/kbound_abstract_disclosures.tex",
        "docs/research/kbound/paper/figure_fallback.tex",
        "docs/research/kbound/paper/float_params.tex",
        "docs/research/kbound/paper/references_kbound_expanded.tex",
        "docs/research/kbound/paper/sections/theory_certificate.tex",
        "docs/research/kbound/paper/sections/theory_core_main.tex",
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
        "docs/research/kbound/scripts/build_dashboard_snapshot.py",
        "docs/research/kbound/scripts/build_docx.py",
        "docs/research/kbound/scripts/build_empirical_data_quality_report_artifact.py",
        "docs/research/kbound/scripts/build_pdfs.sh",
        "docs/research/kbound/scripts/build_release_source_seal.py",
        "docs/research/kbound/scripts/build_result_manifest.py",
        "docs/research/kbound/scripts/build_results_source_compat.py",
        "docs/research/kbound/scripts/build_so2sat_numbers.py",
        "docs/research/kbound/scripts/make_tables.py",
        "docs/research/kbound/scripts/plot_canonical_decision_frontier.py",
        "docs/research/kbound/scripts/plot_conceptual_regime_geometry.py",
        "docs/research/kbound/scripts/refresh_storage_manifest.py",
        "docs/research/kbound/scripts/render_pdf_pages.py",
        "docs/research/kbound/scripts/run_frontier_kga_bridge.py",
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
        "docs/research/kbound/scripts/verify_release_checksums.py",
    ),
    "configuration": (
        "README.md",
        "CITATION.cff",
        "LICENSE",
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
        if path.startswith("kga/") and "__pycache__" not in path and not path.endswith(".pyc"):
            rows.append(("root_kga_package", path))
        if path.startswith("experiments/kbound/so2sat/") and path.endswith(".py"):
            rows.append(("so2sat_code", path))
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
        if not current.is_file():
            raise FileNotFoundError(f"sealed maintained path is missing: {relative}")
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


def build_payload(repo: Path, source_commit: str) -> dict[str, object]:
    resolved = _git("rev-parse", "--verify", f"{source_commit}^{{commit}}", repo=repo)
    head = _git("rev-parse", "--verify", "HEAD", repo=repo)
    if resolved != head:
        raise ValueError(f"source commit must equal HEAD: source={resolved}, HEAD={head}")
    dirty = _dirty_paths(repo)
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


def validate_seal(repo: Path, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unsupported release-source-seal schema")
    commit = str(payload.get("source_commit", ""))
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
    parser.add_argument("--check", action="store_true", help="validate an existing seal")
    args = parser.parse_args()
    output = args.output.resolve()
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
