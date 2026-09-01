#!/usr/bin/env python3
"""Fail-closed verifier for ``KBOUND_RELEASE_SHA256SUMS.txt``."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKSUMS = ROOT / "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt"

# One canonical inventory is used both by the release producer and by the
# default CLI verifier. Generic checksum verification remains an explicit
# lower-level operation; it must not accidentally certify a truncated release.
REQUIRED_RELEASE_PATHS: tuple[str, ...] = (
    "docs/research/kbound/claim_ledger.json",
    "docs/research/kbound/RESULT_MANIFEST.json",
    "docs/research/kbound/STORAGE_MANIFEST.json",
    "docs/research/kbound/results_source.json",
    "docs/research/kbound/audits/empirical_data_quality_2026_08_27/artifact.json",
    "docs/research/kbound/audits/empirical_data_quality_2026_08_27/audit_summary.json",
    "docs/research/kbound/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv",
    "docs/research/kbound/dashboard/data/snapshot.json",
    "docs/research/kbound/paper/generated/kbound_numbers.tex",
    "docs/research/kbound/paper/generated/kbound_result_manifest.json",
    "docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex",
    "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.json",
    "docs/research/kbound/paper/generated/current_policy_interval_diagnostics.tex",
    "docs/research/kbound/paper/generated/current_policy_interval_diagnostics_groups.tex",
    "docs/research/kbound/paper/generated/kbound_primary_accuracy_table.tex",
    "docs/research/kbound/paper/generated/kbound_auxiliary_accuracy_table.tex",
    "docs/research/kbound/paper/generated/kbound_auxiliary_balanced_accuracy_table.tex",
    "docs/research/kbound/paper/generated/cct20_safe_utility_display.tex",
    "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json",
    "docs/research/kbound/paper/generated/uniform_verdicts.json",
    "docs/research/kbound/figures/fig_decision_value_frontier.png",
    "docs/research/kbound/figures/fig_phase_diagram.png",
    "docs/research/kbound/kbound_short_final_draft.pdf",
    "docs/research/kbound/kbound_short_final_draft.docx",
    "docs/research/kbound/kbound_tmlr.pdf",
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
    "experiments/kbound/results/reconciled_panels_v1/CANONICAL_PANEL_RESULTS.md",
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex",
    "experiments/kbound/results/reconciled_panels_v1/source_manifest.json",
    "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
    "experiments/kbound/frontier_sweep_v1/decision_value_results.json",
    "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml",
    "research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json",
    "experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json",
    "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json",
    "experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json",
    "experiments/kbound/results/smoke_pacs_replay_v2/PACS_REPLAY_AUDIT.json",
    "experiments/kbound/results/edge_real_phone_v1/publication_gate.json",
    "docs/research/kbound/audits/phase1_provenance_2026_08_27/provenance_seal.json",
    "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
    "docs/research/kbound/paper/generated/cct20_release_manifest.json",
    "docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json",
    "docs/research/kbound/paper/generated/cct20_numbers.tex",
    "docs/research/kbound/paper/generated/cct20_primary_table.tex",
    "docs/research/kbound/paper/generated/cct20_location_effects.tex",
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml",
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml.sha256",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json.receipt.json",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json",
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json",
    "docs/research/kbound/paper/generated/so2sat_numbers.tex",
    "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json",
    "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json.receipt.json",
    "experiments/kbound/so2sat/prospective_protocol_v1.json",
    "experiments/kbound/so2sat/prospective_protocol_v1.json.receipt.json",
    "docs/research/kbound/audits/formal_foundations_2026_08_31.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_file(
    checksum_path: Path,
    *,
    root: Path,
    required_paths: tuple[str, ...] = (),
) -> int:
    """Verify a caller-selected inventory; the CLI defaults to the full release.

    This primitive is intentionally usable for small fixtures and independent
    byte checks. A publication gate must supply ``REQUIRED_RELEASE_PATHS``.
    """
    root = root.resolve()
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("release checksum file is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed checksum line {line_number}")
        digest, relative = match.groups()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
            raise ValueError(f"unsafe checksum path on line {line_number}: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest

    missing_entries = sorted(set(required_paths) - entries.keys())
    if missing_entries:
        raise ValueError("required checksum entries are missing: " + ", ".join(missing_entries))

    for relative, expected in entries.items():
        parts = PurePosixPath(relative).parts
        path = root.joinpath(*parts)
        # Checking only the final component misses a symlinked parent that can
        # redirect an apparently repository-relative file outside the release.
        linked = any(
            root.joinpath(*parts[:index]).is_symlink()
            for index in range(1, len(parts) + 1)
        )
        if linked or not path.is_file():
            raise FileNotFoundError(f"checksummed release file is missing or a symlink: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"release checksum mismatch for {relative}: expected {expected}, got {observed}"
            )
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checksum_file", nargs="?", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require", action="append", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generic",
        action="store_true",
        help="verify only listed files and explicit --require paths, not a K-Bound release",
    )
    mode.add_argument(
        "--list-required", action="store_true", help="print the canonical release inventory and exit"
    )
    args = parser.parse_args()
    if args.list_required:
        print("\n".join(REQUIRED_RELEASE_PATHS))
        return 0
    required = (() if args.generic else REQUIRED_RELEASE_PATHS) + tuple(args.require)
    try:
        count = verify_checksum_file(
            args.checksum_file.resolve(),
            root=args.root,
            required_paths=required,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    label = "generic checksums" if args.generic else "release checksums"
    print(f"{label}: PASS ({count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
