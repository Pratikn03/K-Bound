#!/usr/bin/env python3
"""Fail-closed verifier for ``KBOUND_RELEASE_SHA256SUMS.txt``."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKSUMS = ROOT / "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt"

CURRENT_RELEASE_PDF_PATHS: tuple[str, ...] = (
    "docs/research/kbound/release/current/kbound_short_main.pdf",
    "docs/research/kbound/release/current/kbound_short_supplement.pdf",
    "docs/research/kbound/release/current/kbound_tmlr.pdf",
    "docs/research/kbound/release/current/kbound_full_report.pdf",
)
CURRENT_RELEASE_CHECKSUM_PATH = (
    "docs/research/kbound/release/current/KBOUND_CURRENT_SHA256SUMS.txt"
)

# This archive embeds the completed checksum file, so including it in that same
# file would create an impossible self-referential hash. Its deterministic bytes
# and internal commitments are verified by build_anonymous_supplement.py.
POST_CHECKSUM_FILE = "docs/research/kbound/KBOUND_POST_CHECKSUM_SHA256SUMS.txt"
POST_CHECKSUM_REQUIRED_PATHS: tuple[str, ...] = ("docs/research/kbound/release/kbound_anonymous_supplement.zip",)
POST_CHECKSUM_RELEASE_PATHS: tuple[str, ...] = (
    POST_CHECKSUM_FILE,
    *POST_CHECKSUM_REQUIRED_PATHS,
)

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
    *CURRENT_RELEASE_PDF_PATHS,
    CURRENT_RELEASE_CHECKSUM_PATH,
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
    "docs/research/kbound/audits/python_environment_2026_09_02.json",
    "docs/research/kbound/release_python_environment_macos_arm64.json",
    "docs/research/kbound/audits/release_toolchain_2026_09_02.json",
    "docs/research/kbound/audits/release_source_seal_2026_08_29.json",
    "docs/research/kbound/audits/repository_test_inventory.json",
    "docs/research/kbound/paper/generated/cct20_release_manifest.json",
    "docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json",
    "docs/research/kbound/paper/generated/cct20_numbers.tex",
    "docs/research/kbound/paper/generated/cct20_primary_table.tex",
    "docs/research/kbound/paper/generated/cct20_location_effects.tex",
    "docs/research/kbound/paper/generated/cct20_reporting_numbers.tex",
    "docs/research/kbound/paper/generated/current_release_identity.tex",
    "docs/research/kbound/paper/release/current_release.json",
    "docs/research/kbound/CURRENT_RELEASE.md",
    "docs/research/kbound/paper/reports/KBOUND_SOURCE_OUTPUT_MAP.md",
    "docs/research/kbound/paper/reports/KBOUND_CURRENT_RELEASE_REPAIR_REPORT.md",
    "docs/research/kbound/paper/reports/TMLR_ANONYMITY_AUDIT.md",
    "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound_short.pdf",
    "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound.pdf",
    "docs/research/kbound/archive/superseded_do_not_cite/originals/kbound_submission.pdf",
    "docs/research/kbound/archive/superseded_do_not_cite/kbound_short_SUPERSEDED.pdf",
    "docs/research/kbound/archive/superseded_do_not_cite/kbound_SUPERSEDED.pdf",
    "docs/research/kbound/archive/superseded_do_not_cite/kbound_submission_SUPERSEDED.pdf",
    "docs/research/kbound/release/cct20_public_evidence_bundle.zip",
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml",
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml.sha256",
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
    checksum_absolute = checksum_path.absolute()
    checksum_parts = checksum_absolute.parts
    if (
        any(Path(*checksum_parts[:index]).is_symlink() for index in range(1, len(checksum_parts) + 1))
        or not checksum_absolute.is_file()
    ):
        raise FileNotFoundError(f"release checksum receipt is missing or a symlink: {checksum_path}")
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
        linked = any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1))
        if linked or not path.is_file():
            raise FileNotFoundError(f"checksummed release file is missing or a symlink: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"release checksum mismatch for {relative}: expected {expected}, got {observed}")
    return len(entries)


def write_checksum_file(
    checksum_path: Path,
    *,
    root: Path,
    required_paths: tuple[str, ...],
) -> int:
    """Atomically write and self-verify one explicit, complete byte inventory."""

    root = root.resolve()
    if not required_paths or len(set(required_paths)) != len(required_paths):
        raise ValueError("checksum inventory must be nonempty and contain unique paths")
    lines: list[str] = []
    for relative in sorted(required_paths):
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
            raise ValueError(f"unsafe checksum inventory path: {relative}")
        path = root.joinpath(*parsed.parts)
        if any(root.joinpath(*parsed.parts[:index]).is_symlink() for index in range(1, len(parsed.parts) + 1)):
            raise FileNotFoundError(f"checksummed release file is a symlink: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"checksummed release file is missing: {relative}")
        lines.append(f"{_sha256(path)}  {relative}\n")

    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", prefix=f".{checksum_path.name}.",
            dir=checksum_path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.writelines(lines)
        count = verify_checksum_file(temporary, root=root, required_paths=required_paths)
        os.replace(temporary, checksum_path)
        temporary = None
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    mode.add_argument("--list-required", action="store_true", help="print the canonical release inventory and exit")
    mode.add_argument("--write", action="store_true", help="atomically regenerate the complete canonical release checksum inventory")
    args = parser.parse_args()
    if args.list_required:
        print("\n".join(REQUIRED_RELEASE_PATHS))
        return 0
    if args.write:
        try:
            count = write_checksum_file(
                args.checksum_file.resolve(), root=args.root, required_paths=REQUIRED_RELEASE_PATHS
            )
        except (OSError, UnicodeError, ValueError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print(f"release checksums: WROTE AND VERIFIED ({count} files)")
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
