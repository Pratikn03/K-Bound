#!/usr/bin/env python3
"""Build the retrospective Phase-1 provenance coverage record.

This audit is deliberately conservative.  A hash computed after an experiment is
recorded as a *post-hoc snapshot*; it is never upgraded to evidence that the same
bytes were used at execution time unless an archived protocol/result already bound
that digest.  The output therefore separates:

* release-authority and archived-source hashes that are verifiable now;
* full configuration hashes recoverable from serialized historical configs;
* current dataset/checkpoint/code snapshots that can help a rerun;
* identities that cannot be recovered without a new, properly sealed run.

No training, model inference, label scoring, or result mutation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/research/kbound/audits/phase1_provenance_2026_08_27/"
    "provenance_seal.json"
)

CANONICAL = Path(
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
)
SOURCE_MANIFEST = Path(
    "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
)

CONFIG_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    (
        "officehome_primary_calibration",
        Path("experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json"),
    ),
    (
        "officehome_primary_test",
        Path("experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json"),
    ),
    (
        "officehome_replication_calibration",
        Path(
            "experiments/kbound/results/officehome_protocol_m_repl_targetval/"
            "result_target_val_eb504dd6.json"
        ),
    ),
    (
        "officehome_replication_test",
        Path(
            "experiments/kbound/results/officehome_protocol_m_repl_targettest/"
            "result_target_test_f761540b.json"
        ),
    ),
    (
        "iwildcam_historical_test",
        Path("experiments/kbound/results/iwildcam_full_test/result_e40faf29.json"),
    ),
    (
        "imagenet_r_protocol_d_historical_manifest",
        Path("experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/result_224624b1.json"),
    ),
    (
        "rxrx1_modelseed0",
        Path("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json"),
    ),
    (
        "rxrx1_modelseed1",
        Path("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed1/result_eef46aea.json"),
    ),
    (
        "rxrx1_modelseed2",
        Path("experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json"),
    ),
    (
        "camelyon17_archived_diagnostic",
        Path("experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json"),
    ),
)

PROTOCOL_PATHS: tuple[Path, ...] = (
    Path("research_lock/STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml"),
    Path("research_lock/CIFAR10C_SAR_REBUILD_PROTOCOL_v2.yaml"),
    Path("research_lock/imagenetc_protocol_E_v1.yaml"),
    Path("research_lock/IMAGENETR_DIVERSE_PANEL_PROTOCOL_D_v1.yaml"),
    Path("research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml"),
    Path("research_lock/IWILDCAM_PROTOCOL_H_v2.yaml"),
    Path("research_lock/PACS_VLCS_PREREG_PROTOCOL_v1.md"),
    Path("research_lock/RXRX1_PROTOCOL_J_v1.yaml"),
    Path("research_lock/CIFAR101_PROTOCOL_K_v1.yaml"),
    Path("research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml"),
    Path("research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml"),
    Path("research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json"),
)

CODE_PATHS: tuple[Path, ...] = (
    Path("scripts/reconcile_result_panels.py"),
    Path("scripts/sync_reconciled_panels.py"),
    Path("docs/research/kbound/scripts/build_results_source_compat.py"),
    Path("docs/research/kbound/scripts/build_result_manifest.py"),
    Path("docs/research/kbound/scripts/make_tables.py"),
    Path("docs/research/kbound/kbound_repro/release_checks.py"),
    Path("src/scripts/validate_manuscript_claims.py"),
    Path("kga/benefit.py"),
    Path("kga/evidence.py"),
    Path("kga/policy.py"),
    Path("kga/routing.py"),
    Path("kga/experiment_contract.py"),
    Path("experiments/kbound/wilds/run_integrity.py"),
    Path("experiments/kbound/wilds/run_camelyon17_kbound.py"),
    Path("experiments/kbound/wilds/run_imagenetr_kbound.py"),
    Path("experiments/kbound/wilds/run_iwildcam_kbound.py"),
    Path("experiments/kbound/wilds/run_rxrx1_kbound.py"),
    Path("experiments/kbound/wilds/run_geoshift_kbound.py"),
    Path("experiments/kbound/officehome/run_officehome_kbound.py"),
    Path("experiments/kbound/officehome/oh_data.py"),
    Path("docs/research/kbound/scripts/pacs_vlcs_runner.py"),
)

TORCHVISION_WEIGHT_FILES: tuple[str, ...] = (
    "convnext_base-6075fbad.pth",
    "convnext_tiny-983f1562.pth",
    "efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b3_rwightman-b3899882.pth",
    "resnet101-cd907fc2.pth",
    "resnet152-f82ba261.pth",
    "resnet50-11ad3fa6.pth",
    "resnext101_32x8d-110c445d.pth",
    "swin_b-68c6b09e.pth",
    "swin_t-704ceda3.pth",
    "vit_b_16-c867db91.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_dataless(path: Path) -> bool:
    """Detect a macOS/iCloud dataless placeholder without materializing it."""

    stat = path.stat()
    flags = int(getattr(stat, "st_flags", 0))
    # UF_DATALESS is 0x40000000 on Darwin.  The st_blocks fallback catches the
    # same condition on Python builds that do not expose st_flags.
    return bool(flags & 0x40000000) or (
        sys.platform == "darwin" and stat.st_size > 0 and stat.st_blocks == 0
    )


def snapshot_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    binding: str,
    logical_path: str | None = None,
) -> dict[str, Any]:
    label = logical_path or path.as_posix()
    if not path.is_file():
        return {
            "path": label,
            "status": "missing",
            "sha256": None,
            "size_bytes": None,
            "binding": binding,
            "expected_sha256": expected_sha256,
            "expected_matches": False if expected_sha256 else None,
        }
    if is_dataless(path):
        return {
            "path": label,
            "status": "dataless_placeholder_not_hashed",
            "sha256": None,
            "size_bytes": path.stat().st_size,
            "binding": binding,
            "expected_sha256": expected_sha256,
            "expected_matches": False if expected_sha256 else None,
        }
    observed = sha256_file(path)
    return {
        "path": label,
        "status": "present_hashed",
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "binding": binding,
        "expected_sha256": expected_sha256,
        "expected_matches": observed == expected_sha256 if expected_sha256 else None,
    }


def canonical_json_sha256(value: Any, *, legacy_spacing: bool = False) -> str:
    if legacy_spacing:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("ascii")
    else:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def config_record(name: str, relative: Path) -> dict[str, Any]:
    path = ROOT / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{relative} does not contain a serialized config object")
    # Historical v0.5/v1 runners used json.dumps(config, sort_keys=True) with
    # default spacing.  Recover the full digest while checking the archived
    # eight-character prefix rather than inventing a new identity.
    digest = canonical_json_sha256(config, legacy_spacing=True)
    recorded = document.get("config_sha8")
    return {
        "name": name,
        "artifact": relative.as_posix(),
        "artifact_sha256": sha256_file(path),
        "config_hash_algorithm": "sha256(json.dumps(config, sort_keys=True))",
        "config_sha256_recovered": digest,
        "recorded_config_sha8": recorded,
        "recorded_prefix_matches": isinstance(recorded, str) and digest.startswith(recorded),
        "binding": (
            "recovered_from_the_exact_serialized_config_inside_the_historical_artifact; "
            "does_not_supply_missing_dataset_checkpoint_or_code_identity"
        ),
    }


def git_output(*args: str) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def code_record(relative: Path) -> dict[str, Any]:
    path = ROOT / relative
    status = git_output("status", "--short", "--untracked-files=all", "--", relative.as_posix())
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "git_status": status or "tracked_clean_at_a_dirty_repository_snapshot",
        "binding": "current_working_copy_snapshot_only_not_historical_execution_identity",
    }


def tree_snapshot(root: Path, *, root_id: str) -> dict[str, Any]:
    allowed = {".jpg", ".jpeg", ".png"}
    if not root.is_dir():
        return {
            "root_id": root_id,
            "status": "missing",
            "file_count": 0,
            "size_bytes": 0,
            "tree_sha256": None,
        }
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in allowed
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    dataless: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if is_dataless(path):
            dataless.append(relative)
            continue
        size = path.stat().st_size
        total_bytes += size
        row = {
            "path": relative,
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            + b"\n"
        )
    complete = not dataless and len(files) > 0
    return {
        "root_id": root_id,
        "status": "present_complete_snapshot" if complete else "present_incomplete_snapshot",
        "file_count": len(files),
        "hashed_file_count": len(files) - len(dataless),
        "dataless_file_count": len(dataless),
        "size_bytes": total_bytes,
        "tree_sha256": digest.hexdigest() if complete else None,
        "tree_hash_algorithm": (
            "sha256(concat(canonical_json({path,size_bytes,sha256}) + newline) "
            "for sorted image-relative paths)"
        ),
        "binding": (
            "posthoc_current_materialization_snapshot_only; historical run did not "
            "precommit this tree digest"
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    canonical_path = ROOT / CANONICAL
    source_manifest_path = ROOT / SOURCE_MANIFEST
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest_sha = sha256_file(source_manifest_path)

    source_checks = []
    for row in source_manifest["files"]:
        original = ROOT / row["source"]
        compact = ROOT / row["destination"]
        source_checks.append(
            {
                "source": row["source"],
                "source_present": original.is_file(),
                "source_sha256_matches": (
                    original.is_file() and sha256_file(original) == row["original_sha256"]
                ),
                "compact": row["destination"],
                "compact_present": compact.is_file(),
                "compact_sha256_matches": (
                    compact.is_file() and sha256_file(compact) == row["compact_sha256"]
                ),
            }
        )

    cifar_checkpoint = snapshot_file(
        ROOT / "experiments/kbound/cifar/resnet18_cifar.pt",
        expected_sha256="43333456a795bbe679966c14812f9964d8b3bf060d30ca2b3d5051cb8c9d7491",
        binding=(
            "historically_bound_for_the_CIFAR10C_SAR_rebuild_by_"
            "research_lock/CIFAR10C_SAR_REBUILD_PROTOCOL_v2.yaml; "
            "Tent/EATA manifests name the path but do not record its digest"
        ),
        logical_path="experiments/kbound/cifar/resnet18_cifar.pt",
    )

    checkpoint_rows = [
        cifar_checkpoint,
        snapshot_file(
            ROOT / "experiments/kbound/results/officehome_f0/f0_resnet50_rw_seed0.pt",
            binding="posthoc_snapshot_only_not_bound_in_the_historical_OfficeHome_result",
            logical_path=(
                "experiments/kbound/results/officehome_f0/f0_resnet50_rw_seed0.pt"
            ),
        ),
        snapshot_file(
            ROOT / "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt",
            binding="posthoc_snapshot_only_not_bound_in_the_historical_iWildCam_result",
            logical_path=(
                "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt"
            ),
        ),
        snapshot_file(
            ROOT / "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0_last.pt",
            binding="not_usable_as_evidence_or_checkpoint_identity",
            logical_path=(
                "experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0_last.pt"
            ),
        ),
        snapshot_file(
            ROOT
            / "experiments/kbound/results/multiseed/iwildcam/"
            "iwildcam_kbound_finder_v1/f0_resnet18_seed0.pt",
            binding="diagnostic_finder_checkpoint_posthoc_snapshot_only",
            logical_path=(
                "experiments/kbound/results/multiseed/iwildcam/"
                "iwildcam_kbound_finder_v1/f0_resnet18_seed0.pt"
            ),
        ),
    ]

    officehome_checkpoint_dir = (
        ROOT
        / "experiments/kbound/results/natural_replication_strengthening_v1/"
        "officehome/checkpoints"
    )
    for seed in range(5):
        meta_path = officehome_checkpoint_dir / f"f0_meta_seed{seed}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint_rows.append(
            snapshot_file(
                officehome_checkpoint_dir / f"f0_resnet50_rw_seed{seed}.pt",
                expected_sha256=meta["checkpoint_sha256"],
                binding=(
                    "bound_to_its_checkpoint_metadata_and_invalidated_posthoc_candidate_audit; "
                    "not_a_promoted_KGA_routing_result"
                ),
                logical_path=(
                    "experiments/kbound/results/natural_replication_strengthening_v1/"
                    f"officehome/checkpoints/f0_resnet50_rw_seed{seed}.pt"
                ),
            )
        )

    torchvision_rows = []
    torch_root = args.torch_checkpoints_root
    for filename in TORCHVISION_WEIGHT_FILES:
        torchvision_rows.append(
            snapshot_file(
                torch_root / filename,
                binding=(
                    "posthoc_current_torchvision_cache_snapshot_only; historical ImageNet-R "
                    "artifact names the weights family but did not store a full digest"
                ),
                logical_path=f"$TORCH_HOME/hub/checkpoints/{filename}",
            )
        )

    cifar10c_files = []
    for filename in (
        "gaussian_noise.npy",
        "defocus_blur.npy",
        "fog.npy",
        "contrast.npy",
        "pixelate.npy",
        "jpeg_compression.npy",
        "labels.npy",
    ):
        expected = (
            "e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b"
            if filename == "labels.npy"
            else None
        )
        cifar10c_files.append(
            snapshot_file(
                ROOT / "experiments/kbound/cifar/CIFAR-10-C" / filename,
                expected_sha256=expected,
                binding=(
                    "labels_are_bound_for_the_SAR_rebuild; corruption_arrays_are_current_"
                    "posthoc_snapshots_only_and_two_are_dataless_in_this_working_copy"
                ),
                logical_path=f"experiments/kbound/cifar/CIFAR-10-C/{filename}",
            )
        )

    cifar101_files = [
        snapshot_file(
            ROOT / "experiments/kbound/cifar/CIFAR-10.1/cifar10.1_v6_data.npy",
            binding="posthoc_current_snapshot_only_historical_protocol_did_not_store_digest",
            logical_path="experiments/kbound/cifar/CIFAR-10.1/cifar10.1_v6_data.npy",
        ),
        snapshot_file(
            ROOT / "experiments/kbound/cifar/CIFAR-10.1/cifar10.1_v6_labels.npy",
            binding="posthoc_current_snapshot_only_historical_protocol_did_not_store_digest",
            logical_path="experiments/kbound/cifar/CIFAR-10.1/cifar10.1_v6_labels.npy",
        ),
    ]

    code = [code_record(path) for path in CODE_PATHS]
    protocols = [
        snapshot_file(
            ROOT / path,
            binding="current_protocol_or_design_history_bytes",
            logical_path=path.as_posix(),
        )
        for path in PROTOCOL_PATHS
    ]

    git_status = git_output("status", "--porcelain=v1") or ""
    config_rows = [config_record(name, path) for name, path in CONFIG_ARTIFACTS]
    config_prefix_failures = [
        row["artifact"] for row in config_rows if not row["recorded_prefix_matches"]
    ]
    source_failures = [
        row
        for row in source_checks
        if not row["source_sha256_matches"] or not row["compact_sha256_matches"]
    ]
    checkpoint_mismatches = [
        row["path"]
        for row in checkpoint_rows
        if row["expected_sha256"] is not None and row["expected_matches"] is not True
    ]

    imagenetr_tree = tree_snapshot(
        args.imagenetr_root, root_id="imagenet_r_current_materialized_images"
    )
    officehome_tree = tree_snapshot(
        args.officehome_root, root_id="officehome_current_materialized_images"
    )

    return {
        "schema_version": "kbound-phase1-provenance-audit-v1",
        "audit_date": "2026-08-27",
        "scope": (
            "Retrospective, read-only provenance audit. Post-hoc hashes bind only the bytes "
            "observed now and are not evidence that identical bytes were used historically."
        ),
        "verdict": "PARTIAL_RETROSPECTIVE_SEAL_NEW_RUNS_STILL_REQUIRED",
        "repository_snapshot": {
            "head_commit": git_output("rev-parse", "HEAD"),
            "worktree_clean": not bool(git_status),
            "status": "DIRTY_WORKING_COPY_SNAPSHOT" if git_status else "CLEAN_COMMIT_SNAPSHOT",
            "attestation": (
                "Current code bytes are hashed individually below. This audit does not claim "
                "that the working tree is clean or that current code produced historical runs."
            ),
        },
        "release_authority": {
            "canonical_panel": snapshot_file(
                canonical_path,
                binding="current_numerical_release_authority",
                logical_path=CANONICAL.as_posix(),
            ),
            "source_manifest": snapshot_file(
                source_manifest_path,
                binding="current_source_lineage_authority",
                logical_path=SOURCE_MANIFEST.as_posix(),
            ),
            "canonical_declared_source_manifest_sha256": canonical.get(
                "source_manifest_sha256"
            ),
            "live_source_manifest_sha256": source_manifest_sha,
            "source_manifest_binding_matches": (
                canonical.get("source_manifest_sha256") == source_manifest_sha
            ),
            "source_file_count": source_manifest.get("file_count"),
            "source_hash_checks_pass": not source_failures,
            "source_hash_failure_count": len(source_failures),
        },
        "historical_configuration_hashes": {
            "algorithm_note": (
                "The full SHA-256 is deterministically recoverable because the exact config "
                "object is serialized and its archived config_sha8 equals the digest prefix."
            ),
            "records": config_rows,
            "all_recorded_prefixes_match": not config_prefix_failures,
            "prefix_failures": config_prefix_failures,
        },
        "protocol_file_snapshots": protocols,
        "current_code_snapshot": {
            "binding": (
                "Current release/rerun implementation bytes only. Historical code identity "
                "exists only where an old manifest or immutable executed-source copy says so."
            ),
            "files": code,
            "canonical_generator_hash_matches": (
                canonical.get("generator_sha256")
                == next(
                    row["sha256"]
                    for row in code
                    if row["path"] == "scripts/reconcile_result_panels.py"
                )
            ),
        },
        "checkpoint_snapshots": {
            "records": checkpoint_rows,
            "recorded_hash_mismatches": checkpoint_mismatches,
            "torchvision_imagenet_r_weights": torchvision_rows,
            "interpretation": (
                "CIFAR SAR and the five Office-Home audit checkpoints have archived expected "
                "digests. Other listed checkpoint hashes are post-hoc snapshots only. ImageNet-R "
                "weights are current torchvision-cache bytes and were not sealed by the old run."
            ),
        },
        "dataset_snapshots": {
            "imagenet_r": imagenetr_tree,
            "officehome": officehome_tree,
            "cifar10c_promoted_subset": {
                "status": (
                    "INCOMPLETE_CURRENT_MATERIALIZATION_NOT_FULLY_SEALABLE"
                    if any(row["status"] != "present_hashed" for row in cifar10c_files)
                    else "CURRENT_FILES_HASHED_POSTHOC"
                ),
                "files": cifar10c_files,
                "historical_binding": (
                    "SAR labels hash was predeclared; corruption-array hashes were not. "
                    "Tent/EATA historical runs did not bind an archive or population digest."
                ),
            },
            "cifar10_1_v6": {
                "status": "CURRENT_FILES_HASHED_POSTHOC",
                "files": cifar101_files,
                "historical_binding": "no archived data digest",
            },
            "not_content_sealed": [
                {
                    "dataset": "ImageNet-C",
                    "reason": "archive/reference checksum absent and historical sample identities absent",
                },
                {
                    "dataset": "PACS",
                    "reason": "Hugging Face revision unpinned and promoted replay lacks full per-cell state",
                },
                {
                    "dataset": "Camelyon17",
                    "reason": "historical internal population was incomplete and no population digest was archived",
                },
                {
                    "dataset": "iWildCam",
                    "reason": "current local population is incomplete and differs from the archived run; no historical population digest",
                },
                {
                    "dataset": "RxRx1",
                    "reason": "historical population and checkpoint digests were not serialized",
                },
            ],
        },
        "known_historical_identity_gaps": [
            "Most historical natural-shift artifacts lack dataset/population content hashes.",
            "Most historical natural-shift artifacts lack checkpoint file and tensor-state hashes.",
            "Historical stream seeds cannot be relabelled as independent model seeds.",
            "ImageNet-R's main v0.5 manifest lists seeds 0-2 while the canonical source set contains seed 3 files.",
            "The historical ImageNet-R manifest did not bind per-backbone native preprocessing transforms.",
            "The old iWildCam result did not bind the official WILDS metric implementation or a complete population.",
            "A post-hoc digest cannot prove historical byte identity; publication-grade closure requires new sealed runs.",
        ],
        "mechanical_checks": {
            "canonical_source_manifest_binding": (
                canonical.get("source_manifest_sha256") == source_manifest_sha
            ),
            "all_106_source_and_compact_hashes_match": not source_failures,
            "all_recoverable_config_prefixes_match": not config_prefix_failures,
            "all_archived_expected_checkpoint_hashes_match": not checkpoint_mismatches,
            "canonical_generator_hash_matches": (
                canonical.get("generator_sha256")
                == next(
                    row["sha256"]
                    for row in code
                    if row["path"] == "scripts/reconcile_result_panels.py"
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--imagenetr-root",
        type=Path,
        default=ROOT / "experiments/kbound/data/imagenet-r",
    )
    parser.add_argument(
        "--officehome-root",
        type=Path,
        default=Path(
            os.environ.get(
                "KBOUND_OFFICEHOME_ROOT",
                str(ROOT / "experiments/kbound/data/officehome"),
            )
        ),
    )
    parser.add_argument(
        "--torch-checkpoints-root",
        type=Path,
        default=Path(
            os.environ.get(
                "KBOUND_TORCH_CHECKPOINTS_ROOT",
                str(Path.home() / ".cache/torch/hub/checkpoints"),
            )
        ),
    )
    args = parser.parse_args()
    payload = build(args)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checks = payload["mechanical_checks"]
    print(f"wrote {output}")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    print(f"  verdict: {payload['verdict']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
