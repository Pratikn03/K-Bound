#!/usr/bin/env python3
"""Fail-closed validator for CIFAR10C_SAR_REBUILD_v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

# --- provenance anchor -------------------------------------------------------
# This tree was produced by a runner that has since been edited in the worktree
# (leave-one-out conformal calibration, kbound_decide routing, EVAL_CHUNK).  Every
# seed's result_manifest.json records the commit it was executed from, so the runner
# is validated against an IMMUTABLE ARCHIVED COPY recovered from that commit -- never
# against the live, mutable worktree path.  The expected digest below is UNCHANGED
# from the original freeze; the recovered source hashes to exactly this value.
EXECUTED_RUNNER_COMMIT = "675ebfcb7a56854123b13250e01843f69007589b"
EXECUTED_RUNNER_SOURCE_PATH = "docs/research/kbound/scripts/cifar_tent_mps_v2.py"
ARCHIVED_RUNNER = "research_lock/executed_sources/cifar_tent_mps_v2__675ebfcb.py"
LIVE_RUNNER_PATH = "docs/research/kbound/scripts/cifar_tent_mps_v2.py"

EXPECTED = {
    ARCHIVED_RUNNER: "f1687904d36114340ae7da055197f6bd44c08e2f617d17703a52824765e62dbc",
    "experiments/kbound/cifar/resnet18_cifar.pt": "43333456a795bbe679966c14812f9964d8b3bf060d30ca2b3d5051cb8c9d7491",
    "experiments/kbound/cifar/CIFAR-10-C/labels.npy": "e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b",
}
REQUIRED_RECORD_FIELDS = {
    "condition", "B", "a0", "a_adapted", "b_hat", "eps_conformal", "kga_decision"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--results", type=Path, default=Path("experiments/kbound/results/cifar10c_sar_rebuild_v2"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    results = args.results if args.results.is_absolute() else root / args.results

    errors: list[str] = []
    for rel, expected in EXPECTED.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"missing frozen input: {rel}")
        elif sha256(path) != expected:
            errors.append(f"hash mismatch: {rel}")

    # The archived runner must be byte-identical to the git object at the executed
    # commit.  This is what makes the archive trustworthy rather than merely asserted.
    archived = root / ARCHIVED_RUNNER
    if archived.is_file():
        try:
            blob = subprocess.run(
                ["git", "show", f"{EXECUTED_RUNNER_COMMIT}:{EXECUTED_RUNNER_SOURCE_PATH}"],
                cwd=root, capture_output=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            errors.append(
                f"cannot recover runner from executed commit {EXECUTED_RUNNER_COMMIT[:12]}: {exc}"
            )
        else:
            if hashlib.sha256(blob).hexdigest() != sha256(archived):
                errors.append(
                    "archived runner does not match the git object at "
                    f"{EXECUTED_RUNNER_COMMIT[:12]}:{EXECUTED_RUNNER_SOURCE_PATH}"
                )

    reference_conditions: list[str] | None = None
    complete: list[int] = []
    for seed in range(5):
        path = results / f"seed{seed}" / f"per_condition_cifar10c_sar_seed{seed}.json"
        if not path.is_file():
            if not args.allow_partial:
                errors.append(f"missing seed file: {path.relative_to(root)}")
            continue
        # Bind this seed to the executed commit.  Without this the archived runner is
        # only asserted to be the one that ran; with it, every seed attests the commit
        # the archive was recovered from.
        manifest_path = path.parent / "result_manifest.json"
        if not manifest_path.is_file():
            errors.append(f"seed {seed}: missing result_manifest.json")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("git_hash") != EXECUTED_RUNNER_COMMIT:
                errors.append(
                    f"seed {seed}: manifest git_hash {manifest.get('git_hash')!r} != "
                    f"executed runner commit {EXECUTED_RUNNER_COMMIT}"
                )
            argv = manifest.get("argv") or []
            if argv and not str(argv[0]).endswith(Path(EXECUTED_RUNNER_SOURCE_PATH).name):
                errors.append(f"seed {seed}: manifest argv[0] is not the frozen runner: {argv[0]!r}")

        payload = json.loads(path.read_text())
        records = payload.get("records", [])
        conditions = [str(record.get("condition")) for record in records]
        if payload.get("seed") != seed or payload.get("method") != "sar":
            errors.append(f"seed/method metadata mismatch: seed {seed}")
        if len(records) != 432 or len(set(conditions)) != 432:
            errors.append(f"seed {seed}: expected 432 unique conditions, got {len(records)}/{len(set(conditions))}")
        for index, record in enumerate(records):
            missing = REQUIRED_RECORD_FIELDS - record.keys()
            if missing:
                errors.append(f"seed {seed} record {index}: missing {sorted(missing)}")
                break
        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            errors.append(f"seed {seed}: condition order differs from first completed seed")
        complete.append(seed)

    if errors:
        print("CIFAR10C SAR REBUILD: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    status = "PARTIAL PASS" if args.allow_partial and complete != list(range(5)) else "PASS"
    print(f"CIFAR10C SAR REBUILD: {status}; complete seeds={complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
