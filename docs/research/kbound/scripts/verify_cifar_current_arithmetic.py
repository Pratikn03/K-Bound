#!/usr/bin/env python3
"""Verify the release-bound current CIFAR arithmetic artifacts without rewriting them."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_AUDIT = Path(
    "experiments/kbound/results/"
    "cifar_paired_array_audit_20260912.Q4GKLg/PAIRED_ARRAY_PROVENANCE_AUDIT.json"
)
DEFAULT_ARTIFACT_DIR = Path(
    "docs/research/kbound/paper/generated/current_cifar_baselines_20260912"
)
EXPECTED_AUDIT_SHA256 = "051905a22f96deed4249f9e39d954223e5a350d323478df45f29fbc49379de30"
EXPECTED_PANEL_SHA256 = "04a2036ad2608e810a8da430c364cdbd6c7c3bc1036d87011686c797962fb462"
PANEL_NAME = "CURRENT_ARITHMETIC_PANEL.json"
CELL_NAME = "cells.jsonl"
LATEX_NAME = "current_arithmetic_panel.tex"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def required_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"missing required current CIFAR artifact: {path}")
    return path.read_bytes()


def load_json(payload: bytes, label: str):
    try:
        return json.loads(payload, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {error}") from error


def verify(
    *,
    repo: Path,
    audit: Path,
    audit_sha256: str,
    artifact_dir: Path,
    expected_panel_sha256: str,
    generator: Path,
) -> None:
    audit_payload = required_bytes(audit)
    observed_audit_sha256 = sha256(audit_payload)
    if observed_audit_sha256 != audit_sha256:
        raise ValueError("paired-array provenance audit SHA-256 mismatch")

    panel_payload = required_bytes(artifact_dir / PANEL_NAME)
    cells_payload = required_bytes(artifact_dir / CELL_NAME)
    latex_payload = required_bytes(artifact_dir / LATEX_NAME)
    if sha256(panel_payload) != expected_panel_sha256:
        raise ValueError(f"{PANEL_NAME} release digest mismatch")
    panel = load_json(panel_payload, PANEL_NAME)
    if panel.get("generation", {}).get("prior_audit_sha256") != observed_audit_sha256:
        raise ValueError(f"{PANEL_NAME} prior-audit binding mismatch")
    if panel.get("cell_record_file") != {
        "n": 2160,
        "path": CELL_NAME,
        "sha256": sha256(cells_payload),
    }:
        raise ValueError(f"{CELL_NAME} digest or row-count binding mismatch")
    if panel.get("latex_file") != {
        "path": LATEX_NAME,
        "sha256": sha256(latex_payload),
    }:
        raise ValueError(f"{LATEX_NAME} digest binding mismatch")

    if not generator.is_file():
        raise ValueError(f"missing current CIFAR arithmetic generator: {generator}")
    with tempfile.TemporaryDirectory(prefix="kbound-current-cifar-check-") as temporary:
        replay_dir = Path(temporary) / "replayed"
        process = subprocess.run(
            [
                sys.executable,
                str(generator),
                "--repo",
                str(repo),
                "--audit",
                str(audit),
                "--audit-sha256",
                audit_sha256,
                "--out-dir",
                str(replay_dir),
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "generator exited nonzero"
            raise ValueError(f"current CIFAR source authentication/replay failed: {detail}")

        replay_cells = required_bytes(replay_dir / CELL_NAME)
        replay_latex = required_bytes(replay_dir / LATEX_NAME)
        if replay_cells != cells_payload:
            raise ValueError(f"{CELL_NAME} regenerated-content mismatch")
        if replay_latex != latex_payload:
            raise ValueError(f"{LATEX_NAME} regenerated-content mismatch")

        replay_panel = load_json(required_bytes(replay_dir / PANEL_NAME), "regenerated panel JSON")
        # These two fields record the invocation, not scientific content.  The
        # stored panel is independently pinned above, so normalize only them
        # before requiring byte-identical canonical JSON for every other field.
        replay_panel["created_utc"] = panel["created_utc"]
        replay_panel["generation"]["python"] = panel["generation"]["python"]
        normalized_replay = (
            json.dumps(replay_panel, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        if normalized_replay != panel_payload:
            raise ValueError(f"{PANEL_NAME} regenerated-content mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256", default=EXPECTED_AUDIT_SHA256)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--expected-panel-sha256", default=EXPECTED_PANEL_SHA256)
    parser.add_argument("--generator", type=Path, default=Path(__file__).with_name("replay_cifar_current_arithmetic.py"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    audit = args.audit.resolve() if args.audit else repo / DEFAULT_AUDIT
    artifact_dir = args.artifact_dir.resolve() if args.artifact_dir else repo / DEFAULT_ARTIFACT_DIR
    try:
        verify(
            repo=repo,
            audit=audit,
            audit_sha256=args.audit_sha256,
            artifact_dir=artifact_dir,
            expected_panel_sha256=args.expected_panel_sha256,
            generator=args.generator.resolve(),
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: current CIFAR arithmetic release binding verified (seven audited inputs; three artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
