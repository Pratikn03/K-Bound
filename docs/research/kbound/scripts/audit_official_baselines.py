#!/usr/bin/env python3
"""Audit whether external baseline artifacts may carry an official-code label.

This script is intentionally fail closed.  Source code being present is not
enough: provenance, a reproducible environment, native output, and a complete
converted decision file must all be available before manuscript promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PATH_BINDING_SCHEMA = "git-repository-relative-posix-v1"
SOURCE_TREE_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "cached_data",
    "dataset",
    "log",
    "raw_logs",
    "public",
}
SOURCE_TREE_EXCLUDED_SUFFIXES = {".ckpt", ".npy", ".npz", ".pkl", ".pt", ".pth"}
_UF_DATALESS = 0x40000000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def tree_hash(root: Path) -> str | None:
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if SOURCE_TREE_EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if path.suffix.lower() in SOURCE_TREE_EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    files.sort()
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def decision_count(path: Path) -> tuple[int, str | None]:
    if not path.is_file():
        return 0, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, sha256_file(path)
    decisions = raw.get("decisions", raw) if isinstance(raw, dict) else {}
    if not isinstance(decisions, dict):
        return 0, sha256_file(path)
    valid = sum(value in {"adapt", "freeze", "abstain"} for value in decisions.values())
    return valid, sha256_file(path)


def repository_relative(path: Path, repo: Path) -> str:
    """Return a portable path bound to the audited Git repository root."""

    resolved_repo = repo.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_repo).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"audited path is outside the repository-root binding: {resolved_path.name}"
        ) from exc


def native_logs(root: Path, *, repo: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    hashes: dict[str, str] = {}
    unavailable: list[str] = []
    failed = []
    for path in files:
        relative = repository_relative(path, repo)
        try:
            if getattr(path.stat(), "st_flags", 0) & _UF_DATALESS:
                unavailable.append(relative)
                continue
            hashes[relative] = sha256_file(path)
        except OSError:
            unavailable.append(relative)
            continue
        if path.suffix.lower() not in {".txt", ".log", ".json", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unavailable.append(relative)
            hashes.pop(relative, None)
            continue
        if "Traceback (most recent call last)" in text or "RuntimeError:" in text:
            failed.append(relative)
    return {
        "count": len(files),
        "sha256": hashes,
        "unavailable": sorted(set(unavailable)),
        "failure_markers": failed,
        "successful": bool(files) and not failed and not unavailable,
    }


def audit_aetta(repo: Path, out: Path) -> dict[str, Any]:
    source = repo / "AETTA"
    license_file = source / "LICENSE.txt"
    environment = source / "aetta.yml"
    decisions = out / "aetta_decisions.json"
    count, decisions_sha = decision_count(decisions)
    archived_run_dir = out / "aetta_native"
    logs = native_logs(
        archived_run_dir if archived_run_dir.is_dir() else source / "raw_logs",
        repo=repo,
    )
    vendor = source / "VENDOR.md"
    vendor_text = vendor.read_text(encoding="utf-8") if vendor.is_file() else ""
    upstream_match = re.search(r"(?im)^upstream_commit_sha:\s*([0-9a-f]{40})\s*$", vendor_text)
    upstream_commit_recorded = upstream_match is not None
    checks = {
        "source_present": source.is_dir(),
        "upstream_commit_recorded": upstream_commit_recorded,
        "license_present": license_file.is_file(),
        "environment_lock_present": environment.is_file(),
        "native_logs_successful": logs["successful"],
        "converted_decisions_nonempty": count > 0,
    }
    return {
        "method": "aetta",
        "source_mode": "vendored",
        "source_tree_sha256": tree_hash(source),
        "vendored_in_git_commit": git_output(repo, "log", "-1", "--format=%H", "--", "AETTA"),
        "upstream_commit": upstream_match.group(1) if upstream_match else None,
        "license_sha256": sha256_file(license_file) if license_file.is_file() else None,
        "environment_sha256": sha256_file(environment) if environment.is_file() else None,
        "native_logs": logs,
        "decision_count": count,
        "decisions_sha256": decisions_sha,
        "checks": checks,
        "official_label_allowed": all(checks.values()),
    }


def audit_poem(repo: Path, out: Path) -> dict[str, Any]:
    source = repo / "external" / "poem"
    commit = git_output(source, "rev-parse", "HEAD") if source.is_dir() else None
    status = git_output(source, "status", "--porcelain") if source.is_dir() else None
    remote = git_output(source, "remote", "get-url", "origin") if source.is_dir() else None
    license_candidates = sorted(source.glob("LICENSE*")) if source.is_dir() else []
    environment_candidates = [source / "requirements.txt", source / "environment.yml"]
    environment_files = [path for path in environment_candidates if path.is_file()]
    decisions = out / "poem_decisions.json"
    count, decisions_sha = decision_count(decisions)
    logs = native_logs(out / "poem_imagenetc", repo=repo)
    checks = {
        "source_present": source.is_dir(),
        "upstream_remote_recorded": bool(remote),
        "commit_recorded": bool(commit),
        "source_clean": status == "",
        "root_license_present": bool(license_candidates),
        "environment_lock_present": bool(environment_files),
        "native_logs_successful": logs["successful"],
        "converted_decisions_nonempty": count > 0,
    }
    return {
        "method": "poem",
        "source_mode": "nested_git_checkout",
        "upstream_remote": remote,
        "upstream_commit": commit,
        "source_dirty_paths": status.splitlines() if status else [],
        "license_sha256": sha256_file(license_candidates[0]) if license_candidates else None,
        "environment_sha256": {
            path.name: sha256_file(path) for path in environment_files
        },
        "native_logs": logs,
        "decision_count": count,
        "decisions_sha256": decisions_sha,
        "checks": checks,
        "official_label_allowed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-promotable", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = (args.out_dir or repo / "experiments/kbound/results/official_repro_v1").resolve()
    output = (args.output or out_dir / "OFFICIAL_BASELINE_AUDIT.json").resolve()
    payload = {
        "schema_version": 2,
        "provenance_path_binding": {
            "schema": PATH_BINDING_SCHEMA,
            "root": ".",
            "root_role": "git_repository_root",
            "content_scope": "working_tree_at_generation",
            "generation_base_git_head": git_output(repo, "rev-parse", "HEAD"),
        },
        "claim_rule": (
            "Label as official implementation under a protocol adapter only when "
            "official_label_allowed is true. Otherwise retain protocol-matched port."
        ),
        "methods": {
            "aetta": audit_aetta(repo, out_dir),
            "poem": audit_poem(repo, out_dir),
        },
    }
    payload["all_promotable"] = all(
        method["official_label_allowed"] for method in payload["methods"].values()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, result in payload["methods"].items():
        failed = [key for key, value in result["checks"].items() if not value]
        print(f"{name}: {'PROMOTABLE' if result['official_label_allowed'] else 'PORT ONLY'}")
        if failed:
            print(f"  failed checks: {', '.join(failed)}")
    print(f"wrote {output}")
    return 0 if payload["all_promotable"] or not args.require_promotable else 2


if __name__ == "__main__":
    raise SystemExit(main())
