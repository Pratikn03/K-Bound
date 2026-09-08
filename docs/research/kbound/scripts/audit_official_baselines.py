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
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_baseline_provenance import (
    EXPECTED_CONDITION_COUNT, _canonical_json_sha256,
    decisions_sha256, is_native_trace_control_file, load_json_strict,
    validate_promotable_audit, verify_native_execution_attestation,
)
from official_decision_artifact import (
    SCHEMA_VERSION, STAGED, UNVERIFIED_LABEL, atomic_json, convert_native_decisions,
    stream_conditions, validate_decisions,
)

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
        "official_label_allowed": False,  # preliminary source/log census, not a schema-3 verdict
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
        # The pinned POEM native entry point is ImageNet-C-only. The existing
        # 432-condition CIFAR adapter is explicitly unavailable in Item11;
        # direct auditor calls must retain that blocker as well.
        "protocol_adapter_compatible": False,
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
        "official_label_allowed": False,  # preliminary source/log census, not a schema-3 verdict
    }


def _native_trace(repo: Path, root: Path, method: str, logs: dict[str, Any], source_tree: str | None) -> None:
    """Validate actual local control files; the strict validator checks the witness again."""
    logs.update(completion_verified=False, runner_receipt_verified=False, execution_attested=False)
    try:
        completion = load_json_strict(root / 'native_completion.json')
        runner = load_json_strict(root / 'native_runner_receipt.json')
        invocation = load_json_strict(root / 'native_invocation.json')
        attestation = load_json_strict(root / 'native_execution_attestation.json')
        logs.update(completion=completion, runner_receipt=runner, execution_attestation=attestation)
        if not all(isinstance(item, dict) for item in (completion, runner, invocation, attestation)):
            return
        source_path = 'AETTA' if method == 'aetta' else 'external/poem'
        producer_path = 'docs/research/kbound/scripts/run_official_native.py'
        producer = dict(path=producer_path, sha256=sha256_file(repo / producer_path))
        command = invocation.get('command')
        invocation_ok = (
            set(invocation) == {'command','method','producer','schema','source','working_directory'}
            and invocation.get('schema') == 'kbound-official-native-invocation-v1'
            and invocation.get('method') == method
            and invocation.get('working_directory') == source_path
            and invocation.get('source') == dict(path=source_path, tree_sha256=source_tree)
            and invocation.get('producer') == producer
            and isinstance(command, list) and bool(command)
            and all(isinstance(value, str) and value for value in command)
            and runner.get('producer') == producer
            and runner.get('invocation_sha256') == sha256_file(root / 'native_invocation.json')
        )
        artifact_hashes = {key:value for key,value in logs['sha256'].items() if not is_native_trace_control_file(key)}
        logs['completion_verified'] = bool(completion.get('log_sha256') == artifact_hashes)
        logs['runner_receipt_verified'] = bool(invocation_ok and runner.get('artifacts_sha256') == artifact_hashes
            and completion.get('runner_receipt_sha256') == _canonical_json_sha256(runner))
        logs['execution_attested'] = verify_native_execution_attestation(attestation, method=method,
            runner_receipt_sha256=_canonical_json_sha256(runner))
    except (OSError, ValueError, TypeError):
        return


def _bind_staged_method(repo, out, record, binding, conditions):
    method = record['method']
    source = repo / ('AETTA' if method == 'aetta' else 'external/poem')
    native = out / ('aetta_native' if method == 'aetta' else 'poem_imagenetc')
    logs = native_logs(native, repo=repo)
    _native_trace(repo, native, method, logs, tree_hash(source))
    record['native_logs'] = logs
    checks = record['checks']
    checks.pop('converted_decisions_nonempty', None)
    checks.update(native_logs_successful=logs['successful'], native_execution_attested=logs['execution_attested'],
        native_invocation_bound=logs['runner_receipt_verified'], converted_decisions_complete=False,
        locked_stream_bound=False, environment_receipt_bound=bool(binding.get('environment_receipt_sha256')),
        toolchain_receipt_bound=bool(binding.get('toolchain_receipt_sha256')))
    record.update(decision_count=0, decision_payload_sha256=None, official_label_allowed=False)
    try:
        stage_path = out / f'{method}_decisions.staged.json'
        stage = load_json_strict(stage_path)
        if (not isinstance(stage, dict) or stage.get('schema_version') != SCHEMA_VERSION
                or stage.get('method') != method or stage.get('status') != STAGED
                or stage.get('label') != UNVERIFIED_LABEL or stage.get('official_label_allowed') is not False):
            raise ValueError('unverified staged decision artifact required')
        decisions = validate_decisions(stage.get('decisions'), conditions)
        record.update(decision_count=len(decisions), decision_payload_sha256=decisions_sha256(decisions),
                      decisions_sha256=sha256_file(stage_path))
        source_log_hash = stage.get('source_log_sha256')
        candidates = [repo/path for path,digest in logs['sha256'].items()
                      if digest == source_log_hash and not is_native_trace_control_file(path)]
        if not candidates:
            raise ValueError('staged decisions do not bind an available native output')
        if convert_native_decisions(method, candidates[0]) != decisions:
            raise ValueError('staged decision payload differs from the native conversion rule')
        checks['converted_decisions_complete'] = len(decisions) == EXPECTED_CONDITION_COUNT
        checks['locked_stream_bound'] = (len(conditions) == EXPECTED_CONDITION_COUNT
            and stage.get('locked_stream_sha256') == binding.get('locked_stream_sha256'))
        record['source_log_sha256'] = source_log_hash
        record['official_label_allowed'] = all(value is True for value in checks.values())
        return decisions
    except (OSError, ValueError, TypeError) as exc:
        record['promotion_error'] = str(exc)
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-promotable", action="store_true")
    parser.add_argument('--locked-stream', type=Path)
    parser.add_argument('--environment-receipt', type=Path)
    parser.add_argument('--toolchain-receipt', type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = (args.out_dir or repo / "experiments/kbound/results/official_repro_v1").resolve()
    output = (args.output or out_dir / "OFFICIAL_BASELINE_AUDIT.json").resolve()
    binding = {}
    conditions = []
    for field, path in (('locked_stream_sha256', args.locked_stream),
                        ('environment_receipt_sha256', args.environment_receipt),
                        ('toolchain_receipt_sha256', args.toolchain_receipt)):
        if path is not None:
            if not isinstance(load_json_strict(path), dict):
                raise ValueError('binding inputs must be JSON objects')
            binding[field] = sha256_file(path)
    if args.locked_stream is not None:
        conditions = stream_conditions(args.locked_stream)
    binding['condition_count'] = len(conditions)
    payload = {
        "schema_version": SCHEMA_VERSION,
        'promotion_binding': binding,
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
    staged = {name:_bind_staged_method(repo, out_dir, record, binding, conditions)
              for name,record in payload['methods'].items()}
    # Validate the candidate audit using the unchanged strict witness validator
    # before publication. A failed required-official run never replaces output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.official-audit-check.', dir=output.parent) as temporary:
        candidate = Path(temporary) / 'audit.json'
        atomic_json(candidate, payload)
        for name, record in payload['methods'].items():
            if record['official_label_allowed']:
                try:
                    validate_promotable_audit(candidate, method=name, decisions=staged[name],
                        source_log_sha256=record['source_log_sha256'],
                        locked_stream_sha256=binding['locked_stream_sha256'],
                        environment_receipt_sha256=binding['environment_receipt_sha256'],
                        toolchain_receipt_sha256=binding['toolchain_receipt_sha256'])
                except (OSError, ValueError, KeyError) as exc:
                    record['official_label_allowed'] = False
                    record['promotion_error'] = str(exc)
    payload["all_promotable"] = all(
        method["official_label_allowed"] for method in payload["methods"].values()
    )
    for name, result in payload["methods"].items():
        failed = [key for key, value in result["checks"].items() if not value]
        print(f"{name}: {'PROMOTABLE' if result['official_label_allowed'] else 'PORT ONLY'}")
        if failed:
            print(f"  failed checks: {', '.join(failed)}")
    if args.require_promotable and not payload['all_promotable']:
        print('refusing to replace output: not every method is promotable', file=sys.stderr)
        return 2
    atomic_json(output, payload)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
