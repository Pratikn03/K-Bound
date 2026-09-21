"""Exact, versioned reconciliation of a helper-only policy source change.

The saved v3 result continues to name its original whole-file source. This
attestation only permits decide_batch consumers to use the separately pinned
current file; it never rewrites results, authorizes new hashes, or certifies
hierarchical selection. A future source change requires a new reviewed version.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

HISTORICAL_PATH = "docs/research/kbound/release/policy_reconciliation_v1/policy_original.txt"
HISTORICAL_SHA256 = "baad38d43f2eda54c701f9c1ab0725c953bf076cadc6ec6eec151526377b0696"
CURRENT_SHA256 = "93152a5b1e8c66d392b9e09b3f28e490591da8f187497d1ef5a4826d5f0d4fce"
CROSSFIT_SHA256 = "a82b16337e12f469f09b732caaa7c8305ff05edac8713fe2f4b2d12ff33ec5ac"


def _bound_bytes(root: Path, relative: str, expected: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"policy reconciliation hash mismatch: {relative}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"policy reconciliation hash mismatch: {relative}")
    return data


def _replay_ast(data: bytes) -> tuple[list[str], str]:
    tree = ast.parse(data)
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "decide_hierarchical_candidates")
    ]
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    if any(isinstance(node, ast.Name) and node.id == "decide_hierarchical_candidates" for node in ast.walk(tree)):
        raise ValueError("replay references the changed hierarchical helper")
    return sorted(ast.dump(node) for node in imports), ast.dump(tree)


def verify_policy_reconciliation(root: Path) -> dict[str, str]:
    old = _bound_bytes(root, HISTORICAL_PATH, HISTORICAL_SHA256)
    new = _bound_bytes(root, "kga/policy.py", CURRENT_SHA256)
    crossfit = _bound_bytes(root, "kga/crossfit.py", CROSSFIT_SHA256)
    if _replay_ast(old) != _replay_ast(new):
        raise ValueError("policy replay AST changed outside the reviewed helper")
    if b"decide_hierarchical_candidates" in crossfit:
        raise ValueError("crossfit references the changed helper")
    return {
        "schema": "kbound-policy-source-reconciliation-v1",
        "scope": "decide_batch replay only",
        "historical_path": HISTORICAL_PATH,
        "historical_sha256": HISTORICAL_SHA256,
        "current_sha256": CURRENT_SHA256,
        "crossfit_sha256": CROSSFIT_SHA256,
    }


def accepts_historical_policy(root: Path, name: str, binding: dict) -> bool:
    if name != "policy" or binding != {"path": "kga/policy.py", "sha256": HISTORICAL_SHA256}:
        return False
    verify_policy_reconciliation(root)
    return True
