from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/research/kbound/scripts/audit_independent_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_audit", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_distinct_checkpoint_hashes_pass(tmp_path: Path) -> None:
    for seed in range(3):
        (tmp_path / f"model_seed{seed}.pt").write_bytes(f"seed={seed}".encode())
    result = MODULE.audit(str(tmp_path / "model_seed{seed}.pt"), [0, 1, 2])
    assert result["status"] == "PASS"
    assert result["all_hashes_distinct"]


def test_relabelled_single_checkpoint_fails(tmp_path: Path) -> None:
    for seed in range(2):
        (tmp_path / f"model_seed{seed}.pt").write_bytes(b"same checkpoint")
    with pytest.raises(ValueError, match="distinct checkpoint bytes"):
        MODULE.audit(str(tmp_path / "model_seed{seed}.pt"), [0, 1])


def test_missing_seed_fails(tmp_path: Path) -> None:
    (tmp_path / "model_seed0.pt").write_bytes(b"seed zero")
    with pytest.raises(FileNotFoundError, match="model-seed 1"):
        MODULE.audit(str(tmp_path / "model_seed{seed}.pt"), [0, 1])
