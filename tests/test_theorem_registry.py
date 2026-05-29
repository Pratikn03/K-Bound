"""Tests for the central theorem registry."""

from __future__ import annotations

from pathlib import Path

from elara.theory.theorem_registry import THEOREM_REGISTRY, artifact_status, list_theorems


def test_registry_covers_t1_through_t7_and_gdr():
    ids = [spec.theorem_id for spec in list_theorems()]
    assert ids == ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "GDR"]


def test_every_theorem_has_core_modules_except_t1_only_one():
    for spec in THEOREM_REGISTRY.values():
        assert spec.core_modules


def test_t3_artifacts_exist_in_repo():
    root = Path(__file__).resolve().parents[1]
    status = artifact_status(root, THEOREM_REGISTRY["T3"])
    assert status["experiments/fusion/craf_real_k_domain_results.json"]
