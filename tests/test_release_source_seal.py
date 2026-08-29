from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_source_seal_binds_head_tree_and_rejects_dirty_maintained_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    maintained = tmp_path / "maintained.txt"
    generated = tmp_path / "generated.json"
    maintained.write_text("source\n", encoding="utf-8")
    generated.write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "maintained.txt", "generated.json")
    _git(tmp_path, "commit", "-qm", "source freeze")
    head = _git(tmp_path, "rev-parse", "HEAD")

    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"test_source": ("maintained.txt",)})
    monkeypatch.setattr(
        seal,
        "GENERATED_OUTPUT_ALLOWLIST",
        frozenset({"generated.json", "release_seal.json"}),
    )

    payload = seal.build_payload(tmp_path, head)
    assert payload["source_commit"] == head
    assert payload["source_tree"] == _git(tmp_path, "rev-parse", "HEAD^{tree}")
    assert payload["sealed_artifact_count"] == 1
    assert payload["artifacts"][0]["path"] == "maintained.txt"
    assert payload["artifacts"][0]["git_blob"]

    generated.write_text("v2\n", encoding="utf-8")
    assert seal.build_payload(tmp_path, head)["artifacts"] == payload["artifacts"]
    seal_path = tmp_path / "release_seal.json"
    seal._write(seal_path, payload)
    assert seal.validate_seal(tmp_path, seal_path) == payload

    maintained.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(tmp_path, head)


def test_source_seal_rejects_non_head_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    path = tmp_path / "maintained.txt"
    path.write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "maintained.txt")
    _git(tmp_path, "commit", "-qm", "one")
    old = _git(tmp_path, "rev-parse", "HEAD")
    path.write_text("two\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "two")
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"test_source": ("maintained.txt",)})
    with pytest.raises(ValueError, match="source commit must equal HEAD"):
        seal.build_payload(tmp_path, old)


def test_tree_enumeration_does_not_request_unrelated_blob_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    oid = "a" * 40

    def fake_git_bytes(*args: str, repo: Path) -> bytes:
        assert repo == tmp_path
        calls.append(args)
        return f"100644 blob {oid}\tmaintained.txt\0".encode("ascii")

    monkeypatch.setattr(seal, "_git_bytes", fake_git_bytes)

    assert seal._tree_blobs(tmp_path, "source-commit") == {"maintained.txt": oid}
    assert calls == [("ls-tree", "-r", "-z", "source-commit")]


def test_source_seal_verifies_checkout_without_materializing_loose_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    maintained = tmp_path / "maintained.txt"
    unrelated = tmp_path / "unrelated.bin"
    maintained.write_bytes(b"sealed source\n")
    unrelated.write_bytes(b"unrelated object\n")
    _git(tmp_path, "add", "maintained.txt", "unrelated.bin")
    _git(tmp_path, "commit", "-qm", "source freeze")
    head = _git(tmp_path, "rev-parse", "HEAD")

    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"test_source": ("maintained.txt",)})
    monkeypatch.setattr(seal, "GENERATED_OUTPUT_ALLOWLIST", frozenset())

    def loose_object(oid: str) -> Path:
        return tmp_path / ".git" / "objects" / oid[:2] / oid[2:]

    unrelated_oid = _git(tmp_path, "rev-parse", "HEAD:unrelated.bin")
    loose_object(unrelated_oid).unlink()
    maintained_oid = _git(tmp_path, "rev-parse", "HEAD:maintained.txt")
    loose_object(maintained_oid).unlink()

    payload = seal.build_payload(tmp_path, head)
    assert payload["sealed_artifact_count"] == 1
    assert payload["artifacts"][0]["git_blob"] == maintained_oid
    assert payload["artifacts"][0]["bytes"] == len(b"sealed source\n")

    maintained.write_bytes(b"different bytes\n")
    with pytest.raises(ValueError, match="checked-out bytes do not match source commit"):
        seal._artifact_rows(tmp_path, head)


def test_release_generated_authorities_are_outer_checksum_outputs() -> None:
    generated = {
        "docs/research/kbound/claim_ledger.json",
        "docs/research/kbound/RESULT_MANIFEST.json",
        "docs/research/kbound/results_source.json",
        "docs/research/kbound/STORAGE_MANIFEST.json",
        "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
        "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json",
        "experiments/kbound/results/reconciled_panels_v1/source_manifest.json",
    }
    source_inventory = {
        path for paths in seal.EXPLICIT_FILES.values() for path in paths
    }
    assert generated.isdisjoint(source_inventory)
    assert generated <= seal.GENERATED_OUTPUT_ALLOWLIST


def test_direct_release_scripts_are_explicitly_source_sealed() -> None:
    required = {
        "docs/research/kbound/scripts/audit_natural_target_provenance.py",
        "docs/research/kbound/scripts/audit_official_baselines.py",
        "docs/research/kbound/scripts/plot_canonical_decision_frontier.py",
        "docs/research/kbound/scripts/plot_conceptual_regime_geometry.py",
        "docs/research/kbound/scripts/render_pdf_pages.py",
        "docs/research/kbound/scripts/run_frontier_kga_bridge.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
    }
    assert required <= set(seal.EXPLICIT_FILES["release_code"])
