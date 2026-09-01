from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
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


def _small_source_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release@example.invalid")
    (repo / "maintained.txt").write_text("source\n", encoding="utf-8")
    (repo / "generated.json").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "maintained.txt", "generated.json")
    _git(repo, "commit", "-qm", "source freeze")
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"test_source": ("maintained.txt",)})
    monkeypatch.setattr(
        seal, "GENERATED_OUTPUT_ALLOWLIST", frozenset({"generated.json", "release_seal.json"})
    )
    return repo, _git(repo, "rev-parse", "HEAD")


def test_clean_start_rejects_even_allowlisted_generated_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    assert seal.build_payload(repo, source, require_clean=True)["source_commit"] == source
    (repo / "generated.json").write_text("new output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completely clean working tree"):
        seal.build_payload(repo, source, require_clean=True)
    # Outputs are allowed only after the clean source has been pinned.
    assert seal.build_payload(repo, source)["source_commit"] == source


def test_phase_check_rejects_source_change_and_commit_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    (repo / "maintained.txt").write_text("changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(repo, source)
    _git(repo, "add", "maintained.txt")
    _git(repo, "commit", "-qm", "source changed during release")
    with pytest.raises(ValueError, match="source commit must equal HEAD"):
        seal.build_payload(repo, source)


def test_source_check_rejects_head_change_while_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    original_rows = seal._artifact_rows

    def changed_head(repo: Path, commit: str) -> list[dict[str, object]]:
        rows = original_rows(repo, commit)
        _git(repo, "commit", "--allow-empty", "-qm", "HEAD advanced while hashing")
        return rows

    monkeypatch.setattr(seal, "_artifact_rows", changed_head)
    with pytest.raises(ValueError, match="HEAD changed during the release source check"):
        seal.build_payload(repo, source)


def test_source_artifact_reader_rejects_symlinked_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_bytes(b"external source\n")
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"source": ("linked/source.py",)})
    monkeypatch.setattr(seal, "_tree_blobs", lambda *args: {"linked/source.py": "a" * 40})
    with pytest.raises(FileNotFoundError, match="missing or a symlink"):
        seal._artifact_rows(repo, source)


def test_final_seal_remains_valid_after_generated_artifact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    payload = seal.build_payload(repo, source, require_clean=True)
    path = repo / "release_seal.json"
    seal._write(path, payload)
    (repo / "generated.json").write_text("release output\n", encoding="utf-8")
    _git(repo, "add", "generated.json", "release_seal.json")
    _git(repo, "commit", "-qm", "generated release artifacts")
    assert _git(repo, "rev-parse", "HEAD") != source
    assert seal.validate_seal(repo, path) == payload
    assert payload["source_commit"] == source


def test_final_seal_rejects_committed_non_output_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    path = repo / "release_seal.json"
    seal._write(path, seal.build_payload(repo, source))
    (repo / "other_source.py").write_text("changed = True\n", encoding="utf-8")
    _git(repo, "add", "other_source.py", "release_seal.json")
    _git(repo, "commit", "-qm", "not an artifact-only commit")
    with pytest.raises(ValueError, match="committed changes since the sealed source"):
        seal.validate_seal(repo, path)


def test_final_seal_rejects_mutable_source_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    payload = seal.build_payload(repo, source)
    payload["source_commit"] = "HEAD"
    path = repo / "release_seal.json"
    seal._write(path, payload)
    with pytest.raises(ValueError, match="full immutable commit ID"):
        seal.validate_seal(repo, path)


def test_formal_pins_and_selected_validation_files_are_source_sealed() -> None:
    prefix = "docs/research/kbound/formal/"
    required = {
        prefix + name
        for name in (
            "KBound.lean", "README.md", "build.sh", "formal_audit.py",
            "lakefile.lean", "lake-manifest.json", "lean-toolchain",
        )
    }
    assert required <= set(seal.EXPLICIT_FILES["formal_source"])
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text()
    executed: set[str] = set()
    for line in runbook.replace("\\\n", " ").splitlines():
        if not line.lstrip().startswith('"$PY" -m pytest'):
            continue
        tokens = shlex.split(line, comments=True)
        if "--collect-only" not in tokens:
            executed.update(t for t in tokens if t.startswith("tests/") and t.endswith(".py"))
    assert executed <= set(seal.EXPLICIT_FILES["release_validation"])
    assert {
        "tests/test_kbound_formal_audit.py", "tests/test_kga_masked_inputs.py",
        "tests/test_kbound_current_policy_bindings.py",
    } <= executed


def test_source_prefix_inventory_excludes_caches_reports_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wanted = {
        "docs/research/kbound/formal/KBound/Probability/NewProof.lean",
        "docs/research/kbound/kbound_repro/release_checks.py",
        "docs/research/kbound/kbound_repro/tests/test_authority.py",
        "docs/research/kbound/kbound_pkg/kbound/certificate.py",
        "docs/research/kbound/kbound_pkg/tests/test_certificate.py",
        "docs/research/kbound/edge/src/kbound_edge/policy.py",
        "docs/research/kbound/edge/tests/test_policy.py",
        "docs/research/kbound/tests/test_protocol.py",
        "kga/_validation.py",
    }
    excluded = {
        "docs/research/kbound/formal/.lake/packages/mathlib/Mathlib.lean",
        "docs/research/kbound/formal/KBound/build/Generated.lean",
        "docs/research/kbound/formal/formal_audit_report.json",
        "docs/research/kbound/kbound_pkg/build/lib/kbound/certificate.py",
        "docs/research/kbound/kbound_repro/__pycache__/stale.py",
        "docs/research/kbound/data/raw.py",
        "experiments/kbound/results/old_history.py",
    }
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {})
    monkeypatch.setattr(seal, "_tree_blobs", lambda *args: {p: "a" * 40 for p in wanted | excluded})
    inventory = {path for _, path in seal._inventory(tmp_path, "source")}
    assert inventory == wanted


def test_generated_formal_receipt_is_output_not_source() -> None:
    receipt = "docs/research/kbound/audits/formal_foundations_2026_08_31.json"
    assert receipt in seal.GENERATED_OUTPUT_ALLOWLIST
    assert receipt not in {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text()
    assert 'bash "$KB/formal/build.sh" --json-out "$REPO/$KB/audits/formal_foundations_2026_08_31.json"' in runbook


@pytest.mark.parametrize("change", ["dirty_start", "source_during_run", "head_during_run"])
def test_runbook_all_enforces_real_source_checks_before_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Exercise the real Bash gate and real seal checker without scientific work."""
    repo, _ = _small_source_repo(tmp_path, monkeypatch)
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text((seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text())
    _git(repo, "add", "docs/research/kbound/runbooks/release_candidate.sh")
    _git(repo, "commit", "-qm", "test runbook")
    event_log = tmp_path / "events.jsonl"
    fake_python = tmp_path / "test-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import importlib.util, json, pathlib, subprocess, sys\n"
        f"event_log = pathlib.Path({str(event_log)!r})\n"
        "args = sys.argv[1:]\n"
        "with event_log.open('a') as log: log.write(json.dumps(args) + '\\n')\n"
        "if args and args[0].endswith('build_release_source_seal.py'):\n"
        f"    spec = importlib.util.spec_from_file_location('release_seal', {seal.__file__!r})\n"
        "    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "    module.ROOT = pathlib.Path.cwd()\n"
        "    module.EXPLICIT_FILES = {'source': ('maintained.txt',)}\n"
        "    module.GENERATED_OUTPUT_ALLOWLIST = frozenset({'generated.json', 'release_seal.json'})\n"
        "    sys.argv = args\n"
        "    raise SystemExit(module.main())\n"
        f"change = {change!r}\n"
        "if args == ['-'] and change != 'dirty_start':\n"
        "    source = pathlib.Path('maintained.txt')\n"
        "    if source.read_text() == 'source\\n':\n"
        "        source.write_text('changed during phase\\n')\n"
        "        if change == 'head_during_run':\n"
        "            subprocess.run(['git', 'add', 'maintained.txt'], check=True)\n"
        "            subprocess.run(['git', 'commit', '-qm', 'mid-run source change'], check=True)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('unexpected scientific command reached')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    if change == "dirty_start":
        (repo / "generated.json").write_text("uncommitted baseline\n")
    environment = dict(os.environ, KBOUND_PYTHON=str(fake_python), PYTHONDONTWRITEBYTECODE="1")
    environment.pop("KBOUND_SOURCE_COMMIT", None)
    environment.pop("RELEASE_SOURCE_COMMIT", None)
    completed = subprocess.run(
        ["bash", str(runbook), "all"], cwd=repo, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert "--preflight" in events[0]
    combined_output = completed.stdout + completed.stderr
    if change == "dirty_start":
        assert len(events) == 1
        assert "completely clean working tree" in combined_output
    else:
        assert any("--check-source" in event for event in events)
        expected = "source commit must equal HEAD" if change == "head_during_run" else "maintained release-source paths are dirty"
        assert expected in combined_output
        assert all(event == ["-"] or event[0].endswith("build_release_source_seal.py") for event in events)
