from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_release_source_seal as seal
from docs.research.kbound.scripts import verify_release_checksums as checksums


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _first_artifact(payload: dict[str, object]) -> dict[str, object]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    assert artifacts
    artifact = artifacts[0]
    assert isinstance(artifact, dict)
    return artifact


def _empty_structural_seal(marker: str = "a") -> dict[str, object]:
    rows: list[dict[str, object]] = []
    return {
        "schema_version": seal.SCHEMA,
        "source_commit": marker * 40,
        "source_tree": marker * 40,
        "working_tree_gate": "fixture",
        "sealed_artifact_count": 0,
        "artifacts_sha256": hashlib.sha256(b"[]").hexdigest(),
        "artifacts": rows,
        "exclusions": [],
    }


def test_structural_source_seal_check_is_environment_and_checkout_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "release-seal.json"
    path.write_bytes(seal._seal_bytes(_empty_structural_seal()))
    monkeypatch.setattr(
        seal,
        "verify_release_python_content",
        lambda: (_ for _ in ()).throw(AssertionError("structural check must not inspect the runtime")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_release_source_seal.py", "--check-structure", "--output", str(path)],
    )

    assert seal.main() == 0
    assert "structure: PASS" in capsys.readouterr().out

    malformed = dict(_empty_structural_seal())
    malformed["artifacts_sha256"] = "0" * 64
    path.write_bytes(seal._seal_bytes(malformed))
    with pytest.raises(ValueError, match="aggregate hash"):
        seal.validate_seal_structure(path)


def test_portable_source_seal_check_keeps_checkout_binding_but_skips_build_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "release-seal.json"
    path.write_bytes(seal._seal_bytes(_empty_structural_seal()))
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        seal,
        "verify_release_python_content",
        lambda: (_ for _ in ()).throw(AssertionError("portable verification must not inspect the build runtime")),
    )
    monkeypatch.setattr(
        seal,
        "validate_seal",
        lambda repo, output: observed.append((repo, output)) or _empty_structural_seal(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_release_source_seal.py", "--check-portable", "--output", str(path)],
    )

    assert seal.main() == 0
    assert observed == [(seal.ROOT, path.absolute())]


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
    artifact = _first_artifact(payload)
    assert artifact["path"] == "maintained.txt"
    assert artifact["git_blob"]

    generated.write_text("v2\n", encoding="utf-8")
    assert seal.build_payload(tmp_path, head)["artifacts"] == payload["artifacts"]
    seal_path = tmp_path / "release_seal.json"
    seal._write(seal_path, payload)
    assert seal.validate_seal(tmp_path, seal_path) == payload

    maintained.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(tmp_path, head)


def test_source_seal_rejects_non_head_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_source_seal_rejects_duplicate_and_nonfinite_json_before_git(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":"x","schema_version":"y"}\n')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        seal.validate_seal(tmp_path, duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"schema_version":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        seal.validate_seal(tmp_path, nonfinite)


def test_tree_enumeration_uses_positive_source_scopes_and_never_requests_blob_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []
    oid = "a" * 40

    def fake_git_bytes(*args: str, repo: Path) -> bytes:
        assert repo == tmp_path
        calls.append(args)
        return f"100644 blob {oid}\tmaintained.txt\0".encode("ascii")

    monkeypatch.setattr(seal, "_git_bytes", fake_git_bytes)

    assert seal._tree_blobs(
        tmp_path,
        "source-commit",
        pathspecs=("maintained.txt",),
    ) == {"maintained.txt": oid}
    assert calls == [
        (
            "ls-tree",
            "-r",
            "-z",
            "source-commit",
            "--",
            "maintained.txt",
        )
    ]


def test_default_source_tree_scopes_are_positive_and_exclude_protected_artifacts() -> None:
    pathspecs = seal._source_tree_pathspecs()

    assert pathspecs
    assert "." not in pathspecs
    assert not any("so2sat" in path.casefold() for path in pathspecs)


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
    artifact = _first_artifact(payload)
    assert artifact["git_blob"] == maintained_oid
    assert artifact["bytes"] == len(b"sealed source\n")

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
    source_inventory = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    assert generated.isdisjoint(source_inventory)
    assert generated <= seal.GENERATED_OUTPUT_ALLOWLIST


def test_required_frontier_release_artifact_is_not_gitignored() -> None:
    """A fresh release checkout must be able to stage the recovered authority."""

    relative = "experiments/kbound/frontier_sweep_v1/decision_value_results.json"
    assert relative in seal.GENERATED_OUTPUT_ALLOWLIST
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=seal.ROOT,
        check=False,
    )
    assert result.returncode == 1, f"{relative} is a required release artifact but is hidden by .gitignore"


@pytest.mark.parametrize(
    "relative",
    [
        "experiments/kbound/frontier_sweep_v1/raw_prediction_rows.csv",
        "experiments/kbound/frontier_sweep_v1/raw_scores.npy",
        "experiments/kbound/frontier_sweep_v1/model_checkpoint.pt",
    ],
)
def test_frontier_release_exception_keeps_neighboring_raw_data_ignored(relative: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=seal.ROOT,
        check=False,
    )
    assert result.returncode == 0, f"neighboring raw artifact escaped .gitignore: {relative}"


def test_cct_release_products_are_outer_artifacts_not_source_inventory() -> None:
    generated = {
        "docs/research/kbound/paper/generated/cct20_release_manifest.json",
        "docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json",
        "docs/research/kbound/paper/generated/cct20_numbers.tex",
        "docs/research/kbound/paper/generated/cct20_primary_table.tex",
        "docs/research/kbound/paper/generated/cct20_location_effects.tex",
    }
    source_inventory = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    assert generated.isdisjoint(source_inventory)
    assert generated <= seal.GENERATED_OUTPUT_ALLOWLIST
    assert generated <= set(checksums.REQUIRED_RELEASE_PATHS)


def test_default_source_seal_does_not_preauthorize_so2sat_gate_or_target_material() -> None:
    """The default source seal must remain safe before explicit authorization."""

    explicit = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    assert not any("so2sat" in path.casefold() for path in explicit)
    assert not any("so2sat" in prefix.casefold() for _, prefix, _ in seal.SOURCE_PREFIX_RULES)


def test_direct_release_scripts_are_explicitly_source_sealed() -> None:
    required = {
        "docs/research/kbound/scripts/audit_natural_target_provenance.py",
        "docs/research/kbound/scripts/audit_official_baselines.py",
        "docs/research/kbound/scripts/plot_canonical_decision_frontier.py",
        "docs/research/kbound/scripts/plot_conceptual_regime_geometry.py",
        "docs/research/kbound/scripts/render_pdf_pages.py",
        "docs/research/kbound/scripts/run_frontier_kga_bridge.py",
        "docs/research/kbound/scripts/validate_closure_protocol.py",
        "docs/research/kbound/scripts/verify_python_environment.py",
    }
    assert required <= set(seal.EXPLICIT_FILES["release_code"])


def test_public_release_builders_and_privacy_tests_are_source_sealed() -> None:
    assert {
        "docs/research/kbound/scripts/build_cct20_public_bundle.py",
        "docs/research/kbound/scripts/build_anonymous_supplement.py",
        "docs/research/kbound/scripts/release_privacy.py",
    } <= set(seal.EXPLICIT_FILES["release_code"])
    assert {
        "tests/test_cct20_public_bundle.py",
        "tests/test_anonymous_supplement.py",
        "tests/test_release_privacy.py",
    } <= set(seal.EXPLICIT_FILES["release_validation"])
    assert {
        "docs/research/kbound/release/cct20_public_evidence_bundle.zip",
        "docs/research/kbound/release/kbound_anonymous_supplement.zip",
    } <= seal.GENERATED_OUTPUT_ALLOWLIST


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
    monkeypatch.setattr(seal, "GENERATED_OUTPUT_ALLOWLIST", frozenset({"generated.json", "release_seal.json"}))
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


def test_source_seal_write_is_atomic_and_rejects_symlink_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    payload = seal.build_payload(repo, source)
    target = repo / "release_seal.json"
    target.write_bytes(b"original\n")
    linked = repo / "linked-seal.json"
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        seal._write(linked, payload)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(seal.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        seal._write(target, payload)
    assert target.read_bytes() == b"original\n"
    assert not list(repo.glob(".release_seal.json.*.tmp"))


def test_source_seal_cli_does_not_resolve_away_output_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    payload = seal.build_payload(repo, source)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"sentinel\n")
    linked = tmp_path / "linked.json"
    linked.symlink_to(outside)
    monkeypatch.setattr(seal, "build_payload", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(seal, "verify_release_python_content", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_release_source_seal.py", "--source-commit", source, "--output", str(linked)],
    )

    with pytest.raises(ValueError, match="symlink"):
        seal.main()

    assert outside.read_bytes() == b"sentinel\n"


def test_source_seal_validator_rejects_a_symlinked_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    target = repo / "release_seal.json"
    seal._write(target, seal.build_payload(repo, source))
    linked = tmp_path / "linked-seal.json"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        seal.validate_seal(repo, linked)


def test_phase_check_rejects_source_change_and_commit_advance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    (repo / "maintained.txt").write_text("changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(repo, source)
    _git(repo, "add", "maintained.txt")
    _git(repo, "commit", "-qm", "source changed during release")
    with pytest.raises(ValueError, match="source commit must equal HEAD"):
        seal.build_payload(repo, source)


def test_source_check_rejects_head_change_while_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    original_rows = seal._artifact_rows

    def changed_head(repo: Path, commit: str) -> list[dict[str, object]]:
        rows = original_rows(repo, commit)
        _git(repo, "commit", "--allow-empty", "-qm", "HEAD advanced while hashing")
        return rows

    monkeypatch.setattr(seal, "_artifact_rows", changed_head)
    with pytest.raises(ValueError, match="HEAD changed during the release source check"):
        seal.build_payload(repo, source)


def test_source_artifact_reader_rejects_symlinked_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_final_seal_rejects_committed_non_output_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    path = repo / "release_seal.json"
    seal._write(path, seal.build_payload(repo, source))
    (repo / "other_source.py").write_text("changed = True\n", encoding="utf-8")
    _git(repo, "add", "other_source.py", "release_seal.json")
    _git(repo, "commit", "-qm", "not an artifact-only commit")
    with pytest.raises(ValueError, match="committed changes since the sealed source"):
        seal.validate_seal(repo, path)


def test_final_seal_rejects_mutable_source_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, source = _small_source_repo(tmp_path, monkeypatch)
    payload = seal.build_payload(repo, source)
    payload["source_commit"] = "HEAD"
    path = repo / "release_seal.json"
    path.write_bytes(seal._seal_bytes(payload))
    with pytest.raises(ValueError, match="full immutable commit ID"):
        seal.validate_seal(repo, path)


def test_formal_pins_and_selected_validation_files_are_source_sealed() -> None:
    from docs.research.kbound.scripts import run_repository_verification

    prefix = "docs/research/kbound/formal/"
    required = {
        prefix + name
        for name in (
            "KBound.lean",
            "README.md",
            "build.sh",
            "formal_audit.py",
            "lakefile.lean",
            "lake-manifest.json",
            "lean-toolchain",
        )
    }
    assert required <= set(seal.EXPLICIT_FILES["formal_source"])
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text()
    assert '"$KB/scripts/run_repository_verification.py"' in runbook
    required_validation = {
        "tests/test_kbound_formal_audit.py",
        "tests/test_kga_masked_inputs.py",
        "tests/test_kbound_current_policy_bindings.py",
    }
    discovered = run_repository_verification.classify_test_paths(required_validation)
    assert set(discovered["pytest_paths"]) == required_validation
    assert required_validation <= set(seal.EXPLICIT_FILES["release_validation"])


def test_task7_executable_surface_is_source_sealed() -> None:
    explicit = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    assert {
        ".pre-commit-config.yaml",
        ".python-version",
        "Dockerfile",
        "requirements-api-py311-linux.lock.txt",
        "requirements-ci-py312-linux.lock.txt",
        "requirements-release.txt",
        "requirements-release-macos-arm64.lock.txt",
        "docs/research/kbound/dashboard/package.json",
        "docs/research/kbound/dashboard/package-lock.json",
        "docs/research/kbound/dashboard/tsconfig.json",
        "docs/research/multiclass_vector_capacity/formal/lake-manifest.json",
        "docs/research/multiclass_vector_capacity/formal/lean-toolchain",
    } <= explicit

    rules = {(category, prefix, suffixes) for category, prefix, suffixes in seal.SOURCE_PREFIX_RULES}
    assert (
        "release_validation",
        "tests/",
        (".py",),
    ) in rules
    assert (
        "release_validation",
        "experiments/kbound/theory_validation/",
        (".py",),
    ) in rules
    assert (
        "formal_source",
        "docs/research/multiclass_vector_capacity/formal/",
        (".lean", ".py"),
    ) in rules


def test_anonymous_bibliography_is_source_bound() -> None:
    bibliography = "docs/research/kbound/paper/references/refs.bib"
    assert bibliography in seal.EXPLICIT_FILES["paper_source"]


def test_all_four_current_manuscript_roles_and_shared_inputs_are_source_bound() -> None:
    required_paper_sources = {
        "docs/research/kbound/kbound_short_main.tex",
        "docs/research/kbound/kbound_short_supplement.tex",
        "docs/research/kbound/kbound_tmlr.tex",
        "docs/research/kbound/kbound_full_report.tex",
        "docs/research/kbound/kbound_abstract_core.tex",
        "docs/research/kbound/kbound_submission_body.tex",
        "docs/research/kbound/kbound_submission_supplement.tex",
        "docs/research/kbound/paper/full_report/evidence_protocol_atlas.tex",
        "docs/research/kbound/paper/full_report/formal_reproducibility_atlas.tex",
        "docs/research/kbound/paper/full_report/theory_exposition.tex",
        "docs/research/kbound/kbound_full_report_extensions.tex",
        "docs/research/kbound/paper/figures/decision_flow.tex",
        "docs/research/kbound/paper/generated/cct20_reporting_numbers.tex",
    }
    required_release_code = {
        "docs/research/kbound/scripts/audit_current_kbound_release.py",
        "docs/research/kbound/scripts/audit_tmlr_anonymity.py",
        "docs/research/kbound/scripts/compare_pdf_renders.py",
        "docs/research/kbound/scripts/empirical_closure.py",
        "docs/research/kbound/scripts/generate_release_identity.py",
        "docs/research/kbound/scripts/verify_pdf_structure.py",
    }
    required_release_validation = {
        "tests/test_kbound_pdf_visual_verification.py",
        "tests/test_tmlr_anonymity_audit.py",
    }

    assert required_paper_sources <= set(seal.EXPLICIT_FILES["paper_source"])
    assert required_release_code <= set(seal.EXPLICIT_FILES["release_code"])
    assert required_release_validation <= set(seal.EXPLICIT_FILES["release_validation"])


def test_current_release_guidance_and_table_crosswalk_are_source_bound() -> None:
    required_configuration = {
        "docs/research/kbound/CIFAR10C_SAR_QUARANTINE.md",
        "docs/research/kbound/G8_EXACTRANK_REGEN.md",
        "docs/research/kbound/KBOUND_SHORT_CLAIM_MANIFEST.md",
        "docs/research/kbound/KBOUND_SHORT_RESULT_AUDIT.md",
        "docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md",
        "docs/research/kbound/RELATED_WORK_POSITIONING.md",
        "docs/research/kbound/REPRODUCE.md",
        "docs/research/kbound/RELEASE_CHECKLIST.md",
        "docs/research/kbound/paper/generated/empirical_audit/claim_matrix.md",
        "docs/research/kbound/paper/RELEASE_TABLE_CROSSWALK.md",
    }

    assert required_configuration <= set(seal.EXPLICIT_FILES["configuration"])


def test_source_seal_allows_and_outer_checksums_bind_only_current_release_pdfs() -> None:
    expected = {
        "docs/research/kbound/release/current/kbound_short_main.pdf",
        "docs/research/kbound/release/current/kbound_short_supplement.pdf",
        "docs/research/kbound/release/current/kbound_tmlr.pdf",
        "docs/research/kbound/release/current/kbound_full_report.pdf",
    }
    allowed_current_documents = {
        path
        for path in seal.GENERATED_OUTPUT_ALLOWLIST
        if path.startswith("docs/research/kbound/release/current/") and path.endswith((".pdf", ".docx"))
    }

    assert allowed_current_documents == expected
    assert expected <= set(checksums.REQUIRED_RELEASE_PATHS)
    assert "docs/research/kbound/release/current/KBOUND_CURRENT_SHA256SUMS.txt" in seal.GENERATED_OUTPUT_ALLOWLIST
    assert not any(path.endswith(".docx") for path in checksums.REQUIRED_RELEASE_PATHS)

    nonrelease_build_outputs = {
        "docs/research/kbound/kbound_short_final_draft.pdf",
        "docs/research/kbound/kbound_short_final_draft.docx",
        "docs/research/kbound/kbound_tmlr.pdf",
        "docs/research/kbound/kbound_short_main.pdf",
        "docs/research/kbound/kbound_short_supplement.pdf",
        "docs/research/kbound/kbound_full_report.pdf",
        "output/pdf/kbound_short_main.pdf",
        "output/pdf/kbound_short_supplement.pdf",
        "output/pdf/kbound_tmlr.pdf",
        "output/pdf/kbound_full_report.pdf",
        "output/pdf/KBOUND_CURRENT_SHA256SUMS.txt",
    }
    assert nonrelease_build_outputs <= seal.GENERATED_OUTPUT_ALLOWLIST
    assert nonrelease_build_outputs.isdisjoint(checksums.REQUIRED_RELEASE_PATHS)


def test_post_commit_release_identity_is_outer_bound_not_source_sealed() -> None:
    identity_outputs = {
        "docs/research/kbound/paper/release/current_release.json",
        "docs/research/kbound/paper/generated/current_release_identity.tex",
    }
    source_inventory = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}

    assert identity_outputs.isdisjoint(source_inventory)
    assert identity_outputs <= seal.GENERATED_OUTPUT_ALLOWLIST
    assert identity_outputs <= set(checksums.REQUIRED_RELEASE_PATHS)


def test_real_source_inventory_has_one_category_per_path() -> None:
    rows = seal._inventory(seal.ROOT, "HEAD")
    paths = [path for _, path in rows]
    assert len(paths) == len(set(paths))
    assert (
        "formal_source",
        "docs/research/multiclass_vector_capacity/formal/lakefile.lean",
    ) in rows


def test_release_environment_receipt_is_lock_bound_sealed_and_checksums_required() -> None:
    receipt_relative = "docs/research/kbound/audits/python_environment_2026_09_02.json"
    lock_relative = "requirements-release-macos-arm64.lock.txt"
    receipt = json.loads((seal.ROOT / receipt_relative).read_text(encoding="utf-8"))
    lock_sha256 = hashlib.sha256((seal.ROOT / lock_relative).read_bytes()).hexdigest()
    explicit = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}

    assert receipt["status"] == "verified"
    assert receipt["runtime"] == {"python": "3.12.13", "platform": "macos-arm64"}
    assert receipt["lock"]["path"] == lock_relative
    assert receipt["lock"]["sha256"] == lock_sha256
    assert receipt_relative in explicit
    assert receipt_relative in checksums.REQUIRED_RELEASE_PATHS

    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    preflight = runbook.split("step_preflight() {", 1)[1].split("\n}", 1)[0]
    verifier = runbook.split("verify_release_python_environment() {", 1)[1].split("\n}", 1)[0]
    assert "verify_release_python_environment" in preflight
    assert "verify_python_environment.py" in verifier
    assert lock_relative in verifier
    assert receipt_relative.removeprefix("docs/research/kbound/") in verifier


def test_all_mode_regenerates_source_bound_results_before_validation() -> None:
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    all_mode = runbook.split("  all)\n", 1)[1].split("    ;;", 1)[0]
    assert all_mode.index("step_generate") < all_mode.index("step_validate_results")
    assert "check_release_source" in all_mode[all_mode.index("step_generate") : all_mode.index("step_validate_results")]


def test_only_explicit_deep_local_generation_refreshes_receipt_bound_cct_authority() -> None:
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    generation = runbook.split("step_generate() {", 1)[1].split("\n}", 1)[0]
    deep_generation = runbook.split("step_generate_deep_local_cct20() {", 1)[1].split("\n}", 1)[0]
    assert '"$KB/scripts/build_cct20_release.py" --refresh-from-existing' not in generation
    assert '"$KB/scripts/build_cct20_release.py" --refresh-from-existing' in deep_generation
    assert '--release-manifest "$KB/paper/generated/cct20_release_manifest.json"' in deep_generation
    assert '--generated-dir "$KB/paper/generated"' in deep_generation
    assert "${TMPDIR:-/tmp}/kbound-cct20-history.XXXXXX" in deep_generation
    assert '--history-dir "$cct_history_dir"' in deep_generation


def test_external_release_toolchain_is_profile_bound_sealed_and_checksums_required() -> None:
    receipt_relative = "docs/research/kbound/audits/release_toolchain_2026_09_02.json"
    profile_relative = "docs/research/kbound/release_toolchain_macos_arm64.json"
    receipt = json.loads((seal.ROOT / receipt_relative).read_text(encoding="utf-8"))
    profile_sha256 = hashlib.sha256((seal.ROOT / profile_relative).read_bytes()).hexdigest()
    explicit = {path for paths in seal.EXPLICIT_FILES.values() for path in paths}

    assert receipt["schema"] == "kbound-release-toolchain-receipt-v1"
    assert receipt["status"] == "verified"
    assert receipt["profile"] == {
        "path": "release_toolchain_macos_arm64.json",
        "sha256": profile_sha256,
    }
    assert receipt["runtime"]["zlib_compile"] == "1.2.12"
    assert receipt["runtime"]["zlib_runtime"] == "1.2.12"
    assert receipt_relative in explicit
    assert profile_relative in explicit
    assert receipt_relative in checksums.REQUIRED_RELEASE_PATHS

    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text(encoding="utf-8")
    preflight = runbook.split("step_preflight() {", 1)[1].split("\n}", 1)[0]
    verifier = runbook.split("verify_release_toolchain_for_phase() {", 1)[1].split("\n}", 1)[0]
    assert "verify_release_toolchain_for_phase" in preflight
    assert "verify_release_toolchain.py" in verifier
    active_profile = "docs/research/kbound/release_toolchain_macos_arm64_v2.json"
    active_receipt = "docs/research/kbound/audits/release_toolchain_2026_09_05_v2.json"
    assert active_profile in explicit
    assert active_receipt in explicit
    assert active_receipt in checksums.REQUIRED_RELEASE_PATHS
    assert {profile_relative, receipt_relative, active_profile, active_receipt}.isdisjoint(seal.GENERATED_OUTPUT_ALLOWLIST)
    assert active_profile.removeprefix("docs/research/kbound/") in verifier
    assert active_receipt.removeprefix("docs/research/kbound/") in verifier
    assert 'cmp -s "$verified_receipt" "$maintained_receipt"' in verifier
    assert '--output "$verified_receipt"' in verifier
    assert "--resolved-tools-output" in verifier


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
    monkeypatch.setattr(seal, "_tree_blobs", lambda *args: dict.fromkeys(wanted | excluded, "a" * 40))
    inventory = {path for _, path in seal._inventory(tmp_path, "source")}
    assert inventory == wanted


def test_protected_natural_shift_authorities_require_a_separate_authorized_seal() -> None:
    explicit = {path.casefold() for paths in seal.EXPLICIT_FILES.values() for path in paths}
    prefixes = {prefix.casefold() for _, prefix, _ in seal.SOURCE_PREFIX_RULES}
    generated = {path.casefold() for path in seal.GENERATED_OUTPUT_ALLOWLIST}
    checksummed = {path.casefold() for path in checksums.REQUIRED_RELEASE_PATHS}

    assert not any("so2sat" in path for path in explicit | prefixes | generated | checksummed)


def test_generated_formal_receipt_is_output_not_source() -> None:
    receipt = "docs/research/kbound/audits/formal_foundations_2026_08_31.json"
    assert receipt in seal.GENERATED_OUTPUT_ALLOWLIST
    assert receipt not in {path for paths in seal.EXPLICIT_FILES.values() for path in paths}
    runbook = (seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text()
    runner = (seal.ROOT / "docs/research/kbound/scripts/run_repository_verification.py").read_text()
    assert "--all-gates" in runbook
    assert "docs/research/kbound/formal/build.sh" in runner
    assert "formal_foundations_2026_08_31.json" in runner


@pytest.mark.parametrize("environment_ok", [True, False])
def test_source_seal_cli_verifies_exact_python_content_before_source_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_ok: bool
) -> None:
    calls: list[str] = []

    def verify_selected_profile(lock: Path, profile: Path) -> None:
        assert lock.name == "requirements-release-macos-arm64.lock.txt"
        assert profile.name == "release_python_environment_macos_arm64_v2.json"
        calls.append("environment")
        if not environment_ok:
            raise ValueError("synthetic Python content rejection")

    monkeypatch.setattr(seal.verify_python_environment, "verify_exact_content_profile", verify_selected_profile)

    def fixture_payload(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("source")
        return {
            "schema_version": seal.SCHEMA,
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "working_tree_gate": "fixture",
            "sealed_artifact_count": 0,
            "artifacts_sha256": hashlib.sha256(b"[]").hexdigest(),
            "artifacts": [],
            "exclusions": [],
        }

    monkeypatch.setattr(
        seal,
        "build_payload",
        fixture_payload,
    )
    monkeypatch.setattr(seal, "_write", lambda _path, _payload: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_release_source_seal.py", "--source-commit", "a" * 40, "--output", str(tmp_path / "seal.json")],
    )

    if environment_ok:
        assert seal.main() == 0
        assert calls == ["environment", "source"]
    else:
        with pytest.raises(ValueError, match="synthetic Python content rejection"):
            seal.main()
        assert calls == ["environment"]


@pytest.mark.parametrize("change", ["dirty_start", "source_during_run", "head_during_run"])
def test_runbook_all_enforces_real_source_checks_before_later_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Exercise the real Bash gate and real seal checker without scientific work."""
    repo, _ = _small_source_repo(tmp_path, monkeypatch)
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text((seal.ROOT / "docs/research/kbound/runbooks/release_candidate.sh").read_text())
    receipt_relative = "docs/research/kbound/audits/release_toolchain_2026_09_05_v2.json"
    receipt = repo / receipt_relative
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "verified"}, sort_keys=True) + "\n")
    _git(repo, "add", "docs/research/kbound/runbooks/release_candidate.sh", receipt_relative)
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
        "    module.verify_release_python_content = lambda: None\n"
        "    module.EXPLICIT_FILES = {'source': ('maintained.txt',)}\n"
        "    module.GENERATED_OUTPUT_ALLOWLIST = frozenset({'generated.json', 'release_seal.json', 'docs/research/kbound/audits/python_environment_2026_09_02.json'})\n"
        "    sys.argv = args\n"
        "    raise SystemExit(module.main())\n"
        "if args and args[0].endswith('verify_python_environment.py'):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text(json.dumps({'status': 'verified'}, sort_keys=True) + '\\n')\n"
        "    raise SystemExit(0)\n"
        "if args and args[0].endswith('verify_release_toolchain.py'):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text(json.dumps({'status': 'verified'}, sort_keys=True) + '\\n')\n"
        "    resolved = pathlib.Path(args[args.index('--resolved-tools-output') + 1])\n"
        "    executable = pathlib.Path(sys.argv[0]).resolve()\n"
        "    names = ('LATEXMK', 'LATEXPAND', 'PANDOC', 'PDFDETACH', 'PDFINFO', 'PDFLATEX', 'PDFTOPPM', 'PDFTOTEXT', 'PERL', 'SOFFICE')\n"
        "    resolved.write_text(''.join(f'KBOUND_TOOL_{name}\\t{executable}\\n' for name in names))\n"
        "    raise SystemExit(0)\n"
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
        ["bash", str(runbook), "all"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
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
        expected = (
            "source commit must equal HEAD"
            if change == "head_during_run"
            else "maintained release-source paths are dirty"
        )
        assert expected in combined_output
        assert all(
            event == ["-"]
            or event[0].endswith("build_release_source_seal.py")
            or event[0].endswith("verify_python_environment.py")
            or event[0].endswith("verify_release_toolchain.py")
            for event in events
        )


# Exact newly maintained inputs, not a broader generated-output exemption.
PUBLICATION_REVISION_INPUTS = {
    "paper_source": (
        "docs/research/kbound/kbound_main_with_appendix.tex",
        "docs/research/kbound/paper/generated/empirical_evidence_numbers.tex",
        "docs/research/kbound/paper/sections/officehome_mechanism_check.tex",
        "docs/research/kbound/paper/sections/proof_traceability.tex",
    ),
    "release_code": (
        "docs/research/kbound/scripts/audit_publication_package.py",
        "docs/research/kbound/scripts/empirical_macros.py",
        "docs/research/kbound/scripts/extract_macro_bundle.py",
        "docs/research/kbound/scripts/word_export.py",
        "docs/research/kbound/scripts/project_proof_receipts.py",
    ),
    "configuration": (
        "docs/research/kbound/paper/release/publication_package_v1.json",
        "docs/research/kbound/paper/release/manuscript_revision.json",
        "docs/research/kbound/formal/actual_fibre_radius_verification_20260907.portable.json",
        "docs/research/kbound/formal/actual_fibre_radius_followon_20260907.portable.json",
        "docs/research/kbound/formal/formal_full_report_bridges_audit_2026_09_08.json",
        "docs/research/kbound/formal/full_report_bridges_semantic_20260908.portable.json",
        "docs/research/kbound/formal/full_report_bridges_verification_20260908.portable.json",
        "docs/research/kbound/paper/release/empirical-evidence-values.json",
        "docs/research/kbound/paper/release/empirical_macro_inputs/manifest.json",
        "docs/research/kbound/paper/release/empirical_macro_inputs/entropy.json",
        "docs/research/kbound/paper/release/empirical_macro_inputs/bridge.json",
        "docs/research/kbound/paper/release/empirical_macro_inputs/officehome.json",
        "docs/research/kbound/paper/release/empirical_macro_inputs/smoke.json",
    ),
    "formal_source": tuple(
        "docs/research/kbound/formal/" + name
        for name in (
            "ActualFibreRadiusExamples.lean",
            "ActualWorldExamples.lean",
            "EvidenceTransportExamples.lean",
            "ExactConformalExamples.lean",
            "ExposureProxyExamples.lean",
            "FeatureRankExamples.lean",
            "HeadlineEvidenceExamples.lean",
            "InformationRefinementExamples.lean",
            "JointKernelExamples.lean",
            "PaperProofExamples.lean",
            "PaperSubclassExamples.lean",
            "RiskAlignmentExamples.lean",
            "WeightedHelpfulExamples.lean",
        )
    ),
}


@pytest.mark.parametrize(
    ("category", "relative"),
    [(category, path) for category, paths in PUBLICATION_REVISION_INPUTS.items() for path in paths],
)
def test_new_publication_input_binds_committed_bytes_and_rejects_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, category: str, relative: str
) -> None:
    """An omitted inventory entry must not silently pass as an empty seal."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic committed source\n")
    _git(tmp_path, "add", relative)
    _git(tmp_path, "commit", "-qm", "synthetic source")
    head = _git(tmp_path, "rev-parse", "HEAD")
    selected = tuple(path for path in seal.EXPLICIT_FILES.get(category, ()) if path == relative)
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {category: selected})
    monkeypatch.setattr(seal, "SOURCE_PREFIX_RULES", ())
    payload = seal.build_payload(tmp_path, head, require_clean=True)
    assert payload["sealed_artifact_count"] == 1
    assert payload["artifacts"][0]["path"] == relative
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(b"synthetic committed source\n").hexdigest()
    assert relative not in seal.GENERATED_OUTPUT_ALLOWLIST
    target.write_bytes(b"uncommitted modification must invalidate source\n")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(tmp_path, head)


def test_new_revision_authority_is_required_at_the_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "docs/research/kbound/paper/release/manuscript_revision.json"
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "synthetic empty source")
    head = _git(tmp_path, "rev-parse", "HEAD")
    selected = tuple(path for path in seal.EXPLICIT_FILES["configuration"] if path == relative)
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {"configuration": selected})
    monkeypatch.setattr(seal, "SOURCE_PREFIX_RULES", ())
    with pytest.raises(FileNotFoundError, match="required release-seal input is not tracked"):
        seal.build_payload(tmp_path, head)


def test_existing_prefixes_bind_new_tests_and_registered_proofs_without_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wanted = {
        "tests/test_kbound_empirical_macros.py",
        "tests/test_kbound_release_identity.py",
        "tests/test_kbound_manuscript_reconciliation.py",
        "docs/research/kbound/formal/KBound/Probability/ActualFibreRadius.lean",
        "docs/research/kbound/formal/KBound/Probability/RiskAlignment.lean",
        "docs/research/kbound/formal/KBound/WeightedHelpful.lean",
        "docs/research/kbound/formal/KBound/FeatureRank.lean",
    }
    excluded = {
        "docs/research/kbound/formal/.lake/packages/mathlib/NewProof.lean",
        "docs/research/kbound/formal/KBound/build/Generated.lean",
        "docs/research/kbound/formal/new_audit_receipt.json",
        "tests/test_so2sat_outcome.py",
        "experiments/kbound/results/private_outcomes.json",
    }
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {})
    monkeypatch.setattr(seal, "_tree_blobs", lambda *args: dict.fromkeys(wanted | excluded, "a" * 40))
    inventory = {path for _, path in seal._inventory(tmp_path, "synthetic source")}
    assert inventory == wanted


PORTABLE_DEPENDENCY_INPUTS = {
    "release_code": (
        "docs/research/kbound/scripts/project_historical_diagnostic_receipt.py",
        "docs/research/kbound/scripts/project_domainnet_stop.py",
    ),
    "configuration": (
        "experiments/audit/historical_diagnostic_recovery_receipt.portable.json",
        "experiments/audit/polarity_diagnostic_log.csv",
        "experiments/audit/canonical_label_semantics.json",
        "protocols/confirmatory_v2/DOMAINNET_DEV_PILOT_v1_STOP_PORTABLE.json",
    ),
}


@pytest.mark.parametrize(
    ("category", "relative"),
    [(category, path) for category, paths in PORTABLE_DEPENDENCY_INPUTS.items() for path in paths],
)
def test_portable_dependency_input_binds_committed_bytes_and_rejects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, category: str, relative: str
) -> None:
    """Missing inventory entries or dirty-source exemptions must not pass."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic portable dependency\n")
    _git(tmp_path, "add", relative)
    _git(tmp_path, "commit", "-qm", "synthetic portable dependency")
    head = _git(tmp_path, "rev-parse", "HEAD")
    selected = tuple(path for path in seal.EXPLICIT_FILES.get(category, ()) if path == relative)
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {category: selected})
    monkeypatch.setattr(seal, "SOURCE_PREFIX_RULES", ())
    payload = seal.build_payload(tmp_path, head, require_clean=True)
    assert payload["sealed_artifact_count"] == 1
    assert payload["artifacts"][0]["path"] == relative
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(b"synthetic portable dependency\n").hexdigest()
    assert relative not in seal.GENERATED_OUTPUT_ALLOWLIST
    target.write_bytes(b"mutated portable dependency\n")
    with pytest.raises(ValueError, match="maintained release-source paths are dirty"):
        seal.build_payload(tmp_path, head)


def test_existing_test_prefix_binds_portable_dependency_consumers_and_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wanted = {
        "tests/test_historical_diagnostic_recovery.py",
        "tests/test_project_historical_diagnostic_receipt.py",
        "tests/test_domainnet_pilot_data_v2.py",
        "tests/test_project_domainnet_stop.py",
        "tests/test_ci_portable_dependency_selectors.py",
    }
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release@example.invalid")
    for relative in wanted:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# synthetic test source\n")
    _git(tmp_path, "add", *sorted(wanted))
    _git(tmp_path, "commit", "-qm", "synthetic portable dependency tests")
    head = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(seal, "EXPLICIT_FILES", {})
    payload = seal.build_payload(tmp_path, head, require_clean=True)
    assert {row["path"] for row in payload["artifacts"]} == wanted
    assert payload["sealed_artifact_count"] == 5
