from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from docs.research.kbound.scripts import run_repository_verification as runner


def _committed_hook_config_repo(tmp_path: Path, content: str) -> tuple[Path, str]:
    repo = tmp_path / "hook-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    (repo / ".pre-commit-config.yaml").write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", ".pre-commit-config.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "hook configuration"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def test_test_inventory_discovers_active_modules_and_records_every_exclusion() -> None:
    paths = [
        "tests/test_unit.py",
        "pkg/feature_test.py",
        "docs/research/kbound/scripts/test_sar_faithful.py",
        "docs/research/kbound/theory_v2/timing_test.py",
        "experiments/kbound/test_3dadam_bootstrap.py",
        "archive/legacy/tests/test_old.py",
        "docs/research/kbound/archive/old/tests/test_old.py",
        "src/archive/tests/test_live.py",
        "src/helper.py",
    ]

    inventory = runner.classify_test_paths(paths)

    assert inventory["pytest_paths"] == [
        "docs/research/kbound/scripts/test_sar_faithful.py",
        "pkg/feature_test.py",
        "src/archive/tests/test_live.py",
        "tests/test_unit.py",
    ]
    excluded = {row["path"]: row["reason"] for row in inventory["excluded_paths"]}
    assert excluded == {
        "archive/legacy/tests/test_old.py": "immutable_archive",
        "docs/research/kbound/archive/old/tests/test_old.py": "immutable_archive",
        "docs/research/kbound/theory_v2/timing_test.py": "timing_utility_without_assertions",
        "experiments/kbound/test_3dadam_bootstrap.py": "historical_target_analysis_writer",
    }
    assert inventory["test_shaped_count"] == 8
    assert inventory["classified_count"] == 8


def test_protected_so2sat_readers_are_excluded_until_explicit_authorization() -> None:
    paths = [
        "tests/test_unit.py",
        "tests/test_canonical_release_data.py",
        "tests/test_empirical_data_quality_audit_remediation.py",
        "tests/test_reconciled_panels.py",
        "tests/test_so2sat_numbers_builder.py",
        "tests/test_so2sat_prospective_v2.py",
    ]

    default = runner.classify_test_paths(paths)
    authorized = runner.classify_test_paths(paths, authorize_protected_so2sat=True)

    assert default["pytest_paths"] == ["tests/test_unit.py"]
    assert {row["path"] for row in default["excluded_paths"]} == set(paths[1:])
    assert set(authorized["pytest_paths"]) == set(paths)


def test_tracked_inventory_is_read_from_head_not_the_mutable_index(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    committed = repo / "tests/test_committed.py"
    committed.parent.mkdir(parents=True)
    committed.write_text("def test_committed(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)
    staged = repo / "tests/test_staged.py"
    staged.write_text("def test_staged(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    assert runner.tracked_paths(repo) == ["tests/test_committed.py"]


def test_tracked_inventory_uses_only_positive_verification_source_pathspecs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        observed.append(args)
        assert kwargs == {"binary": True}
        return f"100644 blob {'a' * 40}\ttests/test_unit.py\0".encode()

    monkeypatch.setattr(runner, "_git", fake_git)

    assert runner.tracked_paths(tmp_path, "a" * 40) == ["tests/test_unit.py"]
    assert len(observed) == 1
    command = observed[0]
    separator = command.index("--")
    assert command[:separator] == (
        "ls-tree",
        "-r",
        "-z",
        "a" * 40,
    )
    assert command[separator + 1 :] == runner.TRACKED_VERIFICATION_PATHSPECS
    assert all(pathspec != "." for pathspec in runner.TRACKED_VERIFICATION_PATHSPECS)
    assert all("results/" not in pathspec for pathspec in runner.TRACKED_VERIFICATION_PATHSPECS)
    assert "experiments/kbound" not in runner.TRACKED_VERIFICATION_PATHSPECS
    assert set(runner.SO2SAT_SOURCE_PATHS) <= set(runner.TRACKED_VERIFICATION_PATHSPECS)
    assert all("*" not in pathspec for pathspec in runner.SO2SAT_SOURCE_PATHS)


def test_tracked_inventory_classifies_cross_project_test_modules_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidates = (
        "AETTA/tests/test_dnn_state_initialization.py",
        "docs/research/kbound/dashboard/tests/test_build_dashboard_snapshot.py",
        "docs/research/multiclass_vector_capacity/formal/tests/test_export_inventory.py",
        "docs/research/multiclass_vector_capacity/tests/test_exact_oracle.py",
        "docs/research/multiclass_vector_capacity/tests/test_statement_bindings.py",
    )

    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        assert kwargs == {"binary": True}
        pathspecs = args[args.index("--") + 1 :]
        selected = [
            path
            for path in candidates
            if any(path == pathspec or path.startswith(f"{pathspec}/") for pathspec in pathspecs)
        ]
        return b"".join(f"100644 blob {'a' * 40}\t{path}\0".encode() for path in selected)

    monkeypatch.setattr(runner, "_git", fake_git)

    inventory = runner.classify_test_paths(runner.tracked_paths(tmp_path, "a" * 40))

    assert inventory["pytest_paths"] == list(candidates[1:])
    assert inventory["excluded_paths"] == [
        {
            "path": "AETTA/tests/test_dnn_state_initialization.py",
            "reason": "third_party_baseline_outside_kbound_release_contract",
        }
    ]


def test_quality_python_paths_are_exact_and_exclude_non_source_roots() -> None:
    assert runner.quality_python_paths(
        [
            "kga/policy.py",
            "tests/test_policy.py",
            "experiments/kbound/so2sat/gate.py",
            "experiments/kbound/results/so2sat_private/reader.py",
            "experiments/kbound/data/so2sat_private/loader.py",
            "docs/research/kbound/archive/retired.py",
            "docs/research/kbound/paper/main.tex",
        ]
    ) == (
        "experiments/kbound/so2sat/gate.py",
        "kga/policy.py",
        "tests/test_policy.py",
    )


def test_authored_source_paths_exclude_protected_subtrees_before_filesystem_access(
    tmp_path: Path,
) -> None:
    public = tmp_path / "tests/public.py"
    public.parent.mkdir(parents=True)
    public.write_text("value = 1\n", encoding="utf-8")
    canary = tmp_path / "outside/canary.py"
    canary.parent.mkdir(parents=True)
    canary.write_text("raise RuntimeError('protected canary was followed')\n", encoding="utf-8")
    protected = tmp_path / "experiments/kbound/so2sat/target_results/private.py"
    protected.parent.mkdir(parents=True)
    protected.symlink_to(canary)

    observed = runner.validate_authored_source_paths(
        tmp_path,
        {
            "tests/public.py": ("100644", "blob"),
            "experiments/kbound/so2sat/target_results/private.py": ("100644", "blob"),
        },
        suffixes=(".py",),
    )

    assert observed == ("tests/public.py",)


def test_authored_source_path_validation_rejects_tracked_symlink_mode(
    tmp_path: Path,
) -> None:
    linked = tmp_path / "tests/linked.py"
    linked.parent.mkdir()
    target = tmp_path / "outside/canary.py"
    target.parent.mkdir()
    target.write_text("raise RuntimeError('symlink canary was followed')\n", encoding="utf-8")
    linked.symlink_to(target)

    with pytest.raises(runner.InventoryError, match="regular Git blob"):
        runner.validate_authored_source_paths(
            tmp_path,
            {"tests/linked.py": ("120000", "blob")},
            suffixes=(".py",),
        )


def test_validated_authored_source_inventory_uses_only_positive_pathspecs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tests/public.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    observed: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        observed.append(args)
        assert kwargs == {"binary": True}
        if args[0] == "ls-tree":
            return f"100644 blob {'a' * 40}\ttests/public.py\0".encode()
        assert args[0] == "diff"
        return b""

    monkeypatch.setattr(runner, "_git", fake_git)

    assert runner.validated_authored_source_paths(
        repo=tmp_path,
        revision="b" * 40,
        suffixes=(".py", ".sh"),
    ) == ("tests/public.py",)
    assert len(observed) == 2
    for command in observed:
        assert command[-len(runner.AUTHORED_SOURCE_PATHSPECS) :] == runner.AUTHORED_SOURCE_PATHSPECS
        assert "experiments/kbound/so2sat/" not in command


def test_validated_authored_source_inventory_excludes_tracked_worktree_deletions(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    live = repo / "tests/live.py"
    retired = repo / "tests/retired.py"
    live.parent.mkdir(parents=True)
    live.write_text("value = 'live'\n", encoding="utf-8")
    retired.write_text("value = 'retired'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)
    retired.unlink()

    assert runner.validated_authored_source_paths(
        repo=repo,
        revision="HEAD",
        suffixes=(".py",),
    ) == ("tests/live.py",)


def test_quality_path_validation_rejects_git_symlink_mode(tmp_path: Path) -> None:
    linked = tmp_path / "tests/linked.py"
    linked.parent.mkdir()
    target = tmp_path / "nonpublic/canary.txt"
    target.parent.mkdir()
    target.write_text("synthetic canary\n", encoding="utf-8")
    linked.symlink_to(target)

    with pytest.raises(runner.InventoryError, match="regular Git blob"):
        runner.validate_quality_paths(
            tmp_path,
            {"tests/linked.py": ("120000", "blob")},
        )


@pytest.mark.parametrize("redirect_parent", [False, True])
def test_quality_path_validation_rejects_worktree_symlink_redirection(tmp_path: Path, redirect_parent: bool) -> None:
    target = tmp_path / "nonpublic/canary.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")
    linked = tmp_path / "tests/linked.py"
    if redirect_parent:
        linked.parent.symlink_to(target.parent, target_is_directory=True)
    else:
        linked.parent.mkdir()
        linked.symlink_to(target)

    with pytest.raises(runner.InventoryError, match="symlink redirection"):
        runner.validate_quality_paths(
            tmp_path,
            {"tests/linked.py": ("100644", "blob")},
        )


def test_quality_path_validation_rejects_nonregular_worktree_entry(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "tests/not_a_file.py"
    directory.mkdir(parents=True)

    with pytest.raises(runner.InventoryError, match="regular working-tree file"):
        runner.validate_quality_paths(
            tmp_path,
            {"tests/not_a_file.py": ("100644", "blob")},
        )


def test_revision_binding_rejects_a_source_mutated_after_inventory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    source = repo / "tests/test_bound.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_bound(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_bound.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)

    bindings = runner.revision_blob_bindings(repo, "HEAD", ("tests/test_bound.py",))
    source.write_text("raise RuntimeError('mutated')\n", encoding="utf-8")

    with pytest.raises(runner.InventoryError, match="differs from the pinned Git blob"):
        runner.verify_worktree_blob_bindings(repo, bindings)


def test_release_pytest_rechecks_binding_before_starting_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "revision_blob_bindings",
        lambda repo, revision, paths: {"tests/test_a.py": "a" * 40},
    )
    monkeypatch.setattr(
        runner,
        "verify_worktree_blob_bindings",
        lambda repo, bindings: (_ for _ in ()).throw(runner.InventoryError("mutated source")),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pytest must not start after source mutation"),
    )

    with pytest.raises(runner.InventoryError, match="mutated source"):
        runner._run_pytest_release(
            ["tests/test_a.py"],
            repo=tmp_path,
            python="/pinned/python",
            source_commit="c" * 40,
        )


def test_so2sat_source_allowlist_guard_queries_only_exact_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "experiments/kbound/so2sat/source.py"
    observed: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        observed.append(args)
        assert kwargs == {"binary": True}
        return f"100644 blob {'b' * 40}\t{relative}\0".encode()

    monkeypatch.setattr(runner, "_git", fake_git)
    monkeypatch.setattr(runner, "SO2SAT_SOURCE_PATHS", (relative,))

    assert runner.verify_so2sat_source_allowlist(tmp_path, "c" * 40) == (relative,)
    assert observed == [
        (
            "ls-tree",
            "-z",
            "c" * 40,
            "--",
            relative,
        )
    ]


def test_so2sat_source_allowlist_matches_current_top_level_git_sources() -> None:
    observed = runner.verify_so2sat_source_allowlist(Path(__file__).resolve().parents[1], "HEAD")

    assert len(observed) == 24
    assert set(observed) == set(runner.SO2SAT_SOURCE_PATHS)


def test_tracked_inventory_cannot_list_a_test_shaped_file_under_experiment_results(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    public_test = repo / "tests/test_public.py"
    public_test.parent.mkdir(parents=True)
    public_test.write_text("def test_public(): pass\n", encoding="utf-8")
    trap = repo / "experiments/kbound/results/synthetic_gate/test_must_not_be_listed.py"
    trap.parent.mkdir(parents=True)
    trap.write_text("raise RuntimeError('must remain unreachable')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)

    assert runner.tracked_paths(repo) == ["tests/test_public.py"]


def test_git_inventory_never_walks_the_repository_object_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    objects = repo / ".git/objects"
    objects.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def reject_object_walk(path: Path):
        if path == objects or objects in path.parents:
            raise AssertionError("release inventory must not walk the Git object store")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", reject_object_walk)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(command, 0, stdout="c" * 40 + "\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._git(repo, "rev-parse", "HEAD") == "c" * 40 + "\n"


def test_git_inventory_fails_closed_without_materializing_an_object_store_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    objects = repo / ".git/objects"
    objects.mkdir(parents=True)
    original_iterdir = Path.iterdir

    def reject_object_walk(path: Path):
        if path == objects or objects in path.parents:
            raise AssertionError("release inventory must not inspect or copy Git objects")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", reject_object_walk)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(runner.InventoryError, match="Git inventory command failed"):
        runner._git(repo, "rev-parse", "HEAD")


def test_unknown_test_shaped_exception_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "EXPLICIT_TEST_EXCLUSIONS",
        {"known_test.py": "documented"},
    )

    with pytest.raises(runner.InventoryError, match="unclassified test-shaped path"):
        runner.classify_test_paths(
            ["known_test.py", "new/location/timing_test.py"],
            require_exception_classification=True,
        )


def test_inventory_payload_is_canonical_and_binds_git_tree(tmp_path: Path) -> None:
    payload = runner.build_inventory_payload(
        tracked_paths=["tests/test_z.py", "tests/test_a.py"],
        source_commit="a" * 40,
        source_tree="b" * 40,
    )
    output = tmp_path / "inventory.json"

    runner.write_canonical_json(output, payload)

    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert raw == runner.canonical_json_bytes(json.loads(raw))
    observed = json.loads(raw)
    assert observed["source_commit"] == "a" * 40
    assert observed["source_tree"] == "b" * 40
    assert observed["pytest_paths"] == ["tests/test_a.py", "tests/test_z.py"]
    assert observed["pytest_process_model"] == "one_module_per_fresh_interpreter"
    assert observed["pytest_process_group_count"] == 2
    assert [row["pytest_paths"] for row in observed["pytest_process_groups"]] == [
        ["tests/test_a.py"],
        ["tests/test_z.py"],
    ]
    assert observed["inventory_sha256"] == runner.inventory_digest(observed)


def test_private_path_scan_rejects_a_real_private_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    source = repo / "kga" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("path = '/Users/release-private'\n", encoding="utf-8")
    subprocess.run(["git", "add", "kga/module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    with pytest.raises(runner.InventoryError, match="private-path scan failed"):
        runner.run_private_path_scan(repo=repo, source_commit=commit)


def test_private_path_scan_returns_a_real_completed_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    source = repo / "kga" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'portable'\n", encoding="utf-8")
    subprocess.run(["git", "add", "kga/module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    receipt = runner.run_private_path_scan(repo=repo, source_commit=commit)

    assert receipt["status"] == "PASS"
    assert receipt["scanned_file_count"] == 1
    assert receipt["scanned_paths_sha256"] == hashlib.sha256(b"kga/module.py").hexdigest()


def test_private_path_scan_rejects_parent_symlink_before_outside_read(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True)
    source = repo / "kga/module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 'committed'\n", encoding="utf-8")
    subprocess.run(["git", "add", "kga/module.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_text("path = '/Users/protected-canary'\n", encoding="utf-8")
    source.unlink()
    source.parent.rmdir()
    (repo / "kga").symlink_to(outside, target_is_directory=True)

    with pytest.raises(runner.InventoryError, match="symlink redirection"):
        runner.run_private_path_scan(repo=repo, source_commit=commit)


def test_expected_source_commit_requires_the_exact_full_head() -> None:
    head = "a" * 40
    assert runner.verify_expected_source_commit(actual=head, expected=head) == head
    with pytest.raises(runner.InventoryError, match="full lowercase"):
        runner.verify_expected_source_commit(actual=head, expected="a" * 12)
    with pytest.raises(runner.InventoryError, match="does not match HEAD"):
        runner.verify_expected_source_commit(actual=head, expected="b" * 40)


@pytest.mark.parametrize("scenario,expected_code", [("pass", 0), ("fail", 1), ("source_changed", 1)])
def test_pytest_only_main_records_outcome_without_claiming_full_release(
    tmp_path: Path, scenario: str, expected_code: int
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    source = repo / "tests/test_receipt_fixture.py"
    source.parent.mkdir()
    source.write_text(f"def test_result():\n    assert {scenario != 'fail'}\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_receipt_fixture.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    commit, _tree = runner.source_identity(repo)
    if scenario == "source_changed":
        source.write_text("raise RuntimeError('changed source must not execute')\n", encoding="utf-8")
    output = tmp_path / "receipt.json"

    result = runner.main(["--repo", str(repo), "--output", str(output), "--expected-source-commit", commit])

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert result == expected_code
    assert receipt["execution_status"] == ("PYTEST_ONLY_PASS" if expected_code == 0 else "PYTEST_ONLY_FAIL")
    assert receipt["execution_scope"] == "pytest_only"
    assert receipt["pytest_exit_code"] == expected_code
    assert receipt["source_commit"] == commit
    assert receipt["private_path_scan"]["status"] == "NOT_RUN"
    assert receipt["immutable_hook_configuration"]["status"] == "NOT_RUN"
    assert "executed_gates" not in receipt
    assert receipt["inventory_sha256"] == runner.inventory_digest(receipt)
    if scenario == "source_changed":
        assert "differs from the pinned Git blob" in receipt["failure"]


def test_inventory_writer_refuses_a_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(runner.InventoryError, match="symlink"):
        runner.write_canonical_json(linked_parent / "inventory.json", {"status": "PASS"})

    assert not (outside / "inventory.json").exists()


def _local_snapshot_fixture(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "original"
    (repo / "kga").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "kga/policy.py").write_text("VALUE = 'committed'\n", encoding="utf-8")
    (repo / "tests/test_deleted.py").write_text("def test_old(): pass\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    return repo, runner.source_identity(repo)[0]


def test_local_source_snapshot_authenticates_current_bytes_as_a_distinct_commit(tmp_path: Path) -> None:
    repo, original_head = _local_snapshot_fixture(tmp_path)
    original_index = (repo / ".git/index").read_bytes()
    (repo / "kga/policy.py").write_text("VALUE = 'local edit'\n", encoding="utf-8")
    (repo / "tests/test_deleted.py").unlink()
    source = repo / "experiments/kbound/so2sat/prospective_v2.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'local-only source'\n", encoding="utf-8")
    (source.parent / "target_private.json").write_text('{"must_not_copy":true}\n', encoding="utf-8")
    destination = tmp_path / "snapshot"

    receipt = runner.create_local_source_snapshot(repo, destination, expected_source_commit=original_head)

    snapshot_head, snapshot_tree = runner.source_identity(destination)
    assert snapshot_head != original_head
    assert receipt["source_commit"] == snapshot_head
    assert receipt["source_tree"] == snapshot_tree
    assert receipt["original_source_commit"] == original_head
    assert receipt["execution_status"] == "NOT_RUN"
    assert runner.source_identity(repo)[0] == original_head
    assert (repo / ".git/index").read_bytes() == original_index
    assert not (destination / "experiments/kbound/so2sat/target_private.json").exists()
    assert not (destination / "tests/test_deleted.py").exists()
    lineage = json.loads((destination / "KBOUND_LOCAL_SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in lineage["files"]}
    assert rows["kga/policy.py"]["relation_to_original_head"] == "MODIFIED"
    assert rows["experiments/kbound/so2sat/prospective_v2.py"]["relation_to_original_head"] == "LOCAL_ONLY"
    assert "tests/test_deleted.py" in lineage["deleted_from_original_head"]
    for relative, row in rows.items():
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == row["sha256"]
    bindings = runner.revision_blob_bindings(destination, snapshot_head, tuple(rows))
    runner.verify_worktree_blob_bindings(destination, bindings)
    with pytest.raises(runner.InventoryError, match="does not match HEAD"):
        runner.verify_expected_source_commit(actual=snapshot_head, expected=original_head)


def test_local_source_snapshot_includes_hook_configuration_gate_input() -> None:
    exact, _roots = runner._local_snapshot_scopes()

    assert runner.PRE_COMMIT_CONFIG_PATH in exact


@pytest.mark.parametrize("symbol_available", [False, True])
def test_local_snapshot_rejects_dataless_before_read_with_optional_stat_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_available: bool
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"VALUE = 'synthetic placeholder'\n")
    real_stat = Path.stat
    metadata = source.stat()
    flagged = SimpleNamespace(**{name: getattr(metadata, name) for name in dir(metadata) if name.startswith("st_")})
    flagged.st_flags = 0x40000060

    def source_stat(path: Path, *args, **kwargs):
        return flagged if path == source else real_stat(path, *args, **kwargs)

    def reject_read(*args, **kwargs):
        pytest.fail("dataless source reached a byte-read helper")

    if symbol_available:
        monkeypatch.setattr(runner.stat, "SF_DATALESS", 0x40000000, raising=False)
    else:
        monkeypatch.delattr(runner.stat, "SF_DATALESS", raising=False)
    monkeypatch.setattr(Path, "stat", source_stat)
    monkeypatch.setattr(runner, "_git_blob_digest", reject_read)
    monkeypatch.setattr(runner, "_read_bound_source_bytes", reject_read)

    with pytest.raises(runner.InventoryError, match="dataless; explicit recovery"):
        runner._local_snapshot_bytes(tmp_path, "source.py")


def test_local_snapshot_reads_materialized_source_without_optional_stat_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_bytes(b"VALUE = 'materialized'\n")
    monkeypatch.delattr(runner.stat, "SF_DATALESS", raising=False)

    assert runner._local_snapshot_bytes(tmp_path, "source.py") == (b"VALUE = 'materialized'\n", "100644")


def test_local_source_snapshot_preserves_occupied_destination(tmp_path: Path) -> None:
    repo, original_head = _local_snapshot_fixture(tmp_path)
    destination = tmp_path / "occupied"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises((FileExistsError, runner.InventoryError)):
        runner.create_local_source_snapshot(repo, destination, expected_source_commit=original_head)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_local_source_snapshot_rejects_source_symlink(tmp_path: Path) -> None:
    repo, original_head = _local_snapshot_fixture(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 'not a source'\n", encoding="utf-8")
    (repo / "kga/redirected.py").symlink_to(outside)
    with pytest.raises(runner.InventoryError, match="symlink"):
        runner.create_local_source_snapshot(repo, tmp_path / "snapshot", expected_source_commit=original_head)


def test_local_source_snapshot_rejects_redirected_ancestor_before_directory_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, original_head = _local_snapshot_fixture(tmp_path)
    outside = tmp_path / "outside"
    (outside / "research/kbound/scripts").mkdir(parents=True)
    (repo / "docs").symlink_to(outside, target_is_directory=True)
    original_walk = os.walk

    def guarded_walk(path, *args, **kwargs):
        if Path(path).as_posix().startswith((repo / "docs").as_posix()):
            pytest.fail("a redirected source ancestor was walked")
        return original_walk(path, *args, **kwargs)

    monkeypatch.setattr(runner.os, "walk", guarded_walk)
    with pytest.raises(runner.InventoryError, match="symlink"):
        runner.create_local_source_snapshot(repo, tmp_path / "snapshot", expected_source_commit=original_head)


def test_local_source_snapshot_rejects_changes_between_copy_and_final_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, original_head = _local_snapshot_fixture(tmp_path)
    read_source = runner._local_snapshot_bytes
    changed = False

    def mutate_after_read(root: Path, relative: str):
        nonlocal changed
        result = read_source(root, relative)
        if relative == "kga/policy.py" and not changed:
            (root / relative).write_text("VALUE = 'changed after copying'\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(runner, "_local_snapshot_bytes", mutate_after_read)
    with pytest.raises(runner.InventoryError, match="changed during snapshot"):
        runner.create_local_source_snapshot(repo, tmp_path / "snapshot", expected_source_commit=original_head)
    assert not (tmp_path / "snapshot/.git").exists()


def test_inventory_records_exact_non_load_bearing_warning_filters() -> None:
    payload = runner.build_inventory_payload(
        tracked_paths=["tests/test_unit.py"],
        source_commit="a" * 40,
        source_tree="b" * 40,
    )
    filters = {row["filter"]: row for row in payload["allowed_pytest_warnings"]}
    assert ("ignore:Using `httpx` with `starlette.testclient` is deprecated:UserWarning") in filters
    assert all(row["scope"] and row["rationale"] for row in filters.values())


def test_run_pytest_uses_exact_discovered_paths_and_propagates_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed.append(command)
        code = 7 if command[-1] == "tests/test_a.py" else 0
        return type("Result", (), {"returncode": code})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "revision_blob_bindings",
        lambda repo, revision, paths: dict.fromkeys(paths, "a" * 40),
    )
    monkeypatch.setattr(runner, "verify_worktree_blob_bindings", lambda repo, bindings: None)
    code = runner.run_pytest_modules(
        ["tests/test_b.py", "tests/test_a.py"],
        repo=tmp_path,
        python="/pinned/python",
        source_commit="c" * 40,
    )

    assert observed == [
        [
            "/pinned/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_a.py",
        ],
        [
            "/pinned/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_b.py",
        ],
    ]
    assert code == 7


def test_pytest_process_plan_is_deterministic_and_covers_each_module_once() -> None:
    groups = runner.pytest_process_groups(["tests/test_z.py", "tests/test_a.py", "pkg/feature_test.py"])
    records = runner.pytest_process_group_records(groups)

    assert [group.paths for group in groups] == [
        ("pkg/feature_test.py",),
        ("tests/test_a.py",),
        ("tests/test_z.py",),
    ]
    assert [record["pytest_paths"] for record in records] == [
        ["pkg/feature_test.py"],
        ["tests/test_a.py"],
        ["tests/test_z.py"],
    ]
    assert len({record["group_id"] for record in records}) == 3
    runner.verify_pytest_process_group_records(
        ["tests/test_z.py", "tests/test_a.py", "pkg/feature_test.py"],
        records,
    )


def test_pytest_process_plan_rejects_duplicate_or_tampered_coverage() -> None:
    with pytest.raises(runner.InventoryError, match="duplicate pytest module"):
        runner.pytest_process_groups(["tests/test_a.py", "tests/test_a.py"])

    records = runner.pytest_process_group_records(runner.pytest_process_groups(["tests/test_a.py", "tests/test_b.py"]))
    with pytest.raises(runner.InventoryError, match="process-group coverage"):
        runner.verify_pytest_process_group_records(
            ["tests/test_a.py", "tests/test_b.py"],
            records[:-1],
        )


def test_release_pytest_aggregates_junit_skips_across_fresh_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        del kwargs
        observed.append(command)
        report = Path(command[command.index("--junitxml") + 1])
        if command[-1] == "tests/test_skip.py":
            report.write_text(
                "<testsuites><testsuite><testcase "
                'classname="docs.research.kbound.kbound_repro.tests.'
                'test_runtime_paths_deps" '
                'name="test_resolve_device_defers_missing_torch">'
                '<skipped message="torch is installed in this environment"/>'
                "</testcase></testsuite></testsuites>",
                encoding="utf-8",
            )
        else:
            report.write_text(
                '<testsuites><testsuite><testcase classname="tests.test_a" name="test_ok"/></testsuite></testsuites>',
                encoding="utf-8",
            )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "revision_blob_bindings",
        lambda repo, revision, paths: dict.fromkeys(paths, "a" * 40),
    )
    monkeypatch.setattr(runner, "verify_worktree_blob_bindings", lambda repo, bindings: None)
    receipt = runner._run_pytest_release(
        ["tests/test_skip.py", "tests/test_a.py"],
        repo=tmp_path,
        python="/pinned/python",
        source_commit="c" * 40,
    )

    assert [command[-1] for command in observed] == [
        "tests/test_a.py",
        "tests/test_skip.py",
    ]
    assert all(command.count("--junitxml") == 1 for command in observed)
    assert len({command[command.index("--junitxml") + 1] for command in observed}) == 2
    assert receipt.accepted_skips == (
        {
            "scope": (
                "docs.research.kbound.kbound_repro.tests.test_runtime_paths_deps."
                "test_resolve_device_defers_missing_torch"
            ),
            "reason": "torch is installed in this environment",
            "rationale": ("inverse optional-dependency branch is exercised only when torch is absent"),
        },
    )
    assert len(receipt.executed_group_ids) == 2


def test_release_pytest_runs_every_group_before_propagating_any_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        del kwargs
        module = command[-1]
        observed.append(module)
        report = Path(command[command.index("--junitxml") + 1])
        report.write_text(
            '<testsuites><testsuite><testcase classname="fixture" name="test"/></testsuite></testsuites>',
            encoding="utf-8",
        )
        return type(
            "Result",
            (),
            {"returncode": 7 if module == "tests/test_a.py" else 0},
        )()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        runner,
        "revision_blob_bindings",
        lambda repo, revision, paths: dict.fromkeys(paths, "a" * 40),
    )
    monkeypatch.setattr(runner, "verify_worktree_blob_bindings", lambda repo, bindings: None)
    with pytest.raises(
        runner.InventoryError,
        match=r"tests/test_a\.py.*exit 7",
    ):
        runner._run_pytest_release(
            ["tests/test_b.py", "tests/test_a.py"],
            repo=tmp_path,
            python="/pinned/python",
            source_commit="c" * 40,
        )

    assert observed == ["tests/test_a.py", "tests/test_b.py"]


def test_pytest_backed_release_gates_are_part_of_exact_once_inventory() -> None:
    runner.verify_pytest_backed_release_gate_coverage(
        [
            "tests/test_production_release_hygiene.py",
            "tests/test_reproducibility_hygiene.py",
            "tests/test_release_privacy.py",
        ]
    )
    with pytest.raises(runner.InventoryError, match="pytest-backed release gate"):
        runner.verify_pytest_backed_release_gate_coverage(
            [
                "tests/test_production_release_hygiene.py",
                "tests/test_reproducibility_hygiene.py",
            ]
        )


def test_quality_gate_phase_does_not_rerun_tracked_pytest_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        del kwargs
        observed.append(command)

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        runner,
        "revision_blob_bindings",
        lambda repo, revision, paths: dict.fromkeys(paths, "a" * 40),
    )
    monkeypatch.setattr(runner, "verify_worktree_blob_bindings", lambda repo, bindings: None)
    exact_quality_paths = (
        "experiments/kbound/so2sat/gate.py",
        "kga/policy.py",
        "tests/test_policy.py",
    )
    runner._run_quality_gates(
        repo=tmp_path,
        python="/pinned/python",
        quality_paths=exact_quality_paths,
        source_commit="c" * 40,
    )

    assert len(observed) == 5
    assert all("pytest" not in command for command in observed)
    assert all("pre_commit" not in command for command in observed)
    critical = observed[0]
    assert critical[:5] == ["/pinned/python", "-m", "ruff", "check", "--force-exclude"]
    excluded = {critical[index + 1] for index, value in enumerate(critical[:-1]) if value == "--extend-exclude"}
    assert excluded == set(runner.BYTE_PRESERVING_ARCHIVE_ROOTS)
    assert "docs/research/kbound/archive" not in excluded
    assert critical[-len(exact_quality_paths) :] == list(exact_quality_paths)
    assert "experiments/kbound" not in critical

    formatter = observed[2]
    assert formatter[:6] == [
        "/pinned/python",
        "-m",
        "ruff",
        "format",
        "--check",
        "--force-exclude",
    ]
    assert "--extend-exclude" not in formatter
    format_excluded = {formatter[index + 1] for index, value in enumerate(formatter[:-1]) if value == "--exclude"}
    assert format_excluded == set(runner.BYTE_PRESERVING_ARCHIVE_ROOTS)
    assert "docs/research/kbound/archive" not in format_excluded

    assert observed[4][1:3] == ["-W", runner.BANDIT_DEPRECATION_FILTER]
    assert all("-W" not in command for command in observed[:4])


@pytest.mark.parametrize("message,category,module,exit_code", [
    ("The verify_requirements argument is now a no-op and is deprecated for removal. Remove the argument from calls.",
     "DeprecationWarning", "stevedore.extension", 0),
    ("A different deprecation", "DeprecationWarning", "stevedore.extension", 1),
    ("The verify_requirements argument is now a no-op and is deprecated for removal. Remove the argument from calls.",
     "UserWarning", "stevedore.extension", 1),
    ("The verify_requirements argument is now a no-op and is deprecated for removal. Remove the argument from calls.",
     "DeprecationWarning", "kga.policy", 1),
])
def test_bandit_known_deprecation_is_visible_and_other_warnings_stay_fatal(
    message: str, category: str, module: str, exit_code: int,
) -> None:
    import subprocess
    import sys

    code = (
        "import warnings; "
        f"warnings.warn_explicit({message!r}, {category}, filename='synthetic.py', lineno=1, module={module!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-W", runner.BANDIT_DEPRECATION_FILTER, "-c", code],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == exit_code
    assert message in completed.stderr


def test_quality_gate_rejects_empty_source_inventory(tmp_path: Path) -> None:
    with pytest.raises(runner.InventoryError, match="quality inventory is empty"):
        runner._run_quality_gates(
            repo=tmp_path,
            python="/pinned/python",
            quality_paths=(),
            source_commit="c" * 40,
        )


def test_immutable_hook_configuration_gate_parses_current_selection_without_running_hooks() -> None:
    repo = Path(__file__).resolve().parents[1]
    source_commit, _source_tree = runner.source_identity(repo)

    receipt = runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)

    assert receipt["status"] == "PASS"
    assert receipt["config_path"] == ".pre-commit-config.yaml"
    assert receipt["repository_count"] == 4
    assert receipt["hook_count"] == 12
    assert receipt["mutating_hook_count"] == 4
    assert receipt["execution"] == "CONFIGURATION_PARSED_NO_HOOKS_EXECUTED"
    assert receipt["config_sha256"] == hashlib.sha256((repo / ".pre-commit-config.yaml").read_bytes()).hexdigest()


def test_immutable_hook_configuration_gate_rejects_broadened_ruff_autofix_selector(tmp_path: Path) -> None:
    current = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    restricted = (
        "        files: ^(?:kga/.*\\.py|src/scripts/kbound/(?:make_synth_archive|smoke_trichotomy)\\.py|"
        "deploy/api/kga_routes\\.py|docs/research/kbound/scripts/(?:validate_closure_protocol|"
        "run_closure_stage)\\.py|tests/(?:test_kga_package|test_kga_experiment_contract|"
        "test_smoke_trichotomy)\\.py)$"
    )
    assert current.count(restricted) == 2
    broadened = current.replace(restricted, "        files: ^", 1)
    repo, source_commit = _committed_hook_config_repo(tmp_path, broadened)

    with pytest.raises(runner.InventoryError, match="mutating hook selector escapes the A\\+ boundary"):
        runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)


def test_immutable_hook_configuration_gate_rejects_missing_result_exclusion(tmp_path: Path) -> None:
    current = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    protected = (
        "        exclude: &immutable_or_external ^(?:AETTA/|external/|archive/|research_lock/|audits/|"
        "experiments/kbound/results/|docs/research/kbound/(?:kbound_pkg/|paper/generated/|results/|audits/))"
    )
    assert current.count(protected) == 1
    underprotected = current.replace(
        protected,
        "        exclude: &immutable_or_external ^(?:AETTA/|external/)",
        1,
    )
    repo, source_commit = _committed_hook_config_repo(tmp_path, underprotected)

    with pytest.raises(runner.InventoryError, match="mutable hygiene hook does not exclude immutable path"):
        runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)


def test_immutable_hook_configuration_gate_rejects_entry_override(tmp_path: Path) -> None:
    current = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    marker = "        name: ruff (A+ lint, autofix)"
    assert current.count(marker) == 1
    overridden = current.replace(marker, f"{marker}\n        entry: python -c 'raise SystemExit(0)'", 1)
    repo, source_commit = _committed_hook_config_repo(tmp_path, overridden)

    with pytest.raises(runner.InventoryError, match="unreviewed control keys"):
        runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)


@pytest.mark.parametrize("hook_name", ["ruff (A+ lint, autofix)", "ruff (A+ format)"])
@pytest.mark.parametrize("exclusion", ["^", r"^kga/policy\.py$"])
def test_immutable_hook_configuration_gate_rejects_ruff_exclusion_of_required_source(
    tmp_path: Path, hook_name: str, exclusion: str
) -> None:
    current = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    marker = f"        name: {hook_name}"
    assert current.count(marker) == 1
    disabled = current.replace(marker, f"{marker}\n        exclude: '{exclusion}'", 1)
    repo, source_commit = _committed_hook_config_repo(tmp_path, disabled)

    with pytest.raises(runner.InventoryError, match=r"excludes required A\+ path: kga/policy\.py"):
        runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)


@pytest.mark.parametrize("hook_name", ["ruff (A+ lint, autofix)", "ruff (A+ format)"])
def test_immutable_hook_configuration_gate_rejects_invalid_ruff_exclusion(tmp_path: Path, hook_name: str) -> None:
    current = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    marker = f"        name: {hook_name}"
    assert current.count(marker) == 1
    invalid = current.replace(marker, f"{marker}\n        exclude: '['", 1)
    repo, source_commit = _committed_hook_config_repo(tmp_path, invalid)

    with pytest.raises(runner.InventoryError, match="invalid exclude regex"):
        runner.validate_immutable_hook_configuration(repo=repo, source_commit=source_commit)


def test_run_all_gates_forwards_validated_exact_quality_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current_hook_config = (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook_repo, source_commit = _committed_hook_config_repo(tmp_path, current_hook_config)
    pytest_paths = sorted(runner.PYTEST_BACKED_RELEASE_GATE_MODULES)
    payload: dict[str, object] = {
        "pytest_paths": pytest_paths,
        "pytest_process_groups": runner.pytest_process_group_records(runner.pytest_process_groups(pytest_paths)),
        "validator_paths": ["validator.py"],
        "source_commit": source_commit,
    }
    exact = ("kga/policy.py", "tests/test_policy.py")
    observed: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        runner,
        "_run_pytest_release",
        lambda *args, **kwargs: runner.PytestRunReceipt((), ()),
    )
    monkeypatch.setattr(runner, "_run_validators", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "validated_quality_python_paths",
        lambda *, repo, revision: exact,
    )

    def record_quality_paths(
        *,
        repo: Path,
        python: str,
        quality_paths: tuple[str, ...],
        source_commit: str,
    ) -> None:
        assert repo == hook_repo
        assert python == "/pinned/python"
        assert source_commit == payload["source_commit"]
        observed.append(quality_paths)

    monkeypatch.setattr(runner, "_run_quality_gates", record_quality_paths)
    monkeypatch.setattr(
        runner,
        "run_private_path_scan",
        lambda *, repo, source_commit: {"status": "PASS"},
    )
    for name in (
        "_run_package_gate",
        "_run_dashboard_gate",
        "_run_lean_gates",
        "_run_docker_gates",
    ):
        monkeypatch.setattr(runner, name, lambda *args, **kwargs: None)

    runner.run_all_gates(payload, repo=hook_repo, python="/pinned/python")

    assert observed == [exact]
    assert payload["immutable_hook_configuration"]["status"] == "PASS"


def test_validator_inventory_is_complete_and_rejects_unclassified_candidates() -> None:
    paths = [
        "experiments/kbound/theory_validation/val_thm1.py",
        "experiments/kbound/conj1_validator.py",
        "docs/research/kbound/theory_v2/val_tight_constants.py",
        "docs/research/kbound/theory_v2/minimax_frontier/validate_v2.py",
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/scripts/validate_closure_seed.py",
        "docs/research/kbound/archive/old/val_retired.py",
    ]

    inventory = runner.classify_validator_paths(paths, authorize_protected_so2sat=True)

    assert inventory["validator_paths"] == [
        "docs/research/kbound/scripts/validate_canonical_release_data.py",
        "docs/research/kbound/theory_v2/minimax_frontier/validate_v2.py",
        "docs/research/kbound/theory_v2/val_tight_constants.py",
        "experiments/kbound/conj1_validator.py",
        "experiments/kbound/theory_validation/val_thm1.py",
    ]
    excluded = {row["path"]: row["reason"] for row in inventory["excluded_validators"]}
    assert excluded == {
        "docs/research/kbound/archive/old/val_retired.py": "immutable_archive",
        "docs/research/kbound/scripts/validate_closure_seed.py": "requires_run_specific_input",
    }
    with pytest.raises(runner.InventoryError, match="unclassified validator"):
        runner.classify_validator_paths(["docs/research/kbound/scripts/validate_new_gate.py"])

    safe = runner.classify_validator_paths(paths)
    assert "docs/research/kbound/scripts/validate_canonical_release_data.py" not in safe["validator_paths"]
    assert {row["path"]: row["reason"] for row in safe["excluded_validators"]}[
        "docs/research/kbound/scripts/validate_canonical_release_data.py"
    ] == ("requires_explicit_protected_so2sat_authority")


def test_docker_gate_builds_from_exact_commit_staging_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    canary = repo / "experiments/kbound/so2sat/target_data/never_touch.bin"
    canary.parent.mkdir(parents=True)
    canary.write_bytes(b"protected")
    payloads = {
        ".dockerignore": b"*\n!Dockerfile\n",
        "Dockerfile": b"FROM scratch\n",
        "requirements-api-py311-linux.lock.txt": b"",
        "deploy/api/main.py": b"app = object()\n",
        "kga/policy.py": b"def decide(): return 'ABSTAIN'\n",
    }
    git_calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        git_calls.append(args)
        assert kwargs == {"binary": True}
        if args[0] == "ls-tree":
            return b"".join(f"100644 blob {'a' * 40}\t{path}\0".encode() for path in payloads)
        assert args[0] == "show"
        relative = args[1].split(":", 1)[1]
        return payloads[relative]

    observed_commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        observed_commands.append(command)
        assert kwargs["repo"] == repo
        if command[:2] == ["docker", "build"]:
            context = Path(command[-1])
            assert context != repo
            assert command[-1] != "."
            present = {path.relative_to(context).as_posix() for path in context.rglob("*") if path.is_file()}
            assert present == set(payloads)
            for relative, expected in payloads.items():
                assert (context / relative).read_bytes() == expected

    monkeypatch.setattr(runner, "_git", fake_git)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_docker_gates(repo=repo, source_commit="c" * 40)

    assert observed_commands[0][0:2] == ["docker", "build"]
    assert git_calls[0] == (
        "ls-tree",
        "-r",
        "-z",
        "c" * 40,
        "--",
        *runner.DOCKER_CONTEXT_PATHSPECS,
    )
    assert all("experiments/kbound/so2sat" not in part for call in git_calls for part in call)


def test_docker_context_rejects_git_symlink_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(_repo: Path, *args: str, **kwargs: object) -> bytes:
        assert args[0] == "ls-tree"
        return (
            f"100644 blob {'a' * 40}\t.dockerignore\0"
            f"100644 blob {'a' * 40}\tDockerfile\0"
            f"100644 blob {'a' * 40}\trequirements-api-py311-linux.lock.txt\0"
            f"120000 blob {'a' * 40}\tdeploy/api/linked.py\0"
            f"100644 blob {'a' * 40}\tkga/policy.py\0"
        ).encode()

    monkeypatch.setattr(runner, "_git", fake_git)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *args, **kwargs: pytest.fail("Docker must not run for a nonregular source entry"),
    )

    with pytest.raises(runner.InventoryError, match="regular Git blob"):
        runner._run_docker_gates(repo=tmp_path, source_commit="c" * 40)


def test_release_gate_plan_covers_every_task7_surface() -> None:
    names = {gate.name for gate in runner.release_gate_plan(python="/pinned/python")}
    assert {
        "pytest",
        "standalone-validators",
        "immutable-hook-configuration",
        "ruff-critical",
        "ruff-lint",
        "ruff-format",
        "mypy",
        "bandit",
        "private-path-scan",
        "package-build-install-cli",
        "dashboard-lockfile-build",
        "lean-kbound",
        "lean-multiclass-vector-capacity",
        "docker-build",
        "docker-health-network-none",
        "docker-content-network-none",
    } <= names
    assert all(gate.required for gate in runner.release_gate_plan(python="/pinned/python"))


def test_release_runbook_executes_full_gate_runner_and_preserves_publication_order() -> None:
    runbook = (Path(__file__).resolve().parents[1] / "docs/research/kbound/runbooks/release_candidate.sh").read_text(
        encoding="utf-8"
    )
    test_body = runbook.split("step_test() {", 1)[1].split("\n}", 1)[0]
    assert "run_repository_verification.py" in test_body
    assert "--all-gates" in test_body
    assert '--expected-source-commit "${RELEASE_SOURCE_COMMIT:?release source was not captured}"' in test_body
    assert "check_repo --staged" not in test_body
    public_body = runbook.split("step_public_bundle() {", 1)[1].split("\n}", 1)[0]
    anonymous_body = runbook.split("step_anonymous_supplement() {", 1)[1].split("\n}", 1)[0]
    assert "--replace-verified" in public_body
    assert "--replace-verified" in anonymous_body
    all_body = runbook.split("  all)", 1)[1].split("    ;;", 1)[0]
    assert "step_deep_local_provenance" not in all_body
    assert "step_generate_deep_local_cct20" not in all_body
    assert "step_deep_local_dataset_preflight" not in all_body
    assert all_body.index("step_verify_public_bundle") < all_body.index("emit_checksums")
    assert all_body.index("emit_checksums") < all_body.index("step_anonymous_supplement")
    assert all_body.index("step_anonymous_supplement") < all_body.index("emit_post_checksums")
    # No combined deep-local mode is required by the portable release. Explicit
    # data access is tested separately; do not reintroduce it to satisfy a
    # historical runbook-layout assertion.
    standalone_test = runbook.split("  test)", 1)[1].split(";;", 1)[0]
    assert "start_release_run" in standalone_test


def _portable_runbook_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(
        (Path(__file__).resolve().parents[1] / "docs/research/kbound/runbooks/release_candidate.sh").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text("[project]\nname = 'portable-release-fixture'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    event_log = tmp_path / "portable-events.txt"
    fake_python = tmp_path / "portable-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "event_log = pathlib.Path(os.environ['KBOUND_EVENT_LOG'])\n"
        "args = sys.argv[1:]\n"
        "with event_log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(' '.join(args) + '\\n')\n"
        "if os.environ.get('KBOUND_FAIL_PORTABLE') and any(os.environ['KBOUND_FAIL_PORTABLE'] in arg for arg in args):\n"
        "    raise SystemExit(23)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return runbook, event_log, fake_python


def test_default_preflight_is_data_portable_and_deep_local_preflight_checks_datasets(
    tmp_path: Path,
) -> None:
    runbook, event_log, _unused_python = _portable_runbook_fixture(tmp_path)
    fake_tool = tmp_path / "verified-tool"
    fake_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    fake_python = tmp_path / "preflight-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "event_log = pathlib.Path(os.environ['KBOUND_EVENT_LOG'])\n"
        "args = sys.argv[1:]\n"
        "stdin = sys.stdin.read() if args == ['-'] else ''\n"
        "with event_log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps({'args': args, 'stdin': stdin}) + '\\n')\n"
        "if args and args[0].endswith('verify_release_toolchain.py'):\n"
        "    destination = pathlib.Path(args[args.index('--resolved-tools-output') + 1])\n"
        "    tool = os.environ['KBOUND_FAKE_TOOL']\n"
        "    names = ('LATEXMK','LATEXPAND','PANDOC','PDFDETACH','PDFINFO','PDFLATEX','PDFTOPPM','PDFTOTEXT','PERL','SOFFICE')\n"
        "    destination.write_text(''.join(f'KBOUND_TOOL_{name}\\t{tool}\\n' for name in names), encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "KBOUND_EVENT_LOG": str(event_log),
        "KBOUND_FAKE_TOOL": str(fake_tool),
        "KBOUND_PYTHON": str(fake_python),
    }

    portable = subprocess.run(
        ["bash", str(runbook), "preflight"],
        cwd=runbook.parents[4],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert portable.returncode == 0, portable.stderr
    portable_events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert not any("imagenetr_root" in event["stdin"] or "pacs_root" in event["stdin"] for event in portable_events)

    event_log.unlink()
    deep_local = subprocess.run(
        ["bash", str(runbook), "deep-local-preflight"],
        cwd=runbook.parents[4],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert deep_local.returncode == 0, deep_local.stderr
    deep_events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert any("imagenetr_root" in event["stdin"] and "pacs_root" in event["stdin"] for event in deep_events)


@pytest.mark.parametrize("mode", ["generate", "validate-results"])
def test_default_release_generation_and_validation_never_invoke_protected_so2sat_readers(
    tmp_path: Path, mode: str
) -> None:
    """Default release modes must stop before any protected natural-shift reader."""

    runbook, event_log, _unused_python = _portable_runbook_fixture(tmp_path)
    fake_tool = tmp_path / "verified-tool"
    fake_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    fake_python = tmp_path / "release-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "event_log = pathlib.Path(os.environ['KBOUND_EVENT_LOG'])\n"
        "args = sys.argv[1:]\n"
        "with event_log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if args and args[0].endswith('verify_release_toolchain.py'):\n"
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.parent.mkdir(parents=True, exist_ok=True)\n"
        "    output.write_text('{\"status\": \"verified\"}\\n', encoding='utf-8')\n"
        "    resolved = pathlib.Path(args[args.index('--resolved-tools-output') + 1])\n"
        "    names = ('LATEXMK','LATEXPAND','PANDOC','PDFDETACH','PDFINFO','PDFLATEX','PDFTOPPM','PDFTOTEXT','PERL','SOFFICE')\n"
        "    tool = os.environ['KBOUND_FAKE_TOOL']\n"
        "    resolved.write_text(''.join(f'KBOUND_TOOL_{name}\\t{tool}\\n' for name in names), encoding='utf-8')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(runbook), mode],
        cwd=runbook.parents[4],
        env={
            **os.environ,
            "KBOUND_EVENT_LOG": str(event_log),
            "KBOUND_FAKE_TOOL": str(fake_tool),
            "KBOUND_PYTHON": str(fake_python),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    flattened = "\n".join(" ".join(event) for event in events).casefold()
    assert "so2sat" not in flattened
    assert "validate_canonical_release_data.py" not in flattened
    assert "audit_natural_target_provenance.py" not in flattened
    sync_events = [event for event in events if event and Path(event[0]).name == "sync_reconciled_panels.py"]
    result_manifest_events = [event for event in events if event and Path(event[0]).name == "build_result_manifest.py"]
    storage_refresh_events = [
        event for event in events if event and Path(event[0]).name == "refresh_storage_manifest.py"
    ]
    if mode == "generate":
        assert sync_events == [["scripts/sync_reconciled_panels.py", "--public-only"]]
        assert result_manifest_events == [
            [
                "docs/research/kbound/scripts/build_result_manifest.py",
                "--public-only",
            ]
        ]
        assert storage_refresh_events == [
            [
                "docs/research/kbound/scripts/refresh_storage_manifest.py",
                "--write-public-only",
            ]
        ]
    else:
        assert sync_events == []
        assert result_manifest_events == []
        assert storage_refresh_events == []
    assert "test_reconciled_panels.py" not in flattened


def test_portable_release_mode_runs_artifact_semantics_without_local_preflight_and_fails_closed(
    tmp_path: Path,
) -> None:
    runbook, event_log, fake_python = _portable_runbook_fixture(tmp_path)
    environment = {
        **os.environ,
        "KBOUND_EVENT_LOG": str(event_log),
        "KBOUND_PYTHON": str(fake_python),
    }
    completed = subprocess.run(
        ["bash", str(runbook), "verify-portable-release"],
        cwd=runbook.parents[4],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert any("build_release_source_seal.py --check-portable" in event for event in events)
    assert any(
        "build_cct20_public_bundle.py" in event and "--portable-check" in event and "--release-manifest" in event
        for event in events
    )
    assert any("build_anonymous_supplement.py" in event and "--portable-check" in event for event in events)
    assert sum("verify_release_checksums.py" in event for event in events) == 2
    assert not any("verify_python_environment.py" in event for event in events)
    assert not any("verify_release_toolchain.py" in event for event in events)

    event_log.unlink()
    failed = subprocess.run(
        ["bash", str(runbook), "verify-portable-release"],
        cwd=runbook.parents[4],
        env={**environment, "KBOUND_FAIL_PORTABLE": "build_cct20_public_bundle.py"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode == 23
    failed_events = event_log.read_text(encoding="utf-8").splitlines()
    assert any("build_cct20_public_bundle.py" in event for event in failed_events)
    assert not any("build_anonymous_supplement.py" in event for event in failed_events)


def test_checksums_mode_uses_only_structural_source_seal_validation(
    tmp_path: Path,
) -> None:
    runbook, event_log, fake_python = _portable_runbook_fixture(tmp_path)
    repo = runbook.parents[4]
    artifact = repo / "artifact.bin"
    artifact.write_bytes(b"portable bytes\n")
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "event_log = pathlib.Path(os.environ['KBOUND_EVENT_LOG'])\n"
        "args = sys.argv[1:]\n"
        "with event_log.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(' '.join(args) + '\\n')\n"
        "if '--list-required' in args:\n"
        "    print('artifact.bin')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(runbook), "checksums"],
        cwd=repo,
        env={
            **os.environ,
            "KBOUND_EVENT_LOG": str(event_log),
            "KBOUND_PYTHON": str(fake_python),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert any("build_release_source_seal.py --check-structure" in event for event in events)
    assert not any("verify_python_environment.py" in event for event in events)
    assert not any("verify_release_toolchain.py" in event for event in events)


def test_pytest_skip_reconciliation_rejects_unexplained_skip(tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_release" name="test_gate">'
        '<skipped message="missing release evidence"/></testcase></testsuite></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(runner.InventoryError, match="unexpected pytest skip"):
        runner.validate_pytest_report(report)


def test_pytest_skip_reconciliation_accepts_exact_non_load_bearing_branch(
    tmp_path: Path,
) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        "<testsuites><testsuite><testcase "
        'classname="docs.research.kbound.kbound_repro.tests.test_runtime_paths_deps" '
        'name="test_resolve_device_defers_missing_torch">'
        '<skipped message="torch is installed in this environment"/>'
        "</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    assert runner.validate_pytest_report(report) == [
        {
            "scope": (
                "docs.research.kbound.kbound_repro.tests.test_runtime_paths_deps."
                "test_resolve_device_defers_missing_torch"
            ),
            "reason": "torch is installed in this environment",
            "rationale": "inverse optional-dependency branch is exercised only when torch is absent",
        }
    ]


def test_dashboard_gate_rejects_stale_unproduced_javascript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dashboard = tmp_path / "docs/research/kbound/dashboard"
    (dashboard / "src").mkdir(parents=True)
    (dashboard / "js").mkdir()
    (dashboard / "src/app.ts").write_text("export const value = 1;\n")
    (dashboard / "js/app.js").write_text("export const value = 1;\n")
    (dashboard / "js/app.js.map").write_text("{}\n")
    (dashboard / "js/stale.js").write_text("export const stale = true;\n")
    (dashboard / "package.json").write_text('{"scripts":{"build":"tsc"}}\n')
    (dashboard / "package-lock.json").write_text('{"name":"fixture","lockfileVersion":3,"packages":{}}\n')
    (dashboard / "tsconfig.json").write_text("{}\n")
    monkeypatch.setattr(runner, "_run", lambda *args, **kwargs: None)

    with pytest.raises(runner.InventoryError, match="unexpected.*dashboard"):
        runner._run_dashboard_gate(repo=tmp_path)


def test_dashboard_gate_uses_hash_locked_install_with_an_empty_isolated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = tmp_path / "docs/research/kbound/dashboard"
    (dashboard / "src").mkdir(parents=True)
    (dashboard / "js").mkdir()
    (dashboard / "src/app.ts").write_text("export const value = 1;\n", encoding="utf-8")
    (dashboard / "js/app.js").write_text("export const value = 1;\n", encoding="utf-8")
    (dashboard / "js/app.js.map").write_text("{}\n", encoding="utf-8")
    (dashboard / "package.json").write_text(
        '{"scripts":{"build":"tsc"},"devDependencies":{"typescript":"5.9.3"}}\n',
        encoding="utf-8",
    )
    (dashboard / "package-lock.json").write_text(
        '{"name":"fixture","lockfileVersion":3,"packages":'
        '{"":{"devDependencies":{"typescript":"5.9.3"}},'
        '"node_modules/typescript":{"version":"5.9.3","integrity":"sha512-fixture"}}}\n',
        encoding="utf-8",
    )
    (dashboard / "tsconfig.json").write_text("{}\n", encoding="utf-8")
    observed: list[list[str]] = []
    cache_was_empty: list[bool] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        observed.append(command)
        if command[:2] == ["npm", "ci"]:
            cache = Path(command[command.index("--cache") + 1])
            cache_was_empty.append(not cache.exists())

    monkeypatch.setattr(runner, "_run", fake_run)
    runner._run_dashboard_gate(repo=tmp_path)

    install = observed[0]
    assert install[:2] == ["npm", "ci"]
    assert "--offline" not in install
    assert "--ignore-scripts" in install
    assert cache_was_empty == [True]


def test_package_gate_requires_exactly_one_wheel_and_source_archive(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "kbound_kga-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    with pytest.raises(runner.InventoryError, match="one wheel and one source archive"):
        runner.validate_distribution_artifacts([wheel])

    source = tmp_path / "kbound_kga-0.1.0.tar.gz"
    source.write_bytes(b"source")
    assert runner.validate_distribution_artifacts([source, wheel]) == (wheel, source)


def test_package_gate_executes_the_installed_kga_console_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CITATION.cff",
    ):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    (tmp_path / "kga").mkdir()
    (tmp_path / "kga/__init__.py").write_text("", encoding="utf-8")
    observed: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        observed.append(command)
        if command[1:3] == ["-m", "build"]:
            output = Path(command[command.index("--outdir") + 1])
            output.mkdir()
            (output / "kbound_kga-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
            (output / "kbound_kga-0.1.0.tar.gz").write_bytes(b"source")

    monkeypatch.setattr(runner, "_run", fake_run)
    runner._run_package_gate(repo=tmp_path, python="/verified/python")

    build_command = next(command for command in observed if command[1:3] == ["-m", "build"])
    assert "--no-isolation" in build_command
    assert "--skip-dependency-check" not in build_command
    assert any(Path(command[0]).name == "kga" and command[1:] == ["--help"] for command in observed)


def test_gate_command_can_remove_source_tree_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str] = {}

    class Completed:
        returncode = 0

    def fake_run(*_args: object, **kwargs: object) -> Completed:
        observed.update(kwargs["env"])
        return Completed()

    monkeypatch.setenv("PYTHONPATH", "/untrusted/source/tree")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._run(["tool"], repo=tmp_path, unset_env=("PYTHONPATH",))
    assert "PYTHONPATH" not in observed


@pytest.mark.parametrize("fail_build", [False, True])
def test_public_bundle_command_verifies_output_and_propagates_failure(tmp_path, fail_build):
    runbook, events, python = _portable_runbook_fixture(tmp_path)
    env = dict(os.environ, KBOUND_EVENT_LOG=str(events), KBOUND_PYTHON=str(python))
    if fail_build:
        env["KBOUND_FAIL_PORTABLE"] = "--replace-verified"
    completed = subprocess.run(
        ["bash", str(runbook), "public-bundle"],
        cwd=runbook.parents[4],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == (23 if fail_build else 0), completed.stderr
    calls = events.read_text().splitlines()
    assert "--replace-verified" in calls[0]
    assert len(calls) == (1 if fail_build else 2)
    if not fail_build:
        assert "--check" in calls[1]


def test_native_aline_group_requires_explicit_method_interpreter(monkeypatch):
    monkeypatch.delenv("ALINE_PYTHON", raising=False)
    with pytest.raises(runner.InventoryError, match="ALINE_PYTHON"):
        runner.pytest_group_runtime(("tests/test_aline_native_estimator.py",), python="release-python")


def test_native_aline_group_uses_declared_runtime_without_changing_parent(monkeypatch):
    monkeypatch.setenv("ALINE_PYTHON", sys.executable)
    monkeypatch.delenv("ALINE_NATIVE_STAGE", raising=False)
    python, env = runner.pytest_group_runtime(("tests/test_aline_native_estimator.py",), python="release-python")
    assert python == sys.executable
    assert env["ALINE_NATIVE_STAGE"] == "1"
    assert env["ALINE_PYTHON"] == sys.executable
    assert "ALINE_NATIVE_STAGE" not in os.environ


def test_regular_group_does_not_select_method_runtime(monkeypatch):
    monkeypatch.setenv("ALINE_PYTHON", "/unavailable/method/python")
    assert runner.pytest_group_runtime(("tests/test_regular.py",), python="release-python") == ("release-python", None)
