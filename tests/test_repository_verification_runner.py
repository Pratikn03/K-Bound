from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.scripts import run_repository_verification as runner


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
    subprocess.run(
        ["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True
    )
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
        return b"tests/test_unit.py\0"

    monkeypatch.setattr(runner, "_git", fake_git)

    assert runner.tracked_paths(tmp_path, "a" * 40) == ["tests/test_unit.py"]
    assert len(observed) == 1
    command = observed[0]
    separator = command.index("--")
    assert command[:separator] == (
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        "a" * 40,
    )
    assert command[separator + 1 :] == runner.TRACKED_VERIFICATION_PATHSPECS
    assert all(pathspec != "." for pathspec in runner.TRACKED_VERIFICATION_PATHSPECS)
    assert all(
        "results/" not in pathspec for pathspec in runner.TRACKED_VERIFICATION_PATHSPECS
    )


def test_tracked_inventory_cannot_list_a_test_shaped_file_under_experiment_results(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release@example.invalid"], cwd=repo, check=True
    )
    public_test = repo / "tests/test_public.py"
    public_test.parent.mkdir(parents=True)
    public_test.write_text("def test_public(): pass\n", encoding="utf-8")
    trap = repo / "experiments/kbound/results/synthetic_gate/test_must_not_be_listed.py"
    trap.parent.mkdir(parents=True)
    trap.write_text("raise RuntimeError('must remain unreachable')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source freeze"], cwd=repo, check=True)

    assert runner.tracked_paths(repo) == ["tests/test_public.py"]


def test_git_inventory_avoids_cloud_only_loose_objects_when_config_is_resident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    loose_object = git_dir / "objects" / "aa" / ("b" * 38)
    (git_dir / "objects" / "pack").mkdir(parents=True)
    loose_object.parent.mkdir()
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="ascii")
    (git_dir / "HEAD").write_text("c" * 40 + "\n", encoding="ascii")
    loose_object.write_bytes(b"cloud placeholder")

    monkeypatch.setattr(runner, "_is_dataless", lambda path: path == loose_object)

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if not any(part.startswith("--git-dir=") for part in command):
            raise subprocess.TimeoutExpired(command, timeout=30)
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(
            command, 0, stdout="c" * 40 + "\n", stderr=""
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._git(repo, "rev-parse", "HEAD") == "c" * 40 + "\n"


def test_git_inventory_avoids_cloud_only_packed_refs_when_config_is_resident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    packed_refs = git_dir / "packed-refs"
    (git_dir / "objects" / "pack").mkdir(parents=True)
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="ascii")
    (git_dir / "HEAD").write_text("c" * 40 + "\n", encoding="ascii")
    packed_refs.write_text(
        "# pack-refs with: peeled fully-peeled sorted\n", encoding="ascii"
    )

    monkeypatch.setattr(runner, "_is_dataless", lambda path: path == packed_refs)

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if not any(part.startswith("--git-dir=") for part in command):
            raise subprocess.TimeoutExpired(command, timeout=30)
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(
            command, 0, stdout="c" * 40 + "\n", stderr=""
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._git(repo, "rev-parse", "HEAD") == "c" * 40 + "\n"


def test_dataless_detection_never_classifies_a_directory_as_a_cloud_file(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "objects" / "aa"
    directory.mkdir(parents=True)

    assert not runner._is_dataless(directory)


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
    source = tmp_path / "kga" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("path = '/Users/release-private'\n", encoding="utf-8")

    with pytest.raises(runner.InventoryError, match="private-path scan failed"):
        runner.run_private_path_scan(repo=tmp_path)


def test_private_path_scan_returns_a_real_completed_receipt(tmp_path: Path) -> None:
    source = tmp_path / "kga" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'portable'\n", encoding="utf-8")

    receipt = runner.run_private_path_scan(repo=tmp_path)

    assert receipt["status"] == "PASS"
    assert receipt["scanned_file_count"] == 1
    assert (
        receipt["scanned_paths_sha256"] == hashlib.sha256(b"kga/module.py").hexdigest()
    )


def test_expected_source_commit_requires_the_exact_full_head() -> None:
    head = "a" * 40
    assert runner.verify_expected_source_commit(actual=head, expected=head) == head
    with pytest.raises(runner.InventoryError, match="full lowercase"):
        runner.verify_expected_source_commit(actual=head, expected="a" * 12)
    with pytest.raises(runner.InventoryError, match="does not match HEAD"):
        runner.verify_expected_source_commit(actual=head, expected="b" * 40)


def test_inventory_writer_refuses_a_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(runner.InventoryError, match="symlink"):
        runner.write_canonical_json(
            linked_parent / "inventory.json", {"status": "PASS"}
        )

    assert not (outside / "inventory.json").exists()


def test_inventory_records_exact_non_load_bearing_warning_filters() -> None:
    payload = runner.build_inventory_payload(
        tracked_paths=["tests/test_unit.py"],
        source_commit="a" * 40,
        source_tree="b" * 40,
    )
    filters = {row["filter"]: row for row in payload["allowed_pytest_warnings"]}
    assert (
        "ignore:Using `httpx` with `starlette.testclient` is deprecated:UserWarning"
    ) in filters
    assert all(row["scope"] and row["rationale"] for row in filters.values())


def test_run_pytest_uses_exact_discovered_paths_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed.append(command)
        code = 7 if command[-1] == "tests/test_a.py" else 0
        return type("Result", (), {"returncode": code})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    code = runner.run_pytest_modules(
        ["tests/test_b.py", "tests/test_a.py"],
        python="/pinned/python",
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
    groups = runner.pytest_process_groups(
        ["tests/test_z.py", "tests/test_a.py", "pkg/feature_test.py"]
    )
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

    records = runner.pytest_process_group_records(
        runner.pytest_process_groups(["tests/test_a.py", "tests/test_b.py"])
    )
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
    receipt = runner._run_pytest_release(
        ["tests/test_skip.py", "tests/test_a.py"],
        repo=tmp_path,
        python="/pinned/python",
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
            "rationale": (
                "inverse optional-dependency branch is exercised only when torch is absent"
            ),
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
    with pytest.raises(
        runner.InventoryError,
        match=r"tests/test_a\.py.*exit 7",
    ):
        runner._run_pytest_release(
            ["tests/test_b.py", "tests/test_a.py"],
            repo=tmp_path,
            python="/pinned/python",
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
    runner._run_quality_gates(repo=tmp_path, python="/pinned/python")

    assert len(observed) == 5
    assert all("pytest" not in command for command in observed)
    assert all("pre_commit" not in command for command in observed)
    critical = observed[0]
    assert critical[:5] == ["/pinned/python", "-m", "ruff", "check", "--force-exclude"]
    excluded = {
        critical[index + 1]
        for index, value in enumerate(critical[:-1])
        if value == "--extend-exclude"
    }
    assert excluded == set(runner.BYTE_PRESERVING_ARCHIVE_ROOTS)
    assert "docs/research/kbound/archive" not in excluded

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
    format_excluded = {
        formatter[index + 1]
        for index, value in enumerate(formatter[:-1])
        if value == "--exclude"
    }
    assert format_excluded == set(runner.BYTE_PRESERVING_ARCHIVE_ROOTS)
    assert "docs/research/kbound/archive" not in format_excluded


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
        runner.classify_validator_paths(
            ["docs/research/kbound/scripts/validate_new_gate.py"]
        )

    safe = runner.classify_validator_paths(paths)
    assert (
        "docs/research/kbound/scripts/validate_canonical_release_data.py"
        not in safe["validator_paths"]
    )
    assert {row["path"]: row["reason"] for row in safe["excluded_validators"]}[
        "docs/research/kbound/scripts/validate_canonical_release_data.py"
    ] == ("requires_explicit_protected_so2sat_authority")


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
    assert all(
        gate.required for gate in runner.release_gate_plan(python="/pinned/python")
    )


def test_release_runbook_executes_full_gate_runner_and_preserves_publication_order() -> (
    None
):
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    test_body = runbook.split("step_test() {", 1)[1].split("\n}", 1)[0]
    assert "run_repository_verification.py" in test_body
    assert "--all-gates" in test_body
    public_body = runbook.split("step_public_bundle() {", 1)[1].split("\n}", 1)[0]
    anonymous_body = runbook.split("step_anonymous_supplement() {", 1)[1].split(
        "\n}", 1
    )[0]
    assert "--replace-verified" in public_body
    assert "--replace-verified" in anonymous_body
    all_body = runbook.split("  all)", 1)[1].split("    ;;", 1)[0]
    assert "step_deep_local_provenance" not in all_body
    assert "step_generate_deep_local_cct20" not in all_body
    assert "step_deep_local_dataset_preflight" not in all_body
    assert all_body.index("step_verify_public_bundle") < all_body.index(
        "emit_checksums"
    )
    assert all_body.index("emit_checksums") < all_body.index(
        "step_anonymous_supplement"
    )
    assert all_body.index("step_anonymous_supplement") < all_body.index(
        "emit_post_checksums"
    )
    deep_body = runbook.split("  all-deep-local)", 1)[1].split("    ;;", 1)[0]
    assert "step_deep_local_dataset_preflight" in deep_body
    assert deep_body.index("step_generate_deep_local_cct20") < deep_body.index(
        "step_deep_local_provenance"
    )
    assert deep_body.index("step_deep_local_provenance") < deep_body.index(
        "step_public_bundle"
    )


def _portable_runbook_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_text(
        (
            Path(__file__).resolve().parents[1]
            / "docs/research/kbound/runbooks/release_candidate.sh"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'portable-release-fixture'\n", encoding="utf-8"
    )
    receipt = repo / "docs/research/kbound/audits/release_toolchain_2026_09_05_v2.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status": "verified"}\n', encoding="utf-8")
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
        "    output = pathlib.Path(args[args.index('--output') + 1])\n"
        "    output.write_text('{\"status\": \"verified\"}\\n', encoding='utf-8')\n"
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
    portable_events = [
        json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    assert not any(
        "imagenetr_root" in event["stdin"] or "pacs_root" in event["stdin"]
        for event in portable_events
    )

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
    deep_events = [
        json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        "imagenetr_root" in event["stdin"] and "pacs_root" in event["stdin"]
        for event in deep_events
    )


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
    events = [
        json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()
    ]
    flattened = "\n".join(" ".join(event) for event in events).casefold()
    assert "so2sat" not in flattened
    assert "sync_reconciled_panels.py" not in flattened
    assert "build_result_manifest.py" not in flattened
    assert "validate_canonical_release_data.py" not in flattened
    assert "audit_natural_target_provenance.py" not in flattened
    assert "refresh_storage_manifest.py" not in flattened
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
    assert any(
        "build_release_source_seal.py --check-portable" in event for event in events
    )
    assert any(
        "build_cct20_public_bundle.py" in event
        and "--portable-check" in event
        and "--release-manifest" in event
        for event in events
    )
    assert any(
        "build_anonymous_supplement.py" in event and "--portable-check" in event
        for event in events
    )
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
    assert any(
        "build_release_source_seal.py --check-structure" in event for event in events
    )
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


def test_dashboard_gate_rejects_stale_unproduced_javascript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dashboard = tmp_path / "docs/research/kbound/dashboard"
    (dashboard / "src").mkdir(parents=True)
    (dashboard / "js").mkdir()
    (dashboard / "src/app.ts").write_text("export const value = 1;\n")
    (dashboard / "js/app.js").write_text("export const value = 1;\n")
    (dashboard / "js/app.js.map").write_text("{}\n")
    (dashboard / "js/stale.js").write_text("export const stale = true;\n")
    (dashboard / "package.json").write_text('{"scripts":{"build":"tsc"}}\n')
    (dashboard / "package-lock.json").write_text(
        '{"name":"fixture","lockfileVersion":3,"packages":{}}\n'
    )
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

    build_command = next(
        command for command in observed if command[1:3] == ["-m", "build"]
    )
    assert "--no-isolation" in build_command
    assert "--skip-dependency-check" not in build_command
    assert any(
        Path(command[0]).name == "kga" and command[1:] == ["--help"]
        for command in observed
    )


def test_gate_command_can_remove_source_tree_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
