"""Result regeneration must not implicitly scan or delete storage metadata.

The real CLI, reconciliation assembly, writers, and renderers run against a
temporary repository. Scientific replay stages return small synthetic panels;
no archived result, dataset tree, import operation, or Git command is accessed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_reconcile_module():
    """Do not resolve the unrelated ``src/scripts`` package used by pytest."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "reconcile_result_panels.py"
    spec = importlib.util.spec_from_file_location("_test_reconcile_no_cleanup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconcile = _load_reconcile_module()


def _score() -> dict:
    return {
        "n": 5,
        "regret": {"kga": 0.125, "always_adapt": 0.25, "always_freeze": 0.375},
        "fa_u": 0.0,
        "adapt_rate": 0.4,
        "decision_coverage": 0.6,
    }


def _transfer() -> dict:
    return {"exact_rank_transfer_score": _score()}


def _candidate_panel() -> dict:
    return {"panel": {"candidates": {name: _score() for name in ("tent", "eata", "sar")}}}


@pytest.fixture
def temporary_reconciliation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    result_root = tmp_path / "experiments/kbound/results/reconciled_panels_v1"
    source_root = result_root / "source"
    source_root.mkdir(parents=True)
    generator = tmp_path / "synthetic_reconcile.py"
    generator.write_text("# Synthetic generator identity for the isolated test.\n", encoding="utf-8")
    generator_hash = hashlib.sha256(generator.read_bytes()).hexdigest()
    original_manifest = {"file_count": 1, "files": [], "generator_sha256": "0" * 64}
    (result_root / "source_manifest.json").write_text(json.dumps(original_manifest), encoding="utf-8")

    preserved_files = {
        tmp_path / "._repository_metadata": b"repository metadata sentinel\n",
        result_root / "._canonical_panel_results.json": b"result metadata sentinel\n",
        source_root / "synthetic" / "._panel.json": b"source metadata sentinel\n",
        source_root / "synthetic" / "panel.json": b'{"records": []}\n',
        tmp_path / "unrelated" / "nested" / "._metadata": b"unrelated metadata sentinel\n",
        tmp_path / "._metadata_directory" / "ordinary_file": b"directory-content sentinel\n",
    }
    for path, content in preserved_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    panels = {
        "officehome": {"primary": _transfer(), "test_stream_seed_replication": _transfer()},
        "iwildcam": {"primary": _transfer(), "release_promotion": {"eligible": False}},
        "imagenetc": _candidate_panel(),
        "pacs": {"pooled_domain_seed_mean": _score()},
        "imagenet_r": {"panel": {"architecture_panel_aggregate": _score()}},
        "cifar10c": _candidate_panel(),
        "camelyon17": {"ood": {"replay": _transfer()}, "b_v2_diagnostic": _candidate_panel()},
        "rxrx1": {"primary_model_seed0": _transfer()},
        "cifar101": {"replay": _transfer()},
    }
    stage_calls: list[str] = []

    def replace_stage(name: str, value: dict) -> None:
        def synthetic_stage() -> dict:
            stage_calls.append(name)
            return copy.deepcopy(value)

        monkeypatch.setattr(reconcile, name, synthetic_stage)

    replace_stage("reconcile_transfer_panels", {name: panels[name] for name in ("officehome", "iwildcam")})
    replace_stage("reconcile_grid_panels", {name: panels[name] for name in ("imagenetc", "imagenet_r", "cifar10c")})
    replace_stage("reconcile_pacs", panels["pacs"])
    replace_stage("reconcile_missing_locked_panels", {name: panels[name] for name in ("camelyon17", "rxrx1", "cifar101")})

    def forbidden_import(*args, **kwargs):
        raise AssertionError("normal reconciliation must not import or clean compact sources")

    monkeypatch.setattr(reconcile, "ROOT", tmp_path)
    monkeypatch.setattr(reconcile, "RESULT_ROOT", result_root)
    monkeypatch.setattr(reconcile, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(reconcile, "__file__", str(generator))
    monkeypatch.setattr(reconcile, "import_sources", forbidden_import)
    # Runtime-pin enforcement is independent of the cleanup regression. Keep
    # this temporary-only assembly test usable in either supported test env.
    monkeypatch.setattr(reconcile, "EXPECTED_NUMPY_VERSION", reconcile.np.__version__)
    monkeypatch.setattr(reconcile, "EXPECTED_SKLEARN_VERSION", reconcile.sklearn.__version__)
    monkeypatch.setattr(reconcile.platform, "platform", lambda: "synthetic-test-platform")
    return SimpleNamespace(
        result_root=result_root,
        generator_hash=generator_hash,
        original_manifest=original_manifest,
        panels=panels,
        stage_calls=stage_calls,
        preserved_files=preserved_files,
    )


@contextmanager
def _forbid_traversal_and_cleanup(monkeypatch: pytest.MonkeyPatch):
    """Fail on attempted cleanup, even if a deletion would be silently ignored."""
    attempted: list[str] = []

    def blocked(operation: str):
        def fail(*args, **kwargs):
            attempted.append(operation)
            raise AssertionError(f"normal reconciliation attempted {operation}")

        return fail

    with monkeypatch.context() as guard:
        for name in ("glob", "rglob", "iterdir", "walk", "unlink", "rmdir"):
            if hasattr(Path, name):
                guard.setattr(Path, name, blocked(f"Path.{name}"))
        for name in ("scandir", "walk", "unlink", "remove", "rmdir", "removedirs", "system", "popen"):
            guard.setattr(os, name, blocked(f"os.{name}"))
        guard.setattr(shutil, "rmtree", blocked("shutil.rmtree"))
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            guard.setattr(subprocess, name, blocked(f"subprocess.{name}"))
        yield
        # A broad exception handler in a cleanup helper must not conceal an
        # attempted traversal/deletion merely because the sentinel survives.
        assert not attempted, f"normal reconciliation attempted prohibited cleanup/traversal: {attempted}"


@pytest.mark.parametrize("reuse_transfer", [False, True], ids=["default", "reuse-transfer"])
def test_normal_main_preserves_metadata_and_scientific_outputs(
    temporary_reconciliation: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reuse_transfer: bool,
) -> None:
    fixture = temporary_reconciliation
    expected_panels = copy.deepcopy(fixture.panels)
    argv = ["reconcile_result_panels.py"]
    if reuse_transfer:
        argv.append("--reuse-transfer")
        # A distinct existing value proves the normal reuse branch is retained.
        expected_panels["officehome"]["primary"]["exact_rank_transfer_score"]["regret"]["kga"] = 0.1875
        (fixture.result_root / "canonical_panel_results.json").write_text(
            json.dumps({"panels": expected_panels}), encoding="utf-8"
        )
    monkeypatch.setattr(sys, "argv", argv)

    with _forbid_traversal_and_cleanup(monkeypatch):
        assert reconcile.main() == 0

    for path, content in fixture.preserved_files.items():
        assert path.is_file()
        assert path.read_bytes() == content
    expected_calls = ["reconcile_grid_panels", "reconcile_pacs", "reconcile_missing_locked_panels"]
    if not reuse_transfer:
        expected_calls.insert(0, "reconcile_transfer_panels")
    assert fixture.stage_calls == expected_calls

    saved = json.loads((fixture.result_root / "canonical_panel_results.json").read_text())
    assert saved["panels"] == expected_panels
    assert saved["schema"] == "kbound-canonical-panel-results-v2"
    assert saved["alpha"] == reconcile.ALPHA
    assert saved["abstention_semantics"] == "retain frozen model"
    assert saved["generator_sha256"] == fixture.generator_hash
    assert saved["source_file_count"] == fixture.original_manifest["file_count"]
    manifest_path = fixture.result_root / "source_manifest.json"
    assert json.loads(manifest_path.read_text()) == {
        **fixture.original_manifest,
        "generator": "scripts/reconcile_result_panels.py",
        "generator_sha256": fixture.generator_hash,
    }
    assert saved["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    markdown = (fixture.result_root / "CANONICAL_PANEL_RESULTS.md").read_text()
    latex = (fixture.result_root / "canonical_panel_table.tex").read_text()
    assert "| Office-Home M-v2 |" in markdown
    assert "0.1250" in markdown and "0.1250" in latex
    assert "withheld: official-metric rerun required" in markdown
    assert "withheld: official-metric rerun required" in latex
    assert capsys.readouterr().out.count("Wrote ") == 3


def test_cli_help_does_not_start_generation_import_or_cleanup(
    temporary_reconciliation: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["reconcile_result_panels.py", "--help"])
    with _forbid_traversal_and_cleanup(monkeypatch), pytest.raises(SystemExit) as raised:
        reconcile.main()
    assert raised.value.code == 0
    assert temporary_reconciliation.stage_calls == []
    assert not (temporary_reconciliation.result_root / "canonical_panel_results.json").exists()
    help_text = capsys.readouterr().out
    assert "--import-from" in help_text
    assert "--clean-import" in help_text
    assert "--reuse-transfer" in help_text
    for path, content in temporary_reconciliation.preserved_files.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("operation", ["traversal", "unlink", "subprocess"])
def test_behavioral_guard_catches_a_reintroduced_cleanup_even_if_errors_are_swallowed(
    temporary_reconciliation: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Counterfactual check: an implicit cleanup tail would fail the guard."""
    fixture = temporary_reconciliation
    sentinel = fixture.result_root / "._canonical_panel_results.json"
    original_write_outputs = reconcile.write_outputs

    def write_then_attempt_cleanup(result: dict) -> None:
        original_write_outputs(result)
        try:
            if operation == "traversal":
                list(reconcile.ROOT.rglob("._*"))
            elif operation == "unlink":
                sentinel.unlink(missing_ok=True)
            else:
                subprocess.run(["find", str(reconcile.ROOT), "-name", "._*", "-type", "f", "-delete"], check=True)
        except Exception:
            pass

    monkeypatch.setattr(reconcile, "write_outputs", write_then_attempt_cleanup)
    monkeypatch.setattr(sys, "argv", ["reconcile_result_panels.py"])
    with pytest.raises(AssertionError, match="attempted prohibited cleanup/traversal"):
        with _forbid_traversal_and_cleanup(monkeypatch):
            assert reconcile.main() == 0
    assert sentinel.read_bytes() == fixture.preserved_files[sentinel]
