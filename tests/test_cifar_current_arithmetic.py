"""Synthetic-only tests for the portable current CIFAR arithmetic replay."""
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/replay_cifar_current_arithmetic.py"
CHECKER = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/verify_cifar_current_arithmetic.py"
RELEASE_RUNBOOK = Path(__file__).resolve().parents[1] / "docs/research/kbound/runbooks/release_candidate.sh"
SPEC = importlib.util.spec_from_file_location("current_arithmetic_panel", SCRIPT)
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)
build_policy_actions = panel.build_policy_actions
score_actions = panel.score_actions


def fixture():
    records = [
        {'a0': 0.4, 'a_adapted': 0.7, 'B': 0.3, 'Z': [0, 0.3, 0, 0, 0.5, 0, 0, 0.2, 0, 0, 0]},
        {'a0': 0.8, 'a_adapted': 0.3, 'B': -0.5, 'Z': [0, 0.5, 0, 0, 0.3, 0, 0, -0.1, 0, 0, 0]},
        {'a0': 0.5, 'a_adapted': 0.5, 'B': 0, 'Z': [0, 0.5, 0, 0, 0.5, 0, 0, 0, 0, 0, 0]},
    ]
    cells = [
        {'sample_id': 'first', 'prediction': 0.2, 'radius': 0.1, 'action': 'ADAPT'},
        {'sample_id': 'second', 'prediction': -0.2, 'radius': 0.1, 'action': 'FREEZE'},
        {'sample_id': 'tie', 'prediction': 0, 'radius': 0.1, 'action': 'ABSTAIN'},
    ]
    return records, cells


def test_shared_point_gate_tie_and_kga_fallback():
    records, cells = fixture()
    actions = build_policy_actions(records, cells)
    assert actions['point_benefit'] == ['ADAPT', 'FREEZE', 'FREEZE']
    assert actions['current_kga'] == ['ADAPT', 'FREEZE', 'ABSTAIN']
    assert actions['confidence_increase'] == ['ADAPT', 'FREEZE', 'FREEZE']
    assert actions['entropy_decrease'] == ['ADAPT', 'FREEZE', 'FREEZE']
    assert score_actions(records, actions['current_kga'])['regret'] == 0
    assert score_actions(records, actions['always_adapt'])['regret'] == pytest.approx(0.5 / 3)
    assert score_actions(records, actions['always_freeze'])['regret'] == pytest.approx(0.3 / 3)
    assert score_actions(records, actions['always_adapt'])['false_adapt_nonpositive_count'] == 2


@pytest.mark.parametrize('mutation,error', [
    ('duplicate', 'unique'), ('mismatched_action', 'stored action'),
    ('nan', 'finite'), ('negative_radius', 'nonnegative'), ('length', 'length'),
])
def test_invalid_authority_fails_closed(mutation, error):
    records, cells = fixture()
    cells = copy.deepcopy(cells)
    if mutation == 'duplicate':
        cells[1]['sample_id'] = cells[0]['sample_id']
    elif mutation == 'mismatched_action':
        cells[0]['action'] = 'FREEZE'
    elif mutation == 'nan':
        cells[0]['prediction'] = float('nan')
    elif mutation == 'negative_radius':
        cells[0]['radius'] = -0.1
    elif mutation == 'length':
        cells.pop()
    with pytest.raises(ValueError, match=error):
        build_policy_actions(records, cells)


def test_latex_is_generated_from_scores():
    records, cells = fixture()
    scores = {name: score_actions(records, actions) for name, actions in build_policy_actions(records, cells).items()}
    latex = panel.render_latex(scores)
    assert 'Current KGA & 0.0000 & 0.6667 & 0.000 & 0.667 & 1 & 1 & 1' in latex
    assert 'AETTA' not in latex and 'POEM' not in latex


@pytest.mark.parametrize('field', ['prediction', 'radius', 'action'])
def test_mismatched_saved_array_digest_is_rejected(field):
    _, cells = fixture()
    hashes = {key: panel.array_sha([cell[key] for cell in cells]) for key in ['prediction', 'radius', 'action']}
    hashes[field] = '0' * 64
    with pytest.raises(ValueError, match=field):
        panel.verify_array_hashes(cells, hashes)


def test_cli_rejects_mismatched_audit_before_creating_output(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"source_inventory": []}))
    output = tmp_path / "must-not-exist"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(tmp_path),
                           "--audit", str(audit), "--audit-sha256", "0" * 64,
                           "--out-dir", str(output)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "prior audit SHA-256 mismatch" in proc.stderr
    assert not output.exists()


def test_cli_rejects_changed_input_before_creating_output(tmp_path):
    relative = panel.CANON + "canonical_panel_results.json"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"source_inventory": [{
        "root_role": "active_checkout", "status": "READ",
        "relative_path": relative, "sha256": "0" * 64}]}))
    output = tmp_path / "must-not-exist"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(tmp_path),
                           "--audit", str(audit), "--audit-sha256", panel.sha(audit.read_bytes()),
                           "--out-dir", str(output)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "input changed since paired-array audit" in proc.stderr
    assert not output.exists()


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=panel.unique_pairs)


@pytest.fixture
def synthetic_release(tmp_path):
    """Build a complete seven-input synthetic authority with the real replay."""
    repo = tmp_path / "synthetic repository"
    manifest_relative = panel.CANON + "source_manifest.json"
    canonical_relative = panel.CANON + "canonical_panel_results.json"
    source_rows = []
    per_file = []
    authenticated_paths = []
    for seed in range(5):
        relative = panel.CANON + f"source/cifar10c/per_condition_cifar10c_tent_seed{seed}.json"
        records = []
        cells = []
        for index in range(432):
            condition = f"synthetic-{index}"
            records.append({
                "seed": seed,
                "condition": condition,
                "a0": 0.25,
                "a_adapted": 0.75,
                "B": 0.5,
                "Z": [0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            })
            cells.append({
                "sample_id": f"tent|seed={seed}|condition={condition}",
                "prediction": 0.2,
                "radius": 0.1,
                "action": "ADAPT",
            })
        payload = json.dumps({"records": records}, sort_keys=True).encode()
        source_path = repo / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
        digest = panel.sha(payload)
        source_rows.append({"destination": relative, "compact_sha256": digest})
        authenticated_paths.append((relative, digest))
        per_file.append({
            "seed": seed,
            "current_prediction_sha256": panel.array_sha([cell["prediction"] for cell in cells]),
            "current_radius_sha256": panel.array_sha([cell["radius"] for cell in cells]),
            "current_action_sha256": panel.array_sha([cell["action"] for cell in cells]),
            "current_cell_authority": {"cells": cells},
        })

    manifest_path = repo / manifest_relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps({"files": source_rows}, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    canonical = {
        "source_manifest_sha256": panel.sha(manifest_bytes),
        "runtime": {"fixture": "synthetic"},
        "panels": {"cifar10c": {"panel": {"candidates": {"tent": {
            "per_file": per_file,
            "regret": {"always_adapt": 0.0, "always_freeze": 0.5, "kga": 0.0},
            "kappa_sweep": [{"kappa": 0, "regret": 0.0}],
        }}}}},
    }
    canonical_path = repo / canonical_relative
    canonical_bytes = json.dumps(canonical, sort_keys=True).encode()
    canonical_path.write_bytes(canonical_bytes)
    authenticated_paths[:0] = [
        (canonical_relative, panel.sha(canonical_bytes)),
        (manifest_relative, panel.sha(manifest_bytes)),
    ]
    audit = tmp_path / "synthetic-audit.json"
    audit.write_text(json.dumps({"source_inventory": [
        {"root_role": "active_checkout", "status": "READ", "relative_path": relative, "sha256": digest}
        for relative, digest in authenticated_paths
    ]}, sort_keys=True))
    artifacts = tmp_path / "generated-artifacts"
    generated = subprocess.run([
        sys.executable, str(SCRIPT), "--repo", str(repo),
        "--audit", str(audit), "--audit-sha256", panel.sha(audit.read_bytes()),
        "--out-dir", str(artifacts),
    ], capture_output=True, text=True)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return {
        "repo": repo,
        "audit": audit,
        "artifacts": artifacts,
        "panel_sha256": panel.sha((artifacts / "CURRENT_ARITHMETIC_PANEL.json").read_bytes()),
        "source": repo / authenticated_paths[-1][0],
    }


def run_checker(release):
    return subprocess.run([
        sys.executable, str(CHECKER), "--repo", str(release["repo"]),
        "--audit", str(release["audit"]),
        "--audit-sha256", panel.sha(release["audit"].read_bytes()),
        "--artifact-dir", str(release["artifacts"]),
        "--expected-panel-sha256", release["panel_sha256"],
    ], capture_output=True, text=True)


def test_checker_accepts_exact_regenerated_artifacts_without_overwriting(synthetic_release):
    before = {path.name: path.read_bytes() for path in synthetic_release["artifacts"].iterdir()}
    proc = run_checker(synthetic_release)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {path.name: path.read_bytes() for path in synthetic_release["artifacts"].iterdir()}
    assert after == before


@pytest.mark.parametrize("name", [
    "CURRENT_ARITHMETIC_PANEL.json", "cells.jsonl", "current_arithmetic_panel.tex",
])
def test_checker_rejects_modified_generated_artifact(synthetic_release, name):
    path = synthetic_release["artifacts"] / name
    path.write_bytes(path.read_bytes() + b"tampered\n")
    proc = run_checker(synthetic_release)
    assert proc.returncode != 0
    assert "mismatch" in proc.stderr


@pytest.mark.parametrize("name", [
    "CURRENT_ARITHMETIC_PANEL.json", "cells.jsonl", "current_arithmetic_panel.tex",
])
def test_checker_rejects_missing_generated_artifact(synthetic_release, name):
    (synthetic_release["artifacts"] / name).unlink()
    proc = run_checker(synthetic_release)
    assert proc.returncode != 0
    assert "missing required current CIFAR artifact" in proc.stderr


def test_checker_rejects_source_changed_after_audit(synthetic_release):
    source = synthetic_release["source"]
    source.write_bytes(source.read_bytes() + b"\n")
    proc = run_checker(synthetic_release)
    assert proc.returncode != 0
    assert "input changed since paired-array audit" in proc.stderr


@pytest.fixture
def release_runbook_fixture(tmp_path):
    repo = tmp_path / "release repository"
    runbook = repo / "docs/research/kbound/runbooks/release_candidate.sh"
    runbook.parent.mkdir(parents=True)
    runbook.write_bytes(RELEASE_RUNBOOK.read_bytes())
    (repo / "pyproject.toml").write_text("[project]\nname = 'release-fixture'\n")
    events = tmp_path / "runbook-events.jsonl"
    outputs = tmp_path / "generation-outputs"
    fake_python = tmp_path / "controlled-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['-']:\n"
        "    sys.stdin.read()\n"
        "events = pathlib.Path(os.environ['CIFAR_RUNBOOK_EVENTS'])\n"
        "with events.open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "name = pathlib.Path(args[0]).name if args else ''\n"
        "if name == 'verify_cifar_current_arithmetic.py' and os.environ.get('CIFAR_FAIL_VERIFY') == '1':\n"
        "    print('controlled current CIFAR verification failure', file=sys.stderr)\n"
        "    raise SystemExit(23)\n"
        "writers = {'sync_reconciled_panels.py', 'build_result_manifest.py', 'refresh_storage_manifest.py'}\n"
        "if name in writers:\n"
        "    outputs = pathlib.Path(os.environ['CIFAR_RUNBOOK_OUTPUTS'])\n"
        "    outputs.mkdir(parents=True, exist_ok=True)\n"
        "    (outputs / (name + '.called')).write_text('generation writer reached\\n')\n"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    for name in (
        "KBOUND_VERIFY_TOOLCHAIN", "KBOUND_REFRESH_CANONICAL", "KBOUND_STRICT_SOURCE_SEAL",
        "KBOUND_PACKAGE_RELEASE",
    ):
        env.pop(name, None)
    env.update({
        "KBOUND_PYTHON": str(fake_python),
        "CIFAR_RUNBOOK_EVENTS": str(events),
        "CIFAR_RUNBOOK_OUTPUTS": str(outputs),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return {"repo": repo, "runbook": runbook, "events": events, "outputs": outputs, "env": env}


def run_release_mode(fixture, mode, *, fail_verify=False):
    env = {**fixture["env"]}
    if fail_verify:
        env["CIFAR_FAIL_VERIFY"] = "1"
    return subprocess.run(
        ["bash", str(fixture["runbook"]), mode],
        cwd=fixture["repo"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def release_events(fixture):
    if not fixture["events"].exists():
        return []
    return [json.loads(line) for line in fixture["events"].read_text().splitlines()]


@pytest.mark.parametrize("mode", ["generate", "all"])
def test_failed_current_cifar_verification_prevents_release_generation_writers(
    release_runbook_fixture, mode,
):
    proc = run_release_mode(release_runbook_fixture, mode, fail_verify=True)
    assert proc.returncode == 23, proc.stdout + proc.stderr
    assert not release_runbook_fixture["outputs"].exists()
    invoked = [Path(args[0]).name for args in release_events(release_runbook_fixture) if args and args[0] != "-"]
    assert invoked[-1] == "verify_cifar_current_arithmetic.py"
    assert not {
        "sync_reconciled_panels.py", "build_result_manifest.py", "refresh_storage_manifest.py",
    }.intersection(invoked)


def test_valid_current_cifar_verification_precedes_release_generation_writers(release_runbook_fixture):
    proc = run_release_mode(release_runbook_fixture, "generate")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    invoked = [Path(args[0]).name for args in release_events(release_runbook_fixture) if args and args[0] != "-"]
    assert invoked == [
        "verify_cifar_current_arithmetic.py",
        "sync_reconciled_panels.py",
        "build_result_manifest.py",
        "refresh_storage_manifest.py",
    ]
    assert {path.name for path in release_runbook_fixture["outputs"].iterdir()} == {
        "sync_reconciled_panels.py.called",
        "build_result_manifest.py.called",
        "refresh_storage_manifest.py.called",
    }
