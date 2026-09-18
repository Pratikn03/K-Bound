from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_official_baseline_audit_cli_witness as witness_support

REPO = Path(__file__).resolve().parents[1]
signed_cli_evidence = witness_support.signed_cli_evidence
pytestmark = pytest.mark.parametrize("signed_cli_evidence", ["aetta"], indirect=True)


@pytest.fixture
def legacy_cifar_package(signed_cli_evidence):
    """Stage real CLI programs, a signed synthetic package, and a test witness launcher."""
    fixture = signed_cli_evidence
    repo = fixture["repo"]
    scripts = repo / "docs/research/kbound/scripts"
    runbooks = repo / "docs/research/kbound/runbooks"
    scripts.mkdir(parents=True)
    runbooks.mkdir(parents=True)
    for filename in ("audit_official_baselines.py", "official_baseline_provenance.py",
                     "official_baselines_headtohead.py"):
        shutil.copy2(REPO / "docs/research/kbound/scripts" / filename, scripts / filename)
    runbook = runbooks / "run_item11_official_baselines.sh"
    shutil.copy2(REPO / "docs/research/kbound/runbooks" / runbook.name, runbook)

    stream = fixture["output"] / "locked_stream.json"
    records = [{"condition": f"condition_{i}", "Z": [float((i + j) % 13) / 13 for j in range(11)],
                "B": 0.1 if i % 2 else -0.1, "a0": 0.5, "a_adapted": 0.6 if i % 2 else 0.4,
                "a_oracle": 0.6 if i % 2 else 0.5} for i in range(432)]
    stream.write_text(json.dumps({"records": records}), encoding="utf-8")
    stream_sha = hashlib.sha256(stream.read_bytes()).hexdigest()
    fixture["audit"]["promotion_binding"]["locked_stream_sha256"] = stream_sha
    fixture["save_audit"]()
    payload = json.loads(fixture["decisions_path"].read_text())
    payload["locked_stream_sha256"] = stream_sha
    fixture["decisions_path"].write_text(json.dumps(payload), encoding="utf-8")

    # The selected package must win over the unrelated historical default file.
    stale_default = repo / "experiments/kbound/results/per_condition_cifar10c_tent_seed0.json"
    stale_default.parent.mkdir(parents=True)
    stale_default.write_text('{"records":[]}\n', encoding="utf-8")

    # This executable exists ONLY under pytest's temp directory. Production code
    # keeps its pinned public key and exposes no test-key environment variable.
    public_key = importlib.import_module("official_baseline_provenance").NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64
    python = repo / "test_witness_python"
    python.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"sys.path[:0] = [{str(scripts)!r}, {str(REPO)!r}]\n"
        "import official_baseline_provenance as provenance\n"
        f"provenance.NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64 = {public_key!r}\n"
        "sys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n",
        encoding="utf-8",
    )
    python.chmod(0o700)
    env = {**os.environ, "KBOUND_PYTHON": str(python), "KBOUND_OFFICIAL_OUT": str(fixture["output"]),
           "KBOUND_CIFAR_STREAM": str(stream),
           "KBOUND_OFFICIAL_VERIFICATION": str(fixture["output"] / "item11_verification.json")}
    env.pop("AETTA_LOG_JSON", None)
    env.pop("POEM_LOG_JSON", None)
    fixture.update(runbook=runbook, env=env, python=python, scripts=scripts, stream=stream, stream_sha=stream_sha)
    return fixture


def run_legacy(fixture):
    return subprocess.run(["bash", str(fixture["runbook"])], env=fixture["env"],
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("legacy_source_dirs", [False, True])
def test_runbook_consumes_existing_signed_package_without_replacing_audit(legacy_cifar_package, legacy_source_dirs):
    fixture = legacy_cifar_package
    if legacy_source_dirs:
        (fixture["repo"] / "AETTA").mkdir()
        (fixture["repo"] / "external/poem/.git").mkdir(parents=True)
    audit = fixture["output"] / "OFFICIAL_BASELINE_AUDIT.json"
    original = audit.read_bytes(), fixture["decisions_path"].read_bytes()
    result = run_legacy(fixture)
    assert result.returncode == 0, result.stderr
    report = json.loads((fixture["output"] / "item11_verification.json").read_text())
    score = json.loads((fixture["output"] / "cifar10c_headtohead.json").read_text())
    assert report["status"] == "PASS"
    assert report["benchmark_complete"] is False
    assert set(report["methods"]) == {"aetta"}
    assert score["official"] == ["aetta"]
    assert score["candidate"] == "tent"
    assert score["input_sha256"] == fixture["stream_sha"]
    assert score["n_conditions"] == 432
    assert "AETTA_official" in score["rows"]
    assert "POEM_official" not in score["rows"]
    assert score["rows"]["AETTA_official"]["regret"] == 0.05
    assert (audit.read_bytes(), fixture["decisions_path"].read_bytes()) == original


def test_scorer_uses_explicit_authenticated_stream(legacy_cifar_package):
    fixture = legacy_cifar_package
    output = fixture["output"] / "direct_score.json"
    result = subprocess.run(
        [str(fixture["python"]), str(fixture["scripts"] / "official_baselines_headtohead.py"),
         "--candidate", "tent", "--stream", str(fixture["stream"]),
         "--decisions", f"aetta={fixture['decisions_path']}", "--provenance-audits",
         f"aetta={fixture['output'] / 'OFFICIAL_BASELINE_AUDIT.json'}", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    score = json.loads(output.read_text())
    assert score["input_sha256"] == fixture["stream_sha"]
    assert score["official"] == ["aetta"]


def test_scorer_rejects_two_selected_input_modes(legacy_cifar_package):
    fixture = legacy_cifar_package
    checkpoint = fixture["output"] / "checkpoint.json"
    checkpoint.write_text(json.dumps({"rows": {"tent": [
        {"condition": f"checkpoint_{i}", "Z": [float(i + j) / 40 for j in range(11)], "a0": 0.5, "aa": 0.6}
        for i in range(40)
    ]}}), encoding="utf-8")
    replay = fixture["output"] / "unexpected_checkpoint_replay"
    result = subprocess.run(
        [str(fixture["python"]), str(fixture["scripts"] / "official_baselines_headtohead.py"),
         "--candidate", "tent", "--stream", str(fixture["stream"]),
         "--checkpoint", str(checkpoint), "--checkpoint-sha256", hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         "--out-dir", str(replay)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not replay.exists()


@pytest.mark.parametrize("field, value", [("B", -0.9), ("a_oracle", 0.9)])
def test_signed_stream_cannot_score_inconsistent_saved_outcomes(legacy_cifar_package, field, value):
    fixture = legacy_cifar_package
    stream_payload = json.loads(fixture["stream"].read_text())
    stream_payload["records"][0][field] = value
    fixture["stream"].write_text(json.dumps(stream_payload), encoding="utf-8")
    selected_hash = hashlib.sha256(fixture["stream"].read_bytes()).hexdigest()
    fixture["audit"]["promotion_binding"]["locked_stream_sha256"] = selected_hash
    fixture["save_audit"]()
    payload = json.loads(fixture["decisions_path"].read_text())
    payload["locked_stream_sha256"] = selected_hash
    fixture["decisions_path"].write_text(json.dumps(payload), encoding="utf-8")
    result = run_legacy(fixture)
    assert result.returncode != 0
    assert not (fixture["output"] / "cifar10c_headtohead.json").exists()
    assert "stream" in result.stderr


@pytest.mark.parametrize("defect", ["missing_audit", "changed_log", "missing_witness", "existing_score"])
def test_runbook_refuses_invalid_evidence_before_scoring(legacy_cifar_package, defect):
    fixture = legacy_cifar_package
    if defect == "missing_audit":
        (fixture["output"] / "OFFICIAL_BASELINE_AUDIT.json").unlink()
    elif defect == "changed_log":
        (fixture["trace"] / "native_stdout.log").write_text("modified\n", encoding="utf-8")
    elif defect == "missing_witness":
        (fixture["trace"] / "native_execution_attestation.json").unlink()
    else:
        (fixture["output"] / "cifar10c_headtohead.json").write_text("preserve previous score\n", encoding="utf-8")
    result = run_legacy(fixture)
    assert result.returncode != 0
    score = fixture["output"] / "cifar10c_headtohead.json"
    if defect == "existing_score":
        assert score.read_text() == "preserve previous score\n"
    else:
        assert not score.exists()
    assert "audit" in result.stderr.lower() or "artifact" in result.stdout.lower() or "score" in result.stderr.lower()
