from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "docs/research/kbound/scripts"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(params=["aetta", "poem"])
def signed_cli_evidence(tmp_path: Path, monkeypatch, request):
    """Synthetic execution with genuine signatures; no production signing key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    monkeypatch.syspath_prepend(str(SCRIPTS))
    provenance = importlib.import_module("official_baseline_provenance")
    spec = importlib.util.spec_from_file_location("audit_cli_under_test", SCRIPTS / "audit_official_baselines.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    method = request.param
    source = tmp_path / "external" / f"{method}_official"
    source.mkdir(parents=True)
    (source / "main.py").write_text(
        "import json\nprint(json.dumps({f'condition_{i}': 'adapt' for i in range(432)}))\n",
        encoding="utf-8",
    )
    output = tmp_path / "results"
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_official_native.py"), "--repo", str(tmp_path),
         "--out-dir", str(output), "--method", method, "--", sys.executable, "main.py"],
        check=False, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    trace = Path(json.loads(run.stdout)["output"])
    invocation = json.loads((trace / "native_invocation.json").read_text())
    receipt = json.loads((trace / "native_runner_receipt.json").read_text())
    completion = json.loads((trace / "native_completion.json").read_text())
    witness = Ed25519PrivateKey.generate()
    public = witness.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    monkeypatch.setattr(provenance, "NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64",
                        base64.b64encode(public).decode("ascii"))
    claim = {
        "method": method, "runner_receipt_sha256": digest(trace / "native_runner_receipt.json"),
        "schema": "kbound-official-native-attestation-v1",
        "witness_key_id": "kbound-official-native-witness-2026-09-02",
    }
    claim_bytes = (json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n").encode()
    attestation = {**claim, "signature_b64": base64.b64encode(witness.sign(claim_bytes)).decode("ascii")}
    (trace / "native_execution_attestation.json").write_text(json.dumps(attestation), encoding="utf-8")
    decisions = {f"condition_{i}": "adapt" for i in range(432)}
    stream = output / "locked_stream.json"
    stream.write_text(json.dumps({"records": [{"condition": item} for item in decisions]}), encoding="utf-8")
    environment = output / "environment_receipt.json"
    environment.write_text('{"environment":"synthetic-only"}\n', encoding="utf-8")
    toolchain = output / "toolchain_receipt.json"
    toolchain.write_text('{"toolchain":"synthetic-only"}\n', encoding="utf-8")
    bindings = {
        "locked_stream_sha256": digest(stream), "environment_receipt_sha256": digest(environment),
        "toolchain_receipt_sha256": digest(toolchain),
    }
    decisions_sha = hashlib.sha256(json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    logs = {
        "successful": True, "unavailable": [], "failure_markers": [],
        "sha256": {path.relative_to(tmp_path).as_posix(): digest(path) for path in trace.iterdir()},
        "completion_verified": True, "runner_receipt_verified": True, "execution_attested": True,
        "completion": completion, "runner_receipt": receipt,
        "execution_attestation": attestation, "invocation": invocation,
    }
    audit = {
        "schema_version": 3, "promotion_binding": {"condition_count": 432, **bindings},
        "methods": {method: {
            "method": method, "official_label_allowed": True,
            "checks": dict.fromkeys(provenance.REQUIRED_CHECKS[method], True),
            "decision_count": 432, "decision_payload_sha256": decisions_sha,
            "source_tree_sha256": invocation["source"]["tree_sha256"],
            "upstream_commit": "4" * 40, "upstream_remote": "https://example.invalid/synthetic-fixture",
            "native_logs": logs,
        }},
    }
    audit_path = output / "OFFICIAL_BASELINE_AUDIT.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    decisions_path = output / f"{method}_decisions.json"
    payload = {
        "schema_version": 3, "method": method,
        "label": "official_implementation_under_protocol_adapter", "official_label_allowed": True,
        "decisions": decisions, "decisions_sha256": decisions_sha,
        "source_log_sha256": digest(trace / "native_stdout.log"),
        "provenance_audit_sha256": digest(audit_path), **bindings,
    }
    decisions_path.write_text(json.dumps(payload), encoding="utf-8")
    attempt = 0

    def invoke():
        nonlocal attempt
        attempt += 1
        report = output / f"verification_{attempt}.json"
        code = cli.main(["--repo", str(tmp_path), "--out-dir", str(output),
                         "--method", method, "--require-promotable", "--output", str(report)])
        return code, json.loads(report.read_text())

    def save_audit():
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        payload["provenance_audit_sha256"] = digest(audit_path)
        decisions_path.write_text(json.dumps(payload), encoding="utf-8")

    return {
        "repo": tmp_path, "source": source, "trace": trace, "output": output,
        "method": method, "audit": audit, "logs": logs, "invoke": invoke,
        "save_audit": save_audit, "decisions_path": decisions_path,
    }


def test_cli_accepts_witnessed_artifacts_without_claiming_benchmark(signed_cli_evidence) -> None:
    fixture = signed_cli_evidence
    preserved = {path: path.read_bytes() for root in (fixture["source"], fixture["output"])
                 for path in root.rglob("*") if path.is_file()}
    code, report = fixture["invoke"]()
    assert code == 0, report
    assert report["status"] == "PASS"
    assert report["official_label_allowed"] is True
    assert report["native_execution_launched"] is False
    assert report["benchmark_complete"] is False
    assert report["methods"][fixture["method"]]["decision_count"] == 432
    assert all(path.read_bytes() == content for path, content in preserved.items())


@pytest.mark.parametrize("target", [
    "native_stdout.log", "native_invocation.json", "source", "decisions",
    "locked_stream.json", "environment_receipt.json", "toolchain_receipt.json",
])
def test_cli_rejects_changed_physical_evidence(signed_cli_evidence, target) -> None:
    fixture = signed_cli_evidence
    assert fixture["invoke"]()[0] == 0
    if target == "source":
        (fixture["source"] / "main.py").write_text("print('modified source')\n", encoding="utf-8")
    elif target == "decisions":
        path = fixture["decisions_path"]
        payload = json.loads(path.read_text())
        payload["decisions"]["condition_0"] = "freeze"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif target in {"locked_stream.json", "environment_receipt.json", "toolchain_receipt.json"}:
        (fixture["output"] / target).write_text('{"changed":true}\n', encoding="utf-8")
    else:
        (fixture["trace"] / target).write_text('{"changed":true}\n', encoding="utf-8")
    code, report = fixture["invoke"]()
    assert code == 2, report
    assert report["status"] == "OPEN"
    assert report["methods"][fixture["method"]]["blockers"]


def test_cli_rejects_unsigned_zero_exit(signed_cli_evidence) -> None:
    fixture = signed_cli_evidence
    del fixture["logs"]["execution_attestation"]
    fixture["save_audit"]()
    code, report = fixture["invoke"]()
    assert code == 2, report
    assert report["official_label_allowed"] is False
    assert "attestation" in " ".join(report["methods"][fixture["method"]]["blockers"])


def test_cli_rejects_decisions_bound_to_a_different_audit(signed_cli_evidence) -> None:
    fixture = signed_cli_evidence
    assert fixture["invoke"]()[0] == 0
    path = fixture["decisions_path"]
    payload = json.loads(path.read_text())
    payload["provenance_audit_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, report = fixture["invoke"]()
    assert code == 2, report
    assert report["status"] == "OPEN"


@pytest.mark.parametrize("filename", [
    "native_invocation.json", "native_runner_receipt.json",
    "native_completion.json", "native_execution_attestation.json",
])
def test_cli_rejects_rehashed_control_file_with_stale_embedded_record(signed_cli_evidence, filename) -> None:
    fixture = signed_cli_evidence
    assert fixture["invoke"]()[0] == 0
    path = fixture["trace"] / filename
    control = json.loads(path.read_text())
    if filename == "native_invocation.json":
        control["command"] = [sys.executable, "-c", "pass"]
    else:
        control["unbound_change"] = True
    path.write_text(json.dumps(control), encoding="utf-8")
    fixture["logs"]["sha256"][path.relative_to(fixture["repo"]).as_posix()] = digest(path)
    fixture["save_audit"]()
    code, report = fixture["invoke"]()
    assert code == 2, report
    assert report["status"] == "OPEN"


@pytest.mark.parametrize("filename", [
    "native_invocation.json", "native_runner_receipt.json",
    "native_completion.json", "native_execution_attestation.json",
])
def test_cli_rejects_omitted_physical_trace_control(signed_cli_evidence, filename) -> None:
    fixture = signed_cli_evidence
    assert fixture["invoke"]()[0] == 0
    path = fixture["trace"] / filename
    del fixture["logs"]["sha256"][path.relative_to(fixture["repo"]).as_posix()]
    path.unlink()
    fixture["save_audit"]()
    code, report = fixture["invoke"]()
    assert code == 2, report
    assert report["status"] == "OPEN"
