from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO / "docs/research/kbound/scripts/audit_official_baselines.py"
CONVERTER = REPO / "docs/research/kbound/runbooks/convert_official_logs_to_decisions.py"
NATIVE_RUNNER = REPO / "docs/research/kbound/scripts/run_official_native.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_official_baselines", AUDIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_native_runner_help_is_importable() -> None:
    proc = subprocess.run(
        [sys.executable, str(NATIVE_RUNNER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--repo" in proc.stdout


def test_tree_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = module.tree_hash(source)
    assert first == module.tree_hash(source)
    (source / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert module.tree_hash(source) != first


def test_tree_hash_excludes_local_data_and_run_logs(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    expected = module.tree_hash(source)
    for directory, name in (
        ("cached_data", "cache.pkl"),
        ("dataset", "sample.npy"),
        ("raw_logs", "run.log"),
    ):
        local_dir = source / directory
        local_dir.mkdir()
        (local_dir / name).write_bytes(b"machine-local payload")
    assert module.tree_hash(source) == expected


def test_tree_hash_authenticates_dataset_loader_code_not_image_payloads(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    dataset = source / "dataset"
    dataset.mkdir(parents=True)
    loader = dataset / "selectedRotateImageFolder.py"
    loader.write_text("PREPROCESS = 'reference'\n", encoding="utf-8")
    before = module.tree_hash(source)
    loader.write_text("PREPROCESS = 'changed'\n", encoding="utf-8")
    after = module.tree_hash(source)
    assert before != after
    (dataset / "unseen.JPEG").write_bytes(b"not source code")
    assert module.tree_hash(source) == after


def test_native_runner_does_not_attest_changed_dataset_loader(tmp_path: Path) -> None:
    source = tmp_path / "external/poem_official"
    (source / "dataset").mkdir(parents=True)
    loader = source / "dataset/loader.py"
    loader.write_text("ORDER = 'locked'\n", encoding="utf-8")
    output = tmp_path / "results"
    command = [sys.executable, "-c",
               "from pathlib import Path; Path('dataset/loader.py').write_text('ORDER = 0\\n')"]
    result = subprocess.run(
        [sys.executable, str(NATIVE_RUNNER), "--repo", str(tmp_path),
         "--out-dir", str(output), "--method", "poem", "--", *command],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    target = output / "poem_imagenetc"
    assert (target / "native_invocation.json").is_file()
    assert not (target / "native_runner_receipt.json").exists()
    assert not (target / "native_completion.json").exists()


def test_native_aetta_runner_uses_same_checkout_as_task3_preflight(tmp_path: Path) -> None:
    source = tmp_path / "external/aetta_official"
    source.mkdir(parents=True)
    (source / "identity.txt").write_text("selected official checkout", encoding="utf-8")
    legacy = tmp_path / "AETTA"
    legacy.mkdir()
    (legacy / "identity.txt").write_text("historical checkout", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(NATIVE_RUNNER), "--repo", str(tmp_path),
         "--out-dir", str(tmp_path / "results"), "--method", "aetta", "--",
         sys.executable, "-c", "from pathlib import Path; print(Path('identity.txt').read_text())"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "results/aetta_native"
    assert (output / "native_stdout.log").read_text().strip() == "selected official checkout"
    invocation = json.loads((output / "native_invocation.json").read_text())
    assert invocation["source"]["path"] == "external/aetta_official"
    assert invocation["working_directory"] == "external/aetta_official"


def test_tree_hash_excludes_static_public_assets(tmp_path: Path) -> None:
    module = load_audit_module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\n", encoding="utf-8")
    expected = module.tree_hash(source)
    public = source / "public/fonts"
    public.mkdir(parents=True)
    (public / "font.woff2").write_bytes(b"static UI asset")
    assert module.tree_hash(source) == expected


def test_native_log_traceback_fails_closed(tmp_path: Path) -> None:
    module = load_audit_module()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("Traceback (most recent call last)\n", encoding="utf-8")
    result = module.native_logs(logs, repo=tmp_path)
    assert result["count"] == 1
    assert result["sha256"].keys() == {"logs/run.log"}
    assert result["failure_markers"] == ["logs/run.log"]
    assert result["successful"] is False


def test_native_logs_reject_paths_outside_root_binding(tmp_path: Path) -> None:
    module = load_audit_module()
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.log").write_text("complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the repository-root binding"):
        module.native_logs(outside, repo=repository)


def test_native_logs_fail_closed_when_a_log_cannot_be_hashed(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_audit_module()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("complete\n", encoding="utf-8")

    def unavailable(_path: Path) -> str:
        raise OSError(89, "operation canceled")

    monkeypatch.setattr(module, "sha256_file", unavailable)
    result = module.native_logs(logs, repo=tmp_path)

    assert result["sha256"] == {}
    assert result["unavailable"] == ["logs/run.log"]
    assert result["successful"] is False


def test_saved_audit_uses_only_repo_relative_provenance_paths() -> None:
    artifact_path = (
        REPO
        / "experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 2
    assert artifact["provenance_path_binding"] == {
        "schema": "git-repository-relative-posix-v1",
        "root": ".",
        "root_role": "git_repository_root",
        "content_scope": "working_tree_at_generation",
        "generation_base_git_head": artifact["provenance_path_binding"][
            "generation_base_git_head"
        ],
    }
    assert len(artifact["provenance_path_binding"]["generation_base_git_head"]) == 40

    absolute_strings: list[str] = []

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.startswith("/"):
                    absolute_strings.append(key)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, str) and value.startswith("/"):
            absolute_strings.append(value)

    inspect(artifact)
    assert absolute_strings == []


def test_converter_writes_unverified_provenance_wrapper(tmp_path: Path) -> None:
    stream = tmp_path / "stream.json"
    logs = tmp_path / "logs.json"
    output = tmp_path / "decisions.json"
    stream.write_text(
        json.dumps({"records": [{"condition": "c1"}, {"condition": "c2"}]}),
        encoding="utf-8",
    )
    logs.write_text(
        json.dumps(
            {
                "c1": {"est_acc_adapted": 0.8, "est_acc_frozen": 0.7},
                "c2": {"est_acc_adapted": 0.6, "est_acc_frozen": 0.7},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--method",
            "aetta",
            "--logs",
            str(logs),
            "--stream",
            str(stream),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["official_label_allowed"] is False
    assert result["label"] == "external_protocol_adapter_unverified"
    assert result["decisions"] == {"c1": "adapt", "c2": "freeze"}
    assert len(result["source_log_sha256"]) == 64
    assert len(result["locked_stream_sha256"]) == 64


def test_converter_rejects_incomplete_locked_stream(tmp_path: Path) -> None:
    stream = tmp_path / "stream.json"
    logs = tmp_path / "logs.json"
    output = tmp_path / "decisions.json"
    stream.write_text(
        json.dumps({"records": [{"condition": "c1"}, {"condition": "c2"}]}),
        encoding="utf-8",
    )
    logs.write_text(json.dumps({"c1": "adapt"}), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--method",
            "poem",
            "--logs",
            str(logs),
            "--stream",
            str(stream),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "missing" in proc.stderr
    assert not output.exists()


@pytest.fixture(params=["aetta", "poem"])
def witnessed_native_audit(tmp_path: Path, monkeypatch, request):
    """Exercise the real runner and Ed25519 verifier with a test-only witness."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    spec = importlib.util.spec_from_file_location(
        "native_provenance_under_test",
        REPO / "docs/research/kbound/scripts/official_baseline_provenance.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method = request.param
    source = tmp_path / "external" / f"{method}_official"
    source.mkdir(parents=True)
    # This is a tiny synthetic command, never a native-method reproduction.
    (source / "main.py").write_text(
        "import json\nprint(json.dumps({f'c{i}': 'adapt' for i in range(432)}))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(NATIVE_RUNNER), "--repo", str(tmp_path),
         "--out-dir", str(tmp_path / "results"), "--method", method,
         "--", sys.executable, "main.py"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    output = Path(json.loads(proc.stdout)["output"])
    invocation = json.loads((output / "native_invocation.json").read_text())
    receipt = json.loads((output / "native_runner_receipt.json").read_text())
    completion = json.loads((output / "native_completion.json").read_text())
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw,
    )
    monkeypatch.setattr(module, "NATIVE_EXECUTION_WITNESS_PUBLIC_KEY_B64",
                        base64.b64encode(public_key).decode("ascii"))
    claim = {
        "method": method,
        "runner_receipt_sha256": hashlib.sha256(
            (output / "native_runner_receipt.json").read_bytes()
        ).hexdigest(),
        "schema": "kbound-official-native-attestation-v1",
        "witness_key_id": "kbound-official-native-witness-2026-09-02",
    }
    signed_bytes = (json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n").encode()
    attestation = {**claim, "signature_b64": base64.b64encode(
        private_key.sign(signed_bytes)
    ).decode("ascii")}
    decisions = {f"c{i}": "adapt" for i in range(432)}
    source_log = hashlib.sha256((output / "native_stdout.log").read_bytes()).hexdigest()
    kwargs = {
        "method": method, "decisions": decisions, "source_log_sha256": source_log,
        "locked_stream_sha256": "1" * 64, "environment_receipt_sha256": "2" * 64,
        "toolchain_receipt_sha256": "3" * 64,
    }
    logs = {
        "successful": True, "unavailable": [], "failure_markers": [],
        "sha256": {path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in output.iterdir() if path.is_file()},
        "completion_verified": True, "runner_receipt_verified": True,
        "execution_attested": True, "completion": completion,
        "runner_receipt": receipt, "execution_attestation": attestation,
        "invocation": invocation,
    }
    audit = {
        "schema_version": 3,
        "promotion_binding": {"condition_count": 432, **{
            key: kwargs[key] for key in (
                "locked_stream_sha256", "environment_receipt_sha256", "toolchain_receipt_sha256"
            )}},
        "methods": {method: {
            "method": method, "official_label_allowed": True,
            "checks": dict.fromkeys(module.REQUIRED_CHECKS[method], True),
            "decision_count": 432,
            "decision_payload_sha256": hashlib.sha256(json.dumps(
                decisions, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "source_tree_sha256": invocation["source"]["tree_sha256"],
            "upstream_commit": "4" * 40,
            "upstream_remote": "https://example.invalid/synthetic-native-fixture",
            "native_logs": logs,
        }},
    }
    path = tmp_path / "audit.json"

    def validate():
        path.write_text(json.dumps(audit), encoding="utf-8")
        return module.validate_promotable_audit(path, **kwargs)

    return audit["methods"][method], kwargs, validate


def test_promotion_accepts_unchanged_witnessed_invocation(witnessed_native_audit) -> None:
    _, _, validate = witnessed_native_audit
    assert len(validate()) == 64


@pytest.mark.parametrize("mutation", ["missing_invocation", "changed_argv", "changed_source"])
def test_promotion_rejects_unbound_native_execution(witnessed_native_audit, mutation) -> None:
    record, _, validate = witnessed_native_audit
    # The same authenticated baseline must pass before the independent claim changes.
    assert len(validate()) == 64
    if mutation == "missing_invocation":
        del record["native_logs"]["invocation"]
    elif mutation == "changed_argv":
        record["native_logs"]["invocation"]["command"] = [sys.executable, "-c", "pass"]
    else:
        record["source_tree_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="invocation|source"):
        validate()


def test_promotion_rejects_trace_metadata_as_converted_output(witnessed_native_audit) -> None:
    record, kwargs, validate = witnessed_native_audit
    assert len(validate()) == 64
    kwargs["source_log_sha256"] = record["native_logs"]["runner_receipt"]["invocation_sha256"]
    with pytest.raises(ValueError, match="converted source log|output"):
        validate()


def test_promotion_rejects_zero_exit_without_independent_witness(witnessed_native_audit) -> None:
    record, _, validate = witnessed_native_audit
    del record["native_logs"]["execution_attestation"]
    with pytest.raises(ValueError, match="attestation"):
        validate()
