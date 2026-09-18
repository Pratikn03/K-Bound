"""Terminal integration tests: synthetic inputs, real full GBR fitting, no native runs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/kbound/scripts/task3_common_analysis.py"
FIXTURE = Path(__file__).with_name("test_task3_common_scoring.py")
spec = importlib.util.spec_from_file_location("common_scoring_fixture", FIXTURE)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def raw_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


@pytest.fixture
def inputs(tmp_path):
    manifest, development, fit, calibration, features, outcomes = fixture_module.panel()
    values = {"manifest": manifest, "development-features": development,
              "fit-outcomes": fit, "calibration-outcomes": calibration,
              "score-features": features, "native-estimates": [], "score-outcomes": outcomes}
    return {key: write_json(tmp_path / (key + ".json"), value) for key, value in values.items()}


def command(phase, paths, output, expected="none"):
    keys = ("manifest", "development-features", "fit-outcomes", "calibration-outcomes",
            "score-features", "native-estimates") if phase == "decide" else (
            "manifest", "decision-packet", "score-outcomes")
    args = [sys.executable, str(SCRIPT), phase, "--output-dir", str(output)]
    for key in keys:
        args += ["--" + key, str(paths[key]), "--" + key + "-sha256", raw_hash(paths[key])]
    if phase == "decide":
        args += ["--expected-methods", expected]
    return args


def invoke(args):
    return subprocess.run(args, text=True, capture_output=True, timeout=45)


def success(args):
    result = invoke(args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_terminal_decide_then_score_and_accurate_separate_fit_timing(inputs, tmp_path):
    result = success(command("decide", inputs, tmp_path / "decision"))
    packet_path = tmp_path / "decision/decision_packet.json"
    receipt = json.loads((tmp_path / "decision/receipt.json").read_text())
    assert result["status"] == "DECISIONS_WRITTEN_NOT_BENCHMARK_EVIDENCE"
    assert receipt["timing"]["total_seconds"] > receipt["timing"]["fit_and_calibration_seconds"] >= receipt["timing"]["predictor_fit_seconds"] > 0
    assert receipt["timing"]["predictor_fit_call_count"] == 1
    assert "calibration" in receipt["timing"]["predictor_fit_scope"]
    assert receipt["outputs"]["decision_packet.json"]["sha256_bytes"] == raw_hash(packet_path)
    packet = json.loads(packet_path.read_text())
    assert receipt["decision_packet_sha256_canonical_body"] == packet["packet_sha256"]
    assert receipt["decision_packet_sha256_canonical_body"] != raw_hash(packet_path)
    assert receipt["expected_native_methods"] == []
    assert receipt["independently_authenticated_native_execution"] is False
    assert packet_path.stat().st_mode & 0o222 == 0
    assert (tmp_path / "decision").stat().st_mode & 0o222 == 0
    score_paths = {**inputs, "decision-packet": packet_path}
    success(command("score", score_paths, tmp_path / "scoring"))
    metrics = json.loads((tmp_path / "scoring/metrics.json").read_text())
    assert metrics["policies"]["point_benefit"]["mean_accuracy"] == 0.625
    assert "coverage" not in metrics["policies"]["KGA"]
    scored_receipt = json.loads((tmp_path / "scoring/receipt.json").read_text())
    assert scored_receipt["timing"]["predictor_fit_seconds"] == 0
    assert scored_receipt["inputs"]["decision-packet"]["sha256_bytes"] == raw_hash(packet_path)


def test_full_refit_decisions_are_byte_identical_after_scored_outcome_interventions(inputs, tmp_path):
    original = json.loads(inputs["score-outcomes"].read_text())
    original_hashes = {key: raw_hash(path) for key, path in inputs.items() if key != "score-outcomes"}
    baseline = None
    for run_index, changed_indexes in enumerate(([], [0], [0, 1], [2, 3])):
        altered = deepcopy(original)
        for index in changed_indexes:
            altered[index]["frozen_correct"] = [0] * 4
            altered[index]["candidate_correct"] = [1] * 4
        write_json(inputs["score-outcomes"], altered)
        decision_dir = tmp_path / f"decision-{run_index}"
        success(command("decide", inputs, decision_dir))
        packet = decision_dir / "decision_packet.json"
        if baseline is None:
            baseline = packet.read_bytes()
        assert packet.read_bytes() == baseline
        receipt = json.loads((decision_dir / "receipt.json").read_text())
        assert "score-outcomes" not in receipt["inputs"]
        assert receipt["timing"]["predictor_fit_call_count"] == 1
        success(command("score", {**inputs, "decision-packet": packet}, tmp_path / f"score-{run_index}"))
    assert {key: raw_hash(inputs[key]) for key in original_hashes} == original_hashes


def test_decide_does_not_open_even_malformed_separate_score_file(inputs, tmp_path):
    inputs["score-outcomes"].write_text("not JSON and must not be consumed")
    success(command("decide", inputs, tmp_path / "decision"))
    result = invoke(command("decide", inputs, tmp_path / "other") + [
        "--score-outcomes", str(inputs["score-outcomes"])])
    assert result.returncode != 0
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("expected,actual", [("AETTA", []), ("Baek_ALine", []),
    ("none", ["AETTA"]), ("AETTA", ["AETTA", "Baek_ALine"]), ("AETTA,AETTA", ["AETTA"]),
    ("Kim_TTALine", [])])
def test_expected_native_method_set_is_explicit_and_exact(inputs, tmp_path, expected, actual):
    features = json.loads(inputs["score-features"].read_text())
    native = [fixture_module.native_pair(row["identity"], method) for method in actual for row in features]
    write_json(inputs["native-estimates"], native)
    result = invoke(command("decide", inputs, tmp_path / "decision", expected))
    assert result.returncode != 0
    assert not (tmp_path / "decision").exists()


def test_both_native_pairs_flow_into_explicit_local_controllers(inputs, tmp_path):
    features = json.loads(inputs["score-features"].read_text())
    native = [fixture_module.native_pair(row["identity"], method)
              for method in ("AETTA", "Baek_ALine") for row in features]
    write_json(inputs["native-estimates"], native)
    success(command("decide", inputs, tmp_path / "decision", "AETTA,Baek_ALine"))
    packet = json.loads((tmp_path / "decision/decision_packet.json").read_text())
    assert all(row["actions"]["AETTA_controller"] == "ADAPT" for row in packet["rows"])
    assert "local estimated-benefit" in packet["policy_descriptions"]["Baek_ALine_controller"]


def test_bad_byte_digest_existing_output_and_symlink_outputs_fail_closed(inputs, tmp_path):
    args = command("decide", inputs, tmp_path / "bad-digest")
    args[args.index("--manifest-sha256") + 1] = "0" * 64
    assert invoke(args).returncode != 0
    assert not (tmp_path / "bad-digest").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_text("preserve")
    assert invoke(command("decide", inputs, existing)).returncode != 0
    assert sentinel.read_text() == "preserve"
    linked = tmp_path / "linked"
    linked.symlink_to(existing, target_is_directory=True)
    assert invoke(command("decide", inputs, linked)).returncode != 0
    assert invoke(command("decide", inputs, linked / "new-output")).returncode != 0
    assert not (existing / "new-output").exists()


@pytest.mark.parametrize("kind", ["mismatch", "malformed", "changed_packet"])
def test_score_rejects_unsealed_or_changed_packet(inputs, tmp_path, kind):
    success(command("decide", inputs, tmp_path / "decision"))
    real_packet = tmp_path / "decision/decision_packet.json"
    packet_copy = tmp_path / "packet-copy.json"
    packet_copy.write_bytes(real_packet.read_bytes())
    score_inputs = {**inputs, "decision-packet": packet_copy}
    args = command("score", score_inputs, tmp_path / "score")
    if kind == "malformed":
        packet_copy.write_text('{"schema": "not-a-packet"}')
        args = command("score", score_inputs, tmp_path / "score")
    elif kind == "changed_packet":
        value = json.loads(packet_copy.read_text())
        value["rows"][0]["actions"]["KGA"] = "ADAPT"
        write_json(packet_copy, value)
        args = command("score", score_inputs, tmp_path / "score")
    else:
        packet_copy.write_bytes(packet_copy.read_bytes() + b"\n")
    assert invoke(args).returncode != 0
    assert not (tmp_path / "score").exists()


def test_json_duplicate_keys_and_symlink_input_rejected(inputs, tmp_path):
    bad_manifest = tmp_path / "duplicate.json"
    bad_manifest.write_text('{"schema": "one", "schema": "two"}')
    args = command("decide", {**inputs, "manifest": bad_manifest}, tmp_path / "duplicate-out")
    assert invoke(args).returncode != 0
    link = tmp_path / "manifest-link.json"
    link.symlink_to(inputs["manifest"])
    args = command("decide", {**inputs, "manifest": link}, tmp_path / "link-out")
    assert invoke(args).returncode != 0
    assert not (tmp_path / "link-out").exists()


def load_analysis():
    spec = importlib.util.spec_from_file_location("common_analysis_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_directory_substitution_is_detected_without_touching_replacement(tmp_path, monkeypatch):
    api = load_analysis()
    output = tmp_path / "output"
    retained = tmp_path / "retained-original"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "sentinel").write_text("preserve")
    real_write = api._write_file

    def substitute_after_artifact(fd, name, data):
        real_write(fd, name, data)
        if name == "decision_packet.json":
            output.rename(retained)
            output.symlink_to(replacement, target_is_directory=True)

    monkeypatch.setattr(api, "_write_file", substitute_after_artifact)
    with pytest.raises(ValueError, match="identity"):
        api._write_output(output, "decision_packet.json", {"retained": True}, {"timing": {}}, time.perf_counter())
    assert json.loads((retained / "decision_packet.json").read_text()) == {"retained": True}
    assert list(replacement.iterdir()) == [replacement / "sentinel"]
    assert (replacement / "sentinel").read_text() == "preserve"
    assert output.is_symlink()


def test_dataless_flag_fails_before_any_input_read(tmp_path, monkeypatch):
    api = load_analysis()
    path = write_json(tmp_path / "input.json", {"valid": "JSON"})
    expected = raw_hash(path)
    observed = path.stat()
    real_fdopen = os.fdopen
    read_attempts = []

    class GuardedInput:
        def __init__(self, *args, **kwargs):
            self.handle = real_fdopen(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, *args):
            read_attempts.append(True)
            return self.handle.read(*args)

    fake_stat = SimpleNamespace(**{key: getattr(observed, key) for key in (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")}, st_flags=0x40000000)
    monkeypatch.delattr(api.stat, "SF_DATALESS", raising=False)
    monkeypatch.setattr(api.os, "fstat", lambda descriptor: fake_stat)
    monkeypatch.setattr(api.os, "fdopen", GuardedInput)
    with pytest.raises(ValueError, match="dataless"):
        api._read_bound(path, expected)
    assert read_attempts == []
