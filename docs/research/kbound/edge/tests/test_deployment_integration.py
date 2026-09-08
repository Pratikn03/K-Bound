"""Synthetic deployment callers: loss of authority must never issue a certificate."""

import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from kbound_edge import model, replay
from kbound_edge.dashboard import LiveDashboard, annotate_frame
from kbound_edge.logging import WindowLogger, read_jsonl
from kbound_edge.profiling import profile_runtime
from kbound_edge.shadow_runtime import ShadowController
from kbound_edge.tent_adapter import EpisodicTentAdapter

from .test_benefit_authority_boundary import identities, seal
from .test_benefit_authority_boundary import pair as pair_fixture
from .test_real_reporting import report_inputs as report_fixture

pair = pair_fixture
report_inputs = report_fixture

SCRIPTS = Path(__file__).parents[1] / "scripts"
CALLERS = ["06_replay_heldout", "07_replay_replication", "07_shadow_live", "12_profile_real_run"]


@pytest.fixture(autouse=True)
def close_loggers_on_test_failure(monkeypatch):
    """Keep expected RED crashes from leaking log descriptors into later tests."""
    opened = []
    original = WindowLogger.__init__
    def track(self, *args, **kwargs):
        original(self, *args, **kwargs)
        opened.append(self)
    monkeypatch.setattr(WindowLogger, "__init__", track)
    yield
    for logger in opened:
        logger.close()


def tiny_model(*args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 4),
    ).eval()


def caller(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_unavailable(d):
    assert d["decision"] == "abstain"
    assert d["availability"] == "unavailable"
    assert d["certificate_status"] == "not_issued"
    assert d["model_action"] == "retain_frozen"
    for key in ("bhat", "eps", "lower", "upper", "authority_sha256", "estimator_sha256",
                "metadata_sha256", "protocol_sha256", "evidence_schema_version", "frozen_model_sha256",
                "candidate_sha256", "fit_sha256", "calibration_sha256", "runtime_versions"):
        assert d[key] is None, key
    json.dumps(d, allow_nan=False)


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    f0 = tiny_model()
    checkpoint = tmp_path / "frozen.pt"
    torch.save(f0.state_dict(), checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(model, "build_model", tiny_model)
    cfg = {"protocol": "edge_real_phone_v1", "seed": 1, "num_classes": 4,
           "device": "cpu", "image_size": 8, "window_size": 2, "alpha": .1,
           "adapter": {"lr": .001, "steps": 1},
           "paths": {"model": str(checkpoint), "kga_edge": str(tmp_path / "missing.joblib"),
                     "kga_edge_meta": str(tmp_path / "missing.json"),
                     "windows_dir": str(tmp_path / "windows"),
                     "heldout_log": str(tmp_path / "heldout.jsonl"),
                     "replication_log": str(tmp_path / "replication.jsonl"),
                     "shadow_log": str(tmp_path / "shadow.jsonl"),
                     "results_dir": str(tmp_path / "results")}}
    sh = {"window_size": 2, "log_every": 1, "policies": {"conf_tau": .5, "entropy_tau": .05},
          "source": {"kind": "fake", "regime_plan": [["clean", 0., 1]],
                     "n_frames_per_condition": 2, "seed": 1}}
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    for split, sid in [("heldout", "S07"), ("replication", "S09"), ("calibration_conformal", "S05")]:
        folder = tmp_path / "windows" / split
        folder.mkdir(parents=True)
        np.savez(folder / f"{sid}_test.npz", frames=frames, window_id="synthetic",
                 source_hashes=np.array(["synthetic"]), labels=np.array([0, 0]))
    return cfg, sh, expected, f0, frames


@pytest.mark.parametrize("name", CALLERS)
@pytest.mark.parametrize("expected", [None, "bad", "0" * 64])
def test_callers_stop_before_loading_untrusted_frozen(name, expected, setup_runtime, monkeypatch):
    cfg, sh, _, _, _ = setup_runtime
    module = caller(name, monkeypatch)
    monkeypatch.setattr(module.C, "load_config", lambda path: sh if "calibration" in path or "shadow" in path else cfg)
    monkeypatch.setattr(sys, "argv", [name] if expected is None else [name, "--expected-frozen-sha256", expected])
    def forbidden(*args, **kwargs):
        pytest.fail("untrusted checkpoint reached deserialization")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="FROZEN_AUTHORITY_INVALID"):
        module.main()


@pytest.mark.parametrize("name", CALLERS)
@pytest.mark.parametrize("fault", ["missing", "malformed", "mismatch", "unreadable", "artifact", "metadata", "gadget", "frozen_scope", "valid", "nan_evidence", "infinite_evidence", "missing_feature", "extraction_failure", "invalid_update_norm"])
def test_actual_callers_preserve_unavailable(name, fault, setup_runtime, tmp_path, monkeypatch, pair):
    cfg, sh, expected, _, frames = setup_runtime
    module = caller(name, monkeypatch)
    monkeypatch.setattr(module.C, "load_config", lambda path: sh if "calibration" in path or "shadow" in path else cfg)
    from types import SimpleNamespace

    from kbound_edge import dataset, real_manifest
    monkeypatch.setattr(dataset, "build_conditions", lambda *a, **k: [SimpleNamespace(frames=frames)])
    monkeypatch.setattr(real_manifest, "expected_windows", lambda *a: [{"window_id": "synthetic"}])
    # Bootstrap resampling requires a scientific session design outside this fixture.
    from kbound_edge import metrics
    monkeypatch.setattr(metrics, "bootstrap_real_metrics", lambda **kwargs: {})
    argv = [name, "--expected-frozen-sha256", expected]
    active = identities()
    active["frozen_model_sha256"] = expected
    sentinel = tmp_path / "executed"
    if fault != "missing":
        authority = tmp_path / "authority.json"
        authority.write_bytes(b"not-json")
        expected_authority = "bad" if fault == "malformed" else "0" * 64
        if fault in {"artifact", "metadata", "gadget", "frozen_scope", "valid", "nan_evidence", "infinite_evidence", "missing_feature", "extraction_failure", "invalid_update_norm"}:
            for obj in (pair["payload"], pair["metadata"], pair["authority"]):
                obj["identities"] = dict(active)
            seal(pair)
            authority.write_bytes(pair["authority_bytes"])
            expected_authority = pair["expected"]
            cfg["paths"]["kga_edge"] = str(pair["path"])
            cfg["paths"]["kga_edge_meta"] = str(pair["metadata_path"])
            if fault == "artifact":
                pair["path"].write_bytes(b"bad-artifact")
            elif fault == "metadata":
                pair["metadata_path"].write_bytes(b"bad-metadata")
            elif fault == "gadget":
                import joblib
                class Gadget:
                    def __reduce__(self):
                        return os.system, (f"touch {sentinel}",)
                joblib.dump(Gadget(), pair["path"])
            elif fault == "frozen_scope":
                active["frozen_model_sha256"] = "f" * 64
        argv += ["--benefit-authority", str(authority), "--expected-authority-sha256",
                 expected_authority, "--active-identities", json.dumps(active)]
        if fault == "unreadable":
            authority.unlink()
    monkeypatch.setattr(sys, "argv", argv)
    if fault in {"nan_evidence", "infinite_evidence", "missing_feature", "extraction_failure"}:
        from kbound_edge import profiling
        def invalid_evidence(*args):
            if fault == "extraction_failure":
                raise ValueError("synthetic extraction failure")
            if fault == "missing_feature":
                return np.array([None] + [0.] * 13, dtype=object)
            return np.full(14, np.nan if fault == "nan_evidence" else np.inf)
        monkeypatch.setattr(replay, "edge_evidence_vector", invalid_evidence)
        monkeypatch.setattr(profiling, "edge_evidence_vector", invalid_evidence)
    if fault == "invalid_update_norm":
        original_adapt = EpisodicTentAdapter.adapt
        def invalid_norm(self, x):
            result = original_adapt(self, x)
            result.upd_norm = float("nan")
            return result
        monkeypatch.setattr(EpisodicTentAdapter, "adapt", invalid_norm)
    module.main()
    assert not sentinel.exists()
    if name == "12_profile_real_run":
        result = json.loads((tmp_path / "results/runtime_profile.json").read_text())
        if fault != "valid":
            assert_unavailable(result["gate_state"])
            assert result["gate"]["mean_ms"] is None
        else:
            assert result["gate_state"]["authority_sha256"] == pair["expected"]
    else:
        path = {"06_replay_heldout": "heldout", "07_replay_replication": "replication", "07_shadow_live": "shadow"}[name]
        records = read_jsonl(str(tmp_path / f"{path}.jsonl"))
        assert records
        for record in records:
            if fault != "valid":
                assert_unavailable(record)
                if fault in {"nan_evidence", "infinite_evidence", "missing_feature", "extraction_failure", "invalid_update_norm"}:
                    assert record["evidence"] == {}
                    assert record["eps"] is None
                    assert record["reason"] == "EDGE_EVIDENCE_INVALID"
                    if fault == "invalid_update_norm":
                        assert record["upd_norm"] is None
            else:
                assert record["availability"] == "available"
                assert record["authority_sha256"] == pair["expected"]
                assert record["frozen_model_sha256"] == expected
            assert record["schema_version"] == "kbound-edge-v2"
        if path != "shadow":
            result = json.loads((tmp_path / f"results/{path}_metrics.json").read_text())
            assert result["unavailable_windows"] == (0 if fault == "valid" else 1)
            if fault != "valid":
                assert result["eps"] is None


def test_failed_reload_clears_authority_and_actual_consumers(pair, setup_runtime, tmp_path):
    from kbound_edge.deployment import DeploymentGate
    _, _, _, f0, frames = setup_runtime
    gate = DeploymentGate()
    kwargs = {"metadata_path": str(pair["metadata_path"]), "authority": pair["authority_bytes"],
                  "expected_authority_sha256": pair["expected"], "active_identities": identities()}
    gate.reload(str(pair["path"]), **kwargs)
    assert gate.decide(np.zeros(14)).as_dict()["authority_sha256"] == pair["expected"]
    gate.reload(str(pair["path"]), **{**kwargs, "expected_authority_sha256": "0" * 64})
    assert gate.estimator is None
    adapter = EpisodicTentAdapter(f0)
    log = tmp_path / "runtime.jsonl"
    with WindowLogger(str(log), "frozen", "config") as logger:
        result = replay.replay_windows([frames], f0, adapter, gate, gate.eps, logger=logger)
    outcome = result["outcomes"][0]
    assert_unavailable(outcome.decision.as_dict())
    assert result["policy_decisions"]["kga_full"] == ["abstain"]
    assert result["policy_decisions"]["kga_no_radius"] == ["abstain"]
    assert result["policy_decisions"]["always_freeze"] == ["freeze"]
    assert_unavailable(read_jsonl(str(log))[0])
    line = LiveDashboard(quiet=True).render(outcome)
    assert "unavailable / retain frozen" in line
    assert "B^=" not in line
    assert annotate_frame(frames[0], outcome).ndim == 3
    ctrl = ShadowController(f0, adapter, gate, gate.eps, window_size=2, image_size=8)
    for frame in frames:
        ctrl.push_frame(frame)
    assert ctrl.official_outputs == [ctrl.outcomes[0].frozen_pred]
    assert ctrl.shadow_decisions == ["abstain"]
    profile = profile_runtime(f0, adapter, gate, gate.eps, [frames], image_size=8, warmup=0)
    assert_unavailable(profile["gate_state"])
    assert profile["gate"]["mean_ms"] is None


@pytest.mark.parametrize("benefit, expected", [(0.4, "adapt"), (-0.4, "freeze"), (0., "abstain")])
def test_sealed_pair_preserves_three_way_policy(pair, benefit, expected):
    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.deployment import DeploymentGate
    est = EdgeBenefitEstimator(max_iter=2).fit(np.zeros((30, 14)), np.full(30, benefit))
    pair["payload"]["model"] = est._model
    pair["payload"]["params"] = est.params
    seal(pair)
    gate = DeploymentGate()
    gate.reload(str(pair["path"]), metadata_path=str(pair["metadata_path"]),
                authority=pair["authority_bytes"], expected_authority_sha256=pair["expected"],
                active_identities=identities())
    d = gate.decide(np.zeros(14)).as_dict()
    assert d["decision"] == expected
    assert d["availability"] == "available"
    assert d["eps"] == .125


def test_trusted_checkpoint_immutable_restrictive_load(setup_runtime, monkeypatch):
    cfg, _, expected, _, _ = setup_runtime
    module = caller(CALLERS[0], monkeypatch)
    original = torch.load
    def inspect(data, **kwargs):
        assert isinstance(data, io.BytesIO)
        assert kwargs["weights_only"] is True
        Path(cfg["paths"]["model"]).write_bytes(b"replacement")
        return original(data, **kwargs)
    monkeypatch.setattr(torch, "load", inspect)
    f0, version = module.C.load_trusted_f0(cfg, expected)
    assert version == expected
    assert model.predict_proba(f0, torch.zeros(2, 3, 8, 8)).shape == (2, 4)


@pytest.mark.parametrize("fault", ["missing", "digest", "symlink", "malicious", "unsupported"])
def test_frozen_failures_never_serve(fault, setup_runtime, tmp_path, monkeypatch):
    cfg, _, expected, _, _ = setup_runtime
    module = caller(CALLERS[0], monkeypatch)
    path = Path(cfg["paths"]["model"])
    sentinel = tmp_path / "frozen-executed"
    if fault == "missing":
        path.unlink()
    elif fault == "digest":
        expected = "0" * 64
    elif fault == "symlink":
        link = tmp_path / "link.pt"
        link.symlink_to(path)
        cfg["paths"]["model"] = str(link)
    elif fault == "malicious":
        class Gadget:
            def __reduce__(self):
                return os.system, (f"touch {sentinel}",)
        torch.save(Gadget(), path)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "unsupported":
        def old_torch(*args, **kwargs):
            assert kwargs["weights_only"] is True
            raise TypeError("weights_only unsupported")
        monkeypatch.setattr(torch, "load", old_torch)
    with pytest.raises(ValueError, match="FROZEN_AUTHORITY_INVALID"):
        module.C.load_trusted_f0(cfg, expected)
    assert not sentinel.exists()


def test_report_renders_unavailable_without_numeric_gate(report_inputs):
    from kbound_edge.reporting import compile_latex_macros
    report_inputs["records"][0].update(decision="abstain", availability="unavailable", certificate_status="not_issued")
    report_inputs["runtime_profile"]["gate"] = {"mean_ms": None, "p95_ms": None}
    macros = compile_latex_macros(**report_inputs)
    assert macros["CameraSFourMeanGate"] == "unavailable / retain frozen"
    assert macros["CameraGateUnavailableWindows"] == "1"
    assert macros["CameraRThreeInterpretationMildLight"] == "unavailable / retain frozen; certificate not issued"
    assert macros["CameraSThreeHOneKga"] == "unavailable / retain frozen"


def test_logger_rejects_nonfinite_before_write(tmp_path):
    with WindowLogger(str(tmp_path / "strict.jsonl"), "frozen", "config") as logger:
        with pytest.raises(ValueError):
            logger.log(0, {"decision": "abstain"}, {}, float("nan"))
        assert logger.n_written == 0


@pytest.mark.parametrize("bad", [
    None, np.full(14, np.nan), np.full(14, np.inf), np.full(14, -np.inf),
    np.array([None] + [0.] * 13, dtype=object), np.ones(14, dtype=bool),
    np.full(14, "0"), np.zeros(13), np.zeros((1, 14)),
    np.ma.array(np.zeros(14), mask=[True] + [False] * 13), np.zeros(14, dtype=complex),
])
def test_invalid_evidence_clears_valid_certificate_before_prediction(pair, bad, monkeypatch):
    from kbound_edge.benefit_estimator import EdgeBenefitEstimator
    from kbound_edge.deployment import DeploymentGate
    est = EdgeBenefitEstimator(max_iter=2).fit(np.zeros((30, 14)), np.full(30, -.4))
    pair["payload"]["model"] = est._model
    pair["payload"]["params"] = est.params
    seal(pair)
    gate = DeploymentGate()
    gate.reload(str(pair["path"]), metadata_path=str(pair["metadata_path"]),
                authority=pair["authority_bytes"], expected_authority_sha256=pair["expected"],
                active_identities=identities())
    assert gate.decide(np.zeros(14)).decision == "freeze"
    original_predict = gate.estimator.predict_one
    calls = []
    def observe(z):
        calls.append(z)
        return original_predict(z)
    monkeypatch.setattr(gate.estimator, "predict_one", observe)
    invalid = gate.decide(bad).as_dict()
    assert_unavailable(invalid)
    assert invalid["reason"] == "EDGE_EVIDENCE_INVALID"
    assert calls == []
    assert gate.estimator is None
    assert gate.eps is None
    assert_unavailable(gate.decide(np.zeros(14)).as_dict())


@pytest.mark.parametrize("consumer", ["replay", "shadow", "profile"])
def test_runtime_evidence_failure_requires_explicit_reload(pair, setup_runtime, monkeypatch, consumer):
    from kbound_edge import profiling
    from kbound_edge.deployment import DeploymentGate
    _, _, _, f0, frames = setup_runtime
    gate = DeploymentGate()
    authority = {"metadata_path": str(pair["metadata_path"]), "authority": pair["authority_bytes"],
                 "expected_authority_sha256": pair["expected"], "active_identities": identities()}
    gate.reload(str(pair["path"]), **authority)
    adapter = EpisodicTentAdapter(f0)
    original_extract = replay.edge_evidence_vector
    attempts = 0
    def intermittent(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise ValueError("synthetic insufficient evidence")
        return original_extract(*args)
    monkeypatch.setattr(replay, "edge_evidence_vector", intermittent)
    monkeypatch.setattr(profiling, "edge_evidence_vector", intermittent)
    if consumer == "replay":
        result = replay.replay_windows([frames] * 3, f0, adapter, gate, gate.eps)
        records = result["gate_records"]
        for outcome in result["outcomes"]:
            np.testing.assert_array_equal(outcome.p0, result["outcomes"][0].p0)
        assert result["policy_decisions"]["kga_full"][1:] == ["abstain", "abstain"]
        assert result["policy_decisions"]["kga_no_radius"][1:] == ["abstain", "abstain"]
    elif consumer == "shadow":
        ctrl = ShadowController(f0, adapter, gate, gate.eps, window_size=2, image_size=8)
        for frame in list(frames) * 3:
            ctrl.push_frame(frame)
        records = [outcome.decision.as_dict() for outcome in ctrl.outcomes]
        assert ctrl.official_outputs == [outcome.frozen_pred for outcome in ctrl.outcomes]
    else:
        result = profiling.profile_runtime(f0, adapter, gate, gate.eps, [frames] * 3, image_size=8, warmup=0)
        records = result["gate_records"]
        assert result["gate"]["mean_ms"] is None
    assert records[0]["availability"] == "available"
    assert records[0]["authority_sha256"] == pair["expected"]
    for record in records[1:]:
        assert_unavailable(record)
    gate.reload(str(pair["path"]), **authority)
    assert gate.decide(np.zeros(14)).availability == "available"


@pytest.mark.parametrize("consumer", ["replay", "profile"])
def test_frozen_prediction_failure_is_not_suppressed(setup_runtime, monkeypatch, consumer):
    from kbound_edge import profiling
    from kbound_edge.deployment import DeploymentGate
    _, _, _, f0, frames = setup_runtime
    gate = DeploymentGate()
    adapter = EpisodicTentAdapter(f0)
    def failed_prediction(*args):
        raise RuntimeError("synthetic frozen inference failure")
    monkeypatch.setattr(replay, "predict_proba", failed_prediction)
    monkeypatch.setattr(profiling, "predict_proba", failed_prediction)
    with pytest.raises(RuntimeError, match="synthetic frozen inference failure"):
        if consumer == "replay":
            replay.replay_windows([frames], f0, adapter, gate, gate.eps)
        else:
            profiling.profile_runtime(f0, adapter, gate, gate.eps, [frames], image_size=8, warmup=0)
