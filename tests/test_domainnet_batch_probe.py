"""Synthetic probe boundary tests; no released tensors or dataset payloads."""

import copy
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"
sys.path.insert(0, str(SCRIPTS))


def probe():
    assert (SCRIPTS / "domainnet_batch_probe.py").is_file(), "bounded batch probe is missing"
    return importlib.import_module("domainnet_batch_probe")


def test_settings_reject_tuning_or_real_input_fields():
    m = probe()
    for field, value in [
        ("batch_size", 4),
        ("device", "mps"),
        ("updates", 2),
        ("data_root", "/forbidden"),
        ("bank_entries", 128),
    ]:
        changed = {**m.SETTINGS, field: value}
        with pytest.raises(ValueError):
            m.validate_settings(changed, m.BUDGET)
    with pytest.raises(ValueError):
        m.validate_settings(m.SETTINGS, {**m.BUDGET, "rss_bytes": 8 * 1024**3})


@pytest.mark.parametrize("fault", ["acceptance", "manifest", "code", "reviewer"])
def test_review_binding_is_required(fault):
    m = probe()
    manifest = {"code": {"x": {"sha256": "b" * 64, "bytes": 1}}}
    review = {
        "schema": "domainnet-batch-probe-review-v1",
        "accepted_for_launch": True,
        "manifest_sha256": "a" * 64,
        "code": manifest["code"],
        "reviewer": "independent",
    }
    if fault == "acceptance":
        review["accepted_for_launch"] = False
    if fault == "manifest":
        review["manifest_sha256"] = "c" * 64
    if fault == "code":
        review["code"] = {}
    if fault == "reviewer":
        review["reviewer"] = ""
    with pytest.raises(ValueError):
        m.validate_review(review, "a" * 64, manifest)


def test_worker_rejects_unrelated_parent_before_source_access(tmp_path):
    m = probe()
    m.write_json(
        tmp_path / "authorization.json",
        {"supervisor_pid": os.getpid() + 100000, "manifest_sha256": "a", "review_sha256": "b", "code": {}},
    )
    with pytest.raises(ValueError, match="supervisor"):
        m.check_supervisor({"output_root": str(tmp_path), "code": {}}, "a", "b")


def test_warnings_are_persisted_even_on_failure(tmp_path):
    import warnings

    m = probe()
    with pytest.raises(RuntimeError):
        with m.retained_warnings(tmp_path):
            warnings.warn("retained probe warning", UserWarning, stacklevel=2)
            raise RuntimeError("failure after warning")
    notice = json.loads((tmp_path / "warning-000.json").read_text())
    assert notice["message"] == "retained probe warning" and notice["category"] == "UserWarning"


def test_synthetic_update_uses_native_session_without_outcomes(tmp_path, monkeypatch):
    # A real tiny session tests the wrapper's update, not full-model costing.
    import domainnet_reference_adapter as adapter
    import torch

    from tests.test_domainnet_reference_adapter import TinyClassifier, tiny_args

    m = probe()
    reference = os.environ.get("KBOUND_ADACONTRAST_REFERENCE")
    assert reference, "supply authenticated source for native test"
    args = tiny_args()
    args.learn.epochs = args.learn.full_progress = 1
    session = adapter.ReferenceSession(TinyClassifier, args, reference_root=reference)
    before = copy.deepcopy(session.model.src_model.state_dict())

    def trap(*a, **kw):
        raise AssertionError("dataset/source payload opened during update")

    import zipfile

    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", trap)
    monkeypatch.setattr(zipfile.ZipFile, "open", trap)
    monkeypatch.setattr(adapter, "read_verified_checkpoint", trap)
    gen = torch.Generator().manual_seed(9817)
    views = [torch.randn(4, 8, generator=gen) for _ in range(3)]
    session.banks = {
        "features": torch.randn(16, 5, generator=gen),
        "probs": torch.softmax(torch.randn(16, 3, generator=gen), 1),
        "ptr": 0,
    }
    result = m.execute_synthetic(session, views)
    assert result["updates"] == 1 and result["queue_ptr"] == 4
    assert result["loss"]["count"] == 1
    assert result["optimizer_state_entries"] > 0
    assert set(result["bn_counter_deltas"].values()) == {2}
    assert not torch.equal(before["fc.weight"], session.model.src_model.state_dict()["fc.weight"])
    assert "accuracy" not in result and "predictions" not in result


def test_partial_update_cannot_be_reported_as_complete():
    m = probe()
    record = {
        "updates": 0,
        "queue_ptr": 0,
        "loss": {"count": 1, "sum": 1.0, "mean": 1.0, "last": 1.0},
        "optimizer_state_entries": 4,
        "bn_counter_deltas": {"student": 2, "teacher": 2},
    }
    with pytest.raises(ValueError):
        m.validate_measurement(record, 128)
    record.update(updates=1, queue_ptr=128)
    record["bn_counter_deltas"]["teacher"] = 0
    with pytest.raises(ValueError):
        m.validate_measurement(record, 128)


@pytest.mark.parametrize("mode", ["wall", "rss"])
def test_supervisor_resource_stop_is_terminal_and_child_reaped(tmp_path, mode):
    m = probe()
    from domainnet_feasibility_runner import run_bounded

    budget = {
        "wall_seconds": 0.3 if mode == "wall" else 5,
        "rss_bytes": 10 * 1024**2 if mode == "rss" else 1024**3,
        "disk_bytes": 1024**2,
    }
    code = "import time; x=bytearray(50*1024**2); time.sleep(10)"
    receipt = run_bounded([sys.executable, "-c", code], tmp_path, tmp_path, budget)
    assert receipt["status"] == "STOP_" + mode.upper() and receipt["child_reaped"]
    m.close_attempt(tmp_path, receipt)
    assert (tmp_path / "failure.json").is_file() and not (tmp_path / "complete.json").exists()


def test_symlink_log_is_rejected_without_touching_target(tmp_path):
    probe()
    from domainnet_feasibility_runner import run_bounded

    target = tmp_path / "original.txt"
    target.write_text("keep")
    (tmp_path / "worker.log").symlink_to(target)
    with pytest.raises(OSError):
        run_bounded(
            [sys.executable, "-c", "raise AssertionError('launched')"],
            tmp_path,
            tmp_path,
            {"wall_seconds": 1, "rss_bytes": 1024**3, "disk_bytes": 1024**2},
        )
    assert target.read_text() == "keep"


def test_completion_requires_measurement_and_closed_log(tmp_path):
    m = probe()
    with pytest.raises((ValueError, OSError)):
        m.close_attempt(tmp_path, {"status": "PASS", "child_reaped": True})
    assert not (tmp_path / "complete.json").exists()


def test_failed_attempt_binds_available_warning_and_phase_evidence(tmp_path):
    m = probe()
    m.write_json(tmp_path / "phase-00.json", {"phase": "construction"})
    m.write_json(tmp_path / "warning-000.json", {"message": "retained"})
    (tmp_path / "worker.log").write_text("stopped here\n")
    m.close_attempt(tmp_path, {"status": "STOP_RSS", "child_reaped": True})
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert set(failure["files"]) == {"phase-00.json", "warning-000.json", "worker.log", "process.json"}
    for name, bound in failure["files"].items():
        assert m.file_identity(tmp_path / name) == bound


def prepared(tmp_path, monkeypatch):
    import domainnet_feasibility_runner as runner

    m = probe()
    manifest = tmp_path / "manifest.json"
    # Unit-only authority: exercise the real identity checks against current
    # code and simulated runtime metadata, without authorizing or reading the
    # historical experiment. Do not mock the runtime comparator itself.
    runtime = {
        "python": "3.12.12",
        "executable": str(tmp_path / "unit-only-python"),
        "packages": {"torch": "2.8.0", "torchvision": "0.23.0"},
        "platform": "unit-test-only",
        "device": "cpu",
    }
    monkeypatch.setattr(runner, "runtime_identity", lambda: copy.deepcopy(runtime))
    base = tmp_path / "unit-only-original.json"
    m.write_json(
        base,
        {
            "code": {Path(m.__file__).name: m.file_identity(m.__file__)},
            "runtime": runtime,
            "checkpoint_path": str(tmp_path / "never-opened-checkpoint.pt"),
            "reference_root": str(tmp_path / "never-opened-reference"),
        },
    )
    monkeypatch.setattr(m, "V4_SHA", m.file_identity(base)["sha256"])
    bound = m.prepare(base, manifest, tmp_path / "run")
    payload = json.loads(manifest.read_text())
    review = tmp_path / "review.json"
    m.write_json(
        review,
        {
            "schema": "domainnet-batch-probe-review-v1",
            "accepted_for_launch": True,
            "manifest_sha256": bound["sha256"],
            "code": payload["code"],
            "reviewer": "unit-test-only-not-real-launch-acceptance",
        },
    )
    return m, manifest, bound["sha256"], review, m.file_identity(review)["sha256"]


def test_prepare_authorize_reads_metadata_only_and_refuses_overwrite(tmp_path, monkeypatch):
    import domainnet_reference_source as source

    def trap(*a, **kw):
        raise AssertionError("source tensor or dataset payload opened")

    monkeypatch.setattr(source, "load_clipart2020", trap)
    m, manifest, sha, review, review_sha = prepared(tmp_path, monkeypatch)
    result = m.authorize(manifest, sha, review, review_sha)
    assert result["settings"]["synthetic_only"] and not Path(result["output_root"]).exists()
    with pytest.raises(FileExistsError):
        m.prepare(result["original_manifest"], manifest, result["output_root"])


def test_unit_authorization_does_not_read_saved_experiment_manifest(tmp_path, monkeypatch):
    m = probe()
    original_read = m.read_json

    def only_fixture_metadata(path, *args, **kwargs):
        assert Path(path).is_relative_to(tmp_path), "real experiment metadata read by unit fixture"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(m, "read_json", only_fixture_metadata)
    m, manifest, sha, review, review_sha = prepared(tmp_path, monkeypatch)
    result = m.authorize(manifest, sha, review, review_sha)
    assert not Path(result["output_root"]).exists()


def test_unit_authorization_still_rejects_changed_observed_runtime(tmp_path, monkeypatch):
    import domainnet_feasibility_runner as runner

    m, manifest, sha, review, review_sha = prepared(tmp_path, monkeypatch)
    changed = {**runner.runtime_identity(), "python": "3.12.0"}
    monkeypatch.setattr(runner, "runtime_identity", lambda: changed)
    with pytest.raises(ValueError, match="approved experiment runtime mismatch"):
        m.authorize(manifest, sha, review, review_sha)


@pytest.mark.parametrize("package", ["torch", "torchvision"])
def test_matching_unit_metadata_cannot_bypass_native_runtime_pins(tmp_path, monkeypatch, package):
    import domainnet_feasibility_runner as runner

    m, manifest, sha, review, review_sha = prepared(tmp_path, monkeypatch)
    payload = m.read_json(manifest, sha)
    changed = copy.deepcopy(payload["runtime"])
    changed["packages"][package] = "0.0.0"
    monkeypatch.setattr(runner, "runtime_identity", lambda: copy.deepcopy(changed))
    # Even identical declared/observed metadata must satisfy the native pins.
    with pytest.raises(ValueError, match="approved experiment runtime mismatch"):
        runner.validate_runtime(changed)


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint_path", "/forbidden.pt"), ("reference_root", "/forbidden"), ("code", {}), ("runtime", {})],
)
def test_changed_identity_is_rejected_even_with_matching_test_review(tmp_path, monkeypatch, field, value):
    m, manifest, sha, review, review_sha = prepared(tmp_path, monkeypatch)
    payload = m.read_json(manifest, sha)
    payload[field] = value
    altered, altered_review = tmp_path / "altered.json", tmp_path / "altered-review.json"
    m.write_json(altered, payload)
    altered_sha = m.file_identity(altered)["sha256"]
    r = m.read_json(review, review_sha)
    r.update(manifest_sha256=altered_sha, code=payload["code"])
    m.write_json(altered_review, r)
    with pytest.raises(ValueError):
        m.authorize(altered, altered_sha, altered_review, m.file_identity(altered_review)["sha256"])
