"""Outcome-blind orchestration tests, no dataset or full model IO."""

import importlib
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts"))


def module():
    return importlib.import_module("domainnet_feasibility_runner")


class Session:
    def __init__(self, fault=None):
        self.completed_steps = 0
        self.model = SimpleNamespace(queue_ptr=0)
        self.banks = None
        self.state = 0
        self.fault = fault
        self.events = []

    def initialize_bank(self, batches):
        self.events.append("bank")
        self.banks = list(batches)
        assert self.banks == ["adaptation"]

    def train_epoch(self, batches, epoch):
        self.events.append(epoch)
        assert batches == ("adaptation", epoch)
        self.completed_steps += 512
        self.model.queue_ptr = self.completed_steps * 4 % 16384
        self.state += 1
        if self.fault == "step":
            self.completed_steps -= 1
        if self.fault == "queue":
            self.model.queue_ptr = 0
        if self.fault == "warning":
            warnings.warn("nonfinite native diagnostic", RuntimeWarning, stacklevel=2)
        summary = {"count": 512, "sum": 1024.0, "mean": 2.0, "last": 3.0}
        if self.fault == "missing_loss":
            return None
        if self.fault == "loss_count":
            summary["count"] = 511
        if self.fault == "loss_nonfinite":
            summary["last"] = float("nan")
        if self.fault == "loss_mean":
            summary["mean"] = 3.0
        return summary


def run(fake, output):
    m = module()

    def infer(refine):
        assert fake.completed_steps == 7680
        fake.events.append("refined" if refine else "direct")
        if fake.fault == "mutation":
            fake.state += 1
        return [2 if refine else 1] * 4096

    return m._execute_schedule(
        fake, ["adaptation"], lambda epoch: ("adaptation", epoch), infer, lambda: str((fake.state, fake.banks)), output
    )


def test_exact_epochs_queue_and_no_evaluation_before_completion():
    fake = Session()
    progress = []
    direct, refined = run(fake, progress.append)
    assert fake.events == ["bank"] + list(range(15)) + ["direct", "refined"]
    assert len(progress) == 15 and progress[-1]["completed_steps"] == 7680
    assert progress[-1]["queue_ptr"] == 14336
    assert progress[7]["initial_queue_fully_replaced"] is True
    assert progress[-1]["loss"] == {"count": 512, "sum": 1024.0, "mean": 2.0, "last": 3.0}
    assert direct == [1] * 4096 and refined == [2] * 4096


@pytest.mark.parametrize("fault", ["step", "queue", "mutation", "warning"])
def test_bad_update_or_inference_stops_without_scoring(fault):
    with pytest.raises((ValueError, RuntimeWarning)):
        with module().warning_receipt([]):
            run(Session(fault), lambda item: None)


@pytest.mark.parametrize("fault", ["missing_loss", "loss_count", "loss_nonfinite", "loss_mean"])
def test_incomplete_or_invalid_loss_summary_stops_before_evaluation(fault):
    fake = Session(fault)
    with pytest.raises(ValueError, match="loss"):
        run(fake, lambda item: None)
    assert "direct" not in fake.events


def test_warning_receipt_keeps_warning_on_failure(capsys):
    m = module()
    recorded = []
    with pytest.raises(RuntimeWarning):
        with m.warning_receipt(recorded):
            warnings.warn("numerical failure", RuntimeWarning, stacklevel=2)
    assert recorded[0]["message"] == "numerical failure"
    assert "numerical failure" in capsys.readouterr().err


def test_known_native_deprecation_is_visible_and_explicit(capsys):
    m = module()
    recorded = []
    with m.warning_receipt(recorded):
        warnings.warn(
            "`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.",
            FutureWarning,
            stacklevel=2,
        )
    assert recorded[0]["disposition"] == "retained_known_deprecation"
    assert "weight_norm" in capsys.readouterr().err


@pytest.mark.parametrize("unit", ["single", "whole_episode"])
def test_all_schedule_operations_are_invariant_to_scored_outcomes(unit, monkeypatch):
    m = module()
    calls = []

    def forbidden(*a, **kw):
        raise AssertionError("outcome authority opened before scoring")

    monkeypatch.setattr(m, "read_truth", forbidden, raising=False)
    truth = [0] * 4096
    before = run(Session(), calls.append)
    truth[0 if unit == "single" else slice(None)] = 1 if unit == "single" else [1] * 4096
    after = run(Session(), calls.append)
    assert before == after and calls[:15] == calls[15:]


def test_runtime_identity_changes_fail_before_payload(monkeypatch):
    m = module()
    with pytest.raises(ValueError):
        m.validate_runtime({"python": "not-the-current-interpreter"})


def test_review_gate_is_bound_to_code_and_manifest():
    m = module()
    with pytest.raises(ValueError):
        m.validate_review({}, "a" * 64, {"code": {}})


@pytest.mark.parametrize("kind", ["wall", "rss", "disk"])
def test_bounded_worker_reaps_only_its_process_group(tmp_path, kind):
    m = module()
    code = {
        "wall": "import time; time.sleep(30)",
        "rss": "import time; x=bytearray(50*1024**2); time.sleep(30)",
        "disk": "from pathlib import Path; import time; Path('large').write_bytes(b'x'*100000); time.sleep(30)",
    }[kind]
    result = m.run_bounded(
        [sys.executable, "-c", code],
        tmp_path,
        tmp_path,
        {
            "wall_seconds": 0.2 if kind == "wall" else 5,
            "rss_bytes": 10**6 if kind == "rss" else 10**9,
            "disk_bytes": 500 if kind == "disk" else 10**8,
        },
    )
    assert result["status"] == "STOP_" + kind.upper() and result["child_reaped"]


def test_missing_review_stops_before_reserving_real_output(tmp_path):
    m = module()
    with pytest.raises((ValueError, OSError)):
        m.main(
            [
                "run",
                "--manifest",
                str(tmp_path / "missing"),
                "--sha256",
                "a" * 64,
                "--review",
                str(tmp_path / "review"),
                "--review-sha256",
                "b" * 64,
            ]
        )
    assert list(tmp_path.iterdir()) == []


def test_scoring_rejects_incomplete_collection_before_truth(tmp_path):
    m = module()
    calls = []

    def truth():
        calls.append("truth")
        raise AssertionError("truth accessed")

    with pytest.raises(ValueError):
        m.score_complete({}, {}, {"status": "INCOMPLETE"}, tmp_path, truth)
    assert calls == []


def test_worker_inputs_require_external_inventory_binding(tmp_path):
    m = module()
    # Self-computing the hash of the current binding file is not authentication.
    with pytest.raises(ValueError, match="stage input"):
        m.validate_stage_inputs("collect", {}, "manifest")


def test_final_inference_serialization_roundtrip_and_tamper(tmp_path, monkeypatch):
    import domainnet_reference_source as source
    import torch
    from torch import nn

    m = module()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 126)

        def forward(self, x, return_feats=False):
            feats = x.mean((2, 3))
            logits = self.fc(feats)
            return (feats, logits) if return_feats else logits

    torch.manual_seed(42)
    net = Tiny().eval()
    session = SimpleNamespace(
        model=SimpleNamespace(src_model=net),
        banks={"features": torch.zeros(10, 3)},
        ref=SimpleNamespace(refine_predictions=lambda f, p, b, a: (p.argmax(1), None, None)),
        args=None,
    )
    batch = torch.zeros(4, 3, 8, 8)
    monkeypatch.setattr(m, "inference_batches", lambda *a: iter([batch]))
    monkeypatch.setattr(source, "build_reference_classifier", Tiny)
    path = tmp_path / "inference.pt"
    identity = m.export_inference(session, path)
    direct = net(batch).argmax(1).tolist()
    m.verify_reloaded(session, path, identity, None, None, direct, direct)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        m.verify_reloaded(session, path, identity, None, None, direct, direct)


def test_staging_worker_stops_and_records_pillow_warning(tmp_path, monkeypatch):
    import io
    import json
    import resource

    import torch
    from domainnet_feasibility_images import rgb_from_bytes
    from PIL import Image

    m = module()
    (tmp_path / "stage").mkdir()
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *a: None)
    im = Image.new("P", (2, 2))
    im.putpalette(list(range(256)) * 3)
    im.info["transparency"] = bytes([0, 128, 255])
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")

    def stage(*a):
        with rgb_from_bytes(buffer.getvalue(), 1024):
            pass

    monkeypatch.setattr(m, "stage_selected", stage)
    with pytest.raises(UserWarning):
        m._worker({}, tmp_path, "stage", None, {"manifest_digest": m.digest({})})
    warnings_saved = json.loads((tmp_path / "stage/warnings.json").read_text())
    assert warnings_saved[0]["disposition"] == "STOP_WARNING"


def test_watchdog_rejects_parent_symlink_without_outside_write(tmp_path):
    m = module()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "stage"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        m.run_bounded(
            [sys.executable, "-c", "pass"], link, tmp_path, {"wall_seconds": 2, "rss_bytes": 10**9, "disk_bytes": 10**9}
        )
    assert list(outside.iterdir()) == []
    with pytest.raises(ValueError):
        m.disk_bytes(link)


def saved_panel(root, version=2):
    """Real JSON/hash files and scorer; synthetic predictions, no native data."""
    from domainnet_feasibility_contract import RELEASE_SHA256, SCOPE, epoch_orders

    m = module()
    manifest = {"code": {}, "runtime": {}, "episodes": []}
    inventory = {"synthetic": True}
    for e in range(3):
        manifest["episodes"].append(
            {
                "stream_seed": 2020 + e,
                "adaptation_ids": [f"painting/a/a{e}_{i}.jpg" for i in range(2048)],
                "evaluation_ids": [f"painting/a/v{e}_{i}.jpg" for i in range(4096)],
                "epoch_order_sha256": [m.digest(x) for x in epoch_orders(2020 + e)],
            }
        )
    packets, truth = [], {}
    for e in range(3):
        folder = root / f"episode-{e}"
        folder.mkdir()
        epoch_hashes = []
        for j in range(15):
            record = {
                "episode": e,
                "epoch": j,
                "stream_seed": 2020 + e,
                "manifest_digest": m.digest(manifest),
                "inventory_digest": m.digest(inventory),
                "completed_steps": (j + 1) * 512,
                "queue_ptr": ((j + 1) * 2048) % 16384,
                "initial_queue_fully_replaced": j >= 7,
                "order_sha256": manifest["episodes"][e]["epoch_order_sha256"][j],
                "candidate_state_sha256": "a" * 64,
                "epoch_seconds": 1.0,
                "elapsed_seconds": j + 1.0,
                "loss": {"count": 512, "sum": 1024.0, "mean": 2.0, "last": 3.0},
            }
            path = folder / f"epoch-{j:02}.json"
            m.write_json(path, record)
            epoch_hashes.append(m.file_identity(path))
        (folder / "inference.pt").write_bytes(b"synthetic inference artifact")
        (folder / "worker.log").write_bytes(b"Native Loss: 2.0\n")
        m.write_json(folder / "warnings.json", [])
        m.write_json(
            root / f"episode-{e}-process.json",
            {
                "status": "PASS",
                "returncode": 0,
                "child_reaped": True,
                "wall_seconds": 15.0,
                "sampled_peak_rss_bytes": 1024,
                "output_bytes": 2048,
            },
        )
        p = {
            "schema": f"painting-feasibility-predictions-v{version}",
            "status": "COLLECTED_NOT_SCORED",
            "episode": e,
            "stream_seed": 2020 + e,
            "manifest_digest": m.digest(manifest),
            "inventory_digest": m.digest(inventory),
            "code": {},
            "runtime": {},
            "scope": SCOPE,
            "sample_ids": manifest["episodes"][e]["evaluation_ids"],
            "adaptation_ids": manifest["episodes"][e]["adaptation_ids"],
            "frozen": [0] * 4096,
            "candidate_direct": [1] * 4096,
            "candidate_refined": [2] * 4096,
            "completed_steps": 7680,
            "queue_ptr": 14336,
            "candidate_state_sha256": "b" * 64,
            "source_receipt": {"release_sha256": RELEASE_SHA256},
            "inference_artifact": m.file_identity(folder / "inference.pt"),
            "reload_predictions_identical": True,
            "warnings": [],
            "epoch_receipts": epoch_hashes,
        }
        m.write_json(folder / "predictions.json", p)
        packets.append(m.file_identity(folder / "predictions.json"))
        truth.update(dict.fromkeys(p["sample_ids"], 1))
    return manifest, inventory, packets, truth


def test_legacy_completion_cannot_open_truth_without_epoch_evidence(tmp_path):
    m = module()
    manifest, inventory, packets, truth = saved_panel(tmp_path, version=1)
    completion = {
        "status": "ALL_THREE_COLLECTED_NOT_SCORED",
        "manifest_digest": m.digest(manifest),
        "inventory_digest": m.digest(inventory),
        "packets": packets,
    }
    (tmp_path / "episode-0/epoch-14.json").unlink()
    opened = []
    with pytest.raises(ValueError):
        m.score_complete(manifest, inventory, completion, tmp_path, lambda: opened.append(True) or truth)
    assert not opened


def bound_panel(root):
    m = module()
    manifest, inventory, _, truth = saved_panel(root)
    completion = {
        "schema": "painting-feasibility-completion-v2",
        "status": "ALL_THREE_COLLECTED_NOT_SCORED",
        "manifest_digest": m.digest(manifest),
        "inventory_digest": m.digest(inventory),
        "episodes": [m.bind_episode_completion(manifest, inventory, root, e) for e in range(3)],
    }
    return manifest, inventory, completion, truth


def test_bound_complete_evidence_scores_all_three(tmp_path):
    m = module()
    manifest, inventory, completion, truth = bound_panel(tmp_path)
    result = m.score_complete(manifest, inventory, completion, tmp_path, lambda: truth)
    assert result["direct_mean_benefit"] == 1.0
    assert result["refined_mean_benefit"] == 0.0
    assert len(completion["episodes"][0]["files"]) == 20


@pytest.mark.parametrize(
    "fault",
    [
        "epoch",
        "missing_epoch",
        "log",
        "warnings",
        "process",
        "predictions",
        "inference",
        "duplicate",
        "reordered",
        "old_schema",
    ],
)
def test_tampered_closed_evidence_stops_before_any_truth(tmp_path, fault):
    m = module()
    manifest, inventory, completion, truth = bound_panel(tmp_path)
    paths = {
        "epoch": "episode-0/epoch-14.json",
        "log": "episode-1/worker.log",
        "warnings": "episode-2/warnings.json",
        "process": "episode-1-process.json",
        "predictions": "episode-0/predictions.json",
        "inference": "episode-2/inference.pt",
    }
    if fault in paths:
        path = tmp_path / paths[fault]
        path.write_bytes(path.read_bytes() + b"tamper")
    elif fault == "missing_epoch":
        (tmp_path / "episode-1/epoch-03.json").unlink()
    elif fault == "duplicate":
        completion["episodes"][1] = completion["episodes"][0]
    elif fault == "reordered":
        completion["episodes"][0]["files"].reverse()
    elif fault == "old_schema":
        completion["schema"] = "painting-feasibility-completion-v1"
    opened = []
    with pytest.raises((ValueError, OSError)):
        m.score_complete(manifest, inventory, completion, tmp_path, lambda: opened.append(True) or truth)
    assert not opened


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", 13),
        ("episode", 2),
        ("completed_steps", 7168),
        ("queue_ptr", 0),
        ("order_sha256", "0" * 64),
        ("elapsed_seconds", -1),
        ("loss", {"count": 511, "sum": 1024.0, "mean": 2.0, "last": 3.0}),
        ("loss", {"count": 512, "sum": 1024.0, "mean": 3.0, "last": 3.0}),
    ],
)
def test_supervisor_rejects_malformed_epoch_even_with_matching_file_hash(tmp_path, field, value):
    import json

    m = module()
    manifest, inventory, _, _ = saved_panel(tmp_path)
    path = tmp_path / "episode-0/epoch-14.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    packet_path = tmp_path / "episode-0/predictions.json"
    packet = json.loads(packet_path.read_text())
    packet["epoch_receipts"][14] = m.file_identity(path)
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        m.bind_episode_completion(manifest, inventory, tmp_path, 0)
