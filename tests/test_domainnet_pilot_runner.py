"""Synthetic real-kernel runner tests. No actual archive/checkpoint/outcome access."""

import copy
import hashlib
import importlib
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.kbound.domainnet import pilot_candidate, source_data, train_source

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols/confirmatory_v2"


def api():
    assert importlib.util.find_spec("experiments.kbound.domainnet.pilot_runner") is not None
    return importlib.import_module("experiments.kbound.domainnet.pilot_runner")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Linear(3, 2)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, 0.2, -0.3], [-0.4, 0.8, 0.1]]))
            self.head.bias.copy_(torch.tensor([0.2, -0.1]))
        self.requires_grad_(False).eval()

    def forward(self, images):
        return self.head(self.bn(images).mean((2, 3)))


def transform(image):
    return torch.tensor(list(image.tobytes()), dtype=torch.float32).reshape(3, 3, 3).permute(2, 0, 1) / 128 - 1


def read(path):
    return source_data.strict_json(path)


def fixture(tmp_path, *, fit=2, radius=1, check=2, observer=None):
    module = api()
    rows = []
    packing = {}
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for partition, count in (("DEV_fit", fit), ("DEV_radius", radius), ("DEV_check", check)):
            cells = []
            for cell in range(count):
                windows = {}
                for window, n in (("U", 64), ("V", 64), ("E", 128)):
                    ids = []
                    for _ in range(n):
                        index = len(rows)
                        sample_id = hashlib.sha256(str(index).encode()).hexdigest()
                        path = f"painting/c{index % 2}/{index}.png"
                        image = Image.new("RGB", (3, 3), (index % 256, (index // 256) % 256, 117))
                        raw = io.BytesIO()
                        image.save(raw, format="PNG")
                        encoded = raw.getvalue()
                        zipped.writestr(path, encoded)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "path": path,
                                "label": index % 2,
                                "official_index": index,
                                "image_sha256": hashlib.sha256(encoded).hexdigest(),
                                "group_id": hashlib.sha256(
                                    struct.pack(">QQ", *image.size) + image.tobytes()
                                ).hexdigest(),
                                "width": 3,
                                "height": 3,
                                "partition": partition,
                                "cell": cell,
                                "window": window,
                            }
                        )
                        image.close()
                        ids.append(sample_id)
                    windows[window] = ids
                cells.append(windows)
            packing[partition] = {"cells": cells, "tail": {"U": [], "V": [], "E": []}}
    public = {
        "schema": "kbound-painting-pilot-public/2",
        "packing": packing,
        "rows": [{k: v for k, v in row.items() if k not in {"label", "path"}} for row in rows],
    }
    hooks = module.SyntheticHooks(
        archive_sha256=source_data.file_hash(archive),
        class_count=2,
        model_factory=Tiny,
        transform=transform,
        observer=observer,
    )
    broker = module.ArchiveBroker(archive, rows, public, class_count=2, transform=transform, observer=observer)
    lock = {
        "schema": "kbound-domainnet-pilot-execution-lock/2",
        "status": "DEVELOPMENT_ONLY",
        "execution_scope": "SYNTHETIC_TEST",
        "input_sha256": {"archive": hooks.archive_sha256},
        "class_count": 2,
        "feature_names": list(pilot_candidate.FEATURE_NAMES),
    }
    return module, public, broker, hooks, lock


def execute_fixture(tmp_path, **kwargs):
    module, public, broker, hooks, lock = fixture(tmp_path, **kwargs)
    out = tmp_path / "out"
    result = module.execute_pilot(Tiny(), public, broker, lock, out, synthetic_hooks=hooks)
    return module, public, broker, hooks, lock, out, result


def test_every_check_action_precedes_e_and_complete_panel_precedes_any_check_label(tmp_path):
    # Deleting/moving either seal boundary makes the sentinel fail during a real kernel execution.
    events = []

    def observe(event, cell_id, output):
        events.append((event, cell_id))
        if event == "images:E" and cell_id.startswith("DEV_check"):
            assert (output / "cells" / cell_id.replace(":", "-") / "action.json").is_file()
            assert (output / "calibration_seal.json").is_file()
        if event == "labels" and cell_id.startswith("DEV_check"):
            panel = read(output / "check_panel_seal.json")
            assert panel["expected_cell_ids"] == ["DEV_check:0", "DEV_check:1"]
            assert len(panel["cells"]) == 2
            for row in panel["cells"]:
                for key in ("action", "predictions"):
                    assert source_data.file_hash(output / row[key]["path"]) == row[key]["sha256"]

    module, public, broker, hooks, lock, out, result = execute_fixture(tmp_path, observer=observe)
    assert result["status"] == "INCONCLUSIVE_DATA_GEOMETRY"
    assert result["execution_scope"] == "SYNTHETIC_TEST"
    assert result["eligible_for_confirmatory"] is False
    assert result["population_certificate"] is False
    for cell in range(2):
        cid = f"DEV_check:{cell}"
        assert events.index(("images:V", cid)) < events.index(("images:E", cid)) < events.index(("labels", cid))
        action = read(out / "cells" / f"DEV_check-{cell}" / "action.json")
        assert action["epsilon"] is None and action["lower"] is None and action["upper"] is None
        assert action["action"] == "ABSTAIN" and action["served_model"] == "frozen"
    first_label = min(events.index(("labels", f"DEV_check:{i}")) for i in range(2))
    assert max(events.index(("images:E", f"DEV_check:{i}")) for i in range(2)) < first_label
    assert read(out / "calibration.json")["fit_cell_ids"] == ["DEV_fit:0", "DEV_fit:1"]
    assert read(out / "calibration.json")["radius_cell_ids"] == ["DEV_radius:0"]
    assert not any(path.stat().st_mode & 0o222 for path in out.rglob("*.json"))
    with pytest.raises(FileExistsError):
        module.execute_pilot(Tiny(), public, broker, lock, out, synthetic_hooks=hooks)


@pytest.mark.parametrize("fault", ["mutated", "missing", "extra", "reordered"])
def test_complete_panel_tampering_stops_before_label_join_and_retains_partial_files(tmp_path, fault):
    labels = []

    def observe(event, cell_id, output):
        if event == "check_panel_sealed":
            path = output / "cells/DEV_check-0/predictions.json"
            if fault == "missing":
                path.unlink()
            elif fault == "mutated":
                path.chmod(0o644)
                path.write_bytes(path.read_bytes() + b" ")
            elif fault == "extra":
                (output / "cells/DEV_check-99").mkdir()
            else:
                path = output / "check_panel_seal.json"
                value = read(path)
                value["expected_cell_ids"].reverse()
                path.chmod(0o644)
                path.write_text(json.dumps(value))
        if event == "labels" and cell_id.startswith("DEV_check"):
            labels.append(cell_id)

    module, public, broker, hooks, lock = fixture(tmp_path, observer=observe)
    out = tmp_path / "out"
    with pytest.raises((ValueError, FileNotFoundError)):
        module.execute_pilot(Tiny(), public, broker, lock, out, synthetic_hooks=hooks)
    assert not labels
    assert read(out / "error_stop.json")["status"] == "ERROR_STOP"
    assert (out / "execution_lock.json").exists()
    assert not (out / "summary.json").exists()


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra", "reordered", "wrong_window", "cross_group"])
def test_public_window_integrity_fails_before_any_model_use(tmp_path, fault):
    module, public, broker, hooks, lock = fixture(tmp_path)
    public = copy.deepcopy(public)
    window = public["packing"]["DEV_check"]["cells"][0]["U"]
    if fault == "duplicate":
        window[-1] = window[0]
    elif fault == "missing":
        window.pop()
    elif fault == "extra":
        window.append("f" * 64)
    elif fault == "reordered":
        window.reverse()
    elif fault == "wrong_window":
        window[0] = public["packing"]["DEV_check"]["cells"][0]["E"][0]
    else:
        public["rows"][64]["group_id"] = public["rows"][0]["group_id"]
    with pytest.raises(ValueError):
        module.execute_pilot(Tiny(), public, broker, lock, tmp_path / "out", synthetic_hooks=hooks)
    assert not (tmp_path / "out/summary.json").exists()


@pytest.mark.parametrize("fault", ["encoded", "rgb", "size", "label"])
def test_broker_revalidates_every_image_and_label_binding(tmp_path, fault):
    module, public, broker, hooks, lock = fixture(tmp_path)
    row = broker._rows[public["packing"]["DEV_fit"]["cells"][0]["U"][0]]
    if fault == "encoded":
        row["image_sha256"] = "0" * 64
    elif fault == "rgb":
        row["group_id"] = "0" * 64
    elif fault == "size":
        row["width"] = 9
    else:
        row["label"] = 2
    with pytest.raises(ValueError):
        module.execute_pilot(Tiny(), public, broker, lock, tmp_path / "out", synthetic_hooks=hooks)


def test_source_mutation_after_v_is_caught_by_candidate_receipt(tmp_path):
    source = Tiny()

    def observe(event, cell_id, output):
        if event == "images:E":
            source.bn.eps *= 2

    module, public, broker, hooks, lock = fixture(tmp_path, observer=observe)
    with pytest.raises(ValueError, match="mutat"):
        module.execute_pilot(source, public, broker, lock, tmp_path / "out", synthetic_hooks=hooks)
    assert read(tmp_path / "out/error_stop.json")["status"] == "ERROR_STOP"


def test_exact_ridge_and_radius_are_bound_before_check(tmp_path):
    _, _, _, _, _, out, _ = execute_fixture(tmp_path, fit=3, radius=10)
    calibration = read(out / "calibration.json")
    estimator = read(out / "estimator.json")
    assert calibration["ridge"] == 10.0 and calibration["alpha"] == 0.1
    assert calibration["epsilon"] == sorted(estimator["residuals"])[9]
    assert estimator["fit_unit"] == "DEV_fit" and estimator["calibration_unit"] == "DEV_radius"
    assert estimator["protocol_sha256"] == source_data.file_hash(out / "execution_lock.json")
    assert calibration["estimator_payload_sha256"] == estimator["payload_sha256"]
    features = [read(out / "cells" / f"DEV_fit-{i}" / "evidence.json")["features"] for i in range(3)]
    assert estimator["feature_center"] == pytest.approx([sum(x[j] for x in features) / 3 for j in range(4)])
    seal = read(out / "calibration_seal.json")
    assert seal["calibration_sha256"] == source_data.file_hash(out / "calibration.json")


@pytest.mark.parametrize(
    "classes,digest",
    [(126, "1" * 64), (True, "1" * 64), (2, "fa47e6d405503ea0286cabd767f176bd30e988b63ddd7b9db16cf030c9770015")],
)
def test_synthetic_hooks_cannot_select_production_identity(classes, digest):
    with pytest.raises(ValueError):
        api().SyntheticHooks(digest, classes, Tiny, transform)


def test_cli_has_no_synthetic_or_backend_override():
    module = api()
    with pytest.raises(SystemExit) as result:
        module.main(["--synthetic-hooks", "anything"])
    assert result.value.code == 2


def source_fixture(tmp_path):
    module = api()
    directory = tmp_path / "source-run"
    (directory / "seed-0").mkdir(parents=True)
    source_inventory = tmp_path / "source-inventory.json"
    identity = {"archive_sha256": "1" * 64, "class_count": 2}
    source_data.write_new_json(source_inventory, {"identity": identity})
    config = PROTOCOLS / "DOMAINNET_SOURCE_PILOT_v1.json"
    policy = PROTOCOLS / "DOMAINNET_SOURCE_POLICY_v2.json"
    if not policy.exists():
        policy = next(
            p for p in PROTOCOLS.glob("*.json") if source_data.file_hash(p) == source_data.SOURCE_POLICY_V2_SHA256
        )
    policy_document = source_data.load_source_policy(policy, source_data.SOURCE_POLICY_V2_SHA256)
    scope = {
        "execution_scope": "SYNTHETIC_TEST",
        "development_pilot": True,
        "purpose": "SOURCE_ONLY_DEVELOPMENT_PILOT",
        "eligible_for_confirmatory": False,
    }
    code = {
        "files": {
            f"experiments/kbound/domainnet/{name}": source_data.file_hash(ROOT / "experiments/kbound/domainnet" / name)
            for name in ("__init__.py", "source_data.py", "train_source.py")
        },
        "git_head": "SYNTHETIC",
        "git_status": "SYNTHETIC",
        "clean_checkout_claim": False,
    }
    context = dict(
        config=read(config),
        config_sha256=source_data.file_hash(config),
        source_identity=identity,
        inventory_sha256=source_data.file_hash(source_inventory),
        code_identity=code,
        runtime={"backend": "cpu"},
        effective_epochs=20,
        effective_batch_size=32,
        source_policy=policy_document,
        **scope,
    )
    state = Tiny().state_dict()
    record = dict(
        status="COMPLETED",
        seed=0,
        seed_streams=train_source.seed_streams(0),
        initial_tensor_sha256="a" * 64,
        final_tensor_sha256=train_source.tensor_hash(state),
        epochs=20,
        processed_count=336220,
        batches=10520,
        checkpoint="seed-0/final.pt",
        candidate="final_epoch_only",
        source_policy=policy_document,
        **scope,
    )
    checkpoint = dict(model=state, record=copy.deepcopy(record), **context)
    torch.save(checkpoint, directory / "seed-0/final.pt")
    record["checkpoint_sha256"] = source_data.file_hash(directory / "seed-0/final.pt")
    source_data.write_new_json(directory / "seed-0/receipt.json", record)
    source_data.write_new_json(directory / "receipt.json", dict(status="COMPLETED", runs=[record], **context))
    epochs = [
        {
            "epoch": i,
            "processed_count": 16811,
            "batches": 526,
            "source_monitor_count": 1892,
            "mean_loss": 3 - i / 20,
            "source_monitor_accuracy": 0.05,
            "next_lr": 0.01,
        }
        for i in range(1, 21)
    ]
    (directory / "seed-0/epochs.jsonl").write_text("\n".join(json.dumps(row) for row in epochs) + "\n")
    paths = {
        "config": config,
        "policy": policy,
        "source_inventory": source_inventory,
        "source_receipt": directory / "receipt.json",
        "source_seed_receipt": directory / "seed-0/receipt.json",
        "source_checkpoint": directory / "seed-0/final.pt",
        "source_epochs": directory / "seed-0/epochs.jsonl",
    }
    expected = {key: source_data.file_hash(path) for key, path in paths.items()}
    expected["source_checkpoint_tensor"] = record["final_tensor_sha256"]
    hooks = module.SyntheticHooks("2" * 64, 2, Tiny, transform)
    return module, directory, paths, expected, hooks


def test_source_final_checkpoint_and_readiness_boundary(tmp_path):
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    result = module.validate_source_run(
        directory, paths["source_inventory"], paths["config"], paths["policy"], expected, synthetic_hooks=hooks
    )
    assert result["readiness"]["ready"] is True
    assert result["readiness"]["final_monitor_accuracy"] == 0.05
    assert train_source.tensor_hash(result["state"]) == expected["source_checkpoint_tensor"]


@pytest.mark.parametrize(
    "fault",
    ["incomplete", "counts", "seed", "best", "flag", "config", "receipt", "bytes", "tensor", "nonfinite", "code"],
)
def test_invalid_source_receipts_checkpoints_or_recipe_fail_before_model_factory(tmp_path, fault):
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    if fault in {"incomplete", "counts"}:
        path = paths["source_epochs"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "incomplete":
            rows.pop()
        else:
            rows[-1]["processed_count"] -= 1
        path.write_text("\n".join(json.dumps(row) for row in rows))
        expected["source_epochs"] = source_data.file_hash(path)
    elif fault in {"seed", "best", "flag", "code", "receipt"}:
        path = paths["source_receipt"]
        value = read(path)
        if fault == "seed":
            value["runs"][0]["seed"] = 1
        elif fault == "best":
            value["runs"][0]["candidate"] = "best_epoch"
        elif fault == "flag":
            value["eligible_for_confirmatory"] = True
        elif fault == "code":
            value["code_identity"]["files"]["experiments/kbound/domainnet/train_source.py"] = "0" * 64
        else:
            value["effective_epochs"] = 19
        path.chmod(0o644)
        path.write_text(json.dumps(value))
        expected["source_receipt"] = source_data.file_hash(path)
    elif fault == "config":
        value = read(paths["config"])
        value["seeds"] = [0, 1]
        paths["config"] = tmp_path / "bad-config.json"
        paths["config"].write_text(json.dumps(value))
        expected["config"] = source_data.file_hash(paths["config"])
    elif fault == "bytes":
        with paths["source_checkpoint"].open("ab") as stream:
            stream.write(b"tamper")
    elif fault == "tensor":
        expected["source_checkpoint_tensor"] = "0" * 64
    else:
        checkpoint = torch.load(paths["source_checkpoint"], weights_only=True)
        checkpoint["model"]["bn.weight"][0] = float("nan")
        torch.save(checkpoint, paths["source_checkpoint"])
        expected["source_checkpoint"] = source_data.file_hash(paths["source_checkpoint"])
    with pytest.raises(ValueError):
        module.validate_source_run(
            directory, paths["source_inventory"], paths["config"], paths["policy"], expected, synthetic_hooks=hooks
        )


@pytest.mark.parametrize("accuracy,last_loss", [(0.049, 1.0), (0.05, 3.0)])
def test_source_optimization_failure_is_inconclusive_not_alternate_model(tmp_path, accuracy, last_loss):
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    rows = [json.loads(line) for line in paths["source_epochs"].read_text().splitlines()]
    rows[-1]["source_monitor_accuracy"] = accuracy
    for row in rows[-5:]:
        row["mean_loss"] = last_loss
    paths["source_epochs"].write_text("\n".join(json.dumps(row) for row in rows))
    expected["source_epochs"] = source_data.file_hash(paths["source_epochs"])
    result = module.validate_source_run(
        directory, paths["source_inventory"], paths["config"], paths["policy"], expected, synthetic_hooks=hooks
    )
    assert result["readiness"]["status"] == "INCONCLUSIVE_SOURCE_LEARNING"
    assert result["readiness"]["ready"] is False


def reseal_source(paths, expected, checkpoint, *, config=None):
    """Synthetic corruption fixture rebinds byte hashes to expose semantic checks."""
    record = checkpoint["record"]
    torch.save(checkpoint, paths["source_checkpoint"])
    record = {**record, "checkpoint_sha256": source_data.file_hash(paths["source_checkpoint"])}
    aggregate = read(paths["source_receipt"])
    aggregate["runs"] = [record]
    if config is not None:
        aggregate["config"] = config
        aggregate["config_sha256"] = expected["config"]
    for key, value in (("source_receipt", aggregate), ("source_seed_receipt", record)):
        paths[key].chmod(0o644)
        paths[key].write_text(json.dumps(value))
    for key in ("source_checkpoint", "source_receipt", "source_seed_receipt"):
        expected[key] = source_data.file_hash(paths[key])


def test_hash_rebound_nonfinite_optimizer_state_is_rejected(tmp_path):
    # A full checkpoint with finite model weights must not hide corrupt optimizer/scheduler state.
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    checkpoint = torch.load(paths["source_checkpoint"], weights_only=True)
    checkpoint["optimizer"] = {"state": {0: {"momentum_buffer": torch.tensor(float("nan"))}}}
    reseal_source(paths, expected, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        module.validate_source_run(
            directory, paths["source_inventory"], paths["config"], paths["policy"], expected, synthetic_hooks=hooks
        )


def test_boolean_seed_cannot_alias_integer_zero_in_exact_configuration(tmp_path):
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    config = read(paths["config"])
    config["seeds"] = [False]
    paths["config"] = tmp_path / "boolean-config.json"
    paths["config"].write_text(json.dumps(config))
    expected["config"] = source_data.file_hash(paths["config"])
    checkpoint = torch.load(paths["source_checkpoint"], weights_only=True)
    checkpoint["config"] = config
    checkpoint["config_sha256"] = expected["config"]
    reseal_source(paths, expected, checkpoint, config=config)
    with pytest.raises(ValueError, match="configuration"):
        module.validate_source_run(
            directory, paths["source_inventory"], paths["config"], paths["policy"], expected, synthetic_hooks=hooks
        )


def test_symlink_source_input_and_output_are_refused(tmp_path):
    module, directory, paths, expected, hooks = source_fixture(tmp_path)
    link = tmp_path / "linked-inventory.json"
    link.symlink_to(paths["source_inventory"])
    with pytest.raises(ValueError, match="symlink"):
        module.validate_source_run(directory, link, paths["config"], paths["policy"], expected, synthetic_hooks=hooks)
    fixture_dir = tmp_path / "images"
    fixture_dir.mkdir()
    module, public, broker, hooks, lock = fixture(fixture_dir)
    output = fixture_dir / "out"
    output.symlink_to(fixture_dir / "absent")
    with pytest.raises(FileExistsError):
        module.execute_pilot(Tiny(), public, broker, lock, output, synthetic_hooks=hooks)


def test_production_entry_failure_has_immutable_error_and_no_summary(tmp_path):
    module = api()
    missing = tmp_path / "missing"
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        module.run_pilot(
            missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, {}, out
        )
    assert read(out / "error_stop.json")["status"] == "ERROR_STOP"
    assert not (out / "summary.json").exists()
    with pytest.raises(FileExistsError):
        module.run_pilot(
            missing, missing, missing, missing, missing, missing, missing, missing, missing, missing, {}, out
        )


def test_cli_rejects_symlinked_expected_hash_authority_before_output_creation(tmp_path):
    authority = tmp_path / "hashes.json"
    authority.write_text("{}")
    linked = tmp_path / "linked-hashes.json"
    linked.symlink_to(authority)
    argv = []
    for name in (
        "prepared-dir",
        "v1-dir",
        "archive",
        "source-run-dir",
        "source-inventory",
        "config",
        "source-policy",
        "design",
        "amendment",
        "v1-stop",
    ):
        argv.extend(["--" + name, str(tmp_path / "absent")])
    argv.extend(["--expected-hashes", str(linked), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="symlink"):
        api().main(argv)
    assert not (tmp_path / "out").exists()


def test_archive_close_failure_cannot_leave_success_summary(tmp_path, monkeypatch):
    # Closing owned input resources belongs before publication of a successful result.
    module, public, broker, hooks, lock = fixture(tmp_path)
    original = broker.close

    def fail_close():
        original()
        raise OSError("synthetic archive close failure")

    monkeypatch.setattr(broker, "close", fail_close)
    with pytest.raises(OSError):
        module.execute_pilot(Tiny(), public, broker, lock, tmp_path / "out", synthetic_hooks=hooks)
    assert not (tmp_path / "out/summary.json").exists()
    assert read(tmp_path / "out/error_stop.json")["status"] == "ERROR_STOP"


def test_observed_runner_clock_includes_engine_input_validation(tmp_path, monkeypatch):
    module, public, broker, hooks, lock = fixture(tmp_path)
    original_clock = module.time.perf_counter
    original_verify = module._verify_inputs
    offset = [0.0]

    def verify(*args, **kwargs):
        if offset[0] == 0:
            offset[0] = 1000.0
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(module.time, "perf_counter", lambda: original_clock() + offset[0])
    monkeypatch.setattr(module, "_verify_inputs", verify)
    result = module.execute_pilot(Tiny(), public, broker, lock, tmp_path / "out", synthetic_hooks=hooks)
    assert result["observed_total_wallclock_seconds"] >= 1000.0
