"""Synthetic source-only integrity and training regressions."""

import hashlib
import importlib
import io
import json
import stat
import zipfile

import pytest
import torch
from PIL import Image
from torch import nn


def api():
    name = "experiments.kbound.domainnet.source_data"
    assert importlib.util.find_spec("experiments.kbound.domainnet"), "source preparation implementation absent"
    return importlib.import_module(name)


def fixture(tmp_path, change=None):
    entries = []
    lines = []
    for i in range(80):
        label = i % 2
        path = f"clipart/class{label}/image{i}.png"
        stream = io.BytesIO()
        Image.new("RGB", (12, 12), (i * 3, 30, 70)).save(stream, format="PNG")
        entries.append((path, stream.getvalue()))
        lines.append(f"{path} {label}")
    if change:
        change(entries, lines)
    archive = tmp_path / "clipart.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, value in entries:
            output.writestr(name, value)
    listing = tmp_path / "source.txt"
    listing.write_text("\n".join(lines) + "\n")
    raw = listing.read_bytes()
    expected = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "list_sha256": hashlib.sha256(raw).hexdigest(),
        "list_git_blob": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
        "class_count": 2,
    }
    return archive, listing, expected


def prepare(tmp_path, change=None):
    archive, listing, expected = fixture(tmp_path, change)
    module = api()
    identity = module.SourceIdentity(**expected)
    dest = tmp_path / "inventory"
    module.prepare_source(archive, listing, dest, identity=identity)
    return module, archive, dest, identity


def test_prepare_roundtrip_grouping_and_immutable_output(tmp_path):
    def duplicate(entries, lines):
        entries.append(("clipart/class0/copy.png", entries[0][1]))
        lines.append("clipart/class0/copy.png 0")

    module, archive, dest, identity = prepare(tmp_path, duplicate)
    doc = module.load_verified_source(dest, archive, identity=identity)
    assert len(doc["rows"]) == 81
    copies = [r for r in doc["rows"] if r["path"] in {"clipart/class0/copy.png", "clipart/class0/image0.png"}]
    assert len({r["group_id"] for r in copies}) == len({r["split"] for r in copies}) == 1
    assert {r["label"] for r in doc["rows"] if r["split"] == "source_fit"} == {0, 1}
    assert {r["split"] for r in doc["rows"]} == {"source_fit", "source_monitor"}
    before = (dest / "inventory.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.prepare_source(archive, dest / "source-list.txt", dest, identity=identity)
    assert (dest / "inventory.json").read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "blob",
        "list_hash",
        "missing",
        "escape",
        "duplicate",
        "label",
        "alias",
        "domain",
        "corrupt",
        "symlink",
        "special",
        "zip_escape",
        "class_alias",
        "label_conflict",
        "truncated",
    ],
)
def test_source_rejects_invalid_inputs_without_success(tmp_path, fault):
    def change(entries, lines):
        if fault == "missing":
            entries.pop()
        elif fault == "escape":
            lines[0] = "clipart/../escape.png 0"
        elif fault == "duplicate":
            lines.append(lines[0])
        elif fault == "label":
            lines[0] = lines[0].replace(" 0", " 00")
        elif fault == "alias":
            lines[0] = lines[0].replace("clipart/", "clipart//")
        elif fault == "domain":
            lines[0] = lines[0].replace("clipart", "real")
        elif fault == "corrupt":
            entries[0] = (entries[0][0], b"not an image")
        elif fault == "truncated":
            entries[0] = (entries[0][0], entries[0][1][:40])
        elif fault in {"symlink", "special"}:
            info = zipfile.ZipInfo("clipart/bad")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK if fault == "symlink" else stat.S_IFIFO) << 16
            entries.append((info, b"anything"))
        elif fault == "zip_escape":
            entries.append(("clipart/../bad", b"anything"))
        elif fault == "class_alias":
            lines[0] = lines[0].replace(" 0", " 1")
        elif fault == "label_conflict":
            entries[1] = (entries[1][0], entries[0][1])

    archive, listing, expected = fixture(tmp_path, change)
    if fault in {"hash", "blob", "list_hash"}:
        expected[{"hash": "archive_sha256", "blob": "list_git_blob", "list_hash": "list_sha256"}[fault]] = "0" * (
            40 if fault == "blob" else 64
        )
    module = api()
    with pytest.raises((ValueError, OSError, zipfile.BadZipFile)):
        module.prepare_source(archive, listing, tmp_path / "out", identity=module.SourceIdentity(**expected))
    assert not (tmp_path / "out" / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["role", "row", "hash"])
def test_inventory_tampering_rejected(tmp_path, mutation):
    module, archive, dest, identity = prepare(tmp_path)
    path = dest / "inventory.json"
    doc = json.loads(path.read_text())
    if mutation == "role":
        doc["role"] = "test"
    elif mutation == "row":
        doc["rows"][0]["split"] = "test"
    else:
        doc["rows"][0]["image_sha256"] = "0" * 64
    path.chmod(0o644)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        module.load_verified_source(dest, archive, identity=identity)


def train_api():
    api()
    return importlib.import_module("experiments.kbound.domainnet.train_source")


def tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 12 * 12, 2))


def tensor_transform(image):
    import numpy as np

    return torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255


def training_fixture(tmp_path):
    module, archive, dest, identity = prepare(tmp_path)
    runner = train_api()
    config = runner.approved_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    hooks = runner.TestHooks(
        identity=identity,
        model_factory=tiny_model,
        fit_transform=tensor_transform,
        monitor_transform=tensor_transform,
        epochs=2,
        batch_size=17,
    )
    return runner, archive, dest, config_path, sha, hooks


def test_training_updates_parameters_final_hash_counts_and_streams(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    output = tmp_path / "trained"
    result = runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert result["status"] == "COMPLETED"
    assert len(result["runs"]) == 5
    initial_hashes = set()
    stream_seeds = []
    inventory = json.loads((dest / "inventory.json").read_text())
    fit_count = sum(r["split"] == "source_fit" for r in inventory["rows"])
    for record in result["runs"]:
        checkpoint = torch.load(output / record["checkpoint"], weights_only=False)
        initial_hashes.add(record["initial_tensor_sha256"])
        stream_seeds.extend(record["seed_streams"].values())
        assert record["final_tensor_sha256"] == runner.tensor_hash(checkpoint["model"])
        assert record["initial_tensor_sha256"] != record["final_tensor_sha256"]
        assert record["processed_count"] == fit_count * 2
        assert record["batches"] == ((fit_count + 16) // 17) * 2
        assert checkpoint["optimizer"]["state"]
        records = [
            json.loads(line) for line in (output / f"seed-{record['seed']}" / "epochs.jsonl").read_text().splitlines()
        ]
        assert [r["epoch"] for r in records] == [1, 2]
        assert all(r["processed_count"] == fit_count for r in records)
    assert len(initial_hashes) == 5
    assert len(set(stream_seeds)) == 15
    with pytest.raises(FileExistsError):
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)


def test_preflight_has_no_eligible_checkpoint(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    output = tmp_path / "preflight"
    result = runner.run_training(
        archive, dest, config, sha, output, backend="cpu", preflight_batches=2, test_hooks=hooks
    )
    assert result["status"] == "PREFLIGHT_ONLY"
    assert result["steps"] == 2 and result["processed_count"] == 34
    assert result["mean_step_seconds"] > 0
    assert not list(output.rglob("*.pt"))


def test_failed_training_preserves_failed_receipt(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)

    def fail():
        raise RuntimeError("intentional synthetic model failure")

    hooks.model_factory = fail
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional"):
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert json.loads((output / "status.json").read_text())["status"] == "FAILED"
    assert not list(output.rglob("*.pt"))


def test_config_tampering_stops_before_model(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    config.write_text("{}")
    with pytest.raises(ValueError):
        runner.run_training(archive, dest, config, sha, tmp_path / "badconfig", backend="cpu", test_hooks=hooks)
    assert not (tmp_path / "badconfig").exists()


def test_cli_rejects_synthetic_identity(tmp_path):
    archive, listing, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        api().main(["--archive", str(archive), "--source-list", str(listing), "--output-dir", str(tmp_path / "out")])


@pytest.mark.parametrize(
    "fault",
    ["duplicate_zip", "encrypted", "outside", "backslash", "absolute", "member_size", "total_size", "member_count"],
)
def test_zip_metadata_gates(tmp_path, fault, monkeypatch):
    def change(entries, lines):
        if fault in {"outside", "backslash", "absolute"}:
            path = {"outside": "real/unselected.png", "backslash": "clipart\\bad", "absolute": "/clipart/bad"}[fault]
            entries.append((path, b"bad"))

    archive, listing, expected = fixture(tmp_path, change)
    raw = archive.read_bytes()
    if fault == "duplicate_zip":
        raw = raw.replace(b"clipart/class0/image2.png", b"clipart/class0/image0.png")
    if fault == "encrypted":
        raw = raw.replace(b"PK\x03\x04\x14\x00\x00\x00", b"PK\x03\x04\x14\x00\x01\x00")
        raw = raw.replace(b"PK\x01\x02\x14\x03\x14\x00\x00\x00", b"PK\x01\x02\x14\x03\x14\x00\x01\x00")
    archive.write_bytes(raw)
    expected["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    module = api()
    if fault in {"member_size", "total_size", "member_count"}:
        monkeypatch.setattr(
            module,
            {"member_size": "MAX_IMAGE_BYTES", "total_size": "MAX_TOTAL_BYTES", "member_count": "MAX_MEMBERS"}[fault],
            1,
        )
    with pytest.raises(ValueError):
        module.prepare_source(archive, listing, tmp_path / "out", identity=module.SourceIdentity(**expected))
    assert not (tmp_path / "out" / "summary.json").exists()


def test_rehashed_config_still_cannot_change_recipe(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    doc = json.loads(config.read_text())
    doc["model"]["weights"] = "DEFAULT"
    config.write_text(json.dumps(doc))
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        runner.run_training(archive, dest, config, sha, tmp_path / "bad", backend="cpu", test_hooks=hooks)


def test_optimizer_changes_each_intended_parameter(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    output = tmp_path / "allweights"
    result = runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    for record in result["runs"]:
        torch.manual_seed(100 + 3 * record["seed"])
        original = tiny_model().state_dict()
        trained = torch.load(output / record["checkpoint"], weights_only=False)["model"]
        assert all(not torch.equal(value, trained[name]) for name, value in original.items())


def test_failure_mid_epoch_has_no_completed_candidate(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    calls = 0

    def transform(image):
        nonlocal calls
        calls += 1
        if calls > 35:
            raise RuntimeError("synthetic mid-epoch failure")
        return tensor_transform(image)

    hooks.fit_transform = transform
    output = tmp_path / "midfailure"
    with pytest.raises(RuntimeError, match="mid-epoch"):
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert json.loads((output / "status.json").read_text())["status"] == "FAILED"
    assert not list(output.rglob("receipt.json"))
    assert not list(output.rglob("*.pt"))


def test_incomplete_json_publication_never_creates_final_receipt(tmp_path, monkeypatch):
    module = api()

    def fail(*args):
        raise OSError("synthetic fsync failure")

    import os

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="fsync"):
        module.write_new_json(tmp_path / "summary.json", {"status": "SUCCESS"})
    assert not (tmp_path / "summary.json").exists()


def test_archive_and_list_identities_checked_before_list_decoding(tmp_path):
    archive, listing, expected = fixture(tmp_path)
    listing.write_bytes(b"\xff")
    expected["list_sha256"] = hashlib.sha256(b"\xff").hexdigest()
    expected["list_git_blob"] = hashlib.sha1(b"blob 1\0\xff").hexdigest()
    expected["archive_sha256"] = "0" * 64
    module = api()
    with pytest.raises(ValueError, match="archive SHA256"):
        module.prepare_source(archive, listing, tmp_path / "out", identity=module.SourceIdentity(**expected))


def test_one_content_group_cannot_produce_two_splits(tmp_path):
    def same_content(entries, lines):
        for i, (name, _) in enumerate(entries):
            entries[i] = (name.replace("class1", "class0"), entries[0][1])
            lines[i] = lines[i].replace("class1", "class0").removesuffix(" 1")
            if not lines[i].endswith(" 0"):
                lines[i] += " 0"

    archive, listing, expected = fixture(tmp_path, same_content)
    expected["class_count"] = 1
    module = api()
    with pytest.raises(ValueError, match="empty split"):
        module.prepare_source(archive, listing, tmp_path / "out", identity=module.SourceIdentity(**expected))


@pytest.mark.parametrize("batches", [0, 21, True])
def test_preflight_out_of_range_is_rejected(tmp_path, batches):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    with pytest.raises(ValueError, match="1..20"):
        runner.run_training(
            archive, dest, config, sha, tmp_path / "bad", backend="cpu", preflight_batches=batches, test_hooks=hooks
        )


def test_training_wrong_source_role_is_rejected(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    inventory_path = dest / "inventory.json"
    doc = json.loads(inventory_path.read_text())
    doc["role"] = "test"
    inventory_path.chmod(0o644)
    inventory_path.write_text(json.dumps(doc))
    summary_path = dest / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    summary["role"] = "test"
    summary_path.chmod(0o644)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="reconstruction"):
        runner.run_training(archive, dest, config, sha, tmp_path / "bad", backend="cpu", test_hooks=hooks)


def test_checked_in_config_is_accepted(tmp_path):
    from pathlib import Path

    runner, archive, dest, _, _, hooks = training_fixture(tmp_path)
    config = Path(__file__).resolve().parents[1] / "protocols/confirmatory_v2/DOMAINNET_SOURCE_TRAINING_v1.json"
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    result = runner.run_training(
        archive, dest, config, sha, tmp_path / "configcheck", backend="cpu", preflight_batches=1, test_hooks=hooks
    )
    assert result["status"] == "PREFLIGHT_ONLY"


def test_checkpoint_safe_weights_only_roundtrip(tmp_path):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    output = tmp_path / "safecheckpoint"
    result = runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    checkpoint = torch.load(output / result["runs"][0]["checkpoint"], weights_only=True, map_location="cpu")
    assert runner.tensor_hash(checkpoint["model"]) == result["runs"][0]["final_tensor_sha256"]
    rng = checkpoint["rng"]["numpy"]
    import numpy as np

    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))


@pytest.mark.parametrize("fault", ["monitor", "gradient", "parameter"])
def test_nonfinite_training_state_rejected(tmp_path, fault):
    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)

    class AdversarialModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 12 * 12, 2)
            if fault == "gradient":
                self.linear.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
            elif fault == "parameter":
                # Finite maximal gradients accumulate to an infinite momentum buffer.
                self.linear.weight.register_hook(lambda grad: torch.full_like(grad, torch.finfo(grad.dtype).max))

        def forward(self, inputs):
            if fault == "monitor" and not self.training:
                return torch.full((len(inputs), 2), float("nan"))
            if fault == "parameter":
                return (self.linear(inputs.flatten(1)) * 0).nan_to_num()
            return self.linear(inputs.flatten(1))

    hooks.model_factory = AdversarialModel
    output = tmp_path / "nonfinite"
    with pytest.raises(ValueError, match="nonfinite"):
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert json.loads((output / "status.json").read_text())["status"] == "FAILED"
    assert not list(output.rglob("receipt.json"))


def test_persistent_aggregate_receipt_io_failure_cannot_publish_completed(tmp_path, monkeypatch):
    import errno
    import os

    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    output = tmp_path / "persistentfailure"
    original_write = runner.write_new_json
    original_error = OSError(errno.EIO, "aggregate receipt storage failure")
    failures = 0

    def persistent_fsync_failure(_fd):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise original_error
        raise OSError(errno.EIO, "subsequent failure-status storage failure")

    def publish(path, value):
        if path == output / "receipt.json":
            monkeypatch.setattr(os, "fsync", persistent_fsync_failure)
        return original_write(path, value)

    monkeypatch.setattr(runner, "write_new_json", publish)
    with pytest.raises(OSError) as caught:
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert not (output / "receipt.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "INCOMPLETE"
    assert caught.value is original_error
    assert len(list(output.glob("seed-*/receipt.json"))) == 5


def test_failure_status_io_error_does_not_replace_original_training_error(tmp_path, monkeypatch):
    import errno
    import os

    runner, archive, dest, config, sha, hooks = training_fixture(tmp_path)
    original_error = RuntimeError("original synthetic training failure")

    def persistent_fsync_failure(_fd):
        raise OSError(errno.EIO, "failure-status storage failure")

    def fail_model():
        monkeypatch.setattr(os, "fsync", persistent_fsync_failure)
        raise original_error

    hooks.model_factory = fail_model
    output = tmp_path / "originalfailure"
    with pytest.raises(RuntimeError) as caught:
        runner.run_training(archive, dest, config, sha, output, backend="cpu", test_hooks=hooks)
    assert caught.value is original_error
    assert not (output / "receipt.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "INCOMPLETE"


@pytest.mark.parametrize("argument, expected_code", [("--help", 0), ("--unknown-option", 2)])
def test_standalone_cli_cleans_only_owned_torch_temp_without_warnings(tmp_path, argument, expected_code):
    import os
    import subprocess
    import sys

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("caller-owned sentinel")
    script = """
import json, pathlib, runpy, sys, torch
torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.05)
owner = sys.modules['torch.distributed.nn.jit.instantiator']
temporary = owner._TEMP_DIR
path = pathlib.Path(temporary.name)
sys.argv = ['train_source', sys.argv[1]]
try:
    runpy.run_module('experiments.kbound.domainnet.train_source', run_name='__main__')
except SystemExit as exc:
    print(json.dumps({'exit_code': exc.code, 'temporary_exists': path.exists(), 'finalizer_alive': temporary._finalizer.alive}))
"""
    process = subprocess.run(
        [sys.executable, "-W", "error", "-c", script, argument],
        env={**os.environ, "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    observed = json.loads(process.stdout.splitlines()[-1])
    assert observed == {"exit_code": expected_code, "temporary_exists": False, "finalizer_alive": False}
    assert "ResourceWarning" not in process.stderr and "Implicitly cleaning up" not in process.stderr
    assert (unrelated / "keep.txt").read_text() == "caller-owned sentinel"


def test_library_cli_function_does_not_cleanup_active_torch_temp(tmp_path):
    import os
    import subprocess
    import sys

    script = """
import json, pathlib, sys, torch
from experiments.kbound.domainnet import train_source
torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.05)
temporary = sys.modules['torch.distributed.nn.jit.instantiator']._TEMP_DIR
try:
    train_source.main(['--help'])
except SystemExit:
    print(json.dumps({'temporary_exists': pathlib.Path(temporary.name).exists(), 'finalizer_alive': temporary._finalizer.alive}))
finally:
    temporary.cleanup()
"""
    process = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env={**os.environ, "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout.splitlines()[-1]) == {"temporary_exists": True, "finalizer_alive": True}
    assert not process.stderr


def test_standalone_cleanup_failure_preserves_original_cli_error(monkeypatch):
    runner = train_api()
    original = ValueError("original CLI failure")

    def fail_main():
        raise original

    def fail_cleanup():
        raise OSError("synthetic scoped cleanup failure")

    # Inject only our local CLI functions; never patch tempfile or Torch globals.
    monkeypatch.setattr(runner, "main", fail_main)
    monkeypatch.setattr(runner, "_cleanup_standalone_torch_temp", fail_cleanup)
    with pytest.raises(ValueError) as caught:
        runner._standalone_main()
    assert caught.value is original
    assert any("synthetic scoped cleanup failure" in note for note in original.__notes__)


@pytest.mark.parametrize("exit_code", [None, 0])
def test_standalone_cleanup_failure_is_reported_on_otherwise_success(monkeypatch, exit_code):
    runner = train_api()

    def fail_cleanup():
        raise OSError("synthetic scoped cleanup failure")

    def successful_main():
        if exit_code is not None:
            raise SystemExit(exit_code)

    monkeypatch.setattr(runner, "main", successful_main)
    monkeypatch.setattr(runner, "_cleanup_standalone_torch_temp", fail_cleanup)
    with pytest.raises(OSError, match="synthetic scoped cleanup failure"):
        runner._standalone_main()


def test_cleanup_helper_rejects_inherited_torch_directory_after_fork(tmp_path):
    import os
    import subprocess
    import sys

    script = """
import json, os, pathlib, psutil, sys, torch
from experiments.kbound.domainnet import train_source
torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.05)
temporary = sys.modules['torch.distributed.nn.jit.instantiator']._TEMP_DIR
path = pathlib.Path(temporary.name)
reader, writer = os.pipe()
child = os.fork()
if child == 0:
    os.close(reader)
    record = {'birth_time': path.stat().st_birthtime, 'child_start_time': psutil.Process().create_time()}
    try:
        train_source._cleanup_standalone_torch_temp()
        record['refused'] = False
    except RuntimeError:
        record['refused'] = True
    os.write(writer, json.dumps(record).encode())
    os._exit(0)  # Isolate helper ownership: do not execute inherited Python finalizers.
os.close(writer)
record = json.loads(os.read(reader, 4096))
os.close(reader)
os.waitpid(child, 0)
record['parent_directory_survived'] = path.exists()
print(json.dumps(record))
temporary.cleanup()
"""
    process = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env={**os.environ, "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    observed = json.loads(process.stdout)
    assert observed["birth_time"] < observed["child_start_time"]
    assert observed["refused"] is True and observed["parent_directory_survived"] is True
    assert not process.stderr


@pytest.mark.parametrize("fault", ["missing_birthtime", "psutil_error"])
def test_cleanup_refuses_missing_process_ownership_evidence(monkeypatch, fault):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import psutil

    runner = train_api()
    torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.05)
    temporary = sys.modules["torch.distributed.nn.jit.instantiator"]._TEMP_DIR
    target = Path(temporary.name)
    if fault == "missing_birthtime":
        original_stat = Path.stat

        def unavailable_birthtime(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            return SimpleNamespace(st_mode=result.st_mode, st_uid=result.st_uid) if path == target else result

        monkeypatch.setattr(Path, "stat", unavailable_birthtime)
    else:

        def unavailable_process(*args, **kwargs):
            raise psutil.AccessDenied()

        monkeypatch.setattr(psutil, "Process", unavailable_process)
    with pytest.raises(RuntimeError, match="process"):
        runner._cleanup_standalone_torch_temp()
    assert target.exists() and temporary._finalizer.alive
