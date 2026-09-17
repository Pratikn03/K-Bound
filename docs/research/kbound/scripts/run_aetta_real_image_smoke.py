#!/usr/bin/env python3
"""Native AETTA estimator on one fixed real-image engineering smoke panel.

Uses authenticated torchvision ResNet-50 BN V2, not POEM's GN checkpoint and
not an exact published AETTA benchmark recipe. The model is frozen: no adapter,
optimization, labels, measured accuracy, KGA action or benchmark claim exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time

import aetta_native_estimator as native_aetta
import task3_image_panel as panel_io

PANEL_SHA256 = "79e6d59fc9240ed2daefe53aadaa4502a35ca0ff25ac8fa211d44adcfe59f92b"
CHECKPOINT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
CHECKPOINT_SIZE = 102540417
CHECKPOINT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
ROW_FIELDS = {"relative_path", "sample_id", "sha256", "size_bytes", "width", "height", "decoded_rgb"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_regular(path, limit, exact_size=None):
    """Bind bytes to one bounded regular descriptor, not a symlink pathname."""
    path = Path(os.path.abspath(path))
    parent = panel_io._open_directory_chain(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("input must be a regular file")
            if not 0 < before.st_size <= limit or (exact_size is not None and before.st_size != exact_size):
                raise ValueError("input size does not match bounded input contract")
            data = handle.read(limit + 1)
            after = os.fstat(handle.fileno())
            if len(data) != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("input changed during read")
            return data
    finally:
        os.close(parent)


def read_checkpoint(path):
    data = read_regular(path, CHECKPOINT_SIZE, exact_size=CHECKPOINT_SIZE)
    if digest(data) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint digest does not match authenticated ResNet-50 BN V2 release")
    return data


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def validate_panel(panel):
    expected = dict(schema="kbound-real-image-smoke-panel-v1", scope="ENGINEERING_SMOKE_NOT_BENCHMARK",
                    outcomes_read=False, upstream_full_dataset_protocol=False, official_image_bytes_authenticated=False,
                    seed=0, batch_size=4, condition="gaussian_noise/5")
    if not isinstance(panel, dict):
        raise ValueError("expected a locked panel object")
    for key, value in expected.items():
        if panel.get(key) != value or type(panel.get(key)) is not type(value):
            raise ValueError(f"unexpected locked smoke {key}")
    if any(not isinstance(panel.get(key), str) or not Path(panel[key]).is_absolute()
           for key in ("clean_root", "corruption_root")):
        raise ValueError("absolute dataset roots required")
    seen = []
    for role, count in (("source", 32), ("target", 104)):
        rows = panel.get(role)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("incorrect fixed panel counts")
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                raise ValueError("unexpected row fields; outcomes are forbidden")
            prefix = "" if role == "source" else "gaussian_noise/5/"
            path = row["relative_path"]
            if (not isinstance(path, str) or not path.startswith(prefix)
                    or not panel_io.NAME.fullmatch(path[len(prefix):])
                    or Path(path).stem != row["sample_id"]):
                raise ValueError("image identity or condition mismatch")
            if (not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    or type(row["size_bytes"]) is not int or not 0 < row["size_bytes"] <= 32 * 1024**2
                    or any(type(row[k]) is not int or row[k] <= 0 for k in ("width", "height"))
                    or row["decoded_rgb"] is not True):
                raise ValueError("invalid image integrity record")
            seen.append(row["sample_id"])
    if len(seen) != len(set(seen)):
        raise ValueError("overlapping or duplicate image IDs")


def read_panel(path):
    data = read_regular(path, 1024**2)
    if digest(data) != PANEL_SHA256:
        raise ValueError("panel digest does not match the reviewed fixed 104-image selection")
    panel = json.loads(data, object_pairs_hook=unique_object)
    validate_panel(panel)
    return panel


def validate_output(output, panel, source):
    output = Path(output).resolve()
    for root in (panel["clean_root"], panel["corruption_root"], source):
        root = Path(root).resolve()
        if output == root or root in output.parents:
            raise ValueError("output must be outside datasets and upstream source")


def write_artifact(directory_fd, name, data):
    if Path(name).name != name:
        raise ValueError("artifact name must be a basename")
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    return digest(data)


def check_resources():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if size > 2 * 1024**3:
        raise RuntimeError("native AETTA smoke exceeded 2 GiB resident ceiling")
    return size


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        value.update(len(metadata).to_bytes(8, "big"))
        value.update(metadata)
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def verify_execution(expected_ids, seen_ids, batch_count, state_before, state_after, forward_ok):
    checks = {
        "all104_target_ids_in_locked_order": expected_ids == seen_ids and len(seen_ids) == 104,
        "all26_batches_completed": batch_count == 26,
        "complete_model_state_unchanged": state_before == state_after,
        "native_11_forwards_per_batch": forward_ok is True,
    }
    if not all(checks.values()):
        raise ValueError("native AETTA real-image checks failed: " + str([key for key, passed in checks.items() if not passed]))
    return checks


def run(args, panel, record):
    # Authenticate source and checkpoint bytes before any model construction.
    native_aetta.authenticate_source(args.aetta_source, auth_receipt=args.auth_receipt)
    weights = read_checkpoint(args.checkpoint)
    import torch
    import torchvision
    from torchvision import transforms

    if torch.__version__ != "2.8.0" or torchvision.__version__ != "0.23.0":
        raise ValueError("This reviewed AETTA smoke requires torch 2.8.0 and torchvision 0.23.0")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = native_aetta.NativeAETTASource(args.aetta_source, auth_receipt=args.auth_receipt)
    model = source.create_model()
    loaded = model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Native ResNet-50 strict checkpoint loading did not match")
    del weights
    episode = source.new_episode(model)
    torch.manual_seed(0)  # Stream dropout RNG is independent of constructor initialization.
    before = state_digest(model)
    record.update(native_estimator=source.provenance, runtime=source.provenance["runtime"],
                  checkpoint={"path": str(args.checkpoint), "sha256": CHECKPOINT_SHA256,
                              "size_bytes": CHECKPOINT_SIZE, "url": CHECKPOINT_URL,
                              "weight_identifier": "torchvision.ResNet50_Weights.IMAGENET1K_V2",
                              "strict_load": True, "native_published_default_checkpoint_claimed": False},
                  model_state_sha256_before=before, model_parameter_count=sum(p.numel() for p in model.parameters()),
                  recipe={"source_model": "full ResNet-50 BN V2", "candidate": "same frozen source model; no adapter",
                          "seed": 0, "dropout_seed_reset_after_weight_load": True, "batch_size": 4,
                          "dropout_draws": 10, "dropout_probability": 0.5,
                          "ema": "native 0.6 previous error + 0.4 corrected error; one retained state over 26 batches",
                          "prediction_timing": "first deterministic native forward before dropout estimation; model never updated",
                          "same_checkpoint_as_poem_gn_smoke": False},
                  preprocessing="PIL RGB; CenterCrop224; ToTensor; exactly one native ImageNet normalization inside AETTA",
                  clean_source_images_read=0, evaluation_labels_read=False, accuracy_measured=False,
                  folder_paths_used_only_to_open_bound_images=True)
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    batches, predictions, seen = [], [], []
    observed = []
    deterministic = []

    def observe_forward(module, inputs, kwargs, output):
        probability = kwargs.get("dropout", 0.0)
        observed.append(probability)
        if probability == 0.0:
            deterministic.append(output.detach())

    handle = model.register_forward_hook(observe_forward, with_kwargs=True)
    forward_ok = True
    started = time.monotonic()
    try:
        for offset in range(0, 104, 4):
            rows = panel["target"][offset:offset + 4]
            images = torch.stack([transform(panel_io.read_bound_image(panel["corruption_root"], row)) for row in rows])
            observed.clear()
            deterministic.clear()
            result = episode.estimate(images)
            valid = observed == [0.0] + [0.5] * 10 and len(deterministic) == 1
            forward_ok = forward_ok and valid
            if not valid or tuple(deterministic[0].shape) != (4, 1000) or not torch.isfinite(deterministic[0]).all():
                raise ValueError("Native deterministic/dropout forward contract failed")
            classes = deterministic[0].argmax(dim=1).tolist()
            ids = [row["sample_id"] for row in rows]
            batches.append({"batch_index": offset // 4, "sample_ids": ids,
                            "image_sha256": [row["sha256"] for row in rows],
                            "input_tensor_sha256": digest(images.numpy().tobytes()),
                            "estimate": result, "observed_dropout_forward_probabilities": list(observed)})
            predictions.extend({"sample_id": row["sample_id"], "predicted_class": int(predicted),
                                "image_sha256": row["sha256"]} for row, predicted in zip(rows, classes))
            seen.extend(ids)
            check_resources()
            print(json.dumps({"stage": "native_aetta_frozen_bn_v2_smoke", "complete": len(seen), "total": 104}), flush=True)
    finally:
        handle.remove()
    after = state_digest(model)
    record.update(estimation_seconds=time.monotonic() - started, model_state_sha256_after=after,
                  final_native_estimator_state=episode.estimator_state, model_updated=False,
                  checks=verify_execution([row["sample_id"] for row in panel["target"]], seen,
                                          len(batches), before, after, forward_ok))
    record["outputs"] = {
        "batch_estimates.json": write_artifact(args.output_fd, "batch_estimates.json",
                                                (json.dumps(batches, indent=2, allow_nan=False) + "\n").encode()),
        "predictions.json": write_artifact(args.output_fd, "predictions.json",
                                            (json.dumps(predictions, indent=2, allow_nan=False) + "\n").encode()),
    }
    record["status"] = "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--aetta-source", required=True, type=Path)
    parser.add_argument("--auth-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runtime = os.environ.get("AETTA_PYTHON")
    if not runtime or Path(runtime).resolve() != Path(sys.executable).resolve():
        parser.error("AETTA_PYTHON must explicitly match this prepared interpreter")
    panel = read_panel(args.manifest)
    validate_output(args.output, panel, args.aetta_source)
    parent_fd = panel_io._open_directory_chain(args.output.parent)
    try:
        os.mkdir(args.output.name, mode=0o700, dir_fd=parent_fd)
        args.output_fd = os.open(args.output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    started = time.monotonic()
    record = {
        "schema": "kbound-native-aetta-real-smoke-v1", "status": "INCOMPLETE",
        "started_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": PANEL_SHA256,
        "manifest": str(args.manifest), "harness_sha256": digest(Path(__file__).read_bytes()),
        "helpers": {Path(module.__file__).name: digest(Path(module.__file__).read_bytes())
                    for module in (native_aetta, panel_io)},
        "scope": "FROZEN_BN_V2_AETTA_ESTIMATOR_ENGINEERING_SMOKE_NOT_BENCHMARK",
        "native_main_executed": False, "cuda_parity_verified": False,
        "benchmark_completed": False, "accuracy_measured": False, "evaluation_labels_read": False,
        "pretrained": True, "model": "full_resnet50_bn_v2", "spatial_size": 224,
        "target_images": 104, "clean_source_images_read": 0, "cpu_threads": 2,
        "wall_deadline_seconds": 180, "no_automatic_downloads": True, "kga_actions_produced": False,
    }

    def deadline(signum, frame):
        raise TimeoutError("180-second bounded AETTA smoke deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    try:
        run(args, panel, record)
        actual, retained = args.output.lstat(), os.fstat(args.output_fd)
        if (actual.st_dev, actual.st_ino) != (retained.st_dev, retained.st_ino):
            raise ValueError("output pathname replaced during execution; retained descriptor used for artifacts")
    except Exception as exc:
        record.update(status="FAILED_REAL_IMAGE_SMOKE", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        record.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        try:
            write_artifact(args.output_fd, "receipt.json",
                           (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        finally:
            os.close(args.output_fd)
    print(json.dumps({"status": record["status"], "output": str(args.output)}), flush=True)
    return 0 if record["status"] == "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
