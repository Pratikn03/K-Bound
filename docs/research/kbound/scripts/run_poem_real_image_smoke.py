#!/usr/bin/env python3
"""Execute authenticated POEM on a fixed real-image smoke panel.

Full pretrained ResNet50-GN and upstream loss/protector/SAR masks are used.
The driver, bounded sample selection and CPU placement are disclosed adaptations.
No labels are loaded, no KGA decisions are claimed, and this is not a benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import types

import task3_image_panel as panel_io
from task3_image_panel import digest
import poem_native_cpu_seam as seam
import poem_dependency_bootstrap as bootstrap

WEIGHTS_SHA256 = "8fe6c4d08094e791f17c7835e1bdd3a64fccc0b849bfc1ea339b3bbf372fac85"
WEIGHTS_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth"
ROW_FIELDS = {"relative_path", "sample_id", "sha256", "size_bytes", "width", "height", "decoded_rgb"}
NATIVE_HASHES = {**seam.HASHES, "dataset/selectedRotateImageFolder.py":
                 "ab6db8b42ab29a3a9857c4a7df4a4d6f67b49e29f05e52a07cf8378d813c5e4b"}


def check_resources():
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if size > 2 * 1024**3:
        raise RuntimeError("CPU smoke exceeded2GiB resident ceiling")
    return size


def write_artifact(directory_fd, name, data):
    if Path(name).name != name:
        raise ValueError("artifact must be a basename")
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    return digest(data)


def validate_panel(panel):
    expected = {"schema": "kbound-real-image-smoke-panel-v1", "scope": "ENGINEERING_SMOKE_NOT_BENCHMARK",
                "outcomes_read": False, "upstream_full_dataset_protocol": False,
                "official_image_bytes_authenticated": False, "seed": 0, "batch_size": 4,
                "condition": "gaussian_noise/5"}
    for key, value in expected.items():
        if panel.get(key) != value or type(panel.get(key)) is not type(value):
            raise ValueError(f"unexpected locked smoke {key}")
    if any(not Path(panel.get(key, "")).is_absolute() for key in ("clean_root", "corruption_root")):
        raise ValueError("absolute dataset roots required")
    identities = []
    for role, count in (("source", 32), ("target", 104)):
        rows = panel.get(role)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("incorrect smoke role counts")
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                raise ValueError("unexpected image fields; outcomes are forbidden")
            path = row["relative_path"]
            prefix = "" if role == "source" else "gaussian_noise/5/"
            if not isinstance(path, str) or not path.startswith(prefix):
                raise ValueError("unexpected condition")
            relative = path[len(prefix):]
            if not panel_io.NAME.fullmatch(relative) or Path(path).stem != row["sample_id"]:
                raise ValueError("underlying sample identity mismatch")
            if (not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    or type(row["size_bytes"]) is not int or not 0 < row["size_bytes"] <= 32 * 1024**2
                    or any(type(row[k]) is not int or row[k] < 1 for k in ("width", "height"))
                    or row["decoded_rgb"] is not True):
                raise ValueError("invalid image integrity record")
            identities.append(row["sample_id"])
    if len(identities) != len(set(identities)):
        raise ValueError("overlapping or duplicate underlying sample IDs")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_panel(path, expected_sha256):
    data = Path(path).read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("manifest digest mismatch")
    panel = json.loads(data, object_pairs_hook=unique_object)
    validate_panel(panel)
    return panel


def derive_temperature(payload, device):
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("unknown device")
    old = b"self.temperature = nn.Parameter(torch.ones(1) * temp).cuda()"
    if payload.count(old) != 1:
        raise ValueError("expected one exact authenticated constructor placement")
    new = f"self.temperature = nn.Parameter(torch.ones(1) * temp).to('{device}')".encode()
    derived = payload.replace(old, new, 1)
    return derived, {"original_sha256": digest(payload), "derived_sha256": digest(derived),
                     "old": old.decode(), "new": new.decode(), "scope": "constructor placement only"}


def load_native(source, record):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    if head != bootstrap.POEM_COMMIT:
        raise ValueError("POEM source revision mismatch")
    payloads = {}
    for name, expected in NATIVE_HASHES.items():
        data = bootstrap.authenticated_bytes(source / name, expected)
        pinned = subprocess.check_output(["git", "-C", str(source), "show", f"{head}:{name}"], timeout=15)
        if pinned != data:
            raise ValueError("working source differs from pinned upstream bytes")
        payloads[name] = data
    derived, placement = derive_temperature(payloads["temperature_scaling.py"], "cpu")
    record.update(upstream_commit=head, source_sha256=dict(NATIVE_HASHES), device_derivation=placement)
    modules = {}
    for name in ("cdf", "protector", "poem", "sar", "temperature_scaling"):
        if name in sys.modules:
            raise RuntimeError("native module already imported; fresh process required")
        module = types.ModuleType(name)
        module.__file__ = str(source / f"{name}.py")
        sys.modules[name] = module
        exec(compile(derived if name == "temperature_scaling" else payloads[name + ".py"],
                     module.__file__, "exec"), module.__dict__)
        modules[name] = module
    return modules


def run(args, panel, record):
    import numpy as np
    import timm
    import torch
    import torchvision
    from torchvision import transforms

    if timm.__version__ != "0.6.11" or torch.__version__ != "2.1.1":
        raise ValueError("explicit POEM runtime requires timm0.6.11 and torch2.1.1")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    native = load_native(args.poem_source, record)
    weights_bytes = args.checkpoint.read_bytes()
    if digest(weights_bytes) != WEIGHTS_SHA256:
        raise ValueError("checkpoint digest does not match approved official GN release")
    record["checkpoint"] = {"path": str(args.checkpoint), "sha256": WEIGHTS_SHA256,
                            "size_bytes": len(weights_bytes), "url": WEIGHTS_URL}
    state = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    del weights_bytes
    backbone = timm.create_model("resnet50_gn", pretrained=False)
    backbone.load_state_dict(state, strict=True)
    del state
    model = native["temperature_scaling"].ModelWithTemperature(backbone.cpu().eval(), 0.90)
    check_resources()
    record["runtime"] = {"python": sys.version, "torch": torch.__version__, "timm": timm.__version__,
                         "torchvision": torchvision.__version__, "numpy": np.__version__, "device": "cpu"}
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    source_transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                                           transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    target_transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    record["preprocessing"] = {"clean": "PIL RGB;Resize256 bilinear;CenterCrop224;ToTensor;ImageNet normalize",
                               "target": "PIL RGB;CenterCrop224;ToTensor;ImageNet normalize",
                               "reference": "pinned POEM selectedRotateImageFolder.prepare_test_data ResNet branch"}
    def batches(role):
        root = panel["clean_root" if role == "source" else "corruption_root"]
        transform = source_transform if role == "source" else target_transform
        for index in range(0, len(panel[role]), 4):
            rows = panel[role][index:index + 4]
            inputs = torch.stack([transform(panel_io.read_bound_image(root, row)) for row in rows])
            yield rows, inputs
            check_resources()
    started = time.monotonic()
    source_ents = []
    with torch.no_grad():
        for _, inputs in batches("source"):
            source_ents.extend(native["poem"].softmax_entropy(model(inputs)).tolist())
    if not np.isfinite(source_ents).all() or len(set(source_ents)) < 2:
        raise ValueError("invalid native source entropy reference")
    record["source_calibration"] = {"count": 32, "entropy_values": source_ents,
                                    "seconds": time.monotonic() - started,
                                    "scope": "bounded32-image reference; not full12500-image native holdout"}
    # Frozen outputs are produced before candidate construction; this does not
    # mutate model state or call the adaptation object twice on a target batch.
    frozen = []
    with torch.no_grad():
        for _, inputs in batches("target"):
            frozen.append(model(inputs).cpu().numpy())
    model = native["sar"].configure_model(model)
    params, names = native["sar"].collect_params(model)
    lr = .00025 / 64 * 4 * 2
    optimizer = torch.optim.SGD(params, lr=lr, momentum=.9)
    protector = native["protector"].get_protector_from_ents(source_ents,
                    types.SimpleNamespace(gamma=1 / (8 * math.sqrt(3)), eps_clip=1.8, device="cpu"))
    adapter = native["poem"].POEM(model, optimizer, protector,
                    e0=.4 * math.log(1000), vanilla_loss=True, episodic=False)
    adapter.eval()
    record["recipe"] = {"temperature": .90, "lr": lr, "momentum": .9,
                         "e0": .4 * math.log(1000), "gamma": 1 / (8 * math.sqrt(3)),
                         "eps_clip": 1.8, "vanilla_loss": True, "steps": 1,
                         "native_warmup_samples": 100, "parameter_names": names,
                         "prediction_timing": "native non-episodic return before current optimizer step",
                         "reset": "fresh native object for this sole stream; no partial-reset reuse"}
    adapted, seen, warmup_unchanged = [], [], False
    started = time.monotonic()
    for rows, inputs in batches("target"):
        with torch.no_grad():
            output = adapter(inputs)
        if not bool(torch.isfinite(output).all()):
            raise ValueError("nonfinite native output")
        adapted.append(output.detach().cpu().numpy())
        seen.extend(row["sample_id"] for row in rows)
        if adapter.curr_n_samples == 96:
            warmup_unchanged = all(torch.equal(v, adapter.model_state[k]) for k, v in model.state_dict().items())
        print(json.dumps({"stage": "native_poem_real_images", "complete": len(seen), "total": 104}), flush=True)
        check_resources()
    record["adaptation_seconds"] = time.monotonic() - started
    record["changed_parameter_tensors"] = sum(not torch.equal(value, adapter.model_state[name])
                                             for name, value in model.named_parameters())
    record["checks"] = {"ordered_ids_match": seen == [r["sample_id"] for r in panel["target"]],
                        "warmup96_model_unchanged": warmup_unchanged,
                        "all104_processed": adapter.curr_n_samples == 104,
                        "optimizer_ran": len(optimizer.state) > 0,
                        "finite_parameters": all(bool(torch.isfinite(v).all()) for v in model.parameters()),
                        "protector_saw104": len(protector.info["z_before"]) == 104,
                        "source_unchanged": all(bootstrap.authenticated_bytes(args.poem_source / n, h)
                                                for n, h in NATIVE_HASHES.items())}
    if not all(record["checks"].values()):
        raise ValueError("real-image native execution checks failed")
    frozen, adapted = np.concatenate(frozen), np.concatenate(adapted)
    logits_buffer = io.BytesIO()
    np.savez_compressed(logits_buffer, frozen=frozen, adapted=adapted, sample_ids=np.array(seen))
    predictions = [{"sample_id": sid, "frozen_class": int(a), "poem_class": int(b)}
                   for sid, a, b in zip(seen, frozen.argmax(1), adapted.argmax(1))]
    record["outputs"] = {
        "logits.npz": write_artifact(args.output_fd, "logits.npz", logits_buffer.getvalue()),
        "predictions.json": write_artifact(args.output_fd, "predictions.json",
                           (json.dumps(predictions, indent=2, allow_nan=False) + "\n").encode())}
    record["prediction_changes"] = int((frozen.argmax(1) != adapted.argmax(1)).sum())
    record["status"] = "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--poem-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runtime = os.environ.get("POEM_PYTHON")
    if not runtime or Path(runtime).resolve() != Path(sys.executable).resolve():
        parser.error("POEM_PYTHON must explicitly match this prepared interpreter")
    panel = read_panel(args.manifest, args.manifest_sha256)
    for root in (panel["clean_root"], panel["corruption_root"], str(args.poem_source)):
        root = Path(root).resolve()
        if args.output.resolve() == root or root in args.output.resolve().parents:
            parser.error("output must be outside datasets and upstream source")
    fd = panel_io._open_directory_chain(args.output.parent)
    try:
        os.mkdir(args.output.name, mode=0o700, dir_fd=fd)
        args.output_fd = os.open(args.output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    start = time.monotonic()
    record = {"schema": "kbound-native-poem-real-smoke-v1", "status": "INCOMPLETE",
              "started_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": args.manifest_sha256,
              "manifest": str(args.manifest), "harness_sha256": digest(Path(__file__).read_bytes()),
              "helpers": {Path(m.__file__).name: digest(Path(m.__file__).read_bytes())
                          for m in (panel_io, seam, bootstrap)},
              "native_main_executed": False, "cuda_parity_verified": False, "benchmark_completed": False,
              "accuracy_measured": False, "evaluation_labels_read": False,
              "pretrained": True, "model": "full_resnet50_gn", "spatial_size": 224,
              "source_images": 32, "target_images": 104, "cpu_threads": 2,
              "wall_deadline_seconds": 600, "no_automatic_downloads": True}
    def deadline(signum, frame):
        raise TimeoutError("600-second bounded smoke deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        run(args, panel, record)
        actual, expected = args.output.lstat(), os.fstat(args.output_fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("output pathname replaced during execution; artifacts retained at original descriptor")
    except Exception as exc:
        record.update(status="FAILED_REAL_IMAGE_SMOKE", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        record["elapsed_seconds"] = time.monotonic() - start
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        try:
            write_artifact(args.output_fd, "receipt.json",
                           (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        finally:
            os.close(args.output_fd)
    print(json.dumps({"status": record["status"], "output": str(args.output)}), flush=True)
    return 0 if record["status"] == "PASS_REAL_IMAGE_ENGINEERING_SMOKE_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
