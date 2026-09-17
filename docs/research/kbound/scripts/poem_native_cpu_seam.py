#!/usr/bin/env python3
"""Bounded engineering diagnostic, not native-driver or benchmark evidence.

Only temperature-parameter placement is derived in memory. POEM, SAR model
configuration, ECDF and Protector execute authenticated upstream bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import types

import poem_dependency_bootstrap as bootstrap
from poem_dependency_bootstrap import MAIN_SHA256, POEM_COMMIT, authenticated_bytes

HASHES = {
    "main.py": MAIN_SHA256,
    "poem.py": "3436e95a07a7edba7f43ff958a3c839fd273032f3cc4c338ec4aa968a56d00d7",
    "protector.py": "f539a54904a9681bee48f881e49ad6c676d63a3385fc11dbec8f19ebd3fee83e",
    "cdf.py": "1c60d4f873baf4a7da1314ad6a708fca1842a67b6e71df75881dec1899aeec26",
    "temperature_scaling.py": "78c994c12675e648a001e84a58b1359d85025ab86ff9ee20dd8b2a681f0bb498",
    "sar.py": "74ab244bf3e2d1d13f02c289fb8272321bbe00e0eeaa9cf8d1337ce2add7d6ab",
    "sam.py": "b0569de29015016996feae257d30be6cc36f80d42d0f51d4a201eb39d2491712",
}


def diagnostic(record, source):
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    if head != POEM_COMMIT:
        raise ValueError("POEM source revision mismatch")
    payloads = {}
    for name, digest in HASHES.items():
        payload = authenticated_bytes(source / name, digest)
        original = subprocess.check_output(["git", "-C", str(source), "show", f"{POEM_COMMIT}:{name}"], timeout=10)
        if payload != original:
            raise ValueError(f"Working source does not match pinned Git bytes: {name}")
        payloads[name] = payload
    import numpy as np
    import scipy
    import timm
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(71)
    np.random.seed(71)
    record.update(source_sha256=HASHES, upstream_commit=head, torch=torch.__version__,
                  timm=timm.__version__, scipy=scipy.__version__)
    original = payloads["temperature_scaling.py"]
    placement = b"self.temperature = nn.Parameter(torch.ones(1) * temp).cuda()"
    replacement = b"self.temperature = nn.Parameter(torch.ones(1) * temp).to('cpu')"
    if original.count(placement) != 1:
        raise ValueError("Expected exactly one authenticated constructor placement")
    derived = original.replace(placement, replacement, 1)
    record["derivation"] = {
        "file": "temperature_scaling.py", "old": placement.decode(), "new": replacement.decode(),
        "original_sha256": HASHES["temperature_scaling.py"],
        "derived_sha256": hashlib.sha256(derived).hexdigest(),
        "scope": "constructor parameter device only; no global CUDA monkeypatch",
        "unmodified_constructor_executed": False,
    }
    modules = {}
    for name in ("cdf", "protector", "poem", "sar", "temperature_scaling"):
        if name in sys.modules:
            raise RuntimeError(f"Module already imported: {name}; use fresh process")
        module = types.ModuleType(name)
        module.__file__ = str(source / f"{name}.py")
        sys.modules[name] = module
        exec(compile(derived if name == "temperature_scaling" else payloads[f"{name}.py"],
                     module.__file__, "exec"), module.__dict__)
        modules[name] = module
    model = timm.create_model("resnet50_gn", pretrained=False).cpu().eval()
    # Fixed low-entropy synthetic fixture, matching the earlier core smoke.
    # This is not pretrained weight loading or source/target calibration tuning.
    with torch.no_grad():
        model.fc.bias[0] = 12.0
    model = modules["temperature_scaling"].ModelWithTemperature(model, 0.90)
    generator = torch.Generator(device="cpu").manual_seed(1729)
    source_images = torch.randn(16, 3, 32, 32, generator=generator)
    target_images = torch.randn(104, 3, 32, 32, generator=generator) * 0.2 + 3.0
    tensor_hash = lambda value: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
    record["fixture"] = {"source_tensor_sha256": tensor_hash(source_images),
                         "target_tensor_sha256": tensor_hash(target_images),
                         "source_ids": [f"synthetic-source-{i:03}" for i in range(16)],
                         "target_ids": [f"synthetic-target-{i:03}" for i in range(104)],
                         "target_distribution": "randn * 0.2 + 3.0; fixed before execution"}
    entropy = modules["poem"].softmax_entropy
    source_ents = []
    with torch.no_grad():
        for images in source_images.split(4):
            source_ents.extend(entropy(model(images)).tolist())
    args = types.SimpleNamespace(gamma=1 / (8 * math.sqrt(3)), eps_clip=1.8, device="cpu")
    protector = modules["protector"].get_protector_from_ents(source_ents, args)
    record["source_entropy"] = {"count": len(source_ents), "values": source_ents,
                                "unique_values": len(set(source_ents))}
    checks = record["checks"] = {
        "temperature_parameter_cpu": model.temperature.device.type == "cpu",
        "source_entropies_finite_and_nonconstant": bool(np.isfinite(source_ents).all() and len(set(source_ents)) > 1),
        "native_cdf_quantiles_equal_observed_source_values": bool(np.array_equal(protector.cdf.q, np.unique(source_ents))),
    }
    model = modules["sar"].configure_model(model)
    params, names = modules["sar"].collect_params(model)
    lr = (0.00025 / 64) * 4 * 2
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    adapt = modules["poem"].POEM(model, optimizer, protector,
                                 e0=0.4 * math.log(1000), vanilla_loss=True)
    # Native main.run uses eval mode and an outer no_grad context; the upstream
    # forward_and_adapt decorator explicitly enables gradients for the update.
    adapt.eval()
    record["native_run_eval_mode"] = not adapt.training and not model.training
    record["expected_batch_sha256"] = [tensor_hash(batch) for batch in target_images.split(4)]
    observed = record["observed_batch_sha256"] = []
    observer = adapt.register_forward_pre_hook(lambda module, inputs: observed.append(tensor_hash(inputs[0])))
    record["optimizer"] = {"type": "SGD", "lr": lr, "momentum": 0.9, "parameter_names": names}
    record["setup_elapsed_seconds"] = time.monotonic() - record.pop("_started")
    start = time.monotonic()
    finite_outputs, seen, warmup_unchanged = True, [], False
    for index, batch in enumerate(target_images.split(4)):
        with torch.no_grad():
            output = adapt(batch)
        finite_outputs &= bool(torch.isfinite(output).all())
        seen.extend(record["fixture"]["target_ids"][index * 4:(index + 1) * 4])
        if adapt.curr_n_samples == 96:
            warmup_unchanged = all(torch.equal(v, adapt.model_state[k]) for k, v in model.state_dict().items())
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) > 2 * 1024**3:
            raise RuntimeError("CPU diagnostic exceeded 2 GiB resident ceiling")
    record["adapt_elapsed_seconds"] = time.monotonic() - start
    observer.remove()
    record["parameter_tensors_changed"] = sum(not torch.equal(p, adapt.model_state[n]) for n, p in model.named_parameters())
    checks.update(finite_outputs=finite_outputs, warmup_96_samples_unchanged=warmup_unchanged,
                  update_after_native_100_sample_warmup=record["parameter_tensors_changed"] > 0,
                  finite_parameters=all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                  ordered_sample_ids_exact=seen == record["fixture"]["target_ids"] and observed == record["expected_batch_sha256"],
                  protector_saw_each_sample_once=len(protector.info["z_before"]) == 104,
                  optimizer_momentum_populated=len(optimizer.state) > 0,
                  finite_protector_trace=bool(np.isfinite(protector.martingales).all() and np.isfinite(protector.epsilons).all()))
    protector_count = len(protector.info["z_before"])
    adapt.reset()
    checks["native_reset_restores_model"] = all(torch.equal(v, adapt.model_state[k]) for k, v in model.state_dict().items())
    checks["native_reset_restores_empty_optimizer_state"] = len(optimizer.state) == 0
    # Characterize upstream behavior; never silently reset its sample counter.
    checks["native_reset_retains_sample_counter"] = adapt.curr_n_samples == 104
    checks["native_reset_retains_protector_history"] = len(protector.info["z_before"]) == protector_count
    record["native_reset"] = {"sample_count_after": adapt.curr_n_samples,
                              "protector_samples_after": len(protector.info["z_before"]),
                              "scope": "model and optimizer reset only; not a fresh episode"}
    protector.reset()
    checks["explicit_native_protector_reset_clears_history"] = protector.C == 1 and protector.epsilons == [0] and not protector.info["z_before"]
    checks["upstream_files_unchanged"] = all(authenticated_bytes(source / n, h) == payloads[n] for n, h in HASHES.items())
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    record["status"] = "PASS_ENGINEERING_DIAGNOSTIC_ONLY" if all(checks.values()) else "FAIL_ENGINEERING_DIAGNOSTIC"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage", action="store_true", required=True,
                        help="Explicit required method-runtime conformance stage, not default source checks")
    parser.add_argument("--poem-source", required=True, type=Path,
                        help="Declared pinned official POEM checkout; never acquired automatically")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    runtime = os.environ.get("POEM_PYTHON")
    if not runtime:
        parser.error("POEM_PYTHON must explicitly name the prepared method-runtime interpreter")
    if Path(runtime).resolve() != Path(sys.executable).resolve():
        parser.error("POEM_PYTHON does not match the running interpreter")
    if not args.poem_source.is_dir() or not (args.poem_source / ".git").exists():
        parser.error("--poem-source must name an existing pinned Git checkout; no acquisition is performed")
    try:
        authenticated_bytes(args.poem_source / "main.py", MAIN_SHA256)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # Exclusive creation rejects accidental replacement of any existing receipt.
    with args.receipt.open("x") as handle:
        started = time.monotonic()
        record = {"status": "INCOMPLETE", "scope": "modified_harness_synthetic_cpu_engineering_only",
                  "device": "cpu", "model": "resnet50_gn", "pretrained": False,
                  "samples": 104, "source_samples": 16, "batch_size": 4, "spatial_size": 32,
                  "seed": 71, "fixture_generator_seed": 1729, "classifier_bias_fixture": 12.0,
                  "temperature": 0.90, "native_main_executed": False, "cuda_parity_verified": False,
                  "accuracy_measured": False, "cdf_source": "native_ecdf_of_fixture_model_source_entropies",
                  "cpu_threads": 2, "wall_deadline_seconds": 110, "_started": started,
                  "execution_stage": "explicit_poem_native_cpu_conformance_required_for_task3",
                  "declared_runtime": runtime, "running_interpreter": sys.executable,
                  "declared_source": str(args.poem_source.resolve()),
                  "source_only_verification": False,
                  "driver_semantics_binding": {"path": "main.py", "sha256": MAIN_SHA256,
                                               "scope": "authenticated eval/no-grad reference; main not executed"},
                  "local_dependencies": [{"path": "docs/research/kbound/scripts/poem_dependency_bootstrap.py",
                                          "sha256": hashlib.sha256(Path(bootstrap.__file__).read_bytes()).hexdigest(),
                                          "source_snapshot_required": True}],
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        def deadline(signum, frame):
            raise TimeoutError("110-second CPU diagnostic deadline reached; no smaller-model substitute")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(110)
        try:
            diagnostic(record, args.poem_source.resolve())
        except Exception as exc:
            record.update(status="BLOCKED_ENGINEERING_DIAGNOSTIC", error=f"{type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
            record.pop("_started", None)
            record["elapsed_seconds"] = time.monotonic() - started
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": record["status"], "receipt": str(args.receipt)}))
    return 0 if record["status"] == "PASS_ENGINEERING_DIAGNOSTIC_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
