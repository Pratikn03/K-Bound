"""Serial independent source-only training; no target inputs or pretrained weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import weakref
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms

from .source_data import (
    SourceIdentity,
    decode_rgb,
    file_hash,
    json_bytes,
    load_source_policy,
    load_verified_source,
    strict_json,
    write_new_json,
)


def approved_config(*, source_policy: dict[str, Any] | None = None, development_pilot: bool = False) -> dict[str, Any]:
    config = {
        "schema": "kbound-domainnet-source-training/1",
        "status": "SOURCE_ONLY_DEVELOPMENT_TRAINING",
        "dataset": "DomainNet-126",
        "domain": "clipart",
        "model": {"name": "resnet18", "weights": None, "outputs": 126, "optimize": "all_parameters"},
        "preprocessing": {
            "rgb": True,
            "crop_size": 224,
            "resize_monitor": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "source_fit": ["RandomResizedCrop224", "RandomHorizontalFlip"],
            "source_monitor": ["Resize256", "CenterCrop224"],
        },
        "optimizer": {"name": "SGD", "lr": 0.05, "momentum": 0.9, "weight_decay": 0.0001},
        "scheduler": {"name": "CosineAnnealingLR", "eta_min": 0.0},
        "epochs": 20,
        "physical_batch_size": 32,
        "drop_last": False,
        "num_workers": 0,
        "seeds": [0, 1, 2, 3, 4],
        "seed_stream_rule": "initialization=100+3*seed; minibatch=101+3*seed; augmentation=102+3*seed",
        "candidate": "final_epoch_only",
        "required_runtime": {"python": "3.12", "numpy": "2.4.4"},
    }
    if source_policy is not None:
        config["schema"] = "kbound-domainnet-source-training/2"
        config["source_policy"] = {
            "sha256": source_policy["sha256"],
            "policy_id": source_policy["document"]["policy_id"],
        }
    if development_pilot:
        if source_policy is None:
            raise ValueError("development pilot requires the approved V2 source policy")
        config["schema"] = "kbound-domainnet-source-pilot/1"
        config["status"] = "SOURCE_ONLY_DEVELOPMENT_PILOT"
        config["seeds"] = [0]
    return config


@dataclass
class TestHooks:
    """Explicit synthetic library injection, unavailable through the CLI."""

    identity: SourceIdentity
    model_factory: Callable[[], nn.Module]
    fit_transform: Callable[[Image.Image], torch.Tensor]
    monitor_transform: Callable[[Image.Image], torch.Tensor]
    epochs: int = 2
    batch_size: int = 17


def tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Length-delimited tensor name/dtype/shape plus contiguous CPU raw bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (metadata, raw):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def seed_streams(seed: int) -> dict[str, int]:
    return {"initialization": 100 + 3 * seed, "minibatch": 101 + 3 * seed, "augmentation": 102 + 3 * seed}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SourceDataset(Dataset):
    def __init__(self, archive: zipfile.ZipFile, rows: list[dict[str, Any]], transform: Callable):
        self.archive = archive
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        raw = self.archive.read(row["path"])
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError("source image changed after inventory validation")
        return self.transform(decode_rgb(raw)), row["label"]


def _progress(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".pending")
    with temp.open("xb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _synchronize(backend: str) -> None:
    if backend == "mps":
        torch.mps.synchronize()


def _code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).with_name(name) for name in ("__init__.py", "source_data.py", "train_source.py")]
    result: dict[str, Any] = {"files": {str(path.relative_to(root)): file_hash(path) for path in paths}}
    for field, command in (
        ("git_head", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain=v1", "--untracked-files=normal"]),
    ):
        process = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        result[field] = process.stdout.strip() if process.returncode == 0 else "UNAVAILABLE"
    result["clean_checkout_claim"] = False
    return result


def _runtime(backend: str) -> dict[str, Any]:
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "platform": platform.platform(),
        "backend": backend,
        "num_workers": 0,
        "nondeterminism": "MPS kernels may be nondeterministic; distinct seeded streams do not establish bitwise reproducibility or environment independence"
        if backend == "mps"
        else "CPU seeded single-process execution; reproducibility across runtime/hardware versions is not asserted",
    }
    if backend == "mps":
        result["mps_memory"] = {
            name: getattr(torch.mps, name)()
            for name in ("current_allocated_memory", "driver_allocated_memory")
            if hasattr(torch.mps, name)
        }
    return result


def run_training(
    archive: Path,
    inventory_dir: Path,
    config_path: Path,
    expected_config_sha256: str,
    output_dir: Path,
    *,
    backend: str,
    preflight_batches: int | None = None,
    test_hooks: TestHooks | None = None,
    policy_path: Path | None = None,
    expected_policy_sha256: str | None = None,
    development_pilot: bool = False,
) -> dict[str, Any]:
    if type(development_pilot) is not bool:
        raise ValueError("development pilot mode must be an explicit boolean")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if backend not in {"cpu", "mps"}:
        raise ValueError("explicit cpu or mps backend required")
    if backend == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS backend is unavailable")
    if preflight_batches is not None and (type(preflight_batches) is not int or not 1 <= preflight_batches <= 20):
        raise ValueError("preflight batches must be an integer in 1..20")
    source_policy = load_source_policy(policy_path, expected_policy_sha256)
    if file_hash(config_path) != expected_config_sha256:
        raise ValueError("externally expected training config SHA256 mismatch")
    config = strict_json(config_path)
    if json_bytes(config) != json_bytes(
        approved_config(source_policy=source_policy, development_pilot=development_pilot)
    ):
        raise ValueError("configuration differs from approved source-only recipe")
    if platform.python_version_tuple()[:2] != ("3", "12") or np.__version__ != "2.4.4":
        raise ValueError("source training requires Python 3.12 and NumPy 2.4.4")
    identity = SourceIdentity() if test_hooks is None else test_hooks.identity
    if test_hooks is not None and (
        identity.archive_sha256 == SourceIdentity().archive_sha256 or identity.class_count == 126
    ):
        raise ValueError("test hooks are restricted to synthetic reduced-class source inputs")
    inventory = load_verified_source(
        inventory_dir,
        archive,
        identity=identity,
        policy_path=policy_path,
        expected_policy_sha256=expected_policy_sha256,
    )
    fit_rows = [row for row in inventory["rows"] if row["split"] == "source_fit"]
    monitor_rows = [row for row in inventory["rows"] if row["split"] == "source_monitor"]
    epochs = config["epochs"] if test_hooks is None else test_hooks.epochs
    batch_size = config["physical_batch_size"] if test_hooks is None else test_hooks.batch_size
    if type(epochs) is not int or type(batch_size) is not int or epochs < 1 or batch_size < 1:
        raise ValueError("invalid effective epochs/batch size")
    output_dir.mkdir(exist_ok=False)
    scope = {"execution_scope": "SYNTHETIC_TEST" if test_hooks else config["status"]}
    if development_pilot:
        scope.update(
            development_pilot=True,
            purpose="SOURCE_ONLY_DEVELOPMENT_PILOT",
            eligible_for_confirmatory=False,
        )
    context = {
        "config": config,
        "config_sha256": expected_config_sha256,
        "source_identity": inventory["identity"],
        "inventory_sha256": file_hash(inventory_dir / "inventory.json"),
        "code_identity": _code_identity(),
        "runtime": _runtime(backend),
        **scope,
        "effective_epochs": epochs,
        "effective_batch_size": batch_size,
    }
    if source_policy is not None:
        context["source_policy"] = source_policy
    try:
        _progress(output_dir / "status.json", dict(status="INCOMPLETE", **context))
        normalize = transforms.Normalize(config["preprocessing"]["mean"], config["preprocessing"]["std"])
        fit_transform = (
            transforms.Compose(
                [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            )
            if test_hooks is None
            else test_hooks.fit_transform
        )
        monitor_transform = (
            transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
            if test_hooks is None
            else test_hooks.monitor_transform
        )
        runs: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive) as source_zip:
            fit = SourceDataset(source_zip, fit_rows, fit_transform)
            monitor = SourceDataset(source_zip, monitor_rows, monitor_transform)
            for seed in config["seeds"]:
                streams = seed_streams(seed)
                seed_all(streams["initialization"])
                model = (
                    torchvision.models.resnet18(weights=None, num_classes=126)
                    if test_hooks is None
                    else test_hooks.model_factory()
                )
                if not all(param.requires_grad for param in model.parameters()):
                    raise ValueError("all source model parameters must be optimized")
                initial_hash = tensor_hash(model.state_dict())
                model.to(backend)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
                sampler_generator = torch.Generator().manual_seed(streams["minibatch"])
                loader_generator = torch.Generator().manual_seed(streams["minibatch"])
                loader = DataLoader(
                    fit,
                    batch_size=batch_size,
                    sampler=RandomSampler(fit, generator=sampler_generator),
                    num_workers=0,
                    drop_last=False,
                    generator=loader_generator,
                )
                monitor_loader = DataLoader(
                    monitor,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                    generator=loader_generator,
                )
                seed_all(streams["augmentation"])
                if preflight_batches is not None:
                    seconds = []
                    processed = 0
                    iterator = iter(loader)
                    model.train()
                    for _ in range(preflight_batches):
                        _synchronize(backend)
                        start = time.perf_counter()
                        try:
                            images, labels = next(iterator)
                        except StopIteration:
                            iterator = iter(loader)
                            images, labels = next(iterator)
                        loss = _step(model, optimizer, images, labels, backend)
                        if not math.isfinite(loss):
                            raise ValueError("nonfinite preflight loss")
                        _synchronize(backend)
                        seconds.append(time.perf_counter() - start)
                        processed += len(labels)
                    average = sum(seconds) / len(seconds)
                    result = dict(
                        status="PREFLIGHT_ONLY",
                        eligible_checkpoint=False,
                        steps=preflight_batches,
                        processed_count=processed,
                        mean_step_seconds=average,
                        projected_full_training_step_seconds=average
                        * math.ceil(len(fit) / config["physical_batch_size"])
                        * config["epochs"]
                        * len(config["seeds"]),
                        projection_scope=f"{len(config['seeds'])} seeds x {config['epochs']} epochs, source-fit steps only; excludes monitor, checkpoint, inventory verification and startup; warmup included",
                        seed_streams=streams,
                        **context,
                    )
                    result["runtime"] = _runtime(backend)
                    write_new_json(output_dir / "preflight.json", result)
                    _progress(output_dir / "status.json", result)
                    return result
                seed_dir = output_dir / f"seed-{seed}"
                seed_dir.mkdir()
                processed = 0
                batches = 0
                for epoch in range(1, epochs + 1):
                    model.train()
                    epoch_count = 0
                    epoch_batches = 0
                    weighted_loss = 0.0
                    for images, labels in loader:
                        loss = _step(model, optimizer, images, labels, backend)
                        if not math.isfinite(loss):
                            raise ValueError("nonfinite source-fit loss")
                        epoch_count += len(labels)
                        epoch_batches += 1
                        weighted_loss += loss * len(labels)
                    if epoch_count != len(fit) or epoch_batches != math.ceil(len(fit) / batch_size):
                        raise ValueError("incomplete source-fit epoch")
                    model.eval()
                    correct = 0
                    monitored = 0
                    with torch.no_grad():
                        for images, labels in monitor_loader:
                            logits = model(images.to(backend))
                            if not bool(torch.isfinite(logits).all()):
                                raise ValueError("nonfinite source-monitor logits")
                            predictions = logits.argmax(1).cpu()
                            correct += int((predictions == labels).sum())
                            monitored += len(labels)
                    if monitored != len(monitor):
                        raise ValueError("incomplete source-monitor epoch")
                    scheduler.step()
                    processed += epoch_count
                    batches += epoch_batches
                    epoch_record = {
                        "epoch": epoch,
                        "processed_count": epoch_count,
                        "batches": epoch_batches,
                        "mean_loss": weighted_loss / epoch_count,
                        "source_monitor_count": monitored,
                        "source_monitor_accuracy": correct / monitored,
                        "next_lr": optimizer.param_groups[0]["lr"],
                    }
                    with (seed_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(epoch_record, allow_nan=False, sort_keys=True) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    _progress(
                        output_dir / "status.json",
                        dict(
                            status="INCOMPLETE",
                            current_seed=seed,
                            completed_epoch=epoch,
                            completed_runs=runs,
                            **context,
                        ),
                    )
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                final_hash = tensor_hash(state)
                record = {
                    "status": "COMPLETED",
                    "seed": seed,
                    "seed_streams": streams,
                    "initial_tensor_sha256": initial_hash,
                    "final_tensor_sha256": final_hash,
                    "epochs": epochs,
                    "processed_count": processed,
                    "batches": batches,
                    "checkpoint": f"seed-{seed}/final.pt",
                    "candidate": "final_epoch_only",
                    **scope,
                }
                if source_policy is not None:
                    record["source_policy"] = source_policy
                checkpoint = dict(
                    model=state,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    rng={
                        "python": random.getstate(),
                        "numpy": [
                            value.tolist() if isinstance(value, np.ndarray) else value
                            for value in np.random.get_state()
                        ],
                        "torch": torch.get_rng_state(),
                        "sampler": sampler_generator.get_state(),
                        "loader": loader_generator.get_state(),
                        "mps": torch.mps.get_rng_state() if backend == "mps" else None,
                    },
                    record=record,
                    **context,
                )
                with (seed_dir / "final.pt").open("xb") as stream:
                    torch.save(checkpoint, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                record["checkpoint_sha256"] = file_hash(seed_dir / "final.pt")
                (seed_dir / "final.pt").chmod(0o444)
                write_new_json(seed_dir / "receipt.json", record)
                runs.append(record)
                del model, optimizer, scheduler, checkpoint, state
        if development_pilot:
            if [r["seed"] for r in runs] != [0]:
                raise ValueError("development pilot requires exactly the single declared seed 0")
        elif len({r["initial_tensor_sha256"] for r in runs}) != 5:
            raise ValueError("initial states are not distinct across five source runs")
        result = dict(status="COMPLETED", runs=runs, **context)
        # Completion becomes visible only after the immutable aggregate receipt.
        # If storage fails here, the last published progress remains INCOMPLETE.
        write_new_json(output_dir / "receipt.json", result)
        _progress(output_dir / "status.json", result)
        return result
    except BaseException as exc:
        try:
            _progress(
                output_dir / "status.json",
                dict(status="FAILED", error_type=type(exc).__name__, error=str(exc), **context),
            )
        except Exception as status_error:
            exc.add_note(f"Failure status could not be published: {type(status_error).__name__}: {status_error}")
        raise


def _step(
    model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, labels: torch.Tensor, backend: str
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(images.to(backend))
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite source-fit logits")
    loss = nn.functional.cross_entropy(logits, labels.to(backend))
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite source-fit loss")
    loss.backward()
    if any(param.grad is not None and not bool(torch.isfinite(param.grad).all()) for param in model.parameters()):
        raise ValueError("nonfinite source-fit gradient")
    optimizer.step()
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("nonfinite source model parameter or buffer")
    if any(
        isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
    ):
        raise ValueError("nonfinite optimizer state")
    return float(loss.detach().cpu())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "mps"], required=True)
    parser.add_argument("--preflight-batches", type=int)
    parser.add_argument("--source-policy", type=Path)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--development-pilot", action="store_true")
    args = parser.parse_args(argv)
    run_training(
        args.archive,
        args.inventory_dir,
        args.config,
        args.expected_config_sha256,
        args.output_dir,
        backend=args.backend,
        preflight_batches=args.preflight_batches,
        policy_path=args.source_policy,
        expected_policy_sha256=args.expected_policy_sha256,
        development_pilot=args.development_pilot,
    )


def _cleanup_standalone_torch_temp() -> None:
    """Close only the loaded pinned Torch JIT module's own temporary directory.

    Torch 2.5.1 creates this resource on importing its remote-module instantiator.
    Its implicit finalizer warns at process exit. Library calls must not close it:
    a caller may continue using Torch after source training returns.
    """
    owner = sys.modules.get("torch.distributed.nn.jit.instantiator")
    if owner is None:
        return
    expected_module = Path(torch.__file__).resolve().parent / "distributed/nn/jit/instantiator.py"
    module_file = getattr(owner, "__file__", None)
    temporary = getattr(owner, "_TEMP_DIR", None)
    if (
        not isinstance(module_file, str)
        or Path(module_file).resolve() != expected_module
        or type(temporary) is not tempfile.TemporaryDirectory
    ):
        raise RuntimeError("cannot identify the standalone Torch temporary-directory owner")
    path = Path(temporary.name)
    finalizer_handle = getattr(temporary, "_finalizer", None)
    if type(finalizer_handle) is not weakref.finalize:
        raise RuntimeError("cannot identify the standalone Torch temporary-directory finalizer")
    finalizer = finalizer_handle.peek()
    if finalizer is None and not path.exists():
        return
    if (
        finalizer is None
        or finalizer[0] is not temporary
        or finalizer[1] != getattr(tempfile.TemporaryDirectory, "_cleanup", None)
        or not finalizer[2]
        or finalizer[2][0] != temporary.name
        or getattr(owner, "INSTANTIATED_TEMPLATE_DIR_PATH", None) != temporary.name
        or not path.is_absolute()
        or path.parent.resolve() != Path(tempfile.gettempdir()).resolve()
        or not path.name.startswith("tmp")
        or len(path.name) <= 3
        or path.is_symlink()
        or not path.is_dir()
        or path.stat().st_uid != os.getuid()
    ):
        raise RuntimeError("refusing cleanup of an unverified standalone Torch temporary directory")
    try:
        import psutil

        birth_time = getattr(path.stat(), "st_birthtime", None)
        process_start = psutil.Process().create_time()
    except Exception as exc:
        raise RuntimeError("cannot establish current-process ownership of the Torch temporary directory") from exc
    if (
        not isinstance(birth_time, (int, float))
        or isinstance(birth_time, bool)
        or not isinstance(process_start, (int, float))
        or isinstance(process_start, bool)
        or not math.isfinite(birth_time)
        or not math.isfinite(process_start)
        or birth_time <= process_start
    ):
        # A child after fork inherits the real object, but not ownership. This
        # helper must not remove its parent's directory. Fresh-process CLI only;
        # inherited Python finalizers are outside this scoped cleanup contract.
        raise RuntimeError("Torch temporary directory is not proven to belong to the current process")
    temporary.cleanup()


def _standalone_main() -> None:
    try:
        main()
    finally:
        original_error = sys.exc_info()[1]
        try:
            _cleanup_standalone_torch_temp()
        except Exception as cleanup_error:
            if original_error is None or (isinstance(original_error, SystemExit) and original_error.code in (None, 0)):
                raise
            original_error.add_note(f"Standalone Torch temporary-directory cleanup failed: {cleanup_error}")


if __name__ == "__main__":
    _standalone_main()
