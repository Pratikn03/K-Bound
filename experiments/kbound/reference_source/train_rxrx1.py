"""Full RxRx1 source replicas from public WILDS v1.2.2, followed separately by KGA.

This implementation does not assert equivalence to the unavailable private
historical source bundle. It never opens test/id_test class labels or trains
on them. The CLI has no reduced-epoch, reduced-batch, or sample-limit options.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import platform
import random
import re
import stat
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile, __version__ as PIL_VERSION
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


REFERENCE_REVISION = "6d96cff360018bdb7c0863ba3976f7fa646aaaab"
INITIALIZATION_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"
COUNTS = {"train": 40612, "id_test": 40612, "val": 9854, "test": 34432}
EPOCHS, BATCH_SIZE, WARMUP_STEPS = 90, 72, 5415


class SourceError(ValueError):
    """A required source-stage identity or execution invariant failed."""


def safe_path(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise SourceError(f"unsafe file path: {path}")
    return path


def read_bytes(path):
    path = safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
                raise SourceError(f"file is not a nonempty regular file: {path}")
            data = f.read()
            after = os.fstat(f.fileno())
            if (before.st_ino, before.st_mtime_ns, before.st_size) != (
                after.st_ino, after.st_mtime_ns, after.st_size
            ) or len(data) != before.st_size:
                raise SourceError(f"file changed during read: {path}")
            return data
    except OSError as exc:
        raise SourceError(f"file unavailable: {path}: {exc}") from exc


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    # Checkpoints can be large; authenticate through a streaming read.
    path = safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise SourceError(f"not a regular file: {path}")
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    except OSError as exc:
        raise SourceError(f"file unavailable: {path}: {exc}") from exc


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    try:
        return json.loads(read_bytes(path))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SourceError(f"invalid JSON file: {path}") from exc


def write_json(path, value, *, replace=False):
    path = safe_path(path)
    if not replace:
        with path.open("xb") as f:
            f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
        return
    temp = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def validate_index(rows, expected):
    seen = set()
    for i, (row, ref) in enumerate(itertools.zip_longest(rows, expected)):
        if not isinstance(row, dict) or not isinstance(ref, dict):
            raise SourceError("population index has missing or extra rows")
        if type(row.get("row_id")) is not int or row["row_id"] != i:
            raise SourceError("population index IDs are missing, duplicate or unordered")
        if any(row.get(k) != ref[k] for k in ("row_id", "path", "split", "group")):
            raise SourceError("population index official split/path membership mismatch")
        if row["path"] in seen:
            raise SourceError("population index duplicate image path")
        seen.add(row["path"])
        if not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise SourceError("population index image hash invalid")
        if any(type(row.get(k)) is not int or row[k] < 1 for k in ("bytes", "width", "height")):
            raise SourceError("population index image dimensions/size invalid")


def source_labels(metadata, rows):
    selected = {r["row_id"] for r in rows if r["split"] in {"train", "val"}}
    labels = {}
    with safe_path(metadata).open(newline="", encoding="utf-8") as f:
        for row_id, row in enumerate(csv.DictReader(f)):
            if row_id not in selected:
                continue  # Never access the class field of test or id_test rows.
            value = row.get("sirna_id")
            if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value) or not 0 <= int(value) < 1139:
                raise SourceError(f"invalid source label at row {row_id}")
            labels[row_id] = int(value)
    if set(labels) != selected:
        raise SourceError("source label IDs missing")
    return labels


def accept_population(directory):
    directory = safe_path(directory)
    receipt_bytes = read_bytes(directory / "completion.json")
    try:
        receipt = json.loads(receipt_bytes)
    except (ValueError, UnicodeError) as exc:
        raise SourceError("invalid population receipt") from exc
    if (not isinstance(receipt, dict) or receipt.get("complete") is not True
            or receipt.get("reference_population_complete") is not True
            or receipt.get("family") != "rxrx1" or receipt.get("counts") != COUNTS
            or receipt.get("expected_counts") != COUNTS
            or receipt.get("index_file") != "image-index.jsonl"):
        raise SourceError("full official RxRx1 population receipt required")
    for field in ("counts", "expected_counts"):
        if any(type(v) is not int for v in receipt[field].values()):
            raise SourceError("population counts must be integers")
    raw_index = read_bytes(directory / "image-index.jsonl")
    if sha(raw_index) != receipt.get("index_sha256"):
        raise SourceError("population index SHA256 mismatch")
    try:
        rows = [json.loads(line) for line in raw_index.splitlines()]
        root = safe_path(receipt["root"])
    except (ValueError, KeyError, TypeError) as exc:
        raise SourceError("population index/root malformed") from exc
    metadata = root / "metadata.csv"
    if file_sha(metadata) != receipt.get("metadata_sha256"):
        raise SourceError("population metadata SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("rx_population", Path(__file__).with_name("population.py"))
    population = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(population)
    try:
        validate_index(rows, population.metadata_entries("rxrx1", root))
    except population.PopulationError as exc:
        raise SourceError(f"population metadata invalid: {exc}") from exc
    if Counter(r["split"] for r in rows) != COUNTS:
        raise SourceError("population index split counts mismatch")
    # Class fields become accessible only after both identities and membership pass.
    labels = source_labels(metadata, rows)
    if file_sha(metadata) != receipt["metadata_sha256"]:
        raise SourceError("population metadata changed during label routing")
    identity = {"receipt_sha256": sha(receipt_bytes), "index_sha256": sha(raw_index),
                "metadata_sha256": receipt["metadata_sha256"], "root": str(root),
                "counts": COUNTS, "population_dir": str(directory)}
    return root, rows, labels, identity


def image_transform(image, *, training):
    if training:
        angle = [0, 90, 180, 270][int(torch.randint(0, 4, (1,)))]
        if angle > 0:
            image = TF.rotate(image, angle)
        if bool(torch.rand(1) < .5):
            image = TF.hflip(image)
    x = TF.to_tensor(image)
    mean, std = x.mean(dim=(1, 2)), x.std(dim=(1, 2), correction=1)
    std[std == 0] = 1
    return TF.normalize(x, mean, std)


class SourceDataset(Dataset):
    def __init__(self, root, rows, labels, *, training):
        self.root, self.rows, self.labels, self.training = Path(root), rows, labels, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        data = read_bytes(self.root / row["path"])
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise SourceError(f"image identity mismatch at row {row['row_id']}")
        old = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                # Explicit campaign boundary for decodable L/RGBA inputs;
                # official RGB RxRx1 images pass through unchanged.
                x = image_transform(image.convert("RGB"), training=self.training)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise SourceError(f"image decode failure at row {row['row_id']}") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old
        return x, self.labels[row["row_id"]]


def authenticated_torch_load(path, expected_sha256):
    """Authenticate and safely load the same real descriptor (supports legacy tar)."""
    path = safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
                raise SourceError("checkpoint must be a nonempty regular file")
            def fingerprint(value):
                return (value.st_dev, value.st_ino, value.st_size,
                        value.st_mtime_ns, value.st_ctime_ns)
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise SourceError("checkpoint SHA256 mismatch")
            if fingerprint(before) != fingerprint(os.fstat(f.fileno())):
                raise SourceError("checkpoint changed during authentication")
            f.seek(0)
            state = torch.load(f, map_location="cpu", weights_only=True)
            if fingerprint(before) != fingerprint(os.fstat(f.fileno())):
                raise SourceError("checkpoint changed during safe loading")
            return state
    except OSError as exc:
        raise SourceError(f"checkpoint unavailable: {path}: {exc}") from exc


def load_initialization(path, expected_sha256):
    if expected_sha256 != INITIALIZATION_SHA256:
        raise SourceError("initialization must have authenticated historical 19c8e357 lineage")
    state = authenticated_torch_load(path, expected_sha256)
    # Public initializer first constructs a 1000-class pretrained model and only
    # then draws a fresh task head. Preserve both RNG consumption and ordering.
    model = torchvision.models.resnet50(weights=None)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise SourceError("initialization is not strict ImageNet1000 ResNet50") from exc
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise SourceError("initialization contains nonfinite tensors")
    model.fc = torch.nn.Linear(model.fc.in_features, 1139)
    model.d_out = 1139
    return model


def lr_factor(step, total_steps, warmup_steps=WARMUP_STEPS):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0., .5 * (1. + math.cos(math.pi * progress)))


def optimizer_scheduler(model, total_steps, warmup_steps=WARMUP_STEPS):
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_steps))
    return optimizer, scheduler


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def run_epoch(model, loader, device, *, optimizer=None, scheduler=None):
    training = optimizer is not None
    if training and scheduler is None:
        raise SourceError("training requires a scheduler")
    model.train(training)  # Includes normal training-mode BN; never freeze it.
    synchronize(device)
    start = time.monotonic()
    loss_sum, correct, examples, batches = 0., 0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise SourceError("nonfinite source loss/logits")
            if training:
                model.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            n = len(y)
            loss_sum += float(loss.detach()) * n
            correct += int((logits.detach().argmax(1) == y).sum())
            examples += n
            batches += 1
    synchronize(device)
    if not examples:
        raise SourceError("empty source epoch")
    return {"ce": loss_sum / examples, "accuracy": correct / examples,
            "examples": examples, "batches": batches,
            "wall_seconds": time.monotonic() - start}


def rng_state(device):
    n = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "mps": torch.mps.get_rng_state() if str(device) == "mps" else None}


def set_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if str(device) == "mps":
        torch.mps.set_rng_state(state["mps"])


def save_training_state(path, model, optimizer, scheduler, identity, history, device):
    synchronize(device)
    state = {"schema": "rxrx1_source_epoch_v1", "identity": identity,
             "algorithm": {"model." + k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "history": history, "rng": rng_state(device)}
    with safe_path(path).open("xb") as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())


def restore_training_state(path, expected_sha256, model, optimizer, scheduler, identity, device):
    data = read_bytes(path)
    if sha(data) != expected_sha256:
        raise SourceError("resume checkpoint identity/hash mismatch")
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if state.get("schema") != "rxrx1_source_epoch_v1" or state.get("identity") != identity:
        raise SourceError("resume configuration/input/runtime identity mismatch")
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["algorithm"].items()}, strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    set_rng(state["rng"], device)
    return state["history"]


def reference_complete(history):
    if len(history) != EPOCHS:
        return False
    return all(r.get("epoch") == i and type(r.get("epoch")) is int
               and all(type(r.get(split, {}).get(k)) is int and r[split][k] == value
                       for split, k, value in [("train", "examples", 40612), ("train", "batches", 565),
                                                ("val", "examples", 9854), ("val", "batches", 137)])
               for i, r in enumerate(history))


def validate_best(out, pointer, history, identity):
    """Do not let a mutable resume pointer change validation model selection."""
    try:
        expected_epoch = max(range(len(history)), key=lambda i: history[i]["val"]["accuracy"])
        if (type(pointer["epoch"]) is not int or pointer["epoch"] != expected_epoch
                or pointer["val_accuracy"] != history[expected_epoch]["val"]["accuracy"]
                or Path(pointer["checkpoint"]).name != pointer["checkpoint"]):
            raise SourceError("resume selection disagrees with completed validation history")
        data = read_bytes(out / pointer["checkpoint"])
        if sha(data) != pointer["sha256"]:
            raise SourceError("resume selection checkpoint hash mismatch")
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if (state.get("schema") != "rxrx1_source_epoch_v1" or state.get("identity") != identity
                or state.get("history") != history[:expected_epoch + 1]):
            raise SourceError("resume selection checkpoint provenance mismatch")
    except (KeyError, IndexError, TypeError) as exc:
        raise SourceError("resume selection pointer malformed") from exc


def prepare_output(path, identity, *, resume):
    path = safe_path(path)
    if resume:
        if not path.is_dir() or read_json(path / "configuration.json") != identity:
            raise SourceError("resume output configuration identity mismatch")
    else:
        try:
            path.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise SourceError("fresh output directory required") from exc
        write_json(path / "configuration.json", identity)
    return path


def runtime_identity(device):
    return {"python": sys.version, "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "pillow": PIL_VERSION, "platform": platform.platform(), "device": device,
            "torch_threads": torch.get_num_threads(), "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")}


def configuration(args, population):
    return {"schema": "rxrx1_public_reference_source_v1", "mode": "preflight" if args.preflight else "train",
            "reference_revision": REFERENCE_REVISION,
            "private_historical_source_equivalence_claimed": False,
            "seed": args.seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "initialization": str(safe_path(args.initialization)), "initialization_sha256": args.initialization_sha256,
            "population": population, "runtime": runtime_identity(args.device), "workers": args.workers,
            "training_code_sha256": file_sha(Path(__file__)),
            "population_code_sha256": file_sha(Path(__file__).with_name("population.py")),
            "optimizer": {"name": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 1e-5},
            "scheduler": {"name": "linear_warmup_half_cosine", "warmup_steps": WARMUP_STEPS, "steps": 50850},
            "normal_train_bn": True, "drop_last": False,
            "preprocessing": "RGB conversion;PIL quarter rotations+horizontal flip for train;ToTensor;per-image/channel sample std;no resize",
            "selection": "strictly greater full official val accuracy;first epoch wins ties",
            "source_validation_ineligible_for_independent_kga_check": True}


def execute(args):
    if args.resume and args.preflight:
        raise SourceError("preflight cannot resume source training")
    if args.workers < 0:
        raise SourceError("workers must be nonnegative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SourceError("requested MPS hardware is unavailable")
    # Reject existing fresh output before label access or model construction.
    if not args.resume and safe_path(args.output_dir).exists():
        raise SourceError("fresh output directory required")
    root, rows, labels, pop_identity = accept_population(args.population_dir)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_initialization(args.initialization, args.initialization_sha256).to(args.device)
    identity = configuration(args, pop_identity)
    out = prepare_output(args.output_dir, identity, resume=args.resume)
    with safe_path(out / "run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SourceError("source output is already owned by another running process") from exc
        if (out / "completion.json").exists():
            raise SourceError("source output already completed; no implicit rerun")
        try:
            loaders = {}
            for split in ("train", "val"):
                data = SourceDataset(root, [r for r in rows if r["split"] == split], labels, training=split == "train")
                loaders[split] = DataLoader(data, batch_size=BATCH_SIZE, shuffle=split == "train",
                                            drop_last=False, num_workers=args.workers, persistent_workers=False)
            if len(loaders["train"]) != 565 or len(loaders["val"]) != 137:
                raise SourceError("reference batch geometry mismatch")
            opt, scheduler = optimizer_scheduler(model, total_steps=EPOCHS * len(loaders["train"]))
            if args.preflight:
                batch = next(iter(loaders["train"]))
                if len(batch[1]) != BATCH_SIZE:
                    raise SourceError("preflight requires the full reference batch72")
                result = run_epoch(model, [batch], args.device, optimizer=opt, scheduler=scheduler)
                write_json(out / "preflight.json", {"mode": "preflight", "complete": False,
                           "source_training_complete": False, "full_batch_exercised": True, "result": result})
                return
            history, best = [], None
            if args.resume:
                journal = read_json(out / "last-epoch.json")
                checkpoint = journal["checkpoint"]
                if Path(checkpoint).name != checkpoint:
                    raise SourceError("resume checkpoint path unsafe")
                history = restore_training_state(out / checkpoint, journal["sha256"], model, opt,
                                                 scheduler, identity, args.device)
                if not history or len(history) > EPOCHS or not reference_complete(
                    history + [{"epoch": i, "train": {"examples": 40612, "batches": 565},
                                "val": {"examples": 9854, "batches": 137}}
                               for i in range(len(history), EPOCHS)]):
                    raise SourceError("resume requires verified completed reference epochs")
                if scheduler.last_epoch != 565 * len(history):
                    raise SourceError("resume scheduler steps disagree with completed epochs")
                best = journal["best"]
                validate_best(out, best, history, identity)
            for epoch in range(len(history), EPOCHS):
                synchronize(args.device); start = time.monotonic()
                train = run_epoch(model, loaders["train"], args.device, optimizer=opt, scheduler=scheduler)
                val = run_epoch(model, loaders["val"], args.device)
                if (train["examples"], train["batches"], val["examples"], val["batches"]) != (40612, 565, 9854, 137):
                    raise SourceError("incomplete epoch; source completion prohibited")
                record = {"epoch": epoch, "train": train, "val": val,
                          "accuracy_gap_train_minus_val": train["accuracy"] - val["accuracy"],
                          "ce_gap_val_minus_train": val["ce"] - train["ce"],
                          "learning_rates": scheduler.get_last_lr(), "scheduler_steps": scheduler.last_epoch,
                          "wall_seconds": time.monotonic() - start}
                history.append(record)
                name = f"epoch-{epoch:03d}-{uuid.uuid4().hex}.pt"
                save_training_state(out / name, model, opt, scheduler, identity, history, args.device)
                digest = file_sha(out / name)
                if best is None or val["accuracy"] > best["val_accuracy"]:
                    best = {"checkpoint": name, "sha256": digest, "epoch": epoch, "val_accuracy": val["accuracy"]}
                journal = {"checkpoint": name, "sha256": digest, "epoch": epoch, "best": best}
                write_json(out / "last-epoch.json", journal, replace=True)
                write_json(out / "learning-curves.json", history, replace=True)
                print(json.dumps(record, sort_keys=True), flush=True)
            if not reference_complete(history):
                raise SourceError("full90-epoch source completion requirements unmet")
            write_json(out / "completion.json", {"complete": True, "stage": "source_training_only",
                       "all_nine_complete": False, "reference_revision": REFERENCE_REVISION,
                       "configuration_sha256": file_sha(out / "configuration.json"),
                       "best": best, "last": journal, "epochs": len(history),
                       "training_examples_visited": sum(r["train"]["examples"] for r in history),
                       "validation_examples_visited": sum(r["val"]["examples"] for r in history),
                       "source_validation_ineligible_for_independent_kga_check": True})
        except (Exception, KeyboardInterrupt) as exc:
            write_json(out / "failure.json", {"complete": False, "stage": "source_training_only",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "resume_policy": "explicit resume of verified last completed epoch only"}, replace=True)
            raise
        finally:
            del model
            if args.device == "mps":
                torch.mps.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--initialization-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=[3, 4], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    try:
        execute(parser.parse_args(argv))
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"complete": False, "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
