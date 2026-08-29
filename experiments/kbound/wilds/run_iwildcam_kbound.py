"""
run_iwildcam_kbound.py - K-Bound finder scan on WILDS iWildCam.

Purpose: quickly test whether iWildCam has the natural mixed help/harm structure
K-Bound needs: some target locations where TTA helps, some where it hurts, and
label-free evidence that separates those cases.  This is a finder/preview runner,
not a locked paper protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision.models as tvm
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
import run_camelyon17_kbound as rc  # noqa: E402
import run_integrity as ri  # noqa: E402
import tta_methods as tm  # noqa: E402

NUM_CLASSES = 182
BATCH_REGIMES = {"tiny": 8, "small": 16}
AGGR = {"mild": {"steps": 10, "lr": 1e-3}, "aggressive": {"steps": 30, "lr": 2.0e-3}}
DEFAULT_CANDIDATES = ["tent_online", "eata_online", "sar_online"]
RESUME_CONTRACT_SCHEMA = "kbound_iwildcam_resume_contract_v2"
_PRESENT_CACHE: dict[str, set[str]] = {}


def macro_f1(y_true, preds):
    """WILDS-standard iWildCam metric: macro-averaged F1 over classes present."""
    y_true = np.asarray(y_true, int)
    preds = np.asarray(preds, int)
    if y_true.shape != preds.shape:
        raise ValueError(f"y_true/preds shape mismatch: {y_true.shape} != {preds.shape}")
    labels = np.unique(y_true)
    if labels.size == 0:
        raise ValueError("macro-F1 requires at least one target label")
    return float(
        f1_score(
            y_true,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def image_transform(train: bool):
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_iwildcam(root: str, split: str, train_tf: bool = False):
    from wilds import get_dataset

    ds = get_dataset(dataset="iwildcam", download=False, root_dir=root)
    sub = ds.get_subset(split, transform=image_transform(train_tf))
    idx = np.asarray(sub.indices)
    # A partial archive changes the scientific population.  Do not silently
    # filter it: a run is valid only when every member of the requested official
    # split exists, and an unreadable selected image fails its cell below.
    data_dir = Path(ds.data_dir) / "train"
    inp = ds._input_array
    present = present_jpgs(data_dir)
    keep = np.fromiter((str(inp[i]) in present for i in idx), dtype=bool, count=len(idx))
    if not bool(keep.all()):
        missing = [str(inp[i]) for i in idx[~keep][:5]]
        raise RuntimeError(
            f"iWildCam split={split!r} is incomplete: {int((~keep).sum())}/{len(idx)} "
            f"official images are missing (first: {missing}); restore the archive before running"
        )
    y = ds.y_array[idx].numpy().astype(int)
    md = ds.metadata_array[idx].numpy()
    loc_i = ds.metadata_fields.index("location")
    locations = md[:, loc_i].astype(int)
    return ds, sub, y, locations


def present_jpgs(data_dir: Path):
    key = str(data_dir)
    if key in _PRESENT_CACHE:
        return _PRESENT_CACHE[key]
    cache = data_dir.parent / "_present_jpgs_cache.json"
    if cache.exists():
        try:
            with cache.open() as f:
                cached = json.load(f)
            # Older cache files were created before a partial iWildCam extract was
            # repaired and can list images that are no longer present. Trust only
            # v2 caches tied to this exact train directory mtime.
            if (
                cached.get("scan_version") == 2
                and cached.get("data_dir") == key
                and cached.get("data_dir_mtime_ns") == data_dir.stat().st_mtime_ns
            ):
                names = set(cached["names"])
                _PRESENT_CACHE[key] = names
                return names
        except Exception:
            pass
    names = {
        ent.name
        for ent in os.scandir(data_dir)
        if ent.is_file() and ent.name.endswith(".jpg") and not ent.name.startswith("._")
    }
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump({
            "scan_version": 2,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_dir": key,
            "data_dir_mtime_ns": data_dir.stat().st_mtime_ns,
            "count": len(names),
            "names": sorted(names),
        }, f)
    os.replace(tmp, cache)
    _PRESENT_CACHE[key] = names
    print(f"[iwildcam] present image cache: {len(names)} files -> {cache}", flush=True)
    return names


def make_model(backbone: str, device: torch.device):
    if backbone == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT
        model = tvm.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def set_trainable(model: nn.Module, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "head":
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif mode == "layer4_head":
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"unknown trainable mode: {mode}")


def atomic_dump(obj, path: Path):
    ri.atomic_json_dump(obj, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_tensor_sha256(path):
    """Hash model tensor content independently of torch's checkpoint container bytes."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {path}")
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError(f"checkpoint contains no tensors: {path}")
    return digest.hexdigest()


def _json_sha256(value):
    return ri.stable_sha256(value)


def _array_sha256(values):
    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _iwildcam_sample_path(dataset, official_index):
    raw_id = str(dataset._input_array[int(official_index)])
    candidates = [
        Path(raw_id).expanduser(),
        Path(dataset.data_dir) / raw_id,
        Path(dataset.data_dir) / "train" / raw_id,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), raw_id
    raise FileNotFoundError(
        f"iWildCam official sample {official_index} has no readable source path for {raw_id!r}"
    )


def iwildcam_population_manifest(sub, y, locations, split, dataset=None):
    sample_ids = np.asarray(sub.indices, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    locations = np.asarray(locations, dtype=np.int64)
    if not (len(sample_ids) == len(y) == len(locations)):
        raise ValueError("iWildCam population arrays have inconsistent lengths")
    core = {
        "split": str(split),
        "n": int(len(sample_ids)),
        "official_sample_ids_sha256": _array_sha256(sample_ids),
        "labels_sha256": _array_sha256(y),
        "locations_sha256": _array_sha256(locations),
    }
    if dataset is not None:
        content_rows = []
        for official_index, label, location in zip(sample_ids, y, locations):
            path, raw_id = _iwildcam_sample_path(dataset, int(official_index))
            content_rows.append({
                "official_sample_id": int(official_index),
                "official_input_id": raw_id.replace("\\", "/"),
                "label": int(label),
                "location": int(location),
                "bytes": int(path.stat().st_size),
                "content_sha256": file_sha256(path),
            })
        core.update({
            "content_identity_status": "VERIFIED",
            "ordered_content_manifest_sha256": _json_sha256(content_rows),
            "content_identity_fields": [
                "official_sample_id", "official_input_id", "label", "location",
                "bytes", "content_sha256",
            ],
        })
    else:
        core["content_identity_status"] = "NOT_COMPUTED_NO_DATASET_ARGUMENT"
    return {**core, "manifest_sha256": _json_sha256(core)}


def build_resume_contract(args, population_manifest, checkpoint_path, checkpoint_file_hash,
                          checkpoint_tensor_hash, resolved_device=None):
    if population_manifest.get("content_identity_status") != "VERIFIED" or not population_manifest.get(
        "ordered_content_manifest_sha256"
    ):
        raise ValueError("iWildCam resume contract requires verified ordered sample-content identities")
    payload = {
        "dataset": "wilds-iwildcam",
        "implementation_sha256": {
            "runner": file_sha256(__file__),
            "tta_methods": file_sha256(tm.__file__),
            "analysis": file_sha256(an.__file__),
            "routing_aggregates": file_sha256(rc.__file__),
        },
        "role": "target_finder",
        "split": str(args.split),
        "candidate_set_ordered": list(args.candidates),
        "split_manifest": dict(population_manifest),
        "checkpoint": {
            "path": str(Path(checkpoint_path).expanduser().resolve()),
            "file_sha256": str(checkpoint_file_hash),
            "tensor_sha256": str(checkpoint_tensor_hash),
        },
        "seed_semantics": {
            "model_seed": int(args.train_seed),
            "stream_seeds_ordered": [int(seed) for seed in args.seeds],
            "condition_seed_rule": "uint32(sha256('{split}/s{stream_seed}/loc{location}/{composition}/{batch_regime}/{aggressiveness}')[:8]); model/checkpoint invariant",
            "quick_eval_seed_rule": "model_seed + 7 for id_val; model_seed + 13 for target split",
        },
        "scientific_config": {
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "backbone": str(args.backbone),
            "trainable": str(args.trainable),
            "retrain": bool(args.retrain),
            "train_epochs": int(args.train_epochs),
            "max_train_batches": int(args.max_train_batches),
            "train_bs": int(args.train_bs),
            "train_lr": float(args.train_lr),
            "balanced_train": bool(args.balanced_train),
            "workers": int(args.workers),
            "num_classes": int(NUM_CLASSES),
            "max_locations": int(args.max_locations),
            "compositions_ordered": list(args.compositions),
            "batch_regimes_ordered": list(args.batch_regimes),
            "batch_sizes": {name: int(BATCH_REGIMES[name]) for name in args.batch_regimes},
            "aggressiveness_ordered": list(args.aggressiveness),
            "aggressiveness_settings": {
                name: {
                    "steps": int(args.steps_override or AGGR[name]["steps"]),
                    "lr": float(args.adapt_lr if args.adapt_lr is not None else AGGR[name]["lr"]),
                }
                for name in args.aggressiveness
            },
            "n_eval": int(args.n_eval),
            "n_batches": int(args.n_batches),
            "eval_bs": int(args.eval_bs),
            "episodic_steps": int(args.episodic_steps),
            "episodic_batch": int(args.episodic_batch),
            "tau_star": float(args.tau_star),
            "kappa": float(args.kappa),
            "route_min_disagreement": 8,
            "smooth_drift_L": float(args.sd_L),
            "route_c_contract": rc.route_c_contract("macro_f1", NUM_CLASSES),
            "steps_override": int(args.steps_override),
            "adapt_lr": None if args.adapt_lr is None else float(args.adapt_lr),
            "device_requested": str(args.device),
            "device_resolved": str(resolved_device if resolved_device is not None else args.device),
            "torch_version": str(torch.__version__),
        },
    }
    return {
        "schema": RESUME_CONTRACT_SCHEMA,
        "sha256": _json_sha256(payload),
        "payload": payload,
    }


def validate_resume_contract(partial_doc, expected, partial_path):
    actual = partial_doc.get("resume_contract")
    if actual is None:
        raise RuntimeError(
            f"refusing legacy resume without {RESUME_CONTRACT_SCHEMA}: {partial_path}; "
            "use --no-resume or a new --run-name"
        )
    if actual != expected:
        actual_payload = actual.get("payload", {}) if isinstance(actual, dict) else {}
        expected_payload = expected["payload"]
        fields = sorted(
            key for key in set(actual_payload) | set(expected_payload)
            if actual_payload.get(key) != expected_payload.get(key)
        )
        raise RuntimeError(
            f"refusing mismatched resume at {partial_path}; scientific contract fields differ: "
            f"{fields or ['schema_or_hash']}"
        )


def _iwc_cell_key(contract_sha256, split, checkpoint_tensor_hash, train_seed,
                  seed, loc, comp, regime, aggr):
    """Resume identity includes target split and exact checkpoint tensor identity."""
    return (
        str(contract_sha256), str(split), str(checkpoint_tensor_hash), int(train_seed),
        int(seed), int(loc), str(comp), str(regime), str(aggr),
    )


def _iwc_scientific_cell_identity(key):
    """Expand the tuple resume key into the named identity hashed as ``cell_id``."""

    if len(key) != 9:
        raise ri.RunIntegrityError("iWildCam scientific cell key must have nine fields")
    contract_sha, split, tensor_sha, model_seed, stream_seed, location, comp, regime, aggr = key
    return {
        "dataset": "wilds-iwildcam",
        "resume_contract_sha256": str(contract_sha),
        "split": str(split),
        "checkpoint_tensor_sha256": str(tensor_sha),
        "model_seed": int(model_seed),
        "stream_seed": int(stream_seed),
        "location": int(location),
        "composition": str(comp),
        "batch_regime": str(regime),
        "aggressiveness": str(aggr),
    }


def _iwc_cell_id(key):
    return ri.make_cell_id(**_iwc_scientific_cell_identity(key))


def _key_token(key):
    return json.dumps(list(key), separators=(",", ":"), default=str)


def _ledger(expected_keys, done, failures, failure_history):
    expected_set = set(expected_keys)
    completed = sorted(expected_set & set(done), key=_key_token)
    pending = sorted(expected_set - set(done) - set(failures), key=_key_token)
    failed_rows = [failures[key] for key in sorted(failures, key=_key_token)]
    complete = len(completed) == len(expected_set) and not failed_rows
    return {
        "status": "complete" if complete else "incomplete",
        "expected": int(len(expected_set)),
        "completed": int(len(completed)),
        "failed": int(len(failed_rows)),
        "pending": int(len(pending)),
        "expected_keys_sha256": _json_sha256([list(key) for key in sorted(expected_set, key=_key_token)]),
        "completed_keys": [list(key) for key in completed],
        "failed_cells": failed_rows,
        "pending_keys": [list(key) for key in pending],
        "failure_history": list(failure_history),
    }


def _close_float(actual, expected, *, atol=1e-12):
    try:
        return bool(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= atol)
    except (TypeError, ValueError):
        return False


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_exact_sample_provenance(
    provenance,
    partial_path,
    *,
    expected_condition_seed,
    expected_stream_n,
    expected_eval_n,
):
    if not isinstance(provenance, dict):
        raise RuntimeError(f"partial condition is missing deterministic sample provenance: {partial_path}")
    hash_pairs = (
        (
            "ordered_stream_requested_subset_positions_sha256",
            "ordered_stream_resolved_subset_positions_sha256",
        ),
        (
            "ordered_eval_requested_subset_positions_sha256",
            "ordered_eval_resolved_subset_positions_sha256",
        ),
    )
    for requested, resolved in hash_pairs:
        if not _is_sha256(provenance.get(requested)) or provenance.get(requested) != provenance.get(resolved):
            raise RuntimeError(f"partial sample requested/resolved identity mismatch: {partial_path}")
    for field in (
        "ordered_stream_official_sample_ids_sha256",
        "ordered_eval_official_sample_ids_sha256",
    ):
        if not _is_sha256(provenance.get(field)):
            raise RuntimeError(f"partial sample provenance field {field!r} is invalid: {partial_path}")
    required_true = (
        "requested_resolved_identity_equal", "stream_eval_disjoint", "stream_unique", "eval_unique",
    )
    if any(provenance.get(field) is not True for field in required_true):
        raise RuntimeError(f"partial sample identity/disjointness assertion is false: {partial_path}")
    if provenance.get("stream_eval_overlap_count") != 0:
        raise RuntimeError(f"partial sample stream/eval overlap is nonzero: {partial_path}")
    if provenance.get("condition_seed") != int(expected_condition_seed):
        raise RuntimeError(f"partial sample condition seed is inconsistent: {partial_path}")
    if provenance.get("stream_n") != int(expected_stream_n):
        raise RuntimeError(f"partial sample stream size is inconsistent: {partial_path}")
    if provenance.get("eval_n") != int(expected_eval_n):
        raise RuntimeError(f"partial sample evaluation size is inconsistent: {partial_path}")


def _iwc_sample_provenance(official_ids, stream_positions, eval_positions, condition_seed):
    """Build the exact identity receipt shared by execution and resume validation."""

    official_ids = np.asarray(official_ids, dtype=np.int64)
    stream_positions = np.asarray(stream_positions, dtype=int)
    eval_positions = np.asarray(eval_positions, dtype=int)
    overlap = np.intersect1d(stream_positions, eval_positions)
    return {
        "condition_seed": int(condition_seed),
        "sample_id_scheme": "WILDS official dataset row indices after present-file filtering",
        "stream_n": int(len(stream_positions)),
        "eval_n": int(len(eval_positions)),
        "ordered_stream_requested_subset_positions_sha256": _array_sha256(stream_positions),
        "ordered_stream_resolved_subset_positions_sha256": _array_sha256(stream_positions),
        "ordered_eval_requested_subset_positions_sha256": _array_sha256(eval_positions),
        "ordered_eval_resolved_subset_positions_sha256": _array_sha256(eval_positions),
        "requested_resolved_identity_equal": True,
        "stream_eval_disjoint": bool(overlap.size == 0),
        "stream_unique": bool(len(np.unique(stream_positions)) == len(stream_positions)),
        "eval_unique": bool(len(np.unique(eval_positions)) == len(eval_positions)),
        "stream_eval_overlap_count": int(overlap.size),
        "ordered_stream_official_sample_ids_sha256": _array_sha256(
            official_ids[stream_positions]
        ),
        "ordered_eval_official_sample_ids_sha256": _array_sha256(
            official_ids[eval_positions]
        ),
    }


def _expected_iwc_resume_samples(key, expected_contract, sub, y, locations):
    """Recompute a cell's sample receipt and labels from the live WILDS index."""

    if sub is None:
        raise ri.RunIntegrityError(
            "iWildCam resume validation requires the current subset/index"
        )
    official_ids = np.asarray(getattr(sub, "indices", None), dtype=np.int64)
    labels = np.asarray(y, dtype=int)
    live_locations = np.asarray(locations, dtype=int)
    if official_ids.ndim != 1 or not (
        len(official_ids) == len(labels) == len(live_locations)
    ):
        raise ri.RunIntegrityError("iWildCam resume sampling context has inconsistent arrays")
    _contract_sha, split, _tensor_sha, _model_seed, stream_seed, location, comp, regime, aggr = key
    scientific = expected_contract.get("payload", {}).get("scientific_config", {})
    try:
        batch_size = int(scientific["batch_sizes"][regime])
        n_eval = int(scientific["n_eval"])
        n_batches = int(scientific["n_batches"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ri.RunIntegrityError("iWildCam resume contract lacks sampling parameters") from exc
    seed_material = f"{split}/s{stream_seed}/loc{location}/{comp}/{regime}/{aggr}"
    condition_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    stream_positions, eval_positions = _iwc_condition_indices(
        labels,
        live_locations,
        location,
        comp,
        batch_size,
        n_eval,
        n_batches,
        np.random.default_rng(condition_seed),
    )
    location_mask = live_locations == int(location)
    return {
        "sample_provenance": _iwc_sample_provenance(
            official_ids, stream_positions, eval_positions, condition_seed
        ),
        "eval_y": labels[eval_positions],
        "location_n": int(np.sum(location_mask)),
        "location_classes": int(len(np.unique(labels[location_mask]))),
    }


def _iwc_record_inventory(records, conditions):
    """Commit every completed condition and its ordered candidate transaction."""

    by_key = {}
    for record in records:
        key = tuple(record.get("_cell_key", ()))
        by_key.setdefault(key, []).append(record)
    condition_keys = {tuple(condition.get("_key", ())) for condition in conditions}
    orphan_keys = set(by_key) - condition_keys
    if orphan_keys:
        raise ri.RunIntegrityError("iWildCam inventory contains orphan candidate records")
    inventory = {}
    for condition in conditions:
        key = tuple(condition.get("_key", ()))
        cell_id = condition.get("cell_id")
        if not key or not isinstance(cell_id, str):
            raise ri.RunIntegrityError("iWildCam inventory requires cell key and cell_id")
        if cell_id in inventory:
            raise ri.RunIntegrityError("iWildCam inventory contains duplicate cell_id values")
        rows = by_key.get(key, [])
        inventory[cell_id] = {
            "key": list(key),
            "candidate_count": len(rows),
            "candidates": [row.get("candidate") for row in rows],
            "condition_sha256": ri.stable_sha256(condition),
            "records_sha256": ri.stable_sha256(rows),
        }
    return inventory


def _validate_iwc_completed_cell(condition, cell_records, expected_contract,
                                 expected_candidates, partial_path,
                                 sub=None, y=None, locations=None):
    key = tuple(condition["_key"])
    if len(key) != 9:
        raise RuntimeError(f"partial iWildCam cell key has the wrong shape: {partial_path}")
    contract_sha, split, tensor_sha, model_seed, stream_seed, location, comp, regime, aggr = key
    scientific_identity = _iwc_scientific_cell_identity(key)
    cell_id = _iwc_cell_id(key)
    expected_identity = {
        "resume_contract_sha256": contract_sha,
        "checkpoint_tensor_sha256": tensor_sha,
        "split": split,
        "model_seed": model_seed,
        "seed": stream_seed,
        "location": location,
        "domain": f"loc{location}",
        "comp": comp,
        "regime": regime,
        "aggr": aggr,
    }
    if contract_sha != expected_contract.get("sha256"):
        raise RuntimeError(f"partial cell is not bound to the resume contract: {partial_path}")
    checkpoint = expected_contract.get("payload", {}).get("checkpoint", {})
    if tensor_sha != checkpoint.get("tensor_sha256"):
        raise RuntimeError(f"partial cell checkpoint tensor identity mismatch: {partial_path}")
    ri.validate_scientific_cell_identity(
        condition.get("cell_id"),
        condition.get("scientific_cell_identity"),
        context="iWildCam completed condition",
    )
    if condition.get("cell_id") != cell_id or condition.get("scientific_cell_identity") != scientific_identity:
        raise RuntimeError(f"partial cell scientific identity is inconsistent: {partial_path}")
    for field, expected in expected_identity.items():
        if condition.get(field) != expected:
            raise RuntimeError(f"partial condition identity field {field!r} mismatches: {partial_path}")
    if len(cell_records) != len(expected_candidates):
        raise RuntimeError(f"partial completed cell has an incomplete candidate transaction: {partial_path}")
    eval_y = np.asarray(condition.get("eval_y"), dtype=int)
    frozen = np.asarray(condition.get("preds_frozen"), dtype=int)
    if eval_y.ndim != 1 or frozen.shape != eval_y.shape or eval_y.size == 0:
        raise RuntimeError(f"partial condition has invalid evaluation labels/predictions: {partial_path}")
    seed_material = f"{split}/s{stream_seed}/loc{location}/{comp}/{regime}/{aggr}"
    condition_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
    scientific = expected_contract.get("payload", {}).get("scientific_config", {})
    expected_stream_n = int(scientific["batch_sizes"][regime]) * int(scientific["n_batches"])
    provenance = condition.get("sample_provenance")
    _validate_exact_sample_provenance(
        provenance,
        partial_path,
        expected_condition_seed=condition_seed,
        expected_stream_n=expected_stream_n,
        expected_eval_n=eval_y.size,
    )
    if eval_y.size > int(scientific["n_eval"]):
        raise RuntimeError(f"partial condition exceeds the configured evaluation cap: {partial_path}")
    if sub is not None:
        expected_samples = _expected_iwc_resume_samples(
            key, expected_contract, sub, y, locations
        )
        if provenance != expected_samples["sample_provenance"]:
            raise RuntimeError(
                f"partial deterministic sample provenance differs from the current iWildCam "
                f"subset/index: {partial_path}"
            )
        if not np.array_equal(eval_y, expected_samples["eval_y"]):
            raise RuntimeError(
                f"partial evaluation labels differ from the current iWildCam subset/index: "
                f"{partial_path}"
            )
        for field in ("location_n", "location_classes"):
            if condition.get(field) != expected_samples[field]:
                raise RuntimeError(
                    f"partial {field} differs from the current iWildCam subset/index: "
                    f"{partial_path}"
                )
    a0 = macro_f1(eval_y, frozen)
    if not _close_float(condition.get("a0"), a0):
        raise RuntimeError(f"partial condition frozen score is inconsistent: {partial_path}")

    aa_all = [a0]
    predictions = [frozen]
    for candidate, record in zip(expected_candidates, cell_records):
        if tuple(record.get("_cell_key", ())) != key or record.get("candidate") != candidate:
            raise RuntimeError(f"partial candidate transaction order/identity mismatch: {partial_path}")
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise RuntimeError(f"partial candidate identity field {field!r} mismatches: {partial_path}")
        if sub is not None:
            for field in ("location_n", "location_classes"):
                if record.get(field) != expected_samples[field]:
                    raise RuntimeError(
                        f"partial candidate {field} differs from the current iWildCam "
                        f"subset/index: {partial_path}"
                    )
        if record.get("cell_id") != cell_id or record.get("scientific_cell_identity") != scientific_identity:
            raise RuntimeError(f"partial candidate scientific cell identity mismatches: {partial_path}")
        if record.get("sample_provenance") != provenance:
            raise RuntimeError(f"partial candidate sample provenance differs from condition: {partial_path}")
        method, mode = candidate.split("_", 1)
        if record.get("method") != method or record.get("mode") != mode:
            raise RuntimeError(f"partial candidate method/mode identity mismatches: {partial_path}")
        ri.validate_evidence_record(
            record,
            tm.EVIDENCE_NAMES,
            expected_tta_protocol=tm.tta_protocol_contract(mode),
            context=f"iWildCam {cell_id}/{candidate}",
        )
        preds = np.asarray(record.get("preds"), dtype=int)
        if preds.shape != eval_y.shape:
            raise RuntimeError(f"partial candidate prediction length mismatch: {partial_path}")
        aa = macro_f1(eval_y, preds)
        if record.get("metric") != "macro_f1" or not _close_float(record.get("a0"), a0):
            raise RuntimeError(f"partial candidate metric/frozen score mismatch: {partial_path}")
        if not _close_float(record.get("aa"), aa) or not _close_float(record.get("B"), aa - a0):
            raise RuntimeError(f"partial candidate score arithmetic is inconsistent: {partial_path}")
        if record.get("c0") != [int(value) for value in frozen == eval_y]:
            raise RuntimeError(f"partial candidate frozen correctness vector is inconsistent: {partial_path}")
        if record.get("ca") != [int(value) for value in preds == eval_y]:
            raise RuntimeError(f"partial candidate adapted correctness vector is inconsistent: {partial_path}")
        if record.get("regime_label") != an.label_regime(aa - a0):
            raise RuntimeError(f"partial candidate regime label is inconsistent: {partial_path}")
        aa_all.append(aa)
        predictions.append(preds)

    names = ["freeze_f0", *expected_candidates]
    if condition.get("cand_names") != names:
        raise RuntimeError(f"partial condition has a mismatched candidate set: {partial_path}")
    if len(condition.get("aa_all", [])) != len(aa_all) or any(
        not _close_float(actual, expected)
        for actual, expected in zip(condition.get("aa_all", []), aa_all)
    ):
        raise RuntimeError(f"partial condition score vector is inconsistent: {partial_path}")
    oracle = max(aa_all)
    best_adapt = max(aa_all[1:])
    if not _close_float(condition.get("oracle"), oracle):
        raise RuntimeError(f"partial condition oracle is inconsistent: {partial_path}")
    if not _close_float(condition.get("best_adapt"), best_adapt):
        raise RuntimeError(f"partial condition best-adapt score is inconsistent: {partial_path}")
    if condition.get("true_best") != names[int(np.argmax(aa_all))]:
        raise RuntimeError(f"partial condition true-best label is inconsistent: {partial_path}")
    if condition.get("regime_label") != an.label_regime(best_adapt - a0):
        raise RuntimeError(f"partial condition regime label is inconsistent: {partial_path}")

    expected_route = an.multicandidate_route(
        np.stack(predictions, 0),
        tau_star=float(scientific["tau_star"]),
        kappa=float(scientific["kappa"]),
        task_type="multiclass_classification",
        n_classes=NUM_CLASSES,
        objective="macro_f1",
        anchor_above_chance=False,
    )
    if condition.get("route") != expected_route:
        raise RuntimeError(f"partial Route-B truth/status payload is inconsistent: {partial_path}")
    if condition.get("route_b_eligible") is not False or condition.get("realized") is not None:
        raise RuntimeError(f"partial unsupported Route-B was treated as scorable: {partial_path}")
    expected_objective = {
        "metric": "macro_f1", "n_classes": NUM_CLASSES,
        "status": "UNSUPPORTED_BINARY_ACCURACY_IDENTITY",
    }
    if condition.get("route_objective") != expected_objective:
        raise RuntimeError(f"partial Route-B objective contract is inconsistent: {partial_path}")
    rc.validate_unsupported_route_c(condition.get("route_c"), "macro_f1", NUM_CLASSES)


def load_partial_iwc(
    partial_path, expected_contract, expected_keys, candidates,
    *, sub=None, y=None, locations=None,
):
    if not partial_path.exists():
        return [], [], set(), {}, []
    doc = ri.strict_json_load(partial_path)
    if not ri.finite_tree(doc):
        raise ri.RunIntegrityError(f"partial contains NaN/Infinity: {partial_path}")
    if not isinstance(doc, dict):
        raise ri.RunIntegrityError(f"partial must be a JSON object: {partial_path}")
    if doc.get("schema") != "kbound_iwildcam_partial_v3":
        raise ri.RunIntegrityError(
            f"refusing legacy iWildCam partial without bound record inventory: {partial_path}"
        )
    validate_resume_contract(doc, expected_contract, partial_path)
    records = doc.get("records", [])
    conditions = doc.get("conditions", [])
    if not isinstance(records, list) or not isinstance(conditions, list):
        raise ri.RunIntegrityError(f"partial records/conditions must be lists: {partial_path}")
    if conditions and (sub is None or y is None or locations is None):
        raise ri.RunIntegrityError(
            "iWildCam resume validation requires the current subset/index and labels"
        )
    expected_set = set(expected_keys)
    keys = [tuple(condition.get("_key", ())) for condition in conditions]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise RuntimeError(f"partial has missing or duplicate completed cell keys: {partial_path}")
    if not set(keys).issubset(expected_set):
        raise RuntimeError(f"partial contains cells outside the current expected grid: {partial_path}")
    completed_set = set(keys)
    expected_candidates = [f"{method}_{mode}" for method, mode in candidates]
    by_key = {}
    for record in records:
        key = tuple(record.get("_cell_key", ()))
        if key not in completed_set:
            raise RuntimeError(f"partial contains an orphan candidate record: {partial_path}")
        by_key.setdefault(key, []).append(record)
    for key in keys:
        _validate_iwc_completed_cell(
            next(condition for condition in conditions if tuple(condition["_key"]) == key),
            by_key.get(key, []), expected_contract, expected_candidates, partial_path,
            sub=sub, y=y, locations=locations,
        )
    ledger_doc = doc.get("ledger", {})
    failures = {
        tuple(row["key"]): row for row in ledger_doc.get("failed_cells", []) if row.get("key")
    }
    if not set(failures).issubset(expected_set):
        raise RuntimeError(f"partial failure ledger contains cells outside the expected grid: {partial_path}")
    if set(failures) & completed_set:
        raise RuntimeError(f"partial marks the same cell completed and failed: {partial_path}")
    failure_history = list(ledger_doc.get("failure_history", []))
    rebuilt_ledger = _ledger(expected_keys, completed_set, failures, failure_history)
    if ledger_doc != rebuilt_ledger:
        raise RuntimeError(f"partial completion ledger is inconsistent: {partial_path}")
    inventory = _iwc_record_inventory(records, conditions)
    if doc.get("record_inventory") != inventory:
        raise RuntimeError(f"partial record inventory commitment is inconsistent: {partial_path}")
    return records, conditions, completed_set, failures, failure_history


def _partial_payload(contract, expected_keys, records, conditions, done, failures,
                     failure_history, elapsed_sec):
    expected_set = set(expected_keys)
    completed_set = {tuple(condition.get("_key", ())) for condition in conditions}
    if completed_set != set(done):
        raise ri.RunIntegrityError("iWildCam done set does not equal completed condition keys")
    if not completed_set.issubset(expected_set) or not set(failures).issubset(expected_set):
        raise ri.RunIntegrityError("iWildCam partial contains state outside the expected grid")
    if completed_set & set(failures):
        raise ri.RunIntegrityError("iWildCam cell cannot be completed and failed")
    expected_candidates = list(contract.get("payload", {}).get("candidate_set_ordered", []))
    by_key = {}
    for record in records:
        by_key.setdefault(tuple(record.get("_cell_key", ())), []).append(record)
    for condition in conditions:
        key = tuple(condition.get("_key", ()))
        _validate_iwc_completed_cell(
            condition,
            by_key.get(key, []),
            contract,
            expected_candidates,
            "in-memory iWildCam partial",
        )
    return {
        "schema": "kbound_iwildcam_partial_v3",
        "resume_contract": contract,
        "progress": f"{len(done)}/{len(set(expected_keys))}",
        "elapsed_sec": round(float(elapsed_sec), 1),
        "ledger": _ledger(expected_keys, done, failures, failure_history),
        "record_inventory": _iwc_record_inventory(records, conditions),
        "records": records,
        "conditions": conditions,
    }


def _require_route(route, name, require_choice=False):
    if not isinstance(route, dict):
        raise RuntimeError(f"{name} returned non-dict result: {type(route).__name__}")
    decision = str(route.get("decision", "ERROR")).upper()
    if decision == "ERROR":
        raise RuntimeError(f"{name} failed: {route.get('reason', 'missing decision')}")
    if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
        raise RuntimeError(f"{name} returned unknown decision: {route.get('decision')!r}")
    route["decision"] = decision
    if require_choice and decision == "ADAPT" and not isinstance(route.get("choice"), (int, np.integer)):
        raise RuntimeError(f"{name} returned ADAPT without an integer candidate choice")


def _commit_cell(records, conditions, cell_records, condition):
    key = tuple(condition.get("_key", ()))
    if not key or any(tuple(record.get("_cell_key", ())) != key for record in cell_records):
        raise ri.RunIntegrityError("staged iWildCam records do not share the condition key")
    if any(tuple(row.get("_key", ())) == key for row in conditions):
        raise ri.RunIntegrityError(f"duplicate completed iWildCam cell: {list(key)}")
    staged = {"records": cell_records, "condition": condition}
    if not ri.finite_tree(staged):
        raise ValueError("iWildCam cell produced NaN/Infinity and cannot be committed")
    records.extend(cell_records)
    conditions.append(condition)


def train_or_load_f0(args, device: torch.device, out_dir: Path):
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / f"f0_{args.backbone}_seed{args.train_seed}.pt"
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    model = make_model(args.backbone, device)
    if ckpt.exists() and not args.retrain:
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"[f0] loaded {ckpt}", flush=True)
        return model, str(ckpt), {"loaded": True}

    print(f"[f0] training {args.backbone} ({args.trainable}) for iWildCam source preview", flush=True)
    _, train_sub, y_train, _ = get_iwildcam(args.data_root, "train", train_tf=True)
    g = torch.Generator().manual_seed(args.train_seed)
    sampler = None
    shuffle = True
    if args.balanced_train:
        counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
        counts[counts == 0.0] = 1.0
        weights = torch.as_tensor(1.0 / counts[y_train], dtype=torch.double)
        n_samples = int((args.max_train_batches or max(1, len(train_sub) // args.train_bs)) * args.train_bs)
        sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True, generator=g)
        shuffle = False
    loader = DataLoader(
        train_sub,
        batch_size=args.train_bs,
        shuffle=shuffle,
        sampler=sampler,
        generator=g,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )
    set_trainable(model, args.trainable)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.train_lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    steps = 0
    losses = []
    for ep in range(args.train_epochs):
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % max(args.log_every, 1) == 0:
                print(f"  [f0] step={steps} loss={np.mean(losses[-args.log_every:]):.4f}", flush=True)
            if args.max_train_batches and steps >= args.max_train_batches:
                break
        if args.max_train_batches and steps >= args.max_train_batches:
            break
    model.eval()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": model.state_dict(),
        "backbone": args.backbone,
        "trainable": args.trainable,
        "balanced_train": bool(args.balanced_train),
        "steps": steps,
        "mean_loss_tail": float(np.mean(losses[-20:])) if losses else None,
        "wall_sec": round(time.time() - t0, 1),
    }
    torch.save(meta, ckpt)
    print(f"[f0] saved {ckpt} steps={steps} wall={meta['wall_sec']}s", flush=True)
    return model, str(ckpt), {"loaded": False, **{k: v for k, v in meta.items() if k != "model"}}


@torch.no_grad()
def eval_subset(model, sub, device, n: int, seed: int, bs: int):
    rng = np.random.default_rng(seed)
    picks = np.arange(len(sub))
    if len(picks) > n:
        picks = rng.choice(picks, n, replace=False)
    loader = DataLoader(Subset(sub, picks.tolist()), batch_size=bs, shuffle=False, num_workers=0)
    preds, ys = [], []
    model.eval()
    for xb, yb, _ in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys).astype(int)
    official_ids = np.asarray(sub.indices, dtype=np.int64)[np.asarray(picks, dtype=np.int64)]
    return {
        "n": int(len(ys)),
        "acc": float((preds == ys).mean()),
        "balanced_acc": tm.balanced_acc(preds, ys),
        "sampling_seed": int(seed),
        "ordered_subset_positions_sha256": _array_sha256(picks),
        "ordered_official_sample_ids_sha256": _array_sha256(official_ids),
    }


def select_locations(y, locations, max_locations: int, min_count: int):
    rows = []
    for loc in sorted(set(locations.tolist())):
        pos = np.where(locations == loc)[0]
        if len(pos) < min_count:
            continue
        n_classes = len(np.unique(y[pos]))
        rows.append((loc, len(pos), n_classes))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return rows[:max_locations]


def load_positions(sub, positions, device, return_positions=False):
    """Load exactly the requested subset positions or fail the cell.

    No substitution is performed under any circumstance.
    """
    xs = []
    used = []
    for p in positions:
        try:
            x, _, _ = sub[int(p)]
        except Exception as exc:
            raise RuntimeError(
                f"unreadable requested iWildCam subset position {int(p)}; "
                "sample substitution is forbidden"
            ) from exc
        xs.append(x)
        used.append(int(p))
    stacked = torch.stack(xs).to(device)
    if return_positions:
        return stacked, np.asarray(used, dtype=int)
    return stacked


def _iwc_condition_indices(y, locations, loc, comp, bs, n_eval, n_batches, rng):
    """Deterministically select the exact stream/evaluation subset positions."""

    y = np.asarray(y, dtype=int)
    locations = np.asarray(locations, dtype=int)
    pos_all = np.where(locations == int(loc))[0]
    if len(pos_all) < bs * n_batches + 2:
        raise RuntimeError(f"location {loc} too small: n={len(pos_all)}")
    classes = np.unique(y[pos_all])
    per = max(1, n_eval // max(1, len(classes)))
    ev = []
    for c in classes:
        ci = pos_all[y[pos_all] == c]
        if len(ci):
            ev.append(rng.choice(ci, min(per, len(ci)), replace=False))
    ev = np.concatenate(ev) if ev else rng.choice(pos_all, min(n_eval, len(pos_all)), replace=False)
    if len(ev) > n_eval:
        ev = rng.choice(ev, n_eval, replace=False)
    rng.shuffle(ev)
    remain = np.setdiff1d(pos_all, ev)
    if len(remain) == 0:
        raise RuntimeError("evaluation pool leaves no disjoint iWildCam adaptation samples")
    n_stream = bs * n_batches
    if comp == "iid":
        if len(remain) < n_stream:
            raise RuntimeError(f"iid stream needs {n_stream} unique samples; only {len(remain)} remain")
        s = rng.choice(remain, n_stream, replace=False)
    elif comp == "imbalanced":
        counts = Counter(y[remain].tolist())
        maj = counts.most_common(1)[0][0]
        mp = remain[y[remain] == maj]
        op = remain[y[remain] != maj]
        n_maj = int(round(0.85 * n_stream))
        if len(mp) and len(op):
            if len(mp) < n_maj or len(op) < n_stream - n_maj:
                raise RuntimeError(
                    f"imbalanced stream needs {n_maj}/{n_stream - n_maj} unique majority/other "
                    f"samples; only {len(mp)}/{len(op)} remain"
                )
            s = np.concatenate([
                rng.choice(mp, n_maj, replace=False),
                rng.choice(op, n_stream - n_maj, replace=False),
            ])
        else:
            raise RuntimeError("imbalanced stream requires both majority and non-majority samples")
    elif comp == "single_class":
        counts = Counter(y[remain].tolist())
        cls = counts.most_common(1)[0][0]
        pool = remain[y[remain] == cls]
        if len(pool) < n_stream:
            raise RuntimeError(
                f"single-class stream needs {n_stream} unique class-{cls} samples; only {len(pool)} remain"
            )
        s = rng.choice(pool, n_stream, replace=False)
    else:
        raise ValueError(f"unknown composition: {comp}")
    rng.shuffle(s)

    if len(np.unique(s)) != len(s) or len(np.unique(ev)) != len(ev):
        raise RuntimeError("iWildCam condition contains duplicate requested identities")
    if np.intersect1d(s, ev).size:
        raise RuntimeError("iWildCam adaptation and evaluation identities overlap")
    return np.asarray(s, dtype=int), np.asarray(ev, dtype=int)


def build_condition(sub, y, locations, loc, comp, bs, n_eval, n_batches, rng, device,
                    return_ids=False):
    s, ev = _iwc_condition_indices(
        y, locations, loc, comp, bs, n_eval, n_batches, rng
    )
    stream_x, resolved_stream = load_positions(sub, s, device, return_positions=True)
    eval_x, resolved_eval = load_positions(sub, ev, device, return_positions=True)
    if not np.array_equal(s, resolved_stream) or not np.array_equal(ev, resolved_eval):
        raise RuntimeError("iWildCam requested/resolved identity mismatch")
    eval_y = y[ev].astype(int)
    stream = [stream_x[i:i + bs] for i in range(0, len(stream_x), bs)]
    if return_ids:
        return stream, eval_x, eval_y, {
            "stream_requested_subset_positions": np.asarray(s, dtype=int),
            "stream_resolved_subset_positions": resolved_stream,
            "eval_requested_subset_positions": np.asarray(ev, dtype=int),
            "eval_resolved_subset_positions": resolved_eval,
            # Backward-compatible exact aliases for consumers of the v0.5 schema.
            "stream_subset_positions": resolved_stream,
            "eval_subset_positions": resolved_eval,
        }
    return stream, eval_x, eval_y


def parse_candidates(names):
    out = []
    for name in names:
        if "_" not in name:
            raise ValueError(f"candidate must look like method_mode, got {name}")
        method, mode = name.split("_", 1)
        out.append((method, mode))
    return out


def run_scan(args, f0, device, out_dir: Path, f0_ckpt):
    partial = out_dir / "_partial.json"
    dataset, sub, y, locations = get_iwildcam(args.data_root, args.split, train_tf=False)
    population_manifest = iwildcam_population_manifest(
        sub, y, locations, args.split, dataset=dataset
    )
    checkpoint_file_hash = file_sha256(f0_ckpt)
    checkpoint_tensor_hash = checkpoint_tensor_sha256(f0_ckpt)
    contract = build_resume_contract(
        args, population_manifest, f0_ckpt, checkpoint_file_hash,
        checkpoint_tensor_hash, device,
    )
    min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
    loc_rows = select_locations(y, locations, args.max_locations, min_count)
    if not loc_rows:
        raise RuntimeError(f"no target locations in split={args.split} with at least {min_count} samples")
    print("[target locations] " + ", ".join(f"{loc}(n={n},classes={c})" for loc, n, c in loc_rows), flush=True)
    candidates = parse_candidates(args.candidates)
    if not candidates or len(args.candidates) != len(set(args.candidates)):
        raise ValueError("candidate set must be non-empty and contain no duplicates")
    expected_keys = [
        _iwc_cell_key(
            contract["sha256"], args.split, checkpoint_tensor_hash, args.train_seed,
            seed, loc, comp, regime, aggr,
        )
        for seed in args.seeds
        for loc, _loc_n, _loc_classes in loc_rows
        for comp in args.compositions
        for regime in args.batch_regimes
        for aggr in args.aggressiveness
    ]
    if len(expected_keys) != len(set(expected_keys)):
        raise ValueError("duplicate seeds/grid values create duplicate expected iWildCam cells")
    n_cells = len(expected_keys)
    records, conditions, done, failures, failure_history = [], [], set(), {}, []
    if args.resume:
        records, conditions, done, failures, failure_history = load_partial_iwc(
            partial, contract, expected_keys, candidates,
            sub=sub, y=y, locations=locations,
        )
        if done:
            print(f"[resume] {len(done)} cells loaded from {partial}", flush=True)
    official_ids = np.asarray(sub.indices, dtype=np.int64)
    t0 = time.time()
    ci = 0

    def dump_partial():
        atomic_dump(
            _partial_payload(
                contract, expected_keys, records, conditions, done, failures,
                failure_history, time.time() - t0,
            ),
            partial,
        )

    # Establish/replace the contract-bound artifact before any expensive cell work.
    dump_partial()

    for seed in args.seeds:
        for loc, loc_n, loc_classes in loc_rows:
            for comp in args.compositions:
                for regime in args.batch_regimes:
                    bs = BATCH_REGIMES[regime]
                    for aggr in args.aggressiveness:
                        ci += 1
                        tag = (
                            f"{args.split}/ckpt{checkpoint_tensor_hash[:12]}/m{args.train_seed}/"
                            f"s{seed}/loc{loc}/{comp}/{regime}/{aggr}"
                        )
                        key = _iwc_cell_key(
                            contract["sha256"], args.split, checkpoint_tensor_hash,
                            args.train_seed, seed, loc, comp, regime, aggr,
                        )
                        scientific_cell_identity = _iwc_scientific_cell_identity(key)
                        cell_id = _iwc_cell_id(key)
                        if key in done:
                            print(f"  [{ci}/{n_cells}] {tag} SKIP (resume)", flush=True)
                            continue
                        seed_material = f"{args.split}/s{seed}/loc{loc}/{comp}/{regime}/{aggr}"
                        cell_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
                        rng = np.random.default_rng(cell_seed)
                        torch.manual_seed(cell_seed)
                        stage = "sample_condition"
                        try:
                            stream, eval_x, eval_y, sample_ids = build_condition(
                                sub, y, locations, loc, comp, bs, args.n_eval,
                                args.n_batches, rng, device, return_ids=True,
                            )
                            stream_requested = sample_ids["stream_requested_subset_positions"]
                            stream_positions = sample_ids["stream_resolved_subset_positions"]
                            eval_requested = sample_ids["eval_requested_subset_positions"]
                            eval_positions = sample_ids["eval_resolved_subset_positions"]
                            identity_match = (
                                np.array_equal(stream_requested, stream_positions)
                                and np.array_equal(eval_requested, eval_positions)
                            )
                            overlap = np.intersect1d(stream_positions, eval_positions)
                            if not identity_match or overlap.size:
                                raise RuntimeError(
                                    "iWildCam sample-identity invariant failed after exact loading"
                                )
                            sample_provenance = _iwc_sample_provenance(
                                official_ids, stream_positions, eval_positions, cell_seed
                            )
                            steps = args.steps_override or AGGR[aggr]["steps"]
                            lr = args.adapt_lr if getattr(args, "adapt_lr", None) is not None else AGGR[aggr]["lr"]
                            stage = "frozen_evaluation"
                            a0_bacc, p0, _ = tm.eval_frozen(
                                f0, eval_x, eval_y, prob_mode="max", bs=args.eval_bs
                            )
                            a0 = macro_f1(eval_y, p0)            # headline metric = WILDS macro-F1
                            a0_acc = float((np.asarray(p0) == np.asarray(eval_y)).mean())
                            preds_all = [p0]
                            aa_all = [a0]
                            cand_names = ["freeze_f0"]
                            cell_records = []
                            for method, mode in candidates:
                                cand = f"{method}_{mode}"
                                stage = f"candidate:{cand}"
                                aa_bacc, z, upd, preds, _ = tm.run_candidate(
                                    method,
                                    mode,
                                    f0,
                                    stream,
                                    eval_x,
                                    eval_y,
                                    NUM_CLASSES,
                                    steps,
                                    lr,
                                    eval_bs=args.episodic_batch,
                                    prob_mode="max",
                                    episodic_steps=args.episodic_steps,
                                )
                                aa = macro_f1(eval_y, preds)     # headline metric = WILDS macro-F1
                                aa_acc = float((np.asarray(preds) == np.asarray(eval_y)).mean())
                                B = float(aa - a0)
                                cell_records.append({
                                    "_cell_key": list(key),
                                    "cell_id": cell_id,
                                    "scientific_cell_identity": scientific_cell_identity,
                                    "resume_contract_sha256": contract["sha256"],
                                    "checkpoint_tensor_sha256": checkpoint_tensor_hash,
                                    "model_seed": int(args.train_seed),
                                    "seed": int(seed),
                                    "domain": f"loc{loc}",
                                    "location": int(loc),
                                    "location_n": int(loc_n),
                                    "location_classes": int(loc_classes),
                                    "split": args.split,
                                    "comp": comp,
                                    "regime": regime,
                                    "aggr": aggr,
                                    "method": method,
                                    "mode": mode,
                                    "candidate": cand,
                                    "tta_protocol": tm.tta_protocol_contract(mode),
                                    "metric": "macro_f1",
                                    "a0": float(a0),
                                    "aa": float(aa),
                                    "B": B,
                                    "a0_bacc": float(a0_bacc),
                                    "aa_bacc": float(aa_bacc),
                                    "a0_acc": float(a0_acc),
                                    "aa_acc": float(aa_acc),
                                    "upd_norm": float(upd),
                                    "Z": [float(v) for v in z],
                                    "preds": [int(v) for v in preds],
                                    "c0": [int(x) for x in (np.asarray(p0) == np.asarray(eval_y))],
                                    "ca": [int(x) for x in (np.asarray(preds) == np.asarray(eval_y))],
                                    "sample_provenance": sample_provenance,
                                    "regime_label": an.label_regime(B),
                                })
                                preds_all.append(preds)
                                aa_all.append(float(aa))
                                cand_names.append(cand)
                                tm.mps_free()
                                gc.collect()
                            stage = "route_b"
                            route = an.multicandidate_route(
                                np.stack(preds_all, 0),
                                tau_star=args.tau_star,
                                kappa=args.kappa,
                                task_type="multiclass_classification",
                                n_classes=NUM_CLASSES,
                                objective="macro_f1",
                                anchor_above_chance=False,
                            )
                            _require_route(route, "multicandidate route", require_choice=True)
                            if route.get("decision") == "ADAPT" and not (
                                1 <= int(route["choice"]) < len(aa_all)
                            ):
                                raise RuntimeError("multicandidate route choice is out of range")
                            realized = rc.route_realized(route, aa_all)
                            oracle = float(max(aa_all))
                            best_adapt = float(max(aa_all[1:]))
                            stage = "route_c"
                            route_c = rc.unsupported_route_c("macro_f1", NUM_CLASSES)
                            condition = {
                                "_key": list(key),
                                "cell_id": cell_id,
                                "scientific_cell_identity": scientific_cell_identity,
                                "resume_contract_sha256": contract["sha256"],
                                "checkpoint_tensor_sha256": checkpoint_tensor_hash,
                                "model_seed": int(args.train_seed),
                                "seed": int(seed),
                                "domain": f"loc{loc}",
                                "location": int(loc),
                                "location_n": int(loc_n),
                                "location_classes": int(loc_classes),
                                "split": args.split,
                                "comp": comp,
                                "regime": regime,
                                "aggr": aggr,
                                "cand_names": cand_names,
                                "aa_all": [float(v) for v in aa_all],
                                "a0": float(a0),
                                "a0_bacc": float(a0_bacc),
                                "metric": "macro_f1",
                                "oracle": oracle,
                                "best_adapt": best_adapt,
                                "true_best": cand_names[int(np.argmax(aa_all))],
                                "route": route,
                                "route_b_eligible": False,
                                "route_objective": {
                                    "metric": "macro_f1",
                                    "n_classes": NUM_CLASSES,
                                    "status": "UNSUPPORTED_BINARY_ACCURACY_IDENTITY",
                                },
                                "route_c": route_c,
                                "realized": realized,
                                "eval_y": [int(v) for v in eval_y],
                                "preds_frozen": [int(v) for v in p0],
                                "sample_provenance": sample_provenance,
                                "regime_label": an.label_regime(best_adapt - a0),
                            }
                            stage = "commit"
                            _commit_cell(records, conditions, cell_records, condition)
                            done.add(key)
                            failures.pop(key, None)
                            print(
                                f"  [{ci}/{n_cells}] {tag} a0={a0:.3f} best={best_adapt:.3f} "
                                f"oracle={oracle:.3f} route={route.get('decision')} "
                                f"tau={route.get('tau', float('nan')):.3f}",
                                flush=True,
                            )
                        except Exception as exc:
                            failure = {
                                "key": list(key), "tag": tag, "stage": stage,
                                "error": repr(exc)[:1000],
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            }
                            failures[key] = failure
                            failure_history.append(failure)
                            print(
                                f"  [{ci}/{n_cells}] {tag} ERROR[{stage}]: {repr(exc)[:180]}",
                                flush=True,
                            )
                        finally:
                            dump_partial()
                            for name in ("stream", "eval_x", "preds_all"):
                                if name in locals():
                                    del locals()[name]
                            gc.collect()
                            tm.mps_free()
    key_order = {key: index for index, key in enumerate(expected_keys)}
    candidate_order = {name: index for index, name in enumerate(args.candidates)}
    conditions.sort(key=lambda row: key_order[tuple(row["_key"])])
    records.sort(key=lambda row: (
        key_order[tuple(row["_cell_key"])], candidate_order[row["candidate"]]
    ))
    ledger = _ledger(expected_keys, done, failures, failure_history)
    dump_partial()
    meta = {
        "target_locations": loc_rows,
        "wall_sec": time.time() - t0,
        "ledger": ledger,
        "resume_contract": contract,
        "population_manifest": population_manifest,
    }
    if ledger["status"] != "complete":
        raise RuntimeError(
            f"iWildCam run incomplete: completed={ledger['completed']}/{ledger['expected']} "
            f"failed={ledger['failed']} pending={ledger['pending']}; partial ledger: {partial}"
        )
    return records, conditions, meta


def summarize(records, conditions):
    if not records:
        return {"note": "no records"}
    B = np.array([r["B"] for r in records], float)
    det = an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"}
    if float(np.mean(B < 0)) < 0.10 and float(B.mean()) > 0:
        classification = "helpful-dominated"
    elif float(np.mean(B < 0)) > 0.60:
        classification = "harmful-dominated"
    else:
        classification = "mixed+detectable" if det.get("detectability_verdict") == "detectable" else "mixed+undetectable"
    return {
        "classification": classification,
        "n_records": int(len(records)),
        "n_conditions": int(len(conditions)),
        "mean_B": float(B.mean()),
        "base_rate_harmful_B<0": float(np.mean(B < 0)),
        "min_B": float(B.min()),
        "max_B": float(B.max()),
        "detectability_verdict": det.get("detectability_verdict"),
        "best_single_feature_harm_AUC": det.get("best_single_feature_harm_AUC"),
    }


def build_manifest(args, f0_ckpt, train_meta, eval_meta, records, conditions, meta):
    ledger = meta.get("ledger", {})
    if ledger.get("status") != "complete":
        raise RuntimeError("refusing to build an iWildCam result manifest from an incomplete ledger")
    if len(conditions) != ledger["expected"]:
        raise RuntimeError("completed iWildCam condition count does not match the expected ledger")
    cfg = {k: getattr(args, k) for k in vars(args)}
    cfg["ckpt_resolved"] = f0_ckpt
    contract = meta["resume_contract"]
    sha = contract["sha256"][:8]
    routing_a = rc.aggregate_single_candidate(records)
    routing_b = rc.aggregate_multicandidate(conditions)
    routing_c = rc.aggregate_smoothdrift(conditions)
    manifest = {
        "schema": "kbound_wilds_iwildcam_finder_v0.5",
        "dataset": "wilds-iwildcam",
        "metric": "diagnostic_per_cell_sklearn_macro_f1",
        "metric_definition": (
            "sklearn macro-F1 over labels present within each sampled cell, zero_division=0; "
            "this is not the official WILDS split-level metric"
        ),
        "official_wilds_metric": False,
        "execution_complete": True,
        "publication_eligible": False,
        "publication_eligibility_note": (
            "official WILDS macro-F1 was not computed, and the opened target lacks a disjoint "
            "validation-locked confirmation"
        ),
        "claim_eligibility": {
            "raw_completed_records": True,
            "official_metric_score": False,
            "route_a_single_candidate": False,
            "route_b_multicandidate": False,
            "route_c_smooth_drift": False,
            "independent_model_ci": False,
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps": bool(torch.backends.mps.is_available()),
        },
        "config": cfg,
        "config_sha8": sha,
        "resume_contract": contract,
        "completion_ledger": ledger,
        "f0_checkpoint": f0_ckpt,
        "f0_checkpoint_sha256": contract["payload"]["checkpoint"]["file_sha256"],
        "f0_checkpoint_tensor_sha256": contract["payload"]["checkpoint"]["tensor_sha256"],
        "model_seed": int(args.train_seed),
        "f0_training": train_meta,
        "f0_quick_eval": eval_meta,
        "num_classes": NUM_CLASSES,
        "evidence_names": tm.EVIDENCE_NAMES,
        "candidates": args.candidates,
        "data": {
            "data_root": args.data_root,
            "split": args.split,
            "population_manifest": meta["population_manifest"],
            "target_locations": [
                {"location": int(loc), "n": int(n), "classes": int(c)} for loc, n, c in meta["target_locations"]
            ],
            "wall_sec": round(meta["wall_sec"], 1),
        },
        "baselines": {
            "metric": "diagnostic_per_cell_sklearn_macro_f1",
            "always_freeze_mean_acc": float(np.mean([r["a0"] for r in records])) if records else None,
            "per_candidate_always_adapt_mean_acc": {
                c: float(np.mean([r["aa"] for r in records if r["candidate"] == c]))
                for c in sorted(set(r["candidate"] for r in records))
            },
            "per_condition_oracle_mean_acc": float(np.mean([c["oracle"] for c in conditions])) if conditions else None,
            "legacy_key_note": (
                "legacy *_acc fields contain per-cell sklearn macro-F1 diagnostics, not the "
                "official WILDS split-level estimand"
            ),
        },
        "routing_a_single_candidate": routing_a,
        "routing_b_multicandidate": routing_b,
        "routing_c_smooth_drift": routing_c,
        "detectability": an.detectability_analysis(records, tm.EVIDENCE_NAMES) if len(records) >= 4 else {"note": "need>=4"},
        "kbound_summary": summarize(records, conditions),
        "records": records,
        "conditions": conditions,
    }
    if not ri.finite_tree(manifest):
        raise ValueError("iWildCam manifest contains NaN/Infinity and cannot be published")
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Finder scan for K-Bound on WILDS iWildCam")
    p.add_argument("--data-root", default=str(REPO / "experiments/kbound/data/wilds"))
    p.add_argument("--results-root", default=str(REPO / "experiments/kbound/results"))
    p.add_argument("--run-name", default="iwildcam_kbound_finder_v1")
    p.add_argument("--ckpt", default="")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--trainable", choices=["head", "layer4_head", "full"], default="head")
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--train-epochs", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=120)
    p.add_argument("--train-bs", type=int, default=32)
    p.add_argument("--train-lr", type=float, default=1e-3)
    p.add_argument("--balanced-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--split", default="val", choices=["val", "test", "id_val", "id_test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-locations", type=int, default=4)
    p.add_argument("--compositions", nargs="+", default=["iid", "single_class"])
    p.add_argument("--batch-regimes", nargs="+", default=["tiny"], dest="batch_regimes")
    p.add_argument("--aggressiveness", nargs="+", default=["mild"])
    p.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    p.add_argument("--n-eval", type=int, default=48, dest="n_eval")
    p.add_argument("--n-batches", type=int, default=2, dest="n_batches")
    p.add_argument("--eval-bs", type=int, default=64, dest="eval_bs")
    p.add_argument("--episodic-steps", type=int, default=3, dest="episodic_steps")
    p.add_argument("--episodic-batch", type=int, default=32, dest="episodic_batch")
    p.add_argument("--tau-star", type=float, default=0.52, dest="tau_star")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument("--sd-L", type=float, default=0.6, dest="sd_L")
    p.add_argument("--steps-override", type=int, default=0, dest="steps_override")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="skip cells already in _partial.json (default: on)")
    # ---- WIN_HUNT_v5: absolute adapter LR override (enters config hash via vars(args)) ----
    p.add_argument("--adapt-lr", type=float, default=None, dest="adapt_lr",
                   help="WIN_HUNT_v5: absolute adapter LR override for tent/eata/sar (AGGR cell lr "
                        "ignored when set). DEFAULT None = per-cell lr (byte-identical). v5 aggressive "
                        "wave sets 0.004 (= 4x the 1e-3 shared-baseline lr). The 'continual' no-reset "
                        "op-point is selected via --candidates tent_online eata_online sar_online.")
    args = p.parse_args(argv)
    if args.smoke:
        args.run_name = "iwildcam_kbound_smoke"
        args.max_train_batches = min(args.max_train_batches, 8)
        args.train_bs = min(args.train_bs, 16)
        args.max_locations = 2
        args.n_eval = 16
        args.n_batches = 1
        args.compositions = ["iid"]
        args.batch_regimes = ["tiny"]
        args.aggressiveness = ["mild"]
        args.candidates = ["tent_online"]
        args.steps_override = 2
    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.results_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        _, _, y, locations = get_iwildcam(args.data_root, args.split, train_tf=False)
        min_count = args.n_eval + max(BATCH_REGIMES[r] for r in args.batch_regimes) * args.n_batches
        loc_rows = select_locations(y, locations, args.max_locations, min_count)
        print("DRY RUN iWildCam finder")
        print(f"split={args.split} locations={loc_rows}")
        print(f"conditions={len(args.seeds) * len(loc_rows) * len(args.compositions) * len(args.batch_regimes) * len(args.aggressiveness)}")
        print(f"candidate_records={len(args.candidates)} per condition")
        return None
    device = tm.pick_device(args.device)
    print(f"[iwildcam] classes={NUM_CLASSES} split={args.split} device={device}", flush=True)
    f0, f0_ckpt, train_meta = train_or_load_f0(args, device, out_dir)
    # Validate the resume contract before any quick target-label evaluation.
    records, conditions, meta = run_scan(args, f0, device, out_dir, f0_ckpt)
    _, id_sub, _, _ = get_iwildcam(args.data_root, "id_val", train_tf=False)
    _, tgt_sub, _, _ = get_iwildcam(args.data_root, args.split, train_tf=False)
    eval_meta = {
        "id_val": eval_subset(f0, id_sub, device, min(256, args.n_eval * 4), args.train_seed + 7, args.eval_bs),
        args.split: eval_subset(f0, tgt_sub, device, min(256, args.n_eval * 4), args.train_seed + 13, args.eval_bs),
    }
    print(f"[f0 eval] {eval_meta}", flush=True)
    manifest = build_manifest(args, f0_ckpt, train_meta, eval_meta, records, conditions, meta)
    out = Path(args.out) if args.out else out_dir / f"result_{manifest['config_sha8']}.json"
    atomic_dump(manifest, out)
    summary = manifest["kbound_summary"]
    mb = manifest["routing_b_multicandidate"]
    print("\n" + "=" * 72, flush=True)
    print(f"records={len(records)} conditions={len(conditions)} wall={meta['wall_sec']:.1f}s", flush=True)
    print(
        f"classification={summary.get('classification')} harmful={summary.get('base_rate_harmful_B<0'):.3f} "
        f"mean_B={summary.get('mean_B'):+.4f} B_range=[{summary.get('min_B'):+.4f},{summary.get('max_B'):+.4f}]",
        flush=True,
    )
    print(
        f"detectability={summary.get('detectability_verdict')} "
        f"best_harm_AUC={summary.get('best_single_feature_harm_AUC')}",
        flush=True,
    )
    print(f"multicand={mb}", flush=True)
    print(f"manifest -> {out}", flush=True)
    return str(out)


if __name__ == "__main__":
    main()
