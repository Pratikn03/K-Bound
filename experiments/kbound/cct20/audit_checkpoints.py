#!/usr/bin/env python3
"""Reject missing, relabelled, or duplicate CCT-20 source checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

try:
    from .integrity import (
        IntegrityError,
        atomic_json_dump,
        file_sha256,
        require_sha256,
        stable_sha256,
    )
except ImportError:  # pragma: no cover - direct script execution
    from integrity import IntegrityError, atomic_json_dump, file_sha256, require_sha256, stable_sha256


CANONICAL_MODEL_SEEDS = (0, 1, 2, 3, 4)
CHECKPOINT_SCHEMA = "kbound_cct20_source_checkpoint_v1"


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor names, types, shapes, and bytes independent of serialization."""

    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            raise IntegrityError(f"checkpoint state entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": str(name),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        # PyTorch forbids dtype-changing ``view`` directly on a scalar (for
        # example BatchNorm's integer ``num_batches_tracked``).  Reshape only
        # that zero-dimensional case; the byte sequence remains exact.
        byte_source = tensor.reshape(1) if tensor.ndim == 0 else tensor
        raw = byte_source.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensor_count += 1
    if tensor_count == 0:
        raise IntegrityError("checkpoint state contains no tensors")
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise IntegrityError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"checkpoint must contain a mapping: {path}")
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise IntegrityError(f"checkpoint has the wrong schema: {path}")
    return value


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing CCT-20 checkpoint: {resolved}")
    checkpoint = _load_checkpoint(resolved)
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise IntegrityError(f"checkpoint lacks model_state: {resolved}")
    tensor_hash = tensor_state_sha256(state)
    claimed = require_sha256(
        checkpoint.get("checkpoint_tensor_sha256"),
        field="checkpoint_tensor_sha256",
    )
    if claimed != tensor_hash:
        raise IntegrityError(
            f"checkpoint tensor hash claim is false for {resolved}: {claimed} != {tensor_hash}"
        )
    seed = checkpoint.get("model_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise IntegrityError(f"checkpoint has invalid model_seed: {resolved}")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("model_seed") != seed:
        raise IntegrityError(f"checkpoint config/model_seed mismatch: {resolved}")
    config_hash = require_sha256(checkpoint.get("config_sha256"), field="config_sha256")
    actual_config_hash = stable_sha256(config)
    if config_hash != actual_config_hash:
        raise IntegrityError(
            f"checkpoint config hash claim is false for {resolved}: "
            f"{config_hash} != {actual_config_hash}"
        )
    config_recipe = dict(config)
    config_recipe.pop("model_seed")
    config_recipe_hash = stable_sha256(config_recipe)
    initial_tensor_hash = require_sha256(
        checkpoint.get("initial_tensor_sha256"),
        field="initial_tensor_sha256",
    )
    backbone_tensor_hash = require_sha256(
        checkpoint.get("imagenet_backbone_tensor_sha256"),
        field="imagenet_backbone_tensor_sha256",
    )
    if tensor_hash == initial_tensor_hash:
        raise IntegrityError(f"checkpoint tensors are unchanged from initialization: {resolved}")
    data_hash = require_sha256(checkpoint.get("data_sha256"), field="data_sha256")
    code_hash = require_sha256(checkpoint.get("code_sha256"), field="code_sha256")
    return {
        "model_seed": seed,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "file_sha256": file_sha256(resolved),
        "tensor_sha256": tensor_hash,
        "initial_tensor_sha256": initial_tensor_hash,
        "imagenet_backbone_tensor_sha256": backbone_tensor_hash,
        "config_sha256": config_hash,
        "config_recipe_sha256": config_recipe_hash,
        "data_sha256": data_hash,
        "code_sha256": code_hash,
    }


def _reject_duplicate(rows: Sequence[dict[str, Any]], field: str, description: str) -> None:
    groups: dict[Any, list[int]] = {}
    for row in rows:
        groups.setdefault(row[field], []).append(row["model_seed"])
    duplicates = {value: seeds for value, seeds in groups.items() if len(seeds) > 1}
    if duplicates:
        raise IntegrityError(f"duplicate {description} across model seeds: {duplicates}")


def audit_checkpoint_set(
    checkpoint_template: str,
    seeds: Sequence[int] = CANONICAL_MODEL_SEEDS,
) -> dict[str, Any]:
    normalized = tuple(int(seed) for seed in seeds)
    if normalized != CANONICAL_MODEL_SEEDS:
        raise IntegrityError(
            f"prospective CCT-20 audit requires model seeds {CANONICAL_MODEL_SEEDS}, found {normalized}"
        )
    rows = []
    for expected_seed in normalized:
        path = Path(checkpoint_template.format(seed=expected_seed))
        identity = checkpoint_identity(path)
        if identity["model_seed"] != expected_seed:
            raise IntegrityError(
                f"checkpoint labelled seed {identity['model_seed']} was supplied for seed {expected_seed}: {path}"
            )
        rows.append(identity)
    _reject_duplicate(rows, "file_sha256", "checkpoint file hash")
    _reject_duplicate(rows, "tensor_sha256", "checkpoint tensor hash")
    _reject_duplicate(rows, "initial_tensor_sha256", "initial model tensor hash")
    _reject_duplicate(rows, "config_sha256", "configuration hash")
    if len({row["config_recipe_sha256"] for row in rows}) != 1:
        raise IntegrityError("five model seeds do not share one training recipe")
    if len({row["imagenet_backbone_tensor_sha256"] for row in rows}) != 1:
        raise IntegrityError("five model seeds do not share one ImageNet backbone initialization")
    if len({row["data_sha256"] for row in rows}) != 1:
        raise IntegrityError("five model seeds were not trained on the same data identity")
    if len({row["code_sha256"] for row in rows}) != 1:
        raise IntegrityError("five model seeds were not trained with the same code identity")
    return {
        "schema": "kbound_cct20_independent_checkpoint_audit_v1",
        "status": "PASS",
        "required_model_seeds": list(CANONICAL_MODEL_SEEDS),
        "n_checkpoints": len(rows),
        "all_file_hashes_distinct": True,
        "all_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "all_config_hashes_distinct": True,
        "shared_config_recipe_sha256": rows[0]["config_recipe_sha256"],
        "shared_imagenet_backbone_tensor_sha256": rows[0][
            "imagenet_backbone_tensor_sha256"
        ],
        "shared_data_sha256": rows[0]["data_sha256"],
        "shared_code_sha256": rows[0]["code_sha256"],
        "checkpoints": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-template", required=True, help="path containing {seed}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_checkpoint_set(args.checkpoint_template)
    if args.output.exists():
        raise IntegrityError(f"refusing to overwrite checkpoint audit: {args.output}")
    atomic_json_dump(args.output, result)
    print(f"checkpoint audit: PASS (5 independent seeds) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
