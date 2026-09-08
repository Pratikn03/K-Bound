"""Shared helpers for the kbound_edge pipeline scripts (01..08).

Resolves the edge-package layout, loads YAML configs, fixes seeds, and provides
tiny JSON IO helpers.  Importing this module also puts ``edge/src`` on
``sys.path`` so the scripts can ``import kbound_edge`` without installation.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, cast

HERE = os.path.dirname(os.path.abspath(__file__))      # edge/scripts
EDGE_ROOT = os.path.dirname(HERE)                       # edge
SRC = os.path.join(EDGE_ROOT, "src")
CONFIGS = os.path.join(EDGE_ROOT, "configs")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return cast(dict[str, Any], yaml.safe_load(fh))


def load_config(name_or_path: str) -> dict[str, Any]:
    """Load a config by bare filename (looked up in configs/) or explicit path."""
    if os.path.isabs(name_or_path) or os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(CONFIGS, name_or_path)
    cfg = load_yaml(path)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def resolve(path: str) -> str:
    """Resolve a (possibly relative) artifact path against the edge root."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(EDGE_ROOT, path))


def ensure_parent(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def save_json(path: str, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str, allow_nan=False)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def plan_tuples(plan):
    """Normalise a YAML plan (list of lists) into a list of tuples."""
    return [tuple(item) for item in plan]


def clean_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config without private keys (stable input for a config hash)."""
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


def is_placeholder_kga_meta(meta: dict[str, Any]) -> bool:
    """True when the edge calibrator was bootstrapped before real S03–S06 captures."""
    eps = float(meta.get("eps", 0.0))
    n_fit = int(meta.get("n_fit", 0))
    n_conf = int(meta.get("n_conformal", 0))
    # Placeholder bootstrap from 00_prepare_real_protocol: tiny splits, zero radius.
    return eps == 0.0 and n_fit <= 20 and n_conf <= 20


def load_f0(cfg: dict[str, Any]):
    """Build the MobileNetV3 head and load the trained f0 checkpoint -> (model, version)."""
    import torch
    from kbound_edge.model import build_model, state_dict_hash

    device = cfg.get("device", "cpu")
    model = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=device)
    sd = torch.load(resolve(cfg["paths"]["model"]), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, state_dict_hash(model)


def add_deployment_arguments(parser):
    """Trust inputs must be supplied independently by the operator/deployment seal."""
    parser.add_argument("--expected-frozen-sha256", help="independently trusted full checkpoint SHA256")
    parser.add_argument("--benefit-authority", help="strict authority JSON path, pinned by the external digest")
    parser.add_argument("--expected-authority-sha256", help="independently trusted authority SHA256")
    parser.add_argument("--active-identities", help="trusted active identity JSON, including exact runtime_versions")


def load_trusted_f0(cfg, expected_sha256):
    """Verify immutable checkpoint bytes before restrictive Torch deserialization."""
    import io
    import re
    from collections import OrderedDict

    import torch
    from kbound_edge.benefit_authority import read_verified_bytes
    from kbound_edge.model import build_model

    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("FROZEN_AUTHORITY_INVALID")
    try:
        data = read_verified_bytes(resolve(cfg["paths"]["model"]), expected_sha256, 256 * 1024 * 1024)
        # No fallback: runtimes without weights_only support cannot serve.
        state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if type(state) not in (dict, OrderedDict):
            raise ValueError
        if not state or any(type(k) is not str or type(v) is not torch.Tensor for k, v in state.items()):
            raise ValueError
        if any(not torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError
        device = cfg.get("device", "cpu")
        model = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=device)
        model.load_state_dict(state, strict=True)
        model.eval()
    except Exception:
        raise ValueError("FROZEN_AUTHORITY_INVALID") from None
    return model, expected_sha256


def load_deployment_gate(cfg, args, frozen_sha256, *, estimator_path=None, metadata_path=None):
    """Bind the verified frozen bytes to the operator's active benefit scope."""
    from kbound_edge.benefit_authority import MAX_JSON_BYTES, EdgeBenefitArtifactError, read_verified_bytes
    from kbound_edge.deployment import DeploymentGate

    gate = DeploymentGate()
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError
                result[key] = value
            return result
        active = json.loads(args.active_identities, object_pairs_hook=pairs)
        if active["frozen_model_sha256"] != frozen_sha256:
            raise ValueError
        authority = read_verified_bytes(args.benefit_authority, args.expected_authority_sha256, MAX_JSON_BYTES)
        default_meta = "artifacts_real/calibration/kga_edge_meta.json" if cfg.get("protocol") == "edge_real_phone_v1" else "artifacts_synth/kga_edge_meta.json"
        gate.reload(resolve(estimator_path or cfg["paths"]["kga_edge"]),
                    metadata_path=resolve(metadata_path or cfg["paths"].get("kga_edge_meta", default_meta)),
                    authority=authority, expected_authority_sha256=args.expected_authority_sha256,
                    active_identities=active)
    except (EdgeBenefitArtifactError, ValueError, TypeError, KeyError, RecursionError):
        gate.invalidate("EDGE_AUTHORITY_INVALID")
    return gate
