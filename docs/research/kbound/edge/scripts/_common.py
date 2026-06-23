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
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))      # edge/scripts
EDGE_ROOT = os.path.dirname(HERE)                       # edge
SRC = os.path.join(EDGE_ROOT, "src")
CONFIGS = os.path.join(EDGE_ROOT, "configs")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(name_or_path: str) -> Dict[str, Any]:
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
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def plan_tuples(plan):
    """Normalise a YAML plan (list of lists) into a list of tuples."""
    return [tuple(item) for item in plan]


def clean_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Config without private keys (stable input for a config hash)."""
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}


def load_f0(cfg: Dict[str, Any]):
    """Build the MobileNetV3 head and load the trained f0 checkpoint -> (model, version)."""
    import torch

    from kbound_edge.model import build_model, state_dict_hash

    device = cfg.get("device", "cpu")
    model = build_model(cfg["num_classes"], pretrained=False, seed=cfg["seed"], device=device)
    sd = torch.load(resolve(cfg["paths"]["model"]), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, state_dict_hash(model)
