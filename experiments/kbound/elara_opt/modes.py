"""modes.py — the three ELARA-Opt modes and how they slot behind the existing
adapter registry, plus the deterministic meta-gate loader.

EXTENDED_TTA_METHODS shows the *one-line* registration that the locked-run runner
would use; the frozen kbound_tta package is NOT mutated here.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from .config import ELARA_OPT_DEFAULTS
from .gate import MetaGate
from .reliability import FEATURE_DIM
from .elara_opt import ELARAOptAdapter, elara_opt_adapt  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))

ELARA_MODES = list(ELARA_OPT_DEFAULTS["modes"])  # elara_uniform / elara_rule / elara_meta


def meta_checkpoint_path(cfg: Optional[Dict] = None) -> str:
    cfg = cfg or ELARA_OPT_DEFAULTS
    return os.path.join(_HERE, cfg["meta"]["checkpoint"])


def load_meta_gate(cfg: Optional[Dict] = None) -> Optional[MetaGate]:
    """Load the dev-trained meta gate if its checkpoint exists; else None."""
    cfg = cfg or ELARA_OPT_DEFAULTS
    path = meta_checkpoint_path(cfg)
    if not os.path.exists(path):
        return None
    model = MetaGate(in_dim=FEATURE_DIM, hidden=int(cfg["meta"]["hidden"]))
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()
    return model


def make_elara_method(mode: str, num_classes: int, *, cfg: Optional[Dict] = None,
                      meta_model: Optional[MetaGate] = None, seed: int = 0):
    """Return a closure with the bare (base, stream, steps, lr) adapter signature."""
    cfg = cfg or ELARA_OPT_DEFAULTS
    if mode == "elara_meta" and meta_model is None:
        meta_model = load_meta_gate(cfg)
    adapter = ELARAOptAdapter(mode=mode, cfg=cfg, meta_model=meta_model, seed=seed)
    return adapter.as_method(num_classes)


#: how ELARA-Opt would extend the frozen registry (documentation, not a mutation).
EXTENDED_TTA_METHODS = {
    "elara_uniform": lambda nc, mm=None, seed=0: make_elara_method("elara_uniform", nc, meta_model=mm, seed=seed),
    "elara_rule": lambda nc, mm=None, seed=0: make_elara_method("elara_rule", nc, meta_model=mm, seed=seed),
    "elara_meta": lambda nc, mm=None, seed=0: make_elara_method("elara_meta", nc, meta_model=mm, seed=seed),
}
