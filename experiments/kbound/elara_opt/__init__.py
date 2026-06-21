"""ELARA-Opt — a label-free test-time *parameter-update* optimizer that K-Bound's
KGA can certify as a candidate. Additive to the repo; the frozen kbound_tta / kga
packages and all K-Bound results, manifests, theorems and locks are untouched.

Public API:
    from experiments.kbound.elara_opt import (
        ELARAOptAdapter, elara_opt_adapt, run_elara_candidate,
        ELARA_MODES, make_elara_method, load_meta_gate, ELARA_OPT_DEFAULTS,
    )
"""
from __future__ import annotations

from .config import ELARA_OPT_DEFAULTS
from .objectives import OBJECTIVE_NAMES
from .reliability import RELIABILITY_NAMES, FEATURE_DIM, compute_features
from .gate import MetaGate, compute_weights
from .elara_opt import ELARAOptAdapter, elara_opt_adapt, trust_radius
from .modes import (
    ELARA_MODES, make_elara_method, load_meta_gate, meta_checkpoint_path,
    EXTENDED_TTA_METHODS,
)
from .run_elara_candidate import run_elara_candidate, kga_decide_multicell

__all__ = [
    "ELARA_OPT_DEFAULTS", "OBJECTIVE_NAMES", "RELIABILITY_NAMES", "FEATURE_DIM",
    "compute_features", "MetaGate", "compute_weights",
    "ELARAOptAdapter", "elara_opt_adapt", "trust_radius",
    "ELARA_MODES", "make_elara_method", "load_meta_gate", "meta_checkpoint_path",
    "EXTENDED_TTA_METHODS", "run_elara_candidate", "kga_decide_multicell",
]
__version__ = "0.1.0"
