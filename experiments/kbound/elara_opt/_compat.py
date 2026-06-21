"""_compat.py — locate and re-export the *validated* K-Bound primitives.

ELARA-Opt is strictly additive: it imports the faithful test-time-adaptation
helpers and the KGA certificate machinery from the existing, frozen packages
(`kbound_tta`, `kga`) and NEVER modifies them.  This module does the path
bootstrap (so imports work whether or not PYTHONPATH was set) and re-exports the
exact symbols ELARA-Opt depends on, with a single, documented source of truth.

No K-Bound result, manifest, theorem validator, or lock file is touched here.
"""
from __future__ import annotations

import os
import sys

# ---- path bootstrap ---------------------------------------------------------
# repo root = .../AutoML_Flagship_V8 (four levels up from this file:
# experiments/kbound/elara_opt/_compat.py)
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
_PKG_SRC = os.path.join(_REPO_ROOT, "packaging", "kbound-tta", "src")
for _p in (_PKG_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO_ROOT = _REPO_ROOT

# ---- faithful TTA primitives (kbound_tta._tta) ------------------------------
from kbound_tta._tta import (  # noqa: E402
    pick_device,
    _entropy,
    _clone_for_tta,
    _bn_affine_params,
    _upd_norm,
    evidence_vector,
    EVIDENCE_NAMES,
    rich_evidence_vector,
    bn_running_stats,
    bn_batch_stats,
    bn_stat_kl_drift,
    predict_logits,
    balanced_acc,
    eval_frozen,
    sar_adapt,          # faithful SAR/SAM (referenced, never re-derived/faked)
    tent_adapt,
)

# ---- KGA certificate / policy (the consumer of the candidate) ---------------
from kbound_tta._analysis import (  # noqa: E402
    decide_kga,
    policy_metrics,
    label_regime,
)
from kga import KGA  # noqa: E402
from kga.policy import Decision  # noqa: E402

__all__ = [
    "REPO_ROOT",
    "pick_device",
    "_entropy",
    "_clone_for_tta",
    "_bn_affine_params",
    "_upd_norm",
    "evidence_vector",
    "EVIDENCE_NAMES",
    "rich_evidence_vector",
    "bn_running_stats",
    "bn_batch_stats",
    "bn_stat_kl_drift",
    "predict_logits",
    "balanced_acc",
    "eval_frozen",
    "sar_adapt",
    "tent_adapt",
    "decide_kga",
    "policy_metrics",
    "label_regime",
    "KGA",
    "Decision",
]
