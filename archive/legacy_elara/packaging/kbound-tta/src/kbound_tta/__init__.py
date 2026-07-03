"""kbound_tta — K-Bound / KGA: a label-free certificate for test-time adaptation.

Given a frozen model f0 and a candidate adapted model on an *unlabeled* batch,
kbound_tta decides **adapt / freeze / abstain** with finite-sample control on the
false-adapt and false-freeze rates — the "Knowability-Guided Adaptation" (KGA)
gate from the K-Bound paper.

Quickstart
----------
    import kbound_tta as kb

    # 1) adapt a frozen model with a candidate TTA method (label-free)
    adapted, update_norm = kb.tent_adapt(f0, stream, steps=1, lr=1e-3)
    # candidates: tent_adapt, eata_adapt, sar_adapt, shot_adapt

    # 2) extract label-free evidence and let the gate decide
    Z = kb.evidence_vector(f0, adapted, batch, num_classes, update_norm)
    decision = kb.decide_kga(Z_train, B_train)   # 'adapt' / 'freeze' / 'abstain'

Reference: "When Is Label-Free Adaptation Knowable?" (Niroula).
Repo: https://github.com/Pratikn03/AutoML_Flagship_V8  (docs/research/kbound)
"""
from ._tta import (  # noqa: F401
    pick_device,
    mps_free,
    evidence_vector,
    rich_evidence_vector,
    tent_adapt,
    eata_adapt,
    sar_adapt,
    shot_adapt,
    balanced_acc,
    predict_logits,
    run_candidate,
    eval_frozen,
)
from ._analysis import (  # noqa: F401
    decide_kga,
    policy_metrics,
    label_regime,
    detectability_analysis,
    multicandidate_route,
    smooth_drift_route,
)

#: registry of label-free test-time-adaptation candidates KGA can wrap
TTA_METHODS = {
    "tent": tent_adapt,
    "eata": eata_adapt,
    "sar": sar_adapt,
    "shot": shot_adapt,
}

__version__ = "0.1.0"

__all__ = [
    "pick_device", "mps_free",
    "evidence_vector", "rich_evidence_vector",
    "tent_adapt", "eata_adapt", "sar_adapt", "shot_adapt", "TTA_METHODS",
    "balanced_acc", "predict_logits", "run_candidate", "eval_frozen",
    "decide_kga", "policy_metrics", "label_regime", "detectability_analysis",
    "multicandidate_route", "smooth_drift_route",
    "__version__",
]
