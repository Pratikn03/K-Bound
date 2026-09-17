"""
aetta_dropout.py - FAITHFUL dropout-AETTA adapt/recover decision for WIN_HUNT_v4
arm_D (closes the "protocol-matched ports" caveat for AETTA).

AETTA  =  Accuracy Estimation for Test-Time Adaptation
          Lee, Chottananurak, Gong, Lee, "AETTA: Label-Free Accuracy Estimation for
          Test-Time Adaptation", CVPR 2024, arXiv:2404.01351.
          Official code: github.com/taeckyung/AETTA (learner/dnn.py).

WHERE THE REAL AETTA COMPUTE LIVES.  AETTA's accuracy estimate needs MC-dropout model
forwards, which cannot be reconstructed from summary logs. So the faithful estimator is
implemented as a RUNNER-SIDE hook (cifar_tent_mps_v2.py::_aetta_accuracy_estimate, run
under --log-samples): N=10 head-MC-dropout passes at the paper's per-dataset rate
(CIFAR-10 0.4, CIFAR-100 0.3, ImageNet 0.2), the paper's IMPROVED estimate
    err_hat = clip( (1 - agreement) / (E_avg/E_max)^alpha , 0, 1 - 1/K ),  alpha = 3,
EMA-accumulated over the eval stream (official 0.6/0.4), reported as accuracy 1 - err_hat.
That per-condition estimate is stored in each samples_*.npz as:
    aetta_acc_est          (adapted model)   and   aetta_acc_est_frozen  (source model).

THIS module is the label-free DECISION rule that consumes those two fields and turns them
into a per-condition ADAPT/FREEZE compatible with the head-to-head scorer, following
AETTA's own model-recovery logic (learner/dnn.py::reset_function, reset_function=="aetta").

--------------------------------------------------------------------------------------
PINNED CHOICES (the paper leaves the per-condition binary rule open; frozen before scoring)
--------------------------------------------------------------------------------------
  (D1) AETTA's recovery has two triggers in the official code:
         (a) hard reset when the estimated accuracy drops below an ABSOLUTE floor
             (`if acc_preds[-1] < 20: hard_reset()`, on a 0-100 scale), and
         (b) hard reset when the recent-5 estimate mean falls >2 below the prior-5 mean
             (a downward TREND of the estimate).
       In our per-condition (non-streaming) setting there is no long trajectory, so we
       map (a) verbatim to an absolute floor and (b) to the natural per-condition analog:
       compare the ADAPTED model's estimated accuracy to the SOURCE model's estimated
       accuracy (both label-free via the same AETTA estimator) -- a predicted DROP vs
       source is exactly "the estimate went down because we adapted", the trend (b) at
       condition granularity.
  (D2) DECISION:
         ADAPT   iff  aetta_acc_est >= aetta_acc_est_frozen - margin   (no predicted
                      degradation vs source; trend-recovery analog, D1b)
                 AND  100 * aetta_acc_est >= floor                     (absolute floor, D1a)
         FREEZE  otherwise  ("recover" -> keep/revert to the source model)
       floor = 20.0 (paper's `< 20` reset threshold, 0-100 scale); margin = 0.0 (strict).
       Both are fixed before scoring and exposed only for the record. AETTA never abstains.

Pure numpy. No torch. Consumes only the aetta_acc_est* scalars logged in the .npz.
"""
from __future__ import annotations
import numpy as np

DEFAULT_FLOOR = 20.0    # (D1a) official AETTA hard-reset floor, on the 0-100 accuracy scale
DEFAULT_MARGIN = 0.0    # (D1b) strict no-degradation-vs-source gate


def aetta_dropout_decision(records, floor=DEFAULT_FLOOR, margin=DEFAULT_MARGIN,
                           return_detail=False):
    """Per-condition ADAPT/FREEZE from the logged AETTA MC-dropout estimates (D1,D2).

    Args:
      records: list of per-condition dicts each carrying scalars
               'aetta_acc_est' (adapted model, fraction in [0,1]) and
               'aetta_acc_est_frozen' (source model, fraction in [0,1]).
      floor:  absolute recovery floor on the 0-100 accuracy scale (paper's 20).
      margin: allowed degradation vs source before recovering (0 = strict).
    Returns: np.array(dtype=object) of {"ADAPT","FREEZE"}; optional per-condition detail.
    """
    dec, detail = [], []
    for r in records:
        est_ad = float(r["aetta_acc_est"])
        est_fr = float(r.get("aetta_acc_est_frozen", np.nan))
        no_degrade = (np.isnan(est_fr)) or (est_ad >= est_fr - margin)   # D1b (if no source est, don't block on it)
        above_floor = (100.0 * est_ad) >= floor                          # D1a
        adapt = bool(no_degrade and above_floor)
        dec.append("ADAPT" if adapt else "FREEZE")
        detail.append({"condition": r.get("condition"),
                       "aetta_acc_est": est_ad, "aetta_acc_est_frozen": est_fr,
                       "no_degrade_vs_source": bool(no_degrade),
                       "above_floor": bool(above_floor), "decision": dec[-1]})
    out = np.array(dec, dtype=object)
    if return_detail:
        return out, {"floor": floor, "margin": margin, "per_condition": detail}
    return out
