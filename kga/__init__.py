"""kga -- Knowability-Guided Adaptation (KGA), the K-Bound decision algorithm.

KGA can decide without deployment target labels after a benefit estimator has
been fitted on labelled development conditions and its residual radius calibrated
on a disjoint split.  Deployment maps label-free evidence ``Z`` to a certificate
``Delta_hat +/- epsilon`` and applies the trichotomy

    ADAPT   if Delta_hat - epsilon > 0
    FREEZE  if Delta_hat + epsilon < 0
    ABSTAIN otherwise

with false-adapt probability bounded by ``alpha`` conditional on the stated
coverage/transfer assumptions (the paper's conditional certificate criterion).

Controlled-grid replay rule
---------------------------
The controlled-grid re-scoring path routes through
:func:`kga.policy.decide_kga`. That function pins the degrees of freedom that
used to vary between copy-pasted historical scripts:

* the radius is the **exact split-conformal rank** quantile
  ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))`` -- never an interpolated
  ``np.quantile``, and ``+inf`` (forced ABSTAIN) when ``k > n``;
* the pool is **leave-one-out-of-pool**: cell ``i``'s radius excludes its own
  labelled residual;
* the trichotomy uses **strict** inequalities, matching the ``|M| > beta``
  commitment convention of the knowability frontier.

Public API
----------
KGA                        -- facade gate (evidence -> frozen estimator -> decide).
BenefitEstimator           -- protocol for a frozen benefit model.
FrozenLinearBenefitEstimator -- auditable reference estimator artifact.
Decision                   -- the ADAPT/FREEZE/ABSTAIN enum.
Certificate                -- a finite-sample certificate ``Delta_hat +/- eps``.
Evidence                   -- the label-free evidence ``Z`` container.
decide_kga                 -- the canonical end-to-end rule (see above).
decide_batch               -- vectorised trichotomy over stored arrays.
split_conformal_rank_radius / conformal_radii_loo / min_calibration_size
                           -- the radius primitives.

The submodules ``kga.certificate`` and ``kga.policy`` remain the stable import
surface for scripts; nothing there has been renamed.

The core decision path is ``numpy``/``scipy`` (no torch) and deterministic.
"""

from __future__ import annotations

from kga._version import __version__
from kga.assumptions import (
    AssumptionReport,
    CoverageClaimBasis,
    CoverageType,
    FallbackAction,
    GateDecision,
    GateThresholds,
    ProtocolRecord,
    Status,
    conformal_radius,
    evidence_support_overlap,
    observed_coverage,
    radius_stability,
    risk_alignment_audit,
    run_gate,
    write_report,
)
from kga.benefit import (
    BenefitEstimator,
    FrozenLinearBenefitEstimator,
    fit_frozen_linear_benefit_estimator,
)
from kga.certificate import (
    Certificate,
    InsufficientCalibrationError,
    conformal_attained_level,
    conformal_radii_loo,
    conformal_split,
    min_calibration_size,
    split_conformal_rank_radius,
)
from kga.evidence import EVIDENCE_FEATURE_NAMES, EVIDENCE_SCHEMA_VERSION, Evidence
from kga.frontier import (
    FrontierAssessment,
    assess_frontier,
    frontier_action,
    frontier_sensitivity,
)
from kga.kga import KGA
from kga.policy import Decision, decide, decide_batch, decide_kga
from kga.routing import (
    AnytimeMulticandidatePanel,
    CandidateCertificate,
    RoutingDecision,
    bonferroni_multicandidate_route,
    multiclass_benefit,
    route_panel,
)

__all__ = [
    "KGA",
    "BenefitEstimator",
    "FrozenLinearBenefitEstimator",
    "fit_frozen_linear_benefit_estimator",
    "Decision",
    "Certificate",
    "Evidence",
    "EVIDENCE_FEATURE_NAMES",
    "EVIDENCE_SCHEMA_VERSION",
    "FrontierAssessment",
    "InsufficientCalibrationError",
    "decide",
    "decide_batch",
    "decide_kga",
    "assess_frontier",
    "frontier_action",
    "frontier_sensitivity",
    "conformal_split",
    "conformal_radii_loo",
    "conformal_attained_level",
    "min_calibration_size",
    "split_conformal_rank_radius",
    "AnytimeMulticandidatePanel",
    "CandidateCertificate",
    "RoutingDecision",
    "bonferroni_multicandidate_route",
    "multiclass_benefit",
    "route_panel",
    "AssumptionReport",
    "CoverageClaimBasis",
    "CoverageType",
    "FallbackAction",
    "GateDecision",
    "GateThresholds",
    "ProtocolRecord",
    "Status",
    "conformal_radius",
    "evidence_support_overlap",
    "observed_coverage",
    "radius_stability",
    "risk_alignment_audit",
    "run_gate",
    "write_report",
    "__version__",
]
