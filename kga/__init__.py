"""kga -- Knowability-Guided Adaptation (KGA), the K-Bound decision algorithm.

KGA decides, **without target labels**, whether to ADAPT, FREEZE, or ABSTAIN on a
test distribution.  It does so from label-free evidence ``Z`` and a finite-sample
certificate ``Delta_hat +/- epsilon`` on the benefit of adapting over freezing,
applying the trichotomy

    ADAPT   if Delta_hat - epsilon > 0
    FREEZE  if Delta_hat + epsilon < 0
    ABSTAIN otherwise

with a false-adapt probability bounded by ``alpha`` (Theorem 3 of the K-Bound
paper, ``docs/research/kbound/K-Bound_paper.pdf``).

The one canonical rule
----------------------
Every driver, re-scoring script and table generator in this repository routes
through :func:`kga.policy.decide_kga` (fix-queue item 15).  That function pins
all three degrees of freedom that used to vary between copy-pasted forks:

* the radius is the **exact split-conformal rank** quantile
  ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))`` -- never an interpolated
  ``np.quantile``, and ``+inf`` (forced ABSTAIN) when ``k > n``;
* the pool is **leave-one-out-of-pool**: cell ``i``'s radius is calibrated on
  the other ``n - 1`` residuals, so ``eps`` is not a function of the label it
  is protecting;
* the trichotomy uses **strict** inequalities, matching the ``|M| > beta``
  commitment convention of the knowability frontier.

Public API
----------
KGA                        -- the facade gate (evidence -> certify -> decide).
Decision                   -- the ADAPT/FREEZE/ABSTAIN enum.
Certificate                -- a finite-sample certificate ``Delta_hat +/- eps``.
Evidence                   -- the label-free evidence ``Z`` container.
decide_kga                 -- the canonical end-to-end rule (see above).
decide_batch               -- vectorised trichotomy over stored arrays.
split_conformal_rank_radius / conformal_radii_loo / min_calibration_size
                           -- the radius primitives.

The submodules ``kga.certificate`` and ``kga.policy`` remain the stable import
surface for scripts; nothing there has been renamed.

The package is pure ``numpy``/``scipy`` (no torch) and deterministic.
"""

from __future__ import annotations

from kga._version import __version__
from kga.assumptions import (
    AssumptionReport,
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
from kga.certificate import (
    Certificate,
    InsufficientCalibrationError,
    conformal_attained_level,
    conformal_radii_loo,
    conformal_split,
    min_calibration_size,
    split_conformal_rank_radius,
)
from kga.evidence import Evidence
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
    "Decision",
    "Certificate",
    "Evidence",
    "InsufficientCalibrationError",
    "decide",
    "decide_batch",
    "decide_kga",
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
