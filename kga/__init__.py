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

Public API
----------
KGA           -- the facade gate (evidence -> certify -> decide -> explain).
Decision      -- the ADAPT/FREEZE/ABSTAIN enum.
Certificate   -- a finite-sample certificate ``Delta_hat +/- epsilon``.
Evidence      -- the label-free evidence ``Z`` container.

The package is pure ``numpy``/``scipy`` (no torch) and deterministic.
"""

from __future__ import annotations

from kga._version import __version__
from kga.certificate import Certificate
from kga.evidence import Evidence
from kga.kga import KGA
from kga.policy import Decision
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
    "AnytimeMulticandidatePanel",
    "CandidateCertificate",
    "RoutingDecision",
    "bonferroni_multicandidate_route",
    "multiclass_benefit",
    "route_panel",
    "__version__",
]
