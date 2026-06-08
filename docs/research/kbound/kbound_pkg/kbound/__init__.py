"""kbound: K-Bound certificate for test-time adaptation.

Exposes the public API:

    conformal_radius  -- Thm thm:cert batch conformal radius
    EProcess          -- anytime-valid betting e-process (Ville's inequality)
    BenefitRouter     -- leave-one-out GBR + conformal decisions
    KGA               -- full adapt/freeze/abstain gate (label-free)
    KBoundOptimizer   -- torch gradient-gating optimizer (torch optional)
"""

from kbound.certificate import conformal_radius, empirical_bernstein_lcb, decide
from kbound.evidence import evidence_vector
from kbound.router import BenefitRouter
from kbound.eprocess import EProcess
from kbound.kga import KGA
from kbound.optimizer import KBoundOptimizer

__all__ = [
    "conformal_radius",
    "empirical_bernstein_lcb",
    "decide",
    "evidence_vector",
    "BenefitRouter",
    "EProcess",
    "KGA",
    "KBoundOptimizer",
]
