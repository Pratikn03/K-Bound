"""Phase 2.G certification utilities.

Two complementary tools:
  - risk_dominance: estimate (q0, q1, Δ0, Δ1, π*) for a (gate, scenario) pair.
  - switching_certificate: paired-bootstrap LCB on the fired-subset
    paired loss difference; CERTIFIED iff LCB > 0.

Both produce retrospective evaluation certificates under defined
stress protocols. They are NOT production safety certificates and
NOT clinical / physical deployment guarantees.
"""

from .risk_dominance import RiskDominanceTerms, estimate_risk_dominance
from .switching_certificate import (
    SwitchingCertificate,
    paired_bootstrap_lcb,
    fired_subset_certificate,
)

__all__ = [
    "RiskDominanceTerms",
    "estimate_risk_dominance",
    "SwitchingCertificate",
    "paired_bootstrap_lcb",
    "fired_subset_certificate",
]
