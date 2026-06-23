"""kbound_edge -- Safe test-time adaptation for camera-based inspection (Tiers 0-2).

This is an ADDITIVE edge/runtime layer that sits ON TOP of the published K-Bound
certificate (the ``kbound`` reproduction package in ``../kbound_pkg``).  It does
NOT fork or re-derive the certificate -- the adapt/freeze/abstain decision rule
(:func:`kbound.certificate.decide`), the label-free evidence vector
(:func:`kbound.evidence.evidence_vector`) and the benefit router
(:class:`kbound.router.BenefitRouter`) are imported and reused verbatim via
:mod:`kbound_edge._bridge`.

What this package adds (and nothing in the paper is modified):

    capture            OpenCV VideoCapture reader + synthetic/fake frame sources
    dataset            frame -> tensor preprocessing, windowing, splits
    model              MobileNetV3-Small (torchvision) + 4-class inspection head
    tent_adapter       EpisodicTentAdapter (deepcopy f0, BN-affine only, 1 Adam step)
    evidence           ~14 label-free features (11 paper features + 3 edge features)
    benefit_estimator  HistGradientBoostingRegressor benefit model
    conformal          split-conformal residual radius (conservative rank, alpha=0.10)
    policy             kga_decide -> adapt/freeze/abstain + lower/upper bound + reason
    metrics            regret, false-adapt (uncond/cond), adapt/abstain rate, latency
    logging            JSONL window logger (NO target labels on the online path)
    replay             offline recorded-stream runner
    shadow_runtime     live window-by-window ShadowController (frozen = official output)
    dashboard          minimal live text view

Tiers
-----
Tier 0  Offline replay of a recorded clip -- the full decision chain, no camera.
Tier 1  Same chain wired to a calibrated benefit estimator + conformal radius.
Tier 2  Live shadow mode: frozen model is the official output, candidate runs in
        shadow and is only logged -- never emitted.

IMPORTANT
---------
Everything ships validated on SYNTHETIC data only.  Synthetic data proves the
CODE runs end-to-end; it is NOT an empirical result.  Real numbers require real
recorded clips (see ``README.md``).
"""

from __future__ import annotations

__version__ = "0.1.0"
SCHEMA_VERSION = "kbound-edge-v1"

# Re-export the reused certificate surface so callers can do
#   from kbound_edge import decide, conformal_radius, evidence_vector
from kbound_edge._bridge import (  # noqa: E402,F401
    decide,
    conformal_radius,
    evidence_vector,
    BenefitRouter,
    PAPER_EVIDENCE_NAMES,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "decide",
    "conformal_radius",
    "evidence_vector",
    "BenefitRouter",
    "PAPER_EVIDENCE_NAMES",
]
