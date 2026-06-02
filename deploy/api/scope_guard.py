"""Scope guard — live out-of-envelope / drift monitoring for the serving path.

Gate P P12/P13: the deployment must (a) declare a validated operating envelope
and (b) detect at inference time when an input falls outside it. This is the
PRODUCTION embodiment of the RGA research idea: down-weight / flag a modality
whose live score distribution drifts from the validated reference.

Design (honest, no overclaim):
  - A reference envelope (per-domain validated score quantiles) is loaded from
    UAIS_SCOPE_REFERENCE (JSON). If absent, the guard runs in ADVISORY mode and
    uses only a reference-free novelty signal (cross-modal disagreement), clearly
    flagged as advisory.
  - evaluate(domain_scores) returns {in_envelope, drift, reasons} per request.
  - A Prometheus gauge tracks the live drift so out-of-envelope rates are
    observable/alertable in production.

The guard does NOT block requests by default; it ANNOTATES responses and emits
metrics, so operators can route, alert, or hold out-of-envelope traffic. The
validated envelope itself is declared in deploy/SCOPE_CONTRACT.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    from prometheus_client import Gauge

    from .monitoring import _existing_collector

    _DRIFT_GAUGE = _existing_collector("uais_scope_drift") or Gauge(
        "uais_scope_drift", "Live out-of-envelope drift score (0=in-envelope)", ["endpoint"]
    )
    _OOE_GAUGE = _existing_collector("uais_out_of_envelope_total") or Gauge(
        "uais_out_of_envelope_total", "Count of out-of-envelope requests", ["endpoint"]
    )
except Exception:  # pragma: no cover - metrics optional
    _DRIFT_GAUGE = None
    _OOE_GAUGE = None


def _load_reference() -> dict | None:
    path = os.getenv("UAIS_SCOPE_REFERENCE")
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


_REFERENCE = _load_reference()
_DRIFT_THRESHOLD = float(os.getenv("UAIS_SCOPE_DRIFT_THRESHOLD", "0.5"))


def reference_loaded() -> bool:
    return _REFERENCE is not None


def evaluate(domain_scores: dict[str, float], *, endpoint: str = "fusion") -> dict:
    """Score an inference request against the validated envelope.

    Returns {in_envelope: bool, drift: float in [0,1], mode: str, reasons: [...]}.
    drift aggregates: (a) per-domain distance outside the validated [q01,q99]
    band (if a reference is loaded), and (b) cross-modal disagreement (always).
    """
    reasons: list[str] = []
    scores = {k: float(v) for k, v in domain_scores.items()}
    vals = list(scores.values())

    # reference-free novelty: cross-modal disagreement (high => modalities conflict)
    disagreement = (max(vals) - min(vals)) if len(vals) > 1 else 0.0

    band_drift = 0.0
    mode = "advisory_no_reference"
    if _REFERENCE is not None:
        mode = "reference_envelope"
        per = []
        for dom, s in scores.items():
            ref = _REFERENCE.get(dom)
            if not ref:
                reasons.append(f"domain '{dom}' not in validated envelope")
                per.append(1.0)
                continue
            lo, hi = float(ref.get("q01", 0.0)), float(ref.get("q99", 1.0))
            span = max(hi - lo, 1e-6)
            if s < lo:
                per.append(min((lo - s) / span, 1.0))
            elif s > hi:
                per.append(min((s - hi) / span, 1.0))
            else:
                per.append(0.0)
        band_drift = max(per) if per else 0.0
        if band_drift > 0:
            reasons.append("input score(s) outside validated range")

    drift = max(band_drift, 0.5 * disagreement)
    if disagreement >= 0.5:
        reasons.append("high cross-modal disagreement")
    in_envelope = drift < _DRIFT_THRESHOLD

    if _DRIFT_GAUGE is not None:
        _DRIFT_GAUGE.labels(endpoint=endpoint).set(drift)
    if _OOE_GAUGE is not None and not in_envelope:
        _OOE_GAUGE.labels(endpoint=endpoint).inc()

    return {
        "in_envelope": bool(in_envelope),
        "drift": round(float(drift), 4),
        "threshold": _DRIFT_THRESHOLD,
        "mode": mode,
        "reasons": reasons,
    }
