"""Runnable KGA quickstart on synthetic scores (numpy-only, ~20 lines).

Run:  python examples/kga_quickstart.py
"""

from __future__ import annotations

import json

import numpy as np

from kga import KGA

rng = np.random.default_rng(0)
kga = KGA(alpha=0.1, method="ebern")

# (1) Label-free evidence Z from calibration vs unlabelled test detector scores.
calib_scores = rng.normal(0.0, 1.0, size=(500, 3))
test_scores = rng.normal(0.5, 1.0, size=(500, 3))  # a covariate shift the gate can see
evidence = kga.evidence(calib_scores, test_scores)
print(f"drift KS(mean/max) = {evidence.ks_mean:.3f} / {evidence.ks_max:.3f}, "
      f"disagree = {evidence.disagree:.3f}, ESS frac = {evidence.ess_frac:.3f}")

# (2) Certificate Delta_hat +/- epsilon from per-sample paired benefits
#     X_i = loss(frozen_i) - loss(adapted_i)  (positive => adapting helps).
paired_benefits = rng.normal(0.25, 0.1, size=400)
certificate = kga.certify(scores=paired_benefits, benefit_range=2.0)
print(f"Delta_hat = {certificate.delta_hat:+.3f}  epsilon = {certificate.epsilon:.3f}  "
      f"lower = {certificate.lower:+.3f}")

# (3) Trichotomy decision and full audit trail.
decision = kga.decide(certificate)
print(f"decision = {decision.value}")
print(json.dumps(kga.explain(), indent=2))
