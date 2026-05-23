# infer_rga

Runtime-only inference for ELARA's reliability-gated attention fusion.

This package is the deployment-side surface of the research codebase.
It exposes a single `InferRGA` class that loads a trained fusion
checkpoint and a fitted reliability estimator and produces fused
anomaly scores plus optional reliability-gate diagnostics.

## What this package is for

Use `infer_rga` if you want to:

- Apply a trained ELARA fusion model to new data without pulling the
  full research codebase (no MVTec / VisA / UNSW loaders, no
  adversarial sweep machinery, no LaTeX emit scripts).
- Get per-domain reliability weights and a binary gate-fired flag at
  inference time, for observe-only deployment monitoring.

Use the research codebase (`src/uais/fusion/...`) directly if you
want to:

- Train a new fusion model.
- Run the benchmark sweep, k-of-D corruption protocol, gradient
  adversarial attacks, or causal-attribution audits.
- Generate paper assets, tables, or figures.

## Install

The package is intentionally light on runtime dependencies. You
need `torch`, `numpy`, and (for the saved reliability estimator)
`joblib`. With the research codebase checked out:

```bash
pip install torch numpy joblib
export PYTHONPATH=$PWD/src:$PWD
```

A standalone wheel build is described in `pyproject.toml`.

## Quickstart

```python
import numpy as np
from infer_rga import InferRGA

rga = InferRGA.from_checkpoint(
    model_path="checkpoints/fusion_visa.pt",
    estimator_path="checkpoints/reliability_visa.joblib",
)

# features: (batch, num_domains, feature_dim) float32 in [0, 1]
# masks:    (batch, num_domains) bool — True = domain missing
features = np.random.rand(8, 2, 5).astype(np.float32)
masks = np.zeros((8, 2), dtype=bool)

scores = rga.predict_proba(features, masks)            # shape (8,)
gate_info = rga.predict_with_gate(features, masks)     # dict with static_probs, mean_reliability, gate_fired
```

## Deployment guarantees

- `predict_proba` is the deployment-grade scorer. It uses the static
  attention fusion path; the reliability gate is **observe-only** in
  this package (the gate-fired flag is reported via
  `predict_with_gate` but it does not switch the score).
- `predict_with_gate` is the right API for the production monitor.
  The `gate_fired` flag should be logged but never used to change
  the deployed score without an explicit human-in-the-loop review
  step.
- The `PerSampleReliabilityEstimator` variant is supported as a
  drop-in. If the estimator was fit with
  `estimator_type: per_sample`, the returned reliability is
  per-sample (vs the default which is per-batch).

## License

Same license as the research codebase. See top-level `LICENSE`.
