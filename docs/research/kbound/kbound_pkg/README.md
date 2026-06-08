# kbound

Python package implementing the K-Bound certificate for test-time adaptation (TTA).
Provides conformal + anytime-valid adaptation gates, label-free evidence vectors,
a benefit router, and an optional gradient-gating PyTorch optimizer.

## Installation

```bash
pip install -e .                    # CPU-only (numpy + scikit-learn)
pip install -e ".[torch]"           # + torch optimizer support
```

## Quickstart

```python
import numpy as np
from kbound import (
    conformal_radius,
    EProcess,
    BenefitRouter,
    KGA,
)
from kbound.certificate import decide
from kbound.evidence import evidence_vector

# --- 1. Conformal radius (Thm thm:cert) ---
rng = np.random.default_rng(0)
residuals = rng.standard_normal(200)       # |Bhat_i - B_i| calibration residuals
eps = conformal_radius(residuals, alpha=0.1)
print(f"eps = {eps:.4f}")

# --- 2. Certificate decision ---
Bhat = 0.05   # predicted benefit from GBR
print(decide(Bhat, eps))   # 'adapt', 'freeze', or 'abstain'

# --- 3. Label-free evidence vector (Thm thm:disagree) ---
# p0 and pa are softmax probability arrays (n_samples x n_classes)
p0 = rng.dirichlet(np.ones(10), size=64)   # frozen model probs
pa = rng.dirichlet(np.ones(10), size=64)   # adapted model probs
z = evidence_vector(p0, pa, upd_norm=0.12)
print(f"evidence vector shape: {z.shape}")  # (11,)

# --- 4. BenefitRouter (leave-one-out GBR + conformal eps) ---
# Build synthetic conditions
Z = rng.standard_normal((60, 11))          # 60 conditions x 11 features
B = rng.uniform(-0.3, 0.5, 60)            # true benefits
router = BenefitRouter()
Bhat, eps, decisions = router.decide_all(Z, B, alpha=0.1)
print(f"decisions: {np.unique(decisions, return_counts=True)}")

# --- 5. Anytime-valid EProcess (Ville's inequality) ---
ep = EProcess(alpha=0.1, a=-1.0, b=1.0)
for benefit_sample in rng.uniform(-0.1, 0.2, 50):
    ep.update(benefit_sample)
    d = ep.decision()
    if d != "abstain":
        print(f"Stopped at t={ep.t}: {d}  (wealth={ep.wealth:.2f})")
        break

# --- 6. KGA (full gate) ---
kga = KGA(alpha=0.1)
# Pass precomputed probability arrays (no-torch path):
d = kga.decide(p0, pa, upd_norm=0.12)
print(f"KGA decision: {d}")

# --- 7. KBoundOptimizer (torch required) ---
# from kbound import KBoundOptimizer
# optimizer = KBoundOptimizer(base_optimizer, evidence_fn=..., cert_fn=...)
# optimizer.step()
```
