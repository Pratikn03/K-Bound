# `kga` — Knowability-Guided Adaptation

A small, pure-`numpy`/`scipy`, torch-free implementation of **KGA**, the decision
algorithm from the paper *K-Bound: When Is Label-Free Adaptation Knowable?*
([`docs/research/kbound/K-Bound_paper.pdf`](../docs/research/kbound/K-Bound_paper.pdf)).

KGA decides — **without any target labels** — whether to **ADAPT**, **FREEZE**, or
**ABSTAIN** on a new test distribution. It does this from label-free *evidence* `Z`
and a finite-sample *certificate* `Δ̂ ± ε` on the benefit of adapting over freezing.

---

## Quickstart

```python
import numpy as np
from kga import KGA

rng = np.random.default_rng(0)
kga = KGA(alpha=0.1, method="ebern")

# (1) Label-free evidence Z from calibration vs unlabelled test scores.
calib = rng.normal(0.0, 1.0, size=(500, 3))
test  = rng.normal(0.0, 1.0, size=(500, 3))
z = kga.evidence(calib, test)
print(z.ks_mean, z.disagree, z.ess_frac)

# (2) Certificate Δ̂ ± ε from per-sample paired benefits X_i = loss(f0_i) - loss(fa_i).
benefits = rng.normal(0.3, 0.1, size=400)          # adapting clearly helps here
cert = kga.certify(scores=benefits, benefit_range=2.0)
print(cert.delta_hat, cert.epsilon, cert.lower)

# (3) Trichotomy decision (ADAPT / FREEZE / ABSTAIN).
print(kga.decide(cert))            # -> Decision.ADAPT

# (4) Audit everything.
import json; print(json.dumps(kga.explain(), indent=2))
```

Command line:

```bash
python -m kga decide --calib calib.npy --test test.npy --alpha 0.1
# {"decision": "ABSTAIN", "delta_hat": 0.0, "epsilon": ..., "method": "conformal", "evidence": {...}}
```

---

## The trichotomy and its guarantee

Let `Δ = R(f0) − R(fa)` be the true benefit of the adapted predictor `fa` over the
frozen one `f0` (positive ⇒ adapting reduces risk). A certificate gives a point
estimate `Δ̂` and a radius `ε` at level `α`. KGA applies:

| condition | decision |
|---|---|
| `Δ̂ − ε > 0` | **ADAPT** — adapting is certified beneficial |
| `Δ̂ + ε < 0` | **FREEZE** — adapting is certified harmful |
| otherwise | **ABSTAIN** — the sign of `Δ` is not knowable at level `α` |

**False-adapt ≤ α (Theorem 3, `thm:cert`).** The radius `ε` is built so that
`Δ ≥ Δ̂ − ε` holds with probability ≥ `1 − α`. ADAPT fires only when `Δ̂ − ε > 0`,
which on that event forces `Δ > 0`. Hence `P(ADAPT and Δ ≤ 0) ≤ α` — the chance of a
*harmful* adaptation is bounded by `α`. The symmetric statement bounds false-freeze.
The **anytime** variant (Theorem 3b, the e-value / testing-by-betting `method="evalue"`)
upgrades this to hold *simultaneously over all sample sizes* via Ville's inequality.

When the evidence `Z` cannot separate the two worlds (the non-identifiability regime
of Theorem 1, `thm:imp`), the certificate brackets zero and KGA **abstains** — the
safe default — rather than guessing.

---

## API reference

### `KGA(alpha=0.1, method="ebern")`
The facade gate. `method` selects the default batch certificate estimator
(`"ebern"`, `"hoeffding"`, or `"evalue"`).

- **`.evidence(calib, test, **kwargs) -> Evidence`** — label-free `Z`.
- **`.certify(...) -> Certificate`** — build `Δ̂ ± ε`. Three conventions:
  - `certify(scores=benefits)` — per-sample paired benefits → batch estimator;
  - `certify(adapt_risk=..., freeze_risk=..., calib_residuals=...)` — two scalar
    risks + held-out residuals → split-conformal;
  - `certify(delta_hat=..., calib_residuals=...)` — explicit estimate + residuals.
- **`.decide(certificate=None) -> Decision`** — the trichotomy (defaults to the
  last certificate built).
- **`.explain() -> dict`** — all intermediate quantities, JSON-serialisable.

### `Evidence` (`kga.evidence`)
Dataclass + `compute_evidence(calib_scores, test_scores, ...)`. Signals:
`ks_mean`/`ks_max` (KS drift), `disagree` (1 − mean pairwise rank correlation),
`entropy_shift`/`conf_shift` and the underlying entropies/decisiveness, and
`ess`/`ess_frac` (Gaussian-ratio importance-weight effective sample size).
Mirrors the `Z` blocks in
`src/scripts/kbound/knowability_experiment.py` and `mixed_regime_experiment.py`.

### `Certificate` (`kga.certificate`)
Frozen dataclass `(delta_hat, epsilon, method, alpha, n)` with `.lower`/`.upper`.
Estimators:
- `empirical_bernstein(...)` — Maurer–Pontil (2009) empirical-Bernstein LCB
  (batch Theorem 3; default). Identical formula to
  `docs/research/kbound/vendored_from_elara/certification/switching_certificate.py`.
- `hoeffding(...)` — distribution-free LCB (conservative baseline).
- `conformal_split(delta_hat, calib_residuals, ...)` — split-conformal radius
  `quantile(|Δ̂ − Δ|, 1 − α)` (the cross-task estimator used in the experiments).
- `evalue_anytime(...)` — anytime-valid betting e-process (Ville; Theorem 3b),
  mirroring `experiments/kbound/theory_validation/val_thm3_evalue.py`.

### `Decision` (`kga.policy`)
`Enum` with `ADAPT`, `FREEZE`, `ABSTAIN` (each compares equal to its string value).
`decide(certificate, alpha=None)` implements the trichotomy with strict
inequalities at the boundaries.

---

## Notes

- **Deterministic & torch-free.** Only `numpy`/`scipy` are required. No randomness
  in the estimators (the e-process is a deterministic recursion over the stream).
- **Provenance.** Every formula mirrors existing, validated K-Bound code; the
  docstrings cite the matching script/theorem. This package is a clean, importable
  API over that math, not a re-derivation.
- See `examples/kga_quickstart.py` for a runnable demo and
  `tests/test_kga_package.py` for the behavioural contract (including an empirical
  false-adapt-rate ≤ α check over thousands of synthetic trials).
