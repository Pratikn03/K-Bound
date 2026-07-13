# `kga` — Knowability-Guided Adaptation

A small, pure-`numpy`/`scipy`, torch-free implementation of **KGA**, the decision
algorithm from the paper *K-Bound: When Is Label-Free Adaptation Knowable?*
([`kbound_short_final_draft.pdf`](../docs/research/kbound/kbound_short_final_draft.pdf)).

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
| otherwise | **ABSTAIN** — do not commit the update |

**Marginal false-adapt control (Theorem 3, `thm:cert`).** If the interval satisfies
`P(|Δ̂-Δ|≤ε)≥1-α`, ADAPT firing at `Δ̂-ε>0` implies
`P(ADAPT and Δ≤0)≤α`. Exchangeability or another valid calibration argument is
needed to justify coverage. The theorem does not control conditional false-adapt
among committed updates.

The package also exposes an experimental e-process interface. Its process-level
validity requires the documented supermartingale and optional-stopping
assumptions; the repository does not claim a complete foundational Lean proof of
that layer.

When the empirical interval brackets zero, KGA abstains from committing the
update. This may reflect structural ambiguity, finite data, estimator inadequacy,
calibration-transfer failure, or conservative width; it does not by itself prove
the population impossibility condition.

### Multicandidate routing (Wave 4)

For **K adapter candidates** per condition, use ``kga.routing``:

```python
from kga.routing import route_panel, AnytimeMulticandidatePanel

# Batch Bonferroni FWER (thm:multicand, thm:multiclass-multicand)
dec = route_panel(deploy_scores, cal_scores, cal_truth, alpha=0.1)
print(dec.selected, dec.decision)  # index or None, adapt/abstain

# Anytime panel (thm:anytime-multicand)
panel = AnytimeMulticandidatePanel(k=4, alpha=0.1)
chosen = panel.update([0.2, -0.1, 0.05, 0.0])
```

Training hook: ``docs/research/kbound/scripts/multicandidate_decide_kga.py`` (LOO GBR +
Bonferroni panel). Repro: ``bash kbtrain.sh theory-v2``.

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

### `Certificate` (`kga.certificate`)
Frozen dataclass `(delta_hat, epsilon, method, alpha, n)` with `.lower`/`.upper`.
Estimators:
- `empirical_bernstein(...)` — Maurer–Pontil (2009) empirical-Bernstein LCB.
- `hoeffding(...)` — distribution-free LCB (conservative baseline).
- `conformal_split(delta_hat, calib_residuals, ...)` — exact-rank
  split-conformal residual radius.
- `evalue_anytime(...)` — experimental betting e-process; its anytime
  interpretation requires the documented process assumptions.

### `Decision` (`kga.policy`)
`Enum` with `ADAPT`, `FREEZE`, `ABSTAIN` (each compares equal to its string value).
`decide(certificate, alpha=None)` implements the trichotomy with strict
inequalities at the boundaries.

---

## Notes

- **Deterministic & torch-free.** Only `numpy`/`scipy` are required. No randomness
  in the estimators (the e-process is a deterministic recursion over the stream).
- **Provenance.** The exact-rank implementation is mapped to the short paper and
  tested in `tests/test_kga_package.py`.
- **Scope.** KGA wraps an externally supplied adapter. This standalone package
  does not ship an ELARA or other adapter implementation.
