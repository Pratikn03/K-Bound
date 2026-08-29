# `kga` — Knowability-Guided Adaptation

A small, lightweight `numpy`/`scipy`/`PyYAML`, torch-free implementation of **KGA**,
the decision algorithm from the paper *K-Bound: When Is Label-Free Adaptation Knowable?*
([`docs/research/kbound/kbound_short_final_draft.pdf`](../docs/research/kbound/kbound_short_final_draft.pdf)).

KGA can decide without **deployment** target labels after a benefit estimator has
been fitted on labelled development conditions and calibrated on a disjoint
labelled residual split. At deployment it maps label-free evidence `Z` to a
finite-sample certificate `Δ̂ ± ε` on the benefit of adapting over freezing.

---

## Quickstart

```python
import hashlib
import numpy as np
from kga import EVIDENCE_FEATURE_NAMES, KGA, fit_frozen_linear_benefit_estimator

rng = np.random.default_rng(0)
kga = KGA(alpha=0.1)

# Before deployment: fit h(Z) on development units and calibrate residuals on
# separate labelled units. The protocol digest binds the model, adapter, splits,
# feature schema, and alpha.
protocol_sha = hashlib.sha256(b"locked-demo-protocol").hexdigest()
x_fit = rng.normal(size=(80, len(EVIDENCE_FEATURE_NAMES)))
y_fit = 0.15 * x_fit[:, 0] - 0.10 * x_fit[:, 1]
x_cal = rng.normal(size=(40, len(EVIDENCE_FEATURE_NAMES)))
y_cal = 0.15 * x_cal[:, 0] - 0.10 * x_cal[:, 1]
estimator = fit_frozen_linear_benefit_estimator(
    x_fit, y_fit, x_cal, y_cal,
    feature_names=EVIDENCE_FEATURE_NAMES,
    evidence_schema_version="kga-generic-score-evidence/1",
    protocol_sha256=protocol_sha,
)

# (1) Label-free evidence Z from calibration vs unlabelled test scores.
calib = rng.normal(0.0, 1.0, size=(500, 3))
test  = rng.normal(0.0, 1.0, size=(500, 3))
z = kga.evidence(calib, test)
print(z.ks_mean, z.disagree, z.ess_frac)

# (2) Deployment certificate: Z actually drives the frozen estimator.
cert = kga.certify_evidence(estimator, protocol_sha256=protocol_sha)
print(cert.delta_hat, cert.epsilon, cert.lower)

# (3) Trichotomy decision (ADAPT / FREEZE / ABSTAIN).
print(kga.decide(cert))

# (4) Audit everything.
import json; print(json.dumps(kga.explain(), indent=2))
```

Command line:

```bash
python -m kga decide --calib calib.npy --test test.npy \
  --estimator-json benefit.json --protocol-sha256 "$PROTOCOL_SHA" --alpha 0.1
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
| otherwise | **ABSTAIN** — the empirical certificate supports no strict update |

**Unconditional false-adapt event (Theorem 3, `thm:cert`).** If the radius construction
attains the stated marginal coverage, then
`Δ ≥ Δ̂ − ε` holds with probability ≥ `1 − α`. ADAPT fires only when `Δ̂ − ε > 0`,
which on that event forces `Δ > 0`. Hence `P(ADAPT and Δ ≤ 0) ≤ α` — the chance of a
*harmful* adaptation is bounded by `α`. The symmetric statement bounds false-freeze.
The **anytime** variant uses a testing-by-betting e-process; its guarantee requires
the declared bounded-stream and predictable-bet assumptions.

An empirical abstention does not diagnose its own cause. It can reflect structural
non-identifiability, a weak estimator, finite calibration size, failed transfer, or
a deliberately conservative radius.

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
- **`.certify_evidence(estimator, protocol_sha256=..., ...) -> Certificate`** —
  the label-free deployment path. It rejects schema/protocol mismatches.
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
`ess`/`ess_frac` (Gaussian-ratio importance-weight effective sample size). This
is the package's generic 10-feature score schema. The benchmark runners use
separate protocol-specific 11-, 17-, and 18-feature schemas and must declare
those identities explicitly.

### `Certificate` (`kga.certificate`)
Frozen dataclass `(delta_hat, epsilon, method, alpha, n)` with `.lower`/`.upper`.
Estimators:
- `empirical_bernstein(...)` — Maurer–Pontil (2009) empirical-Bernstein LCB
  (batch Theorem 3; default). Identical formula to
  `docs/research/kbound/vendored_from_elara/certification/switching_certificate.py`.
- `hoeffding(...)` — distribution-free LCB (conservative baseline).
- `conformal_split(delta_hat, calib_residuals, ...)` — exact-rank split-conformal
  radius on a disjoint exchangeable residual split. Controlled-grid LOO replay
  is empirical residual calibration, not exact split conformal.
- `evalue_anytime(...)` — anytime-valid betting e-process (Ville; Theorem 3b),
  mirroring `experiments/kbound/theory_validation/val_thm3_evalue.py`.

### `Decision` (`kga.policy`)
`Enum` with `ADAPT`, `FREEZE`, `ABSTAIN` (each compares equal to its string value).
`decide(certificate, alpha=None)` implements the trichotomy with strict
inequalities at the boundaries.

### Population frontier (`kga.frontier`)

The optional population companion is deliberately separate from KGA:

```python
from kga.frontier import assess_frontier

assessment = assess_frontier(M=0.30, beta=0.10)
print(assessment.action)  # Decision.ADAPT
```

`M` is the population evidence margin and `beta` is an externally declared
bound on latent drift over a specified target class. `beta=None` fails closed;
it is never replaced by zero or by KGA's empirical radius. Conversely, the KGA
API consumes `delta_hat` and `epsilon` and does not accept `M` or `beta`.
The controlled seven-cell bridge can be regenerated with:

```bash
python docs/research/kbound/scripts/run_frontier_kga_bridge.py
```

That diagnostic reports agreement and disagreement between the two rules. It
does not estimate `beta`, and it is not a real-data performance experiment.

---

## Optional ELARA-U integration

ELARA-U can construct a validation-fitted multimodal detector candidate and KGA
can certify whether to deploy it. This is composition, not a change to KGA's
certificate: ELARA proposes the candidate; KGA returns `ADAPT`, `FREEZE`, or
`ABSTAIN`. Tent, EATA, SAR, and other candidates remain independent alternatives.

```python
import numpy as np

from kga.integrations.elara import ELARAKGAGuard, EvaluationMode

guard = ELARAKGAGuard(alpha=0.1)
probe = np.arange(32)  # fixed before scoring
result = guard.decide(
    s_val=Sval,
    y_val=yval,
    s_test=Stest,
    y_test=ytest,
    mode=EvaluationMode.TARGET_LABEL_LIGHT,
    probe_indices=probe,
)
print(result.decision, result.router_action)
```

The information boundary is explicit:

- `retrospective_audit` uses all target labels and is never claim-eligible.
- `target_label_light` uses only fixed `probe_indices` for its decision.
- `label_free` rejects `y_test` and requires a schema- and protocol-bound
  `FrozenLinearBenefitEstimator` calibrated on disjoint conditions.

Run the current opened-cache audit with:

```bash
bash docs/research/kbound/scripts/kbtrain.sh kga-elara-integrated
```

Its result is retrospective evidence, not a label-free or headline win.

---

## Notes

- **Deterministic & torch-free.** Only `numpy`/`scipy` are required. No randomness
  in the estimators (the e-process is a deterministic recursion over the stream).
- **Provenance.** A frozen benefit artifact includes its protocol digest,
  evidence schema, feature normalization, residuals, and self-check SHA-256.
- See `examples/kga_quickstart.py` for a runnable demo and
  `tests/test_kga_package.py` for the behavioural contract (including an empirical
  false-adapt-rate ≤ α check over thousands of synthetic trials).
