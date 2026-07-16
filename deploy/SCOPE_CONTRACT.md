# Deployment Scope Contract (Gate P / P13)

This contract declares the **validated operating envelope** of the K-Bound
**KGA certificate API** (`deploy/api/`). Deploying or claiming performance
outside this envelope is not supported by the evidence.

## What is validated (deploy within this)

- **KGA certificate decisions** (`POST /decide`): adapt / freeze / abstain from
  label-free calibration and test scores with finite-sample false-adapt control
  at level α (see `kga/` and the K-Bound paper).
- **Scope-guard drift telemetry** on certificate traffic (advisory / reference
  envelope modes via `deploy/api/scope_guard.py`).

## What is NOT validated (do NOT claim/deploy as superiority)

- **Clean cross-domain transfer superiority**: proven *unattainable* against the
  confidence-weighted mean on near-ceiling clean data (Theorem T9). On clean
  inputs the system is *non-inferior at best*; it must not be marketed as beating
  baselines on clean transfer.
- **Universal / unscoped SOTA detection** across arbitrary domains.
- **Production guarantees** beyond this scoped, monitored deployment until the
  remaining Gate P items (P11 model governance, P15 load test) are closed.

## In-deployment safeguards (enforced)

- **Out-of-envelope drift guard** (`deploy/api/scope_guard.py`): every fusion
  inference is annotated with a drift score; out-of-envelope requests increment a
  Prometheus counter (`kga_out_of_envelope_total`) and set `kga_scope_drift`.
  Operators MUST alert on / hold out-of-envelope traffic rather than trust it.
- **Reference envelope**: set `KGA_SCOPE_REFERENCE` (legacy `UAIS_SCOPE_REFERENCE`
  still accepted) to per-domain validated score
  quantiles to enable full envelope checking; otherwise the guard runs in advisory
  (disagreement-only) mode and says so.
- Mandatory auth on `/decide`, no-wildcard CORS in production, rate limiting,
  request timeout. **No pickle/torch model loading** in the KGA-only service.

## Honest status

Gate P verdict: **SCOPED_PRODUCTION_READY** (12/15; all CRITICAL criteria pass) —
security/serving are production-grade and the live out-of-envelope drift guard +
scope contract are in place, so the system is deployable as a **scoped, monitored**
service within the validated envelope above. It is **not** cleared for unscoped
production until the remaining non-critical items close: P5 (distributed rate
limiting), P11 (model versioning/rollback), P15 (load/scale test). The
natural-degradation transfer is development-grade and NOT confirmed (D20). See
`elara_master_c/audits/gate_p_production_audit.json` (authoritative verdict).
