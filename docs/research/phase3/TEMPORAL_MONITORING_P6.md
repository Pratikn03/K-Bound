# Temporal Monitoring + Abstention (Pillar P6)

**Status:** NEW EXPLORATORY (development/synthetic only; not confirmatory; touches
no sealed or final test set).

**Script:** `src/scripts/run_temporal_monitoring_study.py`
**Test:** `tests/test_temporal_monitoring_study.py`
**Result:** `output/phase10/temporal_monitoring_study.json`

## 1. Why this study

The T6 calibration-transfer study (`CALIBRATION_TRANSFER_CONDITION_T6.md`)
established two facts about a validation-calibrated reliability gate:

1. it **helps** when the target stays in-distribution (low target/reference
   divergence), including under an in-distribution domain failure; and
2. it **hurts** once the target score distribution drifts (the Family-D failure),
   because the KS drift term saturates and the gate can no longer down-weight a
   collapsed domain.

T6 also showed the only signal that separates the safe regime from the harmful
one is the **label-free target/reference divergence** — source-side coherence and
the source-validation certificate are blind to it.

This study turns that observation into an **online control policy** rather than an
observe-only monitor: a chronological stream of windows is replayed, and on each
window the system must decide — with no target labels — whether to trust the gate
or abstain to the static baseline.

## 2. Control policy

For each window:

```
drift_t   = mean KS distance of the window's HEALTHY domains vs the frozen
            source reference            (the may-fail domain 0 is excluded,
                                          matching the T6 divergence definition)
alert_t   = drift_t > delta*            (delta* calibrated on clean windows)
cert_t    = INVALID if alert_t else VALID
action_t  = abstain -> static fusion    if cert_t == INVALID
            allow   -> gated fusion      otherwise
```

`delta*` is the `(1 - budget)` quantile of per-window healthy divergence measured
on **clean** calibration windows, so clean periods alert at most `budget` of the
time (default budget = 1%). The gate threshold `tau` is selected, exactly as in
the locked protocol, as the `budget` quantile of clean calibration mean
reliability on a disjoint split. No target labels are used at decision time.

## 3. Schedule and what each regime probes

| Windows | Regime            | Construction                | Expected behaviour |
|--------:|-------------------|-----------------------------|--------------------|
| 0–3     | `CLEAN`           | in-distribution, no failure | quiet; gate ≈ static |
| 4–6     | `IN_DIST_FAILURE` | collapse domain 0, no drift | **no alert**, gate **helps**, policy gates |
| 7–11    | `TRANSFER_DRIFT`  | collapse domain 0 + transfer shift τ∈{1,1.5,2,3,4} | **alert**, gate **hurts**, policy abstains to static |

`IN_DIST_FAILURE` is the critical distractor: a real degradation is present, but
it is *in-distribution*, so the drift statistic stays low and the system must keep
gating (and benefits from it). `TRANSFER_DRIFT` is the harmful regime the policy
must catch and step out of the way for.

## 4. Result (seed 0, D=6, 400 samples/window)

```
 win           regime   drift  alert    gate  a_stat  a_gate   a_act
   0            CLEAN   0.058  False   allow   0.997   0.997   0.997
   1            CLEAN   0.039  False   allow   0.998   0.998   0.998
   2            CLEAN   0.043  False   allow   0.997   0.997   0.997
   3            CLEAN   0.049  False   allow   0.996   0.995   0.995
   4  IN_DIST_FAILURE   0.050  False   allow   0.961   0.971   0.971
   5  IN_DIST_FAILURE   0.044  False   allow   0.956   0.969   0.969
   6  IN_DIST_FAILURE   0.057  False   allow   0.950   0.951   0.951
   7   TRANSFER_DRIFT   0.318   True abstain   0.888   0.858   0.888
   8   TRANSFER_DRIFT   0.436   True abstain   0.829   0.803   0.829
   9   TRANSFER_DRIFT   0.570   True abstain   0.768   0.754   0.768
  10   TRANSFER_DRIFT   0.769   True abstain   0.701   0.701   0.701
  11   TRANSFER_DRIFT   0.899   True abstain   0.736   0.736   0.736

delta* = 0.059   clean false-alarm = 0.000   drift detection = 1.000
mean AUC  static = 0.8980   gated = 0.8942   policy = 0.9000
drift windows  gated = 0.7704   policy = 0.7843
```

Reading:

- **Clean windows raise no false alarms** (0/4) and the gate stays engaged.
- **In-distribution failures are NOT alerted** and the gate adapts, recovering
  ~+0.01–0.02 AUC over static — exactly where gating should help.
- **Every transfer-drift window is detected**, the certificate is invalidated, and
  the system **falls back to static**. Always-gating would have *lost* AUC in those
  windows (0.7704 vs the static/policy 0.7843); the policy avoids that harm.
- **Aggregate acted AUC (0.9000) ≥ both always-static (0.8980) and always-gated
  (0.8942)** — the policy keeps the upside of gating where it is safe and sheds the
  downside where it is not.

## 5. Per-window log schema

Each row in `rows` records the deployment-style monitoring log:

`window_id, regime, tau, domain0_collapsed, mean_reliability, drift_statistic,
alert, certificate_state, gate_state, fallback_state, auc_static, auc_gated,
auc_acted_policy, calibration_ece`.

`certificate_state ∈ {VALID, INVALID}`, `gate_state ∈ {allow, abstain}`,
`fallback_state ∈ {none, static_fallback}`. Certificate invalidation always
co-occurs with an explicit `static_fallback`, satisfying the requirement that an
invalidated certificate produces a visible system response.

## 6. Pass criteria (locked by the test)

1. Clean false-alarm rate ≤ budget (observed 0.0).
2. Drift detection rate = 1.0.
3. In-distribution-failure windows are not alerted and gating helps there.
4. A transfer-drift window exists where always-gating hurts, and the policy
   abstains (falls back to static) on every drift window.
5. Aggregate acted-policy AUC ≥ both always-static and always-gated.
6. Certificate invalidation always pairs with a `static_fallback` action.

## 7. Scope and limits

Synthetic and exploratory. It demonstrates the *mechanism* — a label-free drift
statistic gating a safe fallback — not a benchmark result. It does not establish
real-world detection latency, nor does it claim a specific operating point for any
sealed dataset. Promotion to a confirmatory claim requires replaying a real
chronological corpus through the same policy under the locked protocol.
