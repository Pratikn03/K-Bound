# Gate Decision Rule: *When* a Validation-Derived Drift Gate Should Switch

Status: design + reference implementation (forward / opt-in; does not alter any
locked Phase-2 artifact).
Code: `src/uais/fusion/attention/gate_decision_rule.py`
Tests: `tests/test_gate_decision_rule.py`

## 1. The contrast this rule explains

The locked evidence shows a single, reproducible signature: the **same**
validation-derived KS-drift gate behaves in two opposite ways depending on the
benchmark.

| Regime | Example | Gate behaviour | Why |
| --- | --- | --- | --- |
| Coherent collapse | Family B (label-aligned stress) | helps (positive Δ AUC) | every sample's evidence degrades the same way; the reliability signal is **tightly clustered** |
| Legitimate heterogeneity | Family D (Eyecandies), MVTec 3D-AD | null / hurts | a category mixture produces a **dispersed** reliability signal that the global threshold misreads as drift |

The mechanism is identical in both cases. The open question the manuscript
raises is therefore not *"does the gate work?"* but **"when is a
validation-derived gate safe to trust?"** This document answers that with an
explicit, cheap, finite-sample-guaranteed rule.

## 2. The rule

For a batch with per-(sample, domain) reliability weights `r_{i,d}` (from the
fitted `ReliabilityEstimator` / `PerSampleReliabilityEstimator`) and per-sample
mean reliability `r_i = mean_{d present} r_{i,d}`:

```
switch  iff  drift_coherence(batch) >= coherence_min
        and  bounded_switching_certificate(calibration) is certified
```

- **Drift coherence** `= 1 - min(1, 2 * std_i(r_i))`, in `[0, 1]`.
  - Coherent collapse → `r_i` tightly clustered → low std → coherence → 1.
  - Heterogeneous mixture → `r_i` dispersed → high std → coherence → 0.
- **Switching certificate** is the existing finite-sample condition
  `uais.utils.metrics.bounded_switching_certificate`: the gate may switch only
  if, on a labelled calibration set, the reliability path's loss advantage on
  the *fired* samples exceeds `margin_epsilon` **and** the realized gated policy
  loss is below the static loss.

If either condition fails, **no** sample is routed through the reliability path
and the static prediction is kept. This is exactly the conservative behaviour
that preserves the Family-B gain while avoiding the Family-D / MVTec regression.

## 3. Why this is the right shape

- It is **predictive, not post-hoc**: coherence is computable on an unlabeled
  deployment batch, so the rule decides *before* seeing outcomes whether the
  validation-derived gate transfers.
- It is **auditable**: the certificate term is the same condition the formal
  switching analysis already cites; the coherence term is a one-line statistic.
- It is **conservative by construction**: the default `coherence_min = 0.5`
  vetoes dispersed batches, which is the empirically dangerous regime.
- It **separates the two named failure modes** the paper documents into a
  single decision boundary rather than two anecdotes.

## 4. Relationship to the Phase-1 code fixes (this change set)

These are forward / opt-in fixes; they do **not** mutate locked Phase-2 numbers
(legacy behaviour remains reproducible via explicit config flags).

1. **Early-stopping best-weights restore** — `_train_model` now restores the
   best-validation checkpoint after early stopping (config flag
   `restore_best_weights`, default `true`; set `false` for bit-exact legacy
   reproduction). Previously the model used at evaluation was the
   patience-th epoch *after* the best one.
2. **Honest method labelling** — when `score_blend_on_gate` is enabled (the v4
   exploratory path that blends raw domain scores rather than running the
   attention model), the Family-D cell runner now archives the method as
   `rga_score_blend_*` instead of `base_rga_*`, so the score-blend variant is no
   longer conflated with pure attention-based RGA in downstream analysis.
3. **Per-sample reliability (ISSUE 2)** — `reliability.estimator_type:
   per_sample` selects `PerSampleReliabilityEstimator`, whose weights vary
   sample-by-sample (rank-based local KS + per-sample sharpness), so
   `per_sample_gating` and the per-sample attention reweighting are genuine
   rather than a batch-broadcast constant. Default `batch` preserves locked
   numbers. Benchmark harness: `src/scripts/run_per_sample_gating_benchmark.py`.
4. **Score-blend no longer bypasses the model (ISSUE 3)** — `score_blend_alpha`
   (default `1.0` = legacy pure blend) interpolates between the
   reliability-weighted score blend and the attention-fusion output, so the
   trained model contributes when `alpha < 1.0`. Plumbed through
   `_predict_craf_with_stats` and the Family-D cell runner; alpha-blended runs
   are archived as `rga_score_blend_a{alpha}_*`.
5. **Less-tautological one-class supervision (ISSUE 4)** — the pseudo-target
   aggregation now supports `median`/`quantile`/`trimmed_mean` (which require
   integrating domains rather than copying the strongest score), and
   `one_class_score_input_dropout` (default `0.0`) neutralizes a fraction of
   present-domain score inputs during training so the fusion head cannot
   trivially copy the supervision target. Defaults reproduce the legacy `max`
   target with full score visibility.

## 5. Validation status

- Unit tests encode the contrast directly: coherent collapse → switch allowed;
  heterogeneous mixture → switch suppressed; failed certificate → switch
  vetoed even when coherent. All pass (`tests/test_gate_decision_rule.py`).
- End-to-end audit: `src/scripts/audit_gate_decision_rule_e2e.py` writes
  `experiments/fusion/gate_decision_rule_e2e_audit.json` and
  `docs/research/tables/gate_decision_rule_e2e.tex` (via
  `emit_gate_decision_rule_table.py`). Synthetic scenarios always pass;
  locked B-MECH-1 archive proxies are included when parquet archives are
  present.
- Theorem registry: `src/elara/theory/theorem_registry.py` maps every
  theorem to code, scripts, and artifacts; `validate_theorem_stack.py`
  checks closure after rebuild.
- **Not yet** run end-to-end on the frozen benchmarks. To adopt operationally,
  the rule should be evaluated on the held-out Family-B and Family-D cells and
  the coherence / certificate thresholds calibrated on validation only. That is
  a deliberate, separately-audited experiment, not a silent change to existing
  results.

## 6. Next experiments (proposed, not executed)

1. Compute `drift_coherence` on each locked Family-B and Family-D test cell and
   confirm the coherence ordering matches the observed Δ AUC sign.
2. Calibrate `coherence_min` and `margin_epsilon` on validation-only stress and
   report the gated policy's risk dominance vs. static and vs. always-fire.
3. Re-run one Family-D cell with `restore_best_weights: true` to quantify how
   much the early-stopping fix moves Δ, as a clean v-next track.
