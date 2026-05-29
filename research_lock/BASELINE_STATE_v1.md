# BASELINE_STATE_v1.md — Frozen Current Evidence (pre-Scenario-C)

Frozen snapshot of ELARA's evidence as of this lock. All values are copied from
the authoritative audited artifacts; this file adds **no new numbers**. Each row
carries a Scenario C label.

- Authoritative metrics source: `docs/research/phase3/FINAL_METRICS_MANIFEST.json`
  (source_commit `850804d73175f497a476d0142981426709e99470`).
- Authoritative claim source: `docs/research/phase3/FINAL_CLAIM_MATRIX.csv`.
- Final decision: `docs/research/phase2/PHASE_2_FINAL_DECISION_AUDITED.md`.

## Family A — RGA+ vs fixed static attention (component: RGA+)

Authoritative: `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv`.

| Cell | Benchmark | Pairing | n_test | RGA+ ens AUC | Static ens AUC | ΔAUC | DeLong p (Holm k5) | CI | Label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-POWERED-1 | MVTec 3D-AD | independent_modalities | 278 | 0.7420 | 0.6338 | +0.1082 | 3.35e-4 | [0.052, 0.166] | NEW CONFIRMATORY (prior) |
| A-POWERED-2 | MVTec 3D-AD (held-out cat.) | independent_modalities | 1681 | 0.5216 | 0.4698 | +0.0519 | 4.06e-5 | [0.029, 0.075] | NEW CONFIRMATORY (prior); near-chance absolute |
| A-POWERED-3 | MVTec LOCO-AD | derived_view_proxy | 472 | 0.7392 | 0.6354 | +0.1038 | 4.06e-5 | [0.058, 0.150] | NEW CONFIRMATORY (prior); proxy view |
| A-POWERED-4 | VisA | derived_view_proxy | 648 | 0.8572 | 0.8275 | +0.0297 | 1.53e-3 | [0.012, 0.049] | NEW CONFIRMATORY (prior); proxy view |
| A-POWERED-5 | UNSW-NB15 | naturally_structured_views | 18001 | 0.9897 | 0.9802 | +0.0095 | 1e-15 | [0.008, 0.011] | NEW CONFIRMATORY (prior); small effect |

**Bound:** comparison is vs **fixed static attention only**, NOT strongest
baselines (P2 not addressed). LOCO/VisA are derived-view proxies (P3 partial).

## Family B — Base RGA mechanism (component: Base RGA)

Authoritative: `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv`.

| Endpoint | Scenario | Phase-1 Δ | Phase-2 Δ | CI | DeLong p (Holm k2) | Label |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | zero_attack k=4 mean gate, τ=0.66 | +0.0506 | +0.0507 | [0.0364, 0.0650] | 4.31e-12 | VERIFIED_REPRODUCED |
| B2 | max_attack k=4 mean gate, τ=0.66 | +0.0319 | +0.0939 | [0.0741, 0.1149] | 1e-15 | COMPARABLE_BUT_ESTIMATOR_CHANGED |

**Bound:** B2 must always be shown side-by-side with the Phase-1 target
(+0.0319); not an exact magnitude reproduction. Controlled score-collapse
benchmark only.

## RGA-v2 sensitive gates — REJECTED (component: Base RGA)

Authoritative: `experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv`.

| Gate | Clean false-fire rate | Budget | Label |
| --- | --- | --- | --- |
| G0 (mean) | 0.0000 | 0.0100 | reference gate (kept) |
| G1 / G2 / G3 | 1.0000 | 0.0100 | FAILED (not promoted) |

Domain-composition shift (B-MECH-3S): global + domain-aware gates both fired at
1.000 → **UNRESOLVED** (cohort-mixture theorem open).

## Switching certificate — MIXED (component: Monitor/Certificate)

Authoritative: `experiments/phase2/certification/switching_certificates_v2.csv`.

- max_attack: **certified** (LCB +0.0085, π* 0.000).
- zero_attack: **not certified** (LCB −0.0050, δ₁ −0.0039).
- Retrospective evaluation certificate only — NOT a production safety guarantee.

## Family D — Eyecandies held-out transfer — NOT CONFIRMED (component: Base RGA / transfer)

Authoritative: `docs/research/phase2/FAMILY_D_V3_INFERENCE_REPORT.md`. Sealed
record: `family_d_failure_record.md`.

| Cell | ΔAUC | bootstrap CI | paired-t p | Clean FFR | Label |
| --- | --- | --- | --- | --- | --- |
| D-EYE-1 | −0.0010 | [−0.0114, 0.0092] | 0.3632 | 0.000 (budget ≤0.010) | FAILED (not confirmed) |
| D-EYE-2 | −0.0109 | [−0.0254, 0.0034] | 0.4468 | 0.000 (budget ≤0.010) | FAILED (not confirmed) |

**Main limitation (frozen):** calibration transfer under score-distribution
shift remains unresolved.

## Robustness — white-box (component: Monitor/robustness)

FGSM/PGD substantially damage fusion performance → no adversarial-robustness
claim. Reframed as a *safe-failure / abstention* research path (Phase 9).

## Engineering state relevant to Scenario C

| Capability | Status | Path |
| --- | --- | --- |
| Per-sample reliability estimator | Implemented, opt-in; genuine per-sample (not batch-broadcast) | `src/uais/fusion/attention/reliability_estimator.py` (`PerSampleReliabilityEstimator`) |
| Per-sample gating benchmark | Synthetic-scale only (NEW EXPLORATORY) | `src/scripts/run_per_sample_gating_benchmark.py` |
| Gate decision rule (coherence + certificate) | Implemented + unit-tested; not yet run on real benchmarks | `src/uais/fusion/attention/gate_decision_rule.py`; `docs/research/phase3/GATE_DECISION_RULE.md` |
| Early-stopping best-weights restore | Fixed (opt-in default) | `_train_model` in `src/scripts/run_breakthrough_experiment.py` |
| Score-blend no longer bypasses attention | `score_blend_alpha` (opt-in) | `_predict_craf_with_stats` |
| Less-tautological one-class supervision | aggregations + score-input dropout (opt-in) | `_pseudo_targets_from_domain_scores`, `_dropout_score_input` |
| Per-sample prediction logging | Implemented | `src/elara/evaluation/prediction_archive.py` (`PredictionArchive`) |

> Note: the per-sample / gate-decision / score-blend-α / one-class fixes landed
> as forward, opt-in changes (commit `beb3fba`). They are `NEW EXPLORATORY`
> until run under a frozen confirmatory protocol; they do NOT alter any value
> in this baseline.
