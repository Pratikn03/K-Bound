# Family A — Powered Audited Reproduction Report

**Phase:** 2.C (in-session pilot — 1 of 5 cells executed)
**Cell executed this session:** A-POWERED-1 — MVTec 3D-AD, PatchCore supervised-paired
**Cells deferred (pending_compute):** A-POWERED-2..5 — Real3D × {PatchCore, EfficientAD} × {one-class, supervised-paired}, MVTec 3D-AD × EfficientAD × {one-class, supervised-paired}
**Seeds:** 30 (seeds 42–71)
**Selection rule:** validation-only; per-seed RGA+ chosen head = `argmax_val{rga_meta_router, rga_boosted_fusion}` with tie-break to `rga_boosted_fusion`
**Inference rule:** seed-averaged DeLong paired test on the seed-averaged ensemble prediction vectors + paired bootstrap CI over 278 test samples (10 000 iterations, fixed seed 0), Holm–Bonferroni within the family of 10 named comparators

> **Inferential scope (read first):** This is an audited reanalysis of
> the 30-seed seed-ensemble predictor on a single benchmark cell. It
> is **not** independent confirmatory replication. Inferential
> generalization beyond this seed-ensemble predictor on this cell
> requires Family D (see [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md)).

## 1. Headline numbers (A-POWERED-1 only)

- **RGA+ ensemble ROC-AUC:** **0.7420**
- **Validation-frozen head distribution:** boost = 19 / 30 seeds; router = 11 / 30 seeds (no test-set winner selection — verified in `family_a_selection_log.csv`)
- **Sample size:** n_test = 278; n_positive = 217; n_negative = 61
- **Family Holm K:** 10 (one comparison per named comparator)

| Comparator | Ensemble AUC | Δ vs RGA+ | DeLong p (raw) | DeLong p (Holm K=10) | 95% bootstrap CI | Effect band | Holm-significant α=0.05 | CI excludes 0 |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| static_attention | 0.6338 | **+0.1082** | 1.68e-04 | **1.34e-03** | [+0.0519, +0.1662] | large | ✓ | ✓ |
| craf_attention | 0.5566 | **+0.1854** | 2.09e-07 | **1.88e-06** | [+0.1146, +0.2563] | large | ✓ | ✓ |
| early_fusion_mlp | 0.6707 | **+0.0713** | 8.05e-03 | **4.83e-02** | [+0.0186, +0.1253] | large | ✓ | ✓ |
| late_fusion_ensemble | 0.7008 | +0.0412 | 1.44e-01 | 5.74e-01 | [-0.0138, +0.0983] | large | — | — |
| confidence_weighted_mean | 0.6118 | **+0.1302** | 2.42e-03 | **1.69e-02** | [+0.0458, +0.2182] | large | ✓ | ✓ |
| random_forest | 0.7008 | +0.0412 | 8.27e-02 | 4.13e-01 | [-0.0052, +0.0896] | large | — | — |
| tent_score_adapter | 0.7353 | +0.0067 | 7.80e-01 | 1.00 | [-0.0396, +0.0555] | small | — | — |
| eata_score_adapter | 0.5000 | **+0.2420** | 2.57e-11 | **2.57e-10** | [+0.1678, +0.3102] | large | ✓ | ✓ |
| sar_score_adapter | 0.7354 | +0.0066 | 7.84e-01 | 1.00 | [-0.0396, +0.0555] | small | — | — |
| ttt_pseudo_label_adapter | 0.7237 | +0.0183 | 4.45e-01 | 1.00 | [-0.0270, +0.0664] | moderate | — | — |

Source: [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv) (raw rows), [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv) (Holm summary), [experiments/phase2/statistics/family_a_powered_seed_metrics.csv](../../../experiments/phase2/statistics/family_a_powered_seed_metrics.csv) (per-seed metrics), [experiments/phase2/statistics/family_a_selection_log.csv](../../../experiments/phase2/statistics/family_a_selection_log.csv) (selection trail).

## 2. Honest characterization

For the **seed-averaged ensemble predictor** on this cell:

- RGA+ separates from the static and CRAF baselines with **Holm-corrected DeLong p ≤ 4.83 × 10⁻²** and bootstrap CIs strictly above zero.
- RGA+ also separates from `confidence_weighted_mean`, `early_fusion_mlp`, and `eata_score_adapter` under Holm correction.
- RGA+ does **not** statistically separate from `late_fusion_ensemble`, `random_forest`, `tent_score_adapter`, `sar_score_adapter`, or `ttt_pseudo_label_adapter` after Holm correction. The bootstrap CIs for these comparators all include zero. The point-estimate deltas (+0.0067 to +0.0412) sit inside the "small-to-moderate" practical-effect band and do not support a claim of robust separation.
- The validation-frozen selection rule produced **no test-set winner picking** (router 11 ; boost 19; tie-break followed) — verified by `selection_used_test_metrics=False` in every selection-log row.

## 3. What this row does and does not entitle the manuscript to say

**Entitled:**
- "On the seed-ensemble predictor for MVTec 3D-AD supervised-paired, RGA+ separates from static, CRAF, early-fusion MLP, confidence-weighted mean, and an EATA score adapter at Holm-adjusted α = 0.05 (raw DeLong p between 2.57 × 10⁻¹¹ and 8.05 × 10⁻³)."
- "On the same predictor, RGA+ does not statistically separate from late-fusion ensemble, random forest, or three score-adapter TTA baselines."

**Not entitled (forbidden claims preserved):**
- ELARA is universal
- RGA+ beats every baseline
- Existing Family A cells are confirmatory
- Existing Family A cells are preregistered
- ELARA is SOTA
- ELARA is production-ready or deployment-ready
- ELARA is validated for clinical deployment
- Public benchmark results prove broad cross-domain superiority
- Real3D supports generalization
- Fixed-seed p-values prove robust method superiority

## 4. Remaining A-POWERED cells (pending_compute)

| Cell | Benchmark | Protocol | Status | Reason |
|---|---|---|---|---|
| A-POWERED-2 | MVTec 3D-AD | PatchCore one-class | pending_compute | this-session scope = 1 pilot cell |
| A-POWERED-3 | MVTec 3D-AD | EfficientAD supervised-paired | pending_compute | this-session scope |
| A-POWERED-4 | MVTec 3D-AD | EfficientAD one-class | pending_compute | this-session scope |
| A-POWERED-5 | Real3D | PatchCore supervised-paired | pending_compute | this-session scope |

These rows are recorded as `pending_compute` in the claim matrix and require a separate compute budget for the remaining 4 cells × 30 seeds.

## 5. Reproduction commands

```bash
# 1. Re-run the 30-seed pilot
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --seeds 30 --seed-start 42

# 2. Re-run the audited inference
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_analysis.py

# 3. Verify the prediction archive integrity
PYTHONPATH=src .venv/bin/python src/scripts/validate_phase2_prediction_archives.py
```

## 6. Open gaps explicitly recorded

- 4 of 5 A-POWERED cells remain pending_compute.
- No replication on a held-out benchmark (Family D is the contractual placeholder for this and is **not** executed in this session — see [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md)).
- The inference here is on the **seed-averaged predictor**; it does not characterize the typical single-trained-model AUC.
