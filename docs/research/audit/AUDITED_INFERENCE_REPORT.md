# Audited Inference Report (Phase 1.D)

**Status:** completed.
**Branch:** `fix/elara-phase1-empirical-validity`.
**Source artifacts:**
- [`experiments/audit/rga_plus_validation_frozen_selection.csv`](../../../experiments/audit/rga_plus_validation_frozen_selection.csv)
- [`experiments/audit/audited_comparator_selection.csv`](../../../experiments/audit/audited_comparator_selection.csv)
- [`experiments/audit/statistical_family_registry.csv`](../../../experiments/audit/statistical_family_registry.csv)
- [`experiments/audit/audited_ensemble_inference_results.csv`](../../../experiments/audit/audited_ensemble_inference_results.csv)
- [`experiments/audit/descriptive_seed_variability.csv`](../../../experiments/audit/descriptive_seed_variability.csv)

Phase 1.D replaces the manuscript's invalid Fisher-combined per-seed DeLong + post-test-winner-comparator + test-set-RGA+ inference with an honest audited reanalysis that:

1. Selects the RGA+ head on validation ROC-AUC only (per Phase 1.B).
2. Selects the primary comparator on seed-mean validation ROC-AUC only (per Phase 1.C).
3. Computes a single representative-seed DeLong p-value per cell (raw per-seed predictions are not archived in current JSONs; the policy-preferred ensemble DeLong is deferred to the next runner-patched re-run, Phase 1.F).
4. Applies Holm-Bonferroni only within the locked Family A K=5 confirmatory cells (A2, A3, A5, A7, A8). Family B K=2 (B1, B2) is corrected separately. Family C K=0.

---

## 1. Audited Family A inference results

The Family A audited primary reanalysis uses validation-frozen RGA+ head + validation-frozen primary comparator + single-representative-seed DeLong (seed 42) + Holm-Bonferroni over K=5 confirmatory cells.

| Cell | Benchmark | Protocol | RGA+ head (val-frozen) | Primary comparator (val-frozen) | seed-mean RGA+ ROC | seed-mean comp ROC | Δ AUC | DeLong p (seed 42) | **p_Holm (K=5)** |
|---|---|---|---|---|---|---|---|---|---|
| A2 | MVTec 3D-AD | PatchCore supervised | router | SAR | 0.7389 | 0.7354 | +0.0035 | 0.919 | **0.919** (n.s.) |
| A3 | MVTec 3D-AD | PatchCore held-out | router | Tent | 0.5087 | 0.5031 | +0.0056 | 0.050 | 0.202 (n.s.) |
| A5 | MVTec LOCO-AD | PatchCore supervised | router | Tent | 0.7175 | 0.7260 | **−0.0084** | 0.126 | 0.378 (n.s.) |
| A7 | VisA | RGB+edge supervised | boost | RF | 0.8661 | 0.8548 | +0.0113 | 0.248 | 0.496 (n.s.) |
| A8 | UNSW-NB15 | flow/conn/context | router | RF | 0.9892 | 0.9889 | +0.0003 | 1.3 × 10⁻⁶ | **6.7 × 10⁻⁶ (sig.)** |

**Headline finding:** under the corrected audited-reanalysis statistical procedure, **four of five Family A confirmatory cells lose Holm-corrected significance**. Only UNSW-NB15 flow/conn/context survives Holm correction (and at near-zero practical effect: Δ = 0.0003 ROC-AUC).

MVTec LOCO-AD supervised-paired flips sign: the validation-frozen RGA+ head is **router** (chosen because router beats boost on validation for this cell), and the router test ROC-AUC (0.7175) is **below** the validation-frozen comparator Tent (0.7260). The previous paper claim "+0.008 over Tent, p_Holm = 1.1×10⁻⁵" is incompatible with the audited reanalysis: the corrected sign is negative.

---

## 2. Family A protocol-diagnostic cells (A1, A4, A6)

Family A also contains three protocol-diagnostic cells (canonical one-class on MVTec 3D-AD, MVTec LOCO-AD, and VisA). These cells are **not** confirmatory; they exist to show that under one-class training every supervised fusion head collapses near chance and the RGA gate's delta is essentially zero.

| Cell | RGA+ head (val-frozen) | Primary comparator (val-frozen) | RGA+ ROC | Comp ROC | Δ AUC | p_raw |
|---|---|---|---|---|---|---|
| A1 | MVTec 3D PatchCore canonical | boost | TTT | 0.500 | 0.500 | 0.000 | 1.000 |
| A4 | MVTec LOCO PatchCore canonical | boost | TTT | 0.500 | 0.500 | 0.000 | 1.000 |
| A6 | VisA RGB+edge canonical | boost | TTT | 0.500 | 0.500 | 0.000 | 1.000 |

(The validation-frozen comparator for these cells is the alphabetical tie-break among baselines all at val ROC = 0.5; the cell is descriptive only.)

These cells confirm what the Phase 1.A label/metric semantics audit already established: under canonical one-class training, supervised heads collapse to degenerate constant predictors. No PR-AUC / ECE / Brier claim is allowed from these cells (the canonical-metric block is retained from Phase 1.A).

---

## 3. Family C exploratory cells

| Cell | Benchmark | Protocol | RGA+ head (val-frozen) | Primary comparator (val-frozen) | Δ AUC | p_raw |
|---|---|---|---|---|---|---|
| C1 | Real3D-AD | PCA shape + depth supervised | router | TTT | **−0.0029** | 1.0 |
| C2 | VisA | RGB+random noise-floor | boost | RF | +0.0107 | 0.192 |
| C3 | UNSW-NB15 | held-out attack categories | router | RF | +0.0004 | 0.045 |

Real3D-AD flips sign as well: under the validation-frozen rule the chosen head is router and the chosen comparator is TTT; the corrected delta is **−0.003**. The previous paper claim of "Real3D-AD descriptor upgrade: boost reaches 0.566 above Tent 0.561" was driven by post-hoc test-winner selection.

C3 (UNSW held-out) has a raw p of 0.045 but Family C carries no Holm correction; reported descriptively only.

---

## 4. What is NOT in this report

- **No ensemble DeLong + paired sample bootstrap.** The policy specifies one paired DeLong on seed-averaged predictions plus a paired bootstrap over test samples for a 95 % CI. Both require raw per-seed test prediction arrays, which the current JSONs DO NOT archive. The Phase 1.F runner patch logs these for future runs, but today's audited analysis is limited to the single-representative-seed test that the existing JSONs do support.
- **No Family B audited inference table here.** Family B (mechanism endpoints B1 + B2 at locked τ=0.66) is reported in the manuscript's mechanism subsection; the same single-representative-seed-DeLong rule applies. Family B sweeps and ablations (B3, B4, B5) are descriptive only — no Holm-corrected significance claim.
- **No "RGA+ beats every baseline" framing anywhere.** Per AR-12 of the locked audited-reanalysis policy.

---

## 5. Implications for the manuscript

Phase 1.G must update the paper and thesis to reflect:

1. **Replace the cross-benchmark "RGA+ wins" prose** with the audited inferential summary above. Only one Family A cell survives Holm correction (UNSW SP at Δ=0.0003); MVTec LOCO SP and Real3D actually go negative under the validation-frozen rule.
2. **Remove the headline claim of significant gains on MVTec LOCO SP and VisA SP** (paper §sec:cross-benchmark-master, abstract). These cells are now n.s. after Holm.
3. **Reframe Real3D-AD** as Family C exploratory; remove the "boost reaches 0.5656 above Tent" claim; report Δ = −0.003 vs the validation-frozen comparator (TTT) as the audited descriptive value.
4. **Mark all Family A inferential entries** as "audited reanalysis; single-representative-seed DeLong (seed 42); ensemble DeLong pending raw-prediction archive". This is honest about the methodological limitation.
5. **Note the runner patch** (Phase 1.F): future re-runs will archive raw per-seed predictions and support the policy-preferred ensemble DeLong + paired sample bootstrap.

---

## 6. Comparison to the previous Fisher-combined / test-oracle results

The previous master comparison reported (paper Table sec:cross-benchmark-master):

| Cell | Old RGA+ (test-max) | Old comp (test-winner) | Old Δ | Old p_Holm |
|---|---|---|---|---|
| A2 MVTec 3D SP | 0.739 (router) | SAR (0.735) | +0.004 | 1.000 |
| A3 MVTec 3D held-out | 0.517 (boost) | TTT (0.516) | +0.001 | 1.000 |
| A5 MVTec LOCO SP | 0.734 (boost) | Tent (0.726) | +0.008 | 1.5e−5 |
| A7 VisA SP | 0.866 (boost) | RF (0.855) | +0.011 | 1.8e−4 |
| A8 UNSW SP | 0.989 (router) | RF (0.989) | 0.000 | 5.7e−12 |

Under the corrected audited reanalysis:

| Cell | Corrected RGA+ (val-frozen) | Corrected comp (val-frozen) | Corrected Δ | Corrected p_Holm | Direction of change |
|---|---|---|---|---|---|
| A2 | 0.7389 (router) | SAR | +0.0035 | 0.919 | n.s. (was n.s.) — no change |
| A3 | 0.5087 (router) | Tent | +0.0056 | 0.202 | n.s. (was n.s.) |
| A5 | **0.7175 (router)** | Tent (0.7260) | **−0.0084** | **0.378** | **sign flipped + lost significance** |
| A7 | 0.8661 (boost) | RF | +0.0113 | 0.496 | **lost significance** |
| A8 | 0.9892 (router) | RF | +0.0003 | 6.7e−6 | retained significance (already negligible Δ) |

Per Rule 2 (corrections that lower performance preserve the lower result), the corrected lower / sign-flipped numbers stand. The manuscript headline must be rewritten accordingly.
