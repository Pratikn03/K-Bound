# Family A v2 — Powered Audited Static-Reference Audit

**Surface label:** `PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT`
**Family size:** K = 5
**Primary comparator (locked across all 5 cells):** `static_attention`
**Multiplicity correction:** Holm–Bonferroni across K = 5 cell-level p-values
**Inference rule:** seed-averaged DeLong paired test on 30-seed ensemble prediction vectors + paired sample bootstrap CI (10 000 iterations, fixed seed 0)
**Selection rule:** validation-only RGA+ head (router vs boost, tie-break boost); `selection_used_test_metrics=False` verified per seed.
**Status:** `K5_FULL_FAMILY` — all five cells executed at 30 seeds each.

> **Scope of inference — read first.** This is a powered audited
> **static-reference** reproduction across five previously inspected
> benchmark cells. It evaluates whether validation-frozen RGA+
> improves on a fixed static-attention reference. It is **not**
> confirmatory replication. It is **not** a strongest-baseline
> superiority test. It is **not** evidence of competitive superiority
> over harder comparators. Family A does **not** carry confirmatory
> status; Family-D under a v2 contract is the only path to
> confirmatory language, and that v2 path is `V2_DESIGN_PENDING`.

## 1. Cell roster (registry-locked)

| Cell | Benchmark | Protocol | Pairing strength | Status |
|---|---|---|---|---|
| A-POWERED-1 | MVTec 3D-AD | PatchCore supervised-paired | independent_modalities | 30 seeds (recompute from existing archive) |
| A-POWERED-2 | MVTec 3D-AD | PatchCore held-out category | independent_modalities | 30 seeds |
| A-POWERED-3 | MVTec LOCO-AD | PatchCore supervised-paired | **derived_view_proxy** (rgb + edge_proxy; see [PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md](./PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md)) | 30 seeds |
| A-POWERED-4 | VisA | RGB+edge supervised-paired | **derived_view_proxy** (not independent modalities) | 30 seeds |
| A-POWERED-5 | UNSW-NB15 | flow/conn/context | naturally_structured_views | 30 seeds |

A-POWERED-4's pairing strength is `derived_view_proxy`. Any result
from this cell cannot support an independent-modality generalization
claim.

## 2. Per-cell static-reference audit

| Cell | n_test | RGA+ ens AUC | static ens AUC | Δ | DeLong p (raw) | **Holm K=5 p** | 95% bootstrap CI | Effect band | per-seed mean Δ ± SD | sign-consistent seeds |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| A-POWERED-1 | 278 | 0.7420 | 0.6338 | **+0.1082** | 1.67e-04 | **3.35e-04** | [+0.052, +0.166] | large | +0.1064 ± 0.0101 | 30 / 30 |
| A-POWERED-2 | 1681 | 0.5216 | 0.4698 | **+0.0519** | 1.21e-05 | **4.06e-05** | [+0.029, +0.075] | large | +0.0357 ± 0.0163 | 30 / 30 |
| A-POWERED-3 | 472 | 0.7392 | 0.6354 | **+0.1038** | 1.02e-05 | **4.06e-05** | [+0.058, +0.150] | large | +0.0993 ± 0.0199 | 30 / 30 |
| A-POWERED-4 | 648 | 0.8572 | 0.8275 | **+0.0297** | 1.53e-03 | **1.53e-03** | [+0.012, +0.049] | moderate | +0.0341 ± 0.0059 | 30 / 30 |
| A-POWERED-5 | 18 001 | 0.9897 | 0.9802 | **+0.0095** | < 1e-15 | **< 1e-15** | [+0.008, +0.011] | **small** | +0.0103 ± 0.0008 | 30 / 30 |

Source CSVs:
- [experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv](../../../experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv)
- [experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv](../../../experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv)

All 5 Holm-adjusted p-values fall below α = 0.05 and every paired-bootstrap
CI strictly excludes 0 in the positive direction. Sign-consistency is
30 / 30 in every cell.

## 3. Honest reading

**The signed, magnitude-banded statement is:**

> Across all five registry-locked Family-A cells, the seed-averaged
> RGA+ predictor produces a positive Δ AUC vs the static-attention
> reference, with 30 / 30 seeds sign-consistent per cell and a K = 5
> Holm-adjusted DeLong p ≤ 1.53 × 10⁻³ in every cell. Three cells
> show a **large** practical effect (A-POWERED-1, A-POWERED-2,
> A-POWERED-3), one shows a **moderate** practical effect
> (A-POWERED-4 — derived_view_proxy pairing), and one shows a **small**
> practical effect with very tight CI (A-POWERED-5 — n=18 001).

**The carefully phrased entitled claim is:**

> "Family A now provides powered audited static-reference evidence
> across five previously inspected benchmark cells. It evaluates
> whether validation-frozen RGA+ improves on a fixed static-attention
> reference; it is not confirmatory replication and is not a
> strongest-baseline superiority evaluation."

Per-cell permitted statements:

- **A-POWERED-1** (MVTec 3D-AD PatchCore supervised-paired): RGA+ improves on the static-attention reference by Δ AUC = +0.1082 (95% CI [+0.052, +0.166]), Holm-adjusted DeLong p = 3.35 × 10⁻⁴; large practical effect.
- **A-POWERED-2** (MVTec 3D-AD PatchCore held-out category): both methods sit near chance (RGA+ = 0.522, static = 0.470); RGA+ shows a positive Δ AUC = +0.0519 (CI [+0.029, +0.075]) at Holm-adjusted p = 4.06 × 10⁻⁵. Held-out category transfer remains a fundamentally hard setting; both methods are far below the supervised-paired performance.
- **A-POWERED-3** (MVTec LOCO-AD PatchCore supervised-paired): RGA+ improves on the static-attention reference by Δ AUC = +0.1038 (CI [+0.058, +0.150]) at Holm-adjusted p = 4.06 × 10⁻⁵; large practical effect.
- **A-POWERED-4** (VisA RGB+edge supervised-paired, **derived_view_proxy**): RGA+ improves on the static-attention reference by Δ AUC = +0.0297 (CI [+0.012, +0.049]) at Holm-adjusted p = 1.53 × 10⁻³; **moderate** practical effect. Because pairing strength is derived-view-proxy, this cell cannot support an independent-modality generalization claim.
- **A-POWERED-5** (UNSW-NB15 flow/conn/context, naturally_structured_views): RGA+ improves on the static-attention reference by Δ AUC = +0.0095 (CI [+0.008, +0.011]) at Holm-adjusted p ≪ 0.05; **small** practical effect. The improvement is statistically robust (n = 18 001) but lives in the "small" band; this must always be reported with the band qualifier.

## 4. Forbidden interpretation (verbatim)

- "RGA+ beats the best baselines."
- "RGA+ beats every baseline."
- "Family A confirms generalization."
- "Family A confirmatory."
- "ELARA is SOTA."
- "ELARA is universal."
- "ELARA is production-ready or deployment-ready."
- "ELARA is validated for clinical deployment."
- "Public benchmark results prove broad cross-domain superiority."
- "Real3D supports generalization."
- "Fixed-seed p-values prove robust method superiority."
- "Proceed to Phase 3."

A "strongest-baseline" interpretation is **not** what this surface
provides — the locked comparator is `static_attention`, which is a
reference, not the strongest comparator available. Any text that
states or implies competitive superiority over harder comparators is
forbidden.

## 5. Cell-by-cell pairing-strength caveats

- **A-POWERED-1** (independent_modalities): RGB + 3D depth from PatchCore; multimodal cell. Result reflects RGA+ behaviour vs static reference on a label-aligned supervised-paired protocol.
- **A-POWERED-2** (independent_modalities): MVTec 3D-AD held-out category split — cross-category transfer on the **same** dataset. Any positive Δ does not support cross-dataset generalization. Both methods are near chance on this cell.
- **A-POWERED-3** (**derived_view_proxy**): MVTec LOCO-AD supervised-paired uses rgb + edge_proxy (Sobel-gradient view of the same RGB observation; see [PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md](./PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md)). Logical-anomaly benchmark; mechanism behaviour may differ qualitatively from structural-anomaly benchmarks. Cannot support independent-modality generalization claims.
- **A-POWERED-4** (**derived_view_proxy**): VisA RGB+edge. The "edge" channel is a derived view of the RGB image, not an independent modality. Any Δ on A-POWERED-4 cannot support independent-modality generalization claims.
- **A-POWERED-5** (naturally_structured_views): UNSW-NB15 flow/conn/context. Statistically significant Δ = +0.0095 lives in the **small** practical-effect band; this band qualifier MUST accompany any quote of the p-value or CI.

## 6. Seed-ensemble caveat

The seed-ensemble RGA+ predictor for each cell averages per-seed
validation-frozen head outputs. Because the `rga_boosted_fusion` head
is deterministic across seeds and only the router head has
SGD-trained seed-dependent variation, seed-ensemble pooling is a
weighted mix of one deterministic prediction with the router's
variance. Single-trained-model deployment behaviour cannot be inferred
from the seed-ensemble inference; see [PHASE_2_1_PILOT_PRESERVATION_REPORT.md](./PHASE_2_1_PILOT_PRESERVATION_REPORT.md) §5.

## 7. Relationship to the historical secondary pilot

The historical K = 10 within-cell secondary pilot on A-POWERED-1
remains valid under its `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` label.
It is preserved at:

- [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv)
- [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv)

These files are **not** overwritten or modified by Phase 2.2A. The
v2 surface uses separately-named files prefixed `family_a_v2_`.

## 8. Reproduction commands

See [PHASE_2_2A_REPRODUCTION_COMMANDS.md](./PHASE_2_2A_REPRODUCTION_COMMANDS.md).
