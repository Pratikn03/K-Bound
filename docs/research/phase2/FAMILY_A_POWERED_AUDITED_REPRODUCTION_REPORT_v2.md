# Family A — Powered Audited Reproduction Report — v2

**Status:** repaired report following Phase 2.1 drift detection. **v1 is preserved unchanged** at [FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md](./FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md) as historical drift evidence. This v2 report supersedes v1 for all future citations.

## 1. Cell identities (locked from the original registry)

The Family-A registry-defined cells are:

- **A-POWERED-1** — MVTec 3D-AD, PatchCore supervised-paired. **Pilot complete (30 seeds).**
- **A-POWERED-2** — MVTec 3D-AD, PatchCore held-out category. `pending_compute`.
- **A-POWERED-3** — MVTec LOCO-AD, PatchCore supervised-paired. `pending_compute`.
- **A-POWERED-4** — VisA, RGB+edge supervised-paired. `pending_compute`.
- **A-POWERED-5** — UNSW-NB15, flow/conn/context. `pending_compute`.

The v1 report incorrectly described A-POWERED-2..5 as Real3D / EfficientAD expansions; that description is **withdrawn**. EfficientAD or Real3D expansions may be added to the registry as separately numbered **exploratory** Family-C cells (e.g. `C-EXP-EFFICIENTAD-*`, `C-EXP-REAL3D-*`); they may not silently replace the Family-A cells above.

## 2. A-POWERED-1 — two analytic surfaces

### 2.1 `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` (existing v1 output, frozen, relabelled)

This is the K = 10 within-cell all-comparator Holm output from the v1 report.

- RGA+ ensemble ROC-AUC = **0.7420** on n_test = 278 (217 pos, 61 neg), 30 seeds.
- Validation-frozen RGA+ head distribution: 19 boost / 11 router.
- Five comparators reach Holm-significance under the within-cell K = 10 correction at α = 0.05: `static_attention`, `craf_attention`, `early_fusion_mlp`, `confidence_weighted_mean`, `eata_score_adapter`.
- Five comparators do **not** reach Holm-significance: `late_fusion_ensemble`, `random_forest`, `tent_score_adapter`, `sar_score_adapter`, `ttt_pseudo_label_adapter`.

Underlying data unchanged: [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv), [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv).

This surface may support **only** the statement:

> "On the seed-ensemble predictor for one MVTec 3D-AD supervised-paired pilot cell, RGA+ separates from five of ten named comparators under the all-comparator pilot audit."

It does **not** support any of the forbidden claims listed in [PHASE_2_1_PILOT_PRESERVATION_REPORT.md](./PHASE_2_1_PILOT_PRESERVATION_REPORT.md) §4.

### 2.2 `PRIMARY_FAMILY_A_CELL_LEVEL` — required recompute, not yet executed

Under the repaired v2 policy ([PHASE_2_STATISTICAL_POLICY_v2.md](./PHASE_2_STATISTICAL_POLICY_v2.md) §3), the primary Family-A surface for A-POWERED-1 is:

- DeLong paired test on the seed-averaged ensemble vectors of **RGA+ vs `static_attention` only** (one comparator, not ten).
- Paired bootstrap 95% CI on the same AUROC delta, 10 000 iterations, fixed seed 0.
- Holm correction applied across the K = 5 Family-A cells (not within-cell).
- Reported with the literal label `PRIMARY_FAMILY_A_CELL_LEVEL`.

This computation is **not executed in Phase 2.1** because the Phase 2.1 stop boundary explicitly forbids new analyses. When future compute opens, the primary surface for A-POWERED-1 is recomputed from the existing prediction archive; no new training is required.

A K = 5 Holm-adjusted p-value for the Family-A primary surface **cannot** be reported until A-POWERED-2..5 have also completed runs. Reporting one cell's p-value with a K = 5 correction implicit would understate the correction.

## 3. Forbidden claims (preserved verbatim)

- ELARA is universal.
- RGA+ beats every baseline.
- Existing Family A cells are confirmatory.
- Existing Family A cells are preregistered.
- ELARA is SOTA.
- ELARA is production-ready or deployment-ready.
- ELARA is validated for clinical deployment.
- Public benchmark results prove broad cross-domain superiority.
- Real3D supports generalization.
- Fixed-seed p-values prove robust method superiority.

Neither the secondary all-comparator pilot audit nor the (future) primary cell-level surface entitles any of these claims.

## 4. Seed-ensemble caveat

The seed-ensemble RGA+ predictor on A-POWERED-1 averages 19 identical boost predictions plus 11 router variants — **not** 30 independently retrained full pipelines. The boost head is deterministic; only the router has SGD-trained seed-dependent variation. See [PHASE_2_1_PILOT_PRESERVATION_REPORT.md](./PHASE_2_1_PILOT_PRESERVATION_REPORT.md) §5. Future Family-A reports must include this caveat or drop the seed-ensemble pooling in favour of per-seed per-model inference.

## 5. Pending-compute matrix

| Cell | Benchmark | Protocol | Primary surface status | Secondary surface status |
|---|---|---|---|---|
| A-POWERED-1 | MVTec 3D-AD | PatchCore supervised-paired | recompute pending (from existing archive) | computed, frozen as `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` |
| A-POWERED-2 | MVTec 3D-AD | PatchCore held-out category | `pending_compute` (no archive) | `pending_compute` |
| A-POWERED-3 | MVTec LOCO-AD | PatchCore supervised-paired | `pending_compute` | `pending_compute` |
| A-POWERED-4 | VisA | RGB+edge supervised-paired | `pending_compute` | `pending_compute` |
| A-POWERED-5 | UNSW-NB15 | flow/conn/context | `pending_compute` | `pending_compute` |

The Family-A `PRIMARY_FAMILY_A_CELL_LEVEL` K = 5 Holm correction is `pending_full_family`.
