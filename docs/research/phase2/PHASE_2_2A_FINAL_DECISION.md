# Phase 2.2A — Final Decision

## Decision: **`PASS TO BEGIN FAMILY-B COMPUTE`**

All five registry-locked Family-A cells completed at 30 seeds each with valid prediction archives, the K = 5 Holm correction is applied, all QC tests pass, and no Family-D activity occurred.

## Evidence

| Cell | n_test | RGA+ ens | static ens | Δ | Holm K=5 p | 95% CI | Effect | Sign-consistent |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| A-POWERED-1 | 278 | 0.7420 | 0.6338 | +0.1082 | 3.35e-04 | [+0.052, +0.166] | large | 30/30 |
| A-POWERED-2 | 1681 | 0.5216 | 0.4698 | +0.0519 | 4.06e-05 | [+0.029, +0.075] | large | 30/30 |
| A-POWERED-3 | 472 | 0.7392 | 0.6354 | +0.1038 | 4.06e-05 | [+0.058, +0.150] | large | 30/30 |
| A-POWERED-4 | 648 | 0.8572 | 0.8275 | +0.0297 | 1.53e-03 | [+0.012, +0.049] | moderate | 30/30 |
| A-POWERED-5 | 18 001 | 0.9897 | 0.9802 | +0.0095 | < 1e-15 | [+0.008, +0.011] | small | 30/30 |

All 5 cells reach Holm-significance at K = 5 α = 0.05. All paired-bootstrap CIs strictly exclude 0 in the positive direction. Sign-consistency is 30/30 in every cell.

## Honest claim boundary

**Permitted from this evidence:**

> "Family A now provides powered audited static-reference evidence
> across five previously inspected benchmark cells. It evaluates
> whether validation-frozen RGA+ improves on a fixed static-attention
> reference; it is not confirmatory replication and is not a
> strongest-baseline superiority evaluation."

**Forbidden from this evidence (verbatim, preserved):**

- "RGA+ beats the best baselines."
- "RGA+ beats every baseline."
- "Family A confirms generalization."
- "Family A is confirmatory."
- "Family A is preregistered."
- "ELARA is SOTA."
- "ELARA is universal."
- "ELARA is production-ready or deployment-ready."
- "ELARA is validated for clinical deployment."
- "Public benchmark results prove broad cross-domain superiority."
- "Real3D supports generalization."
- "Fixed-seed p-values prove robust method superiority."
- "Proceed to Phase 3."

## Test suite state

- Before Phase 2.2A: 431 passed / 7 skipped.
- After Phase 2.2A:  **477 passed / 10 skipped** (40 net new tests; new skips correspond to placeholder-guards against archives that already exist).

## Important reading caveats (already in the v2 report)

1. **Static reference, not strongest-baseline.** The locked primary
   comparator is `static_attention`. Significant separation from
   static_attention does **not** imply separation from harder
   comparators (`late_fusion_ensemble`, `random_forest`, TTA score
   adapters). The within-cell K = 10 surface from A-POWERED-1 is
   preserved separately as `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` and
   already shows that 5 of those 10 harder comparators do **not**
   separate from RGA+ on that cell.

2. **A-POWERED-2 sits near chance.** Both methods are below 0.55 on
   the held-out category protocol. The +0.0519 Δ is statistically
   robust but in the "both methods are weak" regime.

3. **A-POWERED-4 is derived-view-proxy.** VisA RGB+edge is not an
   independent-modality cell; the +0.0297 Δ cannot support an
   independent-modality generalization claim.

4. **A-POWERED-5 is in the "small" practical-effect band.** Δ =
   +0.0095 on n = 18 001 produces an exceptionally tiny p but the
   absolute effect is small; the band qualifier MUST accompany any
   citation.

5. **Seed-ensemble caveat (inherited from Phase 2.1 §5).** The
   `rga_boosted_fusion` head is deterministic across seeds; only the
   router has SGD-trained seed-dependent variation. The seed-ensemble
   is therefore a weighted mix of a deterministic prediction with the
   router's seed variance — not 30 independent retrainings.

## What this decision unlocks

- Family-B compute (B-MECH-1..4, B-CERT-1) may now begin under the v2 policy.
- The v2 Family-A static-reference report ([FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md)) is the canonical citation for any Family-A discussion in future Phase-2 work.

## What this decision does NOT unlock

- Paper / thesis edits based on Phase 2.2A results.
- Family-D execution (v1 invalid, v2 design pending).
- RGA-v2 gate sweep execution.
- KS power / mixture-shift execution.
- Phase 3 / ELARA-Universal / ORIUS.

## Provenance

- Driver: [src/scripts/run_phase2_family_a_cell.py](../../../src/scripts/run_phase2_family_a_cell.py), [src/scripts/run_phase2_family_a_analysis.py](../../../src/scripts/run_phase2_family_a_analysis.py)
- Reports: [FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md), [PHASE_2_2A_HOSTILE_REVIEW_REPORT.md](./PHASE_2_2A_HOSTILE_REVIEW_REPORT.md), [PHASE_2_2A_REMAINING_GAPS.md](./PHASE_2_2A_REMAINING_GAPS.md), [PHASE_2_2A_REPRODUCTION_COMMANDS.md](./PHASE_2_2A_REPRODUCTION_COMMANDS.md), [PHASE_2_2A_ARTIFACT_MANIFEST.md](./PHASE_2_2A_ARTIFACT_MANIFEST.md), [PHASE_2_2A_CHANGELOG.md](./PHASE_2_2A_CHANGELOG.md)
- Data: [experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv](../../../experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv), [experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv](../../../experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv)
