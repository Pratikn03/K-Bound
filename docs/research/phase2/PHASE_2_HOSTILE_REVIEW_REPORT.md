# Phase 2 — Hostile Review (self-audit)

Posture: read this as a NeurIPS area-chair stress test of the Phase-2 in-session pilot. The goal is to surface the weakest points so future compute is spent where it matters.

## Critique 1 — "Only one cell was actually executed"

**Reviewer:** *"You ran 30 seeds on one benchmark cell and call this Phase 2. What does this entitle you to claim beyond that single cell?"*

**Response:** Almost nothing. The A-POWERED-1 row is an audited reanalysis on **one** seed-ensemble predictor on **one** dataset / protocol combination. The manuscript is allowed to say: "on this specific predictor on this specific cell, RGA+ separates from 5 of 10 named comparators at Holm-adjusted α = 0.05." It is **not** entitled to any of the ten forbidden claims preserved verbatim in [PHASE_2_INTERIM_REPORT.md](./PHASE_2_INTERIM_REPORT.md) §5. We accept this critique entirely; it is the reason the rest of Family A, all of Family B, and Family D are pre-registered but not executed.

## Critique 2 — "5 of 10 comparators do not separate. That is not a strong result."

**Reviewer:** *"`late_fusion_ensemble`, `random_forest`, and three TTA score adapters all sit inside the bootstrap CI. Doesn't this just mean RGA+ is a slightly better but largely-overlapping method on this cell?"*

**Response:** Substantially yes, on this cell. The honest reading is that RGA+ separates cleanly from the simpler / older baselines (`static_attention`, `craf_attention`, `early_fusion_mlp`, `confidence_weighted_mean`) and from one TTA baseline (`eata_score_adapter`, which is AUC = 0.5000 on this cell — possibly a configuration / training failure rather than a fair comparator). Against the harder baselines (`late_fusion_ensemble`, `random_forest`, three TTA adapters in the 0.72–0.74 range), the seed-ensemble delta is small (+0.0067 to +0.0412) and the bootstrap CI includes zero. The manuscript MUST report these non-significant rows alongside the significant ones — selective reporting would itself be a critique-3-level violation.

## Critique 3 — "The selection rule could still leak."

**Reviewer:** *"You say `validation-only` selection but you wrote the framework yourself. What stops `selection_used_test_metrics=False` from being set incorrectly?"*

**Response:**

1. The schema test [tests/test_phase2_prediction_archive_schema.py::test_write_rejects_test_set_selection_flag](../../../tests/test_phase2_prediction_archive_schema.py) refuses to write any archive entry with `selection_used_test_metrics=True` on the test split.
2. The active-source guard test [tests/test_phase2_validation_only_selection.py::test_phase2_source_no_test_set_selection](../../../tests/test_phase2_validation_only_selection.py) grep-checks the active Phase-2 source files for forbidden patterns like `max(rga_router_test...`, `argmax test_roc`, and "best non-router."
3. The selection log [experiments/phase2/statistics/family_a_selection_log.csv](../../../experiments/phase2/statistics/family_a_selection_log.csv) records `selection_used_test_metrics=False` for every selection event and is verified by [src/scripts/validate_phase2_prediction_archives.py](../../../src/scripts/validate_phase2_prediction_archives.py).
4. The validation-frozen head distribution (19 boost / 11 router) is **not** the same as the test-best distribution; if test-set selection had been used, the distribution would lean toward whatever happened to be a hair higher on each seed's test AUC.

These four checks are mutually independent; a leak would have to defeat all four. Reviewer accepted: not impossible, but expensive.

## Critique 4 — "DeLong on the seed-averaged ensemble is not the same as DeLong on a typical model."

**Reviewer:** *"Your inference is about a hypothetical predictor that averages predictions from 30 trained models. A practitioner deploying one trained model gets neither this AUC nor this CI."*

**Response:** Accepted as a critical scope-of-inference statement. The Family-A report's first sentence makes this explicit:

> This is an audited reanalysis of the 30-seed seed-ensemble predictor on a single benchmark cell. It is **not** independent confirmatory replication.

The per-seed AUC vector (`per_seed_rga_aucs`) is recorded in the inference CSV so a future reader can produce a single-model variance band if needed. But we do **not** report a "single-trained-model" claim from this analysis.

## Critique 5 — "Family D is pre-registered and never run. Is the freeze itself doing any work?"

**Reviewer:** *"You wrote a contract, put dataset names in a JSON, and called it pre-registered. None of it has been executed. Is this just performative?"*

**Response:** The pre-registration only matters if the freeze commit predates the test reads. The contract files explicitly state this:

- [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md) §2 — "The git commit that locks this contract MUST predate the commit that produces any Family-D test-split artefact."
- [FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md](./FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md) §5 — pre-registration integrity contract requires the file hashes at execution time to equal the freeze hashes.
- [FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md) — execution commands are verbatim; deviation invalidates the family.

If the next compute window for Family D does not run those commands verbatim with hashes verified, the result does **not** carry confirmatory weight and the manuscript cannot say "confirmed on a held-out benchmark." The freeze is doing real work because it bounds what future text can claim.

## Critique 6 — "B1/B2 endpoints are inherited, not re-derived."

**Reviewer:** *"You preserve B1 = +0.0506 [0.0315, 0.0681] and B2 = +0.0319 [0.0050, 0.0617] as locked endpoints. Where do those come from in Phase 2?"*

**Response:** They come from prior work and are **claim-matrix targets** for Family B mechanism replication, not Phase-2 derivations. The Family B reports in this session are **scaffolds** marked `pending_compute`. A successful Phase-2 mechanism replication would land an entry in [experiments/phase2/mechanism/family_b_primary_replication_inference.csv](../../../experiments/phase2/mechanism/family_b_primary_replication_inference.csv) labelled `Reproduced`, `Directionally supported`, `Not reproduced`, or `Inconclusive` — none of which has happened this session. The manuscript MUST NOT cite B1/B2 as Phase-2-replicated until that CSV is populated by execution, not by scaffold.

## Critique 7 — "The RGA-v2 gate contract is a YAML, not a result."

**Reviewer:** *"You locked five candidate gates and six promotion criteria but ran zero of them. The contract itself entitles no claim."*

**Response:** Correct. The contract is locked so that, **when** the gate-search compute window opens, the decision rule is pre-frozen and there is no post-hoc tuning. As of this session, no Phase-2 result claims an RGA-v2 promotion — the [RGA_V2_PARTIAL_FAILURE_REPORT.md](./RGA_V2_PARTIAL_FAILURE_REPORT.md) scaffold is labelled `pending_compute`.

## Critique 8 — "Forbidden claims could still leak through unguarded prose."

**Reviewer:** *"You list forbidden claims but the paper / thesis prose is not under contract integrity checks. What stops a future edit from introducing 'ELARA is SOTA' on page 4?"*

**Response:** Accepted as a residual risk. The forbidden claims are preserved verbatim in:

- [PHASE_2_RESEARCH_CONTRACT.md](./PHASE_2_RESEARCH_CONTRACT.md) §2
- [PHASE_2_INTERIM_REPORT.md](./PHASE_2_INTERIM_REPORT.md) §5
- [FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md](./FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md) §3
- [FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md](./FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md) §5

The next session-level safeguard would be a `tests/test_phase2_paper_forbidden_claims.py` that grep-checks `docs/research/PAPER_DRAFT_v1.tex` and `docs/research/THESIS_CHAPTER_v1.tex` for the verbatim forbidden strings. This is recorded as an open gap (see [PHASE_2_REMAINING_OPEN_GAPS.md](./PHASE_2_REMAINING_OPEN_GAPS.md)).
