# Phase 2.2A — Remaining Open Gaps

Each gap is named, bounded, and tied to the test that prevents
silent regression.

## G1 — Family-A K = 5 Holm is conditional on all five cells completing

- **State:** A-POWERED-1 primary surface is computed; A-POWERED-2..5 are pending until their drivers complete.
- **Closing this gap entitles:** the `K5_FULL_FAMILY` Holm-adjusted p-value in `family_a_v2_primary_cell_level_holm_k5.csv`.
- **Test guard:** `test_holm_full_family_has_exactly_five_cells`.

## G2 — Boost head is deterministic across seeds

- **State:** the `rga_boosted_fusion` head produces identical predictions for every seed because it has no SGD-trained parameters. The seed-ensemble is therefore a mix of one deterministic prediction with 30 router variants — not 30 independent retrainings.
- **Closing this gap entitles:** either (a) describing the seed-ensemble inferential statement honestly (already done in [FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md) §6) or (b) dropping the seed-ensemble pooling and reporting per-seed-per-model AUC summaries instead.
- **No test guard** — this is a documentary requirement.

## G3 — A-POWERED-4 cannot support independent-modality generalization

- **State:** A-POWERED-4 (VisA RGB+edge) pairing strength is `derived_view_proxy` — the "edge" channel is a derived view of the RGB image, not an independent modality.
- **Closing this gap entitles:** A-POWERED-4 contributes to the K = 5 family but **only** as a derived-view audit, not as independent-modality evidence. The report records this in §5.
- **Test guard:** `test_family_a_cells_match_locked_identities` ensures the cell's pairing strength remains `derived_view_proxy` in the registry.

## G4 — Family-A is not a strongest-baseline / competitive-superiority test

- **State:** the locked primary comparator is `static_attention`. RGA+ separation from `static_attention` does not imply RGA+ separation from harder comparators (e.g. `late_fusion_ensemble`, `random_forest`, the TTA score adapters).
- **Closing this gap entitles:** the manuscript may say "RGA+ improves on the static-attention reference"; it may not say "RGA+ beats the best baseline."
- **Test guard:** `test_report_contains_no_forbidden_competitive_phrase` (in `tests/test_phase2_family_a_no_competitive_superiority_claim.py`).

## G5 — Family-B mechanism replication still pending

- **State:** B-MECH-1..4, B-CERT-1 are scaffolds with `pending_compute` rows.
- **Phase 2.2A does not open these.** Family-B compute is the next Phase 2.2B / Phase 2.2C task.
- **Closing this gap entitles:** the inherited B1/B2 endpoints (+0.0506 / +0.0319) to be labelled `Reproduced` / `Directionally supported` / `Not reproduced` / `Inconclusive`.

## G6 — RGA-v2 promotion decision still pending

- **State:** [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml) locks 5 candidate gates and 6 promotion criteria. Zero gates have been executed.
- **Phase 2.2A does not run any gate.**

## G7 — Family D v2 design still pending

- **State:** `V2_DESIGN_PENDING`. No execution, no freeze.
- **Phase 2.2A does not touch this.**
- **Test guard:** `tests/test_phase2_family_d_untouched_during_family_a.py` (5 tests).

## G8 — Paper / thesis prose not yet checked for the new forbidden-claim phrases

- **State:** the v2 forbidden phrases ("RGA+ beats the best baselines", "Family A confirms generalization", etc.) are covered by `tests/test_phase2_no_forbidden_claims_in_manuscripts.py` from Phase 2.1. The Phase 2.2A new phrases ("strongest-baseline", "competitive superiority") are checked only in the v2 Family-A report.
- **Closing this gap entitles:** a CI-level guarantee that no future commit can introduce verbatim forbidden text into `PAPER_DRAFT_v1.tex` / `THESIS_CHAPTER_v1.tex` for the new Phase 2.2A phrases.
- **Estimated effort:** ~5 lines to extend the existing test's `FORBIDDEN_CLAIMS` list.

## G9 — Single-trained-model variance band still not in the report

- **Inherited from Phase 2.1.** Per-seed `rga_aucs` / `static_aucs` vectors are archived in `family_a_v2_primary_cell_level_raw.csv` but the report quotes the ensemble AUC, not the per-seed band.
- Pure prose addition; no recompute needed.

## G10 — Phase 3 / ELARA-Universal / ORIUS

- **State:** untouched. Out of scope.
