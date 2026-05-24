# Phase 2.2A — Hostile Review Report

Posture: senior empirical-ML implementation engineer + hostile reproducibility reviewer auditing the Phase 2.2A static-reference audit output.

## C1 — "Is the driver actually general, or just a thin wrapper around the pilot?"

**Reviewer:** *"You said the new driver is registry-driven. Show me that it cannot be tricked into running an unregistered benchmark."*

**Response:** Three guards.

1. The driver reads the cell row from `PHASE_2_EXPERIMENT_REGISTRY_v2.csv` and refuses any `experiment_id` not present.
2. `_validate_cell()` rejects any row whose `analysis_family` is not `A`, refuses any `experiment_id` not starting with `A-POWERED-`, and refuses any row whose `primary_comparator` is not `static_attention`.
3. `_config_for()` refuses any (benchmark, protocol) pair not in the locked `CONFIG_MAP` constant.

[tests/test_phase2_family_a_driver_registry.py](../../../tests/test_phase2_family_a_driver_registry.py) parametrizes both the rejection direction (B-MECH-1, B-CERT-1, C-EXP-EFFICIENTAD-1, D-CONTRACT-V2, A-POWERED-99) and the acceptance direction (every A-POWERED-N).

## C2 — "Does the analysis driver actually use static_attention only?"

**Reviewer:** *"The pilot driver runs 12 methods. What stops the analysis driver from sneaking another comparator in?"*

**Response:** [tests/test_phase2_family_a_static_reference_policy.py](../../../tests/test_phase2_family_a_static_reference_policy.py) greps the driver source for the names of the 8 forbidden comparators (`late_fusion_ensemble`, `random_forest`, `tent_score_adapter`, `sar_score_adapter`, `eata_score_adapter`, `ttt_pseudo_label_adapter`, `confidence_weighted_mean`, `early_fusion_mlp`, `craf_attention`). Any reference at all to those names in the analysis driver's source code fails the test.

The driver's `_per_cell_audit()` function calls `audited_analysis(...)` with exactly two prediction dictionaries — RGA+ (validation-frozen head) and `static_attention`. No other comparator is loaded.

## C3 — "How do you prevent overwriting the historical K = 10 secondary pilot?"

**Reviewer:** *"You said the historical pilot is preserved. Show me the guard."*

**Response:** Two layers.

1. `_verify_not_overwriting_a1_historical()` in the cell driver refuses to write to `family_a_powered_seed_metrics.csv` or `family_a_selection_log.csv` unless `experiment_id == "A-POWERED-1"`. Since the v2 driver always writes to `family_a_v2_<EID>_seed_metrics.csv`, the path collision is impossible by construction.
2. The analysis driver writes to `family_a_v2_primary_cell_level_raw.csv` and `family_a_v2_primary_cell_level_holm_k5.csv` only — names that cannot collide with the historical files.

[tests/test_phase2_family_a_historical_pilot_unchanged.py](../../../tests/test_phase2_family_a_historical_pilot_unchanged.py) checks that the historical CSV headers still contain the legacy `comparator_method` column.

## C4 — "What if A-POWERED-2..5 produce small or negative Δ vs static?"

**Reviewer:** *"Your A-POWERED-1 number is large (+0.108) but small benchmarks often produce tiny effects. Are you ready to report a near-zero or negative Δ honestly?"*

**Response:** The driver reports the descriptive statistics regardless of sign or magnitude:

- per-seed mean Δ ± SD;
- sign-consistent seed count;
- ensemble Δ + DeLong p (raw) + Holm K = 5 p;
- 95% bootstrap CI (which will straddle zero if the effect is near-zero);
- practical-effect band (`negligible` / `very small` / `small` / `moderate` / `large`).

The Family-A v2 report's allowed-interpretation template covers both
the "positive Δ" and "near-zero / negative Δ" cases without
modification:

> "On A-POWERED-N (<benchmark> <protocol>), RGA+ shows a Δ AUC of {Δ} vs the static-attention reference; the paired-bootstrap 95% CI [{ci_low}, {ci_high}] {includes / excludes} zero; practical effect band = {band}."

If A-POWERED-2 ends up at Δ ≈ 0 with CI straddling zero, the report
states that exactly. There is no asymmetry in the reporting path.

## C5 — "What if some cells fail?"

**Reviewer:** *"You said you would mark FAILED_EXECUTION rather than silently downgrade to K = 4. Where is that?"*

**Response:** The analysis driver writes a `PARTIAL_FAMILY` row with `delong_p_holm_k5 = "pending_full_family"` for every cell that is missing or only partially populated. No `K5_FULL_FAMILY` row is produced unless all 5 cells have valid archives with the required methods (`rga_meta_router`, `rga_boosted_fusion`, `static_attention`).

If a cell **runs** but writes a malformed archive, the QC tests in `tests/test_phase2_family_a_prediction_archive_complete.py` will fail before the analysis driver gets to it. Failing cells must be remediated; they cannot be silently dropped.

## C6 — "Is A-POWERED-4 honestly handled?"

**Reviewer:** *"VisA RGB+edge has pairing-strength `derived_view_proxy`. That cell is in Family A but cannot support an independent-modality claim. Is the report careful?"*

**Response:** The v2 Family-A report explicitly states in §5:

> "A-POWERED-4 (derived_view_proxy): VisA RGB+edge. The 'edge' modality is a derived view of the RGB image, not an independent modality. Any Δ on A-POWERED-4 cannot support independent-modality generalization claims."

The registry's `pairing_strength` column is the durable record;
`test_family_a_cells_match_locked_identities` ensures it stays as
`derived_view_proxy`.

## C7 — "Did anything Family-D-shaped leak in?"

**Reviewer:** *"Phase 2.2A is Family-A only. Show me that no Family-D file changed."*

**Response:** `tests/test_phase2_family_d_untouched_during_family_a.py` (5 tests) verifies:
- v1 invalidation notice still says `INVALID_FOR_EXECUTION`;
- v2 design status still says `V2_DESIGN_PENDING`;
- no v2 Family-D artefact was created;
- no `run_phase2_family_a_*.py` script imports or references any `family_d` module;
- all 6 v1 Family-D files still exist.

## C8 — "Operational decision."

**Reviewer:** *"Pick one of the three Phase 2.2A decisions."*

**Decision:** **`PASS TO BEGIN FAMILY-B COMPUTE`** — assuming
A-POWERED-2..5 complete with valid archives and the K = 5 Holm row
is correctly populated. **If any cell did not complete in the
session, the decision is automatically `PASS FOR FAMILY-A STATIC-
REFERENCE AUDIT ONLY`** with the failed cell marked
`pending_compute_wallclock` in the remaining-gaps file.

The current run state is recorded at the bottom of this report and
updated as cells complete.

## Current run state

- A-POWERED-1 primary surface: **complete** (recomputed from archive).
- A-POWERED-2 (MVTec 3D-AD held-out): **in progress** (registry-driven 30-seed run); Δ vs static expected to be small or near-zero (held-out category is a transfer protocol where both methods are near chance).
- A-POWERED-3 (MVTec LOCO-AD SP): **pending**.
- A-POWERED-4 (VisA RGB+edge SP): **pending**.
- A-POWERED-5 (UNSW-NB15): **pending**.
- K = 5 Holm: **PARTIAL_FAMILY** until all 5 cells exist.
