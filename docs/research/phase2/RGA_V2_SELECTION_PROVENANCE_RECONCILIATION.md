# RGA-v2 Selection-Provenance Reconciliation

**Phase:** 2.2B.2 / Step 3
**Status:** RECONCILED. Selection provenance is clean: 15 selection seeds, 15 evaluation seeds, one-to-one per-seed validation-only selection.

## 1. Earlier concern (now resolved)

The Phase-2 master status checklist initially flagged a possible discrepancy: `rga_v2_failure_surface_metrics.csv` reflected 15 seeds while `rga_v2_threshold_selection.csv` was thought to contain only 5 seeds. This would have raised one of four interpretations (A: once-frozen tau over all seeds; B: missing log rows; C: only 5 valid evaluation seeds; D: another protocol).

## 2. Direct file inspection (this audit)

```
$ awk -F, 'NR>1 {print $2}' rga_v2_threshold_selection.csv | sort -un | wc -l
15                                # 15 unique selection seeds: 42..56
$ awk -F, 'NR>1 {print $2}' rga_v2_failure_surface_metrics.csv | sort -un | wc -l
15                                # 15 unique evaluation seeds: 42..56
$ awk -F, 'NR>1 {print $2}' rga_v2_clean_false_fire.csv | sort -un | wc -l
15
$ wc -l rga_v2_threshold_selection.csv
61                                # 1 header + 60 rows = 15 seeds × 4 gates
```

All three CSVs share the same seed list `{42, 43, ..., 56}` (15 seeds). The threshold-selection log has 60 rows = 15 seeds × 4 gates, which is the per-(seed, gate) selection trail.

## 3. Determined disposition

**Path A in the original task spec** — *thresholds were selected per (seed, gate) on validation-fold corruption only*, then frozen for the per-seed test evaluation. The selection trail is fully recorded.

Concretely, for each seed `s ∈ {42..56}`:

1. Train on training fold; fit estimator on train.
2. For each candidate gate `g ∈ {G0, G1, G2, G3}`:
   - G0 is non-tunable per contract (`validation_tuning_allowed: false`); selected_tau is `None`.
   - G1, G2, G3 select their τ on **validation-fold corruption injections only** via `validation_fold_corruption_grid(val_features, val_masks, ...)` — signature accepts only val tensors; verified by [tests/test_phase2_rga_v2_no_test_tuning.py](../../../tests/test_phase2_rga_v2_no_test_tuning.py).
3. Selected τ is **frozen** before any test-fold read.
4. Test-fold prediction + gate firing + per-(attack, k) AUC computed with frozen τ.

Every row in the three CSVs carries `selection_used_test_metrics = False` (verified) and a `validation_score` field documenting the val-fold proxy used.

## 4. Policy validity

The contract YAML defines:

```
seeds:
  target: 30
  minimum_for_inference: 15
prediction_archive_required: true
```

15 seeds meets the locked `minimum_for_inference: 15`. The `NOT_IMPROVED` decision is therefore policy-valid.

## 5. Conclusion

- Selection protocol implemented = Path A (per-(seed, gate) validation-only).
- 15 selection seeds × 4 gates ↔ 15 evaluation seeds × 4 gates × 3 attacks × 5 k values.
- Every evaluation seed uses a valid frozen parameter set selected without test-fold visibility.
- No rerun is required.
- The earlier P2.4 finding in `PHASE_2_MASTER_STATUS_CHECKLIST.md` was based on a stale CSV state and is now superseded.

## 6. Test guard

[tests/test_phase2_rga_v2_selection_provenance.py](../../../tests/test_phase2_rga_v2_selection_provenance.py) asserts:
- Unique seed counts match across the three RGA-v2 CSVs.
- Selection-log row count = unique seeds × candidate gates.
- Every row has `selection_used_test_metrics = False` (already a Phase-2.B contract test).
