# statistical_policy_v1.md (frozen)

The Scenario C statistics policy. It **adopts** the existing audited policy and
adds the freeze rule. It must not change after results appear.

## Adopted authoritative policy

- `docs/research/phase2/PHASE_2_STATISTICAL_POLICY_v2.md`
- `docs/research/audit/STATISTICAL_ANALYSIS_POLICY.md`

## Frozen rules

1. **Threshold / model selection**: validation-only. No test-driven tuning.
2. **Primary endpoint**: one pre-registered metric per family
   (`primary_endpoints_v1.yaml`); decided before any final test evaluation.
3. **Paired comparison**: DeLong paired test for ROC-AUC where assumptions hold;
   otherwise paired bootstrap. Report the test used.
4. **Confidence intervals**: 95% bootstrap CI on the paired delta. A confirmatory
   claim requires the CI to exclude zero.
5. **Multiplicity**: Holm-Bonferroni within each confirmatory endpoint family;
   record the family size k used.
6. **Effect size**: report magnitude + band (small/moderate/large), never a
   p-value alone.
7. **Calibration**: ECE and Brier with uncertainty, under clean and transfer.
8. **False-fire**: report a CI, not just an observed 0.000.
9. **Ensemble vs per-seed**: the primary delta is the pre-declared one (ensemble
   ΔAUC for Family-style cells); the other is secondary and labeled as such.
10. **Negative results**: preserved and reported (label `FAILED`); never deleted.

## Reproducibility requirement (gates confirmatory status)

A `NEW CONFIRMATORY` result is admissible only if it archives:
per-sample predictions for every seed and model (`PredictionArchive`), the
immutable split hash, the validation selection record (pre-test), the config,
hyperparameters, and seeds. Legacy artifacts that lack raw per-seed predictions
cannot back a confirmatory paired ensemble-level claim; they must be re-run with
logging enabled (`src/elara/evaluation/prediction_archive.py`).

## Change control

Any modification creates `statistical_policy_v2.md` with a dated rationale; this
file is never edited in place.
