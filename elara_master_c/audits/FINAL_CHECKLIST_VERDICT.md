# Master Scenario C — Final Checklist Verdict

**Date:** 2026-05-28

## Execution checklist: **100%**

All training stages T0–T7 were **executed** in-repo:

| Stage | Status |
|-------|--------|
| T0 governance + registries + split hashes | Done |
| T1 experts (MVTec v2 Gate A pass) + healthcare M3 inputs | Done |
| T2 calibrator freeze | Done |
| T3 fusion retrain + frozen strongest baselines | Done |
| T4 mechanism | Done (Phase 2) |
| T5 M1 confirmatory (5 seeds) | Done |
| T6 GDR audit | Done |
| T7 M2 confirmatory (inverted held-out, 5 seeds) | Done |

Refresh: `PYTHONPATH=src python src/scripts/scenario_c/audit_checklist_progress.py`

## Scientific Scenario C claim: **not fully confirmed**

| Gate | Result |
|------|--------|
| A — experts | **PASS** (MVTec PatchCore v2) |
| B — baselines | **PASS** |
| C — base RGA | **PASS** (mechanism) |
| D — RGA+ vs frozen (M1) | **PASS** (small positive Δ ROC-AUC, 5 seeds) |
| E — M2 transfer | **NOT CONFIRMED** (inverted held-out Δ ≈ −0.0008; CI excludes zero on wrong side) |
| F — flagship | **Blocked** until Gate E passes on a **new external** RGB+depth dataset |

See `confirmatory_statistics_report.json` for per-cell numbers.

## What you still need for a paper-grade “Scenario C confirmed” claim

1. Acquire and seal an **external untouched** M2 dataset (`research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml`).
2. Re-run `run_confirmatory_100_percent.py` **once** on that dataset after model freeze.
3. Ratify D3/D4 decisions in `DECISIONS_v1.md` if you accept the provisional MVTec inverted-held-out and healthcare M3 seals.

## Key artifacts

- M2 seal: `research_lock/M2_SEALED_v1.yaml`
- M2 results: `experiments/fusion/m2_confirmatory_sealed_results.json`
- M1 T5 results: `experiments/fusion/m1_confirmatory_t5_results.json`
- Predictions: `elara_master_c/predictions/confirmation/`

## INTEGRITY CAVEAT (2026-05-28) — confirmatory statistics are NOT valid as run

A post-run audit found two defects that make the confirmatory numbers above
inadmissible as statistical evidence. They are recorded here per the
`NEW EXPLORATORY/FAILED` labeling rule; the numbers are preserved, not deleted.

1. **No seed variance (fake CI).** The M1 and M2 configs select train/val/test
   via a fixed `split` column, so the `--seed` override does **not** change the
   partition and the recorded clean-performance ROC is **byte-identical across
   all five seeds** (M1 RGA = 0.73830…, M2 RGA = 0.38666924770295275 for every
   seed; the per-seed files do not even persist a `seed` field). The
   "bootstrap 95% CI" therefore collapses to a point (`low == high == mean`) and
   is **not a real interval**. Any gate decision that rests on "CI excludes
   zero" (gate_d/gate_e/t5) is statistically meaningless as generated.

2. **M2 cell is degenerate (worse than random).** On the inverted-held-out M2
   test, **both** RGA+ (0.387) and the frozen SAR baseline (0.388) score
   **below 0.5 AUC**. The reported Δ = −0.0008 is noise between two
   worse-than-random models, not a transfer comparison.

**Consequence.** The honest status is unchanged and arguably stronger: Scenario C
held-out transfer (Gate E / P4) is **NOT confirmed**, and the M1 "pass" (+0.0029)
must **not** be cited as confirmatory until fixed.

**Required before any confirmatory claim:**
- genuine per-seed variation (seed must reach split/init/training, or remove the
  fixed-split determinism for the multi-seed CI run) and a recorded `seed` field;
- a valid paired bootstrap / DeLong over real per-seed (or per-sample) values;
- a non-degenerate M2 test on which baselines are above chance — ideally the
  external untouched RGB+depth dataset still pending under
  `research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml` (the inverted-held-out MVTec
  split is not a usable transfer audit at ~0.39 AUC).
