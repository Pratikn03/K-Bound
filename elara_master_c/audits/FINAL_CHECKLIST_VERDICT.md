# Master Scenario C — Final Checklist Verdict

**Last updated:** 2026-05-29 (full pipeline + external M2 paired inference)

## Execution checklist: **100%**

All training stages T0–T7 were **executed** in-repo. The latest orchestrated run:

`src/scripts/scenario_c/run_full_master_c_pipeline.py` → manifest
`elara_master_c/audits/stage_runs/full_pipeline_20260529T033906Z.json` (exit **2** = gates not passed, not a crash).

| Stage | Status |
|-------|--------|
| T0 governance + registries + split hashes | Done |
| T1 experts + Gate A | Done (Gate A **FAIL** on canonical depth/RGB AUC ~0.58 — see report) |
| T2 calibrator freeze | Done |
| T3 fusion (M0/M1) | Skipped retrain in last run (artifacts exist); prior runs complete |
| T4 mechanism (B-MECH-1) | **Not re-run** in last pipeline (missing `--experiment-id` in that invocation; run manually if needed) |
| T5 powered pilot | **Done** — 30 seeds (42–71), see below |
| T6 GDR audit | **Done** — 4/4 scenarios pass |
| T7 M2 external (3D-ADAM) | **Done** — one-shot 5-seed fusion + per-sample paired inference |
| T7 M2 proxy (inverted MVTec) | Prior run only (below chance; not admissible) |

Refresh: `PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/audit_checklist_progress.py`

**Checklist score:** **34/38** (89.5%) — execution complete; confirmatory superiority/transfer gates not passed.

---

## T5 powered pilot (development, 2026-05-29)

Family A / MVTec supervised-paired development path (`run_phase2_powered_audited_pilot.py`, 30 seeds):

| Summary | Value |
|---------|--------|
| Seeds | 42–71 (30 total) |
| Typical `chosen_test_auc` | **0.737–0.752** (boost or router head) |
| Archive | Phase-2 prediction index updated |

**Bound:** This is **development / pilot** evidence on MVTec, **not** the external M2 confirmatory audit. Do not cite pilot AUC as 3D-ADAM transfer success.

---

## M2 external — 3D-ADAM category-held-out (authoritative Gate E audit)

**Seal:** `research_lock/M2_EXTERNAL_SEALED_v1.yaml`  
**Inputs:** `experiments/fusion/m2_external_3d_adam_sealed_inputs.csv` (2732 samples, 5464 rows, 23 categories)  
**Fusion results:** `experiments/fusion/m2_external_3d_adam_confirmatory_results.json`  
**Paired inference:** `experiments/fusion/m2_external_3d_adam_paired_inference.json`  
**Predictions:** `elara_master_c/predictions/confirmation/` (`M2-EXTERNAL-3D-ADAM`, n_test=**1378**)

### Primary comparison: RGA+ vs frozen SAR

| Metric | RGA+ | SAR (frozen) |
|--------|------|----------------|
| Ensemble test ROC-AUC | **0.508** | **0.546** |
| Δ (RGA+ − SAR) | **−0.038** | |
| DeLong p (Holm) | **0.003** | SAR significantly better |
| Paired bootstrap 95% CI on Δ | **[−0.062, −0.013]** | Excludes 0 (harm to RGA+) |
| `cell_valid` | **true** | Per-sample paired ensemble inference |
| `transfer_confirmed` | **false** | |

**Label:** `NEW CONFIRMATORY / NOT CONFIRMED` for P4 held-out transfer on 3D-ADAM.

RGA+ and SAR scores are **deterministic across fusion seeds** (identical per-seed AUC); inference uses **seed-averaged ensemble predictions** over 1378 held-out test rows (DeLong + 10k paired bootstrap). Static attention **varies by seed** (0.50–0.60); ensemble static vs RGA+ Δ ≈ **−0.070** (also significant harm).

### Secondary: M2 proxy (inverted MVTec) — not admissible

| Metric | Value |
|--------|--------|
| RGA+ / SAR AUC | ~0.387 / ~0.388 |
| Status | **Below chance**; `cell_valid: false` — exploratory proxy only |

---

## Scientific Scenario C claim: **not fully confirmed**

| Gate | Result |
|------|--------|
| A — experts | **FAIL** in latest qualification (RGB/depth ~0.58); prior v2 pass may still exist in archive |
| B — baselines | **PASS** |
| C — base RGA | **PASS** (mechanism / Phase 2) |
| D — RGA+ vs frozen SAR (M1) | **NOT CONFIRMED** (Δ ≈ +0.0029 but degenerate seed CI; `cell_valid: false`) |
| E — M2 transfer (3D-ADAM external) | **NOT CONFIRMED** (valid paired stats: SAR beats RGA+, p=0.003) |
| E — M2 proxy (inverted MVTec) | **NOT CONFIRMED** (below chance) |
| F — flagship Scenario C | **Blocked** — Gate E failed on sealed external benchmark |

See `elara_master_c/audits/confirmatory_statistics_report.json` for full JSON.

**Admissible one-sentence claim today:**

> ELARA/RGA+ shows bounded mechanism gains under controlled collapse and a coherence-certified gate rule, but **does not** confirm held-out superiority over the frozen SAR baseline on the external 3D-ADAM category-held-out audit (Δ ≈ −0.038, paired p = 0.003).

---

## What is still open (not blocking honest thesis)

1. **M1 confirmatory admissibility** — run per-sample paired inference for MVTec M1 (same path as M2 external) if citing RGA+ vs SAR on M1.
2. **T4 B-MECH-1** — re-run if you need a fresh mechanism replication in the pipeline manifest:
   ```bash
   PYTHONPATH=.:src .venv/bin/python src/scripts/run_phase2_mechanism_replication.py \
     --experiment-id B-MECH-1 --seeds 30 --seed-start 42
   ```
3. **Optional `--force-fusion`** — retrain M0/M1 through `run_full_master_c_pipeline.py` if you need new fusion checkpoints.
4. **Thesis/PDF** — rebuild after table emits: `bash scripts/rebuild_paper.sh`

**No further external RGB+depth download is required for Gate E** — 3D-ADAM was acquired, sealed, and evaluated.

---

## Key artifacts

| Artifact | Path |
|----------|------|
| M2 external seal | `research_lock/M2_EXTERNAL_SEALED_v1.yaml` |
| M2 external confirmatory | `experiments/fusion/m2_external_3d_adam_confirmatory_results.json` |
| M2 paired inference | `experiments/fusion/m2_external_3d_adam_paired_inference.json` |
| M2 proxy (legacy) | `experiments/fusion/m2_confirmatory_sealed_results.json` |
| M1 confirmatory | `experiments/fusion/m1_confirmatory_t5_results.json` |
| Confirmatory report | `elara_master_c/audits/confirmatory_statistics_report.json` |
| GDR audit | `experiments/fusion/gate_decision_rule_e2e_audit.json` |
| Full pipeline manifest | `elara_master_c/audits/stage_runs/full_pipeline_20260529T033906Z.json` |
| Predictions (M2 external) | `elara_master_c/predictions/confirmation/` |

---

## INTEGRITY CAVEAT (2026-05-28) — seed-level CI on deterministic fusion methods

For **RGA+ / SAR / score adapters**, clean-test ROC-AUC is often **identical across fusion seeds** because the fusion head is deterministic given fixed train/val/test splits. A **seed-bootstrap CI** on per-seed scalar AUCs then collapses to a point (`cell_valid: false` for M1 and for seed-only M2 rows).

**Mitigation (implemented 2026-05-29 for M2 external):** `confirmatory_statistics.py` runs **per-sample paired DeLong + paired bootstrap** on archived test predictions (`inference_mode: per_sample_paired_ensemble`). M2 external is **`cell_valid: true`** with that path.

**M2 proxy (inverted MVTec)** remains invalid: both methods **below 0.5 AUC** (~0.39).

**M1** still uses seed-only rows in the report until the same paired path is applied to M1 archives.

---

## ENFORCED IN CODE (2026-05-28 / 2026-05-29)

`confirmatory_statistics.py` forces `gate_d` / `gate_e` / `t5` to **fail** when:

- **degenerate seed CI** — `bootstrap_ci_width < 1e-9` or `seed_variance < 1e-12` *and* no valid per-sample paired inference; and/or
- **below-chance** — `min(mean_rga_auc, mean_base_auc) < 0.5`.

Gate E on **3D-ADAM** uses paired inference: transfer requires positive Δ with valid inference and `transfer_confirmed: true` — currently **false**.

`audit_checklist_progress.py` derives `gate_d` only from the confirmatory report (development `gate_bd_evaluation.json` cannot mask failures).

Regression: `tests/test_confirmatory_validity_guard.py`

---

## Theorem stack (2026-05-29 pipeline)

| Item | Status |
|------|--------|
| `validate_theorem_stack.py` | **all_ok: true** |
| GDR E2E | **4/4** pass |
| LaTeX tables | Regenerated (T4, T6 KS, T7 PAC, GDR, theory mapping) |

Theory supports **when to switch / abstain**; it does not overturn the external M2 negative result.
