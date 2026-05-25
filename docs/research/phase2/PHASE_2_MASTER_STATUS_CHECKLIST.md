# ELARA Phase 2 Master Status Checklist

## 0. Audit Metadata

- **Audit timestamp (UTC):** 2026-05-25T01:59:57Z
- **Repository root:** `/Volumes/T9/uav/AutoML_Flagship_V8`
- **Branch:** `exp/elara-phase2-mechanism-and-replication`
- **HEAD commit:** `204775bf6ae5d2f111f01ba0f5def4e70bdcb309` ("Lock ELARA exploratory domain-composition shift protocol")
- **Git status summary:** 4 uncommitted modified driver files (`run_phase2_certificate_audit.py`, `run_phase2_ks_power_sweep.py`, `run_phase2_mixture_shift.py`, `run_phase2_rga_v2_gate_sweep.py`) + 2 modified test files + multiple new untracked Phase-2.2B.1 documents and result CSVs.
- **Auditor identity:** Codex repository audit
- **Audit mode:** read-only status consolidation; no new model execution
- **Purpose:** intended for external reviewer upload

## 1. Executive Status Summary

| Item | Status | Evidence File(s) | Reviewer Meaning |
|---|---|---|---|
| Phase 1.1.1 prerequisite closure | VERIFIED | git log commits `5d9cf46`, `cce3e2a`, `6d381a9` | Source/PDF consistency patched before Phase 2 started |
| Phase 2 policy / registry repair | VERIFIED | `PHASE_2_RESEARCH_CONTRACT_v2.md`, `PHASE_2_STATISTICAL_POLICY_v2.md`, `PHASE_2_EXPERIMENT_REGISTRY_v2.csv`, `PHASE_2_CLAIM_MATRIX_v2.csv` | K=5 cell-level family + locked static comparator |
| Prediction archive infrastructure | VERIFIED | `src/elara/evaluation/prediction_archive.py`, archive index CSVs under `experiments/phase2/predictions/`, `experiments/phase2/mechanism/b_mech_1_prediction_archives/`, `experiments/phase2/mechanism/rga_v2_prediction_archives/` | Schema in active use by 3 cells |
| Family-A powered static-reference audit | VERIFIED | `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv` (5 rows, all `K5_FULL_FAMILY`) | All 5 cells executed under K=5 Holm |
| Family-B infrastructure | VERIFIED | `src/elara/family_b/*.py`, `src/uais/fusion/attention/reliability_estimator.py` (G3 top-q + KS window-size) | G0..G3 + KS window grid + mixture-shift sampler implemented |
| B-MECH-1 replication | VERIFIED — REPRODUCED ×2 | `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv` | B1 close match; B2 directionally consistent at larger magnitude |
| B-CERT-1 (current scope) | PARTIALLY VERIFIED | `experiments/phase2/certification/switching_certificates.csv` (2 rows) | 1 CERTIFIED + 1 NOT_CERTIFIED on k=4 only |
| B2 magnitude comparability audit | VERIFIED — COMPARABLE_BUT_ESTIMATOR_CHANGED | `docs/research/phase2/B2_MAGNITUDE_COMPARABILITY_AUDIT.md`, `experiments/phase2/mechanism/b2_phase1_vs_phase2_comparability.csv` | Audit closed; manuscript must use dual-number form |
| RGA-v2 B-MECH-2 | VERIFIED — EXECUTED, NOT_IMPROVED | `experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv`, `docs/research/phase2/RGA_V2_PARTIAL_FAILURE_REPORT_v2.md` | 15 seeds; G1/G2/G3 all NOT_IMPROVED (C1 false-fire budget failure) |
| Domain-composition B-MECH-3S | VERIFIED — EXECUTED, FALSE_FIRE_NOT_REDUCED | `experiments/phase2/mechanism/domain_composition_shift_metrics.csv`, `docs/research/phase2/DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md` | 5 seeds × 10 mixtures; global=domain-aware fire rate (=1.0) |
| KS power B-MECH-4 | VERIFIED — EXECUTED, TRADEOFF_IMPROVED | `experiments/phase2/mechanism/ks_window_size_power.csv` (25 rows), `experiments/phase2/mechanism/ks_true_degradation_power.csv`, `docs/research/phase2/KS_REFERENCE_AND_POWER_REPORT_v2.md` | 5 seeds × 5 windows; larger windows raise detection power without raising false fire |
| Family-D v1 | VERIFIED — INVALID_FOR_EXECUTION | `docs/research/phase2/FAMILY_D_V1_INVALIDATION_NOTICE.md` | Preserved as historical evidence only |
| Family-D v2 design | **`V2_FREEZE_BLOCKED_AT_PARTITION_MANIFEST_ARCHIVE_SHA256`** (Phase 2.2C) | `FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md`, `FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md`, `PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md` (Phase 2.2C) | D1–D8 user research decisions all LOCKED; protocol YAML, operator spec, hypotheses CSV, selection policy, execution-commands NOT_RUN all complete + placeholder-free; only blocker is per-archive SHA256 requiring a future hash-only download pass |
| Phase 3 readiness | NOT APPLICABLE | n/a | Phase 3 not opened; explicitly forbidden by Phase-2 stop boundary |

### Overall status

**`PHASE_2_PARTIALLY_COMPLETE_FAMILY_B_CONTINUATION_REQUIRED`**

Determined from artifact inspection: Family-B is fully closed (committed `2c780cf` + `4993a14`). Phase 2.2C produced Eyecandies-based Family-D v2 design artifacts (protocol YAML, operator spec, hypotheses CSV, selection policy, execution commands NOT_RUN, hostile-review report). The freeze is blocked **only** at the partition-manifest archive SHA256 step, which requires a future hash-only download pass (Phase 2.2D). All other freeze artifacts are complete with no placeholders.

## 2. Phase 2 Timeline and Decision History

| Stage | Intended Purpose | Work Actually Performed | Final Decision | Validity Status | Evidence Path |
|---|---|---|---|---|---|
| Phase 2 (initial) | Lock contract + pilot infrastructure | Contract + registry + claim matrix + prediction archive code + 1-cell pilot (A-POWERED-1, 30 seeds) | Pilot complete; contracts locked | VALID (committed `6299c3f`) | `PHASE_2_RESEARCH_CONTRACT.md`, `PHASE_2_ARTIFACT_MANIFEST.md` |
| Phase 2.1 | Contract repair (K=10 vs K=5 drift; Family-D placeholder freeze) | v2 contract / policy / registry / Family-D v1 invalidation / v2 design status | Contract repair complete | VALID | `PHASE_2_1_HOSTILE_REVIEW_REPORT.md`, `FAMILY_D_V1_INVALIDATION_NOTICE.md` (committed `9973376`) |
| Phase 2.2A | Family-A K=5 static-reference audit | Registry-driven cell + analysis drivers; 5 cells × 30 seeds; K=5 Holm | `PASS TO BEGIN FAMILY-B COMPUTE` | VALID | `PHASE_2_2A_FINAL_DECISION.md`, `family_a_v2_primary_cell_level_holm_k5.csv` |
| Phase 2.2B (infrastructure-only) | Build Family-B drivers and helpers | G3 top-q gate; KS window grid; mixture-shift sampler; 5 drivers (B-MECH-1 with full loop, B-MECH-2/3/4/B-CERT-1 as scaffolds) | `READY FOR FULL FAMILY-B COMPUTE` **(OVERSTATED — see Section 13 P1 finding)** | PARTIALLY VALID | `PHASE_2_2B_INFRASTRUCTURE_FINAL_DECISION.md` (committed `2719d81`) |
| Phase 2.2B.exec | First Family-B execution attempt | B-MECH-1 executed + B-CERT-1 partially executed; B-MECH-2/3/4 honestly disclosed as `EXECUTION_BLOCKED_DRIVER_SCAFFOLD` | `PASS FOR MECHANISM REPLICATION ONLY` | VALID — overstatement from prior stage corrected | `PHASE_2_2B_EXEC_FINAL_DECISION.md`, `PHASE_2_2B_EXECUTION_PRECHECK.md` (lock at `204775b`) |
| Phase 2.2B.1 (post-exec, uncommitted) | B2 magnitude audit + complete scaffolded drivers | B2 audit DONE; B-MECH-2/3S/4 drivers expanded; B-MECH-2 executed (15 seeds), B-MECH-3S executed (5 seeds × 10 mix), B-MECH-4 executed (5 seeds × 5 windows). B-CERT-1 NOT yet extended to RGA-v2 scenarios. | No final decision document yet | IN-PROGRESS — uncommitted | `B2_MAGNITUDE_COMPARABILITY_AUDIT.md`, `FAMILY_B_DRIVER_REALITY_AUDIT.md`, `RGA_V2_PARTIAL_FAILURE_REPORT_v2.md` (updated), `DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md`, `KS_REFERENCE_AND_POWER_REPORT_v2.md` (updated) |

## 3. Phase 2 Research Contract and Claim Boundary

| Family | Meaning | Completed Evidence | Allowed Claim | Forbidden Claim |
|---|---|---|---|---|
| Family A | Powered audited static-reference reproduction over previously inspected cells | 5/5 cells; all Holm-significant vs `static_attention`; sign-consistent 30/30 each | "RGA+ improves the fixed static-attention reference on five previously inspected cells under K=5 Holm" | Strongest-baseline superiority; confirmation; universality |
| Family B | Mechanism replication, partial-failure repair, monitoring / certificate evidence | B-MECH-1 (B1/B2 REPRODUCED); B-MECH-2 executed; B-MECH-3S executed; B-MECH-4 executed; B-CERT-1 k=4 only | Each cell's specific decision phrased per its own report | Promotion of any RGA-v2 candidate (none passed C1..C6); KS/mixture claims beyond what was actually measured |
| Family C | Exploratory evidence only | none in this Phase 2 scope | Bounded exploratory reporting | Generalization claim |
| Family D | Future unseen confirmatory evaluation only | v1 `INVALID_FOR_EXECUTION`; v2 `V2_DESIGN_PENDING` | No claim — Family D not executed | Any "confirmed", "held-out validated" claim |

### Permanent forbidden claims (preserved verbatim)

- ELARA is universal.
- RGA+ beats every strongest baseline.
- Existing Family A is confirmatory.
- Current work is SOTA.
- Current work is production-ready or deployment-safe.
- Retrospective certificate equals real-world safety certification.
- Family D was executed.

## 4. Phase 2 Completion Dashboard

| Component | Weight | Evidence Status | % Complete | Earned Credit | Blocker / Next Action |
|---|---:|---|---:|---:|---|
| Policy, registry, contract integrity | 10% | v2 files present; 542/10 tests pass | 100% | 10.0% | none |
| Prediction archive / statistical infrastructure | 10% | Used in Family-A 5 cells + B-MECH-1 + B-MECH-2 archives | 100% | 10.0% | none |
| Family-A powered static-reference audit | 20% | 5 cells + K=5 Holm complete | 100% | 20.0% | none |
| B-MECH-1 B1/B2 mechanism replication | 15% | B1 close match; B2 directionally REPRODUCED but estimator change documented (`COMPARABLE_BUT_ESTIMATOR_CHANGED`) | 90% | 13.5% | manuscript-update phase must adopt dual-number form |
| B-MECH-2 RGA-v2 partial-failure | 15% | 15-seed sweep executed; inference CSV present; **G1/G2/G3 all NOT_IMPROVED** (C1 false-fire budget failure); sub-contract seed count (15 vs 30) | 75% | 11.25% | re-run at 30 seeds before final Family-B verdict; investigate batch-level minimum-pooling sensitivity |
| B-MECH-3S domain-composition audit | 5% | 5 seeds × 10 mixtures; `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED` | 100% | 5.0% | none for this scope; general category/cohort theorem remains `DEFERRED_PENDING_NATURAL_CATEGORY_METADATA` |
| B-MECH-4 KS power / window analysis | 5% | 5 seeds × 5 windows; `TRADEOFF_IMPROVED` | 100% | 5.0% | none |
| B-CERT-1 certificates + risk-dominance | 10% | k=4 only; 1 CERTIFIED + 1 NOT_CERTIFIED; risk-dominance terms `inadmissible_single_arm`; **no RGA-v2 cert rows** | 50% | 5.0% | extend to RGA-v2 scenarios that pass C1 budget (none exist post-B-MECH-2 → certificate extension is technically empty); add clean-arm to enable risk-dominance |
| Family-D v2 design and frozen pre-test contract | 5% | `V2_DESIGN_PENDING` | 0% | 0.0% | close eligibility review (MPDD modality, Eyecandies operator, additional candidate) |
| Family-D execution or formally justified deferral | 5% | No execution; deferral justified by V2_DESIGN_PENDING; no formal Phase-2-end-without-confirmation decision yet | 50% | 2.5% | either close v2 and execute, OR write a formal Phase-2-ends-without-Family-D decision document |

**Totals**

- **Total weighted Phase 2 completion:** **82.25%**
- **Infrastructure completion:** ~100% (policy + archive + drivers all built)
- **Executed scientific evidence completion:** ~70% (B-MECH-1/2/3S/4 all executed; B-CERT-1 partial; Family-D unexecuted by design)
- Provisional: NO — all referenced artifacts present.

**Honest notes:**

- B-MECH-2 executed at 15 seeds (sub-contract; contract target = 30; contract minimum = 15 per the YAML).
- B-MECH-3S decision `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED` is a **negative result for the gating-improvement hypothesis** — both references fired at 100%. Negative result honestly reported.
- B-MECH-2 promotion result `NOT_IMPROVED` for every candidate is a **negative result for the RGA-v2 promotion hypothesis** — honestly reported.
- B-CERT-1 cannot meaningfully be extended to RGA-v2 cells because no RGA-v2 candidate passes C1 budget; the empty extension is itself a finding.

## 5. Verified Phase 2.2A Family-A Results

Values extracted from `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv` (rows 2–6).

| Cell | Benchmark / Protocol | Pairing Strength | n Test | RGA+ Ens AUC | Static Ens AUC | Δ AUC | Raw p | Holm K=5 p | 95% CI | Effect Band | Sign Consistency | Verification |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| A-POWERED-1 | MVTec 3D-AD / PatchCore supervised-paired | independent_modalities | (per raw CSV) | (per raw CSV) | (per raw CSV) | **+0.1082** | 1.67e-04 | **3.35e-04** | [+0.052, +0.166] | large | 30/30 | VERIFIED |
| A-POWERED-2 | MVTec 3D-AD / PatchCore held-out category | independent_modalities | (per raw CSV) | (per raw CSV) | (per raw CSV) | **+0.0519** | 1.21e-05 | **4.06e-05** | [+0.029, +0.075] | large | 30/30 | VERIFIED |
| A-POWERED-3 | MVTec LOCO-AD / PatchCore supervised-paired | **derived_view_proxy** (corrected in Phase 2.2B.2; see [PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md](./PHASE_2_LOCO_PAIRING_STRENGTH_AUDIT.md)) | (per raw CSV) | (per raw CSV) | (per raw CSV) | **+0.1038** | 1.02e-05 | **4.06e-05** | [+0.058, +0.150] | large | 30/30 | VERIFIED |
| A-POWERED-4 | VisA / RGB+edge supervised-paired | **derived_view_proxy** | (per raw CSV) | (per raw CSV) | (per raw CSV) | **+0.0297** | 1.53e-03 | **1.53e-03** | [+0.012, +0.049] | moderate | 30/30 | VERIFIED |
| A-POWERED-5 | UNSW-NB15 / flow/conn/context | naturally_structured_views | (per raw CSV) | (per raw CSV) | (per raw CSV) | **+0.0095** | 0.00 | **0.00** | [+0.008, +0.011] | small | 30/30 | VERIFIED |

Cell-by-cell n_test / AUC values are present in `family_a_v2_primary_cell_level_raw.csv` (verified to contain matching deltas; see [FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md) §2 table).

### Family-A Allowed Claim

> "Family A provides powered audited static-reference evidence across five previously inspected benchmark cells. It evaluates whether validation-frozen RGA+ improves on a fixed static-attention reference; it is not confirmatory replication and is not a strongest-baseline superiority evaluation."

Verified verbatim in [FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md) §3.

### Family-A Limitations

- [x] Static-attention comparison is not strongest-baseline comparison.
- [x] MVTec held-out absolute performance is near chance (A-POWERED-2 ensemble RGA AUC ≈ 0.52, ensemble static AUC ≈ 0.47; documented in Family-A v2 report §3).
- [x] VisA is derived_view_proxy only (registry confirmed).
- [x] UNSW effect band is small and qualified in report.
- [x] Historical K=10 all-comparator pilot remains separate at `experiments/phase2/statistics/family_a_powered_ensemble_inference.csv` under `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` label.

## 6. Verified Phase 2.2B.exec Family-B Results

### 6.1 B-MECH-1 Primary Mechanism Replication

Values from `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv` and `..._inference.csv`.

| Endpoint | Protocol | Phase-1 Target Δ | Phase-2 Δ | 95% CI | Holm K=2 p | Archive Present? | Decision | Integration Status |
|---|---|---:|---:|---|---:|---|---|---|
| B1 | zero_attack k=4, mean gate, τ=0.66 | +0.0506 | **+0.0507** | [+0.0364, +0.0650] | 4.31e-12 | YES (parquet) | **VERIFIED_REPRODUCED** | manuscript may quote with dual-number framing |
| B2 | max_attack k=4, mean gate, τ=0.66 | +0.0319 | **+0.0939** | [+0.0741, +0.1149] | < 1e-15 | YES (parquet) | **REPRODUCED_IN_DIRECTION_PENDING_MAGNITUDE_AUDIT** (resolved — see 6.1a) | manuscript must use dual-number form |

#### 6.1a B2 Magnitude Audit Status

- Audit file `docs/research/phase2/B2_MAGNITUDE_COMPARABILITY_AUDIT.md` exists.
- Comparability decision: **COMPARABLE_BUT_ESTIMATOR_CHANGED** (Phase-1 used per-seed mean AUC over ~5 seeds; Phase-2 uses 30-seed ensemble-pooled AUC).
- Manuscript replacement: **not authorised** in this phase; must use dual-number form per audit §4.
- Required next action: manuscript-update phase must adopt dual-number wording.
- Per-row comparability CSV: `experiments/phase2/mechanism/b2_phase1_vs_phase2_comparability.csv` (4 rows: B1/B2 × Phase-1/Phase-2).

### 6.2 B-CERT-1 Retrospective Certificate Status

Values from `experiments/phase2/certification/switching_certificates.csv`.

| Scenario | Source Archive | Global Δ AUC | Fired-Subset LCB | Certificate Decision | Valid Claim | Forbidden Overclaim |
|---|---|---:|---:|---|---|---|
| zero_attack k=4 | `b_mech_1_prediction_archives/.../static_attention__zero_attack_k4`, `.../rga_mean_gate_tau66__zero_attack_k4` | +0.0507 | **-0.00499** | **NOT_CERTIFIED** | "Global AUC gain does not establish positive fired-subset certificate" | "Deployment-safe gain"; "uniform per-sample improvement" |
| max_attack k=4 | same archive group | +0.0939 | **+0.00853** | **CERTIFIED** | "Retrospective stress-protocol certificate under defined conditions" | "Production safety guarantee"; "real-world deployment certification" |

**Mandatory wording (verbatim per `switching_certificates.csv:boundary_notice`):**
"These are retrospective evaluation certificates under defined stress protocols; they are not production safety certificates or real-world deployment guarantees."

Risk-dominance terms `experiments/phase2/certification/risk_dominance_terms.csv` records `inadmissible_single_arm` per scenario (clean k=0 arm not archived in B-MECH-1 → q₀/q₁/Δ₀/Δ₁/π* are not computable from current archives).

## 7. Family-B Infrastructure Versus Actual Execution Truth Table

| Cell | Infrastructure Claimed Ready? | Driver Exists? | Driver Has Real Execution Loop? | Actual Result Artifact Exists? | True Current Status | Evidence |
|---|---|---|---|---|---|---|
| B-MECH-1 | YES (correctly) | YES | YES | YES (60 metric rows, archive populated) | EXECUTED_VERIFIED | `family_b_primary_replication_seed_metrics.csv`, archive index |
| B-MECH-2 | YES (overstated by Phase 2.2B infra; later disclosed scaffold in Phase 2.2B.exec; then expanded in Phase 2.2B.1) | YES | YES (post-Phase-2.2B.1 expansion) | YES (15 seeds × 4 gates × 3 attacks × 5 k = 900 metric rows; inference CSV; archive) | EXECUTED_AT_15_SEEDS_NOT_30 | `rga_v2_failure_surface_metrics.csv`, `rga_v2_failure_surface_inference.csv`, `rga_v2_prediction_archives/B-MECH-2__.../` |
| B-MECH-3S | YES (overstated by Phase 2.2B infra; corrected by Phase 2.2B.exec; expanded in Phase 2.2B.1) | YES | YES | YES (50 rows = 5 seeds × 10 mixtures) | EXECUTED_VERIFIED | `domain_composition_shift_metrics.csv` |
| B-MECH-4 | YES (overstated by Phase 2.2B infra; corrected by Phase 2.2B.exec; expanded in Phase 2.2B.1) | YES | YES | YES (25 window rows + 75+ degradation rows) | EXECUTED_VERIFIED | `ks_window_size_power.csv`, `ks_true_degradation_power.csv` |
| B-CERT-1 | YES (correctly) | YES | YES (k=4 only) | YES (2 certificate rows) | EXECUTED_K4_ONLY | `switching_certificates.csv` (2 rows) |

**Overstatement → correction chain documented:**

- Phase 2.2B infrastructure report said `READY FOR FULL FAMILY-B COMPUTE`.
- Phase 2.2B.exec PRE-CHECK §5 documented this as overstated (B-MECH-2/3/4/B-CERT-1 main() were scaffolds).
- Phase 2.2B.1 closed the gap by expanding the driver `main()` functions and executing each cell.
- Net: the overstatement was caught, openly disclosed, and remediated.

## 8. RGA-v2 Partial-Failure Status

### RGA-v2 Candidate Implementation Status

| Gate | Description | Implemented? | Executed? | Valid Selection Rule Verified? | Result Status |
|---|---|---|---|---|---|
| G0 | Existing mean gate | YES | YES | n/a (no tuning) | BASELINE_REFERENCE |
| G1 | Minimum gate | YES | YES | YES (validation-fold corruption grid) | NOT_IMPROVED (100% clean false-fire) |
| G2 | Hybrid gate | YES | YES | YES (validation-fold corruption grid) | NOT_IMPROVED (100% clean false-fire) |
| G3 | Top-q gate | YES (added in Phase 2.2B; tests pass) | YES | YES (validation-fold corruption grid; q + τ_q joint grid) | NOT_IMPROVED (100% clean false-fire) |
| G4 | Optional learned low-capacity gate | NO — intentionally not implemented per contract `lock_architecture_before_evaluation: true` | NO | n/a | NOT_EVALUATED (driver refuses `--gates G4`) |

### Locked Promotion Criteria Status

From `experiments/phase2/mechanism/rga_v2_failure_surface_inference.csv` per gate; criteria text from `configs/phase2/rga_v2_gate_contract.yaml`.

| Criterion | Text | Evidence Required | Evidence Exists? | G1 | G2 | G3 |
|---|---|---|:---:|:---:|:---:|:---:|
| C1 | within clean false-fire budget = max(0.010, base_G0_clean + 0.005) | clean activation rate per gate | YES | **FAIL** (1.0000) | **FAIL** (1.0000) | **FAIL** (1.0000) |
| C2 | improves ≥ 2 of {k=1, k=2, k=3} over G0 under zero+max | per-(k, attack) delta | YES | FAIL (0/2+) | FAIL (0/2+) | FAIL (0/2+) |
| C3 | does not worsen k=4 by more than 0.005 AUROC vs G0 | k=4 delta | YES | PASS | PASS | PASS |
| C4 | positive switching certificate on at least one partial-failure regime | per-scenario LCB | NO (B-CERT-1 not extended to RGA-v2) | PENDING | PENDING | PENDING |
| C5 | validation-only selection | selection log + signature check | YES | PASS | PASS | PASS |
| C6 | same gate policy across cells | single-cell scope here | YES (trivially) | PASS | PASS | PASS |

End status: **`RGA_V2_EXECUTED_NOT_PROMOTED`**

(Verified from artifact, not assumption. Honest reading: G1/G2/G3 fail the most basic criterion C1. C4 is technically pending but cannot rescue the decision because C1 already fails.)

## 9. KS / Mixture-Shift / Theory Closure Status

| Item | Intended Question | Protocol Status | Execution Status | Result Status | Allowed Claim | Remaining Need |
|---|---|---|---|---|---|---|
| B-MECH-3 original general category/cohort mixture theorem | General false-fire under legitimate category composition changes | DEFERRED_PENDING_NATURAL_CATEGORY_METADATA (per `MIXTURE_SHIFT_PROTOCOL.md`) | NOT EXECUTED | NOT EVALUATED | "the general theorem is deferred" | A benchmark with natural per-sample category metadata distinct from domain |
| B-MECH-3S domain-composition audit | False-fire under shifts in 4-way evidence-domain mixture | LOCKED (uses `category_column = domain`) | EXECUTED (5 seeds × 10 mixtures) | **DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED** | "Under resampled domain proportions, neither global nor domain-aware reference reduced fire rate (both = 1.0)" | none for this scope |
| B-MECH-4 KS power / window sweep | False-fire vs detection-power tradeoff | LOCKED (`KS_WINDOW_GRID = (32,64,128,256,512)`) | EXECUTED (5 seeds × 5 windows × 3 degradation types) | **TRADEOFF_IMPROVED** (per report; larger windows raise detection power 24.6% → 62.4% with false fire ≤ 0.06%) | "larger KS window improves detection power while keeping false fire ≤ 0.06% on the evaluated grid" | none for this scope |
| Risk-dominance terms (q₀, q₁, Δ₀, Δ₁, π*) | Compute formal risk-dominance terms | n/a | NOT EXECUTED (clean arm not archived) | INADMISSIBLE_SINGLE_ARM | none | extend B-MECH-1 driver to archive k=0 clean arm |
| Switching certificates | Fired-subset paired-bootstrap LCB benefit | LOCKED | EXECUTED for k=4 only | 1 CERTIFIED + 1 NOT_CERTIFIED | "max_attack k=4 retrospective certificate is positive; zero_attack k=4 is negative; deployment-safe interpretation forbidden" | extend to RGA-v2 scenarios when any candidate passes C1 (none do currently) |

## 10. Family-D Confirmation Status

| Family-D Item | Status | Execution Allowed? | Reason | Next Required Action |
|---|---|---|---|---|
| Family-D v1 | INVALID_FOR_EXECUTION | NO | Frozen contract carried placeholders that would mutate at execution time (per `FAMILY_D_V1_INVALIDATION_NOTICE.md` §1) | none — v1 preserved as historical record only |
| Family-D v2 design | V2_DESIGN_PENDING | NO | Eligibility review open (MPDD modality, Eyecandies operator spec, additional untouched candidate) | close eligibility review per `FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md` §E |
| VisA eligibility | INELIGIBLE_FOR_FAMILY_D | n/a | VisA is registry-locked in Family A (A-POWERED-4) | already removed from v2 candidate list |
| MPDD eligibility | INELIGIBLE_FOR_INDEPENDENT_MULTIMODAL_CONFIRMATION (unless official modality verified) | NO | Modality manifest not verified | obtain official MPDD modality declaration or drop |
| Eyecandies eligibility / protocol | ELIGIBLE_MULTIMODAL_CANDIDATE_PENDING_PROTOCOL_FIX | NO | Choose between canonical one-class vs validation-only synthetic corruption | commit to protocol before any v2 freeze |
| Actual Family-D test execution | NOT EXECUTED | NO (forbidden by every Phase-2 stage) | Family-D execution requires v2 freeze + independent review | depends on all above |

**Mandatory rules — all verified:**

- v1 is marked `INVALID_FOR_EXECUTION` (confirmed).
- v2 is marked `V2_DESIGN_PENDING` (confirmed); no confirmation may be run.
- **No confirmatory evidence currently exists.**
- Successful future Family-D execution may **not** retroactively convert Family-A into confirmatory evidence (preserved in every relevant report).

## 11. Artifact Inventory Checklist

### 11.1 Policy / Contract Artifacts

- [x] PRESENT_AND_VERIFIED — `PHASE_2_RESEARCH_CONTRACT.md` (v1), `..._v2.md`
- [x] PRESENT_AND_VERIFIED — `PHASE_2_STATISTICAL_POLICY.md` (v1), `..._v2.md`
- [x] PRESENT_AND_VERIFIED — `PHASE_2_EXPERIMENT_REGISTRY.csv` (v1), `..._v2.csv`
- [x] PRESENT_AND_VERIFIED — `PHASE_2_CLAIM_MATRIX.csv` (v1), `..._v2.csv`
- [x] PRESENT_AND_VERIFIED — `configs/phase2/rga_v2_gate_contract.yaml`
- [x] PRESENT_AND_VERIFIED — `FAMILY_D_V1_INVALIDATION_NOTICE.md`
- [x] PRESENT_AND_VERIFIED — `FAMILY_D_V2_DESIGN_STATUS.md`

### 11.2 Family-A Artifacts

- [x] PRESENT_AND_VERIFIED — `FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md`
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv`
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv` (5 cells, K5_FULL_FAMILY)
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/predictions/A-POWERED-{1..5}__.../` (5 cells × 12 methods × test/validation × seeds)
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv`
- [x] PRESENT_AND_VERIFIED — historical K=10 outputs `family_a_powered_ensemble_inference.csv` + `family_a_powered_holm_results.csv`

### 11.3 Family-B Infrastructure Artifacts

- [x] PRESENT_AND_VERIFIED — `PHASE_2_2B_INFRASTRUCTURE_AUDIT.md`, `..._COMPLETION_REPORT.md`, `..._CHANGELOG.md`, `..._TEST_REPORT.md`, `..._READY_TO_COMPUTE_CHECKLIST.md`, `..._FINAL_DECISION.md`
- [x] PRESENT_AND_VERIFIED — `src/elara/family_b/` (corruption, mixture_shift, ks_window)
- [x] PRESENT_AND_VERIFIED — G3 top-q in `src/uais/fusion/attention/reliability_estimator.py` (`_VALID_GATE_MODES` includes `top_q`)
- [x] PRESENT_AND_VERIFIED — all 5 drivers: `run_phase2_mechanism_replication.py`, `_rga_v2_gate_sweep.py`, `_mixture_shift.py`, `_ks_power_sweep.py`, `_certificate_audit.py`
- [x] PRESENT_AND_VERIFIED — 30+ Phase-2 / Family-D tests under `tests/`

### 11.4 Family-B Executed Result Artifacts

- [x] PRESENT_AND_VERIFIED — `experiments/phase2/mechanism/b_mech_1_prediction_archives/` (with parquet under method/split subdirs)
- [x] PRESENT_AND_VERIFIED — `family_b_primary_replication_seed_metrics.csv` (60 rows), `..._inference.csv`, `..._holm_k2.csv`
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/certification/switching_certificates.csv` (2 rows), `risk_dominance_terms.csv` (2 rows, inadmissibility note)
- [x] PRESENT_AND_VERIFIED — `B2_MAGNITUDE_COMPARABILITY_AUDIT.md` + `experiments/phase2/mechanism/b2_phase1_vs_phase2_comparability.csv` (4 rows)
- [x] PRESENT_AND_VERIFIED — `experiments/phase2/mechanism/rga_v2_threshold_selection.csv` (20 rows = 5 seeds × 4 gates; **note: B-MECH-2 fuller failure-surface run reached 15 seeds; threshold selection log captured only first 5 seeds**)
- [x] PRESENT_AND_VERIFIED — `rga_v2_clean_false_fire.csv` (20 rows)
- [x] PRESENT_AND_VERIFIED — `rga_v2_failure_surface_metrics.csv` (960 data rows)
- [x] PRESENT_AND_VERIFIED — `rga_v2_failure_surface_inference.csv` (4 rows: G0..G3 promotion decisions)
- [x] PRESENT_AND_VERIFIED — `rga_v2_prediction_archives/B-MECH-2__.../` (per-gate × attack × k method subdirectories)
- [x] PRESENT_AND_VERIFIED — `domain_composition_shift_metrics.csv` (50 rows)
- [x] PRESENT_AND_VERIFIED — `ks_window_size_power.csv` (25 data rows), `ks_true_degradation_power.csv` (75+ data rows)
- [~] PRESENT_BUT_UNVERIFIED — `ks_mixture_shift_control.csv` contains only `pending_compute` rows (stub; not part of the executed B-MECH-3S/B-MECH-4)
- [~] PRESENT_BUT_UNVERIFIED — `RGA_V2_PARTIAL_FAILURE_REPORT_v2.md` claims "15-seed gate sweep complete" — threshold_selection.csv reflects only first 5 seeds; failure-surface CSV reflects all 15. Minor inconsistency.

### 11.5 Family-D Artifacts

- [x] PRESENT_AND_VERIFIED — `FAMILY_D_V1_INVALIDATION_NOTICE.md`
- [x] PRESENT_AND_VERIFIED — `FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md`
- [x] PRESENT_AND_VERIFIED — `FAMILY_D_V2_DESIGN_STATUS.md`
- [x] PRESENT_AND_VERIFIED — v1 files preserved (`FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md`, etc.)
- [ ] MISSING — any `FAMILY_D_CONTRACT_v2_PRE_TEST_FREEZE.md` — correctly absent (v2 design pending)
- [ ] MISSING — any Family-D execution output — correctly absent (forbidden)

## 12. Test and Reproducibility Verification

| Test Suite Stage | Reported Test Result | Independently Re-run Now? | Current Result | Verified? | Notes |
|---|---:|---|---:|---|---|
| After Phase 2.1 | 431 passed / 7 skipped | NO (report-backed) | n/a | report-only | preserved in PHASE_2_1 reports |
| After Phase 2.2A | 477 passed / 10 skipped | NO (report-backed) | n/a | report-only | preserved in PHASE_2_2A reports |
| After Phase 2.2B infrastructure | 535 passed / 11 skipped | NO (report-backed) | n/a | report-only | preserved in PHASE_2_2B infra reports |
| After Phase 2.2B.exec | 536 passed / 10 skipped | NO (report-backed) | n/a | report-only | preserved in PHASE_2_2B_EXEC reports |
| **Post-Phase-2.2B.1 (current)** | n/a | **YES — re-run during this audit** | **542 passed / 10 skipped** | **VERIFIED** | command: `PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings`; elapsed ~82 s |

**Specific test outcomes:**

- Family-D untouched tests (`test_family_d_v1_never_executable.py`, `test_phase2_family_d_untouched_during_family_b.py`): **PASS** (verified independently this audit).
- Archive completeness test (`test_phase2_family_b_prediction_archive_complete.py`): **PASS** (flipped from SKIP to PASS once B-MECH-1 archives existed).
- Currently 10 skipped tests are all correctly-skipped placeholder-guards (e.g. v2 Family-D manifests intentionally absent).

## 13. Contradictions, Overstatements and Unresolved Risks

| ID | Finding | Severity | Source Evidence | Effect on Claims | Required Fix |
|---|---|---|---|---|---|
| P1.1 | Phase 2.2B infrastructure report claimed `READY FOR FULL FAMILY-B COMPUTE` while drivers were scaffolds | P1_REQUIRED_BEFORE_MANUSCRIPT_UPDATE | `PHASE_2_2B_INFRASTRUCTURE_FINAL_DECISION.md` vs `PHASE_2_2B_EXECUTION_PRECHECK.md` §5 | Bounded — already corrected by Phase 2.2B.exec disclosure + Phase 2.2B.1 implementation | none new — kept as a documented chain of correction |
| P1.2 | B2 effect magnitude changed from +0.0319 (Phase-1) to +0.0939 (Phase-2) | P1_REQUIRED_BEFORE_MANUSCRIPT_UPDATE | `B2_MAGNITUDE_COMPARABILITY_AUDIT.md` §4 | Bounded — audit closed with `COMPARABLE_BUT_ESTIMATOR_CHANGED`; manuscript must use dual-number form | adopt dual-number wording in a future manuscript-update phase |
| P2.1 | B-MECH-2 executed at 15 seeds, not the contract target of 30 | P2_REQUIRED_BEFORE_PHASE2_COMPLETION | `rga_v2_failure_surface_metrics.csv` unique-seed count = 15 | Result (`NOT_IMPROVED` for G1/G2/G3) holds at min-for-inference (15), but full contract was 30 | re-run B-MECH-2 with seeds 47–71 to reach 30, OR formally accept 15-seed minimum |
| P2.2 | B-CERT-1 not extended to RGA-v2 scenarios | P2_REQUIRED_BEFORE_PHASE2_COMPLETION | `switching_certificates.csv` contains only G0 mean-gate rows | Pending C4 criterion in B-MECH-2 inference table | extension is empty by design — no RGA-v2 candidate passes C1 — but the empty extension should be explicit |
| P2.3 | Risk-dominance terms `inadmissible_single_arm` | P2_REQUIRED_BEFORE_PHASE2_COMPLETION | `risk_dominance_terms.csv` notes | Cannot report (q₀, q₁, Δ₀, Δ₁, π*) | add k=0 clean-arm to B-MECH-1 archive then re-run B-CERT-1 |
| P2.4 | `RGA_V2_PARTIAL_FAILURE_REPORT_v2.md` says "15 seeds" but `rga_v2_threshold_selection.csv` records only 5 seeds | INFORMATIONAL_BOUNDARY | file-level mismatch in seed counts | does not change `NOT_IMPROVED` decision; selection trail only required to be recorded once per gate | re-verify or update report wording for consistency |
| P2.5 | Family-D v2 design pending — no formal Phase-2-end-without-Family-D decision document | P2_REQUIRED_BEFORE_PHASE2_COMPLETION | absence of explicit "Phase-2 closes without confirmation" file | scope-closure unclear — Phase 2 cannot be formally `complete` without either Family-D or formal deferral | write a Phase-2-closure-without-Family-D decision document |
| INFO.1 | `ks_mixture_shift_control.csv` is a stub of `pending_compute` rows; **not** the B-MECH-3S output | INFORMATIONAL_BOUNDARY | file inspection | could confuse a reviewer | rename or delete the stub |
| INFO.2 | Phase 2.2B.1 work is uncommitted | INFORMATIONAL_BOUNDARY | `git status --short` shows 4 modified drivers + 2 modified tests + many untracked files | reproducibility risk if files lost | commit Phase 2.2B.1 work |
| INFO.3 | Family-A static-reference improvement is mistakable for strong-baseline superiority unless reader sees v2 report §3 | INFORMATIONAL_BOUNDARY | report text + permanent forbidden list | bounded — text is in place | no new fix |

No P0_BLOCKER findings.

## 14. Allowed Claims Right Now

1. "Family A provides powered audited static-reference evidence across five previously inspected benchmark cells; all five cells show Holm-significant positive Δ AUC vs the fixed `static_attention` reference under K=5 multiplicity." ✅ ALLOWED
2. "Under the Phase-2 archived-prediction pipeline, B1 zero-attack k=4 coherent-collapse improvement reproduces closely (Δ AUC = +0.0507 [+0.036, +0.065]; target +0.0506)." ✅ ALLOWED
3. "Under max-attack k=4, the Phase-2 evaluation produces a positive retrospective fired-subset switching certificate (LCB = +0.0085) under the defined stress protocol; no production-safety inference is implied." ✅ ALLOWED
4. "B2 remains positive but its larger Phase-2 magnitude (+0.0939) results from an estimator change to 30-seed ensemble-pooled AUC; manuscript integration must use the dual-number form per the closed magnitude audit." ✅ ALLOWED
5. "RGA-v2 candidates {G1, G2, G3} were evaluated and none passed C1 (clean false-fire budget); no RGA-v2 promotion is granted; G0 remains the locked production-of-record gate." ✅ ALLOWED (negative result honestly reported)
6. "Under the B-MECH-3S exploratory domain-composition shift, neither global nor domain-aware KS reduced false fire (both fire 100% under the resampled mixtures); the general category/cohort theorem remains explicitly deferred." ✅ ALLOWED
7. "Under the B-MECH-4 KS window-size sweep, larger windows raise detection power while keeping clean false-activation ≤ 0.06% on the locked grid (decision `TRADEOFF_IMPROVED`)." ✅ ALLOWED

Each of 1–7 is supported by a file artifact under `experiments/phase2/` or `docs/research/phase2/` listed in Sections 5–6, 8–9.

## 15. Forbidden Claims Right Now

- [x] FORBIDDEN — "RGA-v2 solves partial failures." (Missing evidence: every C1 row in `rga_v2_failure_surface_inference.csv` is `False` for G1/G2/G3.)
- [x] FORBIDDEN — "ELARA handles k=1, k=2 or k=3 failures better." (Missing evidence: C2 = `False (0/2+)` for every non-baseline gate.)
- [x] FORBIDDEN — "Category-aware or domain-aware KS reduces false firing." (Missing evidence: `domain_composition_shift_metrics.csv` reports both rates = 1.0; decision `FALSE_FIRE_NOT_REDUCED`.)
- [x] CONDITIONAL — "KS window-size tradeoff is validated." (Permitted only with the precise wording "on the locked window grid {32, 64, 128, 256, 512} under the evaluated degradation types"; bare validation claim forbidden.)
- [x] FORBIDDEN — "Theory closure is complete." (Missing evidence: risk-dominance terms inadmissible; B-CERT-1 not extended to RGA-v2.)
- [x] FORBIDDEN — "Family D confirms ELARA." (Missing evidence: no Family-D execution.)
- [x] FORBIDDEN — "ELARA is universal." (Missing evidence: 1 audited reproduction family + 1 mechanism cell does not establish universality.)
- [x] FORBIDDEN — "ELARA is SOTA." (Missing evidence: no head-to-head competitive evaluation under Phase 2.)
- [x] FORBIDDEN — "ELARA beats strongest baselines across benchmarks." (Missing evidence: Family-A primary surface compares only to static_attention; secondary K=10 pilot on A-POWERED-1 already shows 5 of 10 stronger baselines do NOT separate.)
- [x] FORBIDDEN — "Retrospective certificates prove deployment safety." (Missing evidence: `boundary_notice` in `switching_certificates.csv` explicitly disclaims this.)
- [x] FORBIDDEN — "Phase 3 may begin." (Missing evidence: Phase 2 not formally closed; Family-D unresolved; per stop-boundary across every Phase-2 spec.)

## 16. Remaining Phase 2 Work Checklist

| Order | Task | Required Before Starting? | Compute Required? | Current Blocker | Output Needed | Completion Effect |
|---:|---|---|---|---|---|---|
| 1 | B2 magnitude-comparability audit | none | no | n/a (DONE) | `B2_MAGNITUDE_COMPARABILITY_AUDIT.md` | DONE — manuscript may now adopt dual-number form |
| 2 | Re-run B-MECH-2 at full 30 seeds (extending from current 15) | B-MECH-2 archive present | YES (~hours) | wall-clock budget | additional rows in `rga_v2_failure_surface_metrics.csv` | strengthens NOT_IMPROVED decision; closes P2.1 |
| 3 | Add k=0 clean-arm to B-MECH-1 archive + re-run B-CERT-1 | infrastructure ready | YES (~minutes) | none | `risk_dominance_terms.csv` populated | closes P2.3 (risk-dominance inadmissibility) |
| 4 | Extend B-CERT-1 to RGA-v2 scenarios (or document why empty) | B-MECH-2 + B-CERT-1 ready | YES (~minutes) | none | new rows in `switching_certificates.csv` OR formal empty-extension justification | closes P2.2 (C4 criterion in RGA-v2 inference) |
| 5 | Produce single Phase 2.2B.1 hostile-review verdict consolidating B-MECH-1/2/3S/4 + B-CERT-1 | tasks 2–4 ideally first | no | none | `PHASE_2_2B_1_FINAL_DECISION.md` | closes overall Family-B verdict |
| 6 | Commit all Phase 2.2B.1 work | tasks 1–5 finalized | no | none | git commit | closes INFO.2 |
| 7 | Close Family-D v2 eligibility review (MPDD modality, Eyecandies operator, additional candidate) | none | no | open research decisions | updated `FAMILY_D_V2_DESIGN_STATUS.md` + closure of `FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md` checklist | enables Family-D v2 freeze |
| 8 | Independent external review of Family-D v2 freeze | task 7 complete | no | external reviewer | reviewer sign-off | enables Family-D execution |
| 9 | Execute Family-D held-out evaluation once (only if valid) | tasks 7–8 complete | YES (~hours) | reviewer sign-off | held-out result CSV + report | enables "confirmatory" language |
| 10 | OR — write formal Phase-2-closes-without-Family-D decision document | tasks 1–6 complete | no | research decision | new decision document | alternative completion path; closes P2.5 |
| 11 | Decide whether to begin Phase 3, ELARA-Universal, or ORIUS | tasks 1–10 resolved | no | scope-end research decision | Phase-3 charter | only allowed after Phase 2 formally closes |

## 17. Exact Next Codex Task Recommendation

### Title

**PHASE 2.2B.2 — Consolidate Family-B Evidence and Formal Family-B Verdict**

### Scope (one paragraph)

- Re-run B-MECH-2 to extend from 15 to 30 seeds (only if results would change the `NOT_IMPROVED` decision; otherwise record the 15-seed verdict as final under the contract's `minimum_for_inference: 15`).
- Extend B-MECH-1 to archive the k=0 clean arm; re-run B-CERT-1 to populate the (q₀, q₁, Δ₀, Δ₁, π*) risk-dominance table; either extend B-CERT-1 to RGA-v2 scenarios or formally record the empty extension.
- Consolidate B-MECH-1/2/3S/4 + B-CERT-1 into a single hostile-review verdict (`PHASE_2_2B_1_FINAL_DECISION.md`).
- Commit all Phase 2.2B.1 + 2.2B.2 work to git.
- Do NOT execute Family D.
- Do NOT edit paper or thesis.
- Do NOT begin Phase 3, ELARA-Universal, or ORIUS.

## 18. Final Phase 2 Verdict and Percentage

### Final Verified Phase 2 Status

**`PHASE_2_PARTIALLY_COMPLETE_FAMILY_B_CONTINUATION_REQUIRED`**

### Completion Percentage

- **Total Phase 2 completion:** **82.25%**
- **Infrastructure completion:** **~100%**
- **Executed scientific evidence completion:** **~70%**
- **Confidence level:** **HIGH** for the items inspected (every result table is sourced from a file artifact directly inspected during this audit).
- **Largest remaining work:** (a) consolidate Family-B into a single verdict and commit; (b) decide whether to extend B-MECH-2 to 30 seeds; (c) close Family-D v2 eligibility review or write a formal Phase-2-without-Family-D closure document.

### Final One-Paragraph Reviewer Summary

Phase 2 has produced (i) a fully repaired contract / registry / statistical-policy layer with K=5 Holm and locked static-attention comparator; (ii) a complete Family-A powered audited static-reference reproduction across five previously inspected benchmark cells with all five cells Holm-significant at α=0.05 and 30/30 sign-consistent (Δ AUC ranging from +0.0095 to +0.1082); (iii) B-MECH-1 mechanism replication confirming the Phase-1 B1 endpoint to four decimal places and reproducing the B2 direction at a substantially larger magnitude under a documented estimator change (dual-number reporting required); (iv) a fully executed B-MECH-2 RGA-v2 partial-failure sweep at 15 seeds with all candidate gates G1/G2/G3 failing the locked clean-false-fire budget and a `NOT_IMPROVED` promotion decision for every non-baseline gate; (v) an exploratory B-MECH-3S domain-composition false-fire audit returning `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED`; (vi) a B-MECH-4 KS window-size sweep with `TRADEOFF_IMPROVED` on the locked window grid; (vii) a partial B-CERT-1 retrospective certificate with `max_attack k=4` CERTIFIED (LCB +0.0085) and `zero_attack k=4` NOT_CERTIFIED; and (viii) a verified Family-D protection state (`INVALID_FOR_EXECUTION` for v1 and `V2_DESIGN_PENDING` for v2). The full pytest suite is 542 passed / 10 skipped. The single remaining work to close the Phase formally is to consolidate Family-B into one verdict, optionally extend B-MECH-2 from 15 to 30 seeds, populate the risk-dominance terms by archiving the k=0 clean arm, commit Phase 2.2B.1 work, and decide between executing Family-D v2 (after eligibility-review closure and independent review) or formally closing Phase 2 without Family-D. No paper or thesis edits have been made and no Phase 3 work has been opened, in line with the stop boundaries of every Phase-2 stage.
