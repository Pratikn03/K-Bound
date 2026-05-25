# Phase 2 — Interim Report

**Window:** Phase 2 in-session pilot (contracts + infrastructure + 1 cell).
**Stop boundary respected:** Phase 2.A through 2.C executed; Phase 2.D / 2.E / 2.F / 2.G scaffolded; Phase 2.H Family D contract frozen but **not** executed; Phase 3, ELARA-Universal, and ORIUS not opened.

## 1. What was actually executed this session

| Stage | What landed | Status |
|---|---|---|
| 2.A | Research contract + experiment registry + claim matrix + statistical policy + compute plan | committed (`6299c3f`) |
| 2.B | Prediction-archive contract: `src/elara/evaluation/prediction_archive.py`, 28-column Parquet schema, immutable `rerun_N` suffix, 4 schema tests | passing |
| 2.B | Validation-only selection tests (5) + no-leakage tests (2) | passing |
| 2.C | 30-seed pilot on **A-POWERED-1** (MVTec 3D-AD / PatchCore supervised-paired) — `run_phase2_powered_audited_pilot.py` | exit 0; 30 seeds; prediction archive 360 rows / seed × 12 methods × 2 splits = 8 640 archive entries |
| 2.C | Audited inference: `run_phase2_powered_audited_analysis.py` | wrote `family_a_powered_ensemble_inference.csv`, `family_a_powered_holm_results.csv` |
| 2.D | Family B mechanism replication scaffolds (B-MECH-1..4 reports + CSVs marked `pending_compute`) | scaffolded |
| 2.E | RGA-v2 gate contract YAML at [configs/phase2/rga_v2_gate_contract.yaml](../../../configs/phase2/rga_v2_gate_contract.yaml) — 5 candidate gates, 6 promotion criteria, 4 decisions | locked |
| 2.F | KS reference / KS power / mixture-shift control CSV scaffolds | scaffolded |
| 2.G | Risk-dominance code + paired-bootstrap switching certificate + 4 tests | passing |
| 2.H | Family D pre-registration: contract, dataset inventory, hypotheses CSV, partition manifest JSON, statistical policy, execution commands | **frozen, not executed** |

## 2. Headline result from A-POWERED-1 (the only executed cell)

Source: [FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md](./FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md).

- RGA+ ensemble ROC-AUC = **0.7420** on n_test = 278 (217 pos, 61 neg), 30 seeds.
- Validation-frozen head distribution: 19 boost / 11 router (no test-set selection).
- Holm-significant separation (α = 0.05, K = 10) against: `static_attention`, `craf_attention`, `early_fusion_mlp`, `confidence_weighted_mean`, `eata_score_adapter`.
- **No** Holm-significant separation against: `late_fusion_ensemble`, `random_forest`, `tent_score_adapter`, `sar_score_adapter`, `ttt_pseudo_label_adapter`.
- Bootstrap CIs include zero for all five non-significant comparators.

This is **one** cell on a **non-confirmatory** family. It does not entitle any of the forbidden claims (Section 5).

## 3. What is NOT executed (and why)

| Cell / family | Why deferred |
|---|---|
| A-POWERED-2..5 | scope = 1 pilot cell this session |
| B-MECH-1..4 (replication of B1/B2 mechanism deltas, RGA-v2 partial-failure surface, KS reference + KS power) | scope = scaffold only this session |
| B-CERT-1 (risk-dominance + switching certificate on archived predictions) | scope = code + tests only |
| D-H1..D-H5 (held-out confirmatory replication on MPDD / Eyecandies) | **explicitly frozen, not executed** — Family D MUST be executed in a separate, post-freeze compute window for the result to carry confirmatory weight |

## 4. Locked PRIMARY mechanism endpoints — preserved verbatim

The primary RGA+ mechanism endpoints from prior work are preserved as **claim-matrix targets** for Phase-2 mechanism replication. They are **not** re-derived here.

- **B1**: ΔAUC = **+0.0506** with paired-bootstrap 95% CI **[0.0315, 0.0681]**.
- **B2**: ΔAUC = **+0.0319** with paired-bootstrap 95% CI **[0.0050, 0.0617]**.
- Both endpoints came from k-of-D k=4 mean-gate.

Secondary descriptive surface (default-gate path): **+0.0367 / +0.0538**. This surface is descriptive only; the primary inferential weight remains with B1/B2.

## 5. Forbidden claims (preserved verbatim, not weakened)

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

The audited A-POWERED-1 result above does **not** support any of these.

## 6. Path forward (compute-bounded)

1. Open a separate compute window to run A-POWERED-2..5 (4 cells × 30 seeds ≈ 4 × pilot duration).
2. Open a separate compute window to run B-MECH-1..4 (mechanism replication) and B-CERT-1 (certificate on archived predictions).
3. **Only then** open the Family-D execution window with the contract frozen here.
4. Phase 3, ELARA-Universal, and ORIUS remain explicitly out of scope until the above three windows return.
