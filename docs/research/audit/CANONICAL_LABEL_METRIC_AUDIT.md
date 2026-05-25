# Canonical Label / Metric Semantics Audit (Phase 1.A)

**Status:** completed.
**Verdict:** `METRICS_VALID_BUT_MISINTERPRETED`.
**Branch:** `fix/elara-phase1-empirical-validity`.
**Source artifact:** [`experiments/audit/canonical_label_semantics.json`](../../../experiments/audit/canonical_label_semantics.json).
**Audit script:** [`src/scripts/audit_canonical_label_semantics.py`](../../../src/scripts/audit_canonical_label_semantics.py).
**Unit test:** [`tests/test_canonical_label_semantics.py`](../../../tests/test_canonical_label_semantics.py).

The Phase 0 audit flagged canonical MVTec PR-AUC / ECE / Brier ≈ 0.7835 as a P0 Fatal blocker because — with overall dataset prevalence 0.2244 and `pos_label = 1` for anomaly — a constant-score predictor would normally produce metrics near 0.2244, not 0.7835. Phase 1.A re-examined every canonical cell to identify the cause from a finite list: label inversion, `pos_label` mismatch, score-orientation mismatch, stale artifact linkage, metric-implementation bug, or missing raw predictions requiring re-run.

The result is that **all five "bug" hypotheses are rejected**. The 0.7835 number is the canonical test-fold's anomaly prevalence, not the overall dataset's. Under MVTec's canonical one-class protocol the train fold is normal-only and the test fold contains all anomalies — producing a test-fold prevalence far above the overall prevalence. A degenerate predictor on a high-prevalence test fold yields PR-AUC ≈ prevalence, Brier ≈ prevalence, ECE ≈ prevalence. The metrics are mathematically correct but trivially reflect prevalence rather than discrimination ability.

---

## 1. Section A — Label-definition audit

| Benchmark | Overall prevalence | Train prevalence | Validation prevalence | **Test prevalence** | Inferred label semantics |
|---|---|---|---|---|---|
| MVTec 3D-AD canonical one-class | 0.2244 | **0.0000** | **0.0000** | **0.7835** | `label_eq_1_means_anomaly` (canonical one-class consistent) |
| MVTec LOCO-AD canonical one-class | 0.2720 | **0.0000** | **0.0000** | **0.6333** | `label_eq_1_means_anomaly` (canonical one-class consistent) |
| VisA RGB+edge canonical one-class | 0.1109 | **0.0000** | **0.0000** | **0.5550** | `label_eq_1_means_anomaly` (canonical one-class consistent) |

All three canonical cells have train_prevalence = 0 and a *high* test prevalence. This is the canonical one-class structure: train/val are normal-only; the test fold contains all anomalies (and only the residual normals). The dramatic disparity between overall prevalence (~0.22) and canonical test-fold prevalence (0.55–0.78) is the root cause of the 0.7835 surprise.

**Label-inversion hypothesis: rejected.** If labels were inverted, train would have high prevalence and test low prevalence; the data shows the opposite, exactly matching the standard convention.

---

## 2. Section B — Metric-function audit

| Helper | Source | Convention |
|---|---|---|
| `roc_auc_score(y_true, y_prob)` | sklearn | default `pos_label=1`; higher score = positive class |
| `average_precision_score(y_true, y_prob)` | sklearn | default `pos_label=1` |
| `brier_score(y_prob, y_true)` | `src/uais/utils/metrics.py` | `pos_label`-agnostic; uses raw labels |
| `expected_calibration_error(y_true, y_prob, n_bins=10)` | `src/uais/utils/metrics.py` | 10 equal-width bins; reliability gap weighted by bin mass |
| `_compute_from_pred_and_prob` | `src/uais/utils/metrics.py` | all metrics share the same `y_true` / `y_prob`; no `pos_label` override; anomaly-probability convention |

The metric helpers compute everything under `pos_label=1` (sklearn default) and treat the model output as P(anomaly). No inversion happens inside the metric path.

**`pos_label` mismatch hypothesis: rejected.**
**Metric-implementation-bug hypothesis: rejected** (helpers match the standard convention).

---

## 3. Section C — Constant-baseline replay

For each canonical test fold, the audit script computed metrics for synthetic constant-score predictors (probabilities 0.0 / 0.5 / 0.7835 / 1.0) and uniform random predictors (seeds 0 and 42, plus an inverted variant).

Headline finding: the **constant-0.0 anomaly-probability predictor on the MVTec 3D-AD canonical test fold (n=924, prevalence 0.7835)** produces:

| Metric | Value | Comment |
|---|---|---|
| ROC-AUC | 0.5 | true chance |
| PR-AUC | 0.7835 | == test prevalence (sklearn AP convention for constant scores) |
| Brier | 0.7835 | mean((0 - y)^2) = mean(y) = prevalence |
| ECE | 0.7835 | \|0 - prevalence\| at one bin |

This is **exactly** the triple reported for `rga_boosted_fusion` on the same cell (`PR=0.784, ECE=0.784, Brier=0.784`). The match is mechanical: when train has only one class, `ReliabilityBoostedFusion` falls back to a constant predictor at `np.mean(train_labels) = 0.0` (see `reliability_boosted_fusion.py:267-271`), and on a 0.7835-prevalence test fold every degenerate metric equals the prevalence.

The same mechanical match holds for MVTec LOCO (prevalence 0.6333 ≈ reported PR-AUC 0.6333) and VisA RGB+edge (prevalence 0.5550 ≈ reported PR-AUC 0.5550).

---

## 4. Section D — Artifact-reproduction audit

Per-cell `reported PR-AUC` vs `recomputed canonical test-fold prevalence` (delta = absolute difference):

| Benchmark | Method | Reported PR-AUC | Recomputed test prevalence | Δ | Within 0.005? |
|---|---|---|---|---|---|
| MVTec 3D | static_attention | 0.7715 | 0.7835 | 0.0120 | No (small ROC gap from probe-flipped seeds; see polarity audit) |
| MVTec 3D | craf_attention | 0.7857 | 0.7835 | 0.0022 | **Yes** |
| MVTec 3D | rga_meta_router | 0.7857 | 0.7835 | 0.0022 | **Yes** |
| MVTec 3D | rga_boosted_fusion | 0.7835 | 0.7835 | 0.0000 | **Yes** (perfect match: constant-0 predictor) |
| MVTec LOCO | static_attention | 0.6319 | 0.6333 | 0.0014 | **Yes** |
| MVTec LOCO | craf_attention | 0.6548 | 0.6333 | 0.0215 | No (RGA produces some non-degenerate ranking) |
| MVTec LOCO | rga_meta_router | 0.6548 | 0.6333 | 0.0215 | No (matches craf because router selects it under one-class) |
| MVTec LOCO | rga_boosted_fusion | 0.6333 | 0.6333 | 0.0000 | **Yes** (perfect match) |
| VisA RGB+edge | static_attention | 0.5408 | 0.5550 | 0.0142 | No |
| VisA RGB+edge | craf_attention | 0.5393 | 0.5550 | 0.0157 | No |
| VisA RGB+edge | rga_meta_router | 0.5393 | 0.5550 | 0.0157 | No |
| VisA RGB+edge | rga_boosted_fusion | 0.5550 | 0.5550 | 0.0000 | **Yes** (perfect match) |

`rga_boosted_fusion`'s PR-AUC equals the canonical test-fold prevalence to four decimals in every single canonical cell. This is the unambiguous signature of a degenerate constant predictor — exactly the behaviour the runner falls back to when val has one class.

**Stale-artifact-linkage hypothesis: rejected.** The recomputed test-fold prevalence reads directly from the input CSV's `split == 'test'` rows; the audit confirms the JSON is reading the same fold.

**Raw-predictions-missing hypothesis: rejected.** The audit recomputes prevalence from the CSV (no raw test predictions needed); the per-seed and seed-averaged JSON values reproduce as a degenerate constant predictor on this fold.

---

## 5. Section E — Polarity diagnostic audit

The audit emits `experiments/audit/polarity_diagnostic_log.csv` listing the per-seed validation probe AUROC and the historical flip decision. Phase 1.F locks the policy as **no polarity flipping in primary metrics**, so the per-seed flip column is recorded but **not** applied. The CSV's `primary_metrics_use_flip` column is `False` for every row, confirming the lock.

Probe-AUROC distribution for the canonical cells:
- MVTec 3D PatchCore canonical: 3 of 5 seeds had probe AUROC in `[0.45, 0.55]` (borderline); 3 of 5 seeds historically flipped.
- MVTec LOCO PatchCore canonical: borderline behavior similar.
- VisA RGB+edge canonical: borderline behavior similar.

This confirms the Phase-0.5 finding that the existing polarity flip was noise-dominated at canonical-cell probe AUROCs near 0.5. Locking the policy to "no flip" removes this asymmetric noise from the primary path.

---

## 6. Verdict

**`METRICS_VALID_BUT_MISINTERPRETED`.**

All five "bug" hypotheses are rejected:
- Label semantics: **correct** (label=1 means anomaly; canonical one-class structure verified).
- `pos_label` mismatch: **not present** (metrics use sklearn default which matches the convention).
- Score-orientation bug: **not present** (no inversion in the metric path; `rga_boosted_fusion`'s perfect-prevalence match is consistent with a constant-0 predictor, not a flipped-score predictor).
- Stale artifact linkage: **not present** (recompute from CSV agrees with JSON).
- Metric-implementation bug: **not present** (helpers are sklearn defaults).

The 0.7835 number is the **canonical test-fold prevalence** of MVTec 3D-AD. The supervised heads collapse to a degenerate constant predictor under one-class training, and a degenerate predictor on a prevalence-`p` fold produces PR-AUC ≈ `p`, Brier ≈ `p`, ECE ≈ `p`. There is no bug.

---

## 7. Required Phase-1 manuscript actions (no re-runs needed)

1. **Stop promoting canonical PR-AUC / ECE / Brier as headline numbers.** They are not discrimination measures; they are the test-fold prevalence reflected through degenerate predictors.
2. **Mark canonical cells as protocol-diagnostic, not superiority tests.** This was already pre-declared in `STATISTICAL_ANALYSIS_POLICY.md` Phase 0.6 (cells A1, A4, A6 are protocol-diagnostic; not in Family A K-count).
3. **Add a short subsection (or extended caption)** explicitly stating: "Under MVTec's canonical one-class protocol the test fold contains all anomalies, producing test-fold anomaly prevalence ~0.78 (MVTec 3D-AD), ~0.63 (MVTec LOCO-AD), or ~0.56 (VisA). Supervised fusion heads receive normal-only training, collapse to degenerate constant predictors, and trivially yield PR-AUC / ECE / Brier near the test-fold prevalence. We therefore report only ROC-AUC for canonical cells and treat the canonical protocol as a diagnostic of one-class collapse, not a measure of discrimination ability."
4. **Replace canonical PR-AUC / ECE / Brier values in tables with `n/a (degenerate predictor on canonical test fold)`** or a similar explicit label, so a reader who scans the table cannot misread 0.7835 as a quality signal.
5. **No re-runs.** No training run produces a different number; the canonical numbers stand as audited evidence of one-class collapse.

These actions are implemented in Phase 1.G (manuscript repair). The audit is complete; no code or data bug exists.

---

## 8. What this audit unblocks

- Phase 1.B (RGA+ test-set oracle removal) may proceed; canonical cells are not on its critical path because RGA+ headline numbers come from supervised-paired cells (A2, A3, A5, A7, A8), not canonical (A1, A4, A6).
- Phase 1.C (comparator policy repair) may proceed.
- Phase 1.D (statistical inference repair) may proceed.
- Phase 1.E – 1.I may proceed.

The audit does NOT unblock canonical-cell PR-AUC / ECE / Brier prose without the protocol-diagnostic reframing. Phase 1.G must enforce that reframing.
