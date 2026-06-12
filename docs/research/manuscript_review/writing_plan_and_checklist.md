# Final Writing Plan and Quality Checklist for ELARA-U

This document contains the development roadmap, writing plan, and quality checklists for transitioning the ELARA-U framework from a pilot to a flagship manuscript.

## Core Narrative and Title Strategy
The final paper must avoid framing ELARA-U as a universal anomaly detector. Instead, it must center on the following narrative:
> **"Validation-AUROC auto-selection is near-oracle on clean data, but becomes stale under deployment shift. ELARA-U uses shift-aware reliability routing to reduce regret and negative transfer when validation reliability no longer matches deployment reliability."**

Recommended titles:
1. **ELARA-U: Shift-Aware Reliability Routing for Cross-Domain Anomaly Detection**
2. **When Validation Gated Selection Goes Stale: Shift-Aware Reliability Routing for Cross-Domain Anomaly Detection**

---

## Abstract and Introduction Structure
The Abstract must detail:
1. The fragmentation problem: no single anomaly detector wins across all domains.
2. The baseline: validation-AUROC auto-selection is extremely strong on clean data.
3. The gap: validation metrics become stale under test-time distribution shift and sensor degradation.
4. The method: ELARA-U uses a drift-adaptive meta-routing policy and rank-normalized stacking.
5. The evaluation: 123 tasks, 5 families, leave-family-out cross-validation, no test labels.
6. The outcome: rank and regret improvements over stale validation auto-selection.
7. Honest boundaries: pilot scope, synthetic shift vs. natural shift, and boundaries on negative transfer.

---

## Methodology Refinement: Stacking and Stale Gating
The paper must clearly distinguish two operational modes:
* **Mode A — Clean Baseline Mode:** Per-task validation-AUROC auto-selection, which serves as a near-oracle baseline under stable validation-to-test transfer.
* **Mode B — Shift-Aware Stack Mode:** The rank-normalized stacking meta-router, which is activated under test-time degradation.

The score normalization must be formally defined:
$$r_m(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}\{s_m(x_i) \leq s_m(x)\}$$

The final fused score is:
$$g(x) = f_\theta(r_1(x), \dots, r_M(x), z_\tau)$$
where $z_\tau$ represents task-level reliability and drift features.

The Kolmogorov-Smirnov (KS) test-time drift statistic is defined as:
$$D_{\mathrm{KS}}(S^{\mathrm{val}}_m, S^{\mathrm{test}}_m) = \sup_t |F^{\mathrm{val}}_m(t) - F^{\mathrm{test}}_m(t)|$$
which is used to detect test-time score drift without accessing labels.

---

## Degradation and Simulation Protocol
To evaluate robustness, we inject heterogeneous degradations to simulate deployment failures:

| Degradation | Formula | Router Mitigation |
| :--- | :--- | :--- |
| Noise | $s' = s + \epsilon$ | Rank-normalized features |
| Saturation | $s' = c + \epsilon$ | Degenerate-channel guard |
| Inversion | $s' = -s$ | Validation inversion guard |
| Monotone Drift | $s' = a s + b$ | Rank-normalization invariance |
| Missingness | channel removed | Missingness flags |
| Contamination Shift | changed anomaly ratio | Drift-adaptive features |

---

## Flagship Target Layout (14–16 Tables and 7–8 Figures)
* **Tables (14–16 target tables):**
  1. Contribution summary
  2. Detector zoo
  3. Router feature groups
  4. Gate U evidence contract
  5. Dataset manifest
  6. Degradation protocol
  7. Clean-transfer main result
  8. Shifted-transfer main result
  9. Family-level mean rank
  10. Leave-family-out bootstrap
  11. Regret-to-oracle and worst-regret
  12. Negative-transfer rate
  13. Calibration: ECE/Brier/NLL
  14. Ablation table
  15. Runtime/compute table
  16. Failure-case table
  17. Reproducibility checklist
* **Figures (7–8 target figures):**
  1. ELARA-U architecture
  2. Clean vs. shifted story diagram
  3. Main rank comparison
  4. Regret-to-oracle comparison
  5. Family heatmap
  6. Drift/degradation effect curve
  7. Calibration reliability diagram
  8. Router decision lifecycle
  9. Benchmark coverage map

---

## Paper Quality Checklist
* **Claims:** Clean auto-select shown as strong; shifted auto-select failure demonstrated; ELARA-U beats stale auto-select; no universal SOTA overclaims.
* **Method:** Stacking vs. routing mismatch addressed; rank-normalized stacking described; KS drift detector and degenerate guard specified.
* **Ablations:** Stacker variants with removed features: no reliability, no drift, no disagreement, no calibration, no guard, raw-score stacking vs. rank-normalized stacking, select-only, fuse-only.
* **Calibration:** ECE, Brier, NLL, and reliability diagrams.
* **Reproducibility:** Protocol YAML, frozen seeds, dataset hashes, score archive manifest, no-leakage audit.

---

## Recommended Sizing and Page Target
* **Workshop short paper:** 6–8 pages
* **Strong workshop / arXiv paper:** 10–12 pages + appendix (best next target)
* **Conference-style paper:** 8–10 pages + appendix
* **Thesis chapter version:** 25–40 pages
* **Full technical report:** 35–50 pages

The optimal target for the flagship arXiv and journal submission is a **12-page main paper + 20-page appendix**.
