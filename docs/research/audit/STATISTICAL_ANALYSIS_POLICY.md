# Statistical Analysis Policy — Phase 0.6: Locked Audited-Reanalysis Policy

**Status:** PROPOSED Phase-1 policy. Subject to user approval before any Phase-1 work proceeds. **This is the final policy lock; Phase 0.5 is superseded by Phase 0.6 on every conflict.**

This policy pre-declares **for already-observed result cells**: what counts as a locked audited reanalysis vs an exploratory analysis, the multiplicity family, the inference procedure, the orientation/polarity rule, and the seed-ensemble interpretation. It also pre-declares **Family D**, a separate future locked confirmatory replication structure for results that have not yet been inspected.

---

## 0. Framing: this is an audited reanalysis policy, not a pre-registration

**The result matrix was examined before this policy was written.** RGA+ router vs boost performance, baseline rankings, Holm-corrected p-values, positive cells, and negative cells have all been inspected. Therefore:

> **Because the result matrix was examined before this policy was written, corrected analyses of existing cells are post-hoc but locked and reproducible. Confirmatory claims require a newly frozen replication evaluation performed after this policy lock.**

Throughout this document the following relabelling applies to **already-observed** cells:

| Phase 0.5 label | Phase 0.6 replacement |
|---|---|
| "pre-registered" | "Locked Audited-Reanalysis Policy" |
| "confirmatory" (for existing observed cells) | "audited primary reanalysis" |
| "pre-declared comparator" (for existing observed cells) | "locked comparator for audited reanalysis" (validation-frozen by default; never test-frozen) |
| "confirmatory p-value" (for existing results) | "audited inferential summary; not independent confirmatory replication" |

The words "pre-registered" and "confirmatory" are reserved **exclusively** for Family D (see §6).

---

## 1. Four analysis families

Phase 0.6 splits the analysis into four explicit families. Holm correction is applied **inside** each family, never across families.

### Family A — Public cross-benchmark performance (audited reanalysis)

Audited primary reanalysis of RGA+ on public paired benchmarks **on the already-observed test fold**. The primary comparison is RGA+ (validation-frozen head) vs the **validation-frozen primary comparator** (see `HEADLINE_METHOD_POLICY.md` §3).

| # | Benchmark | Protocol | Cell type |
|---|---|---|---|
| A1 | MVTec 3D-AD | PatchCore canonical one-class | protocol-diagnostic |
| A2 | MVTec 3D-AD | PatchCore supervised-paired | audited primary reanalysis |
| A3 | MVTec 3D-AD | PatchCore held-out category | audited primary reanalysis |
| A4 | MVTec LOCO-AD | PatchCore canonical one-class | protocol-diagnostic |
| A5 | MVTec LOCO-AD | PatchCore supervised-paired | audited primary reanalysis |
| A6 | VisA | RGB+edge canonical one-class | protocol-diagnostic |
| A7 | VisA | RGB+edge supervised-paired | audited primary reanalysis |
| A8 | UNSW-NB15 | flow/conn/context | audited primary reanalysis |

**Family A Holm correction:** applied to the **audited-primary-reanalysis** cells only (A2, A3, A5, A7, A8 — **K = 5**). Protocol-diagnostic cells (A1, A4, A6) are reported with raw p-values for descriptive purposes only.

**Canonical block (issue E P0):** A1, A4, A6 are subject to the canonical-metric block — no PR-AUC / ECE / Brier claim until the `audit_canonical_label_semantics.py` audit completes. ROC-AUC at chance level is descriptive evidence only.

**Family A does not support confirmatory superiority claims.** Every Family A result is an audited inferential summary on an inspected test fold; the result matrix was examined before this policy was written.

### Family B — Mechanism stress (audited mechanism endpoints + descriptive surfaces)

Audited mechanism evidence on the secondary stress benchmark ELARA-Bench-LA. Phase 0.6 corrects the Phase 0.5 mistake of treating every sweep / ablation as a single confirmatory cell. The corrected structure:

| # | Benchmark | Protocol | Cell type |
|---|---|---|---|
| B1 | ELARA-Bench-LA | coherent zero-attack all-domain at **locked tau=0.66** | audited mechanism endpoint |
| B2 | ELARA-Bench-LA | coherent max-attack all-domain at **locked tau=0.66** | audited mechanism endpoint |
| B3 | ELARA-Bench-LA | k-of-D corruption sweep | **descriptive mechanism surface only** |
| B4 | ELARA-Bench-LA | tau threshold sweep | **descriptive hyperparameter / gate-activity analysis only** |
| B5 | ELARA-Bench-LA | reliability-component ablation | **descriptive mechanism attribution only** |

**Family B Holm correction:** K = 2 (B1 + B2 only).

**B3–B5 carry no Holm-corrected significance claim.** Each contains multiple inspected contrasts; no significance claim is defensible from existing inspected results. They are reported as descriptive surfaces.

**Future mechanism replication (Family D mechanism subset):** pre-specify exactly one zero-attack contrast, exactly one max-attack contrast, and optionally exactly one no-KS-component ablation contrast. Correct only over those pre-specified endpoint hypotheses. K ≤ 3 for the mechanism subset of Family D.

**Allowed Family B claim (audited):** "the audited mechanism endpoints at tau=0.66 show the RGA gate produces a positive ROC-AUC delta vs static attention under coherent all-domain collapse on ELARA-Bench-LA. The k-of-D, tau, and component-ablation surfaces are descriptive."

**Forbidden Family B claim:** "RGA+ generalises across naturally paired benchmarks because of the mechanism evidence on ELARA-Bench-LA." (Family A is needed for that — and Family A is also not confirmatory.)

### Family C — Exploratory audits

Exploratory cells. No Holm correction inside or across the family. Each cell is reported with raw p-values or descriptive comparisons.

| # | Benchmark | Protocol | Cell type |
|---|---|---|---|
| C1 | Real3D-AD | PCA shape + depth supervised | exploratory |
| C2 | VisA | RGB+random noise-floor | exploratory |
| C3 | UNSW-NB15 | held-out attack categories (already inspected) | exploratory |
| C4 | ELARA-Bench-LA-healthcare-replay | local retrospective replay | exploratory |

### Family D — Future locked confirmatory replication (NEW IN PHASE 0.6)

**The only family permitted to use the words "pre-registered" and "confirmatory."** See §6.

---

## 2. Primary inference rule (audited reanalysis)

For every audited-primary-reanalysis cell (Family A K=5 and Family B K=2), the procedure is:

1. **Train + validate.** Train the fusion model on `train_idx`; validate every candidate head on `val_idx`.
2. **Freeze RGA+.** Select between `rga_meta_router` and `rga_boosted_fusion` on validation ROC-AUC. Freeze the choice per `HEADLINE_METHOD_POLICY.md` §1.
3. **Freeze the primary comparator.** Use the validation-frozen primary comparator (Rule P2 in `HEADLINE_METHOD_POLICY.md` §3). For already-observed cells, **no comparator is pre-declared**. Pre-declared comparators are reserved for Family D.
4. **Produce test predictions for all seeds** for both the frozen RGA+ head and the frozen primary comparator.
5. **Average predicted scores across seeds**, per method, within the cell. This produces one averaged test-prediction vector per method — an **ensemble predictor**.
6. **Run one paired test per cell on the seed-averaged test predictions:**
   - **DeLong** once for ROC-AUC comparison (RGA+ ensemble vs primary-comparator ensemble). One p-value per cell.
   - **Paired bootstrap over test samples** (not seeds) for a 95% CI of `AUROC(RGA+_ensemble) - AUROC(comparator_ensemble)`. The bootstrap resamples test rows; 5000 iterations default.
7. **Report seed mean ± SD across seeds descriptively** as model-instability evidence only.
8. **Apply Holm-Bonferroni correction** to the **one** primary p-value per audited-primary-reanalysis cell, inside its analysis family (A K=5; B K=2; not across families).

**Label this result as an audited inferential summary, not a confirmatory test.**

---

## 3. Seed-averaged predictions are an ensemble — explicit interpretation rule (NEW IN PHASE 0.6)

> **Seed-averaged predictions constitute an ensemble predictor. Any inferential result based on seed-averaged scores applies to that ensemble, not to a single trained RGA+ instance.**

Phase 0.6 makes this explicit because it changes what every audited p-value means.

### 3.1 What seed-averaging is (and is not)

Seed-averaging the predictions across n_seeds training runs **builds an ensemble**. The seed-averaged DeLong + paired sample bootstrap measures the discrimination ability **of that ensemble**, not of a single trained model selected by any criterion.

This matters because:
- A reader who reads "RGA+ ROC-AUC = 0.866" will reasonably assume that figure characterises a typical deployed single trained RGA+ model.
- The seed-averaged number characterises an n-seed ensemble. A single trained model has higher variance.
- The per-seed mean ± SD is the right descriptive statistic for single-model instability.

### 3.2 Reporting requirements for existing audited results

- Report **seed mean ± SD** in every Family A and Family B table as the descriptive model-instability statistic.
- If the seed-averaged DeLong + paired sample bootstrap is retained as the primary statistical readout, **label it "ensemble audited analysis"** in the table caption.
- Do not write "RGA+ is significantly better than X" without the ensemble qualifier; the correct phrasing is "the n-seed RGA+ ensemble is statistically distinguishable from the n-seed comparator ensemble on the audited test fold".

### 3.3 Requirement for Family D

For Family D the user must choose and freeze, before evaluation:
- **(A) deployed seed-ensemble predictor**: same seed-averaged ensemble framework as Family A/B audited reanalysis, but on a previously untouched test partition.
- **(B) single frozen training seed / model-selection protocol**: a single seed is pre-declared (e.g. seed 42), the model trained on that seed is the deployed model, and inference uses that single trained model. No seed-averaging.

The Family D row in `EXPERIMENT_REGISTRY.csv` must record which option was locked. The choice cannot be made after viewing Family D results.

---

## 4. Polarity-flip policy — LOCKED IN PHASE 0.6

**Final locked decision:** no polarity flipping in primary metrics. Phase 0.5 left this as an open user choice; Phase 0.6 locks it.

### 4.1 Decision

- **Primary audited reanalysis (Family A audited primary + Family B audited endpoints) uses NO polarity flipping** in the reported primary metrics.
- **Family D future confirmatory replication also uses NO polarity flipping** by default. If a future deployment variant uses orientation correction, it must be evaluated as a **separately named method** (e.g. `RGA-orient` distinct from `RGA`) — never grafted into the existing method name.

### 4.2 Diagnostic retention

The synthetic-anomaly validation probe is retained **only as a descriptive validation-only score-orientation diagnostic**. Specifically:
- The probe runs at evaluation time per seed (existing `_calibrate_polarity` in `run_breakthrough_experiment.py:535-586`).
- The probe emits per-seed flip logs to the results JSON (`polarity_calibration.flip_required`, `polarity_calibration.calibration_auroc`, `polarity_calibration.n_synthetic`).
- **The probe must not alter primary predictions.** The Phase-1 implementation removes the flip from the primary-prediction path:
  ```python
  # Phase 0.6 lock:
  # The probe's flip_required flag is logged but NOT applied to
  # static_probs / craf_probs in the primary-metric path.
  ```
- The flip logs are reported descriptively in the table or appendix.

### 4.3 Rationale for the lock

- The Phase-0 audit found 3 of 5 canonical-cell seeds flipped at probe AUROC ≈ 0.5 ± noise. Flipping at this borderline produces a mixed-orientation estimator.
- The flip was applied asymmetrically (static + RGA flipped; baselines not), making comparisons non-equivalent.
- Removing the flip from the primary path is the safest path: predictions are reported as the model emits them; orientation is a model property; if the model is in inverse polarity that is a model defect, not something the evaluation should mask.
- Any orientation-correcting deployment variant must be a separately named method so the reader can see exactly which path produced which number.

### 4.4 What this lock forbids

- Reporting `static_attention` ROC-AUC where some seeds were flipped and others not.
- Asymmetric application of the flip across methods.
- Calling the flip a "deployment-grade sanity check".
- Combining flipped and non-flipped seeds into a single mean ± SD.

### 4.5 Implementation note

The Phase-1 implementation must:
1. Remove `static_probs = 1.0 - static_probs` and `craf_probs = 1.0 - craf_probs` (and `_val_probs` equivalents) from the primary-prediction path in `run_breakthrough_experiment.py:2078-2082`.
2. Retain the probe call (`polarity_info = _calibrate_polarity(...)`) and the JSON logging.
3. Update all existing affected JSONs by re-running canonical cells **only if** the issue-E label/metric semantics audit confirms re-running is necessary; otherwise apply the de-flip as a post-processing pass that the audit script verifies is correct.
4. Add a regression test in `tests/test_paper_asset_metadata.py` that asserts no Family A or Family B primary metric is computed on flipped predictions.

---

## 5. Comparator policy — LOCKED IN PHASE 0.6

For **all already-observed cells (Families A, B, C):** the primary comparator is **validation-frozen** (Rule P2 in `HEADLINE_METHOD_POLICY.md` §3).

> **For existing observed cells we do not claim that SAR, Tent, RF, or any other baseline was pre-declared.** Phase 0.5's pre-declaration registry for existing cells is rescinded.

Two reporting options for existing audited reanalyses:

- **(A) Validation-frozen comparator selection.** Choose the highest-validation-ROC-AUC baseline per cell, freeze that name, report the audited inferential summary against it. This is the locked Phase 0.6 default.
- **(B) Full descriptive baseline ranking with no confirmatory comparator.** Show every baseline in the table descriptively. No "RGA+ beats X" framing. Report the audited inferential summary against the validation-frozen comparator separately.

**Do not write "RGA+ beats every baseline" for existing cells under either option.**

### 5.1 Pre-declared comparator registry — reserved for Family D only

A pre-declared comparator registry is allowed **only** for Family D future locked confirmatory replication cells, where the comparator can be genuinely declared before any test inspection. The registry in `HEADLINE_METHOD_POLICY.md` §4 (Phase 0.6) is now Family-D-only.

---

## 6. Family D — Future locked confirmatory replication (NEW IN PHASE 0.6)

**The only family permitted to use the words "pre-registered" and "confirmatory."**

### 6.1 Membership rule

A cell qualifies for Family D **only if** it satisfies **all** of the following:

1. The test partition has not been inspected at any point before this policy lock.
2. The split rule was defined and frozen before any inspection.
3. The seed count was locked before evaluation.
4. The RGA+ head selection rule was frozen before evaluation.
5. The comparator selection rule was frozen before evaluation.
6. The multiplicity family (Family D K-count) was frozen before evaluation.
7. The seed-ensemble option (deployed seed-ensemble vs single frozen seed) was chosen and recorded before evaluation.

A cell that fails any of these conditions is not Family D. It is Family A audited reanalysis or Family C exploratory.

### 6.2 Proposed Family D candidates

These are *candidates* — none has yet been locked, because locking requires a test partition that does not yet exist or has not yet been inspected.

| # | Candidate | Source dataset | Test partition requirement |
|---|---|---|---|
| D1 | MVTec 3D-AD supervised-paired | MVTec 3D-AD | **Newly created untouched locked test split** or **external replication partition**. The currently inspected MVTec 3D supervised-paired test fold cannot be reused. |
| D2 | MVTec LOCO-AD supervised-paired | MVTec LOCO-AD | Newly created untouched locked test split. The current LOCO supervised-paired fold cannot be reused. |
| D3 | UNSW-NB15 with a newly defined temporally or attack-held-out locked evaluation | UNSW-NB15 | A NEW partition. The existing UNSW Family A cell (flow/conn/context) and existing Family C cell (held-out attack categories) have both been inspected; neither may be Family D. |
| D4 | Any newly added naturally paired independent-modality dataset | TBD | Must be `independent_modalities` pairing strength. No `derived_view_proxy` benchmarks admitted. |

### 6.3 Family D inference procedure

Same as §2 (seed-averaged DeLong + paired sample bootstrap if option A; single-seed DeLong if option B) — except:
- No flip applied (per §4).
- Pre-declared comparator (per `HEADLINE_METHOD_POLICY.md` §4 Family D registry).
- Holm correction applied across the Family D K-count (locked before evaluation).
- The resulting p-value may be called **confirmatory** because the policy was locked before the data were inspected.

### 6.4 What Family D enables

Only Family D can support:
- The word "confirmatory" in the paper.
- The word "pre-registered" in the paper.
- A universal-independent-multimodal claim (only via a D4 cell with `independent_modalities` pairing).
- A "RGA+ beats X" claim with statistical force.

Without Family D, the paper's strongest defensible claim about RGA+ is: "In an audited reanalysis of inspected public benchmark data, the RGA+ ensemble (with validation-frozen head and validation-frozen primary comparator) ranks at position X out of n baselines; the audited inferential summary on cell Y is p_Holm = z." That is the ceiling of the paper as it stands. A confirmatory claim requires Family D.

---

## 7. Multiplicity correction

Holm-Bonferroni is applied **inside** each family, never across:

- Family A audited primary K = 5: A2, A3, A5, A7, A8.
- Family B audited mechanism endpoints K = 2: B1, B2.
- Family C K = 0 (no Holm; raw p-values only, descriptive).
- Family D K = locked before evaluation (no value yet).

The protocol-diagnostic cells (A1, A4, A6) are not part of Family A K-count.

---

## 8. P0 canonical-metric block (retained from Phase 0.5)

Issue E (canonical PR-AUC / ECE / Brier = 0.7835, incompatible with prevalence 0.2244) remains **P0 Fatal**. The required Phase-1 gating step is:

1. Create `src/scripts/audit_canonical_label_semantics.py`.
2. Verify on each canonical results JSON: label semantics, `pos_label`, score orientation, constant-baseline behaviour, artifact-to-table linkage.
3. Emit `experiments/audit/canonical_label_semantics.json` + a short narrative.
4. **No canonical PR-AUC / ECE / Brier may be promoted in the paper or thesis until the audit identifies the root cause.**
5. If the cause is a code bug, fix and re-run canonical cells. Verify supervised-paired numbers are unchanged.

---

## 9. Anti-rules (consolidated, Phase 0.6)

| ID | Anti-rule | Source rule |
|---|---|---|
| AR-1 | No test-set selection of RGA+ head | Rule 4 + 5 |
| AR-2 | No test-set selection of the primary comparator | Rule 4 + 5 |
| AR-3 | No Fisher combination of seed p-values | Rule 7 |
| AR-4 | No bootstrap over seeds as the primary significance test | Phase 0.5 update |
| AR-5 | No claim of SOTA / production / deployment-grade / leaderboard / universal superiority | Rule 9 |
| AR-6 | No canonical PR-AUC / ECE / Brier claim until E audit completes | Issue E P0 |
| AR-7 | No mixing of analysis families in the same Holm correction | Phase 0.5 update |
| AR-8 | No asymmetric polarity flip across methods | Issue F |
| AR-9 | No causal language for what is a model-response sensitivity | Issue H |
| AR-10 | No "naturally paired" without `pairing_strength` classification | Phase 0.5 |
| **AR-11** | **No "pre-registered" or "confirmatory" label for already-observed cells.** | **Phase 0.6 NEW** |
| **AR-12** | **No "RGA+ beats every baseline" claim for existing cells. Use audited inferential summary against validation-frozen primary comparator only.** | **Phase 0.6 NEW** |
| **AR-13** | **No polarity flipping in primary metrics for any cell (Families A / B / C / D).** | **Phase 0.6 LOCKED** |
| **AR-14** | **No single-method ROC-AUC reported without the ensemble qualifier when the number comes from seed-averaged predictions.** | **Phase 0.6 NEW** |
| **AR-15** | **No Holm-corrected significance claim for B3 / B4 / B5 sweeps and ablations from already-observed results.** | **Phase 0.6 NEW** |
| **AR-16** | **No reuse of an inspected test partition for Family D. Family D requires a newly created untouched locked test split, an external replication partition, or a newly added dataset.** | **Phase 0.6 NEW** |

---

## 10. What this policy explicitly prevents (consolidated)

- Test-set oracle selection of either RGA+ or the primary comparator (AR-1, AR-2).
- Multiplicity family drift / mis-statement (AR-7).
- Fisher-method independence violation (AR-3).
- Bootstrap-over-seeds as confirmatory (AR-4).
- Causal-language overreach (AR-9).
- Polarity-flip asymmetry across methods (AR-8) — now superseded by the stronger AR-13 (no flipping at all in primary metrics).
- Promoting canonical PR-AUC / ECE / Brier without the semantics audit (AR-6).
- Mixing pairing-strength tiers in a single claim (AR-10).
- Calling an audited reanalysis "confirmatory" or "pre-registered" (AR-11).
- Claiming RGA+ beats every baseline on existing cells (AR-12).
- Treating seed-averaged predictions as single-model predictions (AR-14).
- Claiming Holm-corrected mechanism significance from inspected sweeps / ablations (AR-15).
- Reusing inspected test partitions for confirmatory claims (AR-16).

---

## 11. What this policy still requires from the user

Phase 0.5 left seven open decisions. Phase 0.6 locks four of them and reduces the open decisions to three:

| # | Decision | Phase-0.5 status | Phase-0.6 status |
|---|---|---|---|
| 1 | Polarity-flip path | Open: (A) no flip vs (B) symmetric flip | **LOCKED: (A) no flipping in primary metrics (AR-13).** |
| 2 | Family C comparator rule | Open: Rule P1 vs P2 | **LOCKED: Rule P2 (validation-frozen) for all existing cells. Pre-declaration reserved for Family D only.** |
| 3 | Polarity threshold | Open: 0.5 vs 0.45 | **MOOT under AR-13 (no flipping).** |
| 4 | Family A K count | Open: K=5 vs K=8 | **LOCKED: K=5 (audited-primary only; protocol diagnostics not in K-count).** |
| 5 | MVTec 3D SP comparator pre-declaration | Open: SAR vs Tent | **MOOT under AR-11/AR-12 (no pre-declaration for existing cells).** |
| 6 | Causal-attribution rename | Open: three candidate names | **Still open: pick one of {"Model-Response Sensitivity to Per-Domain Reliability", "Counterfactual Domain-Reliability Perturbation", "Per-Domain Reliability Sensitivity Analysis"}.** |
| 7 | Pairing-strength prose policy | Open: per-cell sentence vs global paragraph | **Still open: prefer per-cell reiteration; alternative is one global paragraph.** |

Three open decisions remain. Two are locked. Two are now moot under AR-13/AR-11. Plus one new decision introduced by Phase 0.6 (the Family D seed-ensemble option) is still open:

- **D-OPT.** Family D seed-ensemble option: **(A) deployed seed-ensemble predictor** vs **(B) single frozen training seed / model-selection protocol**. Must be locked **before** any Family D evaluation; defer this decision until Family D candidates are concrete.

So the remaining open decisions are: causal-attribution rename, pairing-strength prose policy, and Family D seed-ensemble option (deferred until Family D candidates exist).

---

## 12. Phase-0.6 supersession statement

This document supersedes the Phase 0.5 `STATISTICAL_ANALYSIS_POLICY.md` on every conflict. Phase 0.5 remains in the audit history as a record of intermediate policy; readers should treat Phase 0.6 as the active policy lock.
