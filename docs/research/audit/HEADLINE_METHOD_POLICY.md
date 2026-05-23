# RGA+ Headline Method Policy — Phase 0.6: Locked Audited-Reanalysis Policy

**Status:** PROPOSED Phase-1 policy. Subject to user approval before any Phase-1 work proceeds. **This is the final policy lock; Phase 0.5 is superseded by Phase 0.6 on every conflict.**

This policy pre-declares **for already-observed result cells** how RGA+ is defined for the headline number and how the non-RGA primary comparator is chosen. It also defines a **separate pre-declared comparator registry exclusively for Family D future locked confirmatory replication**.

> **Because the result matrix was examined before this policy was written, corrected analyses of existing cells are post-hoc but locked and reproducible. Confirmatory claims require a newly frozen replication evaluation performed after this policy lock.**

---

## 1. RGA+ definition for headline reporting

**RGA+ = the head selected on the validation-fold ROC-AUC and frozen before test evaluation, per cell.**

For **already-observed cells** (Families A, B, C) this is an **audited reanalysis remap** of the existing JSONs:
- The runner already logs both `rga_meta_router` and `rga_boosted_fusion` test metrics per seed.
- The runner already logs per-seed validation ROC-AUC for the router (`router_meta`) and for the boosted head (`rga_boosted.candidate_validation_auc`).
- The Phase-1 emit step rewrites the headline RGA+ field to read the validation-frozen choice from these already-logged values.
- This is **not** a pre-registration; it is a locked reanalysis that uses information (val ROC-AUC) that was always available and was never used as the headline selector before.

For **Family D future cells** the same rule applies, but the policy is locked *before* the data are inspected, so it carries pre-registration force.

Mechanically, per cell (existing or Family D):

1. Train + validate.
2. Compute validation-fold ROC-AUC for both `rga_meta_router` and `rga_boosted_fusion`.
3. Select the head with the higher validation ROC-AUC (ties broken alphabetically). Record the choice in the JSON as `rga_plus_headline.chosen_head ∈ {"router", "boost"}`.
4. Freeze that choice. The test-fold metric of the frozen head is the RGA+ headline number for that cell.

---

## 2. Secondary analyses (allowed)

The router and boost may also be reported separately in the `RGA+ Component Ablation` subsection, with the row that contributed the headline RGA+ visually marked.

A new column "Headline?" carries values `router (val-frozen)` / `boost (val-frozen)` / `not applicable`.

---

## 3. Non-RGA comparator policy — Phase 0.6 lock for already-observed cells

For every already-observed cell in Families A, B, and C, the primary comparator is **validation-frozen (Rule P2)**.

> **No comparator is pre-declared for already-observed cells.** Phase 0.5's pre-declaration registry (which named SAR for MVTec 3D SP, Tent for LOCO SP, RF for VisA SP and UNSW, etc.) is rescinded because that registry was constructed *after* the result matrix had been inspected. Claiming pre-declaration on an inspected result matrix is dishonest pre-registration theatre.

### 3.1 Rule P2 mechanics (the only allowed rule for existing cells)

Per cell, per seed:

1. Compute validation-fold ROC-AUC for every baseline in the candidate set (`RF`, `MLP`, `LFE`, `Tent`, `TTT`, `EATA`, `SAR`, `Conf-mean`).
2. Average the validation ROC-AUC across seeds (per baseline) within the cell.
3. Select the baseline with the highest seed-averaged validation ROC-AUC.
4. Freeze that name. Record it in the JSON as `primary_comparator = {"name": "<method>", "selection_rule": "P2", "val_roc_auc": float, "test_roc_auc": float}`.

The frozen comparator's seed-averaged test ROC-AUC is the comparator for the audited inferential summary.

### 3.2 What the existing-cell prose may say

- "The audited reanalysis selects \<method-name\> as the validation-frozen primary comparator for this cell."
- "The audited inferential summary against the validation-frozen primary comparator is `p_Holm = <value>` (ensemble audited analysis; not confirmatory)."
- "The descriptive baseline ranking on the audited test fold is: ..." (full ordered list of all baselines).

### 3.3 What the existing-cell prose may not say

- "RGA+ beats every baseline" (AR-12).
- "The best non-router baseline is X" (X was chosen by reading the test winner — forbidden).
- "Pre-declared comparator X" (AR-11 forbids pre-declaration framing for existing cells).
- "Confirmatory comparison against X" (existing cells are audited reanalyses only).

### 3.4 If the paper wants a "RGA+ vs every baseline" claim

Under Phase 0.6 the only honest way to make a per-baseline-comparison claim on existing cells is:

- **(A) Descriptive ranking only.** Report the full ordered baseline ranking on the audited test fold. No p-values per baseline. No "beats X" framing.
- **(B) Per-method Holm correction with K = |primary cells| × |baselines| comparisons.** Compute the audited inferential summary against every baseline, all of which are inside the existing Family A K-count expanded to 5 × 8 = 40 comparisons. The corrected p-values will be much weaker. Allowed but not recommended.
- **(C) Defer the claim to Family D.** A genuine "RGA+ beats X" claim requires a Family D replication where X is the pre-declared comparator.

The locked Phase 0.6 default is **(A)** + **single audited inferential summary against the validation-frozen primary comparator (no per-baseline claim)**.

---

## 4. Pre-declared comparator registry — Family D only

**A pre-declared comparator registry is allowed only for Family D future locked confirmatory replication cells.**

The registry below is a placeholder; rows are added only when the Family D split is locked and before any inspection.

| benchmark_family | candidate_baselines | comparator_selection_rule | primary_comparator_status | rationale | locked_before_inspection? |
|---|---|---|---|---|---|
| D1 — MVTec 3D-AD supervised-paired on a newly created untouched locked test split | RF, MLP, LFE, Tent, TTT, EATA, SAR, Conf-mean | Rule P1 (pre-declaration) | **TBD** — to be locked before any D1 evaluation | TBD | Not yet locked |
| D2 — MVTec LOCO-AD supervised-paired on a newly created untouched locked test split | as above | Rule P1 | **TBD** | TBD | Not yet locked |
| D3 — UNSW-NB15 with a newly defined temporally or attack-held-out locked evaluation | as above | Rule P1 | **TBD** | TBD | Not yet locked |
| D4 — Any newly added naturally paired independent-modality dataset | as above | Rule P1 | **TBD** | TBD | Not yet locked |

**Process for filling a Family D row:**
1. Define the test partition and its splitting rule.
2. Confirm the partition has not been inspected (a `git log` audit of every file that touches the partition's data is recommended).
3. Choose a primary comparator from the candidate baselines, justify in writing.
4. Lock the comparator + the seed count + the Holm K-count.
5. *Then* run training and evaluation.
6. *Then* fill the row's `primary_comparator_status` and `locked_before_inspection? = Yes`.

A row cannot move from "Not yet locked" to "Locked" *after* any test evaluation has occurred.

---

## 5. Anti-rules (forbidden constructions, Phase 0.6)

| Forbidden | Why |
|---|---|
| `max(rga_router_test_auroc, rga_boost_test_auroc)` | Test-set oracle selection (Rule 4). |
| `argmax over methods` on the test fold | Test-set oracle selection. |
| Reporting both router and boost in the same row labelled "RGA+" without saying which was frozen | Conceals the selection mechanism. |
| Reporting the winner of router/boost on the test fold as RGA+ | Same as the first row. |
| Re-running the selection rule per metric (boost is RGA+ for ROC, router is RGA+ for PR on the same cell) | Per-metric oracle selection. |
| Selecting the "best non-router" baseline by reading the test fold | Same family of Rule 4 violation, for the comparator. |
| Comparing validation-frozen RGA+ against the post-hoc best-on-test baseline | Asymmetric — gives RGA+ pre-registration credit but lets the comparator p-hack. |
| **Claiming a comparator was pre-declared for an already-observed cell** | **AR-11: dishonest pre-registration theatre.** |
| **Writing "RGA+ beats every baseline" for an existing cell** | **AR-12: existing cells are audited reanalysis, not confirmatory replication.** |
| **Using the Family D pre-declared registry for an existing cell** | **AR-16: Family D is for newly-locked test partitions only.** |

---

## 6. What happens when the frozen choice produces a lower number

Per rule 2: the corrected lower number stands. The manuscript is updated to reflect the validation-frozen number; abstract and master comparison table use the frozen number; the difference between the validation-frozen headline and the test-fold maximum is explicitly noted in the prose if it is material (>0.005 ROC-AUC).

**Predicted impact of Phase 0.6 vs the current paper:** based on the current JSONs:

| Cell | Current test-max RGA+ | Phase 0.6 val-frozen RGA+ | Direction |
|---|---|---|---|
| MVTec 3D PatchCore SP | 0.739 (router) | router OR boost (val tie; resolves alphabetically to "boost") | Δ ≤ 0.001 |
| MVTec LOCO PatchCore SP | 0.734 (boost) | boost (boost wins clearly on val) | likely unchanged |
| VisA RGB+edge SP | 0.866 (boost) | boost OR router (val tie; likely "boost" alphabetically) | Δ ≤ 0.001 |
| Real3D-AD PCA+depth SP | 0.566 (boost) — Family C exploratory | boost (boost beats router by ~0.03) | likely unchanged |
| UNSW-NB15 flow/conn/context | 0.989 (router) | router OR boost (4-decimal tie; likely "boost" alphabetically) | likely unchanged |

The headline numbers are not expected to change materially. The integrity gain is large: no rule-4 violation, no pre-registration theatre, no test-winner comparator selection, and the existing inspected results are correctly labelled as audited reanalysis rather than confirmatory.

---

## 7. Validation procedure

The Phase-1 implementation must include:

1. **Per-cell validation-frozen RGA+ choice logged**:
   - JSON field: `clean_metric_summary.rga_plus_headline = {"chosen_head": "router"|"boost", "val_roc_auc": float, "test_roc_auc": float}`.
2. **Per-cell validation-frozen primary comparator logged**:
   - JSON field: `clean_metric_summary.primary_comparator = {"name": "<method_name>", "selection_rule": "P2", "val_roc_auc": float, "test_roc_auc": float}`.
3. **Unit test** asserting:
   - `chosen_head ∈ {"router", "boost"}` for every cell.
   - `val_roc_auc` for `chosen_head` is `>=` the other candidate's val ROC-AUC.
   - `primary_comparator.name` is the baseline with the highest seed-averaged validation ROC-AUC for the cell.
   - **No** field named `primary_comparator_pre_declared` is set for any Family A / B / C cell.
4. **Emit-script update**: `emit_milestone2_cross_benchmark.py:_cell()` reads `rga_plus_headline.test_roc_auc` and `primary_comparator.test_roc_auc`. The previous `max(rga_router, rga_boost)` and `argmax_test_best_non_router` lines must be deleted.
5. **Regression test** on `tests/test_paper_asset_metadata.py`:
   - The master comparison table caption must include "validation-frozen" or "validation-selected".
   - The comparator description must read from the JSON's `primary_comparator.name` field.
   - The caption / prose **must not** contain "pre-declared", "pre-registered", "best non-router", or "confirmatory" in the context of existing cells. (These tokens are allowed only inside Family D prose.)
   - The caption must include "audited reanalysis" or "audited inferential summary" or equivalent.

---

## 8. Backward-compatibility note

When the Phase-1 fix lands, the existing per-seed `table_1_clean_performance` entries already contain both `rga_meta_router` and `rga_boosted_fusion` test metrics, so no new training run is required for the master comparison table — only the post-processing emit step changes.

The validation ROC-AUC for the router and boost is also already logged per seed. Therefore the val-frozen selection (RGA+ and primary comparator) can be computed from the existing JSONs without re-running any experiment.

This means **the Phase-1 fix to AR-1 / AR-2 / AR-11 / AR-12 is a manuscript + emit-script change, not a re-run**. The Fisher-independence repair (seed-averaged single DeLong + paired sample bootstrap) also reads from existing JSONs.

The only re-runs that may be required are downstream of the label/metric semantics audit (issue E P0): if the audit identifies a code bug that affects test predictions, the affected cells must be re-run. The audit is gated as the first Phase-1 step.

Polarity-flip removal (AR-13) is a post-processing pass — the runner's `polarity_info` is logged, so the emit script can recompute metrics with `flip_required=False` for every method without re-running. Optionally the runner is patched to stop applying the flip in the primary path; that patch is a small Phase-1 code change.

---

## 9. Phase 0.6 supersession statement

This document supersedes the Phase 0.5 `HEADLINE_METHOD_POLICY.md` on every conflict. Phase 0.5's pre-declaration registry for existing cells is rescinded. The registry in §4 above is reserved exclusively for Family D future locked confirmatory replication cells, and no row in that registry is filled until the corresponding Family D test partition is locked and verified untouched.
