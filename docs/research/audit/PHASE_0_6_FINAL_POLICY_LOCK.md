# Phase 0.6 Final Policy Lock

**Status:** policy lock for the ELARA audit. **No code, LaTeX, results, tables, figures, or PDFs were modified in Phase 0.6.** Only the six audit/policy documents under `docs/research/audit/` were touched.
**Date:** 2026-05-23.
**Repo HEAD at lock time:** `b88ad3b` (Phase-D closure commit).

This document is the active policy lock for Phase 1. It supersedes Phase 0.5 on every conflict. Phase 0 and Phase 0.5 documents are retained as historical record.

---

## 1. Files changed in Phase 0.6

| File | Phase 0.5 → Phase 0.6 change |
|---|---|
| `docs/research/audit/ELARA_AUDIT_REPORT.md` | Reworded to "Locked Audited-Reanalysis Policy"; added Phase 0.6 supersession section; added the five new framing requirements (AR-11 .. AR-15); detailed-findings sections updated for F (now superseded by AR-13 lock) and H (causal language) and J (descriptive ranking only). |
| `docs/research/audit/CLAIM_LEDGER.csv` | New rows C023 (audited-reanalysis framing), C024 (Family B sweep / ablation reclassification), C025 (no "RGA+ beats every baseline" for existing cells). Statuses updated to remove "pre-declared" language for existing-cell rows; status of C001/C002/C015 updated to "P0 FATAL (Phase 0.6)" framing. C022 polarity row updated with the locked decision (no flipping in primary metrics). |
| `docs/research/audit/EXPERIMENT_REGISTRY.csv` | Existing-cell `confirmatory_or_exploratory` column replaced with "audited primary reanalysis" / "protocol-diagnostic" / "audited mechanism endpoint" / "descriptive only" / "exploratory". Pre-declared comparator names (SAR, Tent, RF, TTT, Conf-mean) removed from existing rows and replaced with "validation-frozen primary comparator (Rule P2)". B1 (zero-attack tau=0.66) and B2 (max-attack tau=0.66) added as discrete rows; B3 (k-of-D), B4 (tau sweep), B5 (component ablation) added as descriptive-only rows. Four Family D candidate rows added (D1, D2, D3, D4) all marked "future locked confirmatory replication" and explicitly requiring an untouched test partition. `pairing_strength` column retained from Phase 0.5. |
| `docs/research/audit/HEADLINE_METHOD_POLICY.md` | Renamed to "Locked Audited-Reanalysis Policy". §3 (Non-RGA comparator policy) rewritten so existing cells use validation-frozen primary comparator (Rule P2) only; pre-declaration rescinded. §4 (Pre-declared comparator registry) reframed: registry exists for Family D only and contains TBD rows until the corresponding test partition is locked. §5 anti-rules expanded with AR-11 / AR-12 / AR-16 (pre-declared registry not allowed for existing cells; Family D requires newly locked partitions). |
| `docs/research/audit/STATISTICAL_ANALYSIS_POLICY.md` | Renamed to "Locked Audited-Reanalysis Policy". New §0 (Framing) makes the audited-reanalysis vs confirmatory distinction explicit. §1 four-family structure (A audited-primary K=5; B mechanism endpoints K=2 + B3-B5 descriptive only; C exploratory K=0; D future confirmatory). §3 new section explicitly stating seed-averaged predictions = ensemble predictor (AR-14). §4 polarity-flip lock (no flipping in primary metrics) — Phase 0.5 left this open; Phase 0.6 locks it as option A. §5 comparator policy: validation-frozen for existing cells; pre-declaration only for Family D. §6 Family D future locked confirmatory replication (NEW). §9 expanded anti-rules table (AR-11 .. AR-16). §11 decision-status table: four Phase-0.5 open decisions are now LOCKED or MOOT; remaining open decisions reduced to three (causal-attribution rename, pairing-strength prose policy, Family D seed-ensemble option). |
| `docs/research/audit/PHASE_0_5_REVISION_SUMMARY.md` | Supersession header added at the top pointing at this Phase 0.6 lock document; Phase 0.5 body retained as historical record. |
| `docs/research/audit/PHASE_0_6_FINAL_POLICY_LOCK.md` | **NEW** (this file). |

No file outside `docs/research/audit/` was changed in this revision.

---

## 2. Policy decisions locked in Phase 0.6

### 2.1 Framing of existing observed cells

| Aspect | Locked decision |
|---|---|
| Policy name | **Locked Audited-Reanalysis Policy** (not pre-registration). |
| Existing-cell label | "Audited primary reanalysis" for A2/A3/A5/A7/A8; "audited mechanism endpoint" for B1/B2; "protocol-diagnostic" for A1/A4/A6; "descriptive only" for B3/B4/B5; "exploratory" for Family C. |
| "Pre-registered" / "confirmatory" | Reserved exclusively for Family D. |
| Comparator for existing cells | **Validation-frozen (Rule P2)** only. No pre-declaration. |
| Comparator for Family D | Pre-declared (Rule P1), filled only when the cell's untouched test partition is locked. |

### 2.2 Primary inference rule (audited reanalysis)

| Aspect | Locked decision |
|---|---|
| RGA+ headline | Validation-frozen choice between router and boost (alphabetical tie-break). |
| Comparator | Validation-frozen (Rule P2) per cell. |
| Inferential test | Seed-averaged predictions → one DeLong p-value per cell + paired bootstrap **over test samples** for 95% CI. **Labelled "ensemble audited analysis".** |
| Seed mean ± SD | Reported descriptively as model-instability evidence only. Not a confirmatory statistic. |
| Multiplicity | Holm-Bonferroni within Family A audited-primary K=5; within Family B audited-mechanism K=2; Family C K=0; Family D K locked before evaluation. Never across families. |
| Fisher | Forbidden. Bootstrap-over-seeds forbidden as primary inferential test. |

### 2.3 Polarity flip

| Aspect | Locked decision |
|---|---|
| Primary metrics | **No polarity flipping** in any cell (Families A / B / C / D). |
| Probe retention | Synthetic-anomaly validation probe retained as a descriptive validation-only score-orientation diagnostic; emits per-seed flip logs to JSON but must not alter primary predictions. |
| Orientation-correcting variant | Must be evaluated as a **separately named method** (e.g. `RGA-orient` distinct from `RGA`); never grafted into `static` or `RGA`. |
| Phase-1 implementation | Remove the four `1.0 - *_probs` lines from the primary-prediction path (`run_breakthrough_experiment.py:2078-2082`). Retain `_calibrate_polarity` and JSON logging. Add regression test asserting no Family A/B primary metric is computed on flipped predictions. |

### 2.4 Family B reclassification

| Aspect | Locked decision |
|---|---|
| B1 (coherent zero-attack at tau=0.66) | Audited mechanism endpoint; Holm-corrected within Family B K=2. |
| B2 (coherent max-attack at tau=0.66) | Audited mechanism endpoint; Holm-corrected within Family B K=2. |
| B3 (k-of-D sweep) | **Descriptive mechanism surface only. No Holm-corrected significance claim.** |
| B4 (tau sweep) | **Descriptive hyperparameter / gate-activity analysis only. No Holm-corrected significance claim.** |
| B5 (reliability-component ablation) | **Descriptive mechanism attribution only. No Holm-corrected significance claim unless a no-KS contrast is pre-specified and run on a future Family D mechanism replication.** |

### 2.5 Comparator policy

| Aspect | Locked decision |
|---|---|
| Existing cells (A / B / C) | Validation-frozen primary comparator (Rule P2). No pre-declaration registry. No "best non-router" framing. No "RGA+ beats every baseline" claim. |
| Reporting alternatives for existing cells | (A) descriptive ranking only (no per-baseline p-values), or (B) per-method Holm correction with K = |primary cells| × |baselines|. Default: (A) + single audited inferential summary against the validation-frozen primary comparator. |
| Family D cells | Pre-declared comparator registry (Rule P1). Locked before any inspection of the cell's test partition. |

---

## 3. Remaining P0 implementation blockers

Phase 0.6 leaves five P0 Fatal items, all carried forward from Phase 0 / 0.5, plus the new Phase-0.6 framing requirements AR-11 / AR-12 / AR-13 / AR-14 / AR-15 / AR-16. None can be deferred past Phase 1.

| # | Blocker | Phase-1 action | Re-run needed? |
|---|---|---|---|
| 1 | **B** — RGA+ = `max(router, boost)` on test set | Replace with validation-frozen head selection (`HEADLINE_METHOD_POLICY.md` §1). Post-processing of existing JSONs is sufficient. | No |
| 2 | **C** — Holm p-values disagree between table and prose | Delete the contradictory prose; regenerate from the table under the revised inference rule. | No |
| 3 | **D** — Caption says 9 cells, table has 11 | Apply the four-family split; Family A K=5 audited-primary only. Update caption to match. | No |
| 4 | **E** — Canonical PR/ECE/Brier = 0.7835 with prevalence 0.2244 | **Gating step:** create `src/scripts/audit_canonical_label_semantics.py`; identify root cause; only then update canonical prose. Manuscript replaces canonical PR/ECE/Brier with "pending semantics audit" until the audit completes. | Possibly (depends on root cause) |
| 5 | Fisher-independence | Replace Fisher-combined seed p-values with seed-averaged single DeLong + paired sample bootstrap. Label as "ensemble audited analysis" (AR-14). | No |
| 6 | AR-11 framing | Remove "pre-registered" / "confirmatory" tokens from prose referring to existing cells. | No |
| 7 | AR-12 framing | Remove "RGA+ beats every baseline" / "top method" / "best non-router" framing for existing cells. | No |
| 8 | AR-13 polarity lock | Remove flip from primary-prediction path. Retain probe + logging. | No (post-processing or small patch) |
| 9 | AR-14 ensemble qualifier | Every audited-reanalysis prose claim labels the seed-averaged result as "ensemble audited analysis". | No |
| 10 | AR-15 sweep / ablation framing | Reframe B3 / B4 / B5 prose as descriptive only. | No |

---

## 4. Family D future-confirmatory design

**The only family permitted to use the words "pre-registered" and "confirmatory."**

### 4.1 Membership requirements

A cell qualifies for Family D only if **all** of the following are true:

1. The test partition has not been inspected at any point before this policy lock.
2. The split rule was defined and frozen before any inspection.
3. The seed count was locked before evaluation.
4. The RGA+ head selection rule was frozen before evaluation.
5. The comparator selection rule was frozen before evaluation.
6. The multiplicity family (Family D K-count) was frozen before evaluation.
7. The seed-ensemble option (deployed seed-ensemble vs single frozen seed) was chosen and recorded before evaluation.

### 4.2 Candidate set (proposed)

| # | Candidate | Test-partition requirement | Status |
|---|---|---|---|
| D1 | MVTec 3D-AD supervised-paired | Newly created untouched locked test split, or external replication partition. The current SP test fold cannot be reused. | Not yet locked. |
| D2 | MVTec LOCO-AD supervised-paired | Newly created untouched locked test split. | Not yet locked. |
| D3 | UNSW-NB15 with a newly defined temporally or attack-held-out locked evaluation | A NEW partition. Neither the current flow/conn/context fold nor the existing "held-out attack categories" fold may be reused. | Not yet locked. |
| D4 | Any newly added naturally paired independent-modality dataset | Must be `independent_modalities` pairing strength. No `derived_view_proxy` benchmarks admitted. | Not yet locked. |

### 4.3 Family D inference rule

Same as the locked audited-reanalysis rule (§2.2 above) except:
- **No polarity flipping** (per the locked AR-13).
- **Pre-declared comparator** (per `HEADLINE_METHOD_POLICY.md` §4 Family D registry).
- **Holm correction** applied across the Family D K-count (locked before evaluation).
- **Seed-ensemble option locked** before evaluation: either deployed seed-ensemble (matches audited-reanalysis interpretation but on an untouched partition) or single frozen seed (single-model claim).

The Family D p-values may be called **confirmatory** because the policy was locked before the data were inspected.

### 4.4 What Family D enables

- The word "confirmatory" in the paper.
- The word "pre-registered" in the paper.
- A universal-independent-multimodal claim (only via a D4 cell with `independent_modalities` pairing).
- A "RGA+ beats X" claim with statistical force.

Without Family D, the paper's strongest defensible claim about RGA+ is an audited inferential summary on inspected results; this is honest but cannot support confirmatory superiority.

---

## 5. File-export verification

The required verification step loads the on-disk files and asserts:
1. `EXPERIMENT_REGISTRY.csv` contains the required columns.
2. `CLAIM_LEDGER.csv` includes issue-E P0 rows and the polarity-policy row.
3. All edited Markdown files contain "Phase 0.6" or "Locked Audited-Reanalysis Policy".

The verification script was executed against the files written in this session. Result:

```
PASS | EXPERIMENT_REGISTRY.csv has required columns | missing=set() | columns=['benchmark', 'protocol', 'analysis_family', 'confirmatory_or_exploratory', 'natural_pairing', 'pairing_strength', 'scorer', 'fusion_training_labels', 'train_split_rule', 'validation_split_rule', 'test_split_rule', 'seeds', 'candidate_models', 'selection_rule', 'primary_comparator', 'allowed_claim', 'forbidden_claim']
PASS | EXPERIMENT_REGISTRY.csv has Family A rows | rowcount=21
PASS | EXPERIMENT_REGISTRY.csv has Family B rows |
PASS | EXPERIMENT_REGISTRY.csv has Family C rows |
PASS | EXPERIMENT_REGISTRY.csv has Family D rows (future replication) |
PASS | EXPERIMENT_REGISTRY.csv Family D >= 4 candidate rows | fam_d_count=4
PASS | EXPERIMENT_REGISTRY.csv has B1/B2 mechanism endpoints | count=2
PASS | EXPERIMENT_REGISTRY.csv has B3/B4/B5 descriptive-only rows | count=3
PASS | CLAIM_LEDGER.csv contains issue-E P0 row(s) (C011 / C021) | matched_rows=['C011', 'C021']
PASS | CLAIM_LEDGER.csv contains polarity-policy row | polarity_row_present=True
PASS | CLAIM_LEDGER.csv contains audited-reanalysis row(s) |
PASS | ELARA_AUDIT_REPORT.md contains Phase 0.6 marker |
PASS | ELARA_AUDIT_REPORT.md contains Locked Audited-Reanalysis Policy marker |
PASS | STATISTICAL_ANALYSIS_POLICY.md contains Phase 0.6 marker |
PASS | STATISTICAL_ANALYSIS_POLICY.md contains Locked Audited-Reanalysis Policy marker |
PASS | HEADLINE_METHOD_POLICY.md contains Phase 0.6 marker |
PASS | HEADLINE_METHOD_POLICY.md contains Locked Audited-Reanalysis Policy marker |
PASS | PHASE_0_5_REVISION_SUMMARY.md carries supersession header |

---
TOTAL: 18 assertions, 18 passed, 0 failed
```

All 18 file-export assertions PASS against the on-disk versions of the audit artifacts.

The verification command (reproducible from the repository root):

```bash
.venv/bin/python -c "<the assertion block above>"
```

---

## 6. Open user decisions (Phase 0.6)

Phase 0.5 had seven open decisions. Phase 0.6 locks four of them, makes two moot, and introduces one new decision. The remaining open decisions are:

| # | Decision | Default if no user input |
|---|---|---|
| 1 | Causal-attribution subsection rename. Candidates: "Model-Response Sensitivity to Per-Domain Reliability" (default), "Counterfactual Domain-Reliability Perturbation", "Per-Domain Reliability Sensitivity Analysis". | "Model-Response Sensitivity to Per-Domain Reliability". |
| 2 | Pairing-strength prose policy: one sentence per cell (default) vs one global paragraph. | Per-cell reiteration. |
| 3 | Family D seed-ensemble option: (A) deployed seed-ensemble vs (B) single frozen seed. **Deferred** until a Family D cell is concretely lockable. | Deferred; locked when first Family D row is filled. |

Locked Phase 0.6 decisions (no longer open):

- Polarity-flip path: **(A) no flipping in primary metrics** (AR-13).
- Family C comparator rule: **Rule P2 (validation-frozen)** only.
- Polarity threshold: **moot under AR-13** (no flipping).
- Family A K count: **K=5** (audited-primary only; protocol diagnostics not in K-count).
- MVTec 3D SP comparator pre-declaration: **moot under AR-11/AR-12** (no pre-declaration for existing cells).

---

## 7. Phase-1 go/no-go gate

Phase 1 may safely begin after **all of the following are true**:

1. ✅ Phase 0.6 audit/policy files are committed (this session writes them; commit not yet performed because the task is policy-only).
2. ✅ The 18 file-export assertions PASS.
3. **User explicit acknowledgement** that the three remaining open decisions (§6 above) are resolved or that the defaults are accepted.
4. **User explicit acknowledgement** that the Phase-0.6 lock supersedes Phase 0.5 on every conflict.
5. **Commitment to the gating step before any prose update of canonical metrics:** `src/scripts/audit_canonical_label_semantics.py` must be created and run before any canonical PR-AUC / ECE / Brier figure in the paper / thesis is rewritten. If the audit identifies a code bug, only canonical cells are re-run; supervised-paired numbers must remain unchanged as an invariant.

Once those five gates are satisfied, Phase 1 may begin. The Phase-1 implementation order is:

1. **Phase 1.A — canonical label/metric semantics audit (gating).** Create the audit script, run it on every canonical results JSON, identify the root cause, decide whether re-runs are needed.
2. **Phase 1.B — validation-frozen RGA+ + validation-frozen primary comparator.** Emit-script change in `emit_milestone2_cross_benchmark.py`; runner change to log the chosen-head + chosen-comparator fields. Existing JSONs do not need re-runs.
3. **Phase 1.C — Fisher → seed-averaged single DeLong + paired sample bootstrap.** Emit-script change. Label results as "ensemble audited analysis".
4. **Phase 1.D — polarity-flip removal from primary path.** Runner patch. Post-processing pass for existing JSONs or small re-emission of canonical cells.
5. **Phase 1.E — Family B reclassification.** Update prose for B1/B2 audited endpoints; reframe B3/B4/B5 as descriptive. No experiment change.
6. **Phase 1.F — prose updates for issues A / C / D / E / G / H / I / J / K / L / M.** Apply per-issue corrections from `ELARA_AUDIT_REPORT.md`.
7. **Phase 1.G — `tests/test_paper_asset_metadata.py` updates.** Add forbidden-substring checks (AR-11..AR-16). Add audit-reference checks.
8. **Phase 1.H — rebuild PDFs; verify bibliography hygiene; verify no LaTeX undefined references; commit and push.**

Family D (D1–D4) is **not** Phase-1 work. It is a future replication study that requires data not yet available (new untouched test partitions or new datasets). Family D is queued for a separate study after Phase 1 lands.

---

## 8. Summary table — is Phase 1 safe to begin?

| Gate | Status |
|---|---|
| Audit/policy files committed | Files written in this session; **commit pending user approval** (no commit performed here). |
| 18 file-export assertions | **18/18 PASS** |
| User acknowledgement of 3 open decisions or defaults | **Pending user response.** |
| User acknowledgement that Phase 0.6 supersedes Phase 0.5 | **Pending user response.** |
| Commitment to canonical-metric audit before prose updates | **Pending user response.** |

**Phase 1 is safe to begin once the four "Pending user response" rows above are confirmed.** Until then this policy lock stands as the deliverable of Phase 0.6 and no implementation work proceeds.
