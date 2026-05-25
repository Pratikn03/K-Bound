# ELARA Audit Report — Phase 0.6: Locked Audited-Reanalysis Policy

**Auditor role:** senior empirical-ML reproducibility engineer + skeptical conference reviewer.
**Scope:** ELARA / RGA / RGA+ manuscript pair and the JSON artifacts that back them.
**Status:** **no implementation, results, or LaTeX modified** during Phase 0, Phase 0.5, or this Phase 0.6 revision.
**Date of original audit:** 2026-05-22. **Phase 0.5 revision:** 2026-05-22. **Phase 0.6 lock:** 2026-05-23 (this revision).
**Repo HEAD at audit time:** `b88ad3b` (Phase-D closure commit).

---

## Revision history

- **Phase 0 (initial).** Four submission-blocking issues identified (B / C / D / Fisher-independence). Five audit artifacts produced.
- **Phase 0.5.** Issue E escalated to P0; bootstrap-over-seeds rejected as primary inference; comparator registry added; three analysis families (A / B / C) introduced; pairing-strength column added; polarity reframed as a validation-only diagnostic.
- **Phase 0.6 (this revision).** Three final policy problems closed before Phase 1 may begin. Phase 0.6 supersedes Phase 0.5 on every conflict. See [`PHASE_0_6_FINAL_POLICY_LOCK.md`](PHASE_0_6_FINAL_POLICY_LOCK.md) for the complete change list and verification output.

The most consequential Phase-0.6 changes are:
1. The policy for already-observed cells is renamed to a **Locked Audited-Reanalysis Policy**. The words "pre-registered" and "confirmatory" are reserved exclusively for Family D future replication.
2. The comparator pre-declaration registry for existing cells (Phase 0.5) is rescinded; existing cells use **validation-frozen primary comparator** selection only.
3. Family B's mechanism subset is reclassified: B1 (zero-attack tau=0.66) and B2 (max-attack tau=0.66) are audited mechanism endpoints (Holm K=2); B3 (k-of-D), B4 (tau sweep), B5 (component ablation) are descriptive only and carry no Holm-corrected significance claim.
4. Seed-averaged predictions are explicitly an **ensemble predictor**; the audited inferential summary applies to the ensemble, not to a single trained model.
5. The polarity flip is **locked**: no flipping in primary metrics for any cell. The synthetic-anomaly val probe remains as a descriptive validation-only score-orientation diagnostic only.
6. **Family D** future locked confirmatory replication structure introduced.

---

## Executive verdict

The repository is well-structured and the experimental machinery is real, but **five submission-blocking (P0 Fatal) issues** must be closed before any external review, plus a Fisher-independence statistical defect that must also be repaired, plus the new Phase-0.6 framing requirements (no pre-registration / no confirmatory claim for existing cells; no polarity flip in primary metrics; Family B sweep / ablation reclassification):

1. **Rule-4 violation: RGA+ is defined as `max(router, boost)` selected on the test-set ROC-AUC** (`src/scripts/emit_milestone2_cross_benchmark.py` lines 113–115). Every headline RGA+ number in the master comparison table — including the 0.738 / 0.739 abstract claim — is a test-set oracle selection.
2. **Multiplicity family mis-stated:** caption and prose say "Holm-Bonferroni correction across all nine evaluated cells", the rendered table contains 11 rows, and Holm is computed across all 11 (`BENCHMARKS` constant, lines 164–176). Issue D.
3. **Self-contradictory significance reading:** within the same §`sec:cross-benchmark-master`, the rendered Holm column reports `p_Holm = 1.5e-05` and `p_Holm = 1.8e-04` for MVTec LOCO SP and VisA SP, while the prose says the same two cells lose significance after Holm (`p_Holm = 0.116` and `0.199`). Issue C.
4. **Fisher's-method independence violated:** per-seed DeLong p-values share the same test fold; only the training seed varies. Combining them with Fisher under an i.i.d. assumption overstates significance. **Rule 7.**
5. **Canonical-MVTec metric semantics are unexplained (issue E, P0).** With `positive_fraction_actual = 0.2244` and label=1 as the anomaly class, a constant-score predictor should give average precision near 0.2244, not 0.7835. **No canonical MVTec metric claim is allowed in the paper or the thesis until the label / metric semantics audit identifies and fixes the root cause.**

In addition Phase 0.6 introduces five new framing requirements that must be met before Phase 1:
- No "pre-registered" or "confirmatory" label for already-observed cells (**AR-11**).
- No "RGA+ beats every baseline" claim for existing cells (**AR-12**).
- No polarity flipping in primary metrics for any cell (**AR-13**).
- No single-method ROC-AUC reported without the ensemble qualifier when the number comes from seed-averaged predictions (**AR-14**).
- No Holm-corrected significance claim for B3 / B4 / B5 sweeps and ablations from already-observed results (**AR-15**).

The remaining **nine** issues (A, F, G, H, I, J, K, L, M) are real but most are manuscript-only fixes that do not require re-running experiments.

---

## Per-issue table (issues A–M)

| ID | One-line | Severity | Type | Re-run needed? |
|---|---|---|---|---|
| A | Abstract MVTec SP numbers (0.738 / 0.735 / 0.740) | Major (manuscript) | Wrong comparator (Tent vs SAR) plus test-winner comparator selection | No |
| B | RGA+ = `max(router, boost)` on test set | **P0 Fatal** | Rule 4 + Rule 5 violation | No (re-process JSONs) |
| C | Holm p-values disagree between table and prose | **P0 Fatal** | Internal contradiction | No (recompute Holm correctly) |
| D | Caption says 9 cells, table has 11 | **P0 Fatal** | Multiplicity mis-statement | No |
| E | Canonical PR-AUC / ECE / Brier = 0.7835 (incompatible with prevalence 0.2244) | **P0 Fatal** | Label / metric semantics unknown | Possibly — label/score-orientation audit may force re-run |
| F | Polarity flip applied only to static + RGA (now superseded by AR-13) | Major (now also covered by AR-13 lock) | Asymmetric across methods | No (post-processing pass) |
| G | Thesis says clean adapt rate 0.032, table shows 0.000 | Minor | Stale prose | No |
| H | "Interventional ATE under SCM" overclaim | Major | Causal-language overreach | No (reframe) |
| I | Conference / thesis feature drift | Minor | Out-of-sync prose | No |
| J | Baseline list inconsistent across abstract / methods / tables / code | Major | Comparator coverage drift | No |
| K | FGSM/PGD table has duplicate ε headers | Minor | Formatting | No |
| L | Real3D "FPFH+depth" naming stale (now PCA shape stats) | Minor | Stale label | No |
| M | Healthcare replay framing | Minor (verify only) | Language audit | No |

Plus the Fisher-independence defect (separate from issues A–M) and the five Phase-0.6 framing requirements above.

---

## Detailed findings

### A — Abstract MVTec SP numbers

**Locations**
- `docs/research/PAPER_DRAFT_v1.tex:71-72`, `104-105`, `1207-1210`, `2175-2176`
- Source artifact: `experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json`
  - boost = 0.7383, router = 0.7390, SAR = 0.7354, Tent = 0.7353, TTT = 0.7237, RF = 0.7010

**Status:** the *numerical* values are within rounding of the JSON. But:
1. The **best non-router baseline is SAR (0.7354), not Tent (0.7353)**. The abstract's "above Tent (0.735)" is the wrong comparator and is silent about SAR. Tent and SAR are statistically tied at the 4th decimal.
2. The 0.738 / 0.740 boost / router selection is the rule-4 violation (see B).
3. The "best non-router" is itself chosen by reading the test winner (forbidden under Phase 0.6 — AR-2).
4. The abstract framing as "boosted fusion … above Tent" implies a confirmatory comparison on an already-inspected cell (forbidden under Phase 0.6 — AR-11 / AR-12).

**Severity:** Major manuscript fix.

**Proposed correction (Phase 1):** under Phase 0.6's locked audited-reanalysis policy:
- Report a single validation-frozen RGA+ head (per `HEADLINE_METHOD_POLICY.md` §1).
- Report the audited inferential summary against the **validation-frozen primary comparator** (Rule P2; per `HEADLINE_METHOD_POLICY.md` §3) — not against the test-winner "best non-router".
- Reframe the abstract sentence as descriptive: "the validation-frozen RGA+ head reaches 0.738 on the audited test fold; the validation-frozen primary comparator reaches \<value\>. The audited inferential summary is \<p_Holm\>; this is not a confirmatory replication."

---

### B — RGA+ defined as MAX(router, boost) on test set

**Locations**
- `src/scripts/emit_milestone2_cross_benchmark.py:113-119`:
  ```python
  rga_plus_candidates = [v for v in (rga_router, rga_boost) if v is not None]
  rga_plus = max(rga_plus_candidates) if rga_plus_candidates else None
  ```
- The variables `rga_router` and `rga_boost` come from `clean_metric_summary.*.roc_auc.mean` which is the **test-fold** seed-averaged ROC-AUC.
- Caption at `docs/research/PAPER_DRAFT_v1.tex:1537`: `RGA+ = max(router, boost)`.

**Status:** confirmed. Every headline RGA+ value in the master comparison table is a test-set oracle.

**Severity:** **P0 Fatal.** Direct Rule 4 + Rule 5 violation.

**Proposed correction (Phase 1):** per `HEADLINE_METHOD_POLICY.md` §1, select between router and boost on validation ROC-AUC, frozen before test. The runner already logs both the val ROC-AUCs and the test metrics. The Phase-1 fix is a post-processing change to the emit script + a new `rga_plus_headline` JSON field; no re-run required.

---

### C — Holm p-values disagree between table and prose

**Locations**
- Table `docs/research/tables/milestone2_cross_benchmark.tex`:
  - MVTec LOCO PatchCore SP: `p_Holm = 1.5e-05`
  - VisA RGB+edge SP: `p_Holm = 1.8e-04`
- Prose at `docs/research/PAPER_DRAFT_v1.tex:1588-1591`:
  > "VisA `p_raw = 0.050`, MVTec LOCO `p_raw = 0.023` … lose strict significance after Holm correction across nine cells (`p_Holm = 0.199` and `0.116` respectively)."

**Severity:** **P0 Fatal.** Internal contradiction in the same section.

**Proposed correction (Phase 1, manuscript-only):** delete the contradictory prose sentence; rewrite to read from the table. Recompute under Phase 0.6's audited-reanalysis rule: Family A K=5 audited-primary-reanalysis cells only; ensemble seed-averaged DeLong + paired sample bootstrap; label as audited inferential summary (not confirmatory).

---

### D — Caption says 9 cells, table has 11

**Locations**
- Caption at `docs/research/PAPER_DRAFT_v1.tex:1543`: "Holm-Bonferroni correction across all nine evaluated cells".
- `BENCHMARKS` constant at `src/scripts/emit_milestone2_cross_benchmark.py:164-176`: 11 rows.
- Rendered table at `docs/research/tables/milestone2_cross_benchmark.tex`: 11 data rows.

**Severity:** **P0 Fatal.**

**Proposed correction (Phase 1):** apply the four-family split (Family A audited-primary K=5; Family B audited-mechanism K=2; Family C K=0; Family D K to be locked when D rows exist). Holm correction is applied within Family A only for the master comparison subsection. Update the caption to "Holm-Bonferroni correction within Family A audited-primary-reanalysis cells (K=5)".

---

### E — Canonical PR-AUC / ECE / Brier semantics unexplained (P0 Fatal, retained)

**Locations**
- `experiments/fusion/mvtec3d_patchcore_results.json` (canonical one-class):
  - `static_attention`: ROC=0.491, PR=0.771, ECE=0.376, Brier=0.443
  - `craf_attention`: ROC=0.514, PR=0.786, ECE=0.400, Brier=0.443
  - `rga_meta_router`: ROC=0.514, PR=0.786, ECE=0.443, Brier=0.443
  - `rga_boosted_fusion`: ROC=0.500, PR=0.784, ECE=0.784, Brier=0.784 ← three identical values
- Metadata: `positive_fraction_actual = 0.22442653440793553`.

**Why this is P0 Fatal:** with `pos_label = 1` (anomaly) and `prevalence = 0.2244`, a constant-score predictor must have average precision near 0.2244, not 0.7835. The reported PR-AUC of 0.7835 is approximately `1 - 0.2165`, which is the expected average precision of a constant predictor only if `pos_label` is being interpreted as the negative class (label=0, normals). The three identical PR=ECE=Brier=0.7835 values strongly suggest a degenerate constant predictor combined with a mis-set positive class.

**Possible root causes (each must be ruled out by the Phase-1 audit):**
1. Label inversion — the CSV treats normal=1, anomaly=0 (instead of the conventional anomaly=1).
2. `pos_label` mismatch — the metric helper does not specify `pos_label`, and the default differs from the assumed convention given the score's direction.
3. Score-orientation mismatch — model outputs higher-for-normal scores; PR-AUC computed as if higher-for-anomaly.
4. Stale artifact-to-table linkage — the table is populated from a different JSON than the prose claims.
5. Incorrect metric calculation — the helper conflates per-class average precision and reports an average over both classes, producing a number near 0.78 on a 22%-positive dataset.

**Required Phase-1 step (gated before any prose rewrite of canonical metrics):**
1. Create `src/scripts/audit_canonical_label_semantics.py`.
2. Re-compute PR-AUC, ECE, Brier under both `pos_label=1` and `pos_label=0` and under both score orientations.
3. Re-compute the same metrics for a synthetic constant-score predictor at the same prevalence.
4. Verify whether the JSON's values match a known causal pattern.
5. Emit `experiments/audit/canonical_label_semantics.json` and a narrative report.
6. **No canonical PR-AUC / ECE / Brier value may be promoted in the paper or thesis until the audit identifies the root cause.**

**Manuscript constraint:** until the audit completes, every canonical PR-AUC / ECE / Brier figure in the paper / thesis must be replaced with `\emph{see Appendix: canonical-metric semantics audit; values pending verification}`, or removed entirely. ROC-AUC at chance level is the only canonical metric that remains interpretable.

---

### F — Polarity flip applied only to static + RGA (now superseded by AR-13)

**Phase 0.6 update:** Phase 0.5 reframed the polarity flip as a "validation-only score-orientation diagnostic" and left the choice between (A) no flipping and (B) symmetric flipping as a user decision. **Phase 0.6 locks the decision: no polarity flipping in primary metrics (AR-13).**

**Locations**
- `src/scripts/run_breakthrough_experiment.py:2078-2082`:
  ```python
  if polarity_info["flip_required"]:
      static_val_probs = 1.0 - static_val_probs
      static_probs = 1.0 - static_probs
      craf_val_probs = 1.0 - craf_val_probs
      craf_probs = 1.0 - craf_probs
  ```

**Audit of per-seed flip decisions:**
- MVTec 3D PatchCore canonical: 3 of 5 seeds flipped (probe AUROC 0.49–0.53, hovering at threshold).
- MVTec 3D PatchCore SP: 0 of 30 seeds flipped (probe AUROC 0.56–0.59).

**Phase 0.6 locked correction (Phase 1):**
1. Remove the four lines that apply the flip to `static_*_probs` and `craf_*_probs`.
2. Retain the probe call (`_calibrate_polarity`) and the JSON log.
3. Report the per-seed flip log descriptively as a validation-only score-orientation diagnostic.
4. Any orientation-correcting deployment variant must be a separately named method (e.g. `RGA-orient`), never grafted into `static` or `RGA`.
5. Add a regression test asserting no Family A or Family B primary metric is computed on flipped predictions.

**Re-run requirement:** none. The flip can be removed in a post-processing pass because the runner logs both the probe decision and the unflipped predictions in earlier pipeline stages. If re-emission is preferred over post-processing, only the affected canonical cells (where flip_required was True for any seed) need re-emission.

---

### G — Thesis adaptation rate 0.032 vs 0.000

**Locations**
- `docs/research/THESIS_CHAPTER_v1.tex:775`: "At τ=0.66, the clean adaptation rate is 0.032."
- `docs/research/tables/elara_tau_sweep_results.tex` row "clean & 0.66": Adapt rate = 0.000.

**Severity:** Minor. Phase 0.6 reclassifies the tau sweep as Family B4 (descriptive-only); the prose number should read from the regenerated table.

**Proposed correction (Phase 1, manuscript-only):** update the prose to 0.000.

---

### H — "Interventional ATE under SCM" overclaim

**Locations**
- `docs/research/PAPER_DRAFT_v1.tex:2030-2102` (subsection "Causal Reliability Attribution").

**Status:** confirmed mis-framing. The experiment is a model-response sensitivity, not a real-world causal identification.

**Severity:** Major.

**Proposed correction (Phase 1, manuscript-only):**
1. Rename subsection (default: "Model-Response Sensitivity to Per-Domain Reliability"; alternatives "Counterfactual Domain-Reliability Perturbation" or "Per-Domain Reliability Sensitivity Analysis"; user decision still open).
2. Drop `do()` notation; use `Δy_d(r_d → bar_r_d^val)`.
3. Move the Double-ML formulation into an appendix or remove it.
4. Replace "interventional ATE under an SCM" with "the counterfactual response of the fused output when domain $d$'s reliability is set to its validation-fold mean".

---

### I — Conference / thesis feature drift

**Locations**
- Paper has expanded EATA / SAR / k-of-D / FGSM/PGD / T2-T3-T5 validation / fourth-benchmark-scaffold subsections.
- Thesis abstract (lines 47–82) does not mention any of these explicitly.

**Severity:** Minor.

**Proposed correction (Phase 1, manuscript-only):** add a short paragraph to the thesis chapter pointing at the corresponding paper sections.

---

### J — Baseline list inconsistent across abstract / methods / tables / code

**Comparator coverage:**

| Source | RF | MLP | LFE | Tent | TTT | EATA | SAR | Conf-mean | Static | RGA | router | boost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper abstract | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Paper §I.B Strong baselines | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Paper §Benchmark Construction | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | (impl) | ✗ | ✗ |
| Master comparison table | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Code `baselines.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (sep) | (sep) | (sep) | (sep) |

**Severity:** Major.

**Proposed correction (Phase 1, manuscript-only):** add EATA / SAR to the abstract's baseline list and the §I.B contributions list. Add a methods subsection describing the two added adapters. Under Phase 0.6 the prose must use descriptive baseline-ranking language (no "RGA+ beats every baseline" claim).

---

### K — FGSM/PGD table duplicate ε headers

**Locations**
- `docs/research/tables/gradient_adversarial.tex`: the four ε values are repeated twice without an intermediate "FGSM | PGD" header row.

**Severity:** Minor (formatting).

**Proposed correction (Phase 1, manuscript-only):** add a multicolumn header row:
```latex
& & \multicolumn{4}{c}{\textbf{FGSM (1-step)}} & \multicolumn{4}{c}{\textbf{PGD (10-step)}} \\
```

---

### L — Real3D descriptor naming

**Locations**
- Master comparison protocol label: "Real3D-AD | FPFH+depth supervised".
- Actual descriptor: PCA shape statistics + pairwise-angle histogram + radial moments.

**Severity:** Minor (label naming).

**Proposed correction (Phase 1, manuscript-only):** rename the protocol to "PCA shape + depth supervised" everywhere — paper + thesis + emit scripts. Real3D is Family C exploratory under Phase 0.6.

---

### M — Healthcare replay framing

**Locations**
- `docs/research/THESIS_CHAPTER_v1.tex:991` says "should not be interpreted as deployment evidence."
- `THESIS_CHAPTER_v1.tex:1731` mentions "deployment-grade calibration monitor" in an appendix.

**Status:** the thesis already correctly frames healthcare as retrospective replay; the "deployment-grade calibration monitor" language at line 1731 is the package description, not the healthcare claim.

**Severity:** Minor (verify only).

**Proposed correction (Phase 1, manuscript-only):** confirm every healthcare paragraph uses "local retrospective replay" framing; no other change anticipated.

---

## Statistical-method blocker: Fisher independence (Rule 7)

The `emit_milestone2_cross_benchmark.py:_fisher_combine` combines per-seed DeLong p-values into a Fisher chi-square statistic. The seeds use the same test fold; the p-values are correlated. Fisher's independence assumption is violated.

**Phase 0.6 locked replacement:** seed-averaged predictions per method per cell + one paired DeLong test per cell + paired bootstrap **over test samples** (not seeds) for the 95% CI. Results are labelled "ensemble audited analysis" per AR-14. Seed mean ± SD reported descriptively only.

---

## File-by-file change inventory

This list is the *proposed* set of files for Phase 1. **No file has been edited in Phase 0, Phase 0.5, or Phase 0.6 outside of `docs/research/audit/`.**

| File | Issue(s) | Type of change |
|---|---|---|
| `src/scripts/audit_canonical_label_semantics.py` | E | **NEW** — must exist and produce a clean verdict before any canonical metric prose is rewritten |
| `src/scripts/emit_milestone2_cross_benchmark.py` | B, C, D, F, J | Replace `max(router, boost)` with validation-frozen selection; apply Family-A K=5 Holm; Fisher → seed-averaged single DeLong + paired sample bootstrap; comparator is `validation-frozen` (P2); no flip in primary metrics; descriptive baseline ranking |
| `src/scripts/run_breakthrough_experiment.py` | B, F | Log validation-frozen RGA+ choice as a single named field; log validation-frozen primary comparator; remove the flip from the primary-prediction path (AR-13); retain the probe call for logging |
| `docs/research/PAPER_DRAFT_v1.tex` | A, C, D, E, F, G, H, I, J, K, L | Abstract / methods / §sec:cross-benchmark-master / §sec:causal-attribution / FGSM-PGD table reference / Real3D label / EATA/SAR coverage / canonical-metric placeholders / Family B sweep / ablation framing |
| `docs/research/THESIS_CHAPTER_v1.tex` | E, G, I, L, M | τ=0.66 prose; sync to paper expansions; Real3D label; healthcare verification; canonical-metric placeholders; tau-sweep descriptive framing |
| `src/scripts/emit_gradient_adversarial_table.py` | K | Add FGSM/PGD multicolumn header |
| `tests/test_paper_asset_metadata.py` | A, C, D, E, F, AR-11..AR-15 | Add forbidden-substring checks ("max(router", "deployment-grade sanity check", "pre-declared comparator" outside Family D, "RGA+ beats every baseline", "confirmatory" outside Family D, "9 evaluated cells"); assert no canonical PR-AUC / ECE / Brier numbers without semantics-audit reference; assert primary metrics not computed on flipped predictions |
| `docs/research/audit/*` | All | Revised audit artifacts (this report + 4 sibling files + Phase 0.5 summary + Phase 0.6 lock) |

---

## Phase-0.6 deliverable index

| File | Purpose |
|---|---|
| `docs/research/audit/ELARA_AUDIT_REPORT.md` | This report (Phase 0.6 revision). |
| `docs/research/audit/CLAIM_LEDGER.csv` | One row per headline claim; updated with Phase 0.6 audited-reanalysis language and new rows for AR-11/AR-12/AR-13 framing requirements. |
| `docs/research/audit/EXPERIMENT_REGISTRY.csv` | One row per (benchmark, protocol) with `analysis_family`, `confirmatory_or_exploratory`, `pairing_strength`, `allowed_claim`, `forbidden_claim`. Family D rows added. |
| `docs/research/audit/STATISTICAL_ANALYSIS_POLICY.md` | Phase 0.6 lock: four families (A audited-primary K=5; B audited-mechanism K=2 + B3-B5 descriptive; C exploratory; D future confirmatory); seed-averaged ensemble interpretation; polarity-flip lock (no flipping); audited-reanalysis language. |
| `docs/research/audit/HEADLINE_METHOD_POLICY.md` | Phase 0.6 lock: validation-frozen RGA+ + validation-frozen primary comparator for existing cells; pre-declared comparator registry reserved for Family D only. |
| `docs/research/audit/PHASE_0_5_REVISION_SUMMARY.md` | Phase 0.5 historical record (carries a supersession header pointing at Phase 0.6). |
| `docs/research/audit/PHASE_0_6_FINAL_POLICY_LOCK.md` | Phase 0.6 final lock document: files changed, policy decisions locked, remaining P0 blockers, Family D design, Phase-1 go/no-go gate, file-export verification output. |
