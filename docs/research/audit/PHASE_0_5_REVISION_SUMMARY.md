# Phase 0.5 Revision Summary

> **SUPERSEDED BY PHASE 0.6 — 2026-05-23.** See [`PHASE_0_6_FINAL_POLICY_LOCK.md`](PHASE_0_6_FINAL_POLICY_LOCK.md) for the active policy lock. Phase 0.6 supersedes this document on every conflict. The most consequential changes vs Phase 0.5 are:
>
> 1. Existing-cell language switched from "pre-registered / confirmatory" to "Locked Audited-Reanalysis Policy / audited primary reanalysis".
> 2. The Phase-0.5 pre-declaration comparator registry for existing cells (SAR for MVTec 3D SP, Tent for LOCO SP, RF for VisA SP and UNSW, etc.) is **rescinded**. Existing cells use the validation-frozen comparator (Rule P2) only. Pre-declaration is reserved for Family D.
> 3. **Family D** (future locked confirmatory replication) introduced. Only Family D may use the words "pre-registered" and "confirmatory".
> 4. Family B mechanism subset reclassified: B1 (zero-attack tau=0.66) + B2 (max-attack tau=0.66) are audited mechanism endpoints with Holm K=2; **B3 (k-of-D), B4 (tau sweep), B5 (component ablation) carry no Holm-corrected significance claim** under Phase 0.6.
> 5. Polarity-flip decision **LOCKED**: no flipping in primary metrics (Phase-0.5 had left this open as choice A vs B).
> 6. Seed-averaged predictions are explicitly an **ensemble predictor**; the audited inferential summary applies to the ensemble, not to a single trained model.
> 7. New anti-rules AR-11 (no pre-registered/confirmatory framing for existing cells), AR-12 (no "RGA+ beats every baseline"), AR-13 (no polarity flipping in primary metrics), AR-14 (ensemble qualifier required for seed-averaged metrics), AR-15 (no Holm-corrected significance claim for B3-B5 from inspected sweeps), AR-16 (no reuse of inspected test partitions for Family D).
>
> The Phase 0.5 content below is retained as a historical record. **Read the Phase 0.6 lock document for the active policy.**

**Status (historical):** read-only revision of the Phase-0 audit artifacts. No code, LaTeX, results, tables, figures, or PDFs were modified.
**Date:** 2026-05-22 (same session as Phase 0).
**Inputs:** the seven required corrections in the user's Phase-0.5 instruction.
**Outputs (overwritten):**
- [`ELARA_AUDIT_REPORT.md`](ELARA_AUDIT_REPORT.md)
- [`CLAIM_LEDGER.csv`](CLAIM_LEDGER.csv)
- [`EXPERIMENT_REGISTRY.csv`](EXPERIMENT_REGISTRY.csv)
- [`HEADLINE_METHOD_POLICY.md`](HEADLINE_METHOD_POLICY.md)
- [`STATISTICAL_ANALYSIS_POLICY.md`](STATISTICAL_ANALYSIS_POLICY.md)
- **New:** [`PHASE_0_5_REVISION_SUMMARY.md`](PHASE_0_5_REVISION_SUMMARY.md) (this file).

---

## 1. What changed from Phase 0

| # | Change | Files touched | Rationale |
|---|---|---|---|
| 1 | Issue E (canonical PR-AUC / ECE / Brier = 0.7835 with prevalence 0.2244) escalated from Major to **P0 Fatal**. New policy block: no canonical PR-AUC / ECE / Brier claim allowed until a label/metric-semantics audit identifies the root cause. | `ELARA_AUDIT_REPORT.md`, `CLAIM_LEDGER.csv` (rows C011 escalated + C021 added), `STATISTICAL_ANALYSIS_POLICY.md` §5 | Constant-score predictor at prevalence 0.2244 must have average precision near 0.2244, not 0.7835. Possible causes (label inversion, `pos_label` mismatch, score-orientation mismatch, stale linkage, metric bug) cannot be guessed; they must be diagnosed by an audit script before any prose update. |
| 2 | Primary inference rule revised. Removed *paired bootstrap over seed-level AUROC differences* as the primary test. Replaced with: seed-averaged test predictions per method per cell, one DeLong p-value per cell on the averaged predictions, paired bootstrap **over test samples** (not seeds) for a 95% CI of the AUROC difference. Seed mean ± SD is descriptive instability evidence only. Holm applied to the single primary p-value per pre-specified confirmatory cell. | `STATISTICAL_ANALYSIS_POLICY.md` §2, `ELARA_AUDIT_REPORT.md` Fisher-independence section | Seeds reuse the same test fold; bootstrap over seeds estimates training-randomness variability, not test-population uncertainty, and is unstable at n=5 seeds. The seed-averaged single-test approach is the right honest replacement. |
| 3 | New non-RGA comparator policy: primary comparator must be pre-declared per benchmark family or selected on validation only. No confirmatory p-value may compare against a baseline selected by reading the test winner. A comparator registry was added to `HEADLINE_METHOD_POLICY.md` §3.4 with columns `benchmark_family`, `candidate_baselines`, `comparator_selection_rule`, `primary_comparator_status`, `rationale`. | `HEADLINE_METHOD_POLICY.md` §3, `STATISTICAL_ANALYSIS_POLICY.md` §2.1 | Phase 0 fixed the router-vs-boost selection but left the "best non-router" comparator chosen by reading the test winner — also a rule-4 violation. The pre-declaration registry closes the loophole. |
| 4 | Experiment registry and statistical policy reconciled. Three explicit families: A (public cross-benchmark performance), B (mechanism stress), C (exploratory audits). Real3D moved to Family C (exploratory) until 30-seed re-run; ELARA-Bench-LA moved to Family B (mechanism stress, not cross-domain superiority); noise-floor and held-out-attack moved to Family C. Every registry row now has `analysis_family`, `confirmatory_or_exploratory`, `allowed_claim`, `forbidden_claim`. | `EXPERIMENT_REGISTRY.csv`, `STATISTICAL_ANALYSIS_POLICY.md` §1 | Phase 0 marked Real3D as primary while the statistical policy excluded it — internal inconsistency. The three-family split makes the boundary between confirmatory performance, mechanism evidence, and exploratory audit explicit, and pre-declares the allowed/forbidden claim per cell. |
| 5 | New `pairing_strength` column in `EXPERIMENT_REGISTRY.csv` with five levels: `independent_modalities`, `naturally_structured_views`, `derived_view_proxy`, `label_aligned_stress_only`, `local_replay`. Policy: derived-view-proxy benchmarks (VisA RGB+edge_proxy, MVTec LOCO RGB+edge_proxy) cannot by themselves support a universal independent-multimodal claim. | `EXPERIMENT_REGISTRY.csv`, `STATISTICAL_ANALYSIS_POLICY.md` §7 | The current paper labels every paired benchmark as "naturally paired" without distinguishing genuine cross-modal pairing (MVTec 3D RGB+depth) from derived-view proxies (VisA RGB+edge). A reviewer who cares about the multimodal claim will read this conflation immediately. The new column makes the distinction explicit. |
| 6 | Polarity-flip reframed. Removed the phrase "deployment-time sanity check" (or "deployment-grade") from the statistical policy. Replaced with **validation-only score-orientation diagnostic**. The current implementation (flips static + RGA but not score-adapter baselines) is now explicitly marked **not acceptable for confirmatory comparison** until repaired. Two acceptable repair paths offered: (A) no flipping in confirmatory analysis, or (B) symmetric val-only flip across all methods downstream of static fusion. | `STATISTICAL_ANALYSIS_POLICY.md` §4, `CLAIM_LEDGER.csv` (row C022 added) | "Deployment-time sanity check" overstates what the flip actually is (a val-only diagnostic computed at evaluation time). The asymmetric flip across methods is the substantive defect; the policy now makes both the naming and the repair explicit. |
| 7 | Arithmetic / language correction. Phase 0 said "four submission blockers and remaining nine issues (A, E, F, G, H, I, J, K, L, M)" — that list has 10 items. After Issue E is escalated, there are **five P0 Fatal blockers (B, C, D, Fisher-independence, E)** and **nine remaining issues (A, F, G, H, I, J, K, L, M)**. The arithmetic now matches. | `ELARA_AUDIT_REPORT.md` executive verdict | Corrected mis-count plus the E-elevation gives the new counts naturally. |

---

## 2. P0 Fatal items (must fix before any external submission)

After Phase 0.5 the P0 Fatal list is:

1. **B — RGA+ defined as `max(router, boost)` on test ROC-AUC.** Test-set oracle selection. Fixable via post-processing existing JSONs (validation ROC-AUC for router and boost is already logged per seed).
2. **C — Holm p-values disagree between table and prose.** Internal contradiction in the same section. Fixable by deleting the contradictory prose and regenerating from the table under the revised inference rule.
3. **D — Caption says 9 cells, table has 11.** Multiplicity family mis-stated. Fixable via the three-family split (Family A K=5 confirmatory; A1/A4/A6 not in K; Family B K=5 separately; Family C K=0).
4. **E — Canonical PR/ECE/Brier = 0.7835 with prevalence 0.2244 (NEW P0).** Gated by `src/scripts/audit_canonical_label_semantics.py`. May force re-run of canonical cells if the audit identifies a code bug.
5. **Fisher-independence statistical defect.** Per-seed DeLong p-values share the same test fold; Fisher overstates significance. Replaced by seed-averaged single DeLong + paired sample bootstrap per cell.

---

## 3. Final policy decisions still requiring user approval

The Phase-0.5 documents pre-declare reasonable defaults for every decision, but the following items are flagged because they are user-policy choices, not auditor decisions:

1. **Polarity-flip repair path:**
   - (A) No flipping for confirmatory analysis. Polarity diagnostic reported separately.
   - (B) Symmetric val-only orientation rule applied to every method downstream of static fusion (static, RGA, RGA-boost, RGA-router, Tent, TTT, EATA, SAR). Per-seed flip logs exposed.
   - **Default if no decision:** (A) — strictly safer; (B) is preferred only if you want to keep the diagnostic and accept the implementation cost.

2. **Family C primary comparators for Real3D / VisA noise-floor / UNSW held-out:**
   - Default: Rule P2 (validation-frozen comparator per cell, no pre-declaration).
   - Alternative: pre-declare the same comparator as the matched Family A cell. Cheaper to implement but weaker scientifically because the small-n Family C cells have different baseline rankings.
   - **Default if no decision:** Rule P2.

3. **Polarity threshold:** keep at val-probe AUROC `< 0.5`, or lower to `< 0.45` to reduce borderline-seed noise on canonical cells.
   - **Default if no decision:** keep at 0.5 (matches existing implementation; Phase-1 flip-symmetry repair is enough).

4. **Family A K count:** K=5 confirmatory only (A2, A3, A5, A7, A8) versus K=8 including the three canonical protocol-diagnostic cells (A1, A4, A6).
   - **Default if no decision:** K=5. Canonical cells are diagnostic by definition; their inclusion in the Holm K-count would dilute corrected p-values on the genuinely confirmatory cells.

5. **Family-A SAR-vs-Tent pre-declaration for MVTec 3D PatchCore SP comparator:** the registry pre-declares **SAR** based on its 30-seed mean being marginally higher than Tent.
   - Alternative: pre-declare Tent (matches the existing paper prose; would require explaining the prose drift but not breaking it).
   - **Default if no decision:** SAR (registry value).

6. **Causal-attribution section rename:** Phase 0.5 proposes "Model-Response Sensitivity to Per-Domain Reliability".
   - Alternative names that satisfy issue H: "Counterfactual Domain-Reliability Perturbation", "Per-Domain Reliability Sensitivity Analysis".
   - **Default if no decision:** "Model-Response Sensitivity to Per-Domain Reliability".

7. **Pairing-strength labels in the manuscript prose:** the policy requires each cell's claim to reiterate the `pairing_strength` classification. This adds about one sentence per cell to the master comparison subsection.
   - **Default if no decision:** add the one-sentence reiteration per cell. (User may approve a single global paragraph instead.)

---

## 4. Phase-1 go/no-go gate

Phase 1 can safely begin after **all seven user decisions above are made** (or the defaults are explicitly accepted), and the following gating audit step is committed to:

**Gating step (must run before any other Phase-1 work):**
1. Create `src/scripts/audit_canonical_label_semantics.py` per `STATISTICAL_ANALYSIS_POLICY.md` §5.
2. Run it against the canonical results JSONs (`mvtec3d_patchcore_results.json`, `mvtec_loco_patchcore_results.json`, `visa_fusion_results.json`).
3. Identify the root cause of the 0.7835 anomaly.
4. Document the cause in `experiments/audit/canonical_label_semantics.json` and a short narrative.
5. **If the cause is a code bug**, fix the bug, re-run canonical cells, and verify supervised-paired numbers are unchanged. Only then proceed with the rest of Phase 1.
6. **If the cause is a label / metric-orientation confusion that affects reporting but not training**, update the metric helper to report PR-AUC / ECE / Brier under the correct convention and re-emit the canonical tables. No training re-run needed.

After the gating audit completes, Phase 1 implementation can proceed in the order:

1. Phase 1.A — implement the validation-frozen RGA+ selection in the emit script + runner.
2. Phase 1.B — implement the pre-declared primary comparator from the registry in the emit script.
3. Phase 1.C — implement the seed-averaged single DeLong + paired sample bootstrap in the emit script.
4. Phase 1.D — implement the polarity-flip repair (A or B per user decision).
5. Phase 1.E — update paper + thesis prose for issues A, C, D, E, F, G, H, I, J, K, L, M per the per-issue corrections in `ELARA_AUDIT_REPORT.md`.
6. Phase 1.F — update `tests/test_paper_asset_metadata.py` with forbidden-substring checks (no "max(router", no "deployment-grade sanity check", no "9 evaluated cells" without K=5 alternative, etc.).
7. Phase 1.G — rebuild PDFs; verify bibliography hygiene; verify no LaTeX undefined references; commit and push.

The Phase-1 workplan above is **deterministic** given the Phase-0.5 policy documents. No further auditor input is required between approval and execution; only user decisions on the seven items in §3 of this summary.

---

## 5. What this revision did **not** change

- No source code touched. `src/scripts/*`, `src/uais/*`, `tests/*`, `configs/*` are unchanged.
- No LaTeX touched. `docs/research/PAPER_DRAFT_v1.tex`, `docs/research/THESIS_CHAPTER_v1.tex` are unchanged.
- No results, tables, figures, or PDFs touched. `experiments/*`, `docs/research/tables/*`, `docs/research/figures/*`, `output/pdf/*` are unchanged.
- No regeneration scripts (`scripts/rebuild_paper.sh`, asset emitters) touched.
- The only files changed in this revision are the six files inside `docs/research/audit/`.
