# Targeted Scientific-Editing & Numerical Reconciliation Changelog

**Target Drivers:**
- Compact 2-Column: `docs/research/kbound/kbound_submission.tex` $\rightarrow$ `kbound_short_final_draft.pdf` (48 pages)
- Anonymous TMLR Single-Column: `docs/research/kbound/kbound_tmlr.tex` $\rightarrow$ `kbound_tmlr.pdf` (55 pages)

**Date:** September 16, 2026  
**Auditor:** Antigravity AI  
**Pre-edit Recoverable Snapshot:** `docs/research/kbound/snapshot_pre_copyedit_20260916/`  
**Accepted Canonical Result Authority:** `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json` (SHA-256: `66085a363a0a4c28f6eb68aa767bc7d206f4773c52e6dff98b84d4365ba2c4bf`)

---

## 1. Itemized Ledger of Numerical & Presentation Reconciliations

### Item 1: ImageNet-C Tent Action Counts, Regret, and Commitment
- **Affected Locations:**
  - Table 25 (`tab:compact-actions`), `docs/research/kbound/kbound_submission_supplement.tex`, Line 1154
  - Supplementary prose (§B.1), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 541–542
  - Supplementary action audit prose (§J), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 1177–1180
- **Accepted Authority:** `canonical_panel_results.json:56908-57020` (`imagenetc.panel.candidates.tent` at $\kappa=1.0$)
- **Pre-Edit Contradiction:**
  - Table 15 (p. 27): reported KGA regret $0.0139$, adapt regret $0.0191$, freeze regret $0.0145$, commitment $0.0074$, $\mathrm{FA}_u = 0.0000$.
  - Table 25 (p. 37): reported `0 ADAPT, 0 FREEZE, 135 ABSTAIN, 0 false A, FA_u = 0.0000`.
  - Section J prose: claimed "Tent abstains everywhere".
  - Narrative: claimed "ties freeze; no adapts".
  - *Conflict:* With 135 total decisions, commitment $0.0074$ requires exactly 1 committed decision ($1/135 = 0.007407$). A policy that abstains on 100% of cells cannot achieve regret $0.0139$ when always-freeze has regret $0.0145$.
- **Reconciled Resolution:**
  - Table 25 updated to: `ImageNet--C Tent & 135 & 1 & 0 & 134 & 0 & .0000 & 5 seeds $\times$ 27 cells \\`
  - Supplementary §B.1 prose updated: "Tent achieves KGA regret $0.0139$ (vs.\ $0.0191$ for always-adapt and $0.0145$ for always-freeze), issuing $1$ \adapt and $134$ \abstain decisions (commitment $0.0074$, zero false adaptations)."
  - Supplementary §J prose updated: "Tent adapts on only one cell while abstaining on 134".
- **Status:** RESOLVED & VERIFIED.

---

### Item 2: ImageNet-C EATA Regret, Commitment, and Action Counts
- **Affected Locations:**
  - Table 25 (`tab:compact-actions`), `docs/research/kbound/kbound_submission_supplement.tex`, Line 1155
  - Supplementary prose (§B.1), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 542–543
  - Supplementary action audit prose (§J), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 1177–1180
- **Accepted Authority:** `canonical_panel_results.json:54304-54385` (`imagenetc.panel.candidates.eata` at $\kappa=1.0$)
- **Pre-Edit Contradiction:**
  - Table 15: reported KGA regret $0.0031$, adapt regret $0.0001$, freeze regret $0.0342$, $\mathrm{FA}_u = 0.0000$, commitment $0.8000$.
  - Section 8.1 / App B.1 prose: reported stale KGA regret $0.0007$.
  - Table 25: reported `120 ADAPT, 1 FREEZE, 14 ABSTAIN, 1 false A, FA_u = 0.0074`.
  - Section J prose: stated "ImageNet-C EATA has substantial adapt exposure and one false adaptation".
- **Reconciled Resolution:**
  - Table 25 updated to: `ImageNet--C EATA & 135 & 107 & 1 & 27 & 0 & .0000 & 5 seeds $\times$ 27 cells \\`
  - Commitment verified: $(107 + 1) / 135 = 108 / 135 = 0.8000$.
  - Supplementary §B.1 prose updated: "EATA nearly tracks always-adapt but has a larger KGA regret ($0.0031$ vs.\ $0.0001$ for always-adapt and $0.0342$ for always-freeze), committing on $80.0\%$ of cells ($107$ \adapt, $1$ \freeze, $27$ \abstain) with zero false adaptations ($\mathrm{FA}_u = 0.0000$)."
  - Supplementary §J prose updated: "EATA has substantial adapt exposure (107 adapt decisions) and zero false adaptations".
- **Status:** RESOLVED & VERIFIED.

---

### Item 3: ImageNet-C SAR All-Field Reconciliation
- **Affected Locations:**
  - Table 25 (`tab:compact-actions`), `docs/research/kbound/kbound_submission_supplement.tex`, Line 1153
  - Supplementary prose (§B.1), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 539–541
- **Accepted Authority:** `canonical_panel_results.json:55606-55686` (`imagenetc.panel.candidates.sar` at $\kappa=1.0$)
- **Pre-Edit Contradiction:**
  - Table 15: reported KGA regret $0.0348$, adapt regret $0.0529$, freeze regret $0.0319$, $\mathrm{FA}_u = 0.0148$, commitment $0.0741$.
  - Table 25: reported `13 ADAPT, 15 FREEZE, 107 ABSTAIN, 1 false A, FA_u = 0.0074`.
- **Reconciled Resolution:**
  - Table 25 updated to: `ImageNet--C SAR & 135 & 8 & 2 & 125 & 2 & .0148 & 5 seeds $\times$ 27 cells \\`
  - Reconciled fields: $(8 + 2) / 135 = 10 / 135 = 0.07407... \approx 0.0741$; $\mathrm{FA}_u = 2 / 135 = 0.01481... \approx 0.0148$; false $\mathrm{ADAPT} = 2 \le 8$; $8 + 2 + 125 = 135$.
  - Supplementary §B.1 prose: "SAR has KGA/adapt/freeze regrets ($0.0348$ vs.\ $0.0529$ vs.\ $0.0319$), with $8$ \adapt, $2$ \freeze, and $125$ \abstain decisions ($2$ false adaptations among $135$ cells, $\mathrm{FA}_u = 0.0148$)."
- **Status:** RESOLVED & VERIFIED.

---

### Item 4: Camelyon17 B-v2 SAR Regret and Action Reconciliation
- **Affected Locations:**
  - Table 25 (`tab:compact-actions`), `docs/research/kbound/kbound_submission_supplement.tex`, Line 1163
  - Supplementary prose (§B.5), `docs/research/kbound/kbound_submission_supplement.tex`, Lines 863–866
- **Accepted Authority:** `canonical_panel_results.json:1277-1357, 2220-2230`
- **Pre-Edit Contradiction:**
  - Table 16 (p. 27): reported KGA regret $0.0240$, adapt regret $0.0016$, freeze regret $0.1001$.
  - Appendix B.5 (p. 32): reported stale KGA regret $0.0006$ with favorable two-policy point comparison.
  - Table 25: reported `71 ADAPT, 0 FREEZE, 37 ABSTAIN`.
- **Reconciled Resolution:**
  - Table 25 updated to: `Camelyon17 B--v2 SAR & 108 & 50 & 0 & 58 & 1 & .0093 & 3 within-seed grids; diagnostic \\`
  - Action sum: $50 + 0 + 58 = 108$; commitment: $50 / 108 = 0.46296... \approx 0.4630$; $\mathrm{FA}_u = 1 / 108 = 0.009259... \approx 0.0093$.
  - Appendix B.5 prose updated: "In the B--v2 SAR diagnostic, KGA regret is $0.0240$, compared with $0.0016$ for always-adapt and $0.1001$ for always-freeze (with $50$ \adapt, $0$ \freeze, and $58$ \abstain decisions, committing on $46.3\%$ of cells with one false adaptation, $\mathrm{FA}_u = 0.0093$). Point comparisons favor always-adapt, the archived paired interval against always-adapt includes zero, and the Holm-adjusted comparison does not pass."
- **Status:** RESOLVED & VERIFIED.

---

### Item 5: ImageNet-R Architecture Panel Regret & Actions
- **Affected Locations:**
  - Table 25 (`tab:compact-actions`), `docs/research/kbound/kbound_submission_supplement.tex`, Line 1156
  - Supplementary prose (§B.7), `docs/research/kbound/kbound_submission_supplement.tex`, Line 897
- **Accepted Authority:** `canonical_panel_results.json:46546-46627, 46707-46711`
- **Pre-Edit Contradiction:**
  - Table 16: reported KGA regret $0.0136$, adapt regret $0.0064$, freeze regret $0.0325$.
  - Backbone discussion: gave stale regret $0.0150$.
  - Table 25: reported `165 ADAPT, 29 FREEZE, 286 ABSTAIN`.
- **Reconciled Resolution:**
  - Table 25 updated to: `ImageNet--R & 480 & 176 & 36 & 268 & 3 & .0063 & 4 seeds $\times$ 10 backbones $\times$ 12 cells \\`
  - Action sum: $176 + 36 + 268 = 480$; commitment: $(176 + 36) / 480 = 212 / 480 = 0.44166... \approx 0.4417$; $\mathrm{FA}_u = 3 / 480 = 0.00625 \approx 0.0063$.
  - Supplementary §B.7 prose updated: "ImageNet-R KGA regret is $0.0136$, compared with $0.0064$ for always-adapt (and $0.0325$ for always-freeze), across the ten-backbone architecture panel."
- **Status:** RESOLVED & VERIFIED.

---

## 2. Theoretical Scope & Mathematical Terminology Repairs

### Item 6: Evidence-Fibre Audit Theorem Wording and Class Qualification
- **Source Files:**
  - `docs/research/kbound/kbound_abstract_core.tex`, Lines 9–11
  - `docs/research/kbound/kbound_submission_body.tex`, Lines 475–490
- **Pre-Edit Text:** Asserted unconditionally that every valid audit must cover $\beta$.
- **Revised Text:**
  - Abstract: *"Furthermore, an evidence-fibre audit theorem characterizes the smallest residual bound that any uniformly valid audit based on the same observable law must cover at its declared confidence level; additional processing of unchanged evidence cannot reduce the unresolved residual range."*
  - Section 5.3: Boxed equation and surrounding text explicitly qualified with $0 \le \beta \le \frac{1}{2}$:
    $$\text{least valid fibre-wide residual bound} = \text{population abstention radius} = \beta \quad (0 \le \beta \le \tfrac{1}{2})$$
- **Status:** RESOLVED & VERIFIED.

---

### Item 7: Section 6.1 and Table 4 Paired Deployment Decision Framing
- **Source File:** `docs/research/kbound/kbound_submission_body.tex`, Lines 572–608
- **Pre-Edit Text:** Claimed "The common object is $\Delta$".
- **Revised Text:**
  *"K-Bound and KGA address the same paired deployment decision at different inferential levels. K-Bound characterizes population benefit $\Delta$ over a declared class of target laws $\mathcal C_\beta$. KGA predicts measured evaluation-cell benefit $\Delta^{\mathrm{cell}}$ from historical labeled outcomes and uses a separately calibrated residual interval $[\widehat\Delta \pm \varepsilon]$. Transferring that interval to population benefit requires the additional sampling assumptions and radius in Proposition~\ref{thm:population-transfer}; it is not established by the empirical gate alone. Table~\ref{tab:theory-algorithm-bridge} makes this correspondence explicit."*
- **Table 4 Preserved Separate Definitions:**
  - $\Delta$: Population benefit $2\mu_T(D)(M+\gamma)$
  - $\Delta^{\mathrm{cell}}$: Measured evaluation-cell benefit $S(f_a;\mathcal E)-S(f_0;\mathcal E)$
  - $\widehat\Delta$: Fitted prediction $h_\theta(Z)$ of empirical target $\Delta^{\mathrm{cell}}$
  - $\beta$: Externally declared hidden-residual bound
  - $\varepsilon$: Empirical residual-calibration radius
  - $b(m,\delta)$: Separately justified sampling allowance.
- **Status:** RESOLVED & VERIFIED.

---

### Item 8: Scope of Population Guarantees & Alternative Structural Assumptions
- **Source File:** `docs/research/kbound/kbound_submission_body.tex`, Lines 1026–1031
- **Revised Text:**
  *"The historical proxy failure is consistent with the theorem's warning that processing unchanged evidence cannot replace a justified structural assumption. The theorem does not identify the cause or magnitude of this particular proxy's errors. Rather, Theorem~\ref{thm:compact-beta-minimax} shows why $\beta$ cannot be audited from label-free target data alone within this rich kernel class; population guarantees within this framework without coupled target labels or alternative structural assumptions (such as those discussed in Section~\ref{sec:compact-additional-structure}) require treating $\beta$ as an externally declared sensitivity parameter (by analogy to sensitivity analysis in causal inference), while KGA pursues empirical evaluation-cell interval coverage through historical shift benchmarks."*
- **Status:** RESOLVED & VERIFIED.

---

### Item 9: Non-Oracle Diagnostic Directional Error Rate Semantics
- **Source File:** `docs/research/kbound/kbound_submission_body.tex`, Line 681
- **Revised Text:**
  *"An unhedged point rule ($\operatorname{sign}(\widehat\Delta)$) incurs a $30.9\%$--$35.6\%$ directional error rate ($\operatorname{sign}(g) \neq \operatorname{sign}(\Delta)$) with $0\%$ abstention (where $\mathrm{FA}_u$ is defined strictly for false \adapt; here directional errors combine false \adapt and false \freeze on null/tie states); crucially, $100\%$ of these errors occur on null/tie states ($Z=0, \Delta=0$) due to estimation noise around zero, with zero errors on strictly harmful states ($\Delta = -0.6$)."*
- **Status:** RESOLVED & VERIFIED.

---

## 3. Literature, Historical Interpretation, and Protocol Audits

### Item 10: Appendix F.3 Historical Leave-One-Cell-Out Protocol Diagnostic
- **Source File:** `docs/research/kbound/kbound_submission_supplement.tex`, Lines 1241–1246
- **Pre-Edit Ambiguity:** Described LOO order statistics without clarifying protocol membership relative to primary Protocol-B cross-fitting.
- **Revised Text:**
  *"The following diagnostic evaluates retrospective interval inclusion across the 15 compact CIFAR-10-C source files. Unlike the primary gate's 5-fold outcome-disjoint cross-fitting (Protocol B; 104 calibration cells, 241--242 fitting cells, and 86--87 scoring cells per fold; rank 95 at $\alpha=0.10$), this historical replay uses each candidate's stored residual collection under a leave-one-cell-out order-statistic rule, excluding the scored residual from its own calibration pool. No fitting, tuning, new target access, or training is performed. The input, policy, code, and output digests are recorded in the generated diagnostic JSON. Because the leave-one-cell-out calibration pools overlap across cells, the pooled inclusion fraction is algebraically constrained by the order statistics. It is not an independent estimate of coverage on new environments. Family rows reveal the distribution of those same misses, widths, and decisions; they do not supply group-conditional guarantees or validate the primary Protocol-B gate. A nominal 90\% target is a calibration setting, not a promise validated by this replay."*
- **Status:** RESOLVED & VERIFIED.

---

### Item 11: Literature Correction: Kim et al. TTA/ALine Code Availability
- **Source File:** `docs/research/kbound/kbound_submission_body.tex`, Line 1069
- **Pre-Edit Erroneous Claim:** "Kim et al.'s TTA/ALine lacks published code."
- **Primary Literature Authority:** NeurIPS 2024 paper (https://proceedings.neurips.cc/paper_files/paper/2024/hash/d96fcc07d623a9eba68616629911143a-Abstract-Conference.html), linking GitHub code.
- **Revised Text:**
  *"the authors provide code for Kim et al.'s TTA/ALine~\cite{kim2024ttaline}; our current scored panel, however, contains no authenticated outputs from that implementation (0/15 available), reflecting an incomplete local evaluation rather than an absence of published code."*
- **Status:** RESOLVED & VERIFIED.

---

### Item 12: Removal of Causal Overreach in Historical SAR Discussion
- **Source Files:**
  - `docs/research/kbound/kbound_submission_body.tex`, Lines 897–905
  - `docs/research/kbound/kbound_submission_supplement.tex`, Lines 653–657
- **Pre-Edit Text:** Asserted normalization causes the bulk of SAR gains and gradient updates add no value.
- **Revised Text:**
  - Main text: *"On the recorded CIFAR SAR panel, always-adapt has very low paired-oracle regret. KGA's higher regret reflects the net benefit forgone on cells where it retains the frozen predictor. The current results do not isolate whether those decisions arise mainly from prediction error, conservative calibration, or candidate-update dynamics. Normalization and sample-filtering effects remain hypotheses rather than established causes."*
  - Supplement: *"If its underlying records are recovered and re-authenticated, this would confirm the provenance of the historical BN-only comparison. Even with authenticated provenance, observing that normalization adaptation performs comparably to reference SAR does not by itself establish a causal attribution that normalization accounts for the bulk of SAR's gains; rather, it documents that in this historical noise setting, test-time normalization updates alone yielded comparable observed accuracy. Cross-candidate regret must not be pooled as though candidates share a single oracle; task accuracy on common records provides the valid basis for comparison."*
  - Distinct SAR lines preserved: primary CIFAR SAR (Table 8 / 15), historical 60-cell controls (Table 18 / 19), pre-repair audits (Table 20), corrected 135-cell SAR study (Table 17), and 108-cell population sensitivity replay (Table 6).
- **Status:** RESOLVED & VERIFIED.

---

## 4. Presentation, Formatting & Deliverable Verification

### Item 13: Presentation, Table Layout, and Page Budget
- **Table 17 (p. 28):** Converted environment to `table*`, eliminating column clipping for "False A".
- **Figure Pipeline:** Linearized PDF 1.4 xref tables for `fig_decision_flow.pdf`, `fig_frontier_schematic.pdf`, and `fig_certificate.pdf`.
- **Bibliography Spacing & Compact Driver (`kbound_submission.tex`):** Set `\scriptsize` and tightened internal list spacing in `paper/references_kbound_expanded.tex`.
- **Page Budget Verification:**
  - `kbound_short_final_draft.pdf`: **Exactly 48 pages** (main text ends cleanly on page 19, references on page 20, supplement pages 21–48).
  - `kbound_tmlr.pdf`: **Exactly 55 pages** (single-column anonymous TMLR review format).
  - Undefined references/citations in both variants: **0**.
- **Status:** RESOLVED & VERIFIED.
