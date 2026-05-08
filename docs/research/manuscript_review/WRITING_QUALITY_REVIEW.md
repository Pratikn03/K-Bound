# Manuscript Writing-Quality Review — `PAPER_DRAFT_v1.tex`

**Paper:** *Calibration-Aware Reliability-Adaptive Fusion for Heterogeneous Anomaly Detection*
**Source:** [docs/research/PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) (432 lines, IEEEtran 10pt conference)
**Compiled:** [output/pdf/PAPER_DRAFT_v1.pdf](../../../output/pdf/PAPER_DRAFT_v1.pdf)
**Reviewer pass:** writing, structure, claims-vs-code consistency, IEEE conformance, citation hygiene.
**Date:** 2026-05-07

---

## 0. One-paragraph verdict

This is the **most rigorously written artifact in the repository** and the only document on disk whose claims are matched by the code. It is honestly framed as a method-and-protocol manuscript with results withheld until a real benchmark run, which is a defensible scholarly stance. The writing is clean, the math is correct, the IEEEtran scaffolding is right, and the proposed contributions are real and implemented. The work needed is mechanical: a missing comma here, an unbalanced equation reference there, an under-cited related-work paragraph, and one structural omission (no method-vs-baseline figure). Crucially, the paper draft is **inconsistent with the louder marketing docs in the repo** — `README.md`, `UAISV_Final_Project_Summary.md`, and the per-domain `reports/metrics_*.csv` files — which overclaim what was actually run. Those are the documents that need to be brought into alignment with the paper, not the other way around.

Both [reports/uais_final_report.docx](../../../reports/uais_final_report.docx) and [reports/uais_project_plan.docx](../../../reports/uais_project_plan.docx) are **0-byte files**. They cannot be read because they contain nothing. Anything that references "the final report DOCX" should be considered as referring to a placeholder.

---

## 1. Structural review

### 1.1 Section coverage vs. IEEE conference template

| Section | Present? | Comment |
|---|---|---|
| Title + author block | ✅ | [tex:24-33](../PAPER_DRAFT_v1.tex) — single-author IEEEauthorblock. Affiliation field is `Universal Anomaly Intelligence System (UAIS-V)\\Draft manuscript generated from the local research codebase\\\texttt{docs/research/PAPER\_DRAFT\_v1.tex}`. The path-as-affiliation is fine for a working draft but **must be replaced with a real institution** before submission. |
| Abstract | ✅ | [tex:39-56](../PAPER_DRAFT_v1.tex) — 16 lines, dense, hits motivation/method/contribution/status. Good. |
| Keywords | ✅ | [tex:58-61](../PAPER_DRAFT_v1.tex) — 8 keywords, well-chosen. |
| Introduction | ✅ | [tex:63-113](../PAPER_DRAFT_v1.tex) — well-paced, contributions enumerated. |
| Related Work | ✅ | [tex:115-151](../PAPER_DRAFT_v1.tex) — four sub-sections (anomaly, fusion, calibration, explainability). **Thin in places** — see §3 below. |
| Problem Formulation | ✅ | [tex:153-169](../PAPER_DRAFT_v1.tex) — clean notation. |
| Methodology | ✅ | [tex:171-229](../PAPER_DRAFT_v1.tex) — four subsections, two equations, two attribution definitions. |
| Implementation | ✅ | [tex:231-259](../PAPER_DRAFT_v1.tex) — single component table; **lacks pseudocode or training loop**, see §4. |
| Experimental Protocol | ✅ | [tex:261-328](../PAPER_DRAFT_v1.tex) — datasets, baselines, metrics, explicit "results withheld" disclosure, blank result table. |
| Expected Analysis | ✅ | [tex:330-351](../PAPER_DRAFT_v1.tex) — non-standard but appropriate for a draft. Most IEEE papers omit this; for a method-protocol draft, keeping it is a reasonable choice. |
| Limitations | ✅ | [tex:353-363](../PAPER_DRAFT_v1.tex) — three honest limitations. |
| Reproducibility Plan | ✅ | [tex:365-380](../PAPER_DRAFT_v1.tex) — concrete commands. |
| Conclusion | ✅ | [tex:382-390](../PAPER_DRAFT_v1.tex) — appropriate length. |
| References | ✅ | [tex:392-430](../PAPER_DRAFT_v1.tex) — 8 entries, hand-rolled `thebibliography`. **2 cited entries are unused** — see §5. |
| Acknowledgments | ❌ | Missing. Optional in IEEE conference but conventional. |
| **Figure(s)** | ❌ | **No figures at all.** This is the single biggest omission. See §4. |
| Appendix | ❌ | Optional; absence is fine. |

**Verdict on structure:** complete enough for a method-and-protocol draft; the missing piece is *visual material* (architecture diagram, reliability flow, attention-weight illustration).

### 1.2 IEEEtran conformance

| Item | Status |
|---|---|
| `\documentclass[10pt,conference]{IEEEtran}` | ✅ correct combo for IEEE conference. |
| `\IEEEauthorblockN` / `\IEEEauthorblockA` | ✅ used correctly. |
| `\bstctlcite` for IEEEtran biblstyle | ❌ not used; OK because hand-rolled `thebibliography`. |
| `\IEEEpeerreviewmaketitle` | n/a (only needed in non-conference mode). |
| `\bibliographystyle{IEEEtran}` + `\bibliography{...}` | ⚠️ not used; manuscript uses `\begin{thebibliography}` instead. Acceptable, but `.bib` + `IEEEtran.bst` is the more maintainable path once references grow. |
| Page-limit considerations | ⚠️ Cannot verify without compiling; with no figures the paper is likely under-length for IEEE conference (typically 6–8 pages). Adding the architecture figure and a reliability-block figure should bring it into range. |

---

## 2. Writing quality (paragraph by paragraph)

### 2.1 Abstract ([tex:39-56](../PAPER_DRAFT_v1.tex))

Strengths:
- Opens with the *problem* (heterogeneous signals + practical operating constraints), not the *method* — this is the correct order.
- Defines the acronym CRAF the first time it appears.
- Names the three reliability signals (calibration error, drift, sharpness) explicitly.
- Discloses draft status in the last sentence — rare and admirable.

Weaknesses:
- "Modern anomaly detection systems increasingly observe…" — *increasingly* is filler. Cut it.
- "A practical fusion layer must therefore combine domain experts while handling missing modalities, distribution shift, label imbalance, and the need for human-readable explanations." — four-item list is fine but the last item is a different shape from the first three (handling X, handling Y, handling Z, and *the need for* W). Make it parallel: "…distribution shift, label imbalance, and explanation requirements."
- "These weights are injected at inference time so the fusion model can down-weight unreliable or shifted domains without retraining." — strong sentence, keep it.
- The abstract claims "post-hoc reliability estimator" — the code matches: [reliability_estimator.py:29-107](../../../src/uais/fusion/attention/reliability_estimator.py) — fitted *after* training on validation, isotonic calibrators, KS reference distributions. ✅ truthful.

### 2.2 Introduction ([tex:63-113](../PAPER_DRAFT_v1.tex))

Strengths:
- The opening paragraph builds a concrete operational scenario before introducing the method. This is the right rhetorical move for an applied paper.
- The "system-level question" framing in [tex:71-73](../PAPER_DRAFT_v1.tex) is sharp.
- The contribution list at [tex:103-113](../PAPER_DRAFT_v1.tex) is enumerated, not buried.

Weaknesses:
- Paragraph 2 ([tex:75-82](../PAPER_DRAFT_v1.tex)) begins "Simple fusion methods are attractive…" and gestures at "averaging, weighted averaging, stacking, and random forest meta-classifiers" *without a single citation*. These are exactly the baselines cited later in §Related Work and implemented in [baselines.py](../../../src/uais/fusion/attention/baselines.py). Add at least Baltrusaitis 2019 and Zhou 2012 citations here.
- "This assumption is fragile" ([tex:79](../PAPER_DRAFT_v1.tex)) — assertion without evidence. Either drop the claim or anchor it to one cited drift study.
- Paragraph 3 ([tex:84-100](../PAPER_DRAFT_v1.tex)) introduces ReliabilityEstimator before formally defining it. That's fine for an introduction, but the words "test-time reliability adaptation" should be the **bolded contribution** because it is genuinely the novel piece. Currently the contribution sounds equal to "missing-domain masking" which is a known idea.
- "calibration quality on validation data, distribution shift measured by a Kolmogorov-Smirnov test against validation score distributions, and prediction sharpness" — long compound sentence. Consider a colon+list.

### 2.3 Related Work ([tex:115-151](../PAPER_DRAFT_v1.tex))

Strengths:
- Four-subsection structure (anomaly, fusion, calibration, explainability) maps cleanly to the contributions.
- Honest about what is/isn't in scope.

Weaknesses (these will be flagged in peer review):
- **Calibration & Reliability** subsection ([tex:136-143](../PAPER_DRAFT_v1.tex)) cites *nothing*, despite using technical terms (ECE, Brier) that need anchors. Add Guo 2017 (already in the bibliography but unused — see §5) and Zadrozny 2002 (also already in bibliography, also unused).
- The fusion subsection ([tex:126-134](../PAPER_DRAFT_v1.tex)) cites Vaswani 2017 and Baltrusaitis 2019 but **does not cite any test-time-adaptation prior work**. CRAF's "novelty" only stands if test-time adaptive fusion is shown to be non-trivial. Add at least one of: Sun et al. (Test-Time Training), Wang et al. (Tent), or any prior adaptive-fusion paper.
- The explainability subsection only cites Lundberg 2017 (SHAP). Counterfactual explanations have a published lineage (Wachter et al. 2017 on counterfactual explanations, or Mothilal/Sharma/Tan on DiCE). Cite at least one to motivate the choice of the counterfactual style over SHAP.
- No paragraph on **anomaly fusion specifically** — there is real prior work (Lazarevic & Kumar 2005 on combining outlier detectors, Aggarwal & Sathe on ensemble anomaly detection). Even one citation establishes scholarly context.

### 2.4 Problem Formulation ([tex:153-169](../PAPER_DRAFT_v1.tex))

Strengths:
- Correct equation environment, single numbered display.
- Explicit definition of the missing-domain mask convention.

Weaknesses:
- The mask convention is unusual: "$m_{i,d}=1$ means the domain is unavailable" ([tex:158-159](../PAPER_DRAFT_v1.tex)). Most multimodal papers use the opposite convention (mask=1 for *present*). Either is fine, but flag it explicitly: *"We use the masked-out convention…"*. Otherwise reviewers will get confused mid-method.
- The notation $r_i \in [0,1]^D$ ([tex:167](../PAPER_DRAFT_v1.tex)) is correct but the body of [reliability_estimator.py:113-166](../../../src/uais/fusion/attention/reliability_estimator.py) currently broadcasts a *scalar* per domain across the whole batch (`weights[available_mask, i] = rel_d` at line 164). The paper's per-sample $r_{i,d}$ formulation is therefore a *generalization* of what the code does. Either:
  - Update the math to say $r_d$ (batch-level scalar), and add a footnote that per-sample reliability is future work, or
  - Update the code to compute per-sample reliability, and verify that the math matches.
  - **This is a real method-vs-implementation gap. Resolve it before submission.**

### 2.5 Methodology ([tex:171-229](../PAPER_DRAFT_v1.tex))

Strengths:
- Three equations, all correct.
- Equation 4 (counterfactual attribution) ([tex:222-224](../PAPER_DRAFT_v1.tex)) is the simplest possible definition and that is a virtue, not a flaw.
- The paragraph after Eq. (3) ([tex:191-208](../PAPER_DRAFT_v1.tex)) explicitly states the constraint $\alpha + \beta + \gamma = 1$, which the code enforces at [reliability_estimator.py:47-48](../../../src/uais/fusion/attention/reliability_estimator.py). ✅ consistent.

Weaknesses:
- Eq. (2) ([tex:175-177](../PAPER_DRAFT_v1.tex)) is `h_{i,d} = g_d(x_{i,d}) + e_d` — correct, but the corresponding code adds **positional** embeddings as well ([cross_modal_attention.py:90-92](../../../src/uais/fusion/attention/cross_modal_attention.py)). Either drop positional embeddings from the code (they're conceptually weak for a permutation-invariant set of domains) or document them in the equation: `h_{i,d} = g_d(x_{i,d}) + e_d + p_d`.
- Eq. (3) is missing a closing period after `\hat{p}_i = \sigma(z_i).` — actually present. ✅
- Eq. (4): no equation number is referenced anywhere in the prose (`(\ref{eq:reliability})` is used but `eq:reliability` only labels Eq. 3, not Eq. 4). If you intend Eq. 4 to be referenced, add `\label{eq:cda}`.
- The "two practical advantages" pitch at [tex:212-217](../PAPER_DRAFT_v1.tex) is good but the second advantage ("reliability is computed from observable properties…") repeats the abstract verbatim. Vary the phrasing.

### 2.6 Implementation ([tex:231-259](../PAPER_DRAFT_v1.tex))

Strengths:
- The component table at [tex:236-253](../PAPER_DRAFT_v1.tex) cleanly maps method to code.
- Honest about what tests cover.

Weaknesses:
- **No pseudocode**, **no architecture figure**, **no end-to-end algorithm box**. For an IEEE submission this is the section that most needs visualization. Recommended additions:
  1. **Algorithm 1**: training loop pseudocode.
  2. **Algorithm 2**: inference-time CRAF weight injection.
  3. **Figure 1**: the architecture diagram already drawn in Mermaid in [docs/research/ATTENTION_FUSION_ARCHITECTURE.md](../ATTENTION_FUSION_ARCHITECTURE.md). Re-render in TikZ for the paper.
- The table ([tex:241-253](../PAPER_DRAFT_v1.tex)) uses a bare `tabular` instead of `IEEEeqnarraybox` or `tabularx`. With `L{0.34\linewidth}L{0.56\linewidth}` totaling 0.90, you're leaving 10% of the line unused. Either widen to 0.40+0.55 or center the table.

### 2.7 Experimental Protocol ([tex:261-328](../PAPER_DRAFT_v1.tex))

Strengths:
- Section 5.4 ([tex:303-309](../PAPER_DRAFT_v1.tex)) — *"This draft does not report benchmark performance numbers because the full `run_breakthrough_experiment.py` pipeline has not yet been executed in this manuscript workflow."* — this is the right thing to write. Reviewers respect honest withholding of results far more than they punish it.
- The blank result-table skeleton at [tex:311-328](../PAPER_DRAFT_v1.tex) signals exactly what a final version will report.
- Metric list at [tex:296-301](../PAPER_DRAFT_v1.tex) is appropriate for imbalanced anomaly detection: ROC-AUC, PR-AUC, F1, balanced accuracy, Brier, ECE, TPR@FPR — covers ranking, threshold, and calibration.

Weaknesses:
- The Datasets table at [tex:268-283](../PAPER_DRAFT_v1.tex) lists "Behavior — CERT or web-session behavior" and "NLP — Enron or weakly labeled news/email." The "or" framing is honest but **vague**. The actual repository contains [`online_shoppers_intention.csv`](../../../data/raw/behavior/online_shoppers_intention.csv) and [`fake_news_labeled.csv`](../../../data/raw/nlp/fakenews/fake_news_labeled.csv); CERT and Enron are not actually used end-to-end (Enron file exists but the active NLP pipeline is on fake-news, not Enron). Either commit to specific datasets in the paper or reorder the table to put the *as-implemented* datasets first.
- The Baselines list at [tex:285-294](../PAPER_DRAFT_v1.tex) names six baselines; the code in [baselines.py](../../../src/uais/fusion/attention/baselines.py) implements four (`EarlyFusionMLP`, `LateFusionEnsemble`, `RandomForestFusion`, `ConfidenceWeightedMean`) plus "static cross-modal attention" (the same model with `enable_craf=False`). "Simple mean score fusion" is not implemented as a separate class. Either implement the missing baseline (it's ~10 lines), or remove it from the protocol list.
- 70/15/15 split convention is described in [EXPERIMENTAL_PROTOCOL.md](../EXPERIMENTAL_PROTOCOL.md) but **not stated in the paper**. Add: "Splits are 70/15/15 train/val/test with 3 seeds (42, 43, 44)."
- "Statistical support is planned through bootstrap intervals for PR-AUC/F1 and DeLong-style ROC-AUC comparison" — both helpers exist at [src/uais/utils/stats.py](../../../src/uais/utils/stats.py) (`bootstrap_ci`, `delong_roc_test`, `paired_ttest`) and are imported by [run_breakthrough_experiment.py:59](../../../src/scripts/run_breakthrough_experiment.py). ✅ truthful.

### 2.8 Limitations ([tex:353-363](../PAPER_DRAFT_v1.tex))

Excellent paragraph. Three limitations, all real:
1. Results not yet inserted.
2. Multimodal alignment via shared real entities is the hardest data issue.
3. KS testing breaks at small batch sizes; fallback exists.

The second limitation is **especially honest** and matches a real defect in the surrounding non-research pipelines: see [run_fusion_experiment.py:162](../../../src/scripts/run_fusion_experiment.py) where rows from fraud, cyber, and behavior are aligned by index across unrelated entities. The paper's framing of this as a limitation (rather than a solved problem) is correct.

### 2.9 Conclusion ([tex:382-390](../PAPER_DRAFT_v1.tex))

Short, on-message. One nit: "The current codebase contains the mechanisms required for a serious empirical study" — *serious empirical study* is non-academic register. Replace with "the components required for full empirical evaluation."

---

## 3. Citation hygiene

8 bibliography entries. Cited usage:

| Key | Cited in body? | Used at? |
|---|---|---|
| `vaswani2017attention` | ✅ | [tex:133](../PAPER_DRAFT_v1.tex) |
| `baltrusaitis2019multimodal` | ✅ | [tex:128](../PAPER_DRAFT_v1.tex) |
| `pang2021deep` | ✅ | [tex:118](../PAPER_DRAFT_v1.tex) |
| `lundberg2017unified` | ✅ | [tex:147](../PAPER_DRAFT_v1.tex) |
| `zadrozny2002transforming` | ❌ | **never cited** |
| `guo2017calibration` | ❌ | **never cited** |
| `zhou2012ensemble` | ❌ | **never cited** |
| `friedman2001greedy` | ❌ | **never cited** |

Four uncited references will trip a reviewer running `\nocite` checks or a basic `latexmk` warning sweep. Two paths:
1. Cite them where they belong (Guo 2017 → calibration paragraph; Zadrozny 2002 → calibration paragraph; Zhou 2012 → late-fusion baselines mention; Friedman 2001 → boosting baselines if those land in the empirical section).
2. Or remove them from the bibliography until the empirical section actually invokes them.

Recommended: cite Guo 2017 and Zadrozny 2002 at [tex:138-141](../PAPER_DRAFT_v1.tex) (the unanchored calibration sentence). Defer Zhou 2012 and Friedman 2001 to a later draft when baselines are reported.

Missing references that a reviewer is likely to ask for:
- A test-time adaptation foundation paper (Tent or TTT).
- A counterfactual-explanation foundation paper (Wachter et al. 2017 or Mothilal et al. 2020).
- An anomaly-ensemble foundation paper (Aggarwal & Sathe 2017, *Outlier Ensembles*).
- A drift-detection paper to anchor the KS approach (Gama et al. 2014 on concept drift, or Lipton et al. 2018 on detecting/correcting label shift).

---

## 4. Method-vs-implementation gaps (engineering review)

These are the gaps that need to close before submission. Each is a real diff between what the paper claims and what the code does.

| # | Paper says | Code does | Resolution |
|---|---|---|---|
| 1 | "$r_i \in [0,1]^D$" — per-sample reliability ([tex:167](../PAPER_DRAFT_v1.tex)) | Batch-level scalar broadcast: `weights[available_mask, i] = rel_d` ([reliability_estimator.py:164](../../../src/uais/fusion/attention/reliability_estimator.py)) | Either change paper to $r_d$ + footnote, or update code |
| 2 | $h_{i,d} = g_d(x_{i,d}) + e_d$ ([tex:175-177](../PAPER_DRAFT_v1.tex)) | Adds positional embedding too ([cross_modal_attention.py:90-92](../../../src/uais/fusion/attention/cross_modal_attention.py)) | Add positional term to equation or remove from code |
| 3 | "static cross-modal attention without reliability adaptation" listed as a baseline ([tex:293](../PAPER_DRAFT_v1.tex)) | Implemented as the same model with `enable_craf: false` in [attention_config.yaml:66](../../../src/uais/fusion/attention/attention_config.yaml) | Document this in the paper |
| 4 | "simple mean score fusion" listed as baseline ([tex:288](../PAPER_DRAFT_v1.tex)) | Not in [baselines.py](../../../src/uais/fusion/attention/baselines.py) | Add ~10-line `MeanFusion` class or drop from list |
| 5 | "DeLong-style ROC-AUC comparison" ([tex:301](../PAPER_DRAFT_v1.tex)) | `delong_roc_test` helper in [src/uais/utils/stats.py](../../../src/uais/utils/stats.py) used at [run_breakthrough_experiment.py:59](../../../src/scripts/run_breakthrough_experiment.py) | ✅ matches |
| 6 | Datasets: CERT or web-session, Enron or weakly labeled news ([tex:278-280](../PAPER_DRAFT_v1.tex)) | `online_shoppers_intention.csv` (web-session) and `fake_news_labeled.csv` (weak news) | ✅ matches the *vague* version, but commit to specifics |
| 7 | KS-test fallback when sample count is small ([tex:360-363](../PAPER_DRAFT_v1.tex)) | Fallback at [reliability_estimator.py:147](../../../src/uais/fusion/attention/reliability_estimator.py) (`ks_reliability = 1.0` if too few samples) | ✅ matches |
| 8 | "missing-domain masking" ([tex:106-108](../PAPER_DRAFT_v1.tex)) | Implemented via `key_padding_mask` at [cross_modal_attention.py:55-57](../../../src/uais/fusion/attention/cross_modal_attention.py) | ✅ matches |

---

## 5. Mechanical / typographic issues

Sweep before next compile:

- [tex:80](../PAPER_DRAFT_v1.tex): "miscalibrated" — fine but inconsistent with "well-calibrated" elsewhere; pick a hyphenation policy.
- [tex:185-187](../PAPER_DRAFT_v1.tex): the trailing `,` after `\quad` in `f_{\theta}(\mathrm{Pool}(\mathrm{Attention}(H_i, m_i))), \quad \hat{p}_i = \sigma(z_i).` is acceptable but the sentence structure inside one numbered display is awkward. Consider splitting into two equations.
- [tex:198-201](../PAPER_DRAFT_v1.tex): equation has no period at the end of `\gamma S_d,`. IEEEtran style is to terminate a display equation that ends a sentence with `.` or `,` matching the prose. Currently `,` is correct because the sentence continues. ✅
- [tex:228](../PAPER_DRAFT_v1.tex): "if the fraud domain were absent, the risk would change by $\Delta$." — the apostrophe-style quotes here are straight quotes (`\'\'`); IEEEtran prefers TeX-style ``…''. Fix: ``...''.
- [tex:252](../PAPER_DRAFT_v1.tex): "Clean-data, shift, adversarial, missing-domain, calibration, and explanation phases." — this is a sentence fragment in a table cell. Fine for a table but Title Case would be more conventional ("Clean Data, Shift, …").
- [tex:283](../PAPER_DRAFT_v1.tex): two-table-in-a-row with no body text between them. Add at least one sentence between Table II and the Baselines list.
- [tex:309](../PAPER_DRAFT_v1.tex): `Table~\ref{tab:resultsplan}` — label is `tab:resultsplan` (one word). For consistency with `tab:datasets` use `tab:results_plan` or `tab:resultsplan` everywhere. Currently consistent. ✅
- [tex:373-376](../PAPER_DRAFT_v1.tex): `\path{tests/test_baselines.py}` — `\path{}` is from the `url` package which is loaded transitively via `hyperref`. Works, but consistent use: some lines use `\texttt{}` and some `\path{}`. Pick one.
- The blank result table at [tex:319-326](../PAPER_DRAFT_v1.tex) uses `--` for empty cells. IEEE convention prefers `n/a` or `\textendash`. Cosmetic only.
- No `\IEEEcompsocitemizethanks` or `\thanks{}` block — fine for a draft.
- No `\IEEEspecialpapernotice{(Draft)}` — adding it would visually mark the PDF as a draft. Optional but useful.

---

## 6. Comparison with the rest of the repository's writing

| Document | Honesty about results | Technical accuracy | Style |
|---|---|---|---|
| [docs/research/PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) | ✅ explicitly withholds results | ✅ math correct, code matches (with the gaps in §4) | IEEE conference, formal |
| [docs/research/PAPER_DRAFT_v1.md](../PAPER_DRAFT_v1.md) | ✅ stub, says "results intentionally not fabricated" | n/a | Minimal pointer file |
| [docs/research/EXPERIMENTAL_PROTOCOL.md](../EXPERIMENTAL_PROTOCOL.md) | ✅ specifies 3 seeds, mean±std, bootstrap, DeLong | ✅ | Bullet-point |
| [docs/research/ATTENTION_FUSION_ARCHITECTURE.md](../ATTENTION_FUSION_ARCHITECTURE.md) | ✅ neutral | ✅ Mermaid diagram matches code | Bullet-point |
| [docs/research/data/DATASET_INVENTORY.md](../data/DATASET_INVENTORY.md) | ⚠️ "License: TBD" everywhere | ⚠️ lists CERT + Enron + MVTec which are not the datasets actually loaded | Tabular |
| [docs/REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) | ✅ | ✅ paths match real code | Short |
| [README.md](../../../README.md) | ❌ **overclaims** ("LSTM autoencoder for behavior", "DistilBERT NLP", "ResNet/ViT vision") | ❌ none of these match the as-loaded datasets | Marketing |
| [UAISV_Final_Project_Summary.md](../../../UAISV_Final_Project_Summary.md) | ❌ **strongly overclaims** (e.g., "Expected Academic Grade 95-100/100") | ❌ claims CERT r4.2 + Enron + ResNet/ViT — none used in scripted runs | Project pitch |
| [reports/uais_final_report.docx](../../../reports/uais_final_report.docx) | n/a — **0 bytes** | n/a | empty file |
| [reports/uais_project_plan.docx](../../../reports/uais_project_plan.docx) | n/a — **0 bytes** | n/a | empty file |
| [reports/metrics_*.csv](../../../reports/) | ❌ headline numbers are round and `std` column is empty; no real CV produced | ❌ values not reproducible from code | placeholder |
| [IMPROVEMENTS.md](../../../IMPROVEMENTS.md) | ⚠️ describes API auth + monitoring as "after improvements"; matches code | ✅ matches code | Project pitch |

**Reading this matrix:** the paper draft is the *highest-quality* writing in the repository. Every other claim about ML results is either softer (`docs/research/*`) or louder (`README.md`, `UAISV_Final_Project_Summary.md`). The right move before submission is to **demote** the marketing docs, not to inflate the paper.

---

## 7. Recommended edits (priority order)

### Tier 1 — required before submission
1. Replace placeholder affiliation at [tex:29-32](../PAPER_DRAFT_v1.tex) with a real institution.
2. Resolve the per-sample vs batch-level reliability inconsistency (§2.4 and §4 row 1).
3. Cite or remove the four uncited bibliography entries (§3).
4. Add at least one figure (architecture diagram, TikZ port of [ATTENTION_FUSION_ARCHITECTURE.md](../ATTENTION_FUSION_ARCHITECTURE.md)).
5. Either implement `MeanFusion` baseline or remove "simple mean score fusion" from §V.B.
6. Commit to specific datasets in Table II (drop "or").
7. State the 70/15/15 split + seed list explicitly in §V.A.

### Tier 2 — strongly recommended
8. Add Algorithm 1 (training) and Algorithm 2 (CRAF inference injection) using `algorithm2e` or `algorithmicx`.
9. Add citations to test-time adaptation prior work (§3).
10. Add citations to counterfactual-explanation prior work (§3).
11. Run `latexmk` and fix any `Reference X undefined` or `Citation X undefined` warnings.
12. Update the equation for $h_{i,d}$ to include positional embeddings, OR remove positional embeddings from the code.

### Tier 3 — polish
13. Sweep typography (TeX quotes, `\path` vs `\texttt` consistency, em-dash usage).
14. Tighten abstract first sentence (drop "increasingly").
15. Make limitations 2 (alignment) and the run-fusion-experiment alignment defect a single explicit cross-reference.
16. Replace empty `--` in result table with `\textendash` or `n/a`.

### Tier 4 — repo-level cleanup (not paper edits, but blocks honest narrative)
17. Replace [reports/uais_final_report.docx](../../../reports/uais_final_report.docx) with a real PDF/DOCX or delete it.
18. Replace [reports/uais_project_plan.docx](../../../reports/uais_project_plan.docx) likewise.
19. Reconcile [README.md](../../../README.md) and [UAISV_Final_Project_Summary.md](../../../UAISV_Final_Project_Summary.md) with the dataset/model claims that the paper actually substantiates. The paper draft already provides the honest replacement framing — port it.
20. Regenerate [reports/metrics_*.csv](../../../reports/) from a real multi-seed run via [src/uais/reporting/make_tables.py](../../../src/uais/reporting/make_tables.py) before any external mention of those numbers.

---

## 8. Bottom line

The CRAF manuscript is a defensible piece of methods writing. The math is right, the code matches the prose (with the small gaps in §4), the related work is thin but honest, and the protocol section discloses exactly what has and has not been measured. Closing the Tier 1 list and adding a single architecture figure brings this draft within striking distance of a workshop or short-paper venue submission. The longer publication path — e.g., an IEEE TKDE or KDD-ADS submission — requires the actual benchmark run to populate Table III.

The asymmetry to remember: **the paper is too modest about the work, while the README is too proud of it.** Aligning them means trimming the README, not embellishing the paper.
