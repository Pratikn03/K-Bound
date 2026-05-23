# UAV / UAV-RAF — Overall Review, Novelty Audit, Gaps, and 12-Week Plan

**Scope:** synthesis across [WRITING_QUALITY_REVIEW.md](WRITING_QUALITY_REVIEW.md) (v1), [WRITING_QUALITY_REVIEW_v2.md](WRITING_QUALITY_REVIEW_v2.md) (v2), [RESEARCH_QUALITY_AUDIT.md](../../../RESEARCH_QUALITY_AUDIT.md), the rendered PDF, and the on-disk codebase.
**Date:** 2026-05-07
**Audience:** the author, deciding what to do next.

**Execution ladder:** see [EVIDENCE_LADDER.md](EVIDENCE_LADDER.md) for the
workshop / mid-tier / top-venue evidence map and current implementation status.

---

## 0. One-screen verdict

The **engineering** here is solid: clean code, real benchmark generator, honest metadata, runnable end-to-end, real test suite. The **paper** is well-organized and unusually honest about its own limitations.

The **research** is currently sitting at *competent workshop draft*, not yet *PhD/full-venue submission*. The reason is not writing or rigor — those are improving. The reason is that **the proposed mechanism (UAV-RAF) and the benchmark (label-aligned UAV-RealFusion) do not yet isolate a research result**:

- The benchmark is too easy → all methods saturate → no method comparison is meaningful.
- The reliability gate rarely fires → the proposed mechanism is dormant in the headline experiments.
- The novelty is a *combination* of standard components, not a new mechanism with theoretical or empirical separation from prior work.

What the paper needs is not more pages or more polish — it needs **two more weeks of targeted experiments** that change the headline narrative from "we tied static attention" to "we measurably help in conditions where static attention fails." The plan in §6 lays out exactly those experiments.

---

## 1. Novelty audit — what is actually new?

### 1.1 The four claimed contributions, judged honestly

The paper enumerates four contributions in §I.D. Each is graded against the standard "is this novel research, novel engineering, or known practice?"

| Claimed contribution | Component status | Novelty class |
|---|---|---|
| 1. Reframing as a system, not a single fusion module | Documentation choice | **Not research novelty** — a presentation decision |
| 2. UAV-RAF: masked attention + ECE/KS/sharpness reliability + reliability gate | Composition of standard parts | **Modest engineering novelty** — see §1.2 |
| 3. UAV-RealFusion: real-domain label-aligned benchmark | Useful artifact, but explicitly label-aligned | **Reproducibility artifact, not benchmark novelty** |
| 4. Complete evaluation: baselines, missing-domain, drift, adversarial, calibration, CDA | Standard components, well-assembled | **Engineering competence, not research novelty** |

The claim "four contributions" is doing a lot of work. By research standards, the paper has **at most 1.5 contributions** (the UAV-RAF mechanism and a partial-credit benchmark artifact). That's fine for a workshop or short paper, but it must be framed honestly to survive review at venues like KDD, NeurIPS, ICLR, CIKM, or IEEE TKDE.

### 1.2 The actual technical novelty — line-by-line

Each piece of UAV-RAF, against the prior literature:

| Mechanism | UAV-RAF version | Closest prior work | Novelty? |
|---|---|---|---|
| Attention over modalities | Multi-head self-attention with key-padding mask, learned domain embeddings | Vaswani 2017; Baltrusaitis 2019 surveys | None — standard |
| Isotonic calibration on validation | Per-domain isotonic regression | Zadrozny & Elkan 2002; Platt 1999 | None — textbook |
| ECE as reliability signal | Validation ECE, $1-\mathrm{ECE}$ contributes to weight | Guo et al. 2017 | None — standard reliability-of-calibration interpretation |
| KS test for distribution drift | Two-sample KS p-value vs. validation reference | Lipton et al. 2018, label shift; Gama et al. 2014, drift surveys | None — standard drift detector applied at score level |
| Sharpness signal | $(\hat p - 0.5)^2$ averaged | Brier 1950; Gneiting & Raftery 2007 on sharpness | None — standard scoring-rule diagnostic |
| **Combining ECE + KS + sharpness as a weighted scalar** | $r_d = \alpha(1-\mathrm{ECE}_d) + \beta p_{\mathrm{KS},d} + \gamma S_d$ | No exact prior — but each term is standard | **Modest novelty** — the *specific composition* and its application to fusion weighting |
| Reliability gate (use static unless $\bar r < \tau$) | Hard threshold $\tau=0.66$ | Selective prediction (Geifman & El-Yaniv 2017); selective QA (Kamath et al. 2020); early-exit networks | **Modest novelty** — applied as a fusion-mode switch rather than abstain |
| Counterfactual domain attribution | Mask one domain, recompute risk, report $\Delta$ | Leave-one-out attribution (Lipovetsky 2001); permutation importance (Breiman 2001); Shapley (Lundberg 2017); Wachter 2017 counterfactuals; DiCE (Mothilal 2020) | None — standard masking attribution rebadged for fusion |
| Adversarial perturbation engine (zero / max / Gaussian) | Score-level coherent attacks | Standard score-perturbation; Hendrycks & Dietterich 2019 robustness benchmarks | None — standard |

**Net novelty:** UAV-RAF is fundamentally a thoughtful **composition** of well-known tools, plus one **modest design choice** (the reliability gate). That can be a publishable result — but it requires either (a) showing the composition has a measurable advantage on a non-trivial benchmark, or (b) a theoretical result on when the composition is provably better than static attention.

The paper currently has neither. Both are reachable in the plan in §6.

### 1.3 What's *not* in the literature and could become a real contribution

If the author is willing to invest, three directions in increasing ambition:

1. **Empirical-novelty path (lowest effort):** show that UAV-RAF beats Tent and TTT (two real test-time-adaptation baselines from 2020–21) on a benchmark where fusion matters. This converts "modest composition" into "first calibration-aware test-time-adapted fusion." Workshop-grade.

2. **Mechanism-novelty path:** replace the heuristic gate $\tau=0.66$ with a *learned* gate trained on validation drift episodes — a small classifier $g(\bar r) \to \{static, adaptive\}$. Now the reliability mechanism is data-driven rather than threshold-tuned. Conference-grade if paired with #1.

3. **Theory-novelty path (highest effort):** replace the invalid bounded-shift sketch with a theorem stack that proves the mechanism boundaries: quality-blind fusion impossibility, global-KS mixture confounding, mean-gate dilution, risk-dominance switching, finite-sample certification, and KS false-fire control. PhD/journal-grade only if those assumptions are validated by matching experiments.

---

## 2. Research quality — beyond the paper itself

### 2.1 What's strong

- **Code matches prose** (with the theory appendix now requiring theorem-linked experiments): the modules referenced in [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) actually exist and are tested. Compared to most academic codebases, this is unusual.
- **Honest construct flagging**: [experiments/fusion/real_domain_fusion_metadata.json](../../../experiments/fusion/real_domain_fusion_metadata.json) literally contains `"important_limitation": "...not naturally co-observed entities."`. Self-flagging like this is rare.
- **Reproducibility ladder is real**: [src/scripts/prepare_real_fusion_benchmark.py](../../../src/scripts/prepare_real_fusion_benchmark.py) → [configs/attention_real_fusion.yaml](../../../configs/attention_real_fusion.yaml) → [run_breakthrough_experiment.py](../../../src/scripts/run_breakthrough_experiment.py) → [generate_craf_paper_assets.py](../../../src/scripts/generate_craf_paper_assets.py). One command path, deterministic seeds, real outputs.
- **Multi-seed clean evaluation** (5 seeds, mean ± std). Better than the per-domain pipelines audited in [RESEARCH_QUALITY_AUDIT.md](../../../RESEARCH_QUALITY_AUDIT.md), which all run a single seed.
- **Threats-to-validity discussion** (§XII) — internal/construct/external/statistical. The kind of self-criticism that disarms reviewers.

### 2.2 What's weak

- **Saturated benchmark**: 5 of 6 methods at 0.9997 ROC-AUC. Not a method comparison; it's the noise floor.
- **Dormant mechanism**: across the missing-domain ablation (the most-cited robustness experiment), UAV-RAF is mechanistically identical to static attention. This is documented honestly but undermines the contribution.
- **Underpowered statistics**: $n=5$ paired t-test, $p=0.374$. Even if a real effect existed, this test cannot detect it.
- **Single-seed robustness**: drift, adversarial, and CDA experiments are diagnostic on one seed. No CIs.
- **Missing comparators**: no Tent, no TTT, no learned gate baseline, no Bayesian-uncertainty fusion (e.g., MC Dropout averaged across domains). The right test-time-adaptation comparators are absent.
- **Adversarial story is narrow**: gain only on coherent all-domain attacks (a strong threat model that real attackers rarely achieve). Per-domain attacks show no effect.
- **Empty / placeholder docs**: [reports/uais_final_report.docx](../../../reports/uais_final_report.docx) and [reports/uais_project_plan.docx](../../../reports/uais_project_plan.docx) are 0 bytes. The non-research docs ([README.md](../../../README.md), [UAISV_Final_Project_Summary.md](../../../UAISV_Final_Project_Summary.md)) overclaim what was actually run, contradicting the paper's honesty.

### 2.3 Research-quality grade vs. venue

| Venue | Current state | What's missing |
|---|---|---|
| Local workshop / arXiv / thesis chapter | ✅ ready after Tier-0 fixes (rename, affiliation, demote saturated table) | Polish only |
| ML/security workshop (e.g., NeurIPS-W, ICML-W, IEEE S&P workshops) | ⚠️ ready after §6 phase-1 (1–2 weeks) | Better benchmark, gate ablation |
| Mid-tier venue (e.g., CIKM, IEEE BigData, ICDM, EUSIPCO) | ⚠️ ready after §6 phase-2 (4–6 weeks) | Above + Tent/TTT comparators + multi-seed robustness |
| Top-tier venue (KDD, NeurIPS, ICLR, IEEE TKDE) | ❌ not yet | Above + naturally paired multimodal data + theoretical or substantially novel mechanism |

The honest target right now is **mid-tier venue** if the author commits 4–6 weeks of focused experiment work. Top-tier requires a year-scale data effort or a theoretical contribution.

---

## 3. Overall feedback (cross-cutting)

### 3.1 The asymmetry between the paper and the rest of the repo

Worth restating because it shapes how a reader perceives the work:

- The **paper** is appropriately modest and honest.
- The **README** and **UAISV_Final_Project_Summary.md** overclaim heavily ("Expected Academic Grade 95–100", "DistilBERT on Enron phishing", "CERT r4.2 insider threat", "ResNet/ViT vision") — none of which match what the code actually runs.
- The **reports/metrics_*.csv** files have hand-edited round numbers and empty `std` columns; they don't come from any real run.
- The **reports/uais_*.docx** files are 0 bytes.

A reviewer or hiring manager looking at the repo will see the marketing docs first, form a "this is overclaimed" impression, and approach the paper skeptically. **Trim the marketing docs to match the paper, not the other way around.** This is the single biggest win for repo-level credibility and costs a few hours.

### 3.2 The naming problem

"UAV" needs to die. Three reasons:

1. **In-abstract disambiguation against "unmanned aerial vehicle"** is a reviewer red flag. No published paper successfully owns an acronym already saturated in another field.
2. **Search/SEO is poisoned.** A future Google Scholar query for `UAV anomaly detection` returns drone work. Your own paper will be hard to find.
3. **The acronym constraint forces awkward wording.** "UAV system architecture" reads as drone work even after the disambiguation.

Suggested replacements: **MAVERICK**, **VERA**, **CRAFT**, **ARGUS** (see v2 review §3.4 for the long form). Pick one in a single 30-minute renaming session — `\renewcommand` makes it cheap.

### 3.3 The "system vs. method" framing is correct

The v2 reframing (system = UAV, method = UAV-RAF, benchmark = UAV-RealFusion) is the right scholarly move. It demotes the modest mechanism to a subsystem in a larger context, which is honest. Keep the framing; just rename.

### 3.4 The discussion section is too defensive

Sections II (Writing and Research Maturity), XI (Discussion / What the Results Do/Do Not Support), XII (Threats to Validity) collectively spend ~1.5 pages explaining what the paper is *not* claiming. This is overcorrection. Reviewers reward one strong "Limitations" section; they punish multiple defensive sections that read as anticipatory. **Merge II + XI + XII into one Limitations section.** Page count drops, the narrative improves, the modesty is preserved.

---

## 4. Gap analysis — engineering vs. research vs. evidence

This is the most important framing of all the audits combined.

| Layer | What's good | What's missing |
|---|---|---|
| **Engineering** (code, infra, repro) | Modular architecture, real test suite, deterministic scripts, honest metadata, MLOps surface (FastAPI, Streamlit, CI, pre-commit). | A few code-paper math gaps; orphaned figure asset; configs duplicated; synthetic fallbacks silently active. |
| **Research mechanism** (UAV-RAF) | Composition of standard components is sensible; reliability gate is a defensible design choice. | No theoretical justification; no comparison to test-time-adaptation prior work (Tent, TTT); no evidence that the composition outperforms its parts. |
| **Empirical evidence** | 5-seed clean evaluation; threats-to-validity acknowledged. | Saturated benchmark; dormant mechanism in most ablations; single-seed robustness; underpowered stats; narrow adversarial threat model. |
| **Scientific narrative** | System framing is correct; honest about construct validity. | Headline numbers (0.9997) don't match the actual story. Defensive structure repeats the same caveat in three sections. Naming (UAV) actively hurts impression. |
| **Repo-level credibility** | Code-paper consistency; honest in-paper metadata. | Marketing docs (README, summary, docx, csv) systematically overclaim, contradicting the paper. |

The **biggest gap** is empirical: the experiments don't isolate the mechanism. Once that's fixed, the rest of the gaps shrink.

---

## 5. The two questions that decide the next move

Before committing to the plan in §6, the author should answer two questions honestly.

### 5.1 What's the target?

| Target | Effort | Realistic timeline |
|---|---|---|
| Strong thesis chapter / arXiv preprint | Low | 1–2 weeks |
| Workshop submission (e.g., NeurIPS-W, ICML-W, IEEE S&P workshops) | Medium | 4–6 weeks |
| Mid-tier conference (CIKM, ICDM, IEEE BigData) | High | 8–12 weeks |
| Top-tier conference / journal | Very high | 6–12 months (likely with a thesis advisor or collaborator) |

The plan in §6 assumes the **workshop or mid-tier conference** target. For the top-tier target, layer §1.3 path 3 (theory) on top of §6.

### 5.2 What's the available data?

The current bottleneck is paired multimodal data. Decide which is true:

- **(a) No access to naturally co-observed multimodal data is possible.** Then the only path forward is to make the existing label-aligned benchmark *harder* (degrade scorers, increase missingness, multi-domain coherent stress) so that fusion has work to do.
- **(b) Access is possible** (CICIDS + auth logs, MIMIC-III + clinical notes, MMSec, etc.). Then a 2-week data-engineering effort produces a proper benchmark and the paper's headline becomes meaningful.

Most of the plan below is option (a). If option (b) is feasible, replace Phase 1 with a data-acquisition sprint and the paper jumps a venue tier.

---

## 6. 12-week plan

Phased so each phase produces a self-contained deliverable. If you stop after any phase, you still have something to submit at the corresponding venue tier.

### Phase 1 — Two-week credibility sprint *(thesis chapter / arXiv)*

**Goal:** make the paper match its evidence and not embarrass on first read.

| # | Task | Effort | Owner |
|---|---|---|---|
| 1.1 | Rename UAV → chosen alternative across `.tex`, configs, code (single `\renewcommand` + `sed`) | 1 day | author |
| 1.2 | Replace placeholder affiliation with real institution | 0.5 day | author |
| 1.3 | Demote clean benchmark table to appendix; replace headline metric with the all-domain coherent-attack delta | 1 day | author |
| 1.4 | Cite 4 unused bib entries (`zadrozny2002`, `guo2017`, `platt1999`, `zhou2012`) in Calibration / Ensembles paragraphs | 1 hour | author |
| 1.5 | Fix code-paper math gaps: positional embedding term in Eq. (2); per-domain $r_d$ vs per-sample $r_{i,d}$ disambiguation | 0.5 day | author |
| 1.6 | Reference Tables I and II from prose; delete or include orphaned `uav_evidence_map.png` | 1 hour | author |
| 1.7 | Drop $p=0.374$ from §VIII; replace with one-sentence "two methods indistinguishable on saturated split" | 1 hour | author |
| 1.8 | Merge §II + §XI + §XII into a single "Discussion and Limitations" section | 0.5 day | author |
| 1.9 | Reconcile [README.md](../../../README.md) and [UAISV_Final_Project_Summary.md](../../../UAISV_Final_Project_Summary.md) with paper claims; delete the 0-byte docx files | 0.5 day | author |
| 1.10 | Make `_synthetic_fraud` / `_synthetic_cyber` fallbacks opt-in (loud warning when triggered) | 0.5 day | author |

**Phase 1 deliverable:** an honest, internally consistent draft suitable for arXiv, the thesis chapter, or internal review.

### Phase 2 — Three-week mechanism isolation *(workshop submission)*

**Goal:** make UAV-RAF demonstrably do something measurable.

| # | Task | Effort |
|---|---|---|
| 2.1 | $\tau$-sweep ablation: $\tau \in \{0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9\}$. Report gate firing rate per condition + ROC-AUC delta. New table + figure. | 2 days |
| 2.2 | Reliability-component ablation: turn off ECE, then KS, then sharpness, then gate. 4-row table showing each component's contribution to the all-domain attack gain. | 1 day |
| 2.3 | Add Tent + TTT baselines (or a simple entropy-minimization fusion). Replace the "no test-time-adaptation comparator" gap. | 1 week |
| 2.4 | Re-run all robustness experiments across 5 seeds (was 1). Report bootstrap 95% CIs on each cell. | 3 days |
| 2.5 | Construct a *harder* benchmark variant: cap each domain scorer's training set to 5–10% of full data so OOF AUCs drop to ~0.7. Re-run clean evaluation. Now methods will spread out. | 3 days |
| 2.6 | Add learned gate: replace heuristic $\tau$ with a small classifier on $\bar r$. Train on synthetic-drift episodes generated from val. | 3 days |
| 2.7 | Threat-model paragraph: explicitly distinguish "coherent all-domain attack" (worst-case stress) from realistic threat models (independent compromise paths). | 2 hours |

**Phase 2 deliverable:** a paper with a measurable mechanism, real test-time-adaptation comparators, and a non-saturated headline. Workshop-submittable.

### Phase 3 — Six-week empirical depth *(mid-tier conference)*

**Goal:** turn the workshop draft into a conference paper by widening the evidence and adding one of the §1.3 novelty paths.

| # | Task | Effort |
|---|---|---|
| 3.1 | Pick **one** real co-observed multimodal dataset and pair properly (CICIDS+auth, MIMIC+notes, public phishing+URL+content, etc.). | 2 weeks |
| 3.2 | Re-run full benchmark on naturally paired data. This becomes the headline; label-aligned becomes a sanity check. | 1 week |
| 3.3 | Add per-domain coherent attack model (attacker subset $S \subset \{1..D\}$, gradient-aligned) instead of all-or-nothing zero/max. | 4 days |
| 3.4 | Bootstrap CIs everywhere; DeLong test for ROC-AUC comparisons; Holm-Bonferroni correction on the adversarial table. | 3 days |
| 3.5 | Failure-case visualization: hand-picked samples where UAV-RAF and static attention disagree, with CDA explanation, in a figure. | 2 days |
| 3.6 | Run-time / inference-cost section: UAV-RAF adds reliability computation per batch — measure overhead. | 1 day |
| 3.7 | Reviewer-anticipation pass: write the rebuttal in advance to the 5 hardest expected reviewer questions; fix the paper to preempt them. | 3 days |

**Phase 3 deliverable:** a conference-submittable paper with paired-data evidence.

### Phase 4 — Three-week theoretical / breadth pass *(top-tier polish, optional)*

**Goal:** add the §1.3-path-3 contribution if targeting a top venue.

| # | Task | Effort |
|---|---|---|
| 4.1 | Formalize the gate as a switching rule. State a regret/ECE bound under bounded score-shift assumption. Single proposition + proof sketch. | 1 week |
| 4.2 | Connect to the Kittler et al. 1998 combiner-theory frame. One paragraph relating sum-rule, product-rule, and the reliability-weighted attention as combiner instances. | 2 days |
| 4.3 | Generalize the benchmark to D > 4 domains (synthetic add-ons) to show scaling behavior. | 1 week |
| 4.4 | Cross-validate gate threshold via held-out drift episodes; remove the last manual hyperparameter. | 3 days |

**Phase 4 deliverable:** a top-venue-ready paper. Realistically, this phase is overflow into Phase 3 of a thesis-level effort.

---

## 7. The five questions a reviewer will ask first

Anticipate these in the next revision. Each has a 1-paragraph answer in the current paper, but a stronger version exists.

| # | Reviewer question | Current answer | Stronger answer |
|---|---|---|---|
| 1 | "Why is UAV-RAF different from confidence-weighted mean fusion?" | Confidence-weighted uses local score; UAV-RAF uses validation calibration + drift + sharpness. | Above + a 1-row example showing identical scores produce different weights when validation calibration differs across domains. |
| 2 | "Why a hard threshold $\tau=0.66$ instead of soft mixing?" | Conservative; prevents disturbance of clean predictions. | Above + the Phase 2.6 learned-gate result showing the heuristic is competitive with the learned alternative. |
| 3 | "Why these reliability weights $(0.45, 0.35, 0.20)$?" | Configurable hyperparameters. | Above + a sensitivity sweep showing the result is stable across $\pm 0.1$ on each. |
| 4 | "Why are the clean numbers all 0.9997?" | Benchmark is label-aligned and near-saturated. | Above + the Phase 2.5 harder-benchmark result where numbers spread out. |
| 5 | "Why no comparison to Tent/TTT?" | Future work in current draft. | Phase 2.3 head-to-head comparison. |

If the next revision answers all five proactively, the paper goes from "modest workshop-grade" to "defensible mid-tier-conference-grade."

---

## 8. Bottom line, restated

You have **a real piece of engineering** (the codebase) wrapped in **an honestly written paper** that **does not yet isolate a research result**. The fix is empirical, not editorial. Two weeks of $\tau$-sweep, k-of-D corruption, harder benchmarks, and Tent/TTT comparators converts the current draft from "competent but indecisive" into "we measurably help in conditions where prior fusion fails." That's a workshop paper. Add a real co-observed dataset plus the corrected theorem stack with matching experiments and it becomes a stronger conference paper.

The hardest decision is naming. Make it today; everything else flows from it.
