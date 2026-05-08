# Manuscript Writing-Quality Review — `PAPER_DRAFT_v1.tex` (v2 revision)

**Paper:** *UAV: Reliability-Aware Multimodal Anomaly Verification with Real-Domain Score Fusion*
**Source:** [docs/research/PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) (707 lines, IEEEtran 10pt conference)
**Compiled:** [output/pdf/PAPER_DRAFT_v1.pdf](../../../output/pdf/PAPER_DRAFT_v1.pdf)
**Companion:** see [WRITING_QUALITY_REVIEW.md](WRITING_QUALITY_REVIEW.md) for the v1 baseline.
**Date:** 2026-05-07

---

## 0. Verdict

This is a **substantial improvement** over v1. The paper is no longer a method-and-protocol stub; it is now a complete systems paper with real numbers, a real benchmark generator on disk, and an honest discussion of construct validity. Most Tier-1 fixes from the v1 review are addressed. **However**, three concerns now dominate and must be resolved before the paper is publishable at PhD/IEEE level:

1. **The benchmark is near-saturated by construction.** Five of six methods score 0.9997 ROC-AUC with no statistical separation. This is not a property of a good fusion model; it is a property of label-aligned multimodal data. The paper acknowledges the limitation, but the headline numbers still read as breakthrough-ish results when they are not.
2. **The reliability gate never fires in the missing-domain ablation.** Across dropout 0.0–0.5, deltas are essentially zero ([craf_missing_results.tex](../tables/craf_missing_results.tex)). The paper says "the gate reverts to static" — i.e., **\method{} is identical to static attention on the most cited robustness sweep**, except in two extreme adversarial cases. This needs an honest re-positioning.
3. **The system name "UAV" is a high-confidence reviewer-flag.** The author has to write "Here \system{} denotes the local anomaly verification system, not an aerial-vehicle platform" *in the abstract* ([tex:abstract](../PAPER_DRAFT_v1.tex)). That sentence will not survive peer review. Rename.

Everything else (rigor, scope, threats-to-validity, reproducibility) is materially better than v1.

---

## 1. What v1 issues are fixed

| v1 issue | v2 status | Evidence |
|---|---|---|
| Placeholder affiliation (institution missing) | ❌ still present | [tex:30-34](../PAPER_DRAFT_v1.tex) — affiliation is still `\systemfull{} Research System / Local research codebase / Manuscript source: docs/research/PAPER_DRAFT_v1.tex`. Submission-blocker. |
| No figures | ✅ fixed | 8 figures present in [docs/research/figures/](../figures/): `uav_system_architecture.png`, `craf_clean_benchmark.png`, `craf_missing_modality.png`, `craf_drift_curves.png`, `craf_adversarial_delta.png`, `craf_calibration.png`, `craf_cda_impacts.png`, `uav_evidence_map.png`. |
| Bibliography had 4 uncited entries | ⚠️ partial | Now: 9 cited / 13 total. Uncited: `zadrozny2002transforming`, `guo2017calibration`, `platt1999probabilistic`, `zhou2012ensemble`. The Calibration paragraph at §III.C *names* "Post-hoc calibration methods" with three citations queued — but the actual `\cite` calls were dropped from the prose. Check [tex:Related Work, calibration paragraph](../PAPER_DRAFT_v1.tex). |
| `MeanFusion` baseline missing from code | ✅ no longer claimed | v2 lists baselines as "confidence-weighted mean fusion, early-fusion MLP, late-fusion ensemble, random forest fusion, and static attention" — all five exist in [src/uais/fusion/attention/baselines.py](../../../src/uais/fusion/attention/baselines.py). The earlier "simple mean score fusion" entry was correctly removed. |
| Per-sample vs batch reliability inconsistency | ⚠️ partial | [tex:Problem Formulation](../PAPER_DRAFT_v1.tex) still defines $r_i \in [0,1]^D$ as per-sample, but the implementation at [reliability_estimator.py:164](../../../src/uais/fusion/attention/reliability_estimator.py#L164) still broadcasts a domain-level scalar to all rows. Either update the math to $r_d$ + reliability gate operates on $\bar{r}$ (which is what the prose actually does — see §V.C), or implement per-sample reliability. |
| Equation $h_{i,d} = g_d(x_{i,d}) + e_d$ missed positional embedding | ⚠️ unchanged | Still missing $p_d$ from the equation. Code at [cross_modal_attention.py:90-92](../../../src/uais/fusion/attention/cross_modal_attention.py#L90-L92) still adds positional embeddings. |
| Datasets table said "CERT or web-session, Enron or weakly labeled news" | ✅ committed | v2 commits to "Credit Card Fraud, UNSW-NB15, Online Shoppers Intention, and labeled news text" in the abstract and Table III. The honest naming is preserved in §VII.A. |
| Specific seeds + 70/15/15 not stated | ✅ stated | "five seeds 42 through 46" + train/val/test ratio implied by the YAML. The exact split fractions (test_size: 0.2, val_size: 0.1) come from [configs/attention_real_fusion.yaml:44-45](../../../configs/attention_real_fusion.yaml#L44-L45) — call them out in §VII.C. |
| Honest withholding of results | ✅ replaced with measured results | v2 reports five-seed clean numbers and stress-test deltas. |
| No threats-to-validity | ✅ added | Full §X: internal, construct, external, statistical. |
| No algorithm box | ⚠️ partial | Table II ([tex:Algorithm Summary](../PAPER_DRAFT_v1.tex)) is a bulleted procedure, not a formal `\begin{algorithm}` block. For PhD/IEEE, use the `algorithm` package. |

---

## 2. What's new and good

- **Real benchmark generator on disk**: [src/scripts/prepare_real_fusion_benchmark.py](../../../src/scripts/prepare_real_fusion_benchmark.py) builds 8,000 composite samples with explicit OOF score generation per domain and writes [experiments/fusion/real_domain_fusion_inputs.csv](../../../experiments/fusion/real_domain_fusion_inputs.csv) plus a metadata JSON. The metadata literally contains: `"important_limitation": "Domains are sampled from real datasets and aligned by binary label; they are not naturally co-observed entities."` — this kind of self-flagging is good research hygiene.
- **Reliability gate as a separate concept** (§V.C): "If $\bar{r} \geq \tau$, \system{} uses the static attention path." This is conceptually cleaner than v1's continuous reliability injection.
- **Failure analysis section** (§IX.C): correctly identifies four real failure modes — label-aligned construction, conservative gate, RF-fusion-still-wins-F1, missing natural pairing.
- **System reframing** (§I.A–§I.D): the explicit demotion of CRAF to "one subsystem in a larger anomaly-verification architecture" is the right scholarly move. v1 over-claimed novelty for what is essentially a calibration-aware reweighting of the attention block.
- **Construct-validity discussion** (§X.B): naming the construct boundary ("the model has not learned causal or temporal relationships among fraud, cyber, behavior, and text domains") is exactly what reviewers will ask for.
- **Bibliography expanded with relevant entries**: `ovadia2019can` (calibration under shift), `breiman2001random`, `davis2006relationship` (PR vs ROC), `delong1988comparing` — all useful.

---

## 3. Critical concerns (these dominate the v2 review)

### 3.1 The clean benchmark is saturated

Look at [tables/craf_clean_results.tex](../tables/craf_clean_results.tex):

| Method | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Random forest | 0.9997±0.0002 | 0.9994±0.0004 | 0.9906±0.0008 |
| Conf.-weighted mean | 0.9982±0.0007 | 0.9955±0.0027 | 0.9794±0.0042 |
| Early fusion MLP | 0.9997±0.0001 | 0.9993±0.0002 | 0.9837±0.0043 |
| Late fusion ensemble | 0.9996±0.0001 | 0.9992±0.0002 | 0.9832±0.0029 |
| Static attention | 0.9997±0.0001 | 0.9994±0.0002 | 0.9896±0.0012 |
| UAV-RAF attention | 0.9997±0.0001 | 0.9994±0.0002 | 0.9894±0.0010 |

Five of six methods are tied at 0.9997 ROC-AUC. The standard deviations are 1–4 in the fourth decimal. This is not a useful clean comparison; it is the noise floor.

The cause is the benchmark construction. From [prepare_real_fusion_benchmark.py:1-7](../../../src/scripts/prepare_real_fusion_benchmark.py): "domain observations are sampled from real fraud, cyber, behavior, and text datasets, **aligned by binary label**." Combined with the fact that each domain scorer alone reaches OOF ROC-AUC of 0.97 (fraud), 0.97 (cyber), 0.87 (behavior), 0.997 (NLP) ([metadata JSON](../../../experiments/fusion/real_domain_fusion_metadata.json)), label alignment makes any of the four domain scores a near-perfect predictor of the joint label. **There is no fusion problem to solve on the clean split.**

What the paper should do:
1. **Demote the clean table.** Move it to an "appendix sanity check" position in the narrative. Saying "we are competitive on saturated data" is uncontroversial and reviewers won't push back.
2. **Promote a harder split.** Construct a *non-label-aligned* version: pair each composite by random domain assignment, only matching the joint label probabilistically. The current saturation will collapse, and the relative method ranking will become meaningful. This is the experiment that justifies the paper.
3. **Or: shrink the clean benchmark to 1–2 domains per sample.** With only 1–2 domains available per composite and 12% missingness, fusion has more to do. The 4-of-4 high-coverage current config is too informative.
4. The paragraph in §VIII that *acknowledges* the saturation ("clean performance is near saturated") is correct but understates the issue. Reviewers reading the abstract see "0.999736" and judge that as the headline; subsequent prose can't claw it back.

### 3.2 The reliability gate is invisible across the most-cited robustness sweep

Missing-domain ablation deltas from [tables/craf_missing_results.tex](../tables/craf_missing_results.tex):

| Dropout | Static ROC | UAV-RAF ROC | Delta |
|---|---|---|---|
| 0.0 | 0.9999 | 0.9999 | 0.0000 |
| 0.1 | 0.9995 | 0.9994 | -0.0001 |
| 0.2 | 0.9982 | 0.9982 | 0.0000 |
| 0.3 | 0.9959 | 0.9959 | 0.0000 |
| 0.5 | 0.9774 | 0.9774 | 0.0000 |

The paper writes (§IX.A): *"From 0.2 through 0.5 dropout, the reliability gate reverts to the static path and the two methods are identical."* This is honest but it's a damning admission. **The proposed mechanism is mechanistically inactive across the headline robustness experiment.** A reviewer will ask, reasonably: *if your gate threshold $\tau=0.66$ never triggers under these dropouts, what's the mechanism doing?*

Drift summary at [tables/craf_drift_summary.tex](../tables/craf_drift_summary.tex) is similar: per-domain `delta_curve_auc` values are all 0.3000 / 0.2999, with "yes/no" verdicts that are coin flips on the fourth decimal.

**Required action:**
1. Run a $\tau$-sweep ablation: $\tau \in \{0.4, 0.5, 0.66, 0.8, 0.9\}$. Show that the headline gain on the all-domain attacks is preserved while no-harm-on-clean is also preserved. This is the experiment that converts "the gate doesn't fire" from a defect into a design choice.
2. Report when the gate *does* fire across the missing-domain sweep at lower thresholds. If the gate firing rate is, say, 40% under dropout 0.3 at $\tau=0.5$ and the resulting performance change is positive, that's a real signal.
3. Restate in §V.D that the gate is *deliberately* conservative and tuned to protect clean accuracy. Currently the framing is post-hoc.

### 3.3 The headline adversarial gain needs deeper unpacking

From [tables/craf_adversarial_results.tex](../tables/craf_adversarial_results.tex), all single-domain attacks have delta $\leq 0.0001$. The headline gains are exclusively on *all-domain coherent* attacks:

- `zero_attack / all`: 0.7216 → 0.7669 (Δ = +0.0453)
- `max_attack / all`: 0.7648 → 0.8083 (Δ = +0.0435)
- `gaussian_noise / all`: 0.9998 → 0.9997 (Δ = -0.0001)

A reviewer will ask: *what's an "all-domain zero attack" in a real threat model?* It's setting every domain's score to 0 simultaneously across all samples. The fact that ROC-AUC is **0.72 even when scores are zeroed** means the model is reading non-score evidence — embedding columns, domain embeddings, or the missing_embedding parameter ([cross_modal_attention.py:92](../../../src/uais/fusion/attention/cross_modal_attention.py#L92)). Two things follow:

1. The +0.045 ROC-AUC gain when all scores are zeroed is not really a fusion-of-scores story; it's a story about how attention reweights *embedding-level evidence* when scores agree to be uninformative. This is a defensible result, but the paper should reframe it: "When score-level evidence is collapsed, \method{} reweights residual embedding evidence more conservatively than static attention." The current framing implies score-level robustness, which the single-domain attack columns *don't* support.
2. The threat model "an attacker zeros every domain's score simultaneously" is unusually strong. The paper should clarify whether this is intended as a worst-case stress test (fine) or a realistic attack (it is not — domain experts are typically run by independent teams with independent compromise paths).

### 3.4 "UAV" as a system name will not survive peer review

The paper has to disambiguate "UAV" *in the abstract* ([tex:abstract](../PAPER_DRAFT_v1.tex)): "Here \system{} denotes the local anomaly verification system, not an aerial-vehicle platform." Any acronym that requires this kind of pre-emptive defense is the wrong acronym.

Reviewers searching arXiv/IEEE Xplore for "UAV" get unmanned aerial vehicle results. Co-authoring with someone in robotics will get awkward fast. Search/SEO is also poisoned: a Google query for `"UAV" anomaly detection` returns drone-anomaly papers, not this work.

**Recommended replacements (any of these):**
- **MAVERICK** (Multi-domain Anomaly Verification with Reliability-aware Calibrated Knowledge) — keeps the verification framing, no acronym collision
- **ARGUS** (Anomaly Reliability Gate for Unified Scoring) — classical reference, single domain in CS
- **CRAFT** (Calibration- and Reliability-Aware Fusion with Trust gating) — close to v1's `CRAF` but extended; less acronym overload than UAV

If reusing v1's `CRAF` as the *method* (instead of `UAV-RAF`), with **VERA** or similar as the *system* name, would also work. The system/method split is a good v2 idea; it just got paired with bad acronyms.

---

## 4. Statistical reporting issues

### 4.1 n=5 paired t-test is underpowered

§VIII reports $p=0.374$ for static-vs-\method{} clean ROC-AUC over five seeds. With $n=5$ and observed effect size at the $10^{-4}$ scale, the power to detect any difference is approximately zero regardless of whether one exists. Either:

1. Do not report the p-value; just say "the two methods are indistinguishable on the clean split" and move on, **or**
2. Run more seeds (15–30) so the test has any chance of being meaningful.

The current p=0.374 reads as "we tried to claim significance and failed" rather than "we did not claim significance." Removing the p-value entirely is cleaner.

### 4.2 No bootstrap CIs on robustness deltas

§X.A flags this honestly: "robustness ablations are currently diagnostic on the final configured seed." The fix is mechanical — [src/uais/utils/stats.py](../../../src/uais/utils/stats.py) already exposes `bootstrap_ci`. Re-run each adversarial cell with $n_{boot}=1000$ and report 95% CIs in the delta column.

### 4.3 No correction for multiple comparisons

The adversarial table runs 15 cells (3 attacks × 5 targets), each silently a hypothesis test. Reporting raw deltas is fine; if any p-values appear later, apply Holm-Bonferroni at minimum.

---

## 5. Citations and prose-level issues

### 5.1 Uncited bibliography entries

Four entries in `\begin{thebibliography}` are never `\cite{}`-d:

| Key | Right place to cite | Recommended action |
|---|---|---|
| `zadrozny2002transforming` | §III.C calibration paragraph (named "Post-hoc calibration methods") | Insert `\cite{zadrozny2002transforming}` |
| `guo2017calibration` | Same paragraph | Insert `\cite{guo2017calibration}` |
| `platt1999probabilistic` | Same paragraph | Insert `\cite{platt1999probabilistic}` |
| `zhou2012ensemble` | §III.D ensembles paragraph | Insert `\cite{zhou2012ensemble}` after "Ensemble methods remain difficult to beat" |

`latexmk` will warn on these on the next compile. Five-minute fix.

### 5.2 Still-missing prior-work citations

A PhD-level submission needs at least one citation each for:
- **Test-time adaptation** — Wang et al. (Tent) or Sun et al. (Test-Time Training).
- **Counterfactual explanation** — Wachter, Mittelstadt & Floridi (2017) for the foundational definition; Mothilal et al. (2020) on DiCE.
- **Anomaly-detection ensembles** — Aggarwal & Sathe (2017) *Outlier Ensembles*.
- **Score-fusion theory** — Kittler et al. (1998) "On combining classifiers" remains the canonical reference for why mean/product/sum rules behave the way they do under different noise assumptions. Highly relevant to the choice of confidence-weighted mean as a baseline.

### 5.3 Prose issues to address

- Abstract opens with "This manuscript presents \system, a \systemfull{} research system…" — three nouns in one phrase. Tighten to "This paper presents \system, a research system for heterogeneous anomaly verification."
- §II ("Writing and Research Maturity") is unusual for an IEEE conference paper. Most reviewers will read it as defensive. The honest content (claim discipline, distinguishing implementation/measured/future) is good; consider folding it into the Limitations section or omitting entirely. PhD theses can have this section; conference papers usually shouldn't.
- "Out-of-fold" appears 4× in §VII.A and Table III without ever being defined. One sentence: "Domain scorers are trained with 3-fold stratified cross-validation; predicted scores for each fold are produced by the model trained on the other two folds." (See [prepare_real_fusion_benchmark.py:35](../../../src/scripts/prepare_real_fusion_benchmark.py).)
- The phrase "static attention path" appears 6× in §V; alternate with "static fusion" or "attention-only" to prevent fatigue.
- §IX.B closes with: "All differences are small because clean performance is near saturated." That sentence belongs in §VIII, not §IX. Move it.

---

## 6. Method-vs-implementation deltas (refresh of §4 from v1 review)

| # | Paper says | Code does | Status |
|---|---|---|---|
| 1 | $r_i \in [0,1]^D$ per-sample (§IV) | Domain-level scalar broadcast ([reliability_estimator.py:164](../../../src/uais/fusion/attention/reliability_estimator.py#L164)) | ❌ unchanged from v1 |
| 2 | $h_{i,d} = g_d(x_{i,d}) + e_d$ (§VI.A) | Adds positional + missing embeddings ([cross_modal_attention.py:90-92](../../../src/uais/fusion/attention/cross_modal_attention.py#L90-L92)) | ❌ unchanged from v1 |
| 3 | Reliability gate at $\tau=0.66$ (§VI.C) | `clean_gate_threshold: 0.66` in [configs/attention_real_fusion.yaml:72](../../../configs/attention_real_fusion.yaml#L72) | ✅ matches |
| 4 | $(\alpha,\beta,\gamma)=(0.45,0.35,0.20)$ (§VI.B) | `ece_weight 0.45, ks_weight 0.35, sharpness_weight 0.20` ([config](../../../configs/attention_real_fusion.yaml#L68-L71)) | ✅ matches |
| 5 | "five seeds 42 through 46" | `seeds: [42, 43, 44, 45, 46]` ([config](../../../configs/attention_real_fusion.yaml#L56)) | ✅ matches |
| 6 | "8,000 composite samples, four domains, eight embedding features per domain" | `samples: 8000`, `domain_order: [fraud, cyber, behavior, nlp]`, `embedding_dim: 8` ([metadata](../../../experiments/fusion/real_domain_fusion_metadata.json)) | ✅ matches |
| 7 | "0.307 positive rate" | `positive_fraction_actual: 0.306875` (metadata) | ✅ matches |
| 8 | "11.8% nominal missingness" | `missing_probability: 0.12` (metadata) | ⚠️ minor — paper says 11.8%, metadata says 12% nominal. Probably actual sample average is 0.118. Reconcile. |

---

## 7. Required actions to reach PhD-submission quality

### Tier 0 — kill-issues (cannot submit without)
1. Replace placeholder affiliation at [tex:30-34](../PAPER_DRAFT_v1.tex) with a real institution.
2. **Rename the system.** "UAV" requires in-abstract disambiguation against unmanned aerial vehicles.
3. Add the non-label-aligned harder benchmark (or move clean results to appendix). The current headline of 0.9997 is meaningless and a reviewer will say so in one paragraph.

### Tier 1 — required to be defensible
4. Run $\tau$-sweep ablation showing where the reliability gate fires and what changes when it does.
5. Re-run all robustness experiments across 5 seeds; replace single-seed deltas with mean ± bootstrap-CI.
6. Reframe the all-domain adversarial gain as "reliability reweighting of residual embedding evidence" rather than score-level robustness, OR add a real per-domain coherent attack model.
7. Drop the n=5 p-value or run enough seeds to make it meaningful.
8. Resolve the per-sample vs per-domain reliability math/code mismatch.
9. Fix the equation $h_{i,d}$ to include the positional term that's actually used in code.

### Tier 2 — strongly recommended
10. Cite the four uncited entries (`zadrozny2002`, `guo2017`, `platt1999`, `zhou2012`).
11. Add 3–4 missing prior-work citations: test-time adaptation, counterfactual explanations, anomaly ensembles, Kittler et al. on combiner theory.
12. Replace Table II algorithm-summary with a formal `algorithm` block.
13. Add a sentence defining "out-of-fold" in §VII.A.
14. Move §II ("Writing and Research Maturity") into Limitations or remove.
15. Reconcile the 11.8% vs 12% missingness wording.

### Tier 3 — polish
16. Tighten abstract opener.
17. Decimate the repeated phrase "static attention path."
18. Move the "near saturated" sentence from §IX.B to §VIII where it belongs.
19. Add a `\IEEEspecialpapernotice{(Draft)}` marker until results are final.
20. Run `latexmk` and clear all warnings.

---

## 8. Comparison: v1 → v2 progress map

| Dimension | v1 | v2 | Δ |
|---|---|---|---|
| Honesty about results | "results withheld" | full numbers + threats-to-validity | ↑↑ |
| Code-paper consistency | mostly matched | same (with two unchanged math gaps) | = |
| Citation hygiene | 4 unused / 8 total | 4 unused / 13 total | ↑ (more cited, but same uncited count) |
| Figures | 0 | 8 | ↑↑ |
| Statistical rigor | not yet measured | n=5 reported, single-seed robustness | ↑ but not yet PhD-level |
| Construct validity discussion | brief mention in Limitations | dedicated §X.B + abstract disclosure | ↑↑ |
| System framing | method-only | system-then-method | ↑↑ (correct move) |
| Naming | `CRAF` (clean, anomaly-detection neutral) | `UAV / UAV-RAF / UAV-RealFusion` | ↓↓ (UAV is the wrong word) |
| Headline result believability | nothing to evaluate | 0.9997 ROC-AUC reads as breakthrough but is saturation | new concern |
| Reliability mechanism activity | n/a | **inactive across most ablations** | new concern |
| Adversarial robustness story | n/a | strong on coherent all-zero, weak on per-domain | mixed |

---

## 9. Bottom line

v2 turns this into a real systems paper. The scaffolding is right, the code matches the prose, the threats-to-validity discussion is unusually honest, and the `prepare_real_fusion_benchmark.py` artifact is well-built. **The blocker now is empirical, not editorial:** the headline numbers come from a benchmark that is too easy to be informative, and the proposed reliability mechanism is dormant across most of the robustness sweep. Fix those two things — by introducing a non-label-aligned harder benchmark and running a $\tau$-sweep that shows the gate doing measurable work — and the paper crosses from "competent draft" to "submittable."

Also: rename "UAV." Whatever else gets done, that name will not survive review.
