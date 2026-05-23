# Reviewer-Style Rating + Gap Analysis + Upgrade Phase Plan

**Scope:** synthesis across [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex)
(IEEE conference manuscript), [THESIS_CHAPTER_v1.tex](../THESIS_CHAPTER_v1.tex)
(graduate thesis chapter), the rendered PDFs at
[output/pdf/PAPER_DRAFT_v1.pdf](../../../output/pdf/PAPER_DRAFT_v1.pdf) and
[output/pdf/THESIS_CHAPTER_v1.pdf](../../../output/pdf/THESIS_CHAPTER_v1.pdf),
all generated tables and figures, the runner code, and the test suite.
**Reviewer pass:** scope, novelty, evidence, statistical rigor,
code-vs-paper consistency, presentation, and venue fit.
**Date:** 2026-05-14

---

## 0. Bottom line

**Conference manuscript: 5/10 — borderline workshop, weak conference.**
**Thesis chapter: 6/10 — defensible chapter draft with one strong contrastive finding.**

Both documents share the same evidence base. After the recent MVTec 3D-AD
multi-category run, the headline of the work has flipped from "gate helps on
label-aligned coherent attacks" to a more interesting *cross-benchmark
contrast*: the same KS-drift signal that delivers all the label-aligned stress
gain mis-fires on legitimate inter-category variation in naturally paired
data. That contrast is the most reviewer-defensible result on the page right
now. Both documents now report it honestly, but neither yet builds it into a
fully developed scientific narrative — that's the main upgrade target.

---

## 1. Reviewer-style ratings

### 1.1 Conference manuscript ([PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex), 16 pp.)

| Axis | Score | Comment |
|---|---|---|
| **Originality** | 5/10 | The reliability gate is a sensible composition of known signals (ECE + KS + sharpness). The cross-benchmark contrast is the only piece that reads as genuinely original — currently buried in §IX.D.5 rather than promoted. |
| **Significance** | 4/10 | On naturally paired data the proposed method *under-performs* a random forest baseline by a wide margin. The label-aligned gain is real but the threat model is narrow (coherent all-domain collapse only). Significance hinges on positioning the negative result as the contribution. |
| **Soundness** | 7/10 | Five-seed clean evaluation with bootstrap CIs on every delta. Mechanism isolation (τ-sweep, component ablation) is run on both benchmarks. Tent + TTT baselines are included. No silent leakage in the fusion pipeline that I can see. |
| **Clarity** | 7/10 | Section structure is clean (10 sections, see [structure check](../PAPER_DRAFT_v1.tex)). Notation is consistent. Tables and figures are all referenced from prose. The IEEEtran build is clean (1 cosmetic `Overfull` warning, no undefined refs). |
| **Reproducibility** | 8/10 | Single-command rebuild via [scripts/rebuild_paper.sh](../../../scripts/rebuild_paper.sh). Configs, seed lists, and per-seed result JSONs are checked in. 205 pytest pass + smoke pipeline. |
| **Citations** | 7/10 | 17 entries, all cited. Test-time-adaptation prior work (Tent, TTT) and RGB--3D anomaly detection (MVTec 3D-AD line, M3DM, Crossmodal Feature Mapping, Real3D-AD, M3DM-NR) properly cited. Still missing: anomaly-ensemble foundations (Aggarwal & Sathe), Kittler combiner-theory, Wachter counterfactuals. |
| **Overall (conference)** | **5/10** | Sits at the workshop / weak-conference boundary. Needs a stronger benchmark result (or a stronger framing of the negative result) to clear a mid-tier conference like CIKM / IEEE BigData. |

### 1.2 Thesis chapter ([THESIS_CHAPTER_v1.tex](../THESIS_CHAPTER_v1.tex))

| Axis | Score | Comment |
|---|---|---|
| **Scope appropriateness** | 7/10 | Thesis-style: full background section, expanded threats-to-validity discussion, explicit failure-cases subsection, explicit "what did not work" framing. The chapter's voice is more reflective than the conference manuscript. |
| **Evidence development** | 6/10 | Uses the same artifacts as the conference paper. The Cross-Benchmark Contrast subsection (just added) is the strongest piece. The "what did not work" framing is good, but could be elevated to its own dedicated section. |
| **Methodological rigor** | 7/10 | Same as conference: 5-seed CIs, mechanism isolation, real baselines. Construct-validity and statistical-validity discussion are stronger here than in the conference paper. |
| **Originality of contribution** | 5/10 | Same evidence base as the conference paper; same novelty ceiling. The contrast finding could become more central in a chapter than in a 16-page conference paper. |
| **Writing quality** | 7/10 | Clear, honest, reasonably concise. Abstract was stale (still said "bagel-only 78 obs"); just updated to match the new evidence. Conclusion was stale on the same point; updated. |
| **Overall (thesis chapter)** | **6/10** | Defensible chapter draft. The contrastive finding gives it real intellectual content. Needs one more pass that explicitly elevates the negative result as a methodological caveat rather than a sub-finding. |

---

## 2. Strongest pieces (what already lands)

1. **The cross-benchmark contrast** ([PAPER §IX.D.5](../PAPER_DRAFT_v1.tex),
   [THESIS §sec:thesis-contrast](../THESIS_CHAPTER_v1.tex)). On label-aligned
   synthetic pairing the gate fires on coherent attacks and helps; on natural
   MVTec 3D-AD pairing the same gate fires on legitimate category variation and
   hurts. The same KS component drives both behaviors. This is a real,
   publishable methodological finding.
2. **Component ablation as falsification**
   ([rga_component_ablation_results.tex](../tables/rga_component_ablation_results.tex)).
   Removing the ECE term *improves* the LA stress gain; removing the KS term
   eliminates the gain entirely. This is exactly the kind of ablation that
   distinguishes a thoughtfully designed mechanism from a kitchen-sink one.
3. **Honest construct-validity flagging.**
   [`real_domain_fusion_metadata.json`](../../../experiments/fusion/real_domain_fusion_metadata.json)
   literally contains `"important_limitation": "...not naturally co-observed
   entities."`. Both papers reference this property in prose. Self-flagging at
   the artifact level is unusually disciplined.
4. **Single-source-of-truth pipeline.** Both manuscripts pull the same
   `rga_*.tex` and `mvtec3d_*.tex` tables from the same JSON artifacts.
   Changes in evidence cannot diverge between the two documents without
   re-running the pipeline.

---

## 3. Gaps (ordered by impact on reviewer perception)

### G1. The contrastive finding is not the headline of the paper
The current §VIII still has *"Primary Headline Results: MVTec 3D-AD"* as its
title — implying the MVTec results are the headline. They are, but the
*finding* is "the gate hurts," not "the gate helps." That should appear in
the introduction's contribution list and in the abstract's first sentence,
not just inside §IX.D.5. As written, a hurried reviewer can skim §I + abstract
and not realize the paper's central scientific point is a negative result.

### G2. Performance ceiling is held back by the toy MVTec scorer
The MVTec scores come from
[`prepare_mvtec3d_fusion_benchmark.py`](../../../src/scripts/prepare_mvtec3d_fusion_benchmark.py)'s
normal-reference image-statistic baseline (mean/std/quantiles of grayscale
intensities). A reviewer will reasonably say: *"of course attention does badly
on this — random forest fusion at 0.959 is competing against domain scorers
that only see five quantile statistics of an image."* The gate's failure is
visible because the underlying scorer is weak, which creates ambiguity about
whether the gate's regression is intrinsic to the mechanism or an artifact of
the scorer. **Phase B fix:** plug in M3DM-style or PatchCore RGB and depth
features and re-run.

### G3. The label-aligned benchmark has no statistical separation
Five of seven methods are at $0.9997 \pm 0.0001$ ROC-AUC on the clean
label-aligned table ([rga_clean_results.tex](../tables/rga_clean_results.tex)).
The paper acknowledges this honestly ("near saturated"), but a reviewer reading
the table sees rows that all look identical and will ask why the headline
result is built on a benchmark that doesn't separate methods. **Phase B fix:**
re-run the LA benchmark with `--scorer-train-fraction 0.05` so individual
domain scorers degrade below saturation, then re-render the table.

### G4. Missing-domain ablation is unhelpfully flat
[`rga_missing_results.tex`](../tables/rga_missing_results.tex) shows
$\Delta = 0.0000$ for every dropout level. The τ-sweep explains why (gate
adaptation rate stays at 3.2% for clean data at τ=0.66), but a reader who only
looks at the missing-domain table without reading the mechanism-isolation
section will conclude that the mechanism is dead. **Phase B fix:** either
(a) re-run the missing-domain ablation at τ ∈ {0.66, 0.80, 0.90} so the gate
actually fires and the table has variance, or (b) drop the missing-domain
section from the paper.

### G5. No comparison to a learned gate
A learned gate [`learned_gate.py`](../../../src/uais/fusion/attention/learned_gate.py)
exists in the code but is not in any reported table. A reviewer will ask:
*"have you tried a learned gate?"* — and the answer should be in the paper, not
in the codebase. **Phase B fix:** add a row to the τ-sweep table for the
learned gate.

### G6. Threat-model paragraph is single-domain-only
The threat-model paragraph (§VII.G, just added) correctly notes that
all-domain coherent attacks are unrealistic. But the paper still reports the
all-domain gain ($+0.064$ ROC-AUC on `zero_attack:all`) as the main stress
result. **Phase B fix:** promote a *per-domain subset* attack model (e.g.\
"attacker compromises 1 of D domains") as the primary adversarial table; keep
all-domain as the worst-case upper bound.

### G7. Calibration and CDA tables don't have CIs
[`rga_calibration_cda.tex`](../tables/rga_calibration_cda.tex) reports
single-seed numbers. The paper says CIs are computed "where available" but
calibration is one of the more interesting stories (\method{} is
*better-calibrated* than confidence-weighted mean and worse than static), and
it deserves seed-level CIs. **Phase C fix:** aggregate `table_5_calibration`
across seeds in the asset generator.

### G8. Failure-cases table is brief and not visualized
[`rga_failure_cases.tex`](../tables/rga_failure_cases.tex) has 3 rows. A
reviewer will want to see a scatter (static probability vs RGA probability,
colored by label) for many more cases — the
[`plot_failure_cases`](../../../src/scripts/generate_craf_paper_assets.py)
plotter exists but the table is what gets cited. **Phase C fix:** elevate the
scatter from supplementary material to a numbered figure in the paper, and
extend the table to 8–10 rows.

### G9. Thesis chapter still missing a dedicated "What Did Not Work" section
The thesis claims this framing in the abstract but the actual evidence is
scattered: clean ROC penalty in §VIII, missing-domain dormancy in §IX.A, drift
near-zero deltas in §IX.B, MVTec contrast in §IX.D.5. **Phase C fix:** new
§Negative Results section that pulls all five "what did not work" items
together.

### G10. Bibliography missing four foundational citations
Despite being long (17 entries), the bibliography still lacks
(a) Aggarwal & Sathe (anomaly ensembles foundation),
(b) Kittler et al. 1998 (classifier combiner theory),
(c) Wachter et al. 2017 (counterfactual-explanation foundation),
(d) Geifman & El-Yaniv 2017 (selective prediction — the closest prior work
to the gate). **Phase A fix:** add them.

---

## 4. Reviewer Q&A preempt (what reviewers will ask)

| # | Likely reviewer question | Strongest current answer | Strongest possible answer |
|---|---|---|---|
| 1 | *Why does the gate hurt on natural paired data?* | KS-drift mis-fires on inter-category variation; component ablation isolates it. | Above + a τ-sweep table on MVTec showing the *no-firing* baseline (τ ≤ 0.6) is competitive with static attention — meaning the mechanism is well-isolated and the harm is exactly when the gate fires. (Already in table; needs prose.) |
| 2 | *Why is random forest fusion so much better than attention on MVTec?* | The MVTec scorers are lightweight image-statistic features; random forest exploits non-linear interactions better than 32-dim attention at this scorer level. | Above + a Phase-B re-run with strong RGB/3D features that brings attention into competitive range. |
| 3 | *Why a fixed threshold τ = 0.66?* | Conservative; preserves clean ranking. | Above + a learned-gate baseline showing the heuristic threshold is within 1 std of the learned alternative. (Code exists; needs to be wired into the experiment runner.) |
| 4 | *Why is the all-domain attack a meaningful threat model?* | Worst-case stress test, explicit threat-model paragraph in §VII.G. | Above + a per-domain subset attack table that is the primary headline; all-domain as a stress upper bound. |
| 5 | *Why not include CDA explanations as a baseline?* | CDA is included as a domain-attribution mechanism, not as a competing fusion method. | Above + a Spearman correlation between CDA impacts and SHAP attributions (already computed; needs to be cited in prose). |
| 6 | *Why these three reliability signals and not others?* | The component ablation shows KS is necessary, ECE is not. | Above + add a 4th signal (e.g.\ prediction agreement across domains) and ablate it. |
| 7 | *Why MVTec 3D-AD and not a deployment-relevant dataset?* | MVTec is the canonical naturally paired RGB/3D anomaly benchmark; cited in related work. | Above + Phase D plan to extend to medical / industrial deployment datasets. |

---

## 5. Phase plan (upgrade path from current 5/10 to publishable)

The plan below is structured so that each phase produces a self-contained
upgrade. Phase A is the minimum work to fix what's already broken; Phase B is
the work to make the paper actually publishable at a workshop; Phase C is the
work to lift it from workshop to mid-tier conference.

### Phase A — Headline framing + missing citations (~2 days, cosmetic but reviewer-blocking)

| # | Task | File |
|---|---|---|
| A.1 | Rewrite §I.C "Contributions" so contribution #1 is "we report a negative cross-benchmark result for validation-derived drift gates" | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §I.C |
| A.2 | Update first sentence of abstract to lead with the contrastive finding, not the system description | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) abstract |
| A.3 | Add Aggarwal & Sathe, Kittler 1998, Wachter 2017, Geifman & El-Yaniv 2017 to bibliography and cite each in §III | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §III + bibliography |
| A.4 | Add 1-sentence summary of the contrast finding to the thesis abstract's last sentence | [THESIS_CHAPTER_v1.tex](../THESIS_CHAPTER_v1.tex) abstract |
| A.5 | Add a "Negative Results" section to the thesis that consolidates the five "what did not work" items | [THESIS_CHAPTER_v1.tex](../THESIS_CHAPTER_v1.tex) new §after Failure Analysis |

### Phase B — Make the mechanism story complete (~1–2 weeks, real-experiment work)

| # | Task | File / location |
|---|---|---|
| B.1 | Run the LA benchmark with `--scorer-train-fraction 0.05` (already supported in [prepare_real_fusion_benchmark.py](../../../src/scripts/prepare_real_fusion_benchmark.py)) to defeat saturation | new `experiments/fusion/craf_real_results_hard.json` |
| B.2 | Re-render the clean LA table off the harder benchmark; methods now spread out | regenerate `rga_clean_results.tex` |
| B.3 | Wire the learned gate into the breakthrough experiment runner so it produces a "learned" row in the τ-sweep | [run_breakthrough_experiment.py](../../../src/scripts/run_breakthrough_experiment.py) |
| B.4 | Replace the lightweight MVTec scorer with M3DM-style features (pre-trained ResNet penultimate + depth statistics from PointNet++) | extend [prepare_mvtec3d_fusion_benchmark.py](../../../src/scripts/prepare_mvtec3d_fusion_benchmark.py) |
| B.5 | Re-render all MVTec tables off the upgraded scorer; attention should now be competitive with random forest | regenerate `mvtec3d_*.tex` |
| B.6 | Add a per-domain subset attack model ($\|S\| = 1$ and $\|S\| = D/2$) and report it as the primary adversarial table | extend `_evaluate_adversarial` in [run_breakthrough_experiment.py](../../../src/scripts/run_breakthrough_experiment.py) |
| B.7 | Aggregate `table_5_calibration` across seeds; add ECE/Brier confidence intervals to the calibration table | extend [generate_craf_paper_assets.py](../../../src/scripts/generate_craf_paper_assets.py) |
| B.8 | Promote the failure-cases scatter (`plot_failure_cases`) to a numbered figure in both manuscripts | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §XI / [THESIS](../THESIS_CHAPTER_v1.tex) |

### Phase C — Tighten the scientific narrative (~3–5 days)

| # | Task | File |
|---|---|---|
| C.1 | Rewrite §IX.D ("Mechanism Isolation Protocol") so the cross-benchmark contrast is the *first* subsection and component ablation feeds into it | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §IX.D |
| C.2 | Add a "When should you use the gate?" decision table: rows = drift-detector confidence, columns = pairing type (natural / label-aligned), cells = expected gate effect | new table in §IX.D |
| C.3 | Add a one-paragraph reviewer-Q&A preempt at the end of §XII (Discussion) addressing the five questions in §4 above | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §XII |
| C.4 | Update the thesis's "What did not work" abstract framing into a dedicated chapter section | [THESIS_CHAPTER_v1.tex](../THESIS_CHAPTER_v1.tex) |
| C.5 | Add a runtime-overhead row to the calibration table (existing [rga_runtime_overhead.tex](../tables/rga_runtime_overhead.tex) is unreferenced) | [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) §X |

### Phase D — Stretch (months, optional)

| # | Task | Notes |
|---|---|---|
| D.1 | Test the reliability gate on a *third* naturally paired dataset (CICIDS + auth logs, MIMIC + clinical notes, or another industrial paired anomaly benchmark) | Confirms whether the misfire pattern generalizes |
| D.2 | Complete the corrected theorem stack and validate it with k-of-D corruption, category-mixture shift, and finite-sample switching-certificate experiments | The most realistic path from "competent thesis chapter" to "top-venue submission" |
| D.3 | Open-source the runtime-only `infer_rga` package separate from the research codebase | Industry-facing artifact |

---

## 6. Version target per venue

| Target | Required phase work | Confidence |
|---|---|---|
| arXiv preprint, thesis chapter as-is | Phase A only | High |
| Workshop (NeurIPS-W, ICML-W, IEEE S&P-W) | Phase A + Phase B.1, B.6 | High |
| Mid-tier conference (CIKM, ICDM, IEEE BigData, IEEE TKDE) | Phase A + Phase B + Phase C | Medium-high |
| Top-tier conference (KDD, NeurIPS, ICLR, IEEE TPAMI) | Above + Phase D (theory or third paired benchmark) | Low without supervisor + 6+ months |

---

## 7. What I would do next

If I had a one-week budget I would:

1. **Day 1.** Execute Phase A (~2 hours) — rewrite the contribution list,
   abstract first sentence, and add the four missing citations. Then add the
   "Negative Results" section to the thesis. PDF rebuilds clean.

2. **Days 2–3.** Execute Phase B.1–B.3 — re-run the LA benchmark with
   `--scorer-train-fraction 0.05`, render new `rga_*.tex`, wire the learned
   gate into the runner, add it as a row in the τ-sweep table. The paper now
   has (a) a non-saturated clean table, (b) a learned-gate baseline that
   addresses reviewer Q3 in §4.

3. **Days 4–5.** Execute Phase B.4–B.5 — install `m3dm`-style ResNet
   penultimate features for RGB and PointNet++ features for depth, replace
   the image-statistic scorer in
   [`prepare_mvtec3d_fusion_benchmark.py`](../../../src/scripts/prepare_mvtec3d_fusion_benchmark.py),
   re-run on all eight categories. If attention becomes competitive with
   random forest, the paper's argument shifts: "the gate misfire we previously
   reported was real but at the wrong abstraction level — at higher-quality
   features, the gate's harm shrinks to [X]". That is a much stronger paper.
   If attention stays uncompetitive, the negative-result framing becomes
   even more defensible.

4. **Days 6–7.** Execute Phase C — promote the contrast to the headline,
   add the decision table, write the reviewer-Q&A preempt, run final
   linter sweep, recompile both PDFs.

After that one week, the paper is workshop-submittable with high confidence
and mid-tier-conference-submittable with moderate confidence.
