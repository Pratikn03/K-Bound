# Publication Roadmap — arXiv → Workshop → Mid-Tier Conference

**Target outcomes (in order of execution):**
1. **arXiv preprint, well-received** — 95% achievable, ~2 weeks
2. **Top workshop accept** (NeurIPS-W / ICML-W / ICLR-W) — 70–80% with cleanup + targeted submission, ~6–10 weeks total
3. **Mid-tier conference accept** (CIKM, ICDM, IEEE TKDE, ECML-PKDD, IEEE BigData) — 50–60% with extensions, ~4–6 months total

**Author:** Pratik Niroula (Independent Researcher).
**Starting date:** 2026-05-15.
**Working file:** [PAPER_DRAFT_v1.tex](../PAPER_DRAFT_v1.tex) (16 pp, IEEEtran).

---

## 0. The submission story

One sentence per target — keep these front-of-mind, because every decision below serves them:

- **arXiv**: *"We report a contrastive negative result for validation-derived drift gates in multimodal anomaly fusion: the same KS-drift signal helps under label-aligned coherent attack and hurts on naturally paired MVTec 3D-AD. We isolate the mechanism via component ablation and propose what a deployable drift signal should look like."*
- **Workshop**: above + *"We add a learned-gate baseline and per-domain attack model; the contrast is mechanistically robust to gate-design choices."*
- **Conference**: above + *"We extend the contrast to N paired benchmarks with M3DM-style strong RGB+3D features; the misfire pattern generalizes."*

The story compounds: each tier adds *one specific empirical asset* to the previous tier's claim. Do not write a new paper for each — extend.

---

## Target 1 — arXiv preprint (Weeks 1–2)

**Confidence: 95%.** This is the floor outcome; lock it.

### Gating criteria

A preprint counts as "well-received" if:
- The PDF compiles cleanly with zero `Undefined`/`Error` warnings.
- The README/companion docs match what the paper actually claims.
- A third party can clone the repo and rebuild the PDF in one command.
- The headline numbers in the abstract are reproducible from the on-disk JSON artifacts.
- The known split-discipline issue is either fixed or explicitly disclosed in the threats-to-validity section.

### Week 1 — credibility floor

| Day | Task | Where | Done when |
|---|---|---|---|
| 1 | Fix fusion-runner split discipline so MVTec's official train/val/test boundaries are honored | [run_breakthrough_experiment.py:123-138](../../../src/scripts/run_breakthrough_experiment.py#L123-L138) | The `_split` function uses MVTec's `split` column as a partition key when available, falls back to stratified random otherwise |
| 1 | Re-run [run_breakthrough_experiment.py](../../../src/scripts/run_breakthrough_experiment.py) with the MVTec config | terminal | New `mvtec3d_results.json` lands |
| 2 | Re-run [scripts/rebuild_paper.sh](../../../scripts/rebuild_paper.sh) | terminal | All `elara_*.tex`, `mvtec3d_*.tex`, both PDFs rebuilt |
| 2 | Sanity-check the new numbers — does the negative contrast still hold under disciplined splits? | by reading `mvtec3d_results.json` | Either (a) it holds, document it; (b) it weakens, update prose to honest |
| 3 | Prune dead packages: delete `src/uais_v/` except `models/`; delete `config/`; delete `deploy/api/main.py` | filesystem | `git status` reflects the deletion |
| 4 | Fix companion docs: rewrite root `README.md` to describe ELARA accurately; delete or repurpose `UAISV_Final_Project_Summary.md`, `PHASE_2_RESEARCH_PLAN.md`, `IMPROVEMENTS.md` | filesystem | These four files either describe ELARA accurately or are gone |
| 5 | Commit generated assets to git: `docs/research/tables/elara_*.tex` + `mvtec3d_*.tex` + `docs/research/figures/*.png` | `git add` | Clean clone can compile the PDF |

### Week 2 — preprint polish

| Day | Task |
|---|---|
| 6 | Add reviewer-Q&A preempt paragraph at the end of the Discussion section addressing the five top reviewer questions from [REVIEWER_RATING_AND_PHASE_PLAN.md §4](REVIEWER_RATING_AND_PHASE_PLAN.md) |
| 7 | Add 4 missing citations: Aggarwal & Sathe (anomaly ensembles), Kittler et al. 1998 (classifier combiner theory), Wachter et al. 2017 (counterfactual explanations), Geifman & El-Yaniv 2017 (selective prediction) |
| 8 | Run `latexmk` final sweep; clear remaining `Overfull \hbox` warnings; verify all `\ref` / `\cite` resolve |
| 9 | Create arXiv account, fill in metadata (title, abstract, MSC class), upload `PAPER_DRAFT_v1.tex` + `THESIS_CHAPTER_v1.tex` as a single package |
| 10 | Submit to arXiv (cs.LG primary, cs.AI + stat.ML cross-list) |
| 11–12 | Wait for arXiv moderation; share preprint link on Twitter / LinkedIn / Reddit r/MachineLearning |

### Deliverables

- `arxiv.org/abs/2606.NNNNN` (or whatever month)
- Updated GitHub repo with arXiv badge in README
- Tweet thread summarizing the contrastive finding
- LinkedIn post connecting the work to applied ML / robust ML themes

### What success looks like at this tier

- arXiv paper publicly visible
- 50–200 downloads in the first month
- 1–3 citations within 12 months (typical for an independent preprint in a niche)
- Used as a portfolio piece for industry/PhD applications

---

## Target 2 — Top workshop accept (Weeks 3–10)

**Confidence: 70–80% if you pick the right workshop and execute Phase B.**

### Venue selection (in priority order, with rough deadlines)

| Venue | Workshop | Typical deadline | Conference date | Fit |
|---|---|---|---|---|
| **NeurIPS 2026 Workshops** | "Distribution Shifts" / "Reliable ML" / "ML Safety" | mid-Sept 2026 | Dec 2026 | ★★★★★ Best fit. Negative-result story aligns directly. |
| **NeurIPS 2026 Workshops** | "Workshop on Trustworthy ML" / "ML4H" | mid-Sept 2026 | Dec 2026 | ★★★★ if you frame the cross-benchmark misfire as a trust/safety question |
| **AAAI 2027 Workshops** | "Workshop on AI for Cyber Security" / "Trustworthy AI" | early Nov 2026 | Feb 2027 | ★★★★ — security framing of the all-domain attack story fits |
| **ICLR 2027 Workshops** | "Workshop on Reliable ML" | late Feb 2027 | Apr/May 2027 | ★★★★ — backup if NeurIPS-W misses |
| **ICML 2026 Workshops** | (most have already passed for 2026) | n/a | Jul 2026 | ✗ — already past for 2026 |
| **IEEE S&P 2027 Workshops** | DeepSec / "Workshop on ML for Security" | typically Dec 2026–Jan 2027 | May 2027 | ★★★ — adversarial framing fits |

**Primary target: NeurIPS 2026 Workshop on Distribution Shifts** (or equivalent NeurIPS-W). The contrastive misfire of a validation-derived drift signal is *exactly* the kind of finding that workshop wants. Submission deadline mid-September 2026 gives you ~4 months — comfortable.

**Backup: ICLR 2027 Reliable ML workshop** if NeurIPS-W misses.

### Workshop-tier extensions (Phase B from earlier reviews)

| # | Task | Why it matters for workshop reviewers | Effort |
|---|---|---|---|
| B.1 | Add learned-gate baseline row to τ-sweep table; show it's competitive with the heuristic | Preempts "did you try a learned gate?" reviewer question | 2 days |
| B.2 | Add per-domain subset adversarial attack table ($|S|=1, |S|=D/2$ partial-compromise threats) | Preempts "the all-domain attack is unrealistic" reviewer question | 3 days |
| B.3 | Aggregate calibration metrics (ECE, Brier) across seeds; report CIs | Closes the "no CIs on calibration" reviewer concern | 1 day |
| B.4 | Promote failure-case scatter to a numbered figure | Reviewers love concrete examples | 1 day |
| B.5 | Trim paper to workshop length (typically 4–8 pages, varies by venue) | Workshop page limits are hard | 2 days |
| B.6 | Write workshop-specific cover letter / submission notes | Most workshops have a short "why this venue" field | 1 hour |

Total: ~1.5–2 weeks of focused work after Target 1 lands.

### Weeks 3–6 — workshop preparation

| Week | Focus | Deliverable |
|---|---|---|
| 3 | Execute B.1 + B.3 | Updated τ-sweep table with `learned` row; new `*_calibration_ci.tex` |
| 4 | Execute B.2 (per-domain subset attack) — extend `_evaluate_adversarial` in the runner | New `mvtec3d_adversarial_subset_results.tex` + `elara_adversarial_subset_results.tex` |
| 5 | Execute B.4 + write the failure-case figure caption + section text | Promoted figure; supporting prose |
| 6 | Execute B.5 — trim paper to workshop length; rewrite intro/conclusion to fit | Workshop-version `PAPER_WORKSHOP_v1.tex` |

### Weeks 7–10 — submission and review

| Week | Focus |
|---|---|
| 7 | Pick the target workshop (announce when call for papers opens), align paper to its scope statement |
| 8 | Internal review pass: re-read for "do I sound like I believe my own claims?" |
| 9 | Submit to NeurIPS 2026 workshop (assuming mid-Sept deadline) |
| 10 | Buffer for any last-minute fixes; submit secondary workshop if multi-submission is allowed |

### Decision points

- **If NeurIPS-W rejects:** resubmit to ICLR-W with the reviewer feedback incorporated (typically <2 weeks of revision)
- **If both reject:** evaluate whether the negative-result framing is landing; if not, consider promoting the *positive* finding (label-aligned gain) instead

### What success looks like at this tier

- Workshop accept + non-archival presentation
- Citation in your CV
- Likely 2–5 additional citations over 12 months
- Establishes you as someone working in the multimodal-fusion-robustness space
- Often opens door to mid-tier conference submission via "extended workshop paper" track

---

## Target 3 — Mid-tier conference accept (Months 3–6)

**Confidence: 50–60% if you execute Phase C and get genuinely lucky on the M3DM features.**

### Venue selection

| Venue | Typical deadline | Decision date | Conference date | Fit |
|---|---|---|---|---|
| **ICDM 2026** | early June 2026 | early Aug 2026 | Dec 2026 | ★★★★ — TIME-CRITICAL. If you can hit this you save 6 months. |
| **IEEE BigData 2026** | late Sept 2026 | mid-Oct 2026 | Dec 2026 | ★★★ — fits, but data-centric framing |
| **CIKM 2026** | typically already past | — | Oct/Nov 2026 | ✗ — likely past |
| **ECML-PKDD 2027** | April 2027 | June 2027 | Sept 2027 | ★★★★ — European venue, fits multimodal anomaly well |
| **WSDM 2027** | August 2026 | Oct 2026 | Feb 2027 | ★★★ — search/data mining framing |
| **CIKM 2027** | May 2027 | July 2027 | Oct/Nov 2027 | ★★★★ — best fit, give yourself the year to extend |
| **IEEE TKDE** | rolling | rolling | — journal | ★★★★★ — best fit for a fully extended journal version, ~6 months review cycle |

**Primary target: ICDM 2026** (if you can hit early June deadline — aggressive)
**Realistic target: CIKM 2027 or ECML-PKDD 2027** with a journal version submitted to IEEE TKDE later
**Stretch target: IEEE TKDE journal directly** (rolling submission, can be done anytime)

### Conference-tier extensions (Phase C from earlier reviews)

| # | Task | Why it matters | Effort |
|---|---|---|---|
| C.1 | Replace lightweight MVTec image-statistic scorer with M3DM-style ResNet-50 RGB + depth-statistics PointNet++ features | The single biggest empirical lift available — if attention becomes competitive with random forest at higher feature quality, the entire contrast story strengthens | 2–3 weeks |
| C.2 | Add a *third* paired benchmark — CICIDS-2017 + auth logs, or MIMIC-IV + clinical notes, or another industrial paired-anomaly dataset | Converts the contrast from "anecdote on 2 benchmarks" to "pattern on 3 benchmarks" — necessary for a serious conference submission | 3–4 weeks |
| C.3 | Add the "When should you use the gate?" decision table: rows = drift signal confidence, columns = pairing type, cells = expected effect | Makes the contribution prescriptive, not just descriptive | 2 days |
| C.4 | Add bootstrap CIs on every robustness cell (was Phase B for workshop; for conference, must be present) | Reviewer expectation at conference tier | 1 day |
| C.5 | Cross-modal attention runtime overhead measurements vs random forest fusion | Standard "is this practical?" reviewer question | 2 days |
| C.6 | Write rebuttal-anticipation appendix that responds to expected reviewer questions in detail | Sometimes turns borderline accepts into accepts | 1 week |

Total Phase C effort: **~8 weeks** of solo work, parallel-able with Phase B if you have bandwidth.

### Month 3 — empirical depth

- Week 11–12: M3DM-style RGB+depth features (C.1) — biggest single-task block
- Week 13: Re-run all MVTec experiments with new features; regenerate assets

### Month 4 — third benchmark

- Week 14–17: Pick and ingest the third paired dataset; run the same fusion + mechanism-isolation pipeline; render new asset tables `mvtec3d_*` + `cicids_*` (or whatever)

### Month 5 — paper extension

- Week 18: Add C.3 decision table, C.5 runtime overhead
- Week 19: Write C.6 rebuttal-anticipation appendix
- Week 20–21: Final paper extension; target ~10–12 pages

### Month 6 — submission

- Week 22–24: Submit to selected mid-tier conference; buffer for revision

### Decision points

- **If M3DM features don't lift attention performance:** the contrast story actually *strengthens* — "even with strong features, validation-derived drift gates misfire on naturally paired data." Still publishable.
- **If a third benchmark *fixes* the misfire:** the paper's headline flips to "misfire pattern is dataset-specific; we identify when it occurs." Also publishable, different framing.
- **If the third benchmark *confirms* the misfire:** this is the strongest outcome — a real generalizable methodological observation, possibly worth submitting to a top venue instead of mid-tier.

### What success looks like at this tier

- Mid-tier conference accept (CIKM, ICDM, IEEE TKDE, ECML-PKDD)
- ~10–50 citations over 24 months
- Conference talk, proceedings entry, archival publication
- First-author conference paper on CV — strongly differentiates you for PhD admits / industry research roles
- Often promotes to journal extension (IEEE TKDE) once accepted

---

## Cross-cutting practicalities

### Weekly time budget assumed
- ~15–20 hours/week sustained solo effort = the schedule above
- ~30+ hours/week = compress by 30–40%
- <10 hours/week = expand by 2× (still reachable, just slower)

### Backup workshops (if NeurIPS-W misses)
1. ICLR 2027 Reliable ML / Trustworthy ML — Feb 2027 deadline
2. AAAI 2027 Workshop on AI for Cyber Security — Nov 2026
3. IEEE S&P 2027 DeepSec workshop — Jan 2027
4. UAI 2027 workshops — typically May 2027 deadline

### Backup conferences (if ICDM misses)
1. CIKM 2027 — May 2027 deadline
2. ECML-PKDD 2027 — April 2027 deadline
3. IEEE TKDE — rolling, slower but accepts journal-format work
4. IEEE BigData 2027 — September 2027 deadline

### Submission hygiene checklist

Before every submission:
- [ ] All `\ref` / `\cite` resolve
- [ ] All bibliography entries cited
- [ ] All figures and tables referenced from prose
- [ ] Reproducibility statement in paper or supplementary
- [ ] arXiv version up-to-date
- [ ] Anonymous version available if venue requires
- [ ] Code repository pinned at the commit that generated the reported numbers
- [ ] Reviewer-Q&A preempt in Discussion section

### Citations to chase

If you can get even one of the following authors to cite or reference your preprint, your visibility 10×s:
- Authors of the test-time-adaptation foundation papers (Wang et al. on Tent, Sun et al. on TTT)
- Authors of MVTec 3D-AD (Bergmann et al.)
- Authors of M3DM or crossmodal feature mapping (Wang et al. 2023, Costanzino et al. 2023)
- Authors of any "what doesn't generalize in TTA" papers (the negative-result framing aligns with their interests)

Tactic: tweet your preprint at them, present a poster at a small workshop they're attending, or email a polite "I think my work answers your question about X" message.

---

## What success looks like 12 months from today

If you execute the plan above:

| Month | Milestone |
|---|---|
| Month 1 | arXiv preprint live, ~100 downloads |
| Month 3 | Workshop submission to NeurIPS 2026 in progress |
| Month 5 | Workshop accept notification (Sept-Oct 2026) |
| Month 6 | Workshop poster presented at NeurIPS 2026 |
| Month 8 | Conference submission to CIKM 2027 / ECML-PKDD 2027 in progress |
| Month 10 | Conference decision |
| Month 12 | Conference paper accepted (or revising for a different venue) |

12-month deliverable: **1 arXiv preprint + 1 workshop paper + 1 mid-tier conference paper in flight or accepted.** That puts you firmly at the publication output of a strong 2nd-year PhD student, with industry/PhD/founder options all open.

---

## What I would do if I were you (today, this week)

1. **Block 4 days this week** for Target 1 cleanup (split discipline + dead-code prune + companion-doc rewrite).
2. **Pick the target workshop now** — decide between NeurIPS 2026, AAAI 2027, ICLR 2027 — even though you can't submit yet, knowing the audience shapes the writing.
3. **Set a calendar reminder for the workshop's CFP announcement** (usually 3–4 months before deadline).
4. **Start a public-facing landing page** for the work — single-page site, abstract, link to arXiv (once live), link to GitHub, your contact. This is the page reviewers, hiring managers, and PhD admits will find. 1 hour of work; high leverage.
5. **Begin sketching the M3DM-features extension** in parallel — it's the longest-pole item for the conference paper and you want to start the experiment early because the data work always takes longer than you think.

If you complete steps 1–4 in the next 7 days, you're on track for the arXiv preprint by end of May 2026 and a workshop submission by mid-September. That puts you on a credible publication trajectory by year-end 2026.
