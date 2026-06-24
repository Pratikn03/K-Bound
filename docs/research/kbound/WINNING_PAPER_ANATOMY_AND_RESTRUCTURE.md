# Anatomy of a winning main-conference paper — and a one-theorem restructure for `kbound_short`

*Prepared June 2026. Part 1 is how award-winning ML papers of the last decade are actually
written (grounded in award criteria, paper-writing craft, and the canonical impossibility-paper
model). Part 2 maps it directly onto your short paper. The honest calibration is at the end, and it
is the same line held throughout: structure and a clean theorem **amplify** a result into an award —
they do not manufacture one.*

---

## Part 1 — How the papers that win are written

### 1. They serve ONE central idea, said out loud

The single most consistent property of a strong paper is that the reader can state its main idea in
one sentence — because the authors stated it for them. Peyton Jones's classic advice is blunt: a
paper exists to "infect the mind of your reader with your central idea," so identify that one idea
and lead with it. Everything else (theory, method, experiments) is in service of that one sentence.
Award papers do not have five contributions of equal weight; they have **one** that the title, the
abstract's third sentence, and Figure 1 all point at.

Practical consequence: **lead with contributions.** Write the contributions list first — it drives
the whole paper — and make it specific and checkable ("we prove X is identifiable iff Y," not "we
study the problem of Z").

### 2. They win on a new PROBLEM FRAMING, not raw correctness

Conference award and reviewer guidance is remarkably consistent: papers are judged on novelty,
technical quality, **potential impact**, reproducibility, and **clarity** — and reviewers are
explicitly told to reward high-level impact, out-of-the-box ideas, and *novel problems* over
technical correctness alone. Correctness is the price of admission; what wins is a question the field
had not framed that way before. K-Bound already has this: "should the model adapt **at all**?" is a
genuinely fresh framing of test-time adaptation as a *decision* problem. That framing is your single
biggest asset and it should be impossible to miss in the first 30 seconds.

### 3. The one-theorem spine — model it on "Inherent Trade-Offs"

The cleanest model for K-Bound is Kleinberg, Mullainathan & Raghavan, *Inherent Trade-Offs in the
Fair Determination of Risk Scores* (2016/ITCS 2017). It is one of the most influential papers of the
decade, and structurally it is almost minimal: **one impossibility theorem** — you cannot
simultaneously satisfy three reasonable fairness criteria except in narrow, characterized special
cases — plus the exact statement of those escapable cases. It reorganized a whole subfield not
because the proof was heavy (it isn't) but because the *framing* was sharp and the result was a
single quotable statement.

That is the template: **a single theorem that is both a wall and a door.** The wall (you cannot do
X) is the hook; the door (here is exactly when you can, and a method that does) is the contribution.

### 4. The structure (Peyton Jones template)

- **Abstract — 4 sentences:** (1) the problem, (2) why it is hard / unsolved, (3) your one idea, (4)
  the result that follows. No background paragraph.
- **Introduction — ~1 page** ending in an explicit, numbered contributions list and the teaser
  figure.
- **The problem — ~1 page**, made concrete with the running example.
- **The idea — ~2 pages**: the one theorem, stated precisely, with the proof *idea* before the proof.
- **The details — the proof + the method** as the constructive payoff.
- **Related work — at the END**, framed as deltas ("unlike ATC, which predicts accuracy, we certify
  the benefit *sign*").
- **Limitations & honest scope**, then conclusion.

### 5. The honesty the best theory papers share (and why it helps them win)

From the honest-paper craft, two rules matter enormously for a paper shaped like yours:

- **The impossibility is the hook; the constructive result is the centerpiece.** An impossibility /
  two-world argument is usually the *easy* half and is often already half-known. Do not let it
  masquerade as the contribution. The hard, valuable half is the **exact frontier** (when *can* you
  decide) and the **certificate** (a method that provably does, with error control). Put the page
  count and the proof effort there.
- **Negatives are credibility, not weakness.** The papers reviewers trust state plainly what is
  synthetic, what is assumed, and what is *not* claimed. Your kept-in nulls (iWildCam tie under
  resampling, the withdrawn Camelyon17 win) are an asset — they make the positive claims believable.

### 6. Honest calibration (the impact ladder)

- *Incremental* — a tweak on one benchmark.
- *Solid* — a method/result others cite and build on; **most accepted conference papers live here.**
- *Field-shaping* — a framing the field reorganizes around (the verdict the field returns *years
  later*; "Inherent Trade-Offs" is here, but it earned it post-hoc).
- *Foundational* — textbook.

No single paper is "field-shaping" *at submission* — that label is not the author's to assign, and
claiming it reads as naïveté to reviewers. What you *can* do is aim the **paper** at "solid →
memorable" (sharp framing + one clean theorem + honest evidence) and aim the **program** (a sequence
of papers sharing the K-Bound spine) at the legacy.

---

## Part 2 — The one-theorem restructure of `kbound_short`

### Choose the spine: the benefit-sign frontier

K-Bound has several theorems (impossibility `thm:imp`, the frontier `thm:frontier`, the certificate
`thm:cert`, the one-bit dichotomy, and the new τ=1 capacity). For a one-theorem main-conference
paper, the spine should be the **exact benefit-sign frontier**:

> *Adaptation benefit is label-free identifiable **if and only if** an observable margin exceeds the
> calibration-drift budget on the disagreement region.*

Why this one, and not the others:

- It is a **single iff that already contains the other two.** The "only if" direction **is** the
  impossibility (two target worlds, identical evidence, opposite benefit — Le Cam two-point). The
  "if" direction **is** what the adapt/freeze/abstain certificate operationalizes with α-level error
  control. One statement, both the wall and the door — exactly the "Inherent Trade-Offs" shape.
- It is **quotable** and ties directly to the fresh framing ("should it adapt at all?").
- The **impossibility alone is too easy to headline** (it's the hook, not the centerpiece — see 1.5);
  the **certificate alone needs the frontier** to justify its threshold; and the **τ=1 capacity** is,
  by your own honest scope note, a sharpened *instance*, not a new pillar — perfect as a one-paragraph
  "graded refinement," wrong as the spine.

### Section-by-section (target ≈ 9 pages + refs)

1. **Title + abstract (4 sentences).** Problem: TTA can silently hurt and you can't see it without
   labels. Hardness: two worlds can look identical yet need opposite actions. Idea: certify the
   *benefit sign*, not accuracy. Result: an exact frontier + an adapt/freeze/abstain certificate with
   false-adapt ≤ α, validated on the seven-dataset panel.
2. **Introduction (1 page)** → numbered contributions: (i) reframing TTA as adapt/freeze/abstain;
   (ii) the exact frontier (the one theorem); (iii) the certificate with α-control; (iv) the
   empirical decision results. End on the Figure-1 decision teaser (you already have it).
3. **Problem setup + the single theorem.** State the frontier precisely; give the **Le Cam two-point
   proof idea in three sentences** before the formal proof. This is the centerpiece — give it room.
4. **The certificate as the constructive corollary.** The method *is* the "if" direction made
   finite-sample; this is where adapt/freeze/abstain and the α guarantee live.
5. **Experiments — one decisive story.** Foreground a single table that makes the point: the
   **mixed-stream "beats both fixed policies by 13–24×, both CIs exclude zero"** result (currently
   absent from the short — put it back, compressed), plus the three natural-shift headlines. Resist
   six parallel subsections; one table that proves the decision works beats five that sprawl.
6. **Limitations & honest scope (compressed but present).** One paragraph: synthetic-corruption
   caveats, the one-sided datasets, the resampling tie. Keep it — it is why the wins are believed.
7. **Related work at the end**, as deltas against ATC / agreement-on-the-line / AETTA / e-process
   monitoring.
8. **Appendix:** the τ=1 capacity as a one-paragraph "graded refinement" pointer (not the 9-page
   development — that stays in the full `kbound`).

### What changes vs. the current short

The current `kbound_short` is a faithful *miniature* of the full paper — six experiment subsections,
theory folded into setup, no single spine. The restructure is not "cut more"; it is "**re-center**":
pick the frontier as the visible spine, move the mixed-stream win into the body as the decisive
result, state one theorem prominently with its proof idea, and let everything else orbit it.

### Honest bottom line

This restructure will make `kbound_short` a **clean, memorable, "solid"-tier paper with
award-plausible framing** — which is a real and worthwhile target, and exceptional for an
undergraduate solo author. It does **not**, by itself, make it an award winner: that still needs the
one ingredient that isn't editorial — a **decisive, beats-SOTA-at-scale result** (ImageNet-C / ViT,
multi-seed, CI-backed), which remains blocked on a real GPU. The structure is the amplifier; the
scaled result is the signal. Do the restructure because it makes the paper genuinely better and
submission-ready — and run the experiment because that is the only thing that moves the tier.

---

## Sources

- Simon Peyton Jones, *How to Write a Great Research Paper* — https://simon.peytonjones.org/great-research-paper/
- Microsoft Research, *How to write a great research paper* (slides) — https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/
- Kleinberg, Mullainathan & Raghavan, *Inherent Trade-Offs in the Fair Determination of Risk Scores* (2016) — https://arxiv.org/abs/1609.05807
- ICML 2020 Reviewer Guidelines (award/review criteria) — https://icml.cc/Conferences/2020/ReviewerGuidelines
- NeurIPS 2024 Best Paper Awards announcement — https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/
- Curated best-paper list, top venues 2022–2026 — https://github.com/FeijiangHan/Top-Conference-Best-Papers
