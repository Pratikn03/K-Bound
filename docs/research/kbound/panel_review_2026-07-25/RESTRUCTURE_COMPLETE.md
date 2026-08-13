# RESTRUCTURE COMPLETE — build, adversarial verification, and verdict

Written by the build/verify slice, running last. Every number below was recomputed in this
container from the named artifact, or read out of the named build log. Nothing is quoted from the
rewriters' reports without independent re-derivation.

---

## 1. Build status

All three drivers build from a clean `.aux`, four `pdflatex` passes each, **zero errors**.

| driver | output | pages | `!` errors | undefined `\ref` | undefined `\cite` | multiply-defined | overfull `\hbox` |
|---|---|---:|---:|---:|---:|---:|---:|
| `kbound_tmlr.tex` | `kbound_tmlr.pdf` | **87** | **0** | **0** | **0** | **0** | 15 |
| `kbound_short.tex` (IEEEtran) | `kbound_short.pdf` | **52** | **0** | **0** | **0** | **0** | 49 |
| `kbound.tex` (long) | `kbound.pdf` | **49** | **0** | **0** | **0** | **0** | 12 |

Absolute paths:

- `/home/claude/kb/docs/research/kbound/kbound_tmlr.pdf` — 87 pp (the TMLR submission candidate)
- `/home/claude/kb/docs/research/kbound/kbound_short.pdf` — 52 pp (IEEEtran, formatting check only)
- `/home/claude/kb/docs/research/kbound/kbound.pdf` — 49 pp (long companion)

`tmlr.sty` is still absent; the TMLR driver builds under its documented `article` shim, as designed.
IEEEtran and lmodern are installed and the two-column build is real.

**The error count went from 6 / 6 / 16 to 0 / 0 / 0.** Every one of those errors was a missing
figure asset; see §2. Overfull boxes are typographic warnings, not errors, and the long
manuscript's 12 match the pre-restructure baseline exactly.

### 1.1 The six missing figures — fixed without inventing data

Twenty-two `\includegraphics` calls across the compiled tree pointed at PNGs that are not in this
repository snapshot: six in the shared body (`fig_decision_flow`, `fig_frontier_schematic`,
`fig_certificate`, `fig_decisive_decisions_cifar10c`, `fig_decisive_pareto_cifar10c`,
`fig_natural_forest`) and ten more reached only by `kbound.tex`. pdflatex was typesetting a black
rectangle containing the raw path — a broken image box — in all three PDFs.

Three of the six are schematics and three are data figures. **I generated neither.** Drawing a
schematic I invented is not the figure the caption describes, and plotting the data figures from
the artifacts would have produced plots I cannot check against the originals. Both are fabrication
risks that buy a picture.

Instead I added `/home/claude/kb/docs/research/kbound/paper/figure_fallback.tex`, which defines

```latex
\newcommand{\kbgraphics}[2][width=\linewidth]{%
  \IfFileExists{#2}{\includegraphics[#1]{#2}}{\kbplaceholder{#2}}}
```

and rewrote all 22 call sites to `\kbgraphics`. When the asset exists it is used, byte-identically
to before. When it does not, a framed box is typeset reading **"FIGURE ASSET NOT PRESENT IN THIS
BUILD"**, the filename, and: *"This placeholder is not a figure and reports no result... every
quantity it would display is given numerically in the text and tables."* Restoring a PNG to
`figures/` reinstates the real figure with no source edit and no flag.

Rendered and inspected: the boxes are unmistakable for figures and no broken image box survives.
Placeholder counts: 6 in `kbound_short.pdf` and `kbound_tmlr.pdf`, 16 in `kbound.pdf`.

---

## 2. Verification: no fabricated theorem survives

**This check passes.** Method: enumerate every `theorem`/`lemma`/`proposition`/`corollary`/
`conjecture` environment in the resolved `\input` tree of each driver, resolve each label to its
printed number and page via the `.aux`, and cross-check against `SUBMISSION_LEDGER.md` §2 (type key
`[G]` = theorem-level guarantee) and the theory agents' status tables in
`THEORY_BETA_ESTIMABLE.md` §2 / `THEORY_BETA_IMPOSSIBLE.md`.

- 36 theorem-like environments exist in the short-build source tree; **30 compile**.
- The six that do not are exactly the ones the ledger records as excluded: `lem:gate`,
  `prop:lecam-finite`, `prop:cert-sample`, `thm:conj1-dichotomy`, `thm:ev-rate`, `conj:gen` (all
  inside the `\iffalse` block of `theory_appendix_ext.tex`), plus `prop:beatsboth-asym`, recorded
  as cut. All confirmed **absent from the `.aux`**, so none is silently shipped.
- Every compiled result is ledger-status `[G]`. **The single `CONJECTURE`-status result, B9 /
  `epi:conj-open`, is a `conjecture` environment in every build** — Conjecture 1 (p. 29) in
  `kbound_tmlr.pdf`, Conjecture 2 (p. 19) in `kbound.pdf` — followed by "This is stated as a
  conjecture, not a theorem: we neither exhibit such an assumption nor prove that none exists."
  (I had to *add* that sentence to the long manuscript; see §5, defect D3.)

I additionally verified the two load-bearing new proofs by hand rather than trusting the label:

- `cor:beta-is-beta` (Γ_z(C_β) = β). The achievable-drift set is
  `G = [-β,β] ∩ [-1/2-M, 1/2-M]`, so `Γ_z = max{min(β,1/2-M), min(β,1/2+M)} = min(β,1/2+|M|) = β`
  since `β ≤ 1/2`. Part (ii): `sign(M+γ)` is constant and nonzero on `G` iff `0 ∉ [M+inf G, M+sup G]`,
  which reduces to `|M| > β`, and fails for every `β ≥ 0` at `M = 0`. **Correct as printed.**
- `thm:anchor` (labeled data at a non-deployment anchor moves the minimax budget by exactly zero).
  The proof is four lines and is valid *given* its stated product/no-coupling hypothesis, which is
  stated in the theorem rather than smuggled: `lem:fibre` alters only the target label kernel on
  `D`, so `Law(W)` — which contains `S_cal` — is constant on the fibre. **Correct, and honestly
  hypothesised.**

**Proof-pointer audit.** The short/TMLR appendix states three theorems without proof
(`thm:short-audC/DE/G`) and says so in its own text, pointing at the long version. I verified that
claim rather than accepting it: `kbound.pdf` Appendix G, pp. 46–48, does carry `thm:aud-C` (Thm 20),
`thm:aud-DE` (Thm 21) and `thm:aud-G` (Thm 24) with proofs. **The cross-manuscript pointer is true.**

The long manuscript had two *false* proof pointers, now fixed — defect D4 below.

---

## 3. Verification: numbers

I re-derived the load-bearing numbers from the raw artifacts rather than checking them against a
summary. Everything below reproduced **exactly**.

### 3.1 The LOCO stress result — recomputed from `out_cifar_loco_tent_eata.json`, all ten runs

| quantity | shipped | leave-one-corruption-out | manuscript says |
|---|---|---|---|
| ε | 0.0152–0.0219 | 0.0926–0.1122 (**4.28×–6.93×**) | 4.3×–6.9× ✓ |
| commitment rate | 0.509–0.600 | 0.398–0.417 | 0.51–0.60 → 0.40–0.42 ✓ |
| KGA regret | 0.00101–0.00185 | 0.00550–0.00997 (**3.43×–7.61×**) | 3.4×–7.6× ✓ |
| FA_u | 0.0000 (10/10) | **0.0000 (10/10)** | zero in all ten ✓ |
| freeze decisions | 22–76 | **0** | freeze branch empties ✓ |
| beats always-adapt | 10/10 | **5/10 — all five Tent, none of the five EATA (2.01×–2.94× losses)** | ✓ |

The rewriter's new negative is real and I confirm every digit of it. Reporting "FA_u = 0 in 10/10"
alone would have been a selective read.

### 3.2 Abstract numbers — all verified to source

- `66.8%` of `6,480` cells = 4330/6480 = 0.66821 (`decision_value_results.json#headline.cifar10c`) ✓
- regret `5.00×` below better fixed policy = `regret_ratio_bestfixed_over_kga` 5.000259 ✓
- `3.1×` below hindsight-tuned heuristic = 0.0047155/0.0014960 = 3.152 ✓
- `24.5×` effect ratio = `absDelta_ratio_commit_over_abstain` 24.4816 ✓
- ImageNet-R `1.76×` worse = 1/0.567643 = 1.7617 ✓
- `1,113` / `1,244` ADAPT and Clopper–Pearson upper `0.0027` / `0.0024`
  (`NUMBERS_PACK.json#item5.promoted_row_accounting`: 0.0026880, 0.0024052) ✓
- β-sweep: `6,885` cells (6480 + 405), `1.4×`–`50×`, `24–73%`, `0.4–16.6%`, `405` cells, 5 of 10
  configurations at zero commitments (`BETA_SWEEP_FINDINGS.md` §0) ✓
- Episode: control `0.905`, LOCO `0.626`, LOSEV `0.454`, weighted conformal `1.000` at yield
  `0.000`, first feasible at `K = 9`, `470` splits at median `0.34%` with median coverage
  difference exactly zero ✓

### 3.3 The two ImageNet-C coverage ranges are not a contradiction — I checked

Both manuscripts print `0.896–0.985` in one place and `0.904–0.985` in another. This looked like a
mismatch. It is not, and the text names the source at both sites:

- `0.904–0.985` is the five non-genuine rows of `adversarial_ablations_results.json` at its refit
  `β̂ = 0.0293` (values 0.9556, 0.9556, **0.9037**, **0.9852**, 0.9556).
- `0.896–0.985` is Table 4's column at each configuration's own primary `β̂`, from
  `beta_sweep_results.json` (min **0.8963** at `loco|M_gbm`, max **0.9852** at `srclike|M_doc`).
- The one genuine measurement is `srclike|M_gbm`: `0.4704` at `β̂ = 0.02930` in the re-run and
  `0.5111` at the primary `β̂ = 0.030242`. It is also the only ImageNet-C configuration with
  `dev_and_target_disjoint = True`. **Both numbers, both `β̂`s, both correct.**

### 3.4 Regressions checked and clear

- **Retracted uniform-no-harm claim: still dead.** No "uniformly no-harm" survives in any PDF. The
  intro states no-harm as "a claim about the one-sided locked tracks, not a universal property of
  the panel", and the PACS / ImageNet-R / CIFAR-10.1 losses are in the introduction and abstract.
- **One radius rule: intact.** "This is the only radius rule used anywhere in the paper", with the
  interpolated-vs-exact-rank discrepancy (0.0289 vs 0.0122 on ImageNet-C SAR) disclosed.
- **The documented 0.90 null is never read as evidence.** Audited every coverage figure near 0.90.
  Three sites carry it and all three label it, including Table 6's caption: "rows at or near 0.900
  are evidence the estimator is not broken, not evidence that it works." The ImageNet-C
  `0.896–0.985` band is explicitly said to "sit at or above the null and carry almost no information".

---

## 4. Verification: does the spine hold?

Yes, and it is not asserted — each link is discharged by a numbered result in the same build.

| link | claim | discharged by | status |
|---|---|---|---|
| 1 | a label-free adapt/freeze decision exists only relative to a declared β | Lemma 1 → Thm 1 → Cor 1 → Thm 2 | proved, main body |
| 2 | β cannot be supplied label-free | Thm 3 (`thm:beta-minimax`), Cor 3 (`Γ_z(C_β)=β`) | proved, main body |
| 3 | labels at the wrong domain do not rescue it | Thm 4 (`thm:anchor`) | proved, main body |
| 4 | so declaring β from dev data must fail — and it does | §5, 6,885 cells | measured, negative |
| 5 | what escapes is a *population* of deployments, priced | Thms 7–9, §6.8 | proved + measured failure |
| 6 | what remains is the finite-sample certificate | Thm 11, §9.7 | proved + priced |

The chain runs in source order in both manuscripts (TMLR §§4→5→6→7; long §§VI→VII→VIII→IX–X), and
the introduction states it as "a single statement in five links" with each link naming its theorem.

**The one seam a referee will attack is signposted from both sides.** §4 requires validity
*uniformly over the fibre*; §6 substitutes an average over the episode law. §4.6 ends with a block
titled "The one hypothesis that can be attacked, and where we attack it" that names the
substitution and forward-references `epi:prop-escape`; `epi:prop-escape` carries its proof in the
main body and concedes the episode budget is *not* fibre-uniformly valid. This is the honest
handling. It is not hidden.

**Link 4 is the strongest thing in the paper** and it is the one I most wanted to break: the theory
predicts the experiment's failure, and the experiment independently rules out the boring
explanation (F6: the conformal budget and the plug-in quantile differ by a median 0.34% with median
coverage difference exactly zero, so the β-sweep measured E2 failing, not a quantile estimator
failing). That is a genuine closed loop.

---

## 5. Defects I found, and what I did about each

| # | defect | severity | action |
|---|---|---|---|
| D1 | 22 broken image boxes across three PDFs (6 / 6 / 16 hard errors) | **ship-blocking** | **Fixed.** `paper/figure_fallback.tex` + 22 call-site rewrites. 0 errors now. |
| D2 | **The long manuscript's abstract had dropped four disclosures the shared abstract carries.** It claimed no-harm on the four one-sided natural tracks at FA_u = 0 with *no* weak-zeros caveat; it omitted the frontier retraction, the episode-marginal caveat, the "half the severity splits admit no budget" clause, and the entire LOCO EATA negative. | **high — this is an overclaim** | **Fixed additively.** Restored all four into `kbound.tex`'s abstract without deleting the long author's POEM/AETTA and PACS content. Verified in the rebuilt PDF. |
| D3 | `epi:conj-open` in the long manuscript was a `conjecture` environment but lacked the explicit "we neither exhibit such an assumption nor prove that none exists" sentence the short build carries | medium | **Fixed** in `manuscript/theory_spine/theory_beta_estimable.tex`. |
| D4 | **Two theorems in `kbound.pdf`'s main body carry only a "Proof sketch", and one pointed its "full proofs" at a supplementary appendix that does not exist in this build and at `val_conj1_*.py` — i.e. at Python validators.** A validator is not a proof. | **high — a false proof pointer** | **Fixed.** `paper/sections/main_theory_5.tex`: both `thm:conj1-dichotomy` and `thm:ev-rate` now carry an explicit *Proof status* note stating no full proof appears in this manuscript, that the `.py` files are numerical validations and "a validator is not a proof", and that nothing in the impossibility spine or the experiments depends on either. |
| D5 | `tab:experiment-status` shipped two literal `\textsc{tbd}` cells, reading as unfinished draft on rows whose own Key-outcome column says "optional" | low | **Fixed.** Status → "not run", Table → "---". Zero `tbd` in any PDF. |
| D6 | `paper/sections/knowability_capacity.tex:53` says "ε is the 1-D analogue of the calibration-drift budget β" — the exact ε/β conflation the restructure exists to kill | **none shipped** | **Verified dead.** The file is `\input` by no driver and appears in no build log. Reported, not edited: it is a latent hazard if someone re-adds it. |
| D7 | the short rewriter flagged an unresolved ε≠β conflation in `kbound.tex`'s threshold paragraph | **already fixed** | The long rewriter had fixed it in parallel; the short rewriter checked a stale state. The paragraph now reads "it is *not* an estimate of the drift budget β, and no statement in this paper licenses reading it as one," citing `thm:beta-minimax`. All four sites agree. |

Residual, not fixed, reported: `kbound.tex` **duplicates** the abstract rather than `\input`ing
`kbound_abstract.tex` as the other two drivers do. That duplication is what allowed D2 to happen.
I closed the content gap but not the structural cause, because collapsing the two abstracts would
delete the long author's POEM/AETTA and PACS-2.45× sentences, which is a content decision, not a
build fix. **Recommend making `kbound.tex` share `kbound_abstract.tex` before submission.**

---

## 6. Structure: old vs new

| | before (pre-restructure baseline) | after |
|---|---|---|
| `kbound_tmlr.pdf` | 74 pp; theory = matched-evidence + frontier; audit vacuity an appendix aside | **87 pp**; five-act spine |
| `kbound.pdf` | 63 pp | **49 pp** |
| spine | three unrelated findings: an impossibility, a negative β-sweep, a certificate | one statement in five links, each entailing the next |
| `thm:short-audA` | appendix theorem | **main body**, as `cor:audA` of the new `thm:beta-minimax`, triple-labelled |
| β-sweep | §7.2, buried inside Results | **§5, ahead of the method**, framed as the measured prediction of `thm:anchor` |
| episode result | absent | **§6**, new act: what escapes, and what it costs |
| certificate | third pillar of Theory | **§7**, the object that survives §§4–6 |

TMLR outline as built: §1 Intro (p2) · §2 Related (p6) · §3 Setup (p8) · **§4 The drift budget
cannot be supplied label-free (p11)** · **§5 The measured consequence: a β sweep, negative (p18)** ·
**§6 Where a budget can come from, what it costs, and why it does not help here (p24)** · **§7 What
remains: the finite-sample certificate and KGA (p30)** · §8 Setup · §9 Results · §10 Ablations ·
§11 Limitations · §12 Reproducibility · §13 Conclusion · Appendices A–P.

Long manuscript: §V one lemma, three radii · **§VI the exact minimax label-free budget** ·
**§VII does the frontier operationalize? (negative)** · **§VIII where a budget can come from** ·
§IX Method · §X Experiments · §§XI–XIV.

**The main body grew, and I confirm the rewriter's honesty about it.** The plan predicted ~38 pp
and it is 54; promoted theory costs 13 single-column pages, not the 7.2 budgeted. `kbound_short.tex`
at 52 IEEEtran pages is not a submission target and should be kept as a formatting check only.

---

## 7. The new abstract, verbatim

> Test-time adaptation can silently degrade a deployed model on unlabeled target data, and with no
> target labels the system cannot tell a helpful shift from a harmful one. Deciding *whether* to
> adapt is possible only relative to a declared bound on calibration drift—a drift budget β—so the
> question that decides the method is where that budget comes from. This paper answers it, and the
> answer is that it cannot come from the deployment.
>
> We prove that for every declared class of target laws the exact minimax label-free budget is the
> fibre radius Γ_z, attained by a constant audit that reads none of its data; for the deployment
> class C_β the fibre radius equals β exactly, so the best possible label-free audit returns its own
> input, and the number it returns is simultaneously the radius of the band on which a strict
> adapt-or-freeze commitment is unsound. Two consequences follow. Any δ-valid label-free audited
> rule commits with probability at most δ over the unrestricted class—decision yield is bounded by
> the error budget, at every batch size and for every evidence map. And a *fully labeled*
> calibration sample from any other domain leaves the minimax budget unchanged, absent a separately
> declared coupling. One lemma—that the label kernel on the disagreement region is unconstrained by
> every label-free observable—carries the matched-evidence impossibility, the abstention band, and
> the budget impossibility as readings of the same one-parameter family at three radii.
>
> We then measure the prediction. Declaring β as a high quantile of realized drift on source-like
> development cells, exactly as this paper's own method section prescribes, fails on 6,885 real
> evaluation cells: on CIFAR-10-C the declared budget is 1.4× to 50× smaller than soundness
> requires, 24–73% of deployment cells fall outside the declared class, and committed actions are
> wrong at 0.4–16.6% where the frontier promises 0%; on ImageNet-C a large enough budget is
> derivable and returns zero commitments on all 405 cells in five of ten configurations. This is not
> a quantile-estimation failure: across 470 splits the conformal order statistic that replaces the
> plug-in quantile differs from it by a median 0.34%, with a median coverage difference of exactly
> zero. We withdraw the operational reading of the frontier; the proofs are untouched.
>
> What escapes the impossibility is a population of deployments rather than a deployment. Under
> retrospective outcome logging and exchangeability with K logged episodes, β is identified as an
> episode quantile and estimable by a conformal order statistic, with the episode requirement
> bracketed to a factor of e: 1/(eα)−1 ≤ K* ≤ 1/α−1, between 3 and 9 logged deployments at α=0.10,
> and first feasible on real data at exactly K=9. The escape is average-over-episodes instead of
> supremum-over-the-fibre, paid for with historical labels, and neither ingredient suffices alone;
> the guarantee it buys is marginal across deployments, not conditional on the one in hand. The
> assumption that buys it is the one a shifted deployment violates: on CIFAR-10-C the episode budget
> attains its nominal 0.900 coverage under an exchangeable control (0.905) and along nuisance axes,
> and collapses to 0.626 when the corruption family changes and 0.454 when severity changes, with
> half the severity splits admitting no budget at any declared level; weighted conformal restores
> coverage to 1.000 by driving decision yield to 0.000.
>
> What survives is a finite-sample certificate that trades decision yield for a false-adapt
> guarantee, and we measure that frontier exactly. Knowability-Guided Adaptation wraps any adapter
> (Tent, EATA, SAR), estimates the benefit Δ = R_T(f_0) − R_T(f_a) from label-free evidence, and
> commits only when a calibrated interval Δ̂ ± ε excludes zero; ε is a conformal radius and is *not*
> an estimate of β. On the CIFAR-10-C stress grid it commits on 66.8% of 6,480 cells at zero false
> adaptations over 1,113 and 1,244 ADAPT decisions (Clopper–Pearson 95% upper bounds 0.0027 and
> 0.0024 on the conditional rate), cuts regret 5.00× below the better fixed policy and 3.1× below a
> hindsight-tuned label-free drift heuristic held to the same budget, and places its commitments on
> cells whose true effect is 24.5× larger than the cells it declines. Under leave-one-corruption-out
> calibration the radius inflates 4.3×–6.9×, the commitment rate falls from 0.51–0.60 to 0.40–0.42
> and regret worsens 3.4×–7.6×, and the false-adapt count stays at zero in all ten runs; under that
> partition the certificate still beats always-adapt on all five Tent runs and loses to it on all
> five EATA runs, so what survives leave-one-corruption-out is the safety property and not the
> routing-utility one. On four one-sided natural tracks (Camelyon17, iWildCam, RxRx1, Office-Home)
> the certificate ties the better fixed policy at zero observed false adaptation, but three of those
> zeros are weak—Camelyon17's promoted subset contains no harmful cell, and RxRx1 and iWildCam make
> 0 and 1 ADAPT decisions—and the held-out record file behind the promoted iWildCam triple is absent
> from the release, so that row is a sealed summary rather than a reproduced result. On PACS,
> ImageNet-R and CIFAR-10.1 the certificate is a conservative null or fails the declared transfer
> bar, and we report those outcomes rather than withdraw the tracks; on ImageNet-R it is 1.76× worse
> than always-adapt. K-Bound is a safety, validity and abstention layer for test-time adaptation;
> the drift budget its own theory requires is not obtainable at a deployment, and the certificate is
> what remains when that is taken seriously.

---

## 8. Does the restructure achieve what it set out to?

**Yes on the argument. Partly on the artifact. No on length.**

**What it achieved.** The reframing is real and it is now load-bearing rather than rhetorical. The
paper used to present three findings that a referee had to be told were related: an impossibility
theorem, an embarrassing negative result about the authors' own parameter, and a certificate that
survived. It now presents one statement — *the decision needs a budget; no label-free audit can
supply one; here is that failure measured on 6,885 cells; here is exactly what it costs to escape;
here is what is left* — where each step entails the next and each is discharged by a numbered
result in the same document. `thm:short-audA` went from an appendix aside to `cor:audA` of a new
minimax theorem in the main body, and `cor:beta-is-beta` (Γ_z(C_β) = β) is the load-bearing
observation that makes the two halves of the paper one object: *the number bounding the abstention
band and the best number a label-free budget audit can return are the same number.* That is a
genuinely nice result and it was not visible before.

The negative result is now the paper's asset instead of its liability. A β-sweep that fails is a
confession; a β-sweep that fails *exactly where your own theorem says it must*, with the boring
explanation independently excluded (F6), is evidence.

**What it did not achieve.** Length went the wrong way — 87 pp, main body 54. The empirical panel
is still mostly one benchmark family: the strongest evidence is CIFAR-10-C, the natural tracks are
one-sided and three of their four zeros are weak by the paper's own admission, and the LOCO
partition kills the routing-utility claim on all five EATA runs. The escape route in §6 is priced
honestly but the price is high — E2 fails on exactly the deployment axes that motivate TTA — so §6
is closer to "here is why this is hard" than "here is the fix". And the paper still has three
proofless theorems in the appendix and two proof-sketch-only theorems in the long manuscript's
companion stack, all now explicitly marked, none load-bearing.

**Score.** Panel 4.2 → post-cleanup ~6.8 → **I estimate 7.3–7.6 now**, and I would submit it.

Justification against the two anchors. The 4.2 was a structural verdict: over-claimed, three
disconnected results, an unexplained negative. The 6.8 came from 32 fixes that made every
individual claim true but left the structure alone — that was the ceiling the brief correctly
diagnosed. This run buys roughly **+0.6** and no more, and I want to be precise about why it is not
more:

- **+0.5 for the spine.** A referee can now state the paper's thesis in one sentence, and TMLR's
  criterion is claims-supported-by-evidence, which this structure serves better than any other
  arrangement of the same facts. `cor:beta-is-beta` and `thm:anchor` are new, correct, and
  non-obvious.
- **+0.2 for §6 existing at all.** Naming the escape, proving the K bracket, and then *measuring
  the assumption failing* is the difference between an impossibility paper and a defeatist one.
- **−0.1 for length and for the empirical narrowness the restructure did not touch.** 87 pages is a
  real reviewing cost, and no amount of restructuring adds a second benchmark family.

Why not higher: the theory is elegant but each individual proof is short, and a referee may read
`thm:beta-minimax` and `thm:anchor` as careful bookkeeping over `lem:fibre` rather than as deep
results — the paper concedes this in `rem:honest-scope`, which is the right call and also caps the
score. Why not lower: the honesty is now a genuine competitive advantage. I could not find a single
number in either manuscript that did not reproduce from a named artifact, and I tried hard,
including on the two ImageNet-C coverage ranges that looked like a contradiction and were not. The
disclosure discipline — the np.quantile null, the seed-0 heterogeneity, the absent iWildCam record,
the SAR quarantine, the EATA loss under LOCO — is better than most accepted TMLR papers.

**Before submitting:** make `kbound.tex` share `kbound_abstract.tex` (§5, residual); restore or
regenerate the six figure assets; execute the dated re-freeze in `SUBMISSION_LEDGER.md` §0, which
currently and correctly reads NOT FROZEN.
