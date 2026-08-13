# K-Bound restructure plan: the impossibility spine

**Status:** executable specification. A writer follows this literally.
**Scope:** `kbound_abstract.tex`, `kbound_short_body.tex`, `kbound_short_appendix.tex` (shared by
`kbound_short.tex` / `kbound_tmlr.tex`), and `kbound.tex`.
**Inputs already proved and measured:** `/home/claude/kb_fixes/theory_beta_impossible.tex`,
`/home/claude/kb_fixes/theory_beta_estimable.tex`, and the artifacts named in §8 below.
**Author's standing rule in force throughout:** narrow the claim, never hedge it. No "arguably",
no "we believe", no adverb doing load-bearing work.

---

## 0. Executive summary

The paper currently presents three unrelated findings — an impossibility construction, a failed
attempt to declare `beta`, and a conformal wrapper that works — and asks the reader to hold them
apart. They are one statement, and the two theory runs just closed the last link. The restructure
reorders both manuscripts around a five-act spine in which every act is entailed by the one before
it:

1. **A label-free adapt/freeze decision is possible only relative to a declared drift budget `beta`.**
   (Already in the paper: `thm:headline`, whose sufficiency half is interval arithmetic on
   `[M-beta, M+beta]`.)
2. **No label-free procedure can supply that budget — and the exact minimax value is `beta` itself.**
   `thm:short-audA` moves from appendix to main body and is *replaced there* by the sharper
   statement it was a corner of: for every declared class the best label-free budget is the fibre
   radius `Gamma_z`, attained by a constant that reads none of its data; for the deployment class
   `C_beta` that radius equals `beta` exactly (`cor:beta-is-beta`), so the best possible audit
   returns its own input, **and that number is simultaneously the radius of the band on which the
   frontier must abstain**. Two riders: decision yield is bounded by the error budget `delta`
   unconditionally over the unrestricted class (`thm:beta-minimax`(d)), and labeled data from any
   other domain moves the minimax budget by exactly zero (`thm:anchor`). One lemma
   (`lem:fibre`) carries all of it, and also carries `lem:nonid` and `thm:headline`; the paper
   currently proves that lemma three times without naming it.
3. **Therefore the population frontier cannot be operationalized — and we measured that.** The
   `beta`-sweep stops being an isolated embarrassment and becomes the predicted instance of
   `thm:anchor`: the declaration procedure supplies term 1 of `prop:threeterm` and sets terms 2 and 3
   to zero by an undeclared exchangeability assumption. Its coverage column, its
   `1.4x`–`50x` shortfall and its `0.4%`–`16.6%` commit errors are what the theorem says to expect.
   The theory also disposes of the obvious rebuttal: across 470 splits the conformal order statistic
   and the sweep's plug-in quantile differ by a **median 0.34%**, so nothing here is a
   quantile-estimation failure.
4. **What escapes it, priced exactly.** `beta` is a property of a *population* of deployments. Under
   retrospective outcome logging (E1) and episode exchangeability (E2) it is identified as an
   episode quantile, estimable by a conformal order statistic, with the episode requirement
   bracketed to a factor of `e` (`1/(e*alpha) - 1 <= K* <= 1/alpha - 1`; between 3 and 9 logged
   deployments at `alpha = 0.10`, first feasible at exactly `K = 9` on real data). The escape is
   localized to a single logical move — **average over the episode law instead of supremum over the
   fibre, paid for with historical labels** — and neither ingredient works alone (`epi:prop-escape`).
   The honest plausibility assessment is the finding: E2 holds along nuisance axes and **fails along
   the axis that defines a new deployment** (coverage 0.905 control, 0.626 corruption family, 0.454
   severity), and the weighted-conformal relaxation restores coverage to 1.000 by driving decision
   yield to 0.000 — Aud-A's vacuity reappearing inside the method that was supposed to escape it.
5. **What remains achievable, and it is the deployable object.** A finite-sample certificate that
   buys decision yield against a false-adapt guarantee, with the frontier measured exactly:
   CIFAR-10-C commitment on 66.8% of 6,480 cells at `FA_u = 0/6480`, regret 5.00x below the better
   fixed policy, commitments landing on cells whose true effect is 24.5x larger than the declined
   ones; `FA_u = 0.0000` surviving leave-one-corruption-out in all ten runs, at a radius inflated
   4.3x–6.9x and regret 3.4x–7.6x worse; and ImageNet-R, where the same certificate is 1.76x worse
   than always-adapt.

**What this buys.** Act 2 is a main result instead of an appendix statement. Act 3 stops reading as
a self-inflicted wound and reads as a confirmed prediction. Act 4 is entirely new content that
*prices* the impossibility rather than softening it. Act 5 is unchanged in substance but acquires a
reason to exist.

**What this costs, stated plainly.** The main body loses ~14.8 pages of material and gains ~7.2
pages of promoted theory, netting **46 pp → ~38 pp of main body**; the appendix grows from 28 pp to
~38 pp; total goes from 74 pp to ~76 pp. **The restructure does not shorten the paper. It shortens
the part a referee reads first by eight pages and puts the load-bearing theory inside it.** If a
further cut is required, the ruthless option is stated in §7.3 (the one-bit / knowability-capacity /
multicandidate line, ~14 pp, shipped as a companion report) and it is a rewire of 23 `\ref`s.

**The three things that get cut outright and why:** the synthetic wiring check (§7.1 today) — a
50-line explanation of why a retired experiment could not have failed; the physical-study
pre-registration subsection — the paper itself says templates are not evidence; and the standalone
"Weak-Evidence and Negative Regimes" subsection — four sentences duplicating two tables. Details and
ten more cuts in §5.

---

## 1. The new abstract (verbatim — this is the file content)

Replace the entire body of `kbound_abstract.tex` (currently one ~950-word paragraph) with the
following. Keep the two comment lines at the top of the file.

```latex
% Shared abstract for kbound_short.tex (IEEEtran) and kbound_tmlr.tex (article/TMLR).
% Contains the abstract TEXT only -- each driver supplies its own abstract environment.
Test-time adaptation can silently degrade a deployed model on unlabeled target data, and with no
target labels the system cannot tell a helpful shift from a harmful one. Deciding \emph{whether} to
adapt is possible only relative to a declared bound on calibration drift---a drift budget
$\beta$---so the question that decides the method is where that budget comes from. This paper
answers it, and the answer is that it cannot come from the deployment.

We prove that for every declared class of target laws the exact minimax label-free budget is the
fibre radius $\Gamma_z$, attained by a constant audit that reads none of its data; for the
deployment class $\mathcal C_\beta$ the fibre radius equals $\beta$ exactly, so the best possible
label-free audit returns its own input, and the number it returns is simultaneously the radius of
the band on which a strict adapt-or-freeze commitment is unsound. Two consequences follow. Any
$\delta$-valid label-free audited rule commits with probability at most $\delta$ over the
unrestricted class---decision yield is bounded by the error budget, at every batch size and for
every evidence map. And a \emph{fully labeled} calibration sample from any other domain leaves the
minimax budget unchanged, absent a separately declared coupling. One lemma---that the label kernel
on the disagreement region is unconstrained by every label-free observable---carries the
matched-evidence impossibility, the abstention band, and the budget impossibility as readings of the
same one-parameter family at three radii.

We then measure the prediction. Declaring $\beta$ as a high quantile of realized drift on
source-like development cells, exactly as this paper's own method section prescribes, fails on
$6{,}885$ real evaluation cells: on CIFAR-10-C the declared budget is $1.4\times$ to $50\times$
smaller than soundness requires, $24$--$73\%$ of deployment cells fall outside the declared class,
and committed actions are wrong at $0.4$--$16.6\%$ where the frontier promises $0\%$; on ImageNet-C
a large enough budget is derivable and returns zero commitments on all $405$ cells in five of ten
configurations. This is not a quantile-estimation failure: across $470$ splits the conformal order
statistic that replaces the plug-in quantile differs from it by a median $0.34\%$, with a median
coverage difference of exactly zero.

What escapes the impossibility is a population of deployments rather than a deployment. Under
retrospective outcome logging and exchangeability with $K$ logged episodes, $\beta$ is identified as
an episode quantile and estimable by a conformal order statistic, with the episode requirement
bracketed to a factor of $e$: $1/(e\alpha)-1\le K^\star\le 1/\alpha-1$, between $3$ and $9$ logged
deployments at $\alpha=0.10$, and first feasible on real data at exactly $K=9$. The escape is
average-over-episodes instead of supremum-over-the-fibre, paid for with historical labels, and
neither ingredient suffices alone. The assumption that buys it is the one a shifted deployment
violates: on CIFAR-10-C the episode budget attains its nominal $0.900$ coverage under an
exchangeable control ($0.905$) and along nuisance axes, and collapses to $0.626$ when the corruption
family changes and $0.454$ when severity changes, with half the severity splits admitting no budget
at any declared level; weighted conformal restores coverage to $1.000$ by driving decision yield to
$0.000$.

What survives is a finite-sample certificate that trades decision yield for a false-adapt
guarantee, and we measure that frontier exactly. Knowability-Guided Adaptation wraps any adapter
(Tent, EATA, SAR), estimates the benefit $\Delta=R_T(f_0)-R_T(f_a)$ from label-free evidence, and
commits only when a calibrated interval $\widehat\Delta\pm\varepsilon$ excludes zero; $\varepsilon$
is a conformal radius and is \emph{not} an estimate of $\beta$. On the CIFAR-10-C stress grid it
commits on $66.8\%$ of $6{,}480$ cells at zero false adaptations over $1{,}113$ and $1{,}244$
\textsc{adapt} decisions (Clopper--Pearson $95\%$ upper bounds $0.0027$ and $0.0024$ on the
conditional rate), cuts regret $5.00\times$ below the better fixed policy and $3.1\times$ below a
hindsight-tuned label-free drift heuristic held to the same budget, and places its commitments on
cells whose true effect is $24.5\times$ larger than the cells it declines. Under
leave-one-corruption-out calibration the radius inflates $4.3\times$--$6.9\times$, the commitment
rate falls from $0.51$--$0.60$ to $0.40$--$0.42$ and regret worsens $3.4\times$--$7.6\times$, and
the false-adapt count stays at zero in all ten runs. Where the shift is one-sided the certificate
buys nothing and we report it: on ImageNet-R it is $1.76\times$ worse than always-adapt. K-Bound is
a safety, validity and abstention layer for test-time adaptation; the drift budget its own theory
requires is not obtainable at a deployment, and the certificate is what remains when that is taken
seriously.
```

**Word count:** ~620, against the current ~950. Every number in it is traced in §8.

---

## 2. The new contributions list (verbatim — replaces lines 89–99 of `kbound_short_body.tex`)

```latex
\paragraph{Contributions.}
\begin{enumerate}
\item \textbf{One lemma, three radii.} We isolate the construction the paper previously carried out
three times: on the disagreement region the label kernel is a free parameter that no label-free
observable constrains (Lemma~\ref{lem:fibre}). Read at radius $\delta$ it is the matched-evidence
impossibility (Theorem~\ref{lem:nonid}); at radius $\beta$ it is the abstention band of the
strict-commitment frontier (Theorem~\ref{thm:headline}); at the extreme points it is the vacuity of
label-free budget audits (Theorem~\ref{thm:aud-A}). Naming it once is what makes the rest of the
paper a single statement rather than three.
\item \textbf{The exact minimax label-free drift budget, and the identity that closes the loop.}
For every declared class $\mathcal C$ the least label-free-certifiable budget is the fibre radius
$\Gamma_z(\mathcal C)$, and it is attained by a constant audit that reads none of its data
(Theorem~\ref{thm:beta-minimax}(a),(b)). For the paper's own deployment class,
$\Gamma_z(\mathcal C_\beta)=\beta$ exactly (Corollary~\ref{cor:beta-is-beta}): \emph{the best
possible label-free audit of the drift budget returns the budget you declared, and that number is
the radius of the band on which the frontier must abstain.} Two riders make it operational rather
than philosophical: no label-free audit ever converts an abstention into a commitment
(Theorem~\ref{thm:beta-minimax}(c)), and over the unrestricted class the audited rule's decision
yield is at most the false-commitment budget $\delta$, at every batch size, every calibration sample
size and every evidence map (Theorem~\ref{thm:beta-minimax}(d)).
\item \textbf{Labels at the wrong domain buy nothing, and the three currencies that a budget costs.}
A fully labeled calibration sample of any size leaves the minimax budget at $\tfrac12+|M|$ unless a
coupling to the deployment domain is separately declared (Theorem~\ref{thm:anchor}), which is
strictly stronger than audit vacuity and is the statement that bears on our own measurements. The
bound $|\gamma_T|\le|\gamma_{\mathrm{cal}}|+\Delta_{\mathcal F}(\mu_{\mathrm{cal}},\mu_T)+\rho$
separates the three (Proposition~\ref{prop:threeterm}); exactly one term is label-free computable,
and declaring $\rho$ instead of $\beta$ is a renaming, because the budget-declaration problem is a
fixed point of the same construction (Remark~\ref{rem:reading}).
\item \textbf{The theorem explains our own negative experiment.} We ran the declaration procedure
this paper prescribes on $6{,}885$ real evaluation cells and it fails (\S\ref{sec:beta-sweep}):
$1.4\times$--$50\times$ too small on CIFAR-10-C, $24$--$73\%$ of cells outside the declared class,
commit errors of $0.4$--$16.6\%$ against a promised $0\%$, and zero commitments on all $405$
ImageNet-C cells in five of ten configurations. Theorem~\ref{thm:anchor} predicts exactly this: the
procedure supplies term~1 of Proposition~\ref{prop:threeterm} and sets terms~2 and~3 to zero by an
undeclared exchangeability assumption. We check the prediction quantitatively---under the label
shuffle of our own ablation the budget must collapse to $q_{0.90}(|\Delta-\E\Delta|)=0.2376$, and it
measures $0.2468$--$0.2606$ across six channels---and we report the one place the prediction fails
(\S\ref{sec:beta-sweep}, ImageNet-C sign reversal). It is also not a quantile-estimation failure:
across $470$ splits the conformal order statistic differs from the plug-in by a median $0.34\%$ at
a median coverage difference of exactly $0$.
\item \textbf{Where a budget can come from, what it costs, and why that does not help a shifted
deployment.} $\beta$ is a property of a population of deployments. Under retrospective outcome
logging (E1) and episode exchangeability (E2) it is identified as an episode quantile
(Theorem~\ref{epi:thm-ident}(a)) and estimable by a conformal order statistic
(Theorem~\ref{epi:thm-conformal}); it is \emph{not} identified from unlabeled episode observables at
any $K$, so the binding resource is historical \emph{labels}, not historical data
(Theorem~\ref{epi:thm-ident}(b)). We locate the escape from Theorem~\ref{thm:aud-A} to one logical
move---average over the episode law instead of supremum over the fibre---and prove neither
ingredient suffices alone (Proposition~\ref{epi:prop-escape}); we bracket the episode requirement to
a factor of $e$, $1/(e\alpha)-1\le K^\star\le1/\alpha-1$ (Theorem~\ref{epi:thm-floor}), verified at
exactly $K=9$ on real data; and we price the labels, with the calibrated score acting as a control
variate that makes estimating $\gamma$ strictly cheaper than estimating $\bar a-\tfrac12$
(Proposition~\ref{epi:prop-labels}). Then we measure E2 and it fails on the axis a shifted
deployment lives on: coverage $0.905$ under an exchangeable control and $0.626$/$0.454$ when the
corruption family or severity changes, with weighted conformal restoring coverage to $1.000$ at
decision yield $0.000$ (\S\ref{sec:episode-empirics}).
\item \textbf{What remains: a finite-sample certificate whose yield/safety frontier we measure.}
Under interval coverage the rule controls the marginal false-adapt and false-freeze events at level
$\alpha$ (Theorem~\ref{thm:certificate}); we do not claim the interval construction as novel
machinery, and $\varepsilon$ is not an estimate of $\beta$. What is new is the measured exchange
rate. On CIFAR-10-C the certificate commits on $66.8\%$ of $6{,}480$ cells at $\mathrm{FA}_{\mathrm
u}=0/6480$ over $1{,}113$ and $1{,}244$ \textsc{adapt} decisions (CP$_{95}$ $0.0027$/$0.0024$), cuts
regret $5.00\times$ below the better fixed policy and $3.1\times$ below a hindsight-tuned label-free
heuristic at the same budget, and commits on cells whose true effect is $24.5\times$ larger than
those it declines (permutation $p<0.0002$). Under leave-one-corruption-out recalibration the radius
inflates $4.3\times$--$6.9\times$, the commitment rate falls from $0.51$--$0.60$ to $0.40$--$0.42$,
regret worsens $3.4\times$--$7.6\times$, and the false-adapt count is $0$ in all ten runs. Where it
buys nothing we say so: ImageNet-R, $1.76\times$ worse than always-adapt, with per-backbone value
per committed decision exactly $0$ on $9$ of $10$ backbones.
\end{enumerate}
```

**Deltas from the current list:** 8 items → 6. Old item 1 ("TTA as a safety decision") is absorbed
into the Introduction's first paragraph — it is framing, not a contribution, and a referee reading a
contributions list wants results. Old items 2, 3 collapse into new 1+2 (the bookkeeping
decomposition is no longer sold as a contribution; `rem:gamma-residual` already concedes it is a
definition). Old item 6 (coverage certificate) merges into new 6. Old item 7 (KGA wrapper) becomes a
sentence in §7 of the body — a wrapper with no new objective is not a contribution and the paper
already says the machinery is not novel. Old item 8 (nine-track panel) merges into new 6; the panel
accounting is now reported as evidence for a measured frontier rather than as a standalone claim.

---

## 3. `kbound_short_body.tex` — section-by-section specification

Target section numbering after the restructure is in the left column. "Lines" are current line
numbers in `kbound_short_body.tex` as of this plan.

### §1 Introduction (lines 4–126) — **REWRITE**

| Element | Action |
|---|---|
| `fig:teaser` (5–14) | **KEEP**, caption unchanged. It is the only figure that shows the decision object. |
| ¶1 "Test-time adaptation (TTA) updates..." (15–23) | **KEEP**, unchanged. It states the problem. |
| ¶2 "Some cases are information-theoretically unknowable..." (25–47) | **REWRITE.** This paragraph currently narrates the old three-finding structure. It becomes the spine, in five sentences, one per act, each naming its result: `lem:fibre` → `cor:beta-is-beta` → `thm:anchor` → `sec:beta-sweep` → `epi:thm-ident` + `sec:episode-empirics` → `thm:certificate`. Keep the `garg2022atc` / `kalai2021abstain` citations exactly where they are; they are the positioning that stops a referee calling act 2 folklore. **Delete** the sentence "The same construction shows that $\beta$ itself admits no non-vacuous label-free audit (Appendix~\ref{app:audit-short}), which is why K-Bound treats the budget as declared" — it is now the paper's second main result, not a parenthetical. |
| ¶3 "The evidence follows the regime map..." (49–76) | **KEEP with one insertion.** Add, after the ImageNet-R sentence, the leave-one-corruption-out stress result (`FA_u = 0` in all ten runs at a $4.3\times$–$6.9\times$ inflated radius) — it is the strongest single empirical fact the paper owns and it currently appears nowhere in the introduction. |
| Plain-language takeaway (78–87) | **KEEP**, with `|M|\le\beta` phrasing retained. Change "the maximal sound three-way rule when $|M|\le\beta$" to add "and the budget $\beta$ is not obtainable at the deployment (\S\ref{sec:impossible})". |
| Contributions (89–99) | **REPLACE** with §2 of this plan, verbatim. |
| `tab:regime-summary` (101–122) | **KEEP**, 2 refs. Add one row: `Budget declarable?` / `any` / `no label-free audit (Thm. 2)` / `measured: $\widehat\beta$ $1.4$--$50\times$ too small`. |

### §2 Related Work (lines 127–168) — **COMPRESS to ~0.75 pp**

Merge the four subsections (`Test-Time Adaptation`, `Guarded and Monitored Adaptation`,
`Label-Free Performance Estimation`, `Impossibility, Abstention, and Risk Control`) into four
labeled paragraphs under one `\section`. Keep every `\cite`. Keep the "Relation to classical DA and
label-shift non-identifiability" paragraph (142–157) **in full** — it is the paragraph that
distinguishes act 2 from known non-identifiability results and it becomes load-bearing.
**Rewrite** "Surviving Gap, and What Is Not Ours" (158–168): the surviving gap is no longer "nobody
certifies the benefit sign", it is "nobody has priced the budget that any such certificate needs".
Saving: ~1.0 pp.

### §3 Setup and the four quantities (lines 169–290) — **KEEP, one move**

| Element | Action |
|---|---|
| 3.1–3.4 (171–248), incl. `tab:notation-main` | **KEEP.** `tab:notation-main` is referenced and is the reference card. |
| 3.5 Multiclass and Regression Scope + `prop:multiclass` (249–288) | **MOVE** to `kbound_short_appendix.tex` §`app:regression`, which already exists and already holds the derivations. `prop:multiclass` has **0 refs**; keep the label at its new location so the ledger inventory stays valid. Saving: ~0.7 pp. |
| 3.6 Formal Setup / `\input{paper/sections/theory_setup}` (289–290) | **KEEP.** `ass:deploy` (6 refs), `rem:gamma-residual` (3 refs), `def:strict-sound`, `def:risk-align`, `def:regimes` all survive unchanged. |

### §4 The budget cannot be supplied label-free — **NEW MAIN SECTION** `\label{sec:impossible}`

This replaces the current §4 Theory (lines 292–340). New content is
`/home/claude/kb_fixes/theory_beta_impossible.tex`, promoted from appendix status.

| New subsection | Source | Action |
|---|---|---|
| 4.1 The engine | `theory_beta_impossible.tex` §"The engine", `def:audit-data` + `lem:fibre` + proof + `rem:one-construction` | **PROMOTE VERBATIM.** ~0.6 pp. |
| 4.2 What the engine already proved | `paper/sections/theory_core_main.tex` `lem:reduction`, `lem:nonid`, `cor:matched-abstain`, `prop:closed-band`, `thm:headline` | **KEEP ALL FIVE, REWRITE THE CONNECTIVE PROSE.** Their proofs shrink: `lem:nonid`'s proof becomes "Lemma~\ref{lem:fibre} with $\eta\equiv\tfrac12\pm\delta$, $\delta<\min\{\beta-|M|,\tfrac12\}$; class membership and opposite signs as computed there." Saves ~0.4 pp and makes the structural point. **Keep the "the two evidence laws coincide exactly, not approximately / we do not present this as a Le Cam argument" paragraph** (theory_core_main 62–67) — `rem:one-construction` restates it and both must agree. |
| 4.3 The exact minimax label-free budget | `def:fibre-radius`, `thm:beta-minimax` (a)–(d) + proof, `cor:audA`, `cor:beta-is-beta` + proof, and the display paragraph after it | **PROMOTE VERBATIM.** ~1.6 pp. This is the new centre of the paper. |
| 4.4 Labels at the wrong domain buy nothing | `thm:anchor` + proof, `prop:threeterm` + proof, `rem:reading` | **PROMOTE VERBATIM.** ~0.8 pp. |
| 4.5 The escape, and its floor | `thm:lecam` **statement only** (proof → appendix), `thm:dichotomy` in full, `rem:notsharp` | **PROMOTE, PROOF SPLIT.** ~0.65 pp in main, ~0.4 pp in appendix. |
| 4.6 Honest scope | `rem:honest-scope` | **PROMOTE VERBATIM.** This is the paragraph that pre-empts "you can't estimate an unidentified parameter"; it must be in the main body, in the paper's own voice, not buried. ~0.2 pp. |
| — | Current 4.2 "How the Results Fit Together, and Which Steps Are Substantive" (306–325) | **CUT.** Its entire content — which steps are bookkeeping, which are the construction — is now the *organizing principle* of §4.1–4.2 rather than a retrospective apology. Preserve two sentences: the one separating `thm:certificate`'s assumed premise from its conclusion, and the pointer to `\S\ref{sec:fa-identity}`; relocate both into §7. Saving: ~0.5 pp. |
| — | Current 4.3 "Extensions and Deferred Results" (326–336) | **KEEP, REWRITE.** Drop the sentence deferring "the auditability of the drift budget itself" to `app:audit-short` — it is now §4. Keep the iWildCam streaming-script disclosure verbatim; it is a retraction and retractions do not move. |
| — | Current 4.4 "Claim Scope" (338–339) | **KEEP, REWRITE the last clause.** "The framework does not claim that $\beta$ is identifiable from unlabeled deployment data" becomes "Theorem~\ref{thm:beta-minimax} proves $\beta$ is not identifiable from unlabeled deployment data, and Theorem~\ref{epi:thm-ident}(b) proves it is not identifiable from unlabeled *historical* data either." |

**Label bridging, mandatory.** `theory_beta_impossible.tex` references `thm:aud-A`, `thm:aud-B`,
`thm:aud-G`, `thm:aud-H`, `app:auditable` and `conj:aud-maximal`. Those labels exist only in the
**long** build (`paper/sections/auditable_budgets.tex`). The short build has
`thm:short-audA/C/DE/G` and `app:audit-short`. Use the repo's established double-label idiom
(`\label{thm:headline}\label{thm:frontier}` is already in the tree):

- In the short build, the promoted Aud-A theorem carries `\label{thm:aud-A}\label{thm:short-audA}`.
- `app:audit-short` gains `\label{app:auditable}` on the same `\section`.
- `thm:short-audC` gains `\label{thm:aud-C}`; `thm:short-audG` gains `\label{thm:aud-G}`.
- `thm:aud-B` and `thm:aud-H` are **not** in the short build. In `theory_beta_impossible.tex` §4.5
  and `prop:threeterm`, replace `Theorem~\ref{thm:aud-B}` with
  `Theorem~\ref{thm:short-audC}\,(i)` and `Theorem~\ref{thm:aud-H}` with the phrase "the
  witness-class audit of Theorem~\ref{thm:short-audC}". Guard `conj:aud-maximal` (0 refs anywhere)
  by deleting the clause that cites it from `rem:notsharp` in the short build only.

### §5 The measured consequence: a `beta`-sweep (negative result) — **MOVE UP** `\label{sec:beta-sweep}`

Currently §7.2 (lines 820–1079). Move the entire subsection to become §5, a top-level section,
**before** the method and the experimental setup. Keep the label `sec:beta-sweep` (16 refs).

| Element | Action |
|---|---|
| Opening ¶ (822–826) | **REWRITE.** Currently: "the frontier does not operationalize, because $\beta$ cannot be declared from development data." Becomes: "Theorem~\ref{thm:anchor} predicts that a budget bought with labels at a domain other than the deployment domain is worth nothing absent a declared coupling. This section is that prediction, measured." Same conclusion, now entailed. |
| "What is empirically testable, given that the decomposition is an identity" (828–835) | **KEEP VERBATIM.** (C1)/(C2) framing is exactly right and is now supported by `prop:threeterm`: (C2) *is* the claim that terms 2 and 3 are zero. Add one sentence saying so. |
| Design / Label firewall / Scale convention (837–868) | **KEEP VERBATIM.** All three are provenance and none of it moves. |
| (C1), (C2) (870–898) | **KEEP VERBATIM.** |
| `tab:beta-sweep` (900–948) | **KEEP VERBATIM, including the null caveat in the caption.** The 0.90-null disclosure is non-negotiable — it is the documented `np.quantile` trap and the ImageNet-C entries carry no information. |
| `fig:beta-frontier` (950–970) | **KEEP.** |
| Confusion matrix + `tab:beta-confusion` (972–1006) | **KEEP.** |
| "Sensitivity: $\beta$ is a free knob" (1008–1016) | **KEEP.** |
| "Head-to-head against the empirical certificate" (1018–1026) | **KEEP.** |
| "The one positive result, stated precisely" (1028–1043) | **KEEP VERBATIM.** It is the scoped counter-evidence and it must survive a restructure that otherwise reads as prosecuting the frontier. |
| "What this does and does not touch" (1045–1078) | **REWRITE the first three sentences, KEEP the caveat list verbatim.** New opening: the proofs are untouched; what the experiment establishes is that the operational reading fails, *and* Theorem~\ref{thm:anchor} says why. |
| **NEW paragraph at the end** | "**It is not a quantile-estimation failure.**" ~8 lines: the 470-split comparison (median relative gap 0.34%, max 52%, median coverage difference exactly 0.000), and the anchor-collapse prediction check (predicted $0.2376$ vs measured $0.2468$–$0.2606$, ratios $1.04$–$1.10$, zero commitments in 5/5 replicates for `M_doc`/`M_atc4`), **including the ImageNet-C sign reversal that the prediction gets wrong** (real $0.199$–$0.239$ vs shuffled $0.074$–$0.090$) stated as a failed prediction, not omitted. Source: `THEORY_BETA_IMPOSSIBLE.md` §3.2–3.3, `beta_impossible/anchor_collapse_check.json`. |

### §6 Where a budget can come from, what it costs, and why it does not help here — **NEW SECTION** `\label{sec:episode}`

Source: `/home/claude/kb_fixes/theory_beta_estimable.tex`. Promote from its as-written appendix form
(`\section{...}\label{app:episode-beta}`) to a main-body section. Retain **every** `epi:*` label.

| New subsection | Source | Action | Main-body cost |
|---|---|---|---|
| 6.1 The episode model | `epi:def-episode`, `epi:ass-E1`, `epi:ass-E2`, `epi:def-beta-star` | **PROMOTE VERBATIM.** | 0.4 pp |
| 6.2 Identification and its price | `epi:thm-ident` (a) and (b), statement in full; **proof → appendix** | **PROMOTE STATEMENT.** Lead the subsection with (b): the binding resource is historical labels, not historical data. | 0.35 pp |
| 6.3 The estimator | `epi:thm-conformal`, `epi:rem-audG` | **PROMOTE STATEMENT; proof → appendix.** `epi:rem-audG` (the credit to `thm:aud-G`) is mandatory and stays in the main text, not a footnote. | 0.3 pp |
| 6.4 Exactly where the impossibility is escaped | `epi:def-fibreblind`, `epi:prop-escape` (a)(b)(c) + proof, `epi:rem-marginal` | **PROMOTE VERBATIM INCLUDING PROOF.** This is the consistency check between §4 and §6 and it is the single most attackable seam in the restructure; the proof belongs in the main body. `epi:rem-marginal` must be adjacent to `rem:fa-marginal`'s statement of the same distinction. | 0.7 pp |
| 6.5 How many episodes, how many labels | `epi:thm-floor` + `epi:cor-bracket` (statements), `epi:prop-labels` (statement + the control-variate identity), `epi:prop-probe` (statement + the break-even + the ranking sentence) | **PROMOTE STATEMENTS; proofs → appendix.** The "at a single deployment the probe strictly dominates the budget route" sentence stays — it is the uncomfortable honest ranking and cutting it would be exactly the hedging the author forbids. | 0.6 pp |
| 6.6 Relaxing exchangeability | `epi:thm-shift` (a)(b), `epi:cor-not-free` | **PROMOTE STATEMENTS; proofs → appendix.** `epi:cor-not-free`'s conclusion — the assumption moves from unfalsifiable to falsifiable, not from unverified to verified — is a main-text sentence. | 0.3 pp |
| 6.7 Is E2 plausible? | `epi:sec-honesty` | **PROMOTE VERBATIM.** ~0.25 pp. Non-negotiable: it is the honest plausibility assessment the brief requires, and it predicts §6.8 before §6.8 is read. | 0.25 pp |
| 6.8 Measured: E2 holds on nuisance axes and fails on deployment axes `\label{sec:episode-empirics}` | `epi:sec-numerics` incl. the coverage table and findings (i)–(iv) | **PROMOTE VERBATIM.** Add the two facts from `THEORY_BETA_ESTIMABLE.md` §4.3 F3 that are not yet in the .tex and that strengthen it: **5 of 10 `LOSEV` splits are infeasible outright** (no declared level works at any $K$), and the per-split oracle $K$ needed on `LOCO` has median 36 against a nominal 9. Keep the null-is-0.900 statement in the same paragraph as the numbers. | 0.55 pp |
| 6.9 Two consequences for the rest of the paper | `theory_beta_estimable.tex` §"Two consequences" (the 0.34% comparison; $\Delta_{\mathrm{sep}}$ as a one-sided screen) | **SPLIT.** The 0.34% paragraph **moves to §5** (see above) — it belongs next to the experiment it rescues. The $\Delta_{\mathrm{sep}}$ paragraph **stays here**, with its recommendation ("report it; do not treat it as a certificate"). | 0.2 pp |
| 6.10 What is left open | `epi:conj-open` + the e-process paragraph | **PROMOTE VERBATIM.** A clearly-marked conjecture with a statement of why the two natural candidates fail is worth more than a fourth theorem. | 0.15 pp |

**Total §6 main-body cost: ~3.8 pp.** Proofs relocated to appendix: ~4.5 pp.

**Numerics warning that must be written into §6.8 and honoured by the writer:** the null coverage is
0.900 *by construction*. Any sentence that reads a coverage of ~0.90 as a positive finding is the
`frontier_validation.py` failure mode. Only deviations are findings. The same applies to the
`RAND` control: 0.905 and 0.912 are evidence the estimator is *not broken*, not evidence it works.

**Bonus finding to act on before the rewrite** (`THEORY_BETA_ESTIMABLE.md` §4.4): seed 0 of
`stress_grid_multiseed_v1` is not exchangeable with seeds 1–4 (mean $\Delta$ 0.0895 vs 0.128,
update norm 12.39 vs 3.43, `LOSO/0` coverage 0.730/0.731 vs 0.885–0.942). This is a property of the
artifacts, not of the analysis, and it may affect any number pooled over five seeds. **Add a
one-sentence disclosure to §8 (experimental setup) and a row to `SUBMISSION_LEDGER.md` §10;** do not
silently re-pool.

### §7 What remains: the finite-sample certificate and KGA — **MERGE** `\label{sec:method}`

Absorbs current §4's `thm:certificate` block and current §5 (Method, lines 341–557).

| Element | Action |
|---|---|
| `thm:certificate` + proof + `rem:fa-marginal` + `cor:abstain-valid` (theory_core_main 158–204) | **MOVE HERE** from §4. Its role changes from "third pillar of the theory" to "the object that survives §4 and §6". Open the section with one sentence saying that. |
| §5.1 System Overview (342–386) | **KEEP**, with the long conformal-landscape citation paragraph (379–386) **moved to Related Work**. Saving ~0.3 pp. |
| §5.2 Evidence Map and Benefit Estimator (388–433) + `tab:evidence-map` | **KEEP.** `tab:evidence-map` has 0 refs — add a `\ref` in the text rather than cut it; it is the schema a re-implementer needs. |
| §5.3 Out-of-Fold Uncertainty Calibration (434–474) | **KEEP.** |
| §5.4 Population Frontier versus Empirical Certificate (475–498) | **KEEP, SHORTEN by half.** Half of it is now redundant with §4.3 and §5; the surviving content is the single distinction "$\varepsilon$ is not an estimate of $\beta$", which is in the ledger as a mandatory distinction. |
| §5.5 Selecting and Auditing the Drift Budget (499–509) | **CUT.** Entirely subsumed by §4 and §6, which is the point of the restructure. Rewire: whatever refs point into it go to `sec:impossible` / `sec:episode`. Saving ~0.4 pp. |
| §5.6 Deployment Semantics and Failure-Safe Behavior (510–528) | **KEEP.** |
| §5.7 Computational Cost (529–557) + table | **MOVE to appendix** `app:runtime`, which already exists. Saving ~0.5 pp. |

### §8 Experimental setup (lines 558–755) — **KEEP, trim**

Keep `par:quantile-rule`, `par:loo-pool`, `par:multiplicity`, `sec:detectability` /
`def:detectable` (4 refs), `sec:evidence-policy` — all five are pre-registration and none of them
moves. **Add** the seed-0 heterogeneity disclosure (§6.8 note above). **Update RQ list** (559–567):
RQ1a currently refers to `\S\ref{sec:synthetic}`, which is being cut — replace with a reference to
`sec:beta-sweep`. Move `tab:dataset-taxonomy` (0 refs) to appendix or add a `\ref`.

### §9 Results (lines 756–2047) — **KEEP the core, CUT/MOVE seven subsections**

| Current subsection | Action | Rationale |
|---|---|---|
| Opening ¶ (757–762) | **REWRITE.** It currently points readers to §7.2 and §7.5 as "the weakest points". After the restructure §7.2 is §5 and is a main result; the opening becomes a map onto acts 3 and 5. |
| 7.1 Synthetic wiring check (768–818) incl. `fig:frontier-measured` | **CUT.** 51 lines and a figure explaining why a *retired* experiment could not have come out negative. The honest content is three sentences and belongs in the appendix. **Replacement text, exact:** in `app:supp-exp`, "An earlier version opened the results with a synthetic run in which the generator supplies $M$, $\gamma\sim\mathrm U(-\beta,\beta)$ and hence $\sign\Delta$. It cannot come out negative: the evidence is four noisy copies of $M$, so the conformal radius converges to $q_{0.9}(|\mathrm U(-\beta,\beta)|)=0.9\beta$ by algebra and the commit rule reduces to $|M|>0.9\beta$. It is retained in the repository as a unit test and is not evidence." **Rewires:** lines 564, 813, 823, 2082 (`\ref{sec:synthetic}`) and 791 (`\ref{fig:frontier-measured}`). Saving: ~1.6 pp. |
| 7.2 $\beta$-sweep | **MOVED** to §5. |
| 7.3 CIFAR-10-C stress grid (1080–1255) | **KEEP IN FULL.** This is act 5's evidence. **Promote** the "Leave-one-corruption-out calibration" paragraph (1140–1151) from a mid-subsection paragraph to its own labeled paragraph immediately after the headline result, and state the full ten-run table: $\varepsilon$ $0.0152$–$0.0219\to0.0926$–$0.1122$, commitment rate $0.51$–$0.60\to0.398$–$0.417$, regret ratio $3.4\times$–$7.6\times$, **$\mathrm{FA}_{\mathrm u}=0.0000$ in all ten**. Source `out_cifar_loco_tent_eata.json`. |
| 7.4 Decision-baseline comparison (1256–1298) | **KEEP.** |
| 7.5 Mixed head-to-head vs POEM/AETTA (1299–1362) | **KEEP.** |
| 7.6 ImageNet-C SAR (1363–1491) | **KEEP**, unchanged including the demotion to point-estimate. |
| 7.7 Natural Shifts and Consolidated Summary (1492–1585) | **KEEP.** `tab:uniform-panel` and the evidence tiers are the ledger §3 promotion table and must survive intact. **Fix an existing inconsistency while here:** lines 2001 and 2030 still say Camelyon17 is "not reproducible from release" / "unverifiable", contradicting §11 (line 2308ff) and `SUBMISSION_LEDGER.md` §8a, which re-promoted it to recomputable on 2026-07-26. Reconcile to "recomputable; and vacuous, because 18/18 cells are helpful and KGA adapts 18/18." |
| 7.8 What an observed $\mathrm{FA}_{\mathrm u}=0$ establishes (1586–1668) `sec:fa-identity` | **KEEP IN FULL** (11 refs). It is the paper's own refutation of its most quotable number and it now sits inside act 5 where it belongs. |
| 7.9 What the certificate buys (1669–1913) `sec:decision-value` | **KEEP IN FULL** (9 refs). This is act 5's frontier. |
| 7.10 Variance behind the panel means (1914–1967) + `tab:variance` | **MOVE to appendix.** 0 refs to `sec:panel-variance`. It is a variance audit, not a result. Saving ~1.3 pp. |
| 7.11 Weak-Evidence and Negative Regimes (1968–1972) | **CUT.** Four sentences that restate `tab:regime-summary` and two Limitations paragraphs. Nothing references it. Saving ~0.2 pp. |
| 7.12 Physical-Study Pre-registration (1974–1979) | **CUT from main body.** The paragraph's own content is "no physical result is claimed" and the paper elsewhere says "templates are not evidence". `app:edge-physical` stays; move the single `\ref` into the Limitations pre-registration sentence. Saving ~0.25 pp. |
| 7.13 Constructed Heterogeneous Routing (1981–2006) + `tab:primary-numeric` | **MOVE to appendix.** Explicitly a researcher-constructed aggregate, explicitly not transfer. 1 ref (from `tab:claim-status`, which is also moving to the appendix). Saving ~1.0 pp. |
| 7.14 Consolidated Claim and Guarantee Accounting (2008–2047) `sec:guarantees` | **MOVE to appendix**, and **merge with `tab:claim-status`** (2344–2370), which is the same object at a different granularity. One appendix table, two labels, both preserved. Update three rows to the new spine: `Label-free audit of $\beta$ is vacuous` → `The minimax label-free budget is $\Gamma_z$; $\Gamma_z(\mathcal C_\beta)=\beta$` (theorem, §4.3); add `$\beta$ estimable from $K$ exchangeable logged episodes` (theorem, §6.3) and `E2 holds at a novel deployment` (**false, measured**, coverage 0.626/0.454, §6.8). Saving ~1.8 pp. |

### §10 Limitations (lines 2179–2300) — **MERGE §9 in, REWRITE two blocks**

| Element | Action |
|---|---|
| Current §9 Discussion and Failure Modes (2138–2178) | **MERGE INTO LIMITATIONS.** "Why Natural Shifts Mainly Produce No-Harm", "Four Sources of Abstention", "When to Deploy K-Bound" are limitations wearing a different hat; `tab:failure-modes` moves with them. Saving ~1.2 pp. |
| "Setting $\beta$" block (2215–2221) | **REWRITE.** It currently says $\beta$ "can, however, be *audited*" and lists Lipschitz/index-drift purchases. After §4 that reads as contradicted by the paper's own §4.4. New text: no label-free audit supplies a budget; every audit in App.~\ref{app:audit-short} purchases labels at an anchor plus a declared coupling (Prop.~\ref{prop:threeterm}); the only route to a budget that a deployer without deployment labels can take is the episode route of §6, at $K\ge1/\alpha-1$ logged deployments and its exchangeability cost. |
| "$\beta$ is not estimable from development data" block (2223–2246) | **KEEP the caveat list verbatim; CUT the first ten lines**, which now duplicate §5 nearly sentence for sentence. Replace with a two-sentence pointer plus the new limitation: *E2 is not verifiable, only falsifiable, and its observable component is only a one-sided screen ($r=-0.35$, $n=470$).* Saving ~0.8 pp. |
| **NEW block: "The escape is marginal, not conditional."** | ~6 lines. `epi:prop-escape`(a) + `epi:rem-marginal` + `rem:fa-marginal`: a deployer told "the budget was estimated" gets a long-run rate across deployments, not protection today. This is the honest cost of act 4 and it must be in Limitations, not only in §6. |
| **NEW block: "The probe dominates at one deployment."** | ~4 lines. `epi:prop-probe`(b). The budget route is worth taking only because it amortises; if E2 fails, it loses both its guarantee and its economic rationale. |
| "What we do not claim" box (2294–2300) | **KEEP, ADD three items:** that $\beta$ can be supplied by any label-free procedure (now a theorem, not a concession); that episode exchangeability holds at a novel deployment (measured false); that weighted conformal repairs it (measured: valid and vacuous). |
| All other blocks | **KEEP VERBATIM.** Every disclosure about absent artifacts, sealed rows, seed counts and the iWildCam/Office-Home provenance survives untouched. A restructure never removes a disclosure. |

### §11 Reproducibility (2302–2371) — **COMPRESS**

Keep §11.1 (artifact lineage) in the main body verbatim; it is provenance. **Move** §11.2 Formal
Verification Scope to the appendix as a single labeled paragraph (~1.0 pp saved) and **move**
`tab:claim-status` to the merged appendix table above (~0.5 pp saved). Leave a two-sentence pointer.

### §12 Conclusion (2372–2393) — **REWRITE, same length**

Three paragraphs, one per half of the spine. ¶1: the budget cannot come from the deployment, and
that is a theorem with an exact value, not a limitation. ¶2: it can come from a population of
deployments at a priced cost in logged episodes and labels, and the assumption that buys it is the
one a shifted deployment violates — measured. ¶3: what remains is a certificate that trades yield
for a false-adapt guarantee, with the frontier measured, including where it buys nothing. **Delete**
the current ¶3's "we regard it as a contribution rather than a concession" framing — after the
restructure the negative result is entailed by §4 and needs no defence. **Keep** the final
implication paragraph (update mechanism + validity layer) and the calibration-not-sample-size
finding.

---

## 4. `kbound_short_appendix.tex` — specification

| Appendix | Action |
|---|---|
| `app:calib-eval`, `app:evidence-schema` | **KEEP.** `app:calib-eval` has 0 refs — add one from §7. |
| `app:theory-full` (Complete core proofs) | **KEEP and EXPAND.** Now also holds: `thm:lecam`'s proof, `epi:thm-ident` proof, `epi:thm-conformal` proof, `epi:thm-floor` proof, `epi:prop-labels` proof, `epi:prop-probe` proof, `epi:thm-shift` proof. ~+4.9 pp. |
| `prop:beatsboth-asym` (inside `app:theory-full`) | **CUT.** 0 refs anywhere in either build; it sharpens a frontier whose operational reading is withdrawn. Saving ~0.6 pp. Record the cut in `SUBMISSION_LEDGER.md` §2 (it is listed there). |
| `app:regression` | **KEEP, RECEIVES** §3.5's multiclass scope text and `prop:multiclass`. |
| `app:supp-exp` | **KEEP, RECEIVES** the three-sentence synthetic-run note. |
| `app:edge-physical`, `app:d33`, `app:runtime`, `app:formal`, `app:claim-artifact`, `app:imagenetc-ms`, `app:arm-inventory` | **KEEP.** `app:runtime` receives §5.7 (computational cost). `app:formal` receives §11.2. |
| `app:audit-short` (508–586) | **RESTRUCTURE, do not cut.** `thm:short-audA` and its proof **move out** to main §4 (double-labelled `thm:aud-A`). What remains is the *positive* side — `thm:short-audC`, `thm:short-audDE`, `thm:short-audG` — and its new role is stated in one new opening paragraph: **these are the instances of Proposition~\ref{prop:threeterm}; each purchases the first term with labels at an anchor and sets the third to zero by declaration; none omits the first term, which is the non-vacuity certificate for §4.** Keep the "stated here without proof, proved in the long version" honesty note verbatim. Add `\label{app:auditable}` to the section head. |
| **NEW** `app:episode-proofs` | Receives the §6 proofs, the full coverage tables (both benchmarks, both estimators, FA columns, ImageNet-C), the $K$-sweep table, and the two implementation-bug disclosures from `THEORY_BETA_ESTIMABLE.md` §4.1 (the nested cross-fit that manufactured coverage 1.000, and the unshuffled `cross_val_predict` that produced a separability AUC of 0.24 where the truth is 0.5). **The bug disclosures are mandatory** — anyone reusing `cross_val_predict` on these artifacts hits the second one, and the superseded run is kept in the tree as `*_UNSHUFFLED_SUPERSEDED.*`. ~+5.5 pp. |
| **NEW** `app:selfcheck` | One table: the five algebra self-checks C1–C5 from `theory_selfcheck_results.json`, all PASS, including the control-variate identity ($\mathrm{Var}(u)=0.20001$ vs $\mathrm{Var}(C)=0.25000$, $\mathrm{Var}(s)=0.04999$, saving factor $0.800$) and the tightness of `epi:thm-shift`(a) (measured miss $0.0978/0.1452/0.1968/0.2935$ against $\rho\alpha=0.10/0.15/0.20/0.30$). ~0.4 pp. |
| **RECEIVES** from main body | §7.10 variance tables, §7.13 constructed routing + `tab:primary-numeric`, merged §7.14 + `tab:claim-status`, §5.7 cost table, §11.2. ~+4.6 pp. |

---

## 5. The cut list, consolidated

Nothing below is moved; it is deleted from the source tree (the repository artifacts stay).

| # | What | Where | Why it no longer earns its place | Refs to rewire |
|---|---|---|---|---|
| C1 | §7.1 synthetic wiring check, 51 lines + `fig:frontier-measured` | short body 768–818 | Fifty lines arguing that a retired non-experiment could not have failed. Superseded by §5. | 4 × `sec:synthetic`, 1 × `fig:frontier-measured` |
| C2 | §4.2 "How the Results Fit Together" box | short body 306–325 | Its content is now the section order. A paper that has to draw a diagram explaining which of its results are real has the wrong order. | 0 |
| C3 | §5.5 "Selecting and Auditing the Drift Budget" | short body 499–509 | Fully subsumed by §4 and §6. | 0 |
| C4 | §7.11 Weak-Evidence and Negative Regimes | short body 1968–1972 | Duplicates `tab:regime-summary` and two Limitations paragraphs. | 0 |
| C5 | §7.12 Physical-Study Pre-registration | short body 1974–1979 | A pre-registration with no result, in a results section, in a paper that says templates are not evidence. | 1 × `app:edge-physical` (moves to Limitations) |
| C6 | `prop:beatsboth-asym` + proof | short appendix, in `app:theory-full` | 0 refs in either build; sharpens a withdrawn operational reading. | 0 |
| C7 | The `\iffalse` block, short-paper duplicate abstract snippet | `kbound.tex` 1798–1844; `kb_fixes/_snapshot_kbound_short.tex` | Dead source in the build tree. | 0 |
| C8 | Long-build appendix `app:stack` "Companion manuscript: the demoted theorem stack" | `kbound.tex` 2195–2258 | 0 refs. An appendix that announces itself as demoted. | 0 |
| C9 | Long-build `theory_v2` appendix block (12 `\input`s: sequential anytime, multicandidate, multiclass capacity, minimax optimality, tight constants, regression bracketing, Lean appendix) | `kbound.tex` 1869–1888 | Extensions of a frontier whose operational reading is withdrawn. They add referee surface and no line to the spine. Ship as a companion technical report. | 2 × `thm:anytime`, 1 × `thm:multicand`, 3 × `app:extensions` in `kbound.tex`; 10 more inside `paper/sections/*` (see §7.3) |
| C10 | Long-build `knowability_capacity.tex` + `knowability_capacity_general.tex` | `kbound.tex` 2352–2353 | Same reason as C9. A $\tau=1$ threshold in a 1-D location model is a different paper. | 2 × `sec:knowcap`, 4 × `sec:knowcap-gen` |
| C11 | Related-work subsection structure (four `\subsection`s → four `\paragraph`s) | short body 130–141 | Four one-paragraph subsections is a table of contents pretending to be a section. | 0 |
| C12 | Duplicated $\beta$-sweep prose across §5 / Limitations / Conclusion | short body 2223–2235, 2377–2390 | The same ten facts stated three times at full length. Keep once at full length (§5) and twice as pointers. | 0 |
| C13 | §11.2 Formal Verification Scope in the main body | short body 2324–2341 | An inventory of what Lean does *not* cover. Belongs in the appendix inventory it describes. | 0 |
| C14 | The `\ref{app:audit-short}` framing sentence in §4.3 "Extensions and Deferred Results" | short body 329–330 | It defers to an appendix the result that is now §4. | 0 |

**Everything on this list that is a disclosure, a retraction, a caveat, an absent-artifact note, or
a negative result is NOT on this list.** The only negative-adjacent item cut is C1, and C1 is the
retirement notice for something already retired.

---

## 6. Theorem migration table, and what it does to the length

### Appendix → main body

| Result | From | To | Proof in main? | Main-body cost |
|---|---|---|---|---|
| `def:audit-data` | `theory_beta_impossible.tex` (new) | §4.1 | n/a | 0.15 pp |
| `lem:fibre` | new (isolates a construction used 3× in the current paper) | §4.1 | **yes** | 0.45 pp |
| `rem:one-construction` | new | §4.1 | n/a | 0.2 pp |
| `def:fibre-radius` | new | §4.3 | n/a | 0.1 pp |
| `thm:beta-minimax` (a)–(d) | new | §4.3 | **yes** | 0.85 pp |
| `cor:audA` = `thm:aud-A` = `thm:short-audA` | **`app:audit-short` → main** | §4.3 | **yes** | 0.35 pp |
| `cor:beta-is-beta` | new | §4.3 | **yes** | 0.4 pp |
| `thm:anchor` | new | §4.4 | **yes** | 0.3 pp |
| `prop:threeterm` + `rem:reading` | new | §4.4 | **yes** | 0.5 pp |
| `thm:lecam` | new | §4.5 | no (appendix) | 0.25 pp |
| `thm:dichotomy` + `rem:notsharp` | new | §4.5 | **yes** (3 lines) | 0.4 pp |
| `rem:honest-scope` | new | §4.6 | n/a | 0.2 pp |
| `epi:ass-E1`, `epi:ass-E2`, `epi:def-episode`, `epi:def-beta-star` | new | §6.1 | n/a | 0.4 pp |
| `epi:thm-ident` (a),(b) | new | §6.2 | no | 0.35 pp |
| `epi:thm-conformal`, `epi:rem-audG` | new | §6.3 | no | 0.3 pp |
| `epi:def-fibreblind`, `epi:prop-escape`, `epi:rem-marginal` | new | §6.4 | **yes** | 0.7 pp |
| `epi:thm-floor`, `epi:cor-bracket`, `epi:prop-labels`, `epi:prop-probe` | new | §6.5 | no | 0.6 pp |
| `epi:thm-shift`, `epi:cor-not-free` | new | §6.6 | no | 0.3 pp |
| `epi:conj-open` | new | §6.10 | n/a | 0.15 pp |
| **Total promoted** | | | | **~7.2 pp** |

### Main body → appendix (results, not prose)

`prop:multiclass` (§3.5 → `app:regression`); `thm:certificate` stays in the main body but moves
sections (§4 → §7). No other theorem leaves the main body.

### Length accounting

Baseline, measured by building `kbound_tmlr.tex` in this container (74 pp total; Conclusion ends
p. 47, so main body ≈ 46 pp, appendix ≈ 28 pp):

| | pp |
|---|---|
| Main body, before | 46.0 |
| Promotions in (table above) | +7.2 |
| Cuts C1–C6, C11–C14 | −6.4 |
| Moves to appendix (§3.5, §5.7, §7.10, §7.13, §7.14, §11.2/`tab:claim-status`) | −6.8 |
| Related-work compression + §9 merged into §10 + §5.4 halved | −1.6 |
| **Main body, after** | **≈ 38.4** |
| Appendix, before | 28.0 |
| Proofs relocated from §4/§6 | +4.9 |
| New `app:episode-proofs` (tables, $K$-sweep, bug disclosures) | +5.5 |
| New `app:selfcheck` | +0.4 |
| Received from main body | +6.8 |
| `prop:beatsboth-asym` cut | −0.6 |
| **Appendix, after** | **≈ 45.0** |
| **Total, after** | **≈ 83** vs 74 before |

**Stated honestly: the restructure grows the paper by ~9 pages and shrinks the main body by ~8.**
TMLR has no page limit and judges claim–evidence support, so this is the right trade there. It is
**not** the right trade for the IEEEtran two-column build, which is already a desk-reject risk at 23
pages. **Recommendation: `kbound_short.tex` is retired as a submission target and kept only as a
formatting check.** If a two-column short paper is still wanted, the correct object is a new
6–8 page extraction consisting of §1, §4, §5 and a two-page §9 — not a compression of this body.

---

## 7. `kbound.tex` (long manuscript) — specification

### 7.1 Main body

| Current | Action |
|---|---|
| §1 Introduction (125–327) | **REWRITE** onto the spine, mirroring the short body's §1. |
| §2 Related work (328–467) | **KEEP.** The long form has room; only the "surviving gap" framing changes. |
| §3 Problem setup (468–489), §4 Definitions / `theory_setup` (490–492) | **KEEP.** |
| §5 "Theory: four pillars" (493–524) | **RENAME and RESTRUCTURE** to "Theory: one lemma, three radii, and a budget that cannot be supplied". The split-observability framing paragraph (494–506) **stays** — it is the generality claim and it is correctly scoped. The two `\noindent\emph{...}` deferral paragraphs (516–524) pointing at knowability-capacity and certificate extensions are **cut with C9/C10**. |
| `\input{paper/sections/main_theory_5}` (507) | **KEEP the file, REORDER inside it.** New order: `lem:reduction` → **`lem:fibre` (inserted)** → `thm:imp`/`lem:nonid` → `cor:forced-abstain` → `thm:frontier`/`thm:headline` → `thm:cert`/`thm:certificate`. `lem:gate` (2 refs), `prop:lecam-finite` (7 refs), `prop:cert-sample` (0 refs) and `rem:beta-zero` (0 refs) are **cut** from the compiled build with C9; if a reviewer objects, `prop:lecam-finite` is the one to restore. |
| **NEW §6** | `\input{kb_fixes/theory_beta_impossible}` **in full, with all proofs.** In the long build every label it references (`thm:aud-A`, `thm:aud-B`, `thm:aud-G`, `thm:aud-H`, `app:auditable`, `conj:aud-maximal`) **already exists** in `paper/sections/auditable_budgets.tex` — no bridging needed. Move `auditable_budgets.tex` from appendix position (2347) to immediately after this section, since §6 now explains what it is for. |
| **NEW §7** | `\input{kb_fixes/theory_beta_estimable}` **in full, with all proofs and the full numerics tables.** Change its `\section{...}\label{app:episode-beta}` to a main-body `\section`; keep the label. Its internal `\ref{sec:experiments}` must be repointed to `\ref{sec:exp}`. |
| §6 Method: KGA (525–586) | **KEEP, one fix.** The paragraph "The threshold is derived, not tuned" (588 region) claims `thm:frontier` "identifies $\varepsilon$ as the *exact* benefit-sign budget". That is the conflation the ledger forbids ($\varepsilon$ is not an estimate of $\beta$) and §5/§6 of the short body now contradict it explicitly. **Rewrite** it to: $\varepsilon$ is fixed on the calibration split and never selected on target-test data; it is a conformal radius, not a budget; what separates KGA from ATC-style thresholding is the abstain region and the finite-sample false-adapt control, not the value of the threshold. |
| §7.1 $\beta$-sweep long form (596–652) `sec:beta-sweep-long` | **KEEP and MOVE UP** to immediately after the new §6, mirroring the short body. Keep the label (3 refs). |
| Remaining experiments (653–1569) | **KEEP.** |
| §8 Excluded wins (1570–1597) | **KEEP.** |
| §9 Limitations (1598–1711) | **KEEP + the four new blocks specified for the short body's §10.** |
| §10 Discussion (1712–1730) | **MERGE into Limitations.** |
| §11 Conclusion (1731–1755) | **REWRITE** onto the spine. |
| §12 Reproducibility (1756–1797) | **KEEP.** |
| `\iffalse` block (1798–1844) | **CUT** (C7). |

### 7.2 Long-build appendices

**KEEP:** `app:supp` (6 refs), `app:c10c`, `app:status`, `app:weakest` (2 refs), `app:inventory`,
`kbound_frontier_appendix.tex`, `auditable_budgets.tex` (moved to main body per above).
**CUT:** `app:stack` (C8), the twelve `theory_v2` `\input`s (C9), `knowability_capacity*.tex` (C10).
**ADD:** an appendix holding the `theory_beta_estimable` numerics tables that do not fit inline
(`beta_estimability/tables.md` rendered), and the self-check table.

### 7.3 The ruthless option, priced

C9 + C10 remove ~14 pp from the long build. The rewire cost, counted from the tree today:

| Label | refs in `kbound.tex` | refs in `paper/sections/*` | refs in `theory_v2/*` |
|---|---|---|---|
| `thm:conj1-dichotomy` | 14 | 9 | 6 |
| `conj:gen` | 3 | 24 | 2 |
| `thm:multicand` | 1 | 9 | 4 |
| `thm:ev-rate` | 3 | 5 | 0 |
| `thm:anytime` | 2 | 1 | 2 |
| `prop:lecam-finite` | 0 | 4 | 3 |
| `sec:knowcap` / `sec:knowcap-gen` | 2 / 4 | 0 | 0 |

Refs inside files that are themselves cut do not need rewiring. **The binding cost is 29 `\ref`s in
surviving files** (`kbound.tex` and the surviving `paper/sections/*`). This is a two-hour mechanical
job and it removes fourteen pages of results that no longer connect to anything the paper claims.
**Do it.** If it is deferred, `thm:conj1-dichotomy`, `conj:gen` and `thm:ev-rate` must at minimum be
demoted to a single "deferred results" paragraph with statements and no proofs, which recovers ~9 pp.

---

## 8. Number provenance — every figure this plan puts in the abstract or contributions

| Number | Artifact |
|---|---|
| $1.4\times$–$50\times$; $24$–$73\%$; $0.4$–$16.6\%$; $5$ of $10$ zero-commitment ImageNet-C configs; $15$ of $18$ | `frontier_sweep_v1/beta_sweep_results.json`, `BETA_SWEEP_FINDINGS.md`, `tab:beta-sweep` |
| median relative gap $0.34\%$, max $52\%$, median coverage difference $0.000$, $470$ splits | `beta_estimability/compare_to_sweep_results.json` |
| predicted $q_{0.90}(\lvert\Delta-\E\Delta\rvert)=0.2376$ vs measured $0.2468$–$0.2606$ (ratios $1.04$–$1.10$); ImageNet-C $0.0689$ vs $0.0742$–$0.1171$; the sign reversal $0.199$–$0.239$ vs $0.074$–$0.090$ | `kb_fixes/beta_impossible/check_anchor_collapse.py`, `anchor_collapse_check.json` |
| coverage $0.905$ [$0.901$, $0.908$] control; $0.626$ [$0.621$, $0.631$] LOCO; $0.454$ [$0.449$, $0.460$] LOSEV; $M_{\mathrm{atc4}}$ $0.912$/$0.866$/$0.595$; $32{,}400$ deployment cells per axis | `beta_estimability/episode_beta_results.json` (`summary`), `tables.md` |
| $K$-sweep: infeasible at $K=3,5$; $0.907\pm0.088$ at $K=9$ | `beta_estimability/episode_beta_results.json` (`k_sweep`) |
| weighted conformal LOSEV $0.454\to1.000$, yield $0.615\to0.000$; separability AUC $0.497\pm0.033$ control | same |
| $5$ of $10$ LOSEV splits infeasible; median $K$ needed $36$ on LOCO | `THEORY_BETA_ESTIMABLE.md` §4.3 F3, `tripwire_results.json` |
| $\Delta_{\mathrm{sep}}$ screen: $r=-0.346$, $p=1.2\times10^{-14}$, $n=470$; $<0.70$ → mean coverage $0.905$, $7.7\%$ below $0.85$ | same |
| control variate $\mathrm{Var}(u)=0.20001$, $\mathrm{Var}(C)=0.25000$, $\mathrm{Var}(s)=0.04999$, saving $0.800$; `epi:thm-shift`(a) tightness $0.0978/0.1452/0.1968/0.2935$ | `beta_estimability/theory_selfcheck_results.json` |
| yield $0.668$ / regret $0.00150$ / $\mathrm{FA}_{\mathrm u}=0/6480$ (W$_{95}$ $0.0006$); $5.00\times$; $3.1\times$; $24.5\times$ ($p<0.0002$); ImageNet-R $1.76\times$; $7{,}365/7{,}365$ replay | `frontier_sweep_v1/decision_value_results.json`, `DECISION_VALUE_FINDINGS.md` |
| $1{,}113$ / $1{,}244$ adapts, $0$ false, CP$_{95}$ $0.0027$ / $0.0024$ | `SUBMISSION_LEDGER.md` §3 decision-accounting table |
| LOCO: $\varepsilon$ $0.0152$–$0.0219\to0.0926$–$0.1122$ ($4.3\times$–$6.9\times$); commitment rate $0.509$–$0.600\to0.398$–$0.417$; regret $3.4\times$–$7.6\times$; $\mathrm{FA}_{\mathrm u}=0.0000$ in all ten | `frontier_sweep_v1/out_cifar_loco_tent_eata.json` (recomputed for this plan across all ten runs) |
| $6{,}885$ cells; $6{,}480$ + $405$ | `beta_sweep_results.json`, `fs_common.py` |

**Two numbers in the brief that the artifacts do not support as stated, and that the writer must
use in the corrected form:** the LOCO radius inflation is $4.3\times$–$6.9\times$ (brief said
"$\sim0.021\to\sim0.10$", true only of the Tent runs), and the LOCO regret degradation is
$3.4\times$–$7.6\times$ (brief said "$4$–$6\times$"). Both recomputed above from
`out_cifar_loco_tent_eata.json`.

---

## 9. Label survival contract

Every label below is `\ref`'d somewhere in a surviving file and **must exist after the restructure**.
Counts are `\ref` occurrences measured in the tree today (short build / long build).

**Must survive, unchanged location semantics:** `lem:nonid` (26/11), `thm:headline` (24/4),
`thm:frontier` (12/28), `thm:cert` (11/32), `thm:certificate` (12/4), `lem:reduction` (14/18),
`ass:deploy` (6/6), `prop:closed-band` (5/3), `cor:matched-abstain` (4/1),
`rem:gamma-residual` (3/0), `def:strict-sound` (2/2), `thm:imp` (6/44),
`cor:forced-abstain` (0/7), `sec:detectability` (4), `sec:fa-identity` (11),
`sec:decision-value` (9), `sec:limits` (4), `app:theory-full` (2), `app:regression` (2),
`app:d33` (2), `tab:regime-summary` (2), `tab:abl-transfer` (2).

**Must survive at a NEW location:** `thm:short-audA` (3) — moves appendix → main §4.3, gains
`\label{thm:aud-A}`. `sec:beta-sweep` (16) — moves §7.2 → §5. `prop:multiclass` (0) — §3.5 →
`app:regression`; keep the label because `SUBMISSION_LEDGER.md` §2 inventories it.
`sec:guarantees` (1) and `tab:claim-status` — merged into one appendix float carrying both labels.

**Cut, with `\ref`s to rewire (exact call sites listed in §5):** `sec:synthetic` (4),
`fig:frontier-measured` (1), `app:edge-physical` (1 — moves to Limitations),
`sec:constructed-routing` (1 — target moves to appendix), `prop:beatsboth-asym` (0),
`conj:aud-maximal` (0), plus C9/C10's 29 in the long build.

**Zero-ref labels retained deliberately** (they are anchors a reader or the ledger uses):
`sec:intro`, `sec:related`, `sec:setup`, `sec:exp`, `sec:repro-main`, `app:calib-eval`,
`app:formal`, `app:supp-exp`, `fig:teaser`, `tab:notation-main`. Zero-ref labels retained but
requiring a `\ref` to be added in the rewrite: `tab:evidence-map`, `tab:dataset-taxonomy`,
`fig:decisive`, `fig:natural-forest`.

---

## 10. Consistency checks the writer must run before declaring done

1. `grep -c` every label in §9 in the post-restructure tree; zero misses.
2. Build both drivers; `grep "undefined"` in both `.log`s returns nothing new relative to today's
   baseline (`lmodern` and `IEEEtran` are installed; `tmlr.sty` is absent by design and the TMLR
   driver builds under its documented `article` shim).
3. **The Aud-A escape seam.** §4.3 says no fibre-blind budget valid uniformly over the fibre beats
   $\Gamma_z$; §6.3 offers a budget computed from a history that *is* independent of the current
   episode's label kernel. These are consistent only via `epi:prop-escape`. Verify by hand that §6.4
   is present, complete, and cross-referenced from §4.3 with a forward pointer. **If a referee finds
   one seam in this paper, it is this one.**
4. **The 0.900 null.** Search the post-restructure body for every occurrence of a coverage figure
   near 0.90 and confirm each is either labelled a null/control or is a deviation. The
   `np.quantile` trap is already documented in `adversarial_ablations_results.json` and in
   `tab:beta-sweep`'s caption; the episode section reintroduces the same trap in a new place.
5. **Camelyon17 status.** Lines 2001 and 2030 contradict line 2308 and `SUBMISSION_LEDGER.md` §8a.
   One status, stated once: recomputable, and vacuous (18/18 helpful, adapts 18/18).
6. **$\varepsilon$ vs $\beta$.** Ledger §6 lists this as a mandatory distinction. After the
   restructure it appears in §5.4, §7, the abstract, and `kbound.tex`'s "threshold is derived"
   paragraph. All four must say the same thing, and the long-manuscript one currently does not.
7. **No claim of novelty for `epi:thm-conformal` over `thm:aud-G`.** `epi:rem-audG` must be in the
   main text of §6.3.
8. Update `SUBMISSION_LEDGER.md` §2 (theorem inventory), §3 (unchanged), and the claim ledger
   `claim_ledger.json` for: the promoted results, the cut `prop:beatsboth-asym`, the new episode
   results, and the two corrected LOCO ranges from §8.

---

## 11. What this restructure does not fix

Stated so nobody mistakes reorganization for evidence.

- **Two benchmarks.** Six corruption families on CIFAR-10-C, three on ImageNet-C, and all three of
  ImageNet-C's are noise variants. The spine is now coherent; the empirical base is not wider.
- **The effective unit count.** $\Delta$ correlates at $0.913$ across seeds within a
  condition/method and at $0.995$ between design replicates. $6{,}885$ is not $6{,}885$ independent
  units, and the restructure does not change that.
- **Act 2 remains mechanically elementary.** `rem:honest-scope` concedes it: you cannot estimate an
  unidentified parameter. What is not elementary is the *value* $\Gamma_z$, the identity
  $\Gamma_z(\mathcal C_\beta)=\beta$, the yield-$\le\delta$ exchange rate, and `thm:anchor`. A
  referee who rejects all four rejects the paper, and the fallback is act 5 alone.
- **Act 4's guarantee is marginal over deployments, not conditional on one.** A deployer told the
  budget was estimated gets a long-run rate. This is `rem:fa-marginal` again, one level up, and it is
  now stated in three places by design.
- **The highest-value next experiment is still unrun.** `FRONTIER_EXPERIMENT_VERDICT.md` names it:
  exhibit, on real data, two deployment cells with statistically indistinguishable label-free
  evidence and opposite benefit signs. That converts `lem:nonid` from a construction into a measured
  phenomenon. It is runnable on the artifacts already in this tree (11-dim $Z$, 6,885 cells,
  matched-evidence nearest-neighbour search) and it is publishable whether it succeeds or fails.
  **The restructure makes room for it; it does not perform it.**
