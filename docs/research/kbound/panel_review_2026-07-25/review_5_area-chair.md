# Reviewer 5 - Senior Area Chair / Venue Reviewer (framing, novelty, writing)

## Bottom line

This is one of the most carefully self-audited submissions I have handled: the authors maintain a claim ledger, an evidence-tier policy, a leakage audit, a quarantine file for a non-reproducing result, and a Lean formalization with an explicit "what is *not* formalized" inventory. That discipline is real and I want to say so up front. But discipline is not a contribution, and underneath the scaffolding the paper has three problems that a program committee will find quickly. (1) The headline theory is definitional: `γ` is *defined* as `ā − 1/2 − M`, so `sign Δ = sign(M+γ)` is an identity, and "strict commitment is sound iff `|M| > β`" is elementary arithmetic about whether an interval of width `2β` around `M` crosses zero; the impossibility result is the standard fact that unlabeled data does not pin down the label conditional, restricted to the disagreement region. The authors' own internal doc reaches the same verdict ("near-definitional", `VENUE_BENCHMARK_2026-06.md:97`). (2) The theory and the experiments are disjoint: `β` is never numerically supplied to KGA on any of the nine real tracks (the paper says so at `kbound_short.tex:596-598`), so the frontier is exercised only in a simulation whose data-generating process forces the frontier to appear. (3) Several headline sentences are not supported by the paper's own tables — most seriously, the abstract's "across every natural distribution shift we test … *rendition* … KGA is uniformly no-harm: it matches the better fixed policy" is contradicted by the panel's own ImageNet-R row (KGA 0.0112 vs always-adapt 0.0064) and PACS row (0.0431 vs 0.0176), and the ImageNet-C per-seed appendix table was computed under a conformal rule the paper elsewhere disowns, which is the only reason the "5/5 seeds" claim holds.

**Verdict:** Reject at NeurIPS/ICML/ICLR (score 4/10, confidence 4/5). Borderline at a mid-tier venue / TMLR (score 5/10) — but *not* in its current state, because at least two claims in the abstract are falsified by tables inside the same PDF and must be fixed regardless of venue.

---

## What is done well

- **The evidence-tier policy is genuinely good practice.** `kbound_short.tex:570-571` defines locked / reconciled / provisional / diagnostic, and Table `tab:uniform-panel` carries a tier on every row. I have not seen this in a submission before and it should be preserved.
- **Negative results are retained rather than dropped.** CIFAR-10.1 is reported with `FA_u = 0.167` and `FA_c = 0.444` and explicitly labeled a "locked diagnostic fail" (`:917`). PACS and ImageNet-R are retained as nulls. Most authors would have cut these.
- **A non-reproducing result was withheld, not quietly fixed.** `CIFAR10C_SAR_QUARANTINE.md` withholds the archived CIFAR-10-C SAR aggregate and lists six explicit reinstatement gates. I checked that no CIFAR-10-C SAR row appears in the frozen display table (`tab:decisive` caption, `:662`).
- **The `ε ≠ β` distinction is held consistently.** It appears in the abstract, `tab:notation-main`, `rem:four-quantities`, §"Population Frontier versus Empirical Certificate", and the guarantee-accounting table marks "`ε` estimates drift budget `β`" as **false** (`:978`). This is the single most confusable point in the paper and the authors handled it correctly everywhere I looked.
- **The label-requiring iWildCam streaming result is voluntarily demoted** (`:296-302`) rather than reported as a label-free deployment result. That is the right call and it costs them a headline.
- **The Lean scope disclaimer (`:1178-1186`, `app:formal`) is unusually honest** — it enumerates what is *not* formalized, including the witness construction and frontier necessity.

---

## Findings

### [BLOCKER] F5-1 — The abstract, intro, and conclusion claim uniform no-harm on natural shifts that the paper's own panel shows KGA loses on

**Location:** `docs/research/kbound/kbound_short.tex:41` (abstract), `:84-88` (intro), `:1221` (conclusion) vs `:915-916` (Table `tab:uniform-panel`).

**Evidence.** Abstract: *"Across every natural distribution shift we test---hospital, wildlife-camera, laboratory-batch, domain, and rendition shifts---KGA is uniformly \emph{no-harm}: it matches the better fixed policy, averts the failure of the wrong one."* The intro repeats it with the datasets named: *"hospital (Camelyon17), wildlife-camera (iWildCam), laboratory-batch (RxRx1), domain (Office-Home), and **rendition (ImageNet-R)** shifts---KGA is uniformly no-harm."*

The paper defines no-harm as "matching the better fixed policy while avoiding the worse" (`:119-120`). Computing that from the panel rows:

```
Track        KGA     best fixed policy    excess      ratio
ImageNet-R   0.0112  0.0064 (adapt)      +0.0048     1.75x
PACS         0.0431  0.0176 (adapt)      +0.0255     2.45x
CIFAR-10.1   0.0021  0.0017 (freeze)     +0.0004     1.24x   (FA_u = 0.167 > alpha)
```

ImageNet-R is the "rendition shift" the intro explicitly names. PACS is a "domain generalization" track (`tab:dataset-taxonomy:529`) and the abstract's list says "domain … shifts". CIFAR-10.1 is classified as a *natural* resampling shift by the paper's own taxonomy (`:531`) and its `FA_u = 0.167` exceeds the nominal `α = 0.10`, so "keeps observed false adaptation near zero" is also not true of every natural track.

Worse, the paper contradicts itself internally: §Limitations (`:1109`) says "PACS and ImageNet-R are completed, locked null diagnostics; CIFAR-10.1 is a locked diagnostic fail," and the panel's own interpretation column for ImageNet-R reads "candidate-dependent fixed-policy tracking; 0/10 CI beats-both" — i.e. not no-harm.

**Why it matters.** This is the paper's *primary* claim ("Our primary evidence is a safety guarantee"). A reviewer who reads the abstract and then Table XV will conclude the abstract is inaccurate, and everything downstream loses credibility — including the parts that are correct. In a paper whose entire selling point is calibrated honesty, this is fatal.

**Fix.** Rewrite the sentence to the truth the tables support: *"On the four one-sided natural shifts with locked held-out artifacts (Camelyon17, iWildCam, Office-Home, RxRx1) KGA ties the better fixed policy at zero observed false adaptation; on PACS, ImageNet-R and CIFAR-10.1 it is a conservative null or fails the transfer bar."* Remove "rendition" and "domain" from the enumerated no-harm list, or move PACS/ImageNet-R/CIFAR-10.1 into an explicitly named "where it does not hold" clause in the abstract.

---

### [BLOCKER] F5-2 — The ImageNet-C "5/5 seeds" claim survives only under a conformal rule the paper elsewhere disowns; the appendix per-seed table is stale and contradicts the main table

**Location:** `kbound_short.tex:800-805` and `:910`, `:1206`; `kbound_short_appendix.tex:294-312` (Table `tab:imagenetc-perseed`).

**Evidence.** Main text: *"…at `FA_u = 0.000`, **using the exact split-conformal radius** `ε=ρ_(k)`, `k=⌈(n+1)(1−α)⌉`… As a secondary utility check, point estimates improve both fixed-policy regrets on **5/5 seeds**; the per-seed 27-cell CIs exclude zero on 2/5."* Appendix Table VIII pooled row reports regret `0.0107` and `FA_u = 0.007`; the main table reports `0.0264` and `FA_u = 0.000` for the same track.

I recomputed both from the manifest's own named source, `experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json`, replaying the decision rule from the saved `b_hat` and `B`:

```
rule                     FA_u        abstain   regret KGA/adapt/freeze
interpolated q_0.9      1/135=.0074    54       0.0107 / 0.0529 / 0.0319
exact rank ceil((n+1)(1-a)) 0/135=0   109       0.0264 / 0.0529 / 0.0319   <- matches manifest
```

Per-seed under the **exact-rank** rule the main text says it used:

```
seed0 K/A/F = 0.0319/0.0625/0.0319  beats both? NO (exact tie with always-freeze)
seed1 K/A/F = 0.0312/0.0595/0.0312  beats both? NO (exact tie)
seed2 K/A/F = 0.0102/0.0425/0.0284  beats both? yes
seed3 K/A/F = 0.0290/0.0441/0.0290  beats both? NO (exact tie)
seed4 K/A/F = 0.0297/0.0561/0.0389  beats both? yes
```

Under the **interpolated** rule it is 5/5. The appendix per-seed regret column (0.0108, 0.0091, 0.0128, 0.0056, 0.0154) and `FA_u` column (0, 0, **0.037**, 0, 0) reproduce my interpolated run *exactly*, so Table VIII was not regenerated in the G8 exact-rank pass (`SUBMISSION_LEDGER.md:83-89`).

**Why it matters.** The paper takes the aggregate from the exact-rank rule (which gives `FA_u = 0.000`) and the per-seed support from the superseded rule (which gives 5/5 beats-both and hides that 3/5 seeds are *exact ties with always-freeze*). Whether or not that was intentional, it is selecting the more favorable of two rules within a single claim, and it is the paper's only multi-seed beats-both result on a large-scale benchmark. A reviewer who does what I just did will stop reading. It also silently contradicts §"Guarantee accounting" (`:973`, "point-level 5/5") and `tab:claim-status` (`:1206`).

**Fix.** Regenerate Table VIII under the exact-rank rule, report per-seed beats-both as **2/5 point-level with 3/5 exact ties to always-freeze**, and restate the ImageNet-C claim as "pooled paired-bootstrap gaps exclude zero; per-seed the effect is concentrated in the two heavier-harmful-mass seeds." That is still a publishable, defensible result — the current framing is not.

---

### [MAJOR] F5-3 — The headline theorem is definitional, and the impossibility result is exact non-identifiability, not a Le Cam lower bound

**Location:** `paper/sections/theory_setup.tex:19-26`, `paper/sections/theory_core_main.tex:3-62`, `:90-127`.

**Evidence.** `theory_setup.tex:21-26` defines `M := E[s | D] − 1/2` and `γ := E[η_a − s | D]`. Therefore `M + γ = E[η_a | D] − 1/2 = ā − 1/2` **by construction**, and `lem:reduction`'s "`sign Δ = sign(M+γ)` holds for every target law" is an identity, not a result. `thm:headline` (ii) is then a two-line triangle inequality (`:119-120`: "if `M>β` and `|γ|≤β`, then `M+γ>0`"), and (iii) is "pick `γ = −M`". Stripped of vocabulary the theorem is: *an interval `[M−β, M+β]` has a determined sign iff `|M| > β`.*

`lem:nonid`'s construction (`:35-62`) sets the label kernel to `1/2 ± δ` on `D` and leaves `μ_T, f_0, f_a, s` fixed. Because `Z = φ(X_{1:m}, f_0, f_a)` uses no labels, the two evidence laws are **identical**, not merely close — `TV = 0`. That is not a Le Cam two-point argument (a Le Cam argument needs `TV < 1 − 2α`, i.e. distinguishable-but-not-reliably); it is the observation that an unlabeled sample carries zero information about the label conditional. The authors' own positioning file nonetheless calls it "a Le Cam two-point construction" (`paper/sections/related_work_positioning.tex:32`).

**Why it matters.** The paper's contribution list leads with "Exact strict-commitment frontier" as a theorem-level contribution and the Conclusion repeats the "iff" as the central result. An area chair evaluating theory depth will discount all of §Theory to a definition, a triangle inequality, and a one-line coverage implication (`thm:certificate`'s proof is three lines, `:164-168`). The authors' own venue benchmark reaches the same conclusion: *"the adversarial check this session showed the knowability dichotomy is near-definitional"* (`VENUE_BENCHMARK_2026-06.md:97`).

**Fix.** Do not sell the frontier as the theoretical contribution. Present `lem:reduction`+`thm:headline` as a *definitional bookkeeping device* (which is what it is, and it is a useful one), and make the contribution the **decision framing plus the deployable wrapper**. If you want a real theorem, the interesting object is Aud-A (vacuity of label-free budget audits) — that one has content.

---

### [MAJOR] F5-4 — The theory is not tested by any of the nine tracks; the one experiment that does test it is circular by construction

**Location:** `kbound_short.tex:593-610`, `:511` (RQ1); `docs/research/kbound/scripts/frontier_validation.py:52-58`.

**Evidence.** The paper states outright (`:596-598`): *"In the real-data tracks that follow, `β` is **not** numerically supplied to KGA: the empirical decision uses `Δ̂±ε`, and the `β`-frontier is evaluated directly only in this synthetic study."* So RQ1 ("whether the measured commit transition follows the theoretical frontier", `:511`) is answered by the synthetic study alone.

That study's generator (`frontier_validation.py:52-57`) is:

```python
M     = rng.uniform(m_lo, m_hi, n)
gamma = rng.uniform(-beta, beta, n)      # unobserved
B     = M + gamma                        # true benefit
Z     = np.column_stack([M + rng.normal(0, obs_noise, n) for _ in range(4)])  # obs_noise=0.02
```

`Z` is four noisy copies of `M`, so any regressor recovers `B̂ ≈ M`, the residual is exactly `γ`, and the conformal radius converges to `q_0.9(|U(−β,β)|) = 0.9β`. The commit rule `|B̂| > ε` therefore *is* `|M| > 0.9β`. The script's own docstring says it: *"Because gamma is the irreducible residual, the split-conformal radius eps self-calibrates to ~beta, so the decision rule reproduces the |M|>beta frontier automatically."* The paper's defence of this — "The frontier is not imposed; it appears because the synthetic residual scale is controlled by the deliberately hidden drift" (`:609-610`) — concedes the point in the same sentence it denies it.

**Why it matters.** This is the classic theory-vs-experiment disconnect that sinks strong-sounding theory papers, and here it is total: the population theory has *zero* contact with the nine benchmark tracks, and the one experiment that touches it validates an algebraic identity the authors wrote into the simulator. What the nine tracks actually evaluate is a gradient-boosted regressor plus a conformal interval — a competent but ordinary label-free heuristic that shares vocabulary with the theory. The abstract's "KGA turns this frontier into practice" is not supported; the paper even says as much at `:80` ("not a numerical `|M|>β` test").

**Fix.** Either (a) demote §4.1 to an appendix sanity check and stop calling it "validation of the theory", and be explicit in the abstract that the empirical layer is theory-*inspired* rather than theory-*testing*; or (b) construct one real track where `s` and hence `M` is actually computed (you have an ATC-style source-calibrated score already — `:364`), declare a `β` from historical dev-to-deployment calibration gaps, and run the population rule head-to-head against `Δ̂±ε`. (b) would materially strengthen the paper.

---

### [MAJOR] F5-5 — `β` is declared, never measured, never swept on real data, and the paper's own Aud-A theorem says it can never be audited label-free

**Location:** `kbound_short.tex:456-459`, `:1009-1011`, `kbound_short_appendix.tex:321-327`.

**Evidence.** §"Selecting and Auditing the Drift Budget" (`:457`) prescribes: *"Sensitivity analysis should report how decisions change over a pre-specified range of budgets."* No such analysis exists. Ablation (v) (`:1009-1011`) reads in full: *"The population-frontier `β` sweep is the synthetic ground-truth validation of §4.1, kept distinct from the empirical radius `ε`."* — i.e. the only `β` sweep is over the simulation of F5-4, at four values (`0.05…0.20`), with no real data.

Meanwhile Theorem `thm:short-audA` proves that any label-free audit `β̂ = g(Z)` with uniform coverage must satisfy `β̂ ≥ 1/2 + |M|`, "the audited frontier then abstains in every world" (appendix `:321-327`). So the parameter that makes the headline theorem non-vacuous is, by the paper's own theorem, permanently unmeasurable without purchased side information.

**Why it matters.** Every reviewer will ask *"who declares `β`, and what happens if they declare it wrong?"* The paper's answer is: no one does, in any experiment; and if they got it wrong the certificate would silently mis-commit with no diagnostic. Table `tab:failure-modes` lists "True drift exceeds `β`" with diagnostic "external domain monitoring" (`:1083`) — which is not a diagnostic, it is a deferral. Combined with F5-4 this leaves the population theory with no demonstrated operating regime anywhere in the paper.

**Fix.** Run the sensitivity analysis you prescribe: on the CIFAR-10-C stress grid, compute `M` from the ATC-style source-calibrated score, sweep `β ∈ {0, 0.02, 0.05, 0.10, 0.20}`, and report commit rate, regret and `FA_u` for the *population* rule alongside the interval rule. Even a negative result here ("the population rule is dominated by the interval rule at every `β`") would be far more informative than the current silence.

---

### [MAJOR] F5-6 — "Beats both whenever mixed and detectable" is unfalsifiable: "detectable" is never defined and regimes are labeled post hoc

**Location:** `kbound_short.tex:41` (abstract), `:132` (Table `tab:regime-summary`), `:823-825` (Table `tab:imagenetc-faithful`).

**Evidence.** Abstract: *"KGA beats both fixed policies **whenever** helpful and harmful conditions are mixed and detectable."* The word "detectable" occurs ten times (`:41, 91, 104, 114, 132, 586, 805, 1116, 1120, 1221`) and is nowhere given an operational criterion.

Table `tab:regime-summary:132` assigns "Mixed + detectable" to exactly CIFAR-10-C Tent/EATA and ImageNet-C SAR — precisely the tracks where KGA wins. But `tab:imagenetc-faithful:823` shows ImageNet-C **Tent** at **56% harmful cells** — the most mixed track in the entire paper — and there KGA does *not* beat both ("no (≈ties freeze)"). It is simply not listed as "mixed + detectable".

**Why it matters.** A universally quantified claim whose scope condition is defined only by whether the claim came out true is not a scientific claim. This is the single easiest thing for a hostile reviewer to attack, and it undercuts the paper's otherwise-earned reputation for calibrated language.

**Fix.** Define detectability ex ante and measurably — e.g. "a regime is detectable at level `α` if the cross-fitted `|Δ̂| − ε > 0` on at least `p%` of harmful cells" — then classify all nine tracks by that criterion *before* reporting outcomes, and report how many detectable-mixed tracks actually produced beats-both (it will be 2 of 3, with ImageNet-C Tent as the counterexample). Or downgrade "whenever" to "in the mixed regimes we constructed".

---

### [MAJOR] F5-7 — A contributions bullet cites two datasets (FMoW, Poverty) that appear nowhere in the paper

**Location:** `kbound_short.tex:114`.

**Evidence.** *"…honest nulls (**FMoW, Poverty**, CIFAR-10.1) are retained to show the certificate declines unsupported claims."* I grepped both manuscript files: FMoW and Poverty occur exactly once each, in that sentence. They are absent from `tab:dataset-taxonomy`, from the nine-track panel, from `tab:primary-numeric`, from `paper/generated/kbound_result_manifest.json` (12 tracks, neither present), and from `SUBMISSION_LEDGER.md`. Runs do exist in the repo (`experiments/kbound/results/fmow_protocol_L_v1/`, `poverty_protocol_L_dev/`) but were never promoted.

This also contradicts the paper's own scoping note two pages earlier: *"The uniform panel below is the sole empirical index; diagnostic variants and historical runs are not additional evidence for the headline claims"* (`:140-141`).

**Why it matters.** It is in the **contributions list**. A reviewer looking for the FMoW null finds nothing, and reasonably infers that the contribution list was written to a wider evidence base than the paper contains.

**Fix.** Delete "FMoW, Poverty" from `:114`, or add a one-row appendix table with the actual numbers and a tier label.

---

### [MAJOR] F5-8 — The authors' own honest novelty assessment is written down but never compiled into the paper, and the piece they identify as the defensible core was cut as "near-vacuous"

**Location:** `paper/sections/related_work_positioning.tex` (never `\input`); `RELATED_WORK_POSITIONING.md:28-38`; `SUBMISSION_LEDGER.md:50`.

**Evidence.** `related_work_positioning.tex` contains a paragraph titled "What is, and is not, novel here (honest scope)": *"The anytime-valid certificate machinery is **not** our novel contribution: confidence-sequence risk monitoring for TTA already exists and unsupervised accuracy estimation already states the no-assumption impossibility informally."* The companion markdown is blunter: *"The **certificate (Thm 3) is the least novel part**"* (`RELATED_WORK_POSITIONING.md:28`).

I grepped for `related_work_positioning` across every `.tex` in the tree: the only hit is the file's own header comment. It is not `\input` by `kbound_short.tex`, `kbound_short_appendix.tex`, or `kbound.tex`. So none of this reaches a reviewer.

Both documents prescribe the same remedy: *"Lead the paper with the knowability boundary (Thm 1 + **one-bit dichotomy**) … That is the part with no direct competitor."* But `SUBMISSION_LEDGER.md:50` records: `thm:conj1-dichotomy [--] One-bit dichotomy -- NOT COMPILED (\iffalse; **near-vacuous, keep out**)`, which I confirmed at `paper/sections/theory_appendix_ext.tex:73-74`. The one component the authors named as their only uncontested novelty was excised as near-vacuous, and the component they call "the least novel part" is what shipped.

**Why it matters.** The compiled §II Related Work (`:143-159`) contains no equivalent honesty. Its "Surviving Gap" paragraph (`:158-159`) asserts a gap without conceding what is shared machinery. A reviewer who knows POEM (NeurIPS'24) and Schirmer et al. (2025) will supply the concession themselves, less charitably.

**Fix.** Compile the honest-scope paragraph, or a two-sentence version of it, into §II. "Our certificate shares machinery with anytime-valid TTA monitoring; our contribution is the pre-commitment decision object and the abstention region, not the interval construction" costs you three lines and buys enormous goodwill. Also reconcile the two docs with the ledger — right now they prescribe leading with a theorem that was deleted.

---

### [MAJOR] F5-9 — Related Work does not differentiate the impossibility result from the classical domain-adaptation / label-shift identification literature

**Location:** `kbound_short.tex:155-156`; `paper/references_kbound_expanded.tex`.

**Evidence.** The entire treatment is one sentence: *"General label-free target-risk identification requires assumptions on the shift~\cite{garg2022atc,steinhardt2016,bendavid2010theory,rosenfeld2023dis2}."* Ben-David et al. (2010) — whose impossibility/hardness results for domain adaptation are the closest classical relative of `lem:nonid` — is cited inside a four-item list with no statement of what is different.

Checking the 57-entry bibliography (`grep bibitem`): there is **no** Lipton et al. BBSE, no Garg et al. label-shift estimation, no Ben-David & Urner "On the hardness of domain adaptation", no Scott, and no **Suitability Filter (2025)** — which the authors' own positioning file (`related_work_positioning.tex:73-79`) calls "adjacent". Selective prediction is a single citation (`geifman2017selective`); there is no learning-to-defer / Chow's rule / Cortes-DeSalvo-Mohri reject-option citation at all, despite abstention being one of three actions in the title contribution.

**Why it matters.** `lem:nonid` constructs two laws with *identical* unlabeled observables and opposite label-dependent functionals. That is structurally the same move as the standard label-shift/DA non-identifiability arguments, and any reviewer from that community will say "this is known" unless the paper pre-empts it. The differentiation is available and defensible — the localization to the disagreement region `D` and the parameterization by a declared `β` are genuinely the new bits — but the paper never makes the argument.

**Fix.** Add a paragraph: "Classical DA impossibility (Ben-David et al. 2010) and label-shift identification (Lipton et al. 2018; Garg et al. 2020) establish that target risk is not identifiable from unlabeled data without structural assumptions. `lem:nonid` differs in three specific ways: (i) the estimand is a *paired sign*, not a risk level; (ii) the ambiguity is localized to `D`, which shrinks the construction to a `2β`-wide band rather than the full risk range; (iii) the class is parameterized by a declared budget, converting an all-or-nothing impossibility into a frontier." Add the reject-option/learn-to-defer line to §II.D.

---

### [MAJOR] F5-10 — An observed `FA_u = 0.11 > α = 0.10` on a natural-shift track is disclosed in a sub-table and suppressed from the panel and the guarantee accounting

**Location:** `kbound_short.tex:868-895` (Table `tab:multiseed`) vs `:911` (panel) and `:969` (guarantee accounting).

**Evidence.** Table `tab:multiseed` SAR row: *"SAR | 0.041±0.017 | 0.000 | 0.065 | **0.11** | over-freezes (helpful-dom.)"*, with the caption stating "`FA_u` is the per-seed maximum". The main text (`:874-876`) describes it as "KGA over-freezes (worse than always-adapt, `FA_u` up to `0.11`)". The uniform panel row for Camelyon17 (`:911`) reports "`FA_u = 0`" with no candidate qualifier, and the guarantee-accounting table (`:969`) lists "`FA_u ≤ α` (unconditional false-adapt)" as a **theorem** with no note that an observed exceedance exists in the paper.

**Why it matters.** `FA_u ≤ α` is *the* deployable guarantee — the paper says so at `:112` ("the deployable safety guarantee"). An observed 0.11 at nominal 0.10 on a natural track is the most interesting empirical datum about that guarantee in the whole paper, and it is reported only as a parenthetical. A reviewer will read the panel's "`FA_u = 0`" as covering Camelyon17 and then find the SAR row.

**Fix.** Qualify the panel row as "Camelyon17 OOD (EATA candidate)" and add a footnote: "with the aggressive SAR candidate the per-seed maximum `FA_u` reaches 0.11; see Table `tab:multiseed`." Add a row to the guarantee-accounting table acknowledging the exceedance and whether it is within binomial noise at `n = 9` per seed (it is — but say so).

---

### [MAJOR] F5-11 — Venue fit: a 23-page, 26-table, 12-theorem IEEEtran conference submission

**Location:** `SUBMISSION_LEDGER.md:9`; `kbound_short.tex:1` (`\documentclass[conference]{IEEEtran}`).

**Evidence.** Ledger: *"PDF pages: **23**; long manuscript: 60 pages."* Counting the source: 26 `table`/`table*` environments (21 main + 5 appendix), 7 figures, 2 algorithms, and 12 theorem-class environments across the main text, `theory_core_main.tex`, and the appendix. The project docs call this "the short paper" (`SUBMISSION_LEDGER.md:6`) but the manuscript itself never uses that word.

**Why it matters.** IEEE conference tracks are typically 6–8 pages (10 with over-length fees); NeurIPS/ICML/ICLR main-track limits are 8–9 pages plus unlimited appendix, but the appendix here is bound into the same 23 pages via `\appendices`. At two-column IEEEtran density, 23 pages is roughly a 45-page single-column paper. This is a desk-reject risk at most venues and, independent of formatting rules, it is simply too much for a reviewer to hold in mind — which is *why* the internal contradictions in F5-1, F5-2 and F5-7 survived the authors' own multi-phase audit.

**Fix.** Decide the venue first, then cut to it. The 9-track panel does not need 26 tables: `tab:regime-summary`, `tab:data-access`, `tab:assumptions-role`, `tab:notation-main`, `tab:evidence-map`, `tab:failure-modes`, `tab:claim-status` and `tab:baseline-faithfulness` are all meta-tables about the paper rather than results, and at least five of them can merge or move to a supplement. If the target is TMLR (which the authors' own `VENUE_BENCHMARK_2026-06.md:60-63` identifies as the best fit), reformat single-column and the length problem largely dissolves.

---

### [MINOR] F5-12 — Aud-A…G are stated as four theorems with all proofs deferred to a manuscript reviewers cannot see, and are load-bearing for one sentence

**Location:** `kbound_short_appendix.tex:314-360`.

**Evidence.** *"The following results (**proofs in the long manuscript**, appendix 'Auditable drift budgets'…)"* — four `\begin{theorem}` environments (`thm:short-audA`, `audC`, `audDE`, `audG`) with no proofs and no citation a reviewer could follow. `kbound.tex` is a separate, uncited, un-submitted document. The only main-text consumer is one sentence in Limitations (`:1121-1125`) plus a pointer at `:458-459`.

**Why it matters.** Unprovable-as-submitted theorems are not reviewable, and a program committee will treat them as unrefereed. That is a shame, because Aud-A is the most substantive result in the paper (see F5-3) and Aud-G's exact feasibility floor `α ≥ 1/(K+1) + δ` is a genuinely useful practitioner-facing fact.

**Fix.** Promote **Aud-A only** into the main text with its proof (it is a corollary of `lem:nonid`'s construction and should be short), since it is the result that justifies the entire "declared, not measured" design. Cut Aud-C/D/E/G to a single cited remark, or supply their proofs.

---

### [MINOR] F5-13 — The abstract uses `M` and `β` without definition and asserts "the only sound action" without naming the convention that makes it so

**Location:** `kbound_short.tex:41`.

**Evidence.** *"…a strict adapt-or-freeze commitment is uniformly sound if and only if the population evidence margin satisfies `|M|>β`; inside the closed band, matched evidence or zero-benefit ambiguity makes abstention **the only sound action**."* Neither `M` nor `β` is defined in the abstract; a reader cannot evaluate the iff. And "the only sound action" is true only under `def:strict-sound`, which the paper itself flags as "an epistemic validity convention, **stronger than zero regret at the boundary**" (`theory_setup.tex:72-74`) — at `Δ = 0`, adapt and freeze are risk-equivalent, so the claim is about certification semantics, not about loss.

**Why it matters.** An abstract that promises an iff over undefined symbols reads as more precise than it is, and "the only sound action" will be read as "the only *correct* action", which is false at the boundary.

**Fix.** Replace symbols with words ("…if and only if the label-free evidence margin exceeds the declared drift budget") and add four words: "…the only action that certifies a *strict* benefit direction."

---

### [MINOR] F5-14 — Twelve tables apply `\resizebox{\columnwidth}{!}` on top of `\scriptsize`/`\footnotesize`

**Location:** `kbound_short.tex:124, 184, 204, 544, 645, 709, 757, 818, 884, 963` and `kbound_short_appendix.tex` (2 more).

**Evidence.** e.g. `:201-204`: `\begin{table}[t]\centering\scriptsize` immediately followed by `\resizebox{\columnwidth}{!}{%`. Applying `\resizebox` to a `\scriptsize` tabular that is already wider than a column scales the glyphs *below* `\scriptsize`.

**Why it matters.** IEEE (and ACM) camera-ready checks reject arbitrary text scaling below the specified minimum point size, and it is an accessibility problem regardless. Table `tab:uniform-panel` (`:897`) is `\scriptsize` in a `table*` with five prose-heavy columns; Table `tab:assumptions-role` is `\scriptsize` + resized. I could not measure the rendered sizes (PDFs are stripped from this copy), so I flag the construction rather than a specific point size.

**Fix.** Remove `\resizebox` from every table and instead cut columns or abbreviate cell text. This will also force the useful discipline of F5-11.

---

### [NIT] F5-15 — Two of seven figures are conceptual schematics with no measured data

**Location:** `kbound_short.tex:263-270` (`fig:frontier`), `:582-591` (`fig:regime-map`, a full-width `figure*`).

**Evidence.** `fig:regime-map`'s own caption concedes it: *"This is a \emph{conceptual schematic}: axis positions are illustrative regime categories, not measured coordinates."* It occupies a full text width in a 23-page paper. `fig:frontier` is likewise a schematic superseded by the measured `fig:frontier-measured` two sections later.

**Fix.** Cut `fig:regime-map` (its content is Table `tab:regime-summary`) and merge `fig:frontier` into `fig:frontier-measured` as an inset.

---

## What I checked and could NOT fault

These are the things I actively tried to break and could not. I list them so the above findings can be read as targeted rather than indiscriminate.

1. **The ImageNet-C SAR *pooled* headline number is exactly right.** I replayed the decision rule from `experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json` under `ε = ρ_(k)`, `k = ⌈(n+1)(1−α)⌉` per seed and reproduced the manifest exactly: regret `0.0264 / 0.0529 / 0.0319`, `FA_u = 0/135`, `abstain = 109`. `paper/generated/kbound_result_manifest.json#tracks/imagenetc_sar` is accurate; the problem in F5-2 is the *appendix* table, not the manifest.
2. **The CIFAR-10-C panel numbers are consistent across all four places they appear.** `paper/generated/kbound_numbers.tex` (`\CIFARtentKga 0.0016`, `\CIFAReataKga 0.0013`), Table `tab:decisive`, Table `tab:uniform-panel:909`, Table `tab:primary-numeric:948-949`, and `SUBMISSION_LEDGER.md:65` all agree to four decimals. The Phase-7 fix log (`SUBMISSION_LEDGER.md:130-131`) claims the fourth decimals were corrected; they actually were.
3. **`ε` is never conflated with `β`.** Checked the abstract, `tab:notation-main`, `rem:four-quantities`, §"Population Frontier versus Empirical Certificate", `tab:claim-status`, and the guarantee-accounting table. The distinction is stated correctly in all six, and the guarantee table explicitly marks the conflation as **false**.
4. **The withdrawn Camelyon17 "beats both" wording is genuinely gone.** `grep -i "beats both.*camelyon\|camelyon.*beats both"` over both manuscript files returns nothing, as `SUBMISSION_LEDGER.md:111-114` claims. The manuscript states only "reconciled no-harm" for Camelyon17.
5. **`lem:reduction` and `thm:certificate` are correct as written.** The reduction algebra (`theory_core_main.tex:12-23`) checks out, and the certificate proof (`:164-168`) is a valid — if trivial — consequence of the stated coverage hypothesis. `rem:fa-marginal` correctly refuses to extend it to `FA_c`.
6. **The cross-adapter-transfer ablation is an honest self-attack.** Table `tab:abl-transfer` reports `FA_u` up to 0.255 for SAR→Tent, i.e. the authors publish the configuration in which their own guarantee breaks, and use it to justify per-adapter refitting. That is the right instinct and most authors would have omitted it.
7. **The label-requiring streaming result is correctly demoted** (`:296-302`), and CIFAR-10.1's `FA_u = 0.167` failure is retained in the panel rather than dropped. Both cost the authors something and both were the right call.

---

## Open questions for the author

1. **Name one deployment decision, anywhere in the nine tracks, that the population frontier `|M| > β` actually determined.** The paper says `β` is never numerically supplied to KGA on real data (`:596-598`). If the answer is "none", then the theorem is a framing device rather than a method — in which case why is it the abstract's first sentence of contribution, and what would change in any table if Theorems 1–3 were deleted?

2. **Which conformal rule produced Table VIII, and what is the paper's actual per-seed ImageNet-C claim?** Under the exact-rank rule the main text says it used, per-seed beats-both is 2/5 with three *exact ties* to always-freeze and pooled `FA_u = 0/135`; under the interpolated rule it is 5/5 with `FA_u = 1/135` (seed 2 = 0.037). The paper currently takes the aggregate from one rule and the per-seed support from the other. Which is the claim?

3. **How is the abstract's "on every natural distribution shift … including rendition shifts … KGA matches the better fixed policy" defensible** given that the panel reports ImageNet-R at 1.75× and PACS at 2.45× the better fixed policy's regret, and CIFAR-10.1 at `FA_u = 0.167` above the nominal `α = 0.10`? If "natural" is meant to exclude PACS, ImageNet-R and CIFAR-10.1, please state that exclusion in the abstract and explain why "rendition" appears in the enumerated list.

4. *(Secondary, but I would ask it.)* **What is the operational definition of "detectable"?** Given ImageNet-C Tent is 56% harmful — the most mixed regime in the paper — and KGA ties always-freeze there, what ex-ante criterion would have predicted that before the labels were opened?
