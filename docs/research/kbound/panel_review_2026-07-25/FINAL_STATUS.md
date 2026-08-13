# FINAL STATUS — build, adversarial verification, verdict

Written by the final build/verify slice. Every number below was recomputed in this container from
the named artifact or read out of a build log produced in this session. Nothing is quoted from the
integrator's report without independent re-derivation; where I re-derived a number the recomputation
is shown.

---

## 1. Build status — final

All three drivers built from a deleted `.aux`, four `pdflatex` passes each. No bibtex pass is used
(references are `\input` as a manual `thebibliography`, not a `.bib`).

| driver | output | pages | `!` errors | undefined `\ref` | undefined `\cite` | multiply-defined |
|---|---|---:|---:|---:|---:|---:|
| `kbound_tmlr.tex` | `/home/claude/kb/docs/research/kbound/kbound_tmlr.pdf` | **85** | **0** | **0** | **0** | **0** |
| `kbound_short.tex` | `/home/claude/kb/docs/research/kbound/kbound_short.pdf` | **53** | **0** | **0** | **0** | **0** |
| `kbound.tex` | `/home/claude/kb/docs/research/kbound/kbound.pdf` | **50** | **0** | **0** | **0** | **0** |

All four required counters are zero in all three builds. Page counts are unchanged by the three
fixes I applied in §5 (85 / 53 / 50 both before and after), so the corrections cost nothing.

Build logs: `/home/claude/kb_fixes/build_logs_final/{tmlr,short,long}_r{1..4}.log`.

`tmlr.sty` is still absent from the container and from the tree; the TMLR driver builds under its
documented `article` shim. That is a formatting risk at submission, not a content risk — see §7.

---

## 2. Verification 1 — no fabricated result

**Passes.**

Method: enumerate every theorem-like label in each driver's `.aux`, resolve it to its printed number
and page, and trace it to a `proof_status` row in `THEORY_BETA_ESTIMABLE.md` §2 (B1–B9) or
`THEORY_BETA_IMPOSSIBLE.md` §2, or to a pre-existing proved result.

- 63 theorem-like labels resolve in `kbound_tmlr.aux`, 61 in `kbound_short.aux`, 81 in `kbound.aux`
  (including `def:`/`rem:`/`ass:` and the deliberate triple-labelled aliases
  `thm:headline`=`thm:frontier`, `thm:certificate`=`thm:cert`,
  `thm:short-audA`=`thm:aud-A`=`cor:audA`).
- Every load-bearing spine result maps to **PROVED**: `lem:fibre`, `thm:beta-minimax` (a)–(d),
  `cor:audA`, `cor:beta-is-beta`, `thm:anchor`, `prop:threeterm`, `thm:dichotomy`, and the whole
  episode block `epi:thm-ident` (B1), `epi:thm-conformal` (B2), `epi:prop-escape` (B3),
  `epi:cor-bracket`/`epi:thm-floor` (B4), `epi:prop-labels` (B5), `epi:prop-probe` (B6),
  `epi:thm-shift` (B7), `epi:cor-not-free` (B8).
- `thm:lecam` is the one **PROVED-MODULO-STANDARD-RESULT** (Pinsker and `KL ≤ χ²` for Bernoulli,
  both conditions checked in its proof). It is labelled as such in the memo and its proof states the
  dependence.
- The single **CONJECTURE**-status result, B9 / `epi:conj-open`, is a `conjecture` environment in
  **all three** PDFs, and I confirmed the honest sentence survives in each:
  *"This is stated as a conjecture, not a theorem: we neither exhibit such an assumption nor prove
  that none exists."* (`kbound_tmlr.pdf` Conjecture 1; `kbound_short.pdf` Conjecture 1;
  `kbound.pdf` Conjecture 2.)
- The long manuscript's two proof-sketch-only theorems (`thm:conj1-dichotomy`, `thm:ev-rate`) still
  carry the D4 *Proof status* notes including *"a validator is not a proof"*. Verified in the
  rebuilt `kbound.pdf`.

No theorem-like environment in any PDF states a result whose status is weaker than its presentation.

---

## 3. Verification 2 — the necessity claim against the experiments

I re-derived every necessity number in `paper/sections/necessity_observed.tex` from
`necessity_nn_results.json` / `necessity_ceiling_results.json`, and re-ran one computation that is
in neither file. **Every number reproduces.** The verdict the section states matches what the two
scripts returned.

### 3.1 Rate table (`tab:necessity-rate`) — observed *and* null, both checked

| row | pairs | opp. | observed | null [95% CI] | source key | ✓ |
|---|---:|---:|---:|---|---|---|
| CIFAR-10-C cell q=0.25 | 30,555 | 13 | 0.00043 | 0.187 [0.176, 0.199] | `by_quantile.0.25.strict` | ✓ |
| CIFAR-10-C cell q=0.95 | 252,277 | 1,194 | 0.00473 | 0.185 [0.178, 0.193] | `by_quantile.0.95.strict` | ✓ |
| CIFAR-10-C law | 1,172 | 0 | 0.000 | 0.164 [0.138, 0.190] | `group_level` + `law_level_full.permutation_null` | ✓ |
| ImageNet-C cell q=0.25 | 867 | 62 | 0.0715 | 0.250 [0.208, 0.288] | `by_quantile.0.25.strict` | ✓ |
| ImageNet-C cell q=0.95 | 9,901 | 1,588 | 0.1604 | 0.240 [0.225, 0.256] | `by_quantile.0.95.strict` | ✓ |
| ImageNet-C law | 169 | 3 | 0.0178 | 0.218 [0.156, 0.274] | `group_level` + `permutation_null` | ✓ |

`p_below = 0.0004997` (the 2,000-permutation floor) in every row, as the caption says. Random-pair
controls 0.27225 → `0.2723` and 0.22315 → `0.2232` ✓.

**The claimed rate-over-null direction is verified in both directions.** The text says disagreement
is *below* chance by 39× (CIFAR-10-C) and 1.5× (ImageNet-C): 0.185/0.004733 = **39.1** ✓,
0.240/0.16039 = **1.50** ✓. The 39× is quoted from the q=0.95 row, which is the *weakest* of the
available ratios (q=0.25 gives 440×) — the conservative choice, correctly made.

### 3.2 Headline pair — every digit

`hotelling_d2` 2.33630 → `2.34` ✓ · `hotelling_p_value` 0.98497 → `0.985` ✓ ·
`max_abs_per_coordinate_z` 1.00359 → *"no coordinate discrepancy exceeding **1.01** standard
errors"* ✓ (correctly rounded **up**, not down) · `equivalence_lambda_upper95` 3.1115e-61 →
`3×10⁻⁶¹` ✓ (quoted as its actual value, not a rounded 0.00) · re-splits: C(10,5)=252,
`Z_frac_resplits_with_larger_d2` 0.992063 → `99.2%` ✓, median 6.8855 → `6.89` ✓ ·
`B_mannwhitney_U` 0.0, `p_exact` 0.0079365 → `U=0, p=0.0079` ✓ · `B_mean` −0.05330 / +0.03500 with
CIs [−0.08106, −0.02554] / [+0.02344, +0.04656] → `−5.33 [−8.11, −2.55]` and `+3.50 [+2.34, +4.66]`
accuracy points ✓ · `a0_mean` 0.68160 / 0.44080 → `0.682 versus 0.441`, difference 0.2408 → *"24
points of base accuracy"* ✓.

### 3.3 CIFAR-10-C witness pair — every digit

`by_quantile.0.25.top_pairs[0]`, `tent`/`tent`,
`jpeg_compression|s5|large_iid|iid|aggressive|r0` vs `…|single_class|…`: distance 1.66156 → `1.66`
✓, q=0.25 pair floor 2.63759 → `2.64` ✓, `frac_same_law_pairs_farther_apart` 0.97778 → `97.8%` ✓,
`a0` 0.7290 / 0.7295 ✓, `B_group_mean` +0.01580 / −0.01635 → `+1.58` / `−1.64` ✓ with CIs
[+0.82, +2.34] / [−2.82, −0.45] ✓, `zone` = `BLIND` on both and `kga_decision` = `ABSTAIN` on both
✓. The pair does differ only in `iid` vs `single_class` — same method, same corruption, same
severity, as claimed ✓.

### 3.4 Ceiling / band / threat numbers

AUC 0.98297→`0.983`, acc 0.93133→`0.931`, majority 0.74043→`0.740` ✓; LOCO held-out 0.97031 /
0.90957 → `0.970 / 0.910` ✓; permuted null AUC 0.50413→`0.504` ✓; 4,949 evaluable ✓; certified
ambiguous 0.29400→`29.4%` ✓ and 0.20226→`20.2%` inside [0.25,0.75] ✓; median 135 neighbours from 28
distinct conditions ✓; permuted null 0.85434 ± 0.00876 → `85.4 ± 0.9%` ✓; radius sensitivity
0.3248 / 0.2940 / 0.3378 and strict variant 0.2941 → *"moves only between 0.29 and 0.34"* ✓;
`band_B_median` 0.013000→`0.0130` ✓, `band_gamma_median` 0.010832→`0.0108` ✓, `gamma_global.q90_abs`
0.015487→`0.0155` ✓, `gamma_within_fibre_share` 0.51910→`0.519` ✓; `yield_ceiling_upper` 0.70600
→`0.706` ✓, lower 0.54274→`0.543` ✓. Screening 209,628 / 3,240 law pairs ✓; provenance 229 / 94
law pairs with 0 / 3 surviving replicate-averaging ✓; method splits eata 884 / tent 310 / sar 0 and
tent 1,586 / sar 2 / eata 0 ✓; 365 pure-distribution-shift ImageNet-C pairs ✓; weaker pairs
`d2` 12.8353 / 15.2789 → `12.8` / `15.3` ✓. The nine-vs-eleven disclosure is real:
`Z_exact_redundancy_max_residual` is **exactly 0.0** for both `entropy_drop` and `pbal_drop` ✓.

### 3.5 Two numbers not in either JSON — recomputed from raw records

Both by re-running `load` → `parse` → `standardize` → `noise_floor` → `NbrStruct` from
`necessity_ceiling.py` on the raw per-condition files in this session:

- **ImageNet-C selection.** `r* = 0.3945`; evaluable **213/405 = 52.6%**; `Pr(B>0)` = **0.9765**
  among evaluable vs **0.5104** among the 192 non-evaluable. The manuscript's `52.6%`, `97.7%`,
  `51.0%` are correct, and the integrator's recomputation is independently confirmed.
- **CIFAR-10-C selection.** `r* = 0.3734`; 4,949 evaluable; mean `|Δ|` **0.1163** evaluable vs
  **0.1974** non-evaluable. This one was *not* in the manuscript before this session — see defect
  **F3** in §5.

### 3.6 Episode numbers the necessity table imports

Recomputed from `frontier_sweep_v1/beta_estimability/episode_beta_results.json` (470 rows):
`cifar10c|M_gbm` has exactly **135** splits with mean `beta_conformal` **0.018972** → `0.0190` ✓,
range 0.0109–0.0264 ✓, per-protocol means 0.0176–0.0196 ✓. Ratio 0.018972/0.010832 = **1.7515** →
`1.75` ✓. Yield: per-protocol means span **0.5943–0.6569** → `0.594–0.657` ✓.

Also re-verified because the abstract carries them: median relative conformal-vs-plug-in difference
over all 470 splits = **0.3433%** → `0.34%` ✓ with median coverage difference **exactly 0.0** ✓;
`k_sweep` shows `frac_feasible` 0.0 at K=3 and K=5 and 1.0 at K=9 → *"first feasible at exactly
K=9"* ✓.

### 3.7 Do the two methods disagree, and does the paper say so?

**Yes and yes.** Both returned PARTIAL. They disagree on which benchmark carries the evidence: the
pair slice's population witnesses are ImageNet-C-only, the neighbourhood slice's certified ambiguity
is CIFAR-10-C-only. The section states the disagreement explicitly, rules on it, and gives the
reason for the asymmetry — the ImageNet-C zero is discounted as a coverage limitation (verified in
§3.5 above) and its β ratio of 16.8 is quoted nowhere in any PDF (checked); the CIFAR-10-C zero is
credited as a measurement (all 229 witness law pairs separate under averaging — verified). The
structural finding that follows is carried into the spine with its caveat sharpened rather than
over-read. **The paper says so.**

---

## 4. Verification 3 — the four disclosures in *both* abstracts (D2 regression)

**Passes, and the structural cause of D2 is now closed.** The abstracts are factored:
`kbound_abstract_core.tex` (paragraphs 1–4, shared by all three drivers) and
`kbound_abstract_disclosures.tex` (the weak-zeros caveat as a mid-sentence clause, `\input` by both
closing paragraphs). `kbound.tex` no longer duplicates the shared text; it `\input`s both and adds
only its own POEM/AETTA/PACS closing paragraph. The residual the previous slice flagged and did not
fix is fixed.

Verified in the **rendered PDFs**, not the sources, on all three builds (the first pass of this
check produced false negatives from two-column `pdftotext` column interleaving; re-run against
`-raw` extraction of the abstract pages):

| disclosure | `kbound_tmlr.pdf` | `kbound_short.pdf` | `kbound.pdf` |
|---|:--:|:--:|:--:|
| weak-zeros caveat ("three of those zeros are weak") | ✓ | ✓ | ✓ |
| iWildCam record absent → "sealed summary rather than a reproduced result" | ✓ | ✓ | ✓ |
| frontier retraction ("we withdraw the operational reading of the frontier") | ✓ | ✓ | ✓ |
| episode-marginal caveat ("marginal across deployments, not conditional on the one in hand") | ✓ | ✓ | ✓ |
| LOCO EATA negative ("loses to it on all five EATA runs") | ✓ | ✓ | ✓ |
| "half the severity splits admitting no budget at any declared level" | ✓ | ✓ | ✓ |

---

## 5. Verification 4 — nothing lost in the cut

**Passes on results; three under-disclosures found and fixed.**

Method: the pre-cut PDFs are recoverable from `/home/claude/kb_fixes/impossibility_spine.tar.gz`
(87 / 52 / 49 pp). I extracted them and diffed rendered text against the new builds, at two
granularities — the set of named theorem-like environments, and the set of sentences.

**Theorem/result inventory — identical, all three builds.**

| build | pre | post | lost | gained |
|---|---:|---:|---|---|
| `kbound_tmlr` | 43 named environments | 43 | **none** | none |
| `kbound_short` | 43 | 43 | **none** | none |
| `kbound` | 65 | 65 | **none** | none |

**Disclosures and negative findings — none lost.** I checked every candidate absent sentence.
Every apparent loss was a `pdftotext` reflow or float-interleave artifact; the one that looked real
was not. Specifically:

- *"We report FA_u = P[adapt, Δ ≤ 0] and do not interpret FA_c as a certificate"* appeared to have
  vanished from `kbound_tmlr.pdf`. It has not — a table floated between the two halves of the
  sentence. Confirmed present in the rebuilt PDF.
- SAR-as-negative-and-withheld, the one-radius rule, the 0.0289/0.0122 exact-rank discrepancy, the
  labelled 0.90 null, the uniform-no-harm retraction, the Camelyon17 no-harmful-cell disclosure, the
  SAR quarantine, and the seed-0 heterogeneity all survive in the builds that carried them before.
- The long manuscript never carried the *verbatim* "only radius rule used anywhere in the paper"
  sentence, before or after the cut; it carries the equivalent disclosure in its own wording
  (*"0.0112 under this manuscript's interpolated radius, 0.0151 under the exact-rank rule"*). This
  is a pre-existing wording gap, not a cut casualty. Noted in §7.

### Defects I found in the new material, and fixed

| # | defect | severity | action |
|---|---|---|---|
| **F1** | `tab:necessity-band`'s row *"Episode-conformal β̂, mean over 135 splits — 0.0190"* dropped the benefit-model qualifier that its source memo carries. The value is `M_gbm`-specific; the same estimator on the **same 135 splits** with the alternative confidence-based `M` returns **0.0536**, making the headline "three unrelated routes agree to a factor of 1.75" a factor of **4.9** instead. Reporting only the favourable model overstates the agreement. | **medium — makes a convergence look more robust than it is** | **Fixed.** Row now reads "(gradient-boosted `M`)"; caption now states *"The agreement is conditional on the benefit model"*, gives the 0.0536 / factor-4.9 counter-number, and explains why (γ = B − M(Z) is defined relative to M, so a weaker M leaves a larger residual). Recomputed from source: `M_atc4` mean over the same 135 splits = 0.053561, ratio 4.944. |
| **F2** | *"the deployed certificate's measured yield of 0.594–0.657 sits inside that bracket, near its top"* omitted the source memo's own caveat — *"puts my neighbourhood-derived bracket next to a yield measured under a different splitting protocol; it is indicative, not a matched comparison"* — and presented a range of **per-protocol means** as if it were the range of splits (raw per-split range is 0.246–0.987). | **medium — an unmatched comparison read as a confirmation** | **Fixed.** The sentence now says "with the same benefit model", carries the memo's indicative-not-matched caveat verbatim in substance, and gives the raw split range. |
| **F3** | *"29.4% of the 4,949 evaluable cells (**22.5% of all 6,480**)"* quoted the all-cells restatement with no note that it counts every non-evaluable cell as unambiguous — while the same section carefully discounts the *ImageNet-C* evaluability selection. The CIFAR-10-C evaluable subset is also selected (mean \|Δ\| 0.116 vs 0.197), which the memo records under "not addressed / open". | **medium — asymmetric handling of the same threat** | **Fixed.** 22.5% moved out of the headline sentence into its own clause that states it is a conservative convention rather than a measurement, quotes the 0.116/0.197 selection I recomputed in §3.5, and names 29.4%-over-evaluable as the claim actually made. Checked that no other site in any manuscript quotes 22.5% (the guarantee-accounting row in `kbound_short_appendix.tex` already said "of evaluable"). |

All three fixes are in the shared `paper/sections/necessity_observed.tex`, so they land in all three
manuscripts. Verified rendered in all three rebuilt PDFs. **Page counts did not move.**

---

## 6. Verification 5 — standing regressions

| regression | status |
|---|---|
| Retracted uniform-no-harm claim stays dead | **Clear.** The string "uniformly no-harm" occurs in each PDF only inside explicit retraction framing — TMLR/short: *"the clearest counterexample in the paper to a blanket 'uniformly no-harm' reading, and it comes from an artifact we ourselves promote"*; long: *"It is **not** uniformly no-harm across the panel"*. No affirmative use anywhere. |
| One radius rule | **Intact** in TMLR and short (*"the only radius rule used anywhere in the paper"*), with the 0.0289 vs 0.0122 interpolated/exact-rank discrepancy disclosed. The long manuscript carries the discrepancy but not the uniqueness sentence — unchanged from before the cut. |
| The documented 0.90 null is never read as evidence | **Clear.** Every coverage figure at or near 0.900 is accompanied by its label; Table 6's caption still reads *"rows at or near 0.900 are evidence the estimator is not broken, not evidence that it works"*, and the ImageNet-C 0.896–0.985 band is still said to *"sit at or above the null and carry almost no information"*. The abstract's 0.905 exchangeable-control figure is presented as the **control** against which 0.626 and 0.454 are the finding, which is the correct reading. |
| Necessity section does not read the below-chance result as support | **Clear.** The section states the direction went against the excess-over-chance test and calls it *"a statement that Z is informative, not that it is sufficient"*, and the introductions carry the 39× rarity so no reader meets the existence claim without it. |

---

## 7. What the necessity experiment established, honestly

**It established existence and refuted prevalence, and the refutation is the more interesting half.**

- The `lem:nonid` configuration **is** observed in logged deployments. Three population-level
  ImageNet-C witnesses survive replicate-averaging, one of them with a genuine equivalence result
  (λ upper bound 3×10⁻⁶¹, exact 252-fold re-split at 99.2%, exact Mann–Whitney U=0) rather than a
  failure to reject; 1,194 CIFAR-10-C and 1,588 ImageNet-C cell-level matched pairs disagree in
  benefit sign; and 29.4% of evaluable CIFAR-10-C cells sit in fibres certified ambiguous against a
  permuted-label null of 85.4%.
- It **also** established that matched-evidence disagreement is 39× *below* chance on CIFAR-10-C.
  `Z` is highly informative about the sign of Δ (AUC 0.983, 0.970 held out across corruption
  families) and still not sufficient to determine it. The section reports this as a bound on the
  ambiguity, which is the right framing and the narrow claim.
- The ambiguity's **width** independently lands where the spine says it should: 0.0108–0.0155 from
  evidence geometry, 0.0190 from an episode-conformal order statistic over labelled history that
  shares no estimator with it. That is `cor:beta-is-beta` as a measurement — but only for the
  benefit model the paper deploys (F1), and the section now says so.
- What it did **not** establish: prevalence, generality beyond this logged evidence vector, or
  anything about ImageNet-C fibres (under-powered, and the evaluable subset is selected 97.7% vs
  51.0% in the adverse direction). The section says all three.

**Did the necessity result land?** Yes, but as a *calibration* of the impossibility rather than a
second proof of it. Its real value to the paper is that it converts "a referee may say your
worst case never happens" from an unanswered objection into a measured answer with a number
attached — and the number is unflattering to the naive reading, which the paper reports rather than
buries. Its structural by-product (non-identifiability dissolves under a population of replicates on
CIFAR-10-C but not on ImageNet-C) is the spine's own architecture appearing in the data, and the
caveat that replicate-averaging is not population-averaging is correctly sharpened rather than
over-read.

## 8. What was cut

Net effect over this session's two slices: **87→85 (TMLR), 52→53 (short), 49→50 (long)**. The
necessity section costs 4 pp in TMLR and 3 pp in each of the others; gross cuts were 6 / 3 / 4 pp.
No theorem, no disclosure, and no negative finding was removed — verified by the pre/post PDF diff
in §5, which is a stronger check than a source diff because it sees what the reader sees. The
integrator did not hit "cut hard" on the two-column and long builds, and the manuscripts say so.

---

## 9. What remains open

Specific, and none of it is closed by this session.

1. **`tmlr.sty` is absent.** The TMLR driver builds under an `article` shim. The 85-page count is
   therefore an estimate under the wrong class file, and TMLR's own template will change it. This is
   the single cheapest remaining item and it is blocking an accurate length statement. Obtain
   `tmlr.sty`, rebuild, and re-measure before claiming any page count to a reviewer.
2. **Camelyon17 reconciliation.** `camelyon_reconciliation/` is absent and not returning. The
   promoted row *is* recomputable from `results/camelyon17_richZ_F_v1/_partial.json`, and
   `FINAL_CLOSURE.md` §3 records that the labelling is reversed relative to what the paper once
   said. The abstract's weak-zeros clause now covers it honestly ("promoted subset contains no
   harmful cell"), but the underlying row remains a sealed summary rather than a reproduced result.
3. **The edge capture never ran.** `publication_gate.json` records `"passed": false` with 0 of 632
   expected clips observed in every session; `edge_real_phone_v1/calibration_summary.json` and
   `split_audit.json` are outputs that do not exist. Two tests skip for this reason, correctly. The
   21 recovered edge artifacts describe the *protocol*, not results.
4. **The SAR official-settings ImageNet-C control is unrun.** The shipped operating point is
   lr 4e-3, 16× SAR's published learning rate, and the paper concedes it chose the regime in which
   SAR collapses. Until the control at lr 2.5e-4 with `layer4` frozen runs (~1 day), "KGA helps SAR"
   means "KGA helps SAR at a setting we chose because SAR fails there." This is the highest-value
   cheap experiment remaining and the paper's most attackable empirical claim.
5. **Six figure assets are still missing** and render as the honest "FIGURE ASSET NOT PRESENT IN
   THIS BUILD" placeholder (6 in short and TMLR, 16 in the long build). Every quantity they would
   display is in the text and tables, but a submission with 6 placeholder boxes is not a submission.
6. **LOCO for EATA was never run**, which is the arm whose cluster-robust interval fails. The paper
   says so rather than implying coverage.
7. **The dated re-freeze in `SUBMISSION_LEDGER.md` §0 has not been executed**; it correctly reads
   NOT FROZEN.
8. **The long manuscript lacks the "only radius rule" uniqueness sentence** that TMLR and short
   carry. Pre-existing, low severity, one sentence to fix.

---

## 10. Score estimate

**7.4–7.7.** Against the anchors 4.2 → 6.8 → 7.3–7.6, this session buys roughly **+0.1**, and I want
to be precise that it is small.

**Why it moved at all.** The necessity result is a real addition and it is the specific thing a
referee would have asked for: the impossibility spine's opening move is a *constructed* pair, and
the obvious attack is that the construction is a worst case practice never visits. That attack is
now answered with measurement rather than assertion, in two analyses that share no estimator, on
6,885 real deployment cells, with the direction of the surprise reported against the authors' own
interest. The three-route convergence on the band width (0.0108 / 0.0155 / 0.0190) is genuinely
striking and it is exactly `cor:beta-is-beta` showing up as a number. The abstract de-duplication is
worth something too — not intellectually, but D2 was a live overclaim risk and it is now
structurally impossible rather than merely fixed.

**Why it did not move more.** The necessity result is a *calibration*, not a new theorem: it cannot
bound what any label-free statistic can do, only what this one does, and the section says so. Its
strongest single finding — 39× below chance — argues that the paper's evidence vector works well,
which is good news for the method and mildly deflationary for the impossibility narrative; the paper
handles this with the right framing but a referee will still notice. The ImageNet-C half is
under-powered and correctly discounted, which means the population-witness claim and the
certified-ambiguity claim rest on disjoint benchmarks — a genuine weakness the section names but
cannot remove. And the three defects I fixed in §5 were all in the same direction (an agreement
reported without the qualifier that weakens it), which is the failure mode a fast integration
produces; that they existed at all is a small mark against the new material even though they are now
gone.

**Why not lower.** Nothing in this session found a fabricated number. I re-derived every necessity
figure from raw JSON, re-ran two computations from raw records that appear in no result file, and
re-checked the abstract numbers, the episode K-sweep, and the 470-split conformal comparison. All of
it reproduced, including the two figures I recomputed specifically because the memo asserted them
without an artifact. The rounding is honest in the one place it could have been shaded — 1.0036
standard errors is reported as "not exceeding 1.01", rounded up — and a 3×10⁻⁶¹ equivalence bound is
quoted at its actual value rather than as 0.00.

**The ceiling is not here.** What caps this paper at high sevens is not the theory and not the
honesty; it is that the empirical panel is one benchmark family with a one-sided natural-track
supplement, and no restructuring or measurement of the existing artifacts changes that. Item 4 in §9
(the SAR control) and a LOCO run for EATA are the two experiments that would move it, and both are
cheap. If the intent is to go past 8, that is where the next day goes — not into the manuscript.
