# Rewrite report — short paper and shared inputs (impossibility-spine restructure)

**Scope executed:** `kbound_abstract.tex`, `kbound_short_body.tex`, `kbound_short_appendix.tex`,
`paper/sections/**`. Drivers untouched (no content lives there). `kbound.tex` and `manuscript/**`
NOT touched — verified no file I own is `\input` by them.

**Build state (both drivers, three passes each, from clean `.aux`):**

| driver | pages | undefined refs | undefined citations | undefined control seqs |
|---|---:|---:|---:|---:|
| `kbound_tmlr.tex` | 90 | **0** | **0** | **0** |
| `kbound_short.tex` (IEEEtran) | 53 | **0** | **0** | **0** |

The six `figures/fig_*.png` "not found" errors are byte-identical to the pre-restructure baseline;
pre-existing missing assets, not caused by this rewrite.

---

## 1. Length: the plan's estimate was wrong and I am reporting the real number

| | plan predicted | **measured** | baseline |
|---|---:|---:|---:|
| main body | ~38 pp | **54 pp** | 46 pp |
| appendix | ~45 pp | **36 pp** | 28 pp |
| total | ~83 pp | **90 pp** | 74 pp |

**The main body grew by 8 pages instead of shrinking by 8.** Cause, measured from the `.aux`: the
promoted theory costs 13 pp in single column (Sec. 4 = pp. 11-17, Sec. 6 = pp. 23-28), not the
7.2 pp the plan's per-theorem table budgeted. Every cut and move in the plan was executed and they
recovered roughly what the plan said; the promotion cost nearly double. I did not cut further to
hit the number, because the only remaining levers were results tables and disclosures.

The plan's recommendation stands and hardens: `kbound_short.tex` at 53 IEEEtran pages is not a
submission target. Kept as a formatting check only.

---

## 2. Abstract — old vs new

| | old | new |
|---|---|---|
| length | ~950 words, one paragraph | ~640 words, five paragraphs |
| opening claim | "We recast TTA as a safety decision" | "Deciding whether to adapt is possible only relative to a declared drift budget beta — and it cannot come from the deployment" |
| theory sold | matched-evidence construction; audit vacuity as an aside | exact minimax Gamma_z; Gamma_z(C_beta)=beta; yield <= delta; anchor destruction; one lemma at three radii |
| beta-sweep | "we report a negative result about our own parameter" | same numbers, now entailed by thm:anchor, plus the rebuttal-killer (median 0.34% over 470 splits) |
| episode result | absent | new paragraph: identification, conformal estimator, factor-e bracket, K=9, E2 measured failing (0.905 -> 0.626/0.454, weighted conformal 1.000 at yield 0.000) |
| LOCO stress | absent | radius 4.3x-6.9x, regret 3.4x-7.6x, FA_u = 0 in 10/10 — and the disclosure that beats-both does NOT survive it |

**Disclosures I put back that the architect's draft abstract had dropped** (a restructure never
removes a disclosure): the three weak natural-shift zeros (Camelyon17 has no harmful cell;
RxRx1/iWildCam make 0 and 1 adapt decisions), the absent iWildCam record file, the PACS /
ImageNet-R / CIFAR-10.1 nulls, and "we withdraw the operational reading of the frontier; the proofs
are untouched".

---

## 3. Contributions — 8 items -> 6

Replaced verbatim with plan Sec. 2, with two label corrections forced by the short build
(`thm:short-audA` -> `Corollary~\ref{cor:audA}`; `epi:cor-bracket` added to the bracket cite).

| old item | fate |
|---|---|
| 1. TTA as a safety decision | absorbed into Intro para 1. Framing, not a result. |
| 2. matched-evidence + audit vacuity | split into new 1 (one lemma, three radii) and new 2 (exact minimax) |
| 3. bookkeeping decomposition | demoted; rem:gamma-residual already concedes it is a definition |
| 4. negative result about beta | new 4, reframed from confession to confirmed prediction of thm:anchor, with the quantitative check (0.2376 predicted vs 0.2468-0.2606 measured) and the failed prediction kept |
| 5, 6, 8 | merged into new 6 |
| 7. KGA wrapper | cut as a contribution; one sentence in Sec. 7 |
| — | new 3 (labels at the wrong domain buy nothing; three currencies) — entirely new |
| — | new 5 (where a budget can come from, priced) — entirely new |

---

## 4. Every claim that changed role

| claim | old role | new role |
|---|---|---|
| thm:short-audA (Aud-A) | appendix theorem | MAIN BODY Sec. 4.3 as Corollary cor:audA of thm:beta-minimax, triple-labelled cor:audA = thm:aud-A = thm:short-audA. Recovered and sharpened. |
| lem:nonid | the load-bearing construction, proved from scratch | a reading of lem:fibre at one radius. Proof now 8 lines and cites the lemma. Construction identical. |
| thm:headline | the headline result | the first link of five. Sec. 4.3 adds that its abstention radius EQUALS the minimax audit value (cor:beta-is-beta). |
| beta-sweep (sec:beta-sweep) | Sec. 7.2, buried in Results | Sec. 5, top-level, BEFORE the method: "Theorem 4 predicts...; this section is that prediction, measured." |
| thm:certificate | third pillar of Theory | the object that survives Sec. 4-6; opens Sec. 7. |
| rem:empirical | draft remark in the theory file | Sec. 5's closing remark, next to the experiment it interprets |
| eps vs beta | a paragraph inside Sec. 5.4 | the TITLE of Sec. 7.4, rewritten so all four sites agree |
| Camelyon17 | "not reproducible from release" (3 sites) vs "recomputed 2026-07-26" (2 sites) | one status, four sites: recomputable AND vacuous (18/18 helpful, adapts 18/18) |
| LOCO recalibration | mid-subsection paragraph, Tent-only, 1 run | own labelled paragraph + tab:loco-ten, all ten runs, with a new negative |

---

## 5. Three corrections I had to make to the plan's own numbers

1. **rem:empirical said the genuine ImageNet-C coverage measurement is 0.470.** The primary table
   says 0.5111. Both correct, different runs: 0.470 is `adversarial_ablations_results.json` at its
   refit beta_hat = 0.0293; 0.5111 is `beta_sweep_results.json` at the primary beta_hat = 0.0302.
   The text now states both with their sources.

2. **rem:empirical said "five of its six recorded configurations".** Verified: the ablation file
   records exactly six ImageNet-C rows at 0.9556 / 0.9556 / 0.9037 / 0.9852 / 0.9556 / 0.4704.
   Correct; the text now names the file so the count is checkable.

3. **NEW NEGATIVE, not in the plan or the brief.** Recomputing all ten LOCO runs from
   `out_cifar_loco_tent_eata.json`:

   | | shipped | LOCO |
   |---|---|---|
   | eps | 0.0152-0.0219 | 0.0926-0.1122 (4.3x-6.9x) |
   | commitment rate | 0.509-0.600 | 0.398-0.417 |
   | KGA regret | 0.00101-0.00185 | 0.00550-0.00997 (3.4x-7.6x) |
   | FA_u | 0.0000 (10/10) | **0.0000 (10/10)** |
   | freeze decisions | 22-76 | **0** |
   | beats always-adapt | **10/10** | **5/10 — Tent only; loses on all five EATA runs at 2.01x-2.94x** |

   The brief and the plan both presented "FA_u = 0 in all ten runs" as the strongest empirical fact
   the paper owns. It is — but the beats-both claim does NOT survive that partition and the freeze
   branch empties entirely. Stating only the zero would have been a selective read. All three facts
   are now in the LOCO paragraph, the table caption, the introduction, the abstract, and
   SUBMISSION_LEDGER.md Sec. 10b.

---

## 6. Deviations from the spec, and why

| # | plan said | I did | why |
|---|---|---|---|
| D1 | rem:empirical not promoted; Sec. 5 gets a hand-written paragraph | wrote that paragraph AND promoted rem:empirical verbatim as `beta_sweep_correspondence.tex` | the plan's Sec. 5 spec asks for the anchor-collapse check and the ImageNet-C sign reversal, which IS rem:empirical. Paraphrasing a proved remark violates "use those statements verbatim". |
| D2 | prop:threeterm cites "Aud-B, Aud-C, Aud-F, Aud-G, Aud-H" | cites App. app:audit-short without naming Aud-B/F/H | those three do not exist in the short build. Naming absent theorems is a fabricated citation. Verified the claim holds for the three that ARE there. |
| D3 | rem:notsharp should drop the conj:aud-maximal clause | dropped the \ref, KEPT the open question in words | deleting the clause would remove an honest statement of non-sharpness. |
| D4 | thm:lecam proof -> app:theory-full | put it in app:episode-proofs, which app:theory-full inputs | keeps every relocated proof in one file; app:theory-full still resolves as the pointer target. |
| D5 | epi:thm-conformal cites four named conformal papers | cites the line generically in epi:rem-audG + \cite{tibshirani2019conformal} | those four bib keys are not in references_kbound_expanded.tex. Inventing keys is a fabricated citation. The no-novelty statement is intact and in the main text. |
| D6 | Sec. 6.9 splits | done; Delta_sep got its own subsection heading | it is a recommendation, not a result, and needed not to read as one. |
| D7 | Sec. 11.2 -> appendix paragraph; tab:claim-status -> merged float | both into a new app:panel-detail together with 7.10/7.13/7.14; 11.2 -> app:formal | sec:guarantees and tab:claim-status both survive with their labels. |

---

## 7. Checks from plan Sec. 10 — results

1. **Label survival contract.** All 48 contracted labels present, script-verified. Both builds:
   0 undefined references.
2. **Build check.** Above. Zero new warnings relative to baseline.
3. **The Aud-A escape seam.** Sec. 4.3 now ends with an explicit block, "The one hypothesis that
   can be attacked, and where we attack it", naming the supremum-vs-average substitution and
   forward-pointing to Sec. 6.4. Sec. 6.4 carries epi:prop-escape WITH ITS PROOF IN THE MAIN BODY.
   Signposted from both sides.
4. **The 0.900 null.** Every coverage figure near 0.90 audited. Three sites carry it and all three
   label it: tab:beta-sweep's caption (np.quantile trap), tab:episode-coverage's caption ("The null
   is 0.900 by construction — rows at or near 0.900 are evidence the estimator is not broken, not
   evidence that it works"), and the jackknife paragraph in Sec. 8. Finding (i) in Sec. 6.8 is
   worded so the control reads as a not-broken check.
5. **Camelyon17.** Four sites reconciled. Grep for "not reproducible from release" and
   "unverifiable" returns nothing in either file.
6. **eps vs beta.** Three of four sites are mine and now agree. **The fourth is in kbound.tex,
   which I do not own, and it still asserts that thm:frontier identifies eps as the exact
   benefit-sign budget. That is the conflation and it must be fixed by whoever owns that file.**
   Recorded in SUBMISSION_LEDGER.md Sec. 6.
7. **No novelty claimed for epi:thm-conformal.** epi:rem-audG is in the main text of Sec. 6.3, in
   bold, not a footnote.
8. **Ledger updated.** Sec. 2 (full new theorem inventory, prop:beatsboth-asym recorded as cut,
   prop:multiclass relocation, proof-coverage restated), Sec. 6 (two new mandatory distinctions
   incl. the unfixed long-manuscript site), Sec. 10a (seed-0 heterogeneity), Sec. 10b (corrected
   LOCO ranges + the new EATA negative).

---

## 8. Proof-status discipline

- Proofs in the main body: lem:fibre, lem:nonid, cor:matched-abstain, prop:closed-band,
  thm:headline, thm:beta-minimax, cor:audA, cor:beta-is-beta, thm:anchor, prop:threeterm,
  thm:dichotomy, thm:certificate, epi:prop-escape.
- Pointered to app:episode-proofs: thm:lecam, epi:thm-ident, epi:thm-conformal, epi:thm-floor,
  epi:prop-labels, epi:prop-probe, epi:thm-shift.
- **epi:conj-open is in a `conjecture` environment and the sentence after it says "This is stated
  as a conjecture, not a theorem: we neither exhibit such an assumption nor prove that none
  exists."** Nothing with GAP/CONJECTURE status is presented as a theorem anywhere.
- thm:short-audC/DE/G keep their verbatim "stated here without proof, proved in the long version"
  note. No main-text claim depends on them.

---

## 9. Cuts executed, and what was NOT cut

Executed: C1 (synthetic wiring check, 51 lines + fig:frontier-measured; replaced by the exact
3-sentence note in app:supp-exp, 4 refs rewired), C2, C3, C4, C5 (its ref moved into Limitations
with the "no physical result is claimed" sentence), C6 (prop:beatsboth-asym, comment left at the
cut site), C11, C12, C13, C14.

**Not cut, deliberately:** every disclosure, retraction, caveat, absent-artifact note and negative
result. Machine-verified: 20 disclosure markers counted before and after; none lost, several added.
The iWildCam streaming-script retraction is verbatim. The SAR quarantine is verbatim. The
np.quantile null caveat is verbatim and now appears in a second place. The seed-0 heterogeneity is
a NEW disclosure, in Sec. 8, and seed 0 is not dropped from any five-seed figure.

---

## 10. New files created (all under paper/sections/)

| file | contents |
|---|---|
| theory_fibre_engine.tex | Sec. 4.1 — def:audit-data, lem:fibre + proof, rem:one-construction |
| theory_beta_minimax.tex | Sec. 4.3-4.6 — def:fibre-radius, thm:beta-minimax, cor:audA, cor:beta-is-beta, thm:anchor, prop:threeterm, rem:reading, thm:lecam, thm:dichotomy, rem:notsharp, rem:honest-scope, + the escape-seam signpost |
| theory_certificate.tex | split out of theory_core_main.tex: thm:certificate, rem:fa-marginal, cor:abstain-valid |
| theory_episode_main.tex | all of Sec. 6 (statements + epi:prop-escape's proof + tab:episode-coverage) |
| theory_episode_proofs.tex | app:episode-proofs + app:selfcheck: seven relocated proofs, full 26-row coverage grid, K-sweep table, TWO IMPLEMENTATION-BUG DISCLOSURES, five algebra self-checks |
| beta_sweep_correspondence.tex | rem:empirical, promoted into Sec. 5 |

theory_core_main.tex was MODIFIED (certificate split out; lem:nonid's proof shortened to cite
lem:fibre). Verified: no file outside my slice inputs it.

Artifacts copied into the repo tree so the paper's path citation resolves:
experiments/kbound/frontier_sweep_v1/beta_impossible/{check_anchor_collapse.py,anchor_collapse_check.json}

---

## 11. What a referee will still hit, and I could not fix from this slice

1. **The main body is 54 pages.** See Sec. 1. The honest cost; the plan understated it.
2. **kbound.tex's eps-vs-beta paragraph still contradicts Sec. 7.4.** Not my file.
3. **Act 2 is mechanically elementary** and rem:honest-scope says so in the main body, in the
   paper's own voice. Kept verbatim.
4. **Two benchmarks.** The spine is coherent; the empirical base is not wider.
5. **The highest-value next experiment is still unrun** — two real cells with indistinguishable
   evidence and opposite benefit signs. The restructure makes room for it; it does not perform it.
