# Review 6 — Overall Panel Assessment (Chief Reviewer)

Panel: 5 specialists (mathematician, TTA systems engineer, empirical methodologist, reproducibility
auditor, area chair), 82 findings raised, every one re-checked by an adversarial verifier. Post-verification
severity distribution: **1 BLOCKER, 27 MAJOR, 42 MINOR, 11 NIT, 1 REFUTED**. After deduplication across
reviewers the majors collapse to **12 distinct defects**, listed in the fix queue below.

I ran my own arbitration checks on the three findings where specialists disagreed; those results are in
`## Where the panel disagrees` and they change two severities.

---

## Verdict

**Major revision — not submittable in its current state, at any venue.**

The reason is narrow and fixable, and it is not the theory. It is that **the abstract makes a universally
quantified empirical claim that is falsified by a table inside the same PDF** (F5-1), and that **two
different conformal calibration rules are used within a single claim, with the more favourable one chosen
for each half** (F1-2 / F4-10 / F5-2). Those two are disqualifying on their own: the first is the kind of
thing a program committee finds in ten minutes and never forgives in a paper whose entire selling point is
calibrated honesty; the second looks like rule-shopping even though I believe it is staleness.

**At a top venue (NeurIPS / ICML / ICLR): reject.** Beyond the two items above, the theory section will be
discounted to a definition plus a triangle inequality (F5-3, and the authors' own
`VENUE_BENCHMARK_2026-06.md:80` says as much), the population theory has zero contact with the nine
benchmark tracks (F5-4: β is never supplied to KGA on real data, `kbound_short.tex:596-598`), and the one
experiment that tests the frontier is circular by construction (F1-4). A 23-page IEEEtran `[conference]`
submission with 26 tables (F5-11) is a desk-reject risk before any of that is read.

**At a mid-tier venue or TMLR: major revision, then a plausible accept.** The CIFAR-10-C stress-grid safety
result is real, well-powered and survives every robustness check the panel threw at it — including one I ran
myself (see below). The engineering hygiene is genuinely above field average. TMLR's criteria (claims
supported by evidence, not novelty magnitude) actually fit this work, and reformatting single-column
dissolves F5-11.

**The single biggest risk to the work** is not any individual defect. It is that the paper's *product* is
trustworthiness — a certificate that tells you when to trust an unlabeled decision — and the panel found
that the project's own audit documents certify things the source code contradicts: `PHASE6_LEAKAGE_AUDIT.md`
declares "PASS (clean). No live promoted track computes ε in-sample on the cells it scores" while four
scripts do exactly that (F3-13 / F4-2 / F4-3); `SUBMISSION_LEDGER.md` marks G8 `[RESOLVED = PASS]` while the
appendix table it ordered regenerated was not regenerated (F1-2) and the ledger itself reverts the panel to
the values G8 said to drop (F3-4). A reviewer who discovers that the self-audit is unreliable stops
believing the parts that are correct, and most of this project *is* correct. Fixing the audit-vs-source
discrepancies is worth more than any new experiment.

**Bottom line for the author:** you are approximately three weeks of disciplined cleanup away from a
defensible submission, and roughly none of that work is new experiments. The evidence you have mostly
supports a *narrower* paper than the one you wrote.

---

## Panel scorecard

| Reviewer | Perspective | Score /10 | Confidence | Biggest concern |
|---|---|---|---|---|
| R1 | Mathematician / theoretical statistician | 5 | 4/5 | The finite-sample layer: `thm:certificate` is `if coverage then safety`, and on the promoted rule FA_u ≤ α is an arithmetic identity, not a measurement (F1-1) |
| R2 | AI/ML systems engineer (TTA) | 4 | 5/5 | The shipped `kga` package produced no reported number; seven copy-pasted `decide_kga()` forks did, using the rule `certificate.py` exists to replace (F2-1) |
| R3 | Empirical scientist / methodologist | 4 | 4/5 | Wrong unit of analysis: the ImageNet-C beats-both CI bootstraps 135 correlated cell-seed rows as independent; the paper says it seed-averaged (F3-1) |
| R4 | Research integrity / reproducibility | 4 | 5/5 | Neither documented one-command reproduction runs; 142 committed artifacts are NUL placeholders; 2 of 9 promoted tracks have no verifiable source (F4-7/8/9/12) |
| R5 | Area chair / venue | 4 | 4/5 | The abstract's universal no-harm claim is contradicted by the paper's own Table XV (F5-1) |

Panel mean **4.2/10**. No reviewer recommended acceptance; two recommended reject, three major revision.
Notably, all five independently praised the project's self-audit discipline before criticising it — the
scores reflect claim-evidence mismatch, not sloppiness.

---

## Where the panel agrees

Five clusters were found independently by two or more specialists. These are the real problems.

**1. The radius is calibrated on the cells it scores (4 reviewers: F1-1, F2-2, F4-2, F4-3; audit-doc
contradiction F3-13).** `eps = np.quantile(np.abs(Bhat - B), 1-alpha)` is taken over all N residuals
including cell *i*'s own, then used to decide cell *i*. This appears at
`docs/research/kbound/scripts/cifar_tent_mps_v2.py:162` (the CIFAR headline runner),
`g8_canonical_pooling.py:11`, `g8_exactrank_regen.py:26`, `run_wilds_camelyon17.py:56`,
`ablation_exactrank.py:88-89`, and in seven `decide_kga` forks. It means ε is a function of the test labels,
which is the object the FA_u ≤ α guarantee attaches to. `PHASE6_LEAKAGE_AUDIT.md:5,11,16` certifies the
opposite, citing a line range (`:143-156`) that stops one line short of the offending line.

**2. Two conformal rules are mixed inside one claim (3 reviewers: F1-2, F4-10, F5-2).** ImageNet-C SAR's
aggregate (0.0264, FA_u = 0.000) comes from the exact-rank rule; the per-seed table
(`kbound_short_appendix.tex:303-310`, pooled 0.0107, FA_u = 0.007) comes from the interpolated rule that
`SUBMISSION_LEDGER.md:88-89` ordered dropped. Three reviewers independently replayed both rules from the
same JSONs and got identical answers. Under the promoted rule, `kbound_short.tex:801` and
`kbound_short_appendix.tex:287` — "improve both fixed-policy regrets on 5/5 seeds" — are **false**: seeds 0,
1 and 3 are bit-identical ties with always-freeze.

**3. Headline sentences contradicted by the paper's own tables (2 reviewers: F5-1 = F3-2).** Abstract
`:41`: "Across every natural distribution shift we test — hospital, wildlife-camera, laboratory-batch,
domain, and rendition shifts — KGA is uniformly no-harm: it matches the better fixed policy." Table XV
`:915-916`: PACS KGA 0.0431 vs always-adapt 0.0176 (2.45×); ImageNet-R 0.0112 vs 0.0064 (1.76×). Both
reproduce from raw. The body already calls these null diagnostics (`:1109`) — the abstract and conclusion
did not get the memo.

**4. The flagship CIFAR-10-C track is not reproducible from the release (2 reviewers: F2-3 = F4-7).**
`stress_grid_multiseed_v1/seed0/` has no per-condition records; `_locked_analysis_script.py` raises
`FileNotFoundError` at line 28 as released; `ablation_exactrank.py:36` and `gate_baseline_comparison.py`
both point at files that do not exist. R4 additionally found that `reproduce_submission.sh` — the documented
one-command verification — aborts at step 1 (F4-9), and that 142 committed text artifacts are NUL-filled
iCloud placeholders (F4-8), including every ablation JSON and the entire Office-Home runner.

**5. FA_u = 0 is definitional wherever it is reported outside CIFAR-10-C (2 reviewers: F3-8, and F1-1's
arithmetic).** RxRx1 adapt rate 0.0 (60/60 frozen), iWildCam 1/72, ImageNet-C SAR 12 adapts across 5 seeds.
Clopper-Pearson 95% upper bounds on conditional false-adapt: CIFAR-10-C 0.0027, Office-Home 0.127,
ImageNet-C SAR 0.221, RxRx1 undefined. The panel reports none of this. Only one track actually exercises
the guarantee.

Two further items were raised by one reviewer each but corroborated by another's evidence and belong here:
the **quarantined CIFAR-10-C SAR arm is printed in the body with comparative wording** (F3-3 = F4-1), where
the favourable direction comes entirely from the seed the quarantine exists for (seed 0 harmful fraction
0.528 vs 0.074-0.102; drop it and KGA loses to always-adapt); and the **`--quick` smoke configuration was
used for the headline stress grid** (F2-5: 6 of 15 corruptions, severities {1,5}, `"quick": true` in the run
manifest, and `kbound.tex:807` affirmatively states the grid covers "the official CIFAR-10-C corruptions").

---

## Where the panel disagrees

**1. Is `FA_u ≤ α` structurally forced, or is the ceiling above α?**
R1 (F1-1) says it is an arithmetic identity that cannot be violated. R2's verifier (F2-2) says the forced
ceiling is 44/432 = 0.1019, *above* α, so the result is not vacuous.

*My ruling: both are right about different rules, and R1 is right about the rule the paper promotes.* I
recomputed the ceilings directly. Under the exact-rank rule `k = ceil((n+1)(1-α))`, the ceiling is
`(n-k)/n` = **0.0972 at n=432** and **0.0370 at n=27** — both below α = 0.10, so FA_u ≤ α cannot be violated
for any data. Under the interpolated `np.quantile` rule the exceedance fraction is **0.1019 at n=432** and
**0.1111 at n=27** — above α, so a violation is arithmetically possible though barely. The promoted
headline uses exact rank; the archived artifacts use interpolated. Verdict: on the promoted rule the
guarantee is untestable by construction on the stress grids, and the paper must say so. Severity stays
MAJOR, not BLOCKER, because the natural-shift tracks use a genuine dev/test domain split — but see the next
ruling, which limits how much comfort that provides.

**2. Does the ImageNet-C "beats both" survive?**
R3 (F3-1) says no: the CI bootstraps 135 correlated cell-seed rows as independent while the text
(`kbound_short.tex:797-802`) says it seed-averaged. R5 (F5-2's verifier) says the pooled beats-both survives
the exact rule.

*My ruling: R3 is right, and the fix is a demotion, not a withdrawal.* I re-ran it. Pooled point estimates
0.026422 / 0.052933 / 0.031894 reproduce exactly. i.i.d. over 135 rows: adapt-gap **[-0.0524, -0.0032]**,
freeze-gap **[-0.0087, -0.0027]** — matches the manifest. Seed-averaged over 27 conditions (the design the
text describes): adapt-gap **[-0.0812, +0.0179]** — includes zero; freeze-gap **[-0.0092, -0.0023]** —
excludes zero. So the gap to the *worse* fixed policy is not CI-supported at the correct unit; the gap to
the *better* fixed policy (always-freeze) survives every clustering I tried. The honest claim is "beats the
better fixed policy at FA_u ≈ 0", which is still the interesting claim. R5's verifier reproduced the
i.i.d. interval and stopped there.

**3. How much does the in-pool radius actually change?**
R4 (F4-2/F4-3) treats it as near-blocker; R1 and R2 treat it as structural but bounded.

*My ruling: it is a labelling and audit-integrity defect with material numerical effect only at small n.* I
recomputed leave-one-out-of-pool radii across all 8 archived CIFAR tent+eata files (**3,456 cells**):
**zero decisions change, FA_u stays 0 under both rules.** On ImageNet-C (n=27) it moves FA_u from 0/135 to
1/135 and regret from 0.0264 to 0.0289 — beats-both against freeze survives (0.0289 < 0.0319). On Camelyon17
Table VIII (n=9) the realized ε is 0.153-0.372, which is large enough to be the *cause* of the "over-freezes"
verdict. So: fix the code and the audit doc everywhere; re-run Table VIII; the CIFAR-10-C safety result is
unaffected and should be defended, not hedged.

**4. Is `lem:nonid` a real theorem or a restatement of a known fact?**
R1 calls it "the one genuinely non-trivial theorem" with correctly ordered quantifiers; R5 (F5-3) calls it
the standard fact that unlabeled data does not determine the label conditional.

*My ruling: R1 on correctness, R5 on novelty — these are compatible and the paper should adopt both.* The
Bernoulli-kernel construction is sound, stays inside `C_β`, and the β = 0 carve-out is necessary; R1 verified
it and I see no error. But it is the same *move* as classical DA/label-shift non-identifiability, and the
paper never differentiates (F5-9: no Ben-David & Urner, no Lipton BBSE, no Garg label-shift in the
57-entry bibliography). R5's verifier was also right to knock down R1's jab that a TV = 0 construction "is
not Le Cam" — it is the degenerate, strongest case.

**5. R2-F2-13 (three incompatible evidence implementations) was REFUTED.** The docstring's provenance
citations in `kga/evidence.py:6-9` are exact, and `ks_mean` does appear in shipped result artifacts. Do not
act on that finding; the cosmetic residue (three coexisting evidence schemas) is in the "nice to have" list.
Similarly, R2-F2-8's causal link between the salted-`hash()` seeding and the G2 quarantine is refuted —
`rebuild_cifar10c_sar.sh:9` shows the quarantined track runs through `cifar_tent_mps_v2.py`, which has no
`hash()` call. The seeding bug is real; the story attached to it is not.

**6. R2-F2-4 (BN-statistic confound) — verifier downgraded MAJOR → MINOR; I restore it to MAJOR-adjacent
priority.** The mechanism is confirmed (`cifar_tent_mps_v2.py:1054` evaluates f0 with `train_mode=False`
while `:1061` evaluates f_a with `train_mode=True` on a clone whose running stats were nulled at `:694`), so
Δ bundles BN-statistic replacement with the gradient update, and no BN-only baseline exists anywhere in the
repo. The verifier is right that this biases *toward* adaptation looking helpful, so reported harmful
fractions are conservative — but "add a `bn` arm" is the first thing any TTA reviewer will ask for, it costs
a few hours, and its absence undercuts the 0.124 always-freeze regret gap. Priority high, severity MINOR.

---

## The fix queue

Ordered by (severity, then how much each item unblocks). Effort estimates assume the author knows the tree.

### Must fix before submission

**1. [F5-1, F3-2] BLOCKER — Abstract/conclusion claim uniform no-harm on shifts where the paper's own panel
shows KGA losing.**
File: `docs/research/kbound/kbound_short.tex:41`, `:85`, `:1221`.
Fix: delete "domain, and rendition" from the enumeration, or restate as: "on the four one-sided natural
tracks with locked held-out artifacts (Camelyon17, iWildCam, Office-Home, RxRx1) KGA ties the better fixed
policy at zero observed false adaptation; on PACS, ImageNet-R and CIFAR-10.1 it is a conservative null or
fails the transfer bar." The scoped version already exists at `:580` — promote it.
Effort: **30 minutes.** This is the single highest value-per-minute edit in the project.

**2. [F1-2, F4-10, F5-2] MAJOR — One quantile rule, everywhere; the "5/5 seeds" sentence is false under the
promoted rule.**
Files: `kbound_short_appendix.tex:287`, `:303-310`; `kbound_short.tex:801`, `:910`, `:1206`;
`paper/generated/decision_metrics.json`; `paper/generated/uniform_verdicts.json` (stale, still carries the
superseded 27-cell row).
Fix: regenerate `tab:imagenetc-perseed` under exact rank; replace the claim with "point estimates improve
both fixed-policy regrets on 2/5 seeds; on seeds 0, 1 and 3 KGA abstains throughout and is bit-identical to
always-freeze; the pooled win is driven by seeds 2 and 4." Declare the rule once, in the config table.
Effort: **2-3 hours.**

**3. [F3-1] MAJOR — Re-run the ImageNet-C bootstrap at the unit of analysis the text claims, and demote the
adapt-side claim.**
File: `docs/research/kbound/scripts/g8_exactrank_ci.py:18` (`idx=rng.integers(0,n,(5000,n))` over 135 rows).
Fix: seed-average to 27 conditions, as `_locked_analysis_script.py:54` already does for CIFAR. Report
adapt-gap [-0.081, +0.018] and freeze-gap [-0.009, -0.002]. Restate as "beats the better fixed policy
(always-freeze) with a CI excluding zero; the gap to always-adapt is not CI-supported at the condition
level." Update Table XV, `tab:primary-numeric`, `uniform_verdicts.json`.
Effort: **2 hours.**

**4. [F4-2, F4-3, F2-2, F1-1, F3-13] MAJOR — Remove the scored cell from its own radius pool, everywhere;
correct the leakage audit.**
Files: `scripts/cifar_tent_mps_v2.py:162`, `g8_canonical_pooling.py:11`, `g8_exactrank_regen.py:26`,
`run_wilds_camelyon17.py:56`, `ablation_exactrank.py:88-89`, plus the 7 `decide_kga` forks.
Fix: leave-one-out-of-pool radius (or a genuine held-out calibration split). I verified this changes **0 of
3456** CIFAR decisions, so the flagship number is safe; ImageNet-C becomes 0.0289 with FA_u = 1/135 (say so
— beats-both against freeze still holds); Camelyon17 Table VIII must be re-scored (its ε of 0.15-0.37 at
n = 9 is what produces "over-freezes"). Then rewrite `PHASE6_LEAKAGE_AUDIT.md:5,11,16` — its VERDICT line is
false as stated. Add a regression test asserting the scored index is excluded.
Effort: **1 day**, most of it re-scoring and doc correction.

**5. [F1-1, F3-8] MAJOR — Stop reporting FA_u ≤ α as if it were measured where it is forced or untested.**
Files: `kbound_short.tex:511` (RQ2), Table XV `:908-918`, `paper/generated/empirical_audit/decision_metrics.json`.
Fix: (a) state that with in-sample rank calibration FA_u ≤ (N−k)/N is an identity, so the informative
statistic is FA_u = 0 versus the ceiling, not "FA_u ≤ α"; (b) add ADAPT/FREEZE/ABSTAIN counts and a
Clopper-Pearson upper bound on FA_c to every panel row; (c) mark tracks with < 10 ADAPT decisions
"guarantee untested" — that is RxRx1 (0 adapts), iWildCam (1), Office-Home (22), ImageNet-C SAR (12);
(d) delete the Wilson intervals on deterministic in-sample counts (F1-11).
Effort: **half a day.** This is the honesty fix that makes the CIFAR-10-C result look *better*, because it
is the one track with real power (0/1114 adapts wrong, CP upper bound 0.0027).

**6. [F3-3, F4-1] MAJOR — Remove or restate the quarantined CIFAR-10-C SAR paragraph.**
File: `kbound_short.tex:637-642`.
Fix: the sentence claims the aggregate was "rebuilt from all five saved per-condition seed files" — only
four exist for the 432-cell grid (a fifth complete set exists at `win_hunt_v5/cifar10c_aggr/`, but that is a
270-cell grid). Drop the "intervals exclude zero" clause. Replace with: "SAR's five-seed aggregate is
dominated by seed 0, whose harmful base rate (0.53) is 5× the other seeds'; on seeds 1-4 KGA's regret
(0.00160) exceeds always-adapt's (0.00031). No comparative verdict is drawn." This is what your own
`LOCKED_ANALYSIS_FINDINGS.md` and `LOCK_SEAL.json`'s `"not_locked"` field already say.
Effort: **30 minutes.**

**7. [F2-5, F2-6] MAJOR — Disclose the actual operating points.**
Files: `kbound_short.tex:622-625`, `kbound.tex:807`, `kbound_short_appendix.tex:127` (`tab:adapter-hparams`).
Fix: (a) state that the CIFAR-10-C grid is 6 of 15 corruptions (name them) at severities {1,5} — the
driver's `--quick` mode, `"quick": true` in every run manifest; `kbound.tex:807`'s "the official CIFAR-10-C
corruptions" is false as written. (b) State the ImageNet-C SAR operating point inline: lr 4e-3 (16× the
2.5e-4 your own docstring at `cifar_tent_mps_v2.py:759` calls official), batch 16, 50 steps, layer4
adapted contrary to Niu et al. `tab:adapter-hparams` currently says "mild & aggressive (per cell)", which is
wrong — the manifests show aggressive-only, small-batch-only.
Effort: **1 hour** to disclose. Add **1 day** if you also run the official-settings control, which you
should — "beats both because SAR collapses" is much weaker if SAR only collapses at 16× its own lr.

**8. [F4-7, F2-3, F4-9] MAJOR — Make the two documented reproductions actually run.**
Files: `experiments/kbound/results/stress_grid_multiseed_v1/seed0/` (missing per-condition dumps);
`scripts/ablation_exactrank.py:36`; `scripts/gate_baseline_comparison.py:213`;
`tests/test_calibration_split_integrity.py:10-11` (computes `REPO/"docs"/"experiments"/...`, a path that
cannot exist); `reproduce_submission.sh:3` (`set -euo pipefail` means steps 2-9 never run).
Fix: commit `per_condition_cifar10c_{tent,eata,sar}_seed0.json` for the 432-cell grid, or repoint the
harness at the head-to-head seed-0 dump — but note `ablation_exactrank.py`'s `load()` also reads
`r['a_oracle']`, absent from every committed 432-cell dump, so repointing alone will not work. Fix the test
path, and `skipif` on missing edge artifacts so the guard degrades honestly. Add CI that runs
`_locked_analysis_script.py` against the archived tree.
Effort: **1 day.**

**9. [F4-8] MAJOR — Materialize the 142 NUL-filled placeholder artifacts.**
Files: every ablation JSON, `cost_profile.json`, `experiments/kbound/officehome/run_officehome_kbound.py`
(17,202 bytes, zero readable), `oh_analyze.py`, edge CSVs.
Fix: iCloud "Download Now", then add a release guard rejecting any tracked text artifact that is
NUL-filled — note the naive whitespace test does *not* catch these (I checked: a strict whitespace scan
returns 0 files, a NUL scan returns 142). Extend `STORAGE_MANIFEST.json` checksums (currently 3 files) to
every artifact a table depends on.
Effort: **hours** if the source Mac is available; **days** if any are genuinely lost.

**10. [F3-6, F4-12] MAJOR — Two promoted tracks have no verifiable source.**
Files: `scripts/bootstrap_win_cis.py:37,43,47` (all four record files absent);
`audits/integrity_2026-06-20/camelyon_reconciliation/` (does not exist — these are the only 2 of 69 sealed
files missing, and they are also the sole basis for the KB-CLAIM-022 withdrawal argument in
`PHASE6_LEAKAGE_AUDIT.md:43-45`).
Fix: restore the directory (its SHA-256s are already sealed, so restoration is independently verifiable) or
re-run `camelyon_G_reconciliation.py` and re-seal. Until then mark Camelyon17 OOD "not reproducible from
release" rather than "locked". The promoted row 0.0000/0.0000/0.1381 appears in **no** artifact on disk,
while live Camelyon artifacts show FA_u of 0.026 and 0.033.
Effort: **hours** to restore, **1 day** to re-run.

**11. [F3-4, F4-5, F4-11] MAJOR — Fix the provenance layer that the appendix calls "the authoritative index".**
Files: `paper/generated/kbound_result_manifest.json`; `nine_track_lock_v1/LOCK_SEAL.json`.
Fix: the CIFAR rows' `source` points at `LOCKED_ANALYSIS_RESULTS.json`, which contains different numbers;
the promoted values live in `mixed_headtohead_v1/HEADTOHEAD_RESULTS_*.json`. Repoint, add a `source` field
to `cifar10_1_K` and `rxrx1_J` (currently absent), seal the files that actually contain each number, and add
a CI check asserting each sealed file contains the manifest's value. Reconcile `SUBMISSION_LEDGER.md:88-89`
(G8 says drop interpolated) with `:130-131` (P2 reverts to interpolated) — the ledger currently contradicts
itself while marking G8 `[RESOLVED = PASS]`.
Effort: **half a day.**

**12. [F5-11] MAJOR — Decide the venue and cut to its limit.**
File: `kbound_short.tex:1` (`\documentclass[conference]{IEEEtran}`, 23 pages, 26 tables, 7 figures,
12 theorem environments, appendix bound in via `\appendices`).
Fix: at least eight tables are meta-tables about the paper rather than results (`tab:regime-summary`,
`tab:data-access`, `tab:assumptions-role`, `tab:notation-main`, `tab:evidence-map`, `tab:failure-modes`,
`tab:claim-status`, `tab:baseline-faithfulness`); merge or supplement five of them. Cut `fig:regime-map`
(its own caption admits it is a schematic duplicating `tab:regime-summary`). If TMLR, reformat
single-column and most of this dissolves. Also scrub identity before any double-blind submission —
`CITATION.cff` names the author while `kbound_short.tex:1167` says "anonymized repository" (F4-18).
Effort: **1-2 days.**

### Should fix

**13. [F5-3, F1-9] MAJOR framing — Stop selling the frontier as the theoretical contribution.**
Files: `kbound_short.tex:112` (contributions), `paper/sections/theory_setup.tex:21-27`.
Fix: γ is *defined* as `ā − 1/2 − M`, so `sign Δ = sign(M+γ)` is an identity and the sufficiency half of
`thm:headline` is interval arithmetic. Present `lem:reduction` honestly as a decomposition (a useful one),
put the weight on `lem:nonid`'s construction and on the necessity half, and consider promoting
`thm:short-audA` — the vacuity of label-free budget audits — which is the result with actual content.
Effort: **half a day** of rewriting, no new math.

**14. [F5-4, F1-4] MAJOR — Relabel the frontier validation, or give it teeth.**
File: `scripts/frontier_validation.py:53-58`; `kbound_short.tex:593-612`.
Fix: `Z` is four noisy copies of `M`, so the residual is exactly γ and ε → 0.9β by algebra — the script's own
docstring (`:19-22`) says so, and the reported "90.0% empirical coverage" is `np.quantile`'s definition
(verified: exactly 0.90 at n = 400 and n = 220 for arbitrary data). Either call it an illustration, or
re-run with `Z` that is not a noisy copy of `M`, γ whose 0.9-quantile is not β, and a held-out set. Better:
compute `M` from the ATC-style source-calibrated score you already have at `:364` on the CIFAR grid, declare
β from historical dev-to-deployment gaps, sweep β ∈ {0, 0.02, 0.05, 0.10, 0.20} and run the population rule
against Δ̂ ± ε. A negative result there is far more informative than the current silence (F5-5).
Effort: **1 hour** to relabel; **2-3 days** for the real experiment. The real experiment is the single
biggest upgrade available to this paper.

**15. [F2-1] MAJOR — Route every driver through the shipped library, or stop shipping it as the artifact.**
Files: `kga/certificate.py`, `kga/policy.py` vs. 7 `decide_kga` forks (`cifar_tent_mps_v2.py:151`,
`wilds/analysis.py:55`, `kga_breadth.py:83`, `frontier_validation.py:41`, `run_decision_baselines.py:83`,
`run_wilds_camelyon17.py:45`, `theory_v2/realdata/eps_recal/_probe2.py:14`, plus two inlined in
`wilds/per_condition_serialize.py:60,84`).
Fix: delete the forks; re-score every promoted track from stored `b_hat`/`B` through one path. Also resolve
the internal contradiction between `kbound_short.tex:549-550` ("LOO jackknife q_0.9") and `:800-801`
("exact split-conformal radius") — the config table is the stale one.
Effort: **1 day.**

**16. [F2-4] MINOR but high-priority — Add a BN-statistics-only baseline arm.**
File: `scripts/cifar_tent_mps_v2.py` (`TTA_METHODS` at `:818`).
Fix: add a `bn` method (source weights, target BN statistics, zero gradient steps) and report Δ against it
as well; or evaluate f0 in `train()` mode so both arms use target statistics and Δ isolates the gradient
update. Also state the eval chunk size (512 CIFAR / 256 WILDS) in the config table — it silently changes
`a_adapted` because the adapted arm recomputes BN stats per chunk.
Effort: **half a day** to implement, hours to re-run.

**17. [F3-5, F3-12] MINOR — Report cluster-robust intervals and a leave-one-corruption-out row.**
File: `_locked_analysis_script.py:60`; ablation table.
Fix: r0/r1 replicate pairs correlate 0.948-0.999, so effective n ≤ 216, not 432; clustering by corruption
family widens CIs 2.4-3.9× (all still exclude zero) and reveals one family (gaussian_noise, +0.0019) where
KGA is worse than always-adapt. Separately, leave-one-corruption-out calibration triples estimator MAE
(0.0102 → 0.0322) and quadruples ε (0.021 → 0.092) while FA_u stays 0 — that ablation costs six model fits
and about one second, and it answers the strongest objection to your calibration design. Add both.
Effort: **2 hours.** Strengthens the paper.

**18. [F3-10, F3-11] MINOR — Baseline parity and the radius's own value.**
Files: `scripts/gate_baseline_comparison.py:26-28,49,52`; `tab:gates`.
Fix: two of the four gates are unfitted sign rules while KGA gets 432 LOO GBR fits; the docstring claims all
gates are leave-one-task-out calibrated "exactly like KGA", which is false for gates 1-2 *and* for KGA
(leave-one-cell-out). Either port the LOO-tuned rows from `kbound.tex:1948-1958` or say in the caption that
the gates are untuned. Separately, report explicitly that the radius-free variant meets the declared budget
(FA_u = 0.049 < α = 0.10) at 4.25× lower regret, and identify the regime where the radius pays for itself —
the harmful-cell column (0.141 → 0.000) is your answer; make it the argument.
Effort: **3 hours.**

**19. [F4-6, F4-14] MAJOR/MINOR — Disclose or eliminate the seed-0 environment heterogeneity.**
Files: `stress_grid_multiseed_v1/seed*/result_manifest.json`; `requirements.lock.txt`.
Fix: seed 0 ran on a different commit, Python 3.12.13 and torch 2.5.1; seeds 1-3 on Python 3.14.3 and torch
2.12.0; seed 4 on a third commit. ImageNet-C seed 0's argv also omits `--severities 1 3 5` and
`--max-images 4000`. A five-seed variance claim requires seeds to differ only in seed. Re-run seed 0 under
the seeds-1-4 stack, or footnote it. Also record the scikit-learn version in `result_manifest.json` — `b_hat`
comes from `GradientBoostingRegressor(subsample=0.8)` and no manifest pins it, so ε and every decision are
version-dependent. Add `result_manifest.json` to `pooled_5seed/`.
Effort: **hours** to disclose, **1 day** to re-run.

**20. [F4-13] MAJOR — Write `DATA.md`.**
File: `STORAGE_MANIFEST.json`.
Fix: three entries say "see DATA.md" and no such file exists anywhere; the ImageNet-R command
(`bash scripts/download_data.py --dataset imagenet-r`) is not a real invocation (that script accepts only
`--enron`, `--cifar10`, `--all`); the CIFAR-10-C command references an `AETTA/` directory not in the release.
Only 2 of 9 datasets have a working path. Per dataset: canonical URL/DOI, version, split definition, licence,
archive checksum. Pin `wilds==<version>`.
Effort: **half a day.**

**21. [F1-6, F1-8, F5-12] MINOR — Proof hygiene in the compiled build.**
Files: `paper/sections/theory_core_main.tex:60-61`; `theory_appendix_ext.tex:36,46`;
`kbound_short_appendix.tex:155-157,321,329,344,353`.
Fix: 8 of 13 compiled theorem-level results have no proof. Concretely: delete the dangling "An explicit
Gaussian witness appears in Appendix (Theorem thm:imp)" (the witness is inside an `\iffalse` block and does
not compile; the Bernoulli kernel already in `lem:nonid` suffices); give `thm:imp(ii)` a two-line
Neyman-Pearson proof and define `\TV`, which is a bare macro; repair `prop:beatsboth-asym`, which is
currently a subordinate clause with no main verb inside a numbered proposition of a frozen submission, and
add its two-line partition proof; supply six-line proofs for Aud-A/C/DE/G or mark them "stated here; proved
in [long version]" with a retrievable citation. Also fix `theory_appendix_ext.tex:37` — it claims `lem:nonid`
is parts (i)-(ii) of `thm:imp`, but `lem:nonid` establishes only (i), and `SUBMISSION_LEDGER.md:45` records
that exact xref as already fixed. Add `prop:multiclass` to the ledger's compiled inventory.
Effort: **half a day.**

**22. [F5-6, F5-7, F5-8, F5-9, F5-13] MINOR — Framing repairs.**
File: `kbound_short.tex:41,114,132,155-156`.
Fix: (a) define "detectable" ex ante and measurably, then classify all nine tracks *before* reporting
outcomes — ImageNet-C Tent at 56% harmful is the most mixed track in the paper and is not labelled
mixed+detectable because KGA does not win there; (b) delete "FMoW, Poverty" from the contributions bullet
(they appear nowhere else in the paper); (c) compile two sentences of `related_work_positioning.tex:97-108`
into Section II — your own honest-scope paragraph is written and never `\input`; (d) add Ben-David et al.
2010, Lipton BBSE, Garg label-shift and a reject-option/learn-to-defer citation, and state the three
differences (paired sign not risk level; ambiguity localized to D; class parameterized by declared budget);
(e) in the abstract replace `|M| > β` with words and qualify "the only sound action" as "the only action that
certifies a strict benefit direction".
Effort: **3 hours.**

**23. [F5-10, F3-15, F3-16] MINOR — Panel rows that hide their own variance.**
Files: `kbound_short.tex:911,915,916`; `PACS_MULTISEED_RESULTS.json`.
Fix: Camelyon17's panel row reports FA_u = 0 with no candidate qualifier while `tab:multiseed`'s SAR row
reports 0.11 > α — qualify the row and note that 1/9 is inside binomial noise. PACS art_painting seed 1 has
FA_u = 0.1111 > α and seed 2 abstains on all 18 cells; report k/n with a Wilson interval (the per-cell files
exist for 2 of 3 seeds). ImageNet-R's mean-across-backbone hides convnext_tiny at 0.0207 vs always-adapt
0.0015 (14×) and convnext_base at a degenerate 0% harmful base rate — report min/median/max and the harmful
base rate per backbone.
Effort: **3 hours.**

**24. [F3-9] MINOR — Declare the comparison family prospectively.**
File: `paper/generated/uniform_verdicts.json` (`_meta.wave_holm_family`).
Fix: the family is "the 3 beats-both candidates ... all survive Holm" — exactly the three that won, declared
after the results were known, against 1,387 recorded `beats_both` determinations project-wide (23.5% true).
Most of those are exploratory dev-screen rows and should not be treated as a multiplicity family, but the
paper publishes no arm inventory at all. Publish protocol → arms → verdicts as an appendix table and define
the family over the pre-registered beats-both bars.
Effort: **3 hours.**

### Nice to have

**25. [F1-5, F2-7] MINOR — `kga/certificate.py:261` silently under-covers at small n.**
`k = min(n, ceil((n+1)*(1-alpha)))` returns `max(residuals)` when the clamp fires (n ≤ 8 at α = 0.1), giving
attainable coverage n/(n+1) < 1−α while the docstring still promises 1−α. Three sibling scripts
(`ablation_exactrank.py:57`, `official_baselines_headtohead.py:48`, `reproduce_headlines.py:32`) already
return `inf` correctly, so this is a bug, not a convention. Return `inf` → ABSTAIN; apply the same guard
inside `route_panel`, where `bonf = alpha/k` makes it reachable (I confirmed `route_panel(K=5, n_cal=10)`
returns `committed=True` at an unattainable level). Add tests at n ∈ {3,5,8}. **1 hour.**

**26. [F1-12, F2-12, + R1's missed item] MINOR — Certificate estimator defaults.**
`empirical_bernstein` (`:181`) and `hoeffding` (`:236-239`) both default to a data-estimated range, voiding
Maurer-Pontil. Make `benefit_range` required. Separately — and this is the sharpest unflagged item in the
whole panel — both return **one-sided** radii (`ln(2/alpha)`, `ln(1/alpha)`) while `kga/policy.py:79-81`
applies ε in both directions, so `thm:certificate`'s two-sided premise holds only at 1−2α. The FA_u/FF_u
conclusions survive via the one-sided argument, so the right fix is to restate `thm:certificate` with two
separate one-sided coverage conditions rather than `Pr[|Δ̂−Δ| ≤ ε] ≥ 1−α`. **2 hours**, and it makes the
theorem stronger.

**27. [F1-3] MINOR — Algorithm 1's two `\Comment` labels say "exact split-conformal rank quantile" for
branches that are leave-one-out and CV respectively.** The body text (`kbound_short.tex:338-341`, `:402`,
`:692-693`) is already honest about this and even names jackknife+ as unimplemented; only the algorithm
labels are wrong. Promote the manifest's own string ("leave-one-condition-out cross-fitted empirical
residual calibration; not exact split conformal") into the algorithm. **15 minutes.**

**28. [F2-9] MINOR — One definition of false-adapt.** `wilds/analysis.py:87` and
`cifar_tent_mps_v2.py:182` compute `mean(B[adapt] < 0)` (conditional, strict) under the field name
`false_adapt_rate_B<0`, and it gates `beats_both`; `_locked_analysis_script.py:43` uses the marginal
`mean(is_adapt & (B <= 0))` that the theorem controls. 500 archived cells have B exactly 0.0, 102 of them
ADAPT (97 in baseline arms). Define `false_adapt_unconditional` once in `kga/policy.py`, emit `fa_u` and
`fa_c` as separate fields. `SUBMISSION_LEDGER.md:89` already carries "Still fix FA_u marginal code label"
inside a block marked `[RESOLVED = PASS]`. **2 hours.**

**29. [F2-10, F2-11, F2-15] MINOR — Shipped library defects with no paper impact.** `kga/cli.py:62-64`
hard-codes `delta_hat = 0.0`, so `python -m kga decide` is a constant-ABSTAIN generator (I ran it: maximal
detected drift, still ABSTAIN) — this is the first thing a reader will execute. `kga/evidence.py:218`
computes the reciprocal of the importance weight its docstring specifies, giving opposite conclusions on
variance-shrink shifts (ess_frac 0.883 vs 0.0008 in my run). `kga/routing.py:202-205` early-returns, so
candidates after the first rejecter never ingest that step. **3 hours total.**

**30. [F2-8, F3-17, F2-16] MINOR/NIT — Reproducibility hygiene.** `src/scripts/kbound/cifar10c_suite.py:70`
seeds per-cell RNG with salted `hash()` (three interpreters gave three different values) — replace with a
stable digest. Six scripts hard-code `~/Documents/AutoML_Flagship_V8` or `/Volumes/T9/...`, including both
scripts that produce the ImageNet-C headline (`g8_canonical_pooling.py:2`, `g8_exactrank_ci.py:2`), against
`EXTERNAL_STORAGE_POLICY.md:18`'s own ban on machine-local paths. `test_sar_faithful.py:58` asserts
`(w.clone() - w).norm() < 1e-9`, which is 0 by construction and touches no project code — replace with an
assertion on the real parameter tensors. **2 hours.**

**31. [F4-4, F4-15, F4-17, F4-19] MINOR — Ledger and seal hygiene.** Re-freeze (the pinned PDF sha256 cannot
match a `.tex` edited a day later per `EDIT_NOTES_2026-07-23.md`); regenerate `STORAGE_MANIFEST.json` (its
recorded hash for `claim_ledger.json` — the file it designates the wording authority — no longer matches
disk); stamp superseded audit docs "SUPERSEDED BY SUBMISSION_LEDGER §N" or delete them (three of them tell
three different stories about RxRx1 0.2531 vs 0.2587, and `REVIEWER_REPRO_PACKET.md:92-93` tells external
reviewers that Office-Home carries a CI-backed beats-both that the ledger deliberately demoted); make the
forbidden-phrase gate context-aware, since it currently fires 25 times on the paper's own disclaimers.
**half a day.**

**32. [F1-7, F1-14, F5-14, F5-15] NIT — Out-of-build and cosmetic.** `theory_v2/tight_constants_closure.tex:93-97`
invokes MLR/Karlin-Rubin where Neyman-Pearson with a p+q ≤ α budget split is needed (the constant κ(α) is
correct, the argument is not; also its stated range [4, 5.2] holds only for α ≤ 0.10 — κ(0.20) = 6.36).
Not in the frozen build, so fix it before the long version. Rename `Impossibility.lean`'s
`forced_abstention_probability` (proved `by linarith`) and `exchangeable_scores_miss_le_alpha` (which takes
its hard step as a hypothesis; no permutation-invariance definition exists anywhere in the 27 `.lean` files)
to match content — or formalize exchangeability properly, which is the highest-value addition available and
a tractable mathlib exercise. Drop `\resizebox` from 12 tables. **1 day.**

---

## What is genuinely strong

Do not change these. Several are better than field standard and two are better than anything else the panel
has reviewed recently.

- **The CIFAR-10-C stress-grid safety result is real and it is your paper.** 2,160 cell-seed pairs, 1,114
  ADAPT decisions, 0 false adapts, Clopper-Pearson 95% upper bound on FA_c of 0.0027. R3 reproduced
  0.00157361 / 0.00792338 / 0.12409792 to 8 decimals from raw. It survives cluster-robust bootstrapping at
  the corruption level, leave-one-twin-pair-out refitting, leave-one-corruption-out calibration (FA_u stays
  0), and — my own check — leave-one-out-of-pool radius recomputation across 3,456 cells with **zero**
  decision changes. This is the one place where the guarantee is actually exercised and it holds.
- **No TTA state leakage.** `_clone_for_tta` (`cifar_tent_mps_v2.py:702`) deep-copies the frozen model with
  a fresh optimizer per cell; the adaptation stream and evaluation pool are disjoint on CIFAR
  (`:838-840`); the eval pool is class-balanced. This is the single most common source of fake TTA gains
  and R2 confirmed it is absent.
- **β is never fitted.** R3 grepped `kga/`, `src/scripts/kbound/` and all 69 `research_lock/*.yaml`: the
  obvious fatal flaw — a "declared" budget silently tuned on test — does not exist, and α is fixed at 0.10
  everywhere.
- **The evidence seal works.** Two reviewers independently recomputed all 69 SHA-256 entries in
  `LOCK_SEAL.json`: 67 verify byte-for-byte, 0 mismatches, 2 missing. The machinery is sound; the problem is
  only that two seals hash files that do not contain the promoted number (item 11).
- **Negative results are retained and the promotion direction is honest.** CIFAR-10.1 is reported at
  FA_u = 0.167 and labelled a diagnostic fail; PACS and ImageNet-R are kept as nulls; the exact-rank
  regeneration moved ImageNet-C SAR from 0.0108 to 0.0264 and you promoted the worse number; Office-Home's
  promoted scoring is the *less* flattering of the two available. `tab:abl-transfer` publishes the
  configuration where your own guarantee breaks (SAR→Tent, FA_u = 0.255). This is rare and you should say so
  more loudly, not less.
- **Definitional hygiene.** `rem:four-quantities` (`theory_setup.tex:77-96`) separates M, γ, β and ε with an
  explicit table and states that ε is not an estimate of β; R1 searched for a place where they are conflated
  in the compiled build and found none. `def:risk-align` is stated as a definition with "Fitting a benefit
  regressor does not certify risk alignment" immediately after. Boundary cases (|M| = β > 0, M = β = 0,
  β = 0) each get their own clause.
- **`lem:nonid` is correct.** The Bernoulli-kernel construction stays inside `C_β`, the quantifiers are
  correctly ordered, the evidence laws genuinely coincide because Z is label-free, and the β = 0 carve-out
  is both correct and necessary. R1 re-derived the whole impossibility chain independently. The theorem is
  sound; only its novelty framing overreaches.
- **Zero `sorry`, `axiom` or `admit` in all 27 Lean files**, with an accurate volunteered inventory of what
  Lean does *not* cover (`kbound_short_appendix.tex:261-269`). Better disclosure than most formalization
  appendices, even though the file names overclaim.
- **`theory_v2/minimax_optimality_theorem.tex` contains a real Le Cam two-point argument** with the
  three-way-to-two-way reduction done cleanly, the correct `ε_n < Δ/2` achievability step (the one most
  people get wrong), and an explicit statement that it proves order-optimality, not tight constants.
- **The pre-registration is genuine**, with a report-all-arms rule, a schema guard that exits rather than
  fabricate, and `prohibited_after_unblinding: any re-tuning`. The withdrawals are substantive:
  `claim_ledger.json` records KB-CLAIM-022's `calibration_method: "in_sample_radius"` in plain text.

---

## Recommended next 5 concrete actions

1. **Fix the abstract (30 min).** Delete "domain, and rendition" from `kbound_short.tex:41`, mirror the edit
   at `:85` and `:1221`, and promote the already-correct scoped sentence at `:580`. Nothing else in this
   list matters if a reviewer stops reading at the abstract. Then grep every universal quantifier in the
   paper ("every", "all", "uniformly", "5/5") against Table XV before you touch anything else.

2. **Declare one quantile rule and regenerate everything downstream (half a day).** Pick exact rank.
   Regenerate `tab:imagenetc-perseed`, `decision_metrics.json`, `uniform_verdicts.json` and the CIFAR
   manifest rows under it. Change `kbound_short.tex:801` and `kbound_short_appendix.tex:287` to "2/5 seeds,
   with three exact ties to always-freeze". Fix the config table at `:549-550`. This closes items 2 and 11
   and removes the appearance of rule-shopping, which is the accusation that would do the most damage.

3. **Re-run every radius leave-one-out-of-pool and rewrite `PHASE6_LEAKAGE_AUDIT.md` (1 day).** You already
   know the answer for CIFAR — I measured it: nothing changes. ImageNet-C becomes 0.0289 / FA_u = 1/135 and
   still beats freeze. Camelyon17 Table VIII needs a genuine re-score. Publishing "we found this ourselves,
   here is the corrected number, the conclusion is unchanged" is a much stronger position than having a
   reviewer find it, and after this fix your leakage audit becomes true.

4. **Make `reproduce_submission.sh` and `_locked_analysis_script.py` run on a clean checkout (1 day).**
   Commit the seed-0 per-condition dumps (or repoint and add the missing `a_oracle` field), fix
   `test_calibration_split_integrity.py:10-11`'s impossible path, materialize the 142 NUL artifacts, and
   write `DATA.md`. A reviewer who can run one command and see your numbers appear will forgive a great
   deal; one who runs your own command and gets a `FileNotFoundError` in step 1 will forgive nothing.

5. **Run the two experiments that convert your weakest claims into strong ones (2-3 days).** (a) The
   real-data β sweep: compute M from the ATC-style score at `kbound_short.tex:364` on the CIFAR grid, declare
   β from historical dev-to-deployment gaps, sweep β ∈ {0, 0.02, 0.05, 0.10, 0.20}, and run the population
   frontier rule against Δ̂ ± ε. This is the only thing that connects your theory to your experiments, and a
   negative result still publishes. (b) The BN-statistics-only arm plus a SAR control at official settings
   (lr 2.5e-4, layer4 frozen). Both are questions every reviewer will ask; answering them pre-emptively is
   worth more than any additional benchmark track.

**What to defend, not fix:** the CIFAR-10-C safety result, `lem:nonid`, the evidence-tier policy, the
quarantine practice, and the retained negatives. When you narrow the paper, narrow it *toward* those.
