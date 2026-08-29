# Comparison family, arm inventory, and multiplicity correction

> **SUPERSEDED HISTORICAL SEARCH CENSUS.** This 2026-07-26 document did not freeze a reproducible
> file list and commit for its repository-wide counts, and its proposed global preregistration/Holm
> interpretation is not part of the Phase-1 release. Retain it only as search-history context. The
> current authority is `claim_ledger.json` plus `paper/generated/uniform_verdicts.json`, whose
> `wave_holm` family is empty. A separate retrospective current-policy six-family sensitivity now
> exists, but every candidate fails the preregistered six-comparison Holm gate; current-policy
> POEM/AETTA inference remains pending.

**Written 2026-07-26** to close fix-queue item 24 (F3-9). This document declares the confirmatory
comparison family, publishes the search that produced it, and states what correction applies.
It is the prose companion to `paper/generated/uniform_verdicts.json`'s `wave_holm_family` field and
to the appendix arm-inventory table.

---

## 1. The problem this document exists to fix

Before 2026-07-26, `uniform_verdicts.json` declared the multiplicity family as:

> `"wave_holm_family": "3 beats-both candidates (CIFAR-10-C tent/eata vs adapt; ImageNet-C sar vs
> freeze); all survive Holm at 0.05"`

The family consisted of **exactly the three comparisons that came out positive**, and it was
written down after the results were known. That is not a multiplicity family; it is the selected
subset. A Holm correction over the winners is arithmetically valid and inferentially empty.

The scale of the surrounding search is not small, and the paper published no arm inventory at all.
Measured on this tree on 2026-07-26 (786 readable JSON files under `experiments/kbound/**`,
`research_lock/*.json` and `docs/research/kbound/**`):

| quantity | count |
|---|---|
| `"beats_both"` determinations recorded | **1 427** |
| of which `true` | **345** (24.2%) |
| of which `false` | 1 070 |
| distinct directories containing at least one | 118 |
| `"verdict_win"` determinations | 97 (40 true, 57 false) |
| pre-registered protocol files in `research_lock/` | 69 `.yaml` + 12 `.md` = **81** |
| named win-search campaigns | WIN_HUNT v2 (arms A, C), v3 (D, E, F, G), v4 (A, B, C, D, F), v5 |

(The review panel's independent count over a slightly narrower file set was 1 387 / 326 / 23.5%.
The two counts agree to within the file-set difference and the conclusion is identical.)

The directory names — `win_hunt_v*`, `win_finder_v*`, `win_loop_v1`, `hard_dataset_win_loop_v1` —
describe the process accurately. There is nothing wrong with searching. What is wrong is
reporting a family-wise error statement computed over three of 1 427 determinations without
publishing the denominator.

---

## 2. The confirmatory family, declared prospectively

**Membership rule.** A comparison is a member of the confirmatory family if and only if all four
hold:

1. It is a `(track, candidate)` pair named in a file under `research_lock/` **before any number
   for that pair was computed**, evidenced by that file's own `Registered:` date and
   `status: preregistered_before_heldout` / `locked_pre_registration`.
2. The protocol states its own beats-both bar (both regret gaps, or the gap to the better fixed
   policy, with a CI excluding zero, plus `FA_u <= alpha`) *in the same file*.
3. It is evaluated **once**, on the declared held-out cells, with no re-roll.
4. The protocol version was not superseded before scoring.

**Explicitly excluded, and why:**

| excluded class | reason |
|---|---|
| development-screen rows (`dev_screen`, `cal_seeds`) | by construction they exist to select the arm that is then tested once; counting them in the confirmatory family would double-count the selection |
| WIN_HUNT v2-v5 campaign arms | exploratory search over datasets and adapters; these are the *reason* a correction is needed, not members of the family being corrected |
| superseded protocol versions (`OFFICEHOME_PROTOCOL_M_v1`, `KBOUND_6_DATASET_PANEL_v2`, `CAMELYON17_PROTOCOL_G_v1`) | replaced before scoring; their numbers are not promoted |
| post-hoc re-scorings (`g8_exactrank_regen`, the 2026-07-26 LOO re-scoring) | these change the *rule* on already-registered arms; they are sensitivity analyses of family members, not new members |
| smoke, debug and `_partial` runs | not scored against a bar |

**Consequence that must be stated in the paper:** the exploratory rows are *not* corrected for, and
they are not claimed to be. The correction below applies to the confirmatory family only, and the
1 427-determination denominator is published so a reader can price the exploratory layer for
themselves.

---

## 3. The arm inventory

Every pre-registered `(track, candidate)` pair with a stated beats-both bar, with its registration
date, its declared expectation, and its realized verdict. Registration dates and expectations are
quoted from the protocol files.

| # | protocol file | registered | track / candidate | declared expectation | realized verdict | in confirmatory family? |
|---|---|---|---|---|---|---|
| 1 | `STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml` | 2026-06-11 | CIFAR-10-C / tent | beats-both bar stated | **beats-both** | yes |
| 2 | " | " | CIFAR-10-C / eata | " | **beats-both** | yes |
| 3 | " | " | CIFAR-10-C / sar_fixed | " | **withheld** (seed-0 aggregate non-reproducing) | yes — and it is a *failure*, reported as one |
| 4 | `STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml` | 2026-06-25 | CIFAR-10-C strict v2 / tent, eata, sar_fixed | `beats_both: regret_kga < adapt AND < freeze with Holm CI excluding 0` | **unrun** (gap G7) | no — never executed |
| 5 | `imagenetc_protocol_E_v1.yaml` | 2026-06-21 | ImageNet-C / sar (official gentle schedule, 19 corruptions, sev 4-5) | "all publishable; declared before any result" | **unrun** (gap G5-adjacent) | no — never executed |
| 6 | ImageNet-C SAR 5-seed pooled (scored under `WIN_HUNT_v5` shell, re-scored under G8 exact rank) | 2026-07-09 / re-scored 2026-07-20 | ImageNet-C / sar | beats-both vs the better fixed policy | **beats-both on the point estimate; CI does NOT survive the item-4 radius fix** (see §5) | yes |
| 7 | `OFFICEHOME_PROTOCOL_M_v2.yaml` | 2026-06-19 | Office-Home / sar_online_aggressive (dev-locked from a 6-arm panel) | `heldout_beats_both: true` | **no-harm only**; LOO beats-both explicitly NOT promoted (`KBOUND_WIN_BOOTSTRAP_CIS_oof.json`: `beats_both_robust: false`) | yes — and it is a *failure of the declared bar*, reported as one |
| 8 | `IWILDCAM_PROTOCOL_H_v2.yaml` | 2026-06-19 | iWildCam / dev-locked adapter | `heldout_beats_both: true` | **no-harm** (ties freeze) | yes — declared bar not met |
| 9 | `CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml` | 2026-06-20 | Camelyon17 / eata_online | supersedes the v1 win claim | **`NOT_A_BEATS_BOTH_WIN`** — reclassified `no_harm_safety_abstention` | yes — declared bar not met, and the reclassification is self-reported |
| 10 | `RXRX1_PROTOCOL_J_v1.yaml` | 2026-06-16 | RxRx1 / sar_online | `expected_outcome: FREEZE_ORACLE_AUDIT_PASS_not_beats_both` | **no-harm** — as predicted | yes; a pre-declared negative |
| 11 | `CIFAR101_PROTOCOL_K_v1.yaml` | 2026-06-16 | CIFAR-10.1 v6 / tent | `expected_outcome: LOW_MARGIN_AUDIT_PASS_not_beats_both` | **transfer bar FAILED** (FA_u 0.167, FA_c 0.444) | yes; a pre-declared negative that came out worse than declared |
| 12 | `IMAGENETR_DIVERSE_PANEL_PROTOCOL_D_v1.yaml` | 2026-06-13 | ImageNet-R / 10-backbone panel | tests whether a non-co-adapted panel passes the structural diagnostic | **null diagnostic**; 0 of 10 backbones CI-supported beats-both; KGA worse than always-adapt on 7 of 10 | yes; a null |
| 13 | `PACS_VLCS_PREREG_PROTOCOL_v1.md` | 2026-06-22 | PACS / dev-locked adapter, 4 LODO targets | "a null is reported as a null, with no re-tuning" | **null diagnostic** | yes; a null |
| 14 | " | " | VLCS / dev-locked adapter, 4 LODO targets | same | **unrun** | no — never executed |
| 15 | `mixed_protocol_oof_v2.yaml` | 2026-06-25 | three-source constructed mixture (Office-Home + iWildCam + Camelyon OOD, n=143) | `beats_both: both regret gaps CI exclude 0` | **beats-both** — but it is a *constructed routing mixture, not transfer*, and is labelled so in `LOCK_SEAL.json` | yes, with the construction caveat |

**Family size: 11 executed confirmatory arms** (rows 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, plus
row 15 as a constructed-mixture arm = 12 if the mixture is counted; the paper should state which).
Three registered arms (4, 5, 14) were never executed and are reported as unrun.

**Positive rate inside the declared family: 3 of 12** (CIFAR-10-C Tent, CIFAR-10-C EATA,
three-source mixture), plus ImageNet-C SAR as a point-estimate-only positive after the radius fix.
That is the number a reader should compare against the 24.2% project-wide `beats_both` base rate —
and it is *lower*, which is the honest and slightly reassuring reading: the pre-registered arms
were harder than the exploratory ones.

---

## 4. The correction that applies

**Holm-Bonferroni over the executed confirmatory family, one p-value per arm, reported for every
arm including the failures.** Not over the winners.

Two facts about how much work Holm can actually do here, both of which must be stated:

1. **The bootstrap p-values are at the floor.** `_locked_analysis_script.py` applies Holm honestly
   over its own six comparisons, but all six raw p-values equal the bootstrap floor
   `1/(10^4 + 1) = 9.999e-5`. At that resolution Holm at m = 6 or m = 12 changes nothing, because
   `12 x 9.999e-5 = 1.2e-3 < 0.05`. Increase the replicate count if a finer p-value is wanted;
   otherwise report the floor explicitly and say Holm is not binding.
2. **The binding constraint is not multiplicity, it is the unit of analysis and the radius.**
   The ImageNet-C freeze-gap interval survives the in-pool radius at the cell level and does *not*
   survive the leave-one-out-of-pool radius at the seed-averaged condition level
   (`[-0.0085, +0.0038]`, `NUMBERS_PACK.md §0.1`). Similarly, CIFAR-10-C EATA's adapt-gap
   interval excludes zero at 432 i.i.d. cells and does **not** exclude zero when clustered by
   corruption family (`[-0.00436, +0.00035]`, `NUMBERS_PACK.md §0.2`). No multiplicity correction
   touches either of those; they are design questions and they are larger than the Holm
   adjustment.

**Recommended statement for the paper:**

> The confirmatory family is the set of pre-registered `(track, candidate)` beats-both bars listed
> in Appendix `tab:arm-inventory` — 12 arms, of which 3 met the bar. Holm-Bonferroni over that
> family leaves all three positives significant; the correction is not binding because every raw
> bootstrap p-value is at the resolution floor `1/(B+1)`. We also record that 1 427 `beats_both`
> determinations exist project-wide across the exploratory search campaigns (24.2% positive), and
> we do not correct for those: they selected which arms to pre-register, and no post hoc
> correction can undo that. Readers should treat the exploratory layer as hypothesis generation.

---

## 5. What changes if the item-4 radius fix is adopted

The radius fix is not a multiplicity issue but it moves a family member across the bar, so it
belongs in this document:

| arm | before the fix | after the fix |
|---|---|---|
| CIFAR-10-C Tent | beats-both, FA_u 0 | **unchanged** — 0 of 9 504 decisions change |
| CIFAR-10-C EATA | beats-both, FA_u 0 | **unchanged** — 0 of 9 504 decisions change |
| ImageNet-C SAR | beats-both (CI on the freeze gap excludes zero) | **point-estimate no-harm only**; 0.0289 vs 0.0319, freeze-gap CI `[-0.0085, +0.0038]` includes zero; FA_u 0/135 -> 1/135 |
| three-source mixture | beats-both | not re-scored (built from tracks whose record files are absent) |

So the defensible count of CI-supported beats-both arms after the fix is **2** (CIFAR-10-C Tent and
EATA), plus the constructed mixture if it is counted, plus ImageNet-C SAR as a point-estimate
no-harm. Holm over 2 or 3 arms at p = 9.999e-5 is trivially satisfied and should be described as
such rather than presented as a hurdle cleared.

---

## 6. Maintenance rule

Any new `(track, candidate)` arm must be added to §3 **at registration time**, with an empty
verdict column, before it is run. An arm that first appears in this table with its verdict already
filled in is by definition not a member of the confirmatory family. The appendix table
`tab:arm-inventory` is generated from this document; keep them in sync.
