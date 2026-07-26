# K-Bound Phase 6 — Data-Leakage & Timing/Ordering Audit

Read-only audit, originally run 2026-07-21. Scope: split disjointness, in-sample-radius defect,
certificate/oracle timing, multi-seed pooling.
Auditor replay: `numpy` on the canonical raw JSONs. Every finding is quoted `file:line`.

---

# CORRECTION 2026-07-26 — the 2026-07-21 VERDICT was WRONG

**This document certified, on 2026-07-21, that no live promoted track computed the conformal
radius in sample on the cells it scored. That certification was false.** Five shipped scripts and
seven copy-pasted `decide_kga` forks did exactly that. The error was found by an external review
panel (findings F1-1, F2-2, F3-13, F4-2, F4-3) and confirmed by an independent recompute
(`NUMBERS_PACK.md §4`). The original verdict line and the original section (b) are preserved
verbatim below under **§(b-OLD)** so the correction can be diffed, not just read.

Nothing about this correction was a quiet edit. The 2026-07-21 verdict is retracted in full.

## What was actually found

The 2026-07-21 audit inspected the *natural-shift* scoring path
(`analyze_F.py`, `score_kbound_holdout.py`) and generalized its clean bill of health to the whole
panel. Those two files were and are clean. The **controlled** tracks — the two that carry the
paper's headline beats-both claims — go through a different code path, and that path pooled the
scored cell's own residual into its own radius:

| file (at the time of the audit) | offending construction | track it produces |
|---|---|---|
| `scripts/cifar_tent_mps_v2.py:162` | `eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))` over all N residuals | CIFAR-10-C stress grid + head-to-head (headline) |
| `scripts/g8_canonical_pooling.py:11` | `rho = np.abs(bh - B); eps = cexact(rho)` per file, then used on every cell in that file | ImageNet-C SAR pooled (headline) |
| `scripts/g8_exactrank_regen.py:26` | same construction, exact-rank branch | ImageNet-C exact-rank regeneration (G8) |
| `scripts/run_wilds_camelyon17.py:56` | same construction, interpolated branch | Camelyon17 Table VIII |
| `scripts/ablation_exactrank.py:88-89` | same construction | `tab:abl-exactrank` |
| 7 further `decide_kga` forks | copies of the same body | see fix-queue item 15 |

`B` is the realized test benefit. Taking a quantile over `|Bhat − B|` **including cell i's own
residual** and then deciding cell i makes ε a function of the very test labels that the
`FA_u ≤ α` guarantee attaches to. That is in-sample calibration, and it is the exact defect this
audit was commissioned to look for.

**The audit's own citation concealed it.** Row 1 of the §(a) table cited
`scripts/cifar_tent_mps_v2.py:143-156` as the evidence that the CIFAR path was cross-fit. The
offending line was **162** — six lines past the end of the quoted range. The cited range covers
the estimator's leave-one-cell-out loop, which *is* cross-fit; the radius computed immediately
after it was not. Cross-fitting the estimator and pooling the radius are two different things,
and this document conflated them.

## Blast radius (recomputed, not asserted)

Source: `NUMBERS_PACK.md §4` — every row below was re-derived from the committed per-cell
artifacts under both the in-pool rule (as published) and the leave-one-out-of-pool rule (the fix).

**CIFAR-10-C — nothing changes. This is the strongest single fact in the correction.**

| tree | cells | decisions changed by the fix | FA_u |
|---|---|---|---|
| stress grid Tent+EATA, seeds 1-4 | 3 456 | **0** | 0 before and after |
| head-to-head Tent, seeds 0-4 | 2 160 | **0** | 0 before and after |
| head-to-head EATA, seeds 0-4 | 2 160 | **0** | 0 before and after |
| stress grid SAR, seeds 1-4 | 1 728 | **0** | 0 before and after |
| **total** | **9 504** | **0** | |

The regret triples are bit-identical in-pool vs LOO on every one of these groups. The flagship
CIFAR-10-C safety result does not depend on the defect.

**ImageNet-C SAR — the fix costs one false adapt.**

| rule | KGA | adapt | freeze | FA_u | ADAPT/FREEZE/ABSTAIN |
|---|---|---|---|---|---|
| exact rank, in-pool (as published) | 0.026422 | 0.052933 | 0.031894 | 0/135 | 12 / 14 / 109 |
| exact rank, LOO (the fix) | 0.028893 | 0.052933 | 0.031894 | **1/135 = 0.0074** | 13 / 15 / 107 |

2 of 135 decisions change. The point-estimate no-harm ordering against always-freeze survives
(0.0289 < 0.0319). The **interval** does not: under the LOO radius at the seed-averaged unit of
analysis the freeze-gap CI is `[-0.0085, +0.0038]` and includes zero (`NUMBERS_PACK.md §0.1`).
ImageNet-C SAR therefore supports a point-estimate claim and **not** a CI-supported beats-both
once this defect is fixed. Also affected: ImageNet-C EATA FA_u 0/135 -> 1/135 (3 decisions).

**Camelyon17 Table VIII — must be re-scored, and the fix makes the SAR row worse.**

| candidate | rule | KGA | adapt | freeze | FA_u | FA_c | ADAPT |
|---|---|---|---|---|---|---|---|
| SAR | interp in-pool (published) | 0.041016 | 0.000217 | 0.065430 | 1/36 = 0.0278 | 0.143 | 7 |
| SAR | interp LOO (the fix) | 0.041124 | 0.000217 | 0.065430 | **2/36 = 0.0556** | **0.250** | 8 |
| Tent | either | 0.020074 | 0.138021 | 0.020074 | 0/36 | — | 0 |
| EATA | interp in-pool (published) | 0.039280 | 0.041667 | 0.042426 | 0/36 | 0.000 | 1 |
| EATA | interp LOO | 0.035373 | 0.041667 | 0.042426 | 0/36 | 0.000 | 2 |

Two structural facts about this table that the 2026-07-21 audit should have stated and did not:

1. At n = 9 per seed, the exact-rank index is `k = min(9, ceil(10 × 0.9)) = 9`, so ε **is the
   maximum residual** and FA_u is forced to exactly 0. The exact-rank column of Table VIII
   carries no information. The same degeneracy holds for RxRx1 and ImageNet-R (n = 12, k = 12).
2. The published "over-freezes" verdict for Camelyon17 SAR is a consequence of an ε of 0.07-0.09
   on a track whose benefits are ~0.001 in magnitude. It is a statement about the radius, not
   about the adapter.

**Every other track.** Decisions changed by the LOO fix (exact rank): Camelyon17 SAR 1,
CIFAR-10.1 Tent 1, CIFAR-10.1 SAR 1, ImageNet-R efficientnet_b0 1, ImageNet-R resnet152 1,
everything else 0. PACS cannot be re-scored from the release at all — its per-cell dumps carry
`Z, a0, aa, B` but no `b_hat`, no `eps_conformal` and no decision, and seed 0 has no per-cell dump.

## What was fixed

- **Code (fix-queue item 4, owned by the library slice).** The default calibration in
  `kbound_decide.decide_kga` — the single path that `cifar_tent_mps_v2.py`,
  `run_wilds_camelyon17.py`, `g8_canonical_pooling.py`, `g8_exactrank_regen.py` and
  `ablation_exactrank.py` now all call — is leave-one-out-of-pool: cell i's radius uses the other
  n−1 residuals only, and ε is returned as a per-cell ndarray rather than a scalar.
  `calibration="in_pool"` is retained solely to reproduce a pre-fix archived number.
- **A regression test** asserting the scored index is excluded from its own radius pool.
- **This document.** The verdict is retracted; §(a) row 1's citation is corrected to the post-fix
  `decide_kga`, and the pre-fix line number 162 is recorded above so the history is auditable.

## What is still open after this correction

1. **Camelyon17 OOD is `sealed but not recomputable from release`, not `locked`.** The promoted
   regret triple `0.0000/0.0000/0.1381 (n=18)` is recorded in exactly one sealed file
   (`research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29`) and in no computable artifact;
   the promoted `FA_u = 0` is recorded nowhere at all. The directory that file cites as its
   evidence, `audits/integrity_2026-06-20/camelyon_reconciliation/`, does not exist — those are
   the only 2 of 72 sealed files that are missing, and they are also the sole documentary basis
   for the KB-CLAIM-022 withdrawal argument reproduced in §(c) below. See `SUBMISSION_LEDGER.md §8`
   for the restoration procedure. Until it is executed, §(c) below is an argument whose underlying
   record is a summary, not data.
2. **The exact-rank FA_u ceiling.** Under in-pool rank calibration `FA_u ≤ (N−k)/N` is an
   arithmetic identity — 0.0972 at n = 432, 0.0370 at n = 27, exactly 0 at n ≤ 9 — so
   "FA_u ≤ α holds on every track" is not a measurement. The informative statistic is FA_u = 0
   *versus that ceiling*, plus the ADAPT count and a Clopper-Pearson bound on FA_c. Fix-queue
   item 5.
3. **`claim_ledger.json:168`** still points KB-CLAIM-022 at
   `archive/audit_only/camelyon17_protocol_G_pooled_beats_both`, which is not materialized on disk.
4. **`research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97`** still re-lists `id_val` in the Camelyon
   split reference. `SUBMISSION_LEDGER.md` marked this `[RESOLVED]` under G9; only the
   `bootstrap_win_cis.py` half was actually done.

---

## (a) Per-track split / ε-source table — CORRECTED

Two things are being tracked in this table and the 2026-07-21 version merged them: whether the
**estimator** `Bhat` is cross-fit, and whether the **radius** ε is. The natural-shift tracks were
clean on both. The controlled tracks were clean on the first and not on the second.

| Track (promoted) | Calibration / dev split | Test split scored | Estimator cross-fit? | Radius ε in-pool before the fix? | Scorer |
|---|---|---|---|---|---|
| CIFAR-10-C stress | leave-one-cell-out (per cond., within seed) | the held-out cell (jackknife) | yes | **YES — defect** (`:162`, pre-fix) | `scripts/cifar_tent_mps_v2.py` `decide_kga` (post-fix) |
| ImageNet-C SAR (5-seed) | per-seed leave-one-cell-out (27 cells) | same seed's 27 cells; decisions pooled | yes | **YES — defect** (`g8_canonical_pooling.py:11`, pre-fix) | `scripts/g8_canonical_pooling.py` |
| PACS (LODO, 4 targets) | one source domain (`calibration_domain`) | target domain, 18 cells | source ≠ target | unverifiable — no `b_hat`/ε in the released per-cell dumps | `results/win_hunt_v5/pacs_aggr/pacs_result.json` |
| Camelyon17 OOD | dev seeds {0,1} | OOD test-only seeds {2,3,4}, n=18 | yes (LOO OOF on dev) | no (dev-only ε) — **but see §(c): the artifact is absent** | `scripts/analyze_F.py:186-199` |
| Camelyon17 Table VIII (per-candidate) | in-pool over 9 cells × 4 seeds | the same cells | yes | **YES — defect** (`run_wilds_camelyon17.py:56`, pre-fix) | `scripts/run_wilds_camelyon17.py` |
| iWildCam H v2 | calib seed {0} | held-out seed {1}, n=72 | yes | no (`eps_global=0.02937` from cal only) | `scripts/score_kbound_holdout.py:75-91` |
| Office-Home M v2 | calib target-**val** {0,1} | target-**test** {0,1}, n=35 | yes | no (`eps_global=0.00102` from cal only) | `scripts/score_kbound_holdout.py:64-106` |
| RxRx1 J | dev seeds {0..4} | test seeds {5..9}, n=60 | yes | no (LOO OOF on dev) | `scripts/analyze_F.py:186-199` |

The split-disjointness half of the original audit stands: no promoted track scores a test
partition that overlaps its calibration partition, and no track uses target labels to select the
adapter, α, or the decision threshold. That was never the defect. The defect was the radius.

## (b) Leakage found — CORRECTED

**Found: in-sample radius calibration on 5 shipped scripts and 7 `decide_kga` forks**, listed in
the correction header above. Fixed in code 2026-07-26 (fix-queue item 4).

The two *natural-shift* scoring scripts do carry a genuine in-fold guard, and that part of the
2026-07-21 finding is confirmed:

- `analyze_F.py:186-193` — "Out-of-fold (leave-one-out) residuals for the conformal radius -> no
  in-sample leakage. (The in-sample radius was ~10x too small on small dev sets; see audit
  2026-06.)" ε is `conformal_rank_radius(resid_c)` where `resid_c` = LOO residuals on the
  **calibration** rows only (`:189-193, :198`); decisions are `decide_global(Bhat_t, eps)` on test
  (`:199`).
- `score_kbound_holdout.py:75-85` — same LOO guard on the calibration file; ε applied to a
  *separate* test file (`:89-91`).

Note the irony the 2026-07-21 audit missed: the June-2026 fix quoted in that docstring
("the in-sample radius was ~10x too small on small dev sets") is a fix for **precisely** the
defect that was still live in the controlled path.

Decision-time label use is confirmed clean and unchanged:

- `certificate.py` `decide(Bhat, eps)` takes only `(Bhat, ε)`.
- `analyze_F.py:118-135` `metrics()` and `run_wilds_camelyon17.py` `policy_metrics()` compute
  false-adapt from true `B` **for post-hoc scoring only**; `dec` is produced independently.

Replay evidence (ImageNet-C SAR, `results/win_hunt_v5_imagenetc_ms/pooled_5seed`), retained from
the original audit and still correct *as a description of the published, in-pool numbers*:

- Per-seed exact-rank in-pool ε = [0.084, 0.108, 0.046, 0.084, 0.072]; per-seed pooled →
  KGA/adapt/freeze = 0.0264/0.0529/0.0319, FA_u = 0. Under the LOO fix this becomes
  0.0289/0.0529/0.0319, FA_u = 1/135.
- The original audit read "ε(seed0)=0.084 is substantial (not ~0)" as evidence *against*
  in-sample calibration. That inference was invalid: a pooled-residual quantile at n = 27 is not
  driven to zero by including one extra residual, so its magnitude carries no information about
  whether the scored cell was in the pool. Magnitude is not a leakage test; index exclusion is.

### (b-OLD) The retracted 2026-07-21 text, preserved verbatim

> **VERDICT: PASS (clean). No live promoted track computes ε in-sample on the cells it scores. No
> promoted number uses target labels to choose ε or the decision threshold. KB-CLAIM-022 confirmed
> quarantined.**
>
> **(b) Leakage found: NONE.** "The two scoring scripts carry an explicit in-fold guard (the fix
> for the June-2026 defect)."
>
> §(a) row 1, CIFAR-10-C stress: "ε is q0.9 of **LOO** residuals … Scorer
> `scripts/cifar_tent_mps_v2.py:143-156` … PASS (cross-fit; no cell's estimator saw it)."

Retracted 2026-07-26. The VERDICT's sentence 1 is false as stated (5 scripts + 7 forks).
Sentence 2 is true and still holds. Sentence 3 is unverifiable from the release — see §(c). The
§(a) citation is off by six lines in the direction that hides the defect.

## (c) KB-CLAIM-022 quarantine — ARGUMENT INTACT, ARTIFACT MISSING

The withdrawal reasoning below is unchanged and, as reasoning, we still endorse it. What changed
is its evidentiary status: **the file it rests on is not in the release.**

- Ledger: `claim_ledger.json:159-173` — `status="withdrawn"`,
  `calibration_method="in_sample_radius"`, `test_split="pooled id_val (invalid)"`. This is on disk
  and verifiable.
- The root-cause reconstruction below is quoted from
  `audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json`, **which does not exist
  in this repository**. Of 69 files sealed in `nine_track_lock_v1/LOCK_SEAL.json`, 67 verify
  byte-for-byte; the only 2 missing are this file and its sibling `VERDICT_phase1.md`, both
  Camelyon17.
  - `POOLED_test_val_idval` (the withdrawn artifact): `beats_both=true, preregistered_win=true,
    regret_kga=3.6e-05, n_test=54`. The "win" exists only because `id_val` (frac_harm `B<0` =
    0.767, mean_B = −0.0075) is pooled into the genuinely-helpful OOD test/val domains.
  - `OOD_test_only` (the promoted reconciled result): `regret_kga=0.0, regret_adapt=0.0,
    regret_freeze=0.1381, false_adapt=0.0, n_test=18, beats_both=FALSE`.
- **Independent check of the promoted triple — and a correction to the review panel.** The panel
  (F3-6, F4-12) reported that `0.1381` "appears in no artifact on disk". That grep was restricted
  to `*.json`. Widened to all file types, the triple appears in exactly one place:
  `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29` —
  `OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381,
  beats_both: false}`. That file **is** sealed and its hash verifies byte-for-byte. So the number
  was written down, under change control, before the paper cited it.
  Three things remain wrong, and they are what matters:
  1. The YAML is a hand-transcribed summary of a rerun. **Nothing recomputes the triple.** The
     three artifacts it names as its own evidence (`camelyon_G_reconciliation.py`,
     `recon_results.json`, `VERDICT_phase1.md`) are all absent; two are sealed-and-missing, the
     script was never sealed at all.
  2. **The promoted `FA_u = 0` is recorded nowhere.** The YAML's only Camelyon false-adapt figure
     is `idval_only: {false_adapt: 0.80}`; the `OOD_test_only` entry has no false-adapt field.
  3. The nearest live artifacts give nonzero false-adapt on their own (different) slices:
     `camelyon17_protocol_G_v1/analyze_F_results.json` 0.0256 at n = 54,
     `camelyon17_richZ_F_v1/analyze_F_results.json` 0.0329 at n = 324.

Consequence, applied throughout the release documentation as of 2026-07-26: **Camelyon17 OOD is
labelled "sealed but not recomputable from release"**, not "locked" and not "reconciled". A reader
can verify the number was recorded before it was cited; a reader cannot verify it is correct.
Restoration procedure: `SUBMISSION_LEDGER.md §8`.

The rest of the quarantine mechanics are confirmed and unchanged: the withdrawn artifact path
`archive/audit_only/camelyon17_protocol_G_pooled_beats_both` is not materialized on disk; no
result JSON under `experiments/` asserts the pooled beats-both; `uniform_scorer.py:108` still
hard-fails any contaminated split → `WITHDRAWN`.

## (d) Certificate timing / multi-seed pooling — UNCHANGED (this section stands)

- Timing: α is fixed at 0.10 everywhere (`certificate.py`, `analyze_F.py:40`,
  `run_wilds_camelyon17.py:42`, all protocol JSONs); the threshold is the fixed constant 0 in
  `Bhat±ε`. Neither α nor the threshold is selected on test. Adapter/estimator/evidence selection
  is done on dev only.
- Multi-seed pooling: `g8_canonical_pooling.py` computes ε **inside** the per-seed loop on that
  seed's own 27-cell residuals, forms per-seed decisions from `b_hat`, then pools decisions for
  the aggregate. It does not fit one ε across all 135 pooled cells. Replay contrast: a single
  pooled ε gives KGA 0.0109 at ε 0.0431, against the promoted per-seed 0.0264 — confirming the
  promoted path is per-seed. This is correct and is not the defect; the defect was *within* each
  per-seed pool.

## Open fix list (superseding the 2026-07-21 "all non-blocking" list)

The original list characterised its three items as "traceability, low" and "documentation, low".
Two of them are now MAJOR.

1. **[MAJOR] Camelyon17 reconciliation directory absent.** See §(c). Restore
   `audits/integrity_2026-06-20/camelyon_reconciliation/` (its SHA-256s are already sealed, so
   restoration is independently verifiable) or re-run `camelyon_G_reconciliation.py` and re-seal.
   Blocks: the Camelyon17 panel row and the KB-CLAIM-022 argument.
2. **[MAJOR] `claim_ledger.json:168`** points KB-CLAIM-022 at a path not materialized on disk.
   Repoint to the live rationale once item 1 is done.
3. **[MINOR, resolved] RxRx1 0.2531 vs 0.2587.** Settled from the artifact:
   `rxrx1_protocol_J_v1/analyze_F_results.json` carries `"candidate": "sar_online"` and
   `regret_adapt = 0.2530598958`. The printed 0.2531 is correct. This document's 2026-07-21 claim
   that "the promoted value is the 5-seed real-ckpt rerun" was **wrong**, and so was
   `PHASE7_INTEGRATION_AUDIT`'s stated reason. One sentence in `SUBMISSION_LEDGER.md §7` now
   supersedes all three stories.
4. **[MINOR, superseded] `run_wilds_camelyon17.py:45-59` in-pool LOO ε.** This document called it
   a documentation nit affecting only a raw input file. It was in fact the Camelyon17 Table VIII
   defect (§ correction header). Fixed in code.
