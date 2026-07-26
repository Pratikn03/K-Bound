# Fix-queue report — DOCS slice

Owner: docs agent. Files owned: every `.md` and `.json` under `docs/research/kbound/` except
`paper/**`, plus `/home/claude/kb/{README.md, GAP_AUDIT.md, INTEGRITY_FIXES.md, DATA.md}`, plus
`experiments/kbound/results/nine_track_lock_v1/LOCK_SEAL.json` (explicitly assigned by item 11).
Items owned: **4 (doc half), 9, 10, 11, 19 (disclosure), 20, 24, 31**, plus the closing instruction
to update `docs/research/kbound/README.md` and `SUBMISSION_LEDGER.md` to the current state
including the TMLR venue decision.

---

## Resume audit

An earlier attempt at this slice was interrupted. State found on disk at the start of this run:

| item | state found | action taken |
|---|---|---|
| 4 (doc half) | **untouched** — `PHASE6_LEAKAGE_AUDIT.md` still opened with `## VERDICT: PASS (clean)...` | rewritten with a dated retraction |
| 9 | **untouched** — `docs/research/kbound/PLACEHOLDER_INVENTORY.md` did not exist | created |
| 10 | **untouched** — `LOCK_SEAL.json` `camelyon17_ood.status` was `"locked"`; every doc said "reconciled"/"locked" | demoted everywhere |
| 11 | **untouched** — `LOCK_SEAL.json` had no `source`/`promoted_value_location` on any track; ledger G8/P2 still contradicted | fixed |
| 19 | **untouched** — no environment-heterogeneity disclosure anywhere | added to `REPRODUCE.md §0a`, `SUBMISSION_LEDGER.md §10`, `CIFAR10C_SAR_QUARANTINE.md` |
| 20 | **untouched** — `DATA.md` did not exist anywhere in the tree | created |
| 24 | **partially done by another agent** — `paper/generated/uniform_verdicts.json`'s `wave_holm_family` was already rewritten prospectively (paper agent's file, not mine) and referenced an appendix table that did not exist | wrote the docs-side companion `COMPARISON_FAMILY.md` with the arm inventory it points at |
| 31 | **untouched** — `SUBMISSION_LEDGER.md §0` still pinned the stale PDF sha256; `STORAGE_MANIFEST.json` still carried the drifted `claim_ledger.json` hash; no superseded stamps anywhere | all four sub-items done |

Two things done by *other* agents that I detected and worked around, rather than duplicating:

- The library agent had already applied the **code** half of item 4 (`kbound_decide.decide_kga`
  now defaults to leave-one-out-of-pool; the five shipped scripts call through it). My item-4 work
  is documentation only and describes the fix in the past tense, correctly.
- `docs/research/kbound/experiments/kbound/results/ablation_exactrank.json` had been regenerated
  and is now readable, which is why my placeholder count is **143** where the recompute agent's was
  145. I re-scanned rather than copying their number.

---

## Item 4 (doc half) — **DONE**

`docs/research/kbound/PHASE6_LEAKAGE_AUDIT.md` — rewritten (now ~250 lines vs 61).

**The core rhetorical change.**

OLD (`:5`, the entire verdict, one line):
> `## VERDICT: PASS (clean). No live promoted track computes ε in-sample on the cells it scores. No promoted number uses target labels to choose ε or the decision threshold. KB-CLAIM-022 confirmed quarantined.`

NEW (a titled correction block at the top of the file):
> `# CORRECTION 2026-07-26 — the 2026-07-21 VERDICT was WRONG`
> "**This document certified, on 2026-07-21, that no live promoted track computed the conformal
> radius in sample on the cells it scored. That certification was false.** Five shipped scripts and
> seven copy-pasted `decide_kga` forks did exactly that. [...] Nothing about this correction was a
> quiet edit. The 2026-07-21 verdict is retracted in full."

OLD (section b heading): `## (b) Leakage found: NONE`
NEW: `## (b) Leakage found — CORRECTED` … "**Found: in-sample radius calibration on 5 shipped
scripts and 7 `decide_kga` forks**".

The retracted text is **preserved verbatim** in a new `### (b-OLD)` block so the author can diff the
rhetoric, including the off-by-six-lines citation.

**The citation defect is named explicitly**: the audit cited `cifar_tent_mps_v2.py:143-156` as
evidence the CIFAR path was cross-fit; the offending line was **162**, six lines past the end of the
quoted range. The new text explains *why* that citation was misleading rather than just correcting
it — the cited range covers the estimator's LOO loop, which *is* cross-fit; the radius computed
immediately after it was not. "Cross-fitting the estimator and pooling the radius are two different
things, and this document conflated them."

**Blast radius, every number from `NUMBERS_PACK.md §4`:**
- CIFAR-10-C: **0 of 9 504** decisions change across four trees (the queue's "0 of 3456" is the
  stress-grid subtotal; the full committed set is 9 504). Regret triples bit-identical.
- ImageNet-C SAR: 0.026422 → 0.028893, FA_u 0/135 → **1/135**, ADAPT 12→13, ABSTAIN 109→107. Point
  estimate still beats always-freeze (0.0289 < 0.0319); the seed-averaged freeze-gap CI becomes
  `[-0.0085, +0.0038]` and **includes zero** (`NUMBERS_PACK §0.1` — flagged because fix-queue item 4's
  own text says "beats-both against freeze still holds", which is true of the point estimate and
  false of the interval).
- Camelyon17 Table VIII re-scored in full (three candidates × four rule/pool combinations). The fix
  makes the SAR row **worse**: FA_u 1/36 → 2/36, FA_c 0.143 → 0.250.
- Every other track's delta enumerated (1 decision each on CIFAR-10.1 ×2 and ImageNet-R ×2).
- Two structural facts added that the original audit should have stated: at n = 9 the exact-rank
  index is `k = 9`, so ε **is** the max residual and FA_u is forced to 0 — the exact-rank column of
  Table VIII carries no information; and the "over-freezes" verdict is a statement about the radius,
  not the adapter.

**One inference in the original audit is called out as invalid**: it read "ε(seed0)=0.084 is
substantial (not ~0)" as evidence against in-sample calibration. New text: "Magnitude is not a
leakage test; index exclusion is."

**Deliberately not done:** I did not delete the original document and start over. The (a) split
table and the (d) timing/pooling section were *correct* and are retained with their verdicts intact
— section (d) is explicitly marked "UNCHANGED (this section stands)". Retracting a document
wholesale when only one of four sections is wrong would itself be a misrepresentation.

## Item 9 — **DONE** (the guard is a spec; the download is BLOCKED-NEEDS-DATA by construction)

Created `docs/research/kbound/PLACEHOLDER_INVENTORY.md` (336 lines).

**Census is my own, run fresh today**, not copied: 1 895 text-extension files scanned;
**143 zero-byte-or-NUL-in-first-4KB**; **0 whitespace-only**; 0 `OSError`. By extension: 76 `.json`,
45 `.py`, 10 `.csv`, 9 `.md`, 3 `.sh`. The document leads with the two facts that matter — a
whitespace scan returns 0 and a NUL scan returns 143, and these are iCloud dataless placeholders
carrying their full nominal byte length (`run_officehome_kbound.py` reports 17 202 bytes and reads
as NUL from byte 0).

**Grouped by what depends on them**, eight groups A-H, each with the complete file list and byte
sizes, plus a dependency table stating the consequence of each group. Key honest finding stated up
front: **no promoted panel number is blocked by a placeholder** — what blocks numbers is *absent*
files, a different failure recorded in `SUBMISSION_LEDGER.md §8`. The placeholders block
ablation/cost recomputation (group A), the entire Office-Home audit trail (group B), the physical
study (group C), and the `GAP_AUDIT.md` / `INTEGRITY_FIXES.md` evidence base (group F).

**Recovery instruction** for the author: one `find … | xargs brctl download` + `brctl log --wait`
command, the Finder equivalent, a re-verification snippet, and how to stop eviction recurring.

**Release guard written as a specification, not code** (the library agent owns `tests/`):
- **Guard 1** NUL scan — name, exact rule, why 4 KB, why `content.strip()` provably fails on this
  tree, expected state (0) vs today (143), and an allowlist policy requiring reason + owner +
  expiry, with `kbound_pkg/build/**` marked *delete, do not allowlist*.
- **Guard 2** checksum coverage — three rules (coverage of every `source` in the result manifest and
  every sealed path; integrity; existence with an explicit `"status": "absent"` rather than a silent
  null), plus the requirement that `STORAGE_MANIFEST.json` be regenerated by
  `reproduce_submission.sh` at freeze time.
- **Guard 3** freshness — non-blocking; fails only if the count increases.

**BLOCKED-NEEDS-DATA:** materializing the files themselves. They are iCloud placeholders on the
author's Mac and cannot be downloaded from this container. That is the item's stated premise.

## Item 10 — **DONE, with a correction to the review panel**

**Material finding that changes the item.** Findings F3-6 and F4-12 state the Camelyon triple
"appears in no artifact on disk", based on `grep -rn "0.1381" --include=*.json`. Widening the grep
to all file types, it appears in exactly one place:

```
research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29
  OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381, beats_both: false}
```

That file **is** sealed in `LOCK_SEAL.json` and its hash verifies byte-for-byte. Three things
remain wrong and are what actually justify the demotion:
1. the YAML entry is a hand-transcribed **summary** of a rerun — nothing recomputes the triple;
2. the promoted **`FA_u = 0` is recorded nowhere** (the YAML's only Camelyon false-adapt figure is
   `idval_only: {false_adapt: 0.80}`);
3. all three artifacts the YAML names as its evidence are absent — and I found a **third** missing
   file the review did not list: `camelyon_G_reconciliation.py`, which was **never sealed**, so its
   restoration cannot be verified against a recorded hash.

I therefore used the label **"sealed but not recomputable from release"** rather than the item's
suggested "not reproducible from release". It is more precise in both directions: a reader can
verify the number was recorded under change control before the paper cited it, and cannot verify it
is correct.

Applied in: `LOCK_SEAL.json` (`status`, `verdict`, `promoted_value_location`,
`also_absent_never_sealed`, `restoration_procedure`), `SUBMISSION_LEDGER.md §3` + new `§8a`,
`PHASE6_LEAKAGE_AUDIT.md §(c)`, `DATA.md §4b`, `results_source.json`
(`verdict`, `source`, `tier`, new `false_adapt_provenance`), `RESULT_MANIFEST.json`,
`claim_ledger.json`, `KBOUND_SHORT_RESULT_AUDIT.md`, `KBOUND_SHORT_CLAIM_MANIFEST.md`,
`REPRODUCE.md`, `REPRO_HARDENING_REPORT.md`, `REVIEWER_REPRO_PACKET.md`, `README.md` (docs and root).

**Prose diff, `results_source.json` `camelyon17_ood`:**
- OLD `"verdict": "locked no-harm (OOD reconciliation)"`,
  `"source": "audits/integrity_2026-06-20/camelyon_reconciliation/"`
- NEW `"verdict": "sealed but NOT RECOMPUTABLE from release (one-sided no-harm; OOD
  reconciliation)"`, `"source": "research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml ::
  evidence.rerun_locked_decision_gbr_global.OOD_test_only -- the only place the regret triple is
  recorded. The previously cited directory […] DOES NOT EXIST."`, plus a new
  `"false_adapt_provenance"` field saying the `0.0` is carried from the manuscript, not an artifact.

**Restoration procedure recorded** at `SUBMISSION_LEDGER.md §8` as a 7-row table: path, what it
blocks, how to restore, and — the column that matters — **whether the restoration is verifiable**
(yes for the two sealed files, no for the other five). Also registered machine-readably in
`STORAGE_MANIFEST.json` under a new `absent_required_artifacts` key.

**Also covered by this item:** the four `bootstrap_win_cis.py` record files. All four confirmed
absent; registered in `STORAGE_MANIFEST.json` and `SUBMISSION_LEDGER.md §8` rows 4-7; the affected
tracks (Office-Home, iWildCam) annotated in `results_source.json` and `RESULT_MANIFEST.json`.

## Item 11 — **DONE** (manifest half delegated to the paper agent — see below)

### `LOCK_SEAL.json`

- **`cifar10c_tent_eata`**: sealed the two files that actually contain the promoted triple
  (`mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_{tent_primary,eata_secondary}.json`, hashes
  computed from disk). Added `promoted_value_location` naming the JSON path and the exact values,
  and `canonical_aggregate` explaining that `LOCKED_ANALYSIS_RESULTS.json` is a *different* 5-seed
  aggregate (0.0016259/0.0079757/0.1239368) retained as provenance, not the panel source.
- **`cifar10_1_K`**: sealed `cifar101_protocol_K_v1/analyze_F_results.json` — the artifact the
  promoted row actually comes from — added the missing `source` field, and a `note` explaining that
  the five `cifar101_multiseed_v1/seed*/result_manifest.json` files already sealed belong to a
  different 24-condition run.
- **`rxrx1_J`**: added the missing `source` and `promoted_value_location`, plus the settled
  0.2531-vs-0.2587 account.
- **`camelyon17_ood`**: demoted (item 10).
- **Top level**: new `corrections` entry (dated, findings cited, with a `panel_correction`
  sub-field recording where the review was itself wrong), `integrity_rule` requiring every track to
  carry a `promoted_value_location`, `sealed_utc_original`, and `reseal_required`.
- Verified all 72 sealed hashes against disk: **70 present and verifying, 2 absent, 0 mismatches.**

### `SUBMISSION_LEDGER.md` G8 vs P2

The item says "un-mark it, or mark it resolved with the correct rule". I did the second, because
the contradiction turned out to be resolvable and the resolution is useful.

- OLD `:83-89` (G8, `[RESOLVED = PASS]`): "…ALL beats-both SURVIVE (… **CIFAR Tent
  0.0016/0.0080/0.1239; CIFAR EATA 0.0013/0.0033/0.1313**) … ACTION: update panel numbers to
  exact-rank values; … drop interpolated-quantile from headline path."
- OLD `:130-131` (P2): "Uniform-panel CIFAR-10-C Tent/EATA 4th-decimals **0.0080/0.1239, 0.1313 ->
  canonical 0.0079/0.1241, 0.1314**".
- NEW: a dedicated **`§5 G8 reconciliation`** showing the two are orthogonal — **G8 governs the
  RULE** (exact rank), **P2 governs the SOURCE AGGREGATE** (head-to-head, not stress grid) — with
  the arithmetic: under exact rank on the head-to-head aggregate Tent is
  `0.00158518 / 0.00792338 / 0.12409792` → `.0016/.0079/.1241`, which is exactly the published
  panel. G8's recorded `0.0080/0.1239` were the stress-grid values and G8 was never updated when the
  source aggregate changed. The item is re-labelled **G8a `[RESOLVED = PASS, restated 2026-07-26]`**
  with the canonical rule quoted in a block. The sub-item the old ledger buried *inside* a
  `[RESOLVED]` block ("Still fix FA_u marginal code label") is called out and its separate closure
  recorded.

### For the paper agent (coordination — I did **not** touch `paper/**`)

`paper/generated/kbound_result_manifest.json` needs, and I have not made, these changes:

1. `cifar10c_tent.source` and `cifar10c_eata.source` currently name
   `experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`, which
   contains `0.0016259256 / 0.0079756946 / 0.1239368049` — **not** the promoted triple. Repoint to
   `experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json`
   (`policy_mean_regret.{kga 0.001573610885275735, always_adapt 0.007923379871580337,
   always_freeze 0.1240979162355264}`) and `…_eata_secondary.json`
   (`{kga 0.0012675924985497084, always_adapt 0.003268287358460603,
   always_freeze 0.13137893428405126}`). Sha256s are now recorded in `LOCK_SEAL.json`.
2. `cifar10_1_K` and `rxrx1_J` have **no `source` field**. Add
   `experiments/kbound/results/cifar101_protocol_K_v1/analyze_F_results.json` (`test_locked`:
   `regret_kga 0.0020625007649262748`, `regret_adapt 0.019020830591519673`,
   `regret_freeze 0.0017083333333333333`, `adapt_rate 0.375`, `false_adapt` (FA_c) `0.4444444444444444`,
   `n_test 48`; `candidate: tent`) and
   `experiments/kbound/results/rxrx1_protocol_J_v1/analyze_F_results.json` (`regret_adapt
   0.2530598958333333`, `candidate: sar_online`).
3. `camelyon17_ood.source` names a directory that does not exist. The only live pointer is
   `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml` (regret triple only — the `FA_u = 0` has
   no source).
4. The appendix calls this manifest "the authoritative index"; `LOCK_SEAL.json` now carries an
   `integrity_rule` requiring a `promoted_value_location` per track. A CI check asserting each
   sealed file contains the manifest's value is specified as Guard 2 in `PLACEHOLDER_INVENTORY.md`.

## Item 19 — **DONE** (disclosure; the re-run is BLOCKED-NEEDS-DATA)

Disclosed in three places, at three levels of detail:

- **`REPRODUCE.md §0a`** — new section "Disclosure: the committed multi-seed runs were NOT produced
  under one environment", with both per-seed tables (CIFAR-10-C: three stacks; ImageNet-C: seed 0
  from a third), the four supporting facts (seed-0 `argv` omits `--severities 1 3 5` and
  `--max-images 4000`; `pooled_5seed/`'s seed-0 file is md5-identical to the older `win_hunt_v5`
  copy; `pooled_5seed/` has **no** `result_manifest.json`; **0 of 43** manifests record a
  scikit-learn version despite `b_hat` coming from `GradientBoostingRegressor(subsample=0.8)`), and
  an explicit **permitted / forbidden** rule:
  > *Permitted:* reporting the five seeds as five runs, and their spread as an upper bound on seed
  > variance. *Forbidden:* calling it a five-seed variance estimate, or attributing any seed-0
  > outlier to the seed.
- **`SUBMISSION_LEDGER.md §10`** — the ledger-level version, plus the confound with G2.
- **`CIFAR10C_SAR_QUARANTINE.md`** — this is where the disclosure bites hardest, and it added a
  gate. NEW text: "Seed 0 is not only the seed whose aggregate fails to replay. It is also the only
  CIFAR-10-C seed that ran on a different Python, a different torch and a different commit […]
  **This adds a gate 0 to the list below: the rebuild must run all five seeds under one pinned
  stack** […] A rebuild that reproduces the anomaly under a *different* environment than the
  original settles nothing."

**BLOCKED-NEEDS-DATA:** re-running seed 0 under the seeds-1-4 stack needs GPU/MPS hardware and the
raw datasets. Recorded as an open item in `SUBMISSION_LEDGER.md §10`.

## Item 20 — **DONE, with three rows honestly marked UNPINNED**

Created `/home/claude/kb/DATA.md` (11 sections). Ground rule stated at the top and followed: every
version string, split, corruption list and grid shape was read out of a committed artifact or
script **and the file it was read from is named**; checksums I could not verify are written
**NOT RECORDED** rather than guessed.

Pinned from the repository:
- **CIFAR-10-C** Zenodo **2535967** (`run_decisive_cifar.sh:19`), and the operating point verified
  by decomposing 864 `condition` strings: **6 of the 15 corruptions**
  (`gaussian_noise, defocus_blur, fog, contrast, pixelate, jpeg_compression`) at **severities
  {1,5}**, × 3 batch × 3 label regime × 2 aggressiveness × 2 replicate = **432 cells**.
- **ImageNet-C** Zenodo **2235448** (`download_all_datasets.sh:74`), operating point verified the
  same way: **3 corruptions** (`gaussian_noise, shot_noise, impulse_noise` — all one family, which
  is why the corruption-family clustering has only 3 clusters) at severities **{1,3,5}**,
  small/aggressive only, 3 label regimes = **27 cells/seed**. Also noted that `IMAGENET_C_QUICK`
  names a *different* 6-corruption subset that the promoted run did **not** use.
- **WILDS**: `wilds 2.0.0` (three independent sources named), datasets `camelyon17_v1.0`,
  `iwildcam_v2.0`, `rxrx1_v1.0`. **Pin instruction**: `download_all_datasets.sh:52` currently does
  `$PIP install -q wilds` unpinned.
- **CIFAR-10.1 v6** with both raw `.npy` URLs from `cifar_tent_mps_v2.py:467-468` — the only fully
  automatic, fully pinned dataset in the panel.
- **PACS** via HF `flwrlabs/pacs` and `export_pacs_hf.py` (revision unpinned — pin the HF commit).

Marked **UNPINNED / unobtainable**, with the author action spelled out:
- **Office-Home** — no URL, and the **split definition is unrecoverable**: `run_officehome_kbound.py`
  (17 202 B), `oh_data.py`, `oh_analyze.py` and 8 more are NUL placeholders, and both source record
  files are absent. Four-step author action listed.
- **ImageNet-R** — no acquisition path anywhere in the release. Canonical upstream named with an
  explicit **"VERIFY … before release; it is not recorded anywhere in this tree"**.

**New disclosure I found while doing this** (`§4a`): `experiments/kbound/wilds/READINESS.md:18-20`
records that the Camelyon17 runs used an internal copy that was **90.9% complete** (414 389 /
455 954 patches; center 2 = `test` 100% present, disk-filter drops the rest, logged), and
`T9_AUDIT.md:17` confirms the active run used that copy rather than the complete one. A third party
downloading complete `camelyon17_v1.0` will not reproduce the non-test-center conditions
cell-for-cell. This is separate from, and additional to, the missing-reconciliation problem.

`§10` is a blunt obtainability table (2 of 9 unobtainable, 1 partial) and `§11` is a six-item
release checklist.

`STORAGE_MANIFEST.json` correspondingly fixed: the three broken `reproduction_command` values
replaced (each with a sentence saying *why* the old one was wrong), four missing dataset entries
added (ImageNet-C, WILDS, CIFAR-10.1, Office-Home), and `data_documentation` pointing at `DATA.md`.

## Item 24 — **DONE**

Created `docs/research/kbound/COMPARISON_FAMILY.md`.

**Denominator measured myself** over 786 readable JSONs: **1 427** `beats_both` determinations,
**345 true (24.2%)**, 1 070 false, across 118 directories; 97 `verdict_win` (40/57); **81**
pre-registered protocol files (69 `.yaml` + 12 `.md`); campaigns WIN_HUNT v2-v5. The review's
independent count (1 387 / 326 / 23.5%) is cited alongside so the two are reconcilable.

**Prospective membership rule**: four conditions (named in `research_lock/` before any number for
that pair was computed; the bar stated in the same file; evaluated once on declared held-out cells;
version not superseded), plus an explicit exclusion table with reasons — dev screens, WIN_HUNT
campaign arms, superseded protocol versions, post-hoc re-scorings, smoke/debug runs.

**Full arm inventory**, 15 rows, each with the protocol file, registration date, declared
expectation quoted from the file, and realized verdict — **including the failures**: Office-Home's
declared `heldout_beats_both: true` was **not** met, iWildCam's was not met, Camelyon17 self-reports
`NOT_A_BEATS_BOTH_WIN`, CIFAR-10-C SAR is a withheld family member, and three registered arms
(strict stress grid v2, ImageNet-C protocol E, VLCS) were **never executed** and are reported as
unrun.

**The finding that makes this worth publishing**: the positive rate *inside* the declared family is
**3 of 12**, against the 24.2% project-wide base rate. The pre-registered arms were harder than the
exploratory ones — which is the reassuring direction, and it can only be said because the
denominator is published.

**Correction stated honestly**: Holm over the executed confirmatory family, one p-value per arm,
failures included — and then two reasons Holm does almost nothing here: every raw bootstrap p-value
is at the floor `1/(10^4+1) = 9.999e-5`, and the binding constraints are the unit of analysis and
the radius, not multiplicity. Includes a ready-to-paste paragraph for the paper and a §5 showing
that after the radius fix the defensible count of CI-supported beats-both arms is **2**.

## Item 31 — **DONE** (all four sub-findings)

**F4-4, re-freeze.** `SUBMISSION_LEDGER.md §0` — the stale pins are quoted in a blockquote and then
demolished:
- OLD: "Git commit (HEAD at freeze): `ff9be6b2…`; PDF sha256: `5b01e5e7…`; PDF pages: 23"
- NEW: "**All three are stale, and the ledger should not have carried them as if they were live.**"
  with the four reasons (12 post-freeze edits, two of which change compiled output; the page-count
  drift the notes themselves concede; **no `.git` directory in the release**, so the commit hash is
  not a checkable claim; and the 2026-07-26 changes). Replaced by a 5-step dated re-freeze procedure
  and a standing rule: "this section carries either a complete, same-day freeze record produced by
  the procedure above, or the words **NOT FROZEN**. It never carries a partial one." Current value:
  **NOT FROZEN**.

**Reconciliation with `EDIT_NOTES_2026-07-23.md`** — annotated in place with a "KEEP THIS FILE"
header explaining that it is *the evidence* the freeze is invalid, naming the two output-changing
edits, confirming its "simplify wording, never scope" rule is not in dispute, promoting its two
correct flags into tracked items, and qualifying its one over-claim ("pass the forbidden-phrase
greps" → "pass after manual review of 7 negation hits").

**F4-15, `STORAGE_MANIFEST.json` regenerated.** The `claim_ledger.json` entry was `23179` bytes /
`81b9d1e0…`; disk was `25336` / `bff76f3c…`. Refreshed, with the old values retained as
`previous_sha256_2026_07_21` and a `drift_note` naming the finding. Then extended from **3
checksums to 75**: a new `sealed_evidence_checksums` block hashing every file in `LOCK_SEAL.json`
fresh from disk (72 files: 70 present, 2 absent) plus a summary, and a `coverage_gap` field pointing
at the Guard-2 spec. (Re-refreshed after my own `claim_ledger.json` annotation pass, so the recorded
hash matches disk right now.)

**F4-17, superseded docs stamped** — five documents, each stamp naming *what it still gets wrong*
rather than a generic banner:
- `GAP_AUDIT.md`, `INTEGRITY_FIXES.md` (root) — superseded; both note their `frontier_decisive/**`
  evidence is now unreadable, and `INTEGRITY_FIXES.md`'s stamp notes its item 5 describes a
  Camelyon win that was later withdrawn as KB-CLAIM-022.
- `EVIDENCE_MATRIX.md` — three specific errors enumerated (stress-grid vs head-to-head rows; "RxRx1
  fresh 0.0/**0.2587**/0.0 real ckpt confirmed"; `[TODO-local]` items the ledger marks `[RESOLVED]`).
- `PHASE7_INTEGRATION_AUDIT.md` — stamped, **plus** a correction of its own [P0] reasoning.
- `REVIEWER_REPRO_PACKET.md` — stamped "PARTIALLY SUPERSEDED" **and corrected in place**:
  - OLD `§A5`: "iWildCam is a **point-estimate** win (no CI claim); Office-Home and the CIFAR stress
    grid carry the CI-backed beats-both."
  - NEW: "…**CORRECTED 2026-07-26 (F4-17):** an earlier version of this line said […] That was wrong
    about Office-Home and it was the version handed to external reviewers. Office-Home is promoted
    as **OOF no-harm only** […] its own artifact records `\"beats_both_robust\": false` […] The
    CI-backed beats-both tracks are the **CIFAR-10-C stress grid (Tent and EATA)** and the
    constructed three-source mixture."

**The three-stories RxRx1 item settled in one sentence** at `SUBMISSION_LEDGER.md §11a`: the
artifact carries `"candidate": "sar_online"` and `regret_adapt = 0.2530598958`, so 0.2531 is right
and **both** `PHASE6`'s and `PHASE7`'s stated reasons were wrong; 0.258724 is the seeds 0-4
extraction — a different seed set, not a different candidate.

**F4-19, forbidden-phrase gate** — specified as `SUBMISSION_LEDGER.md §12.8` with the four
self-denial hits named and two concrete fixes (proximity requirement, or explicit negation
whitelist), plus the argument for why it matters: "A gate that fires on correct text gets overridden
by habit and then protects nothing." **Not implemented** — the gate lives in code I do not own.

**Also re-opened as G9**: `WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` still re-lists `id_val` while the
ledger marked G9 `[RESOLVED]`. Verified the `bootstrap_win_cis.py` half *is* done. Recorded in both
`SUBMISSION_LEDGER.md §4` and `KBOUND_REMAINING_TODOS.md` with the two halves separated.

## Closing instruction (README + ledger current state, TMLR) — **DONE**

**`docs/research/kbound/README.md`** — new "State of the project as of 2026-07-26" banner as the
first thing after the title: eight bullets covering the venue change, NOT FROZEN, the radius defect
and its blast radius, the retracted PHASE6 verdict, the three demotions, the 143 placeholders, the
environment heterogeneity, and the post-hoc family. Closes with the one-sentence version:
> "the CIFAR-10-C stress-grid safety result is real, well-powered and survives every check; most of
> the rest of the panel supports a narrower claim than the one originally written, and the documents
> now say so."

"Start Here" reordered to put `SUBMISSION_LEDGER.md` first and add `DATA.md`,
`PLACEHOLDER_INVENTORY.md`, `COMPARISON_FAMILY.md`, `PHASE6_LEAKAGE_AUDIT.md`, `REPRODUCE.md`.
"Evidence Tiers" rewritten into four honest tiers (CI-supported / point-estimate / one-sided
no-harm with source problems / diagnostic-negative-withheld) — the old version listed ImageNet-C SAR
under "Promoted controlled results" as a beats-both and Camelyon17 as a plain "natural no-harm
result". "Manuscript Policy" now leads with **"Target venue: TMLR, single-column"** and states that
no result is cut for length. A caveat added about `reproduce_submission.sh`'s `set -euo pipefail`.

**Root `README.md`** — "Scientific Status" table rebuilt: the old row "Controlled mixed shifts |
CIFAR-10-C Tent/EATA **and ImageNet-C SAR beats-both tracks**" is split into a CI-supported row and
a point-estimate row; the old "Natural shifts | Office-Home, iWildCam, **Camelyon17**, RxRx1
no-harm results" is split so Camelyon17 gets its own provenance-failure row. Three release caveats
added (placeholders, `DATA.md`, environment). The minimal-use section now states the declared LOO
rule and the FA_u-ceiling caveat.

**`SUBMISSION_LEDGER.md`** — rewritten; §§0, 3, 4, 5, 8, 9, 10, 11, 12 new or rewritten, including
a full decision-accounting table (ADAPT/FREEZE/ABSTAIN, false adapts, CP95 upper on FA_c) for all
nine tracks plus D33, and the conclusion that **only CIFAR-10-C certifies `FA_c <= 0.10`** —
0 false adapts in 1 113 and 1 244 ADAPT decisions, CP95 upper 0.0027 and 0.0024.

`DOCS_INDEX.md` reconciled to 2026-07-26 with the three new documents and a superseded-docs section.

---

## Deliberately not done

1. **`paper/generated/kbound_result_manifest.json`** — owned by the paper agent. Requirements
   written up above under "For the paper agent".
2. **The forbidden-phrase gate, the NUL-scan test, and the `STORAGE_MANIFEST` coverage test** —
   specified in full (`PLACEHOLDER_INVENTORY.md` Guards 1-3, `SUBMISSION_LEDGER.md §12.8`) but not
   implemented; `tests/` belongs to the library agent.
3. **`research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97`** — the open half of G9. It is a `.yaml`
   outside my ownership; recorded as an open item in two places instead.
4. **Deleting `kbound_pkg/build/`** — recommended in `PLACEHOLDER_INVENTORY.md` (8 of the 143
   placeholders are a stale build copy of the library) but not executed, since deleting a directory
   is not a documentation change and the library agent owns that tree.
5. **Retracting `PHASE6_LEAKAGE_AUDIT.md` wholesale.** Sections (a) and (d) were correct; §(d) is
   explicitly marked "UNCHANGED (this section stands)". Over-retracting would be its own
   misrepresentation.
6. **`archive/**` and `reports/**`** — left alone. `reports/NONTRAINING_CLOSURE_REVIEW_2026-07-21.md`
   still says "Camelyon17 remains 'reconciled no-harm'"; it is a dated meeting record rather than a
   status document, and `SUBMISSION_LEDGER.md §11` establishes that the ledger overrides it. Flagging
   rather than editing, in case the author wants dated records left untouched.

## Numbers audit

Every number written into a file came from `NUMBERS_PACK.md`, from an artifact I read directly, or
from a scan I ran myself in this session. The self-run measurements are: the 143-placeholder census
(full-tree NUL scan); the 1 427 / 345 / 24.2% `beats_both` count over 786 JSONs; the 81 protocol
files; the 72-file / 70-present / 2-absent `LOCK_SEAL.json` verification; all sha256/byte values
written into `LOCK_SEAL.json` and `STORAGE_MANIFEST.json`; the CIFAR-10-C 432-cell and ImageNet-C
27-cell grid decompositions; and the `claim_ledger.json` drift (23 179 → 25 336 bytes). No number
was carried forward from prior text without being checked.
