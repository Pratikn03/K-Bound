# FIXES_APPLIED — K-Bound fix queue, applied, audited, and closed

**Final changelog.** Three passes are folded into this file:

1. Four agents (PAPER, SCRIPTS, LIBRARY, DOCS) applied the 32-item queue in `kb_review/review_6_overall.md`.
2. A verifier audited that work and found **seven surviving defects** (§What was found wrong, retained
   below for the record as defects A–L).
3. Two cleanup agents closed those defects (D1–D11), and a **final verification pass on 2026-07-26**
   re-checked every one of them against the files, re-ran the sweeps, rebuilt the paper, re-ran the
   test suite, and re-verified the seals.

Everything below was re-derived from the files on disk. Where an agent's report and the file
disagreed, the file won. Numbers come from `kb_fixes/NUMBERS_PACK.json` or from an artifact read
directly; where the pack and the review panel disagree, the pack wins (pack §0).

---

## Summary

**32 of 32 items attempted. 28 DONE, 2 PARTIAL, 2 BLOCKED-NEEDS-DATA. All seven verifier defects
are CLOSED and independently confirmed.**

The paper now claims something materially narrower than it did, and every headline sentence is
traceable to a committed artifact.

* **The uniform-no-harm BLOCKER is dead everywhere.** It survived the first fix round twice in
  `kbound.tex` (:236, :303); both are now rewritten. A tree-wide sweep of every `.tex` and `.md`
  finds **zero surviving assertions**: every remaining occurrence is a negation, a dated correction,
  or sits inside a document carrying a `SUPERSEDED 2026-07-26` banner. The claim is now scoped to
  the **four** one-sided natural tracks with locked held-out artifacts (Camelyon17, iWildCam,
  Office-Home, RxRx1), with PACS, ImageNet-R and CIFAR-10.1 named as the losses.
* **One radius rule, stated once and implemented once.** `k = ceil((n+1)(1-alpha))`, unclamped,
  leave-one-out-of-pool; `k > n` warns, returns `+inf`, and forces ABSTAIN. The clamp is gone from
  **both** implementations — the library (`kga/certificate.py`) and the driver shim
  (`docs/research/kbound/scripts/kbound_decide.py`), which had kept `clamp="min_n"` as its default
  and was the shim every table-producing script actually calls. The normative statement lives once,
  quotably, in `THEORY_TO_CODE_MAP.md` §1a, and `tests/test_one_radius_rule.py` (34 tests) fails if
  anyone re-adds a clamp, an infeasibility knob, or an interpolated quantile in a radius function.
* **The promoted ImageNet-C number is the clean one.** `0.0289 / FA_u = 1/135 / 13 ADAPT` is
  promoted in the paper, `uniform_verdicts.json`, `kbound_result_manifest.json`,
  `decision_metrics.json`, `RESULT_MANIFEST.json` and `results_source.json` alike. The leaky in-pool
  `0.0264 / 0 / 12` survives only as an explicitly labelled sensitivity. The claim is written as the
  pack requires: *point-estimate no-harm against always-freeze, and NOT a CI-supported beats-both
  against either fixed policy.*
* **The headline is a safety claim, not an accuracy claim.** 1,113 ADAPT decisions, zero false,
  Clopper–Pearson 95% upper 0.0027, on CIFAR-10-C — and the paper says plainly that `FA_u <= alpha`
  is an arithmetic identity under in-sample rank calibration and therefore *not a measurement* on
  the other eight tracks.
* **Item 17, which no agent owned, is done and it moved a headline.** Cluster-robust intervals at
  four resampling units for both candidates (16 intervals, `tab:cifar-cluster`) show **Tent survives
  every unit and EATA does not**. The paper now claims a CI-supported beats-both **for Tent only**.
  `migration.cluster_robust_beats_both` has exactly one member.
* **The provenance layer is real.** `kbound_result_manifest.json` — which the appendix calls "the
  authoritative index" — was never touched in the first round. It is now schema v2, with CIFAR
  sources repointed to `mixed_headtohead_v1/`, `source` fields added where they were missing,
  Camelyon17 flagged `NOT_REPRODUCIBLE_FROM_RELEASE`, and the 1109-vs-1113 discrepancy resolved with
  both aggregates preserved and labelled.
* **94 files carrying machine-local paths are down to 9**, every one of them allowlisted with a
  reason, and every one either *detects* or *documents* the pattern rather than depending on it. A
  whole class nobody had looked for — 11 files hard-coding an ephemeral Cowork session-sandbox mount
  — was found and fixed.

**What is still not true of this release:** two tracks (Camelyon17's `n=18` OOD triple, and the
Office-Home / iWildCam promoted regrets) rest on record files that are absent from disk; six figures
are missing so no PDF builds without stubs; and 140 files are unreadable iCloud placeholders. All are
itemised in §Still blocked.

---

## What the final verification pass checked, and what it found

| Sweep | Result |
|---|---|
| Uniform-no-harm, every `.tex` + `.md` | **0 surviving assertions.** 6 live `.md` survivors found and closed (see below). |
| One radius rule (`np.quantile`, `decide_kga`, clamping) | 6 `decide_kga` forks, **all bodiless delegations** (AST-verified). 20 interpolated-radius survivors, all allowlisted with reasons and frozen by test. `on_infeasible` is `{'inf','raise'}`; `'clamp'` raises. |
| Numbers vs `NUMBERS_PACK.json` | All 16 cluster intervals, all 12 per-corruption gaps, the LOCO row, PACS, ImageNet-R, CIFAR-10.1, Camelyon17, the SAR quarantine, the ε coefficients of variation and all three ImageNet-C intervals match to printed precision. **One mismatch found and fixed** (below). |
| Cross-artifact agreement | paper / `uniform_verdicts` / `kbound_result_manifest` / `decision_metrics` / `RESULT_MANIFEST` / `results_source` tell **one** story: 0.0289, FA_u 1/135, 13 ADAPT, 1113/1244. |
| LaTeX, all four drivers | **0 unresolved refs, 0 duplicate labels, 0 genuinely missing `\input`s.** TMLR build: exit 0, **61 pages**, 0 errors, 0 undefined refs/citations, 8 overfull hboxes (worst 22.18 pt). |
| Python | **490 files, 0 syntax errors.** 44 `.py` are NUL placeholders. |
| Test suite | **169 passed / 26 failed / 11 collection errors / 41 skipped.** Every failure traced; **none attributable to this work.** |
| Seals | `STORAGE_MANIFEST` 69/71 match, `LOCK_SEAL` 70/72 match, **0 mismatches** (2 absent each = the known Camelyon pair), re-verified *after* all edits. |

### Defects found and fixed in this final pass

1. **`tab:cifar-cluster` printed interpolated-rule point estimates inside an exact-rank table.**
   `kbound_short_body.tex:599` explicitly declares *"$0.0015736\to0.0015852$ Tent,
   $0.0012676\to0.0012799$ EATA … We report the exact-rank value in every case"* — and then :907
   and :913 printed `0.00157` and `0.00127`, the interpolated values, in a table whose caption says
   "declared exact-rank radius", in a paper that prints the exact-rank adapt counts 1113/1244
   everywhere. Table panel (b) three rows below printed the correct `0.00159`. **Fixed:**
   `0.00157 -> 0.00159`, `0.00127 -> 0.00128` (pack `item4.cifar10c_*_headtohead_2160cells.loo_radius`,
   `regret_exact_*`). This is the same "two rules inside one table" defect the PAPER agent found and
   fixed in `tab:imagenetc-faithful`; it survived one table over.
2. **Six live `.md` files still asserted uniform no-harm.** The first-round sweep covered `.tex`
   only. Fixed by kind, not by blanket rewrite:
   * `PROJECT_STATUS_AND_OPEN_PROBLEMS.md` (self-described "single source of truth") — rewritten to
     the four-track claim with a dated `CORRECTION 2026-07-26` block quoting the old sentence.
   * `audits/MAIN_PAPER_CLAIM_MATRIX.md` — this is the *wording-governance* file; its row licensed
     the retracted phrase. Row rescoped and `"uniform/uniformly no-harm"` added to its forbidden
     column.
   * `claim_ledger.json` KB-CLAIM-020 — `allowed_wording` was literally
     `"uniformly no-harm under stated held-out protocol"`. Corrected, and both variants added to
     `forbidden_wording`, matching how PACS is already handled in the same file (KB-CLAIM-041).
   * Four dated records (`MAIN_PAPER_REVISION_AUDIT.md`, `KBOUND_10X_FINAL_GATE.md`,
     `NONTRAINING_CLAIM_MATRIX_2026-07-21.md`, `MIXED_BENCHMARK_EXT_PROTOCOL.md`) — **not** rewritten.
     Rewriting a dated audit or a pre-registration falsifies the record. Each got a
     `SUPERSEDED 2026-07-26` banner naming exactly what it still gets wrong.
3. **Three live `.md` instruction lines still carried machine-local paths**, missed because the D8
   sweep was scoped to `.py`/`.sh`: `REVIEWER_REPRO_PACKET.md:205` (`cd /path/to/AutoML_Flagship_V8`
   — also the *old repo name*, in the reviewer-facing packet), `wilds/READINESS.md:54`,
   `multimodal_natural/README.md:29,37`. All three repointed to the established
   `$KBOUND_REPO_ROOT` / `$KBOUND_EXTERNAL_ROOT` convention.

### Claimed CLOSED but not found in the file

**None.** Every one of D1–D11 was located in the files and matches its description. Spot-checks that
confirmed the more surprising claims:

* D1's `tab:cifar-cluster` exists with all **16** intervals, and every one matches
  `NUMBERS_PACK.json` to five decimals; the 12 per-corruption gaps match to five decimals; the
  leave-one-corruption-out row matches the `[exact]` variant (MAE 0.030904, R² 0.895377,
  ε 0.098815, adapt 41.34%, regret 0.006032), which is the right choice because the paper declares
  one rule.
* D6's three named `\resizebox` are gone **and** so are the three extra ones in the
  `manuscript/main.tex` tree that the verifier did not enumerate. The two survivors in
  `edge/kbound_camera_supp_tables.tex` are confirmed reachable from **no** driver.
* D9's consequence is disclosed in the paper's own voice at `kbound_short_body.tex:1224–1231`:
  Table VIII "cannot use the declared leave-one-out-of-pool calibration … the released library would
  return ε = +∞ and abstain on all 36 cells … We report the in-pool scoring and treat the FA_u
  column as void rather than as evidence."
* D3's `tab:imagenetc-faithful` is arithmetically self-consistent under one rule: mean oracle
  accuracy 0.4379 gives KGA 0.4379 − 0.0289 = **0.409** and always-adapt 0.4379 − 0.0529 = **0.385**,
  both as printed.
* D7's replacement test passes (3/3) and the README genuinely now expands KGA on first use, closing a
  real gap — the README used the acronym throughout and never once defined it.

---

## Item-by-item

| # | Sev | Status | Files changed | What happened |
|---|---|---|---|---|
| 1 | BLOCKER | **DONE** | `kbound.tex`, `kbound_short_body.tex`, `kbound_abstract.tex`, + 6 `.md`, `claim_ledger.json` | Short paper, TMLR and abstract rescoped in round 1. Round 2 closed the two survivors in `kbound.tex` (:236 caption now names PACS's 2.45× loss *inside the same caption*; :303 rewritten). Final pass closed 6 `.md` survivors and the claim-ledger `allowed_wording` that licensed the phrase. **0 assertions tree-wide.** |
| 2 | MAJOR | DONE | `kbound_short_body.tex`, `uniform_verdicts.json` | Rule declared once at :560; "5/5 seeds" → "2 of 5 seeds (seeds 2 and 4)"; all five §0.4 rule-shift rows disclosed, including the three where exact-rank is *less* flattering. |
| 3 | MAJOR | DONE | `kbound_short_body.tex`, `g8_exactrank_ci.py`, `uniform_verdicts.json` | Bootstrap re-run at 27 seed-averaged conditions. Adapt gap [−0.0806,+0.0175] (includes zero), freeze gap [−0.0092,−0.0023]. `--unit` flag warns on correlated units. |
| 4 | MAJOR | **DONE** | 6 scripts, `PHASE6_LEAKAGE_AUDIT.md`, `kbound_short_body.tex`, `kbound_short_appendix.tex`, `kbound_numbers.tex` | Round 1 fixed the code and retracted the audit but left the paper promoting the in-pool number. Round 2 promoted the LOO triple **0.0289 / FA_u 1/135 / 13 ADAPT** everywhere and demoted 0.0264 to a labelled sensitivity in exactly three places. |
| 5 | MAJOR | DONE | `kbound_short_body.tex`, `decision_metrics.json` | New §`sec:fa-identity` + `tab:decision-accounting`; 29 meaningless Wilson intervals deleted with per-entry reasons; CP95 on FA_c added to all 29 tracks; <10-adapt tracks daggered. |
| 6 | MAJOR | DONE | `kbound_short_body.tex` | SAR quarantine rewritten: four not five seed files; "intervals exclude zero" withdrawn as "they exclude zero on the wrong side"; three positive intervals printed. |
| 7 | MAJOR | DONE | `kbound_short_body.tex`, `kbound.tex`, `kbound_short_appendix.tex` | 6-of-15 corruptions named, `"quick": true` disclosed, ImageNet-C operating point stated inline (lr 4e-3 = 16× official), `tab:adapter-hparams` corrected. |
| 8 | MAJOR | **PARTIAL** | `ablation_exactrank.py`, `ablation_sweep.py`, `gate_baseline_comparison.py`, `reproduce_submission.sh`, `test_calibration_split_integrity.py` | Verified by execution twice: 9/9 steps, `DONE (PASS)`, exit 0 (~18–25 min). **Edge split-integrity artifacts still absent → 2 tests SKIP, the property is UNVERIFIED in this release.** |
| 9 | MAJOR | **BLOCKED-NEEDS-DATA** | `PLACEHOLDER_INVENTORY.md` (new) | 140–143 placeholders inventoried, grouped A–H, with recovery commands. Materialising them needs the author's Mac. |
| 10 | MAJOR | **BLOCKED-NEEDS-DATA** | 14 files incl. `LOCK_SEAL.json`, `results_source.json`, `SUBMISSION_LEDGER.md`, `kbound_result_manifest.json` | Camelyon17 relabelled "sealed but NOT RECOMPUTABLE from release" across the tree and `NOT_REPRODUCIBLE_FROM_RELEASE` in the manifest. Two sealed files + one unsealed script still absent. |
| 11 | MAJOR | **DONE** | `kbound_result_manifest.json`, `decision_metrics.json`, `uniform_verdicts.json`, `LOCK_SEAL.json`, `SUBMISSION_LEDGER.md` | Round 1 left the manifest untouched (mtime predated the fix run). Round 2 regenerated it to schema v2 with all four required edits, and resolved 1109-vs-1113 from artifacts. |
| 12 | MAJOR | DONE | `kbound_tmlr.tex` (new), `kbound_short_body.tex`, `kbound.tex` | Single-column TMLR driver, now 61 pages, shares 100% of content with the IEEE build. No result table deleted. |
| 13 | MAJOR | DONE | `kbound_short_body.tex`, `theory_setup.tex` | γ stated as a residual *by definition*, so the decomposition is an identity and sufficiency is interval arithmetic; weight moved to `lem:nonid` and `thm:short-audA`. |
| 14 | MAJOR | DONE | `kbound_short_body.tex`, `frontier_validation.py`, `frontier_sweep.py` | Retitled "Synthetic illustration of the frontier (not a test of it)"; script retitled ILLUSTRATION with a runtime circularity assertion. |
| 15 | MAJOR | **DONE** | 9 driver files + 6 `src/scripts/kbound/` shims | All forks are bodiless delegations — **AST-verified in the final pass**: 6 forks, each exactly one `Return`. Six byte-identical copies on the installed package path replaced by 6-line shims. |
| 16 | MINOR | DONE | `cifar_tent_mps_v2.py`, `run_wilds_camelyon17.py` | `bn` arm wired (not run); `EVAL_CHUNK=512` stamped into manifests; WILDS eval batch recorded at its real value (64). |
| 17 | MINOR | **DONE** | `kbound_short_body.tex`, `kbound_short_appendix.tex`, `uniform_verdicts.json`, `kbound_result_manifest.json` | **Was assigned to nobody and flips a headline.** New `tab:cifar-cluster`: 16 intervals (2 candidates × 4 units × 2 gap sides) + the leave-one-corruption-out refit. Tent survives every unit; EATA does not. Claim narrowed to **Tent only**. |
| 18 | MINOR | DONE | `gate_baseline_comparison.py` | False parity claim retracted on both sides; per-rule calibration budget printed as a column; two equal-budget KGA rows added. |
| 19 | MAJOR/MINOR | DONE | `REPRODUCE.md`, `SUBMISSION_LEDGER.md`, `CIFAR10C_SAR_QUARANTINE.md` | Both per-seed environment tables published, explicit permitted/forbidden rule, new gate-0 on the SAR rebuild. |
| 20 | MAJOR | DONE | `DATA.md` (new), `STORAGE_MANIFEST.json` | Every version/split read from a named committed artifact; unverifiable checksums written NOT RECORDED; 2 of 9 datasets marked unobtainable. |
| 21 | MINOR | DONE | `theory_core_main.tex`, `theory_appendix_ext.tex`, `kbound_short_appendix.tex` | Dangling Gaussian-witness pointer gone; `\TV` defined in all drivers; `thm:imp(ii)` given a Neyman–Pearson proof; explicit proof-status block added. |
| 22 | MINOR | DONE | `kbound_short_body.tex`, `refs.bib` | `def:detectable` applied *ex ante* to all nine tracks conceding both failure directions; FMoW/Poverty deleted; 7 citations added. |
| 23 | MINOR | DONE | `kbound_short_body.tex` | PACS min/median/max/sd printed; all ten ImageNet-R backbones printed; corrected to 7-of-10 backbones and 4 degenerate 0% harmful base rates. |
| 24 | MINOR | DONE | `COMPARISON_FAMILY.md` (new), `uniform_verdicts.json`, `kbound_short_body.tex` | Retrospective Holm family withdrawn in the author's voice; 1387/326/23.5% census published; `_meta.wave_holm_family` replaced with a SUPERSEDED block. |
| 25 | MINOR | **DONE** | `kga/certificate.py`, `kga/policy.py`, `kga/routing.py`, `kbound_decide.py`, `THEORY_TO_CODE_MAP.md`, `tests/test_one_radius_rule.py` | Round 1 unclamped the library but **not the shim every driver calls**, which kept `clamp="min_n"` as its default — two implementations, two rules, and the paper declared a third. Round 2 removed it from both, deleted the knob from `decide_kga`, and wrote the rule down once in `THEORY_TO_CODE_MAP.md` §1a. |
| 26 | MINOR | DONE | `kga/certificate.py`, `kga/cli.py` | `benefit_range` keyword-only with no default in **both** estimators; sidedness stated at module top; `Certificate.interval_level` machine-readable. |
| 27 | MINOR | DONE | `kbound_short_appendix.tex` | Both `\Comment` labels in `alg:calib-eval` carry the manifest's own "…; NOT exact split conformal" string; grid branch rewritten for the LOO pool. |
| 28 | MINOR | DONE | 5 analysis scripts | `kbound_decide.false_adapt` is the single definition; `fa_u`/`fa_c` emitted separately everywhere; two `beats_both` gates re-pointed to the marginal rate. |
| 29 | MINOR | DONE | `kga/cli.py`, `kga/evidence.py`, `kga/routing.py` | `decide` no longer a constant-ABSTAIN generator; importance-weight direction corrected; anytime panel advances all K e-processes before checking rejection. |
| 30 | MINOR/NIT | **DONE** | `kbound_repro/paths.py`, `tests/test_reproducibility_hygiene.py`, + ~110 `.py`/`.sh`/`.md` | Salted `hash()` → blake2b; sklearn version + PYTHONHASHSEED stamped. Machine-local paths **94 → 9**, all allowlisted with reasons. Guard widened from `kga/`+`tests/` to the whole tree, `.sh` included, comments **not** exempted. |
| 31 | MINOR | DONE | 9 doc files | Stale freeze pins replaced with a dated re-freeze procedure; `STORAGE_MANIFEST` regenerated from 3 to 71 checksums; five superseded docs stamped. |
| 32 | NIT | **PARTIAL** | `kbound.tex`, `theory_setup.tex`, `edge/kbound_camera_main_tables.tex`, `kbound_frontier_appendix.tex`, `paper/sections/{gamma_meter,unification_reach,reach_unification_v05}.tex` | Karlin–Rubin → Neyman–Pearson done. **All `\resizebox` in every compiled build are gone** (3 named + 3 more nobody had enumerated); the 2 survivors are in a file no driver reaches. **The Lean renames are still not done:** `forced_abstention_probability` and `exchangeable_scores_miss_le_alpha` still carry names that overstate their content in `formal/KBound/`. |

---

## Claim changes

Old text quoted from the pre-fix files; new text from disk today. This is the complete, quotable set.

**1. The headline empirical claim (abstract).**
OLD — *"Across held-out natural shifts, KGA is uniformly no-harm: it matches the better fixed policy
and beats the worse."*
NEW — *"On the four one-sided natural tracks with locked held-out artifacts—hospital (Camelyon17),
wildlife-camera (iWildCam), laboratory-batch (RxRx1), and domain (Office-Home) shifts—KGA ties the
better fixed policy at zero observed false adaptation; on PACS, ImageNet-R, and CIFAR-10.1 it is a
conservative null or fails the declared transfer bar, and we report those outcomes rather than
withdraw the tracks."*

**2. The same claim in the long manuscript** (`kbound.tex:303`, the second surviving BLOCKER).
OLD — *"K-Bound does not claim universal improvement. **The headline deployment claim is uniform
no-harm on natural shifts**."* — which contradicted :183 of the same file, 120 lines earlier.
NEW — *"K-Bound does not claim universal improvement, and it does not claim no-harm across the whole
panel. **The headline deployment claim is no-harm on the four one-sided natural tracks with locked
held-out artifacts** — Camelyon17, iWildCam, Office-Home and RxRx1 — where KGA matches the better
fixed policy and beats the worse one at zero observed false adaptation."*

**3. The regime-summary table caption** (`kbound.tex:236`, the first surviving BLOCKER — a universal
claim falsified by the rows printed directly beneath it).
OLD — *"**Headline rows** are held-out natural shifts (uniform no-harm)."*
NEW — *"**Headline rows** are held-out natural shifts. No-harm holds on the four one-sided tracks
with locked held-out artifacts (Office-Home, iWildCam, Camelyon17, RxRx1); it does **not** hold
uniformly — PACS, in this very block, loses to always-adapt by 2.45× (0.0431 versus 0.0176)."*

**4. The safety guarantee.**
OLD — the panel reported `FA_u <= alpha` on eight tracks as eight confirmations.
NEW — *"The guarantee is exercised with real statistical power on exactly one track: the CIFAR-10-C
stress grid, where 1,113 ADAPT decisions produce zero false adaptations (Clopper–Pearson 95% upper
bound 0.0027 on the conditional rate)"* — plus, in §`sec:fa-identity`: *"on the stress grids
'FA_u <= alpha' holds for any data whatsoever and is not a measurement… the ceiling is 0.0972 at
n=432, 0.0370 at n=27, and exactly 0 at n in {9,12,18}, where k=n."*

**5. The CIFAR-10-C beats-both, separated by candidate** (item 17; this is the claim that moved most).
OLD — *"For both Tent and EATA, the regret gaps to **both** fixed policies are positive with 95%
bootstrap CIs excluding zero."*
NEW — *"**Tent survives every unit.** The gap to always-adapt stays negative with the interval
excluding zero from 432 cells down to 6 corruption clusters ([−0.00952,−0.00259]) … **EATA does
not.** Its adapt-side gap loses the interval as soon as the corruption structure is respected:
[−0.00483,+0.00043] at 12 corruption×severity clusters and [−0.00436,+0.00035] at 6 corruption
families, both containing zero. **We therefore claim a CI-supported beats-both for Tent only.**"*

**6. ImageNet-C beats-both.**
OLD — *"KGA beats both fixed policies with 95% CIs excluding zero"* (adapt-side [−0.052, −0.003]).
INTERMEDIATE (round 1, still wrong) — *"beats always-freeze with an interval excluding zero at the
condition level, at FA_u = 0"*, using the leaky in-pool radius.
NEW — *"ImageNet-C SAR therefore supports a point-estimate no-harm statement against always-freeze
(0.0289 versus 0.0319) and **not** a CI-supported beats-both against either fixed policy"*, with
[−0.0085,+0.0038] freeze / [−0.0755,+0.0181] adapt at 27 conditions and [−0.0079,+0.0036] even at
the generous i.i.d.-135 design.

**7. ImageNet-C per-seed.**
OLD — *"point estimates improve both fixed-policy regrets on 5/5 seeds."*
NEW — *"Point estimates improve both fixed-policy regrets on 2 of 5 seeds (seeds 2 and 4). On seeds
0, 1, and 3 KGA adapts on no cell at all and its regret is bit-identical to always-freeze… An
earlier draft reported '5/5 seeds', which was an artifact of the interpolated quantile rule and is
withdrawn."*

**8. ImageNet-C EATA vs always-adapt.**
OLD — *"EATA ties always-adapt."*
NEW — *"essentially a tie, and not a win"* — under the declared rule it **trails**, 0.00067 versus
0.00010, on 120 ADAPT decisions with 1 false adapt.

**9. The radius rule.**
OLD — *"ε = ρ_(k), k = min{n, ⌈(n+1)(1−α)⌉}, α = 0.10 … This is the only rule used anywhere in the
paper"*, with the clamp load-bearing in the FA_u argument: *"the clamp k = min{n, ·} makes ε the
maximum residual at n ≤ 9, so FA_u is forced to 0."*
NEW — *"ε = ρ_(k), k = ⌈(n_cal+1)(1−α)⌉ … There is **no min{n_cal,·} clamp**"*, with the
infeasibility branch, the n ≥ 9 feasibility threshold, and the reason the clamp is wrong (it attains
only n/(n+1) < 1−α). The load-bearing sentence is replaced by a verifiable one: the clamp is
**never binding** at any calibration size in the paper (n=9→k=9, 12→12, 18→18, 27→26, 135→123,
432→390), so the distinction is about what the paper *claims*, not what the numbers are.

**10. Camelyon17 Table VIII, the one place the threshold bites.**
OLD — silent; the table was presented as ordinary evidence.
NEW — *"This is also the one table in the paper that cannot use the declared leave-one-out-of-pool
calibration: a pool of 8 residuals needs k = 9, which no finite radius supplies, so the released
library would return ε = +∞ and abstain on all 36 cells. We report the in-pool scoring and treat the
FA_u column as void rather than as evidence."*

**11. The leakage audit verdict.**
OLD — *"## VERDICT: PASS (clean). No live promoted track computes ε in-sample on the cells it
scores."*
NEW — *"# CORRECTION 2026-07-26 — the 2026-07-21 VERDICT was WRONG… That certification was false.
Five shipped scripts and seven copy-pasted decide_kga forks did exactly that."*

**12. The multiplicity family.**
OLD — *"the 3 beats-both candidates … all survive Holm at 0.05."*
NEW — *"SUPERSEDED. The previous value … named exactly the three comparisons that came out positive,
after the results were known. That is not a multiplicity family and it is withdrawn."* Replaced by a
prospectively defined family with a published 1,387-determination denominator.

**13. The theoretical contribution.**
OLD — the frontier sold as the theorem.
NEW — γ is *defined* as the residual, so the decomposition is an identity and `thm:headline`'s
sufficiency half is interval arithmetic; the weight moves to `lem:nonid`'s matched construction and
`thm:short-audA`, *"a statement no amount of better label-free evidence can repeal."*

**14. The frontier experiment.**
OLD — *"frontier validation"* reporting *"90.0% empirical coverage."*
NEW — *"Synthetic illustration of the frontier (not a test of it)"*, conceding *"the frontier is not
discovered, it is substituted in"* and that 90.0% is `np.quantile`'s definition at that n, exact for
arbitrary data.

**15. Camelyon17 provenance.**
OLD — *"locked no-harm (OOD reconciliation)"*, sourced to a directory.
NEW — *"row retained for completeness; NOT reproducible from the release"*, with
`reproducibility_status: NOT_REPRODUCIBLE_FROM_RELEASE` and *"The promoted triple 0.0000 / 0.0000 /
0.1381 at n=18 appears in NO artifact on disk"* recorded in the manifest.

**16. The repository claim (identity).**
OLD — *"public at an anonymized repository."*
NEW — a factual note at `kbound.tex:1667` that the tree is **not** anonymised and must be redacted
before a double-blind submission; *"Earlier drafts described it as 'an anonymized repository', which
was false."*

**17. Reviewer packet.**
OLD — *"Office-Home and the CIFAR stress grid carry the CI-backed beats-both."*
NEW — *"That was wrong about Office-Home and it was the version handed to external reviewers… its
own artifact records `beats_both_robust: false`."*

**18. Project status ledger** (final pass).
OLD — *"an impossibility/frontier theorem + a certificate that provably controls false-adapt +
beats-both on synthetic stress grids + **uniform no-harm on five real benchmarks**."*
NEW — *"… + no-harm on the **four** one-sided natural tracks with locked held-out artifacts
(Camelyon17, iWildCam, Office-Home, RxRx1)"*, with a dated CORRECTION block quoting the old sentence
and naming the three losses.

**19. The claim ledger's own permission** (final pass) — the file that governs allowed wording.
OLD — KB-CLAIM-020 `allowed_wording: "uniformly no-harm under stated held-out protocol"`.
NEW — `"no-harm under the stated held-out protocol: ties always-freeze, beats always-adapt,
FA_u = 0 over 22 ADAPT decisions"`, with `"uniformly no-harm"` and `"uniform no-harm"` added to
`forbidden_wording`.

---

## Numbers that changed

Every value re-checked against `NUMBERS_PACK.json` and, where the pack names an artifact, against the
artifact itself.

| Quantity | Old | New | Source of the new value |
|---|---|---|---|
| ImageNet-C SAR pooled regret (KGA) | 0.0107 → 0.0264 (in-pool) | **0.0289** (LOO, **promoted**) | pack `item4.imagenetc_sar.loo_radius`; `pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json` |
| ImageNet-C SAR FA_u | 0.000 | **1/135 = 0.0074**, CP95 0.3163 | same |
| ImageNet-C SAR decisions | 65 → 12/14/109 (in-pool) | **13 ADAPT / 15 FREEZE / 107 ABSTAIN**, 1 false adapt | same |
| ImageNet-C SAR mean accuracy | 0.427 (interpolated-rule column) | **0.409** | recomputed under one rule; self-consistent with oracle 0.4379 − 0.0289 |
| ImageNet-C Tent accuracy / regret | 0.407 / 0.0139 | **0.406 / 0.0145** | same one-rule recomputation |
| ImageNet-C EATA accuracy / regret | 0.440 / 0.0003 | **0.440 / 0.0007** | same; and EATA **trails** always-adapt (0.00067 vs 0.00010) rather than tying it |
| ImageNet-C adapt-gap CI (95%) | [−0.0518, −0.0038] on 135 i.i.d. rows | **[−0.0806, +0.0175]** on 27 seed-averaged conditions; **[−0.0755, +0.0181]** under the promoted LOO radius | pack `item3…seedavg27.exact_rank`, `…after_item4_loo_radius` |
| ImageNet-C freeze-gap CI | [−0.0086, −0.0026] | **[−0.0085, +0.0038]** (27 conditions, LOO); **[−0.0079, +0.0036]** (i.i.d.-135) | same |
| ImageNet-C seeds beating both | 5 of 5 | **2 of 5** (seeds 2 and 4) | pack `item2.imagenetc_sar.seeds_beating_both` |
| CIFAR-10-C Tent regret triple | 0.0016259 / 0.0079757 / 0.1239368 (stress grid) | **0.0015852 / 0.0079234 / 0.1240979** (head-to-head, **exact rank**) | `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json` + pack `item4.cifar10c_tent_headtohead_2160cells.loo_radius.regret_exact_loo` |
| …the same, printed in `tab:cifar-cluster` panel (a) | 0.00157 (**interpolated**, wrong rule) | **0.00159** | fixed in the final pass; the paper declares the exact-rank value at :599 |
| CIFAR-10-C EATA regret triple | 0.0012676 / 0.0032683 / 0.1313789 (interpolated) | **0.0012799 / 0.0032683 / 0.1313789** (exact rank) | pack `item4.cifar10c_eata_headtohead_2160cells.loo_radius.regret_exact_loo` |
| …the same, printed in `tab:cifar-cluster` panel (a) | 0.00127 (**interpolated**, wrong rule) | **0.00128** | fixed in the final pass |
| CIFAR-10-C ADAPT counts | 1114 / 1246 (interpolated) | **1113 / 1244** (exact rank) | pack `item4.cifar10c_*_headtohead` |
| CIFAR-10-C aggregate provenance | 1109 (shipped audit) vs 1113 (paper) | **1113 promoted**; 1109 preserved in a labelled `superseded_aggregate` block | `decision_metrics.json` schema v3. 1109 = `stress_grid_multiseed_v1` under the archived interpolated rule (seed 0's stored summary, 221 adapts, + seeds 1–4 recomputed, 888); its glob matched 4 files (1728 cells) while claiming n=2160 |
| CIFAR-10-C CP95 upper on FA_c | not reported | **0.0027** (Tent), **0.0024** (EATA) | pack `item5.promoted_row_accounting` |
| **CIFAR-10-C Tent adapt-gap CI, 6 corruption-family clusters** | not computed | **[−0.00952, −0.00259]**, excludes zero | pack `item17.cifar10c_tent.cluster_robust` (item 17) |
| **CIFAR-10-C EATA adapt-gap CI, 12 clusters / 6 families** | claimed to exclude zero | **[−0.00483, +0.00043]** and **[−0.00436, +0.00035]** — both **include zero** | pack `item17.cifar10c_eata.cluster_robust`; the pack flags this as an explicit reversal of review_6's prediction |
| CIFAR-10-C adverse corruption families | not reported | Tent **1** (`gaussian_noise` +0.00189); EATA **2** (`gaussian_noise` +0.00022, `jpeg_compression` +0.00292) | pack `item17.*.per_corruption` |
| CIFAR-10-C leave-one-corruption-out refit | not run | MAE 0.00959→**0.03090**, R² 0.993→**0.895**, ε 0.02132→**0.09882**, adapt 51.5%→**41.3%**, regret 0.00159→**0.00603** (still below 0.00792 and 0.12410, FA_u = 0 throughout) | pack `item17.cifar10c_tent.leave_one_corruption_out`, `[exact]` column |
| ImageNet-R panel row | 0.0112 (interpolated) | **0.0151** (exact rank), vs always-adapt 0.0064 | pack `item23.imagenet_r`, §0.4 |
| ImageNet-R backbones worse than always-adapt | 1 | **7 of 10** (4 with a degenerate 0% harmful base rate) | pack §0.3 |
| CIFAR-10.1 SAR | 0.0045, FA_u 0.0250 | **0.0057**, FA_u 0.0083 | pack §0.4 |
| Camelyon17 Tab. VIII SAR | 0.0410, FA_u 0.0278 | **0.0425**, FA_u 0.0000 (and the FA_u column declared **void**) | pack §0.4 |
| PACS KGA regret | mean 0.0431 only | mean **0.0431**, min **0.00529**, median **0.03616**, max **0.15344**, sd **0.03895**; vs always-adapt **0.0176** (2.45×) | `PACS_MULTISEED_RESULTS.json`, pack `item23.pacs` |
| PACS pooled FA_u | not reported | **2/216 = 0.00926**; FA_c 2/12 = 0.167, CP95 0.438 | same |
| CIFAR-10-C SAR seed-0 harmful base rate | not reported | **0.5278**, 5.8× the seeds-1–4 mean **0.0909** | pack `item6.cifar10c_sar.quarantine` |
| CIFAR-10-C SAR, seeds 1–4 | claimed a win | KGA **0.0015990** vs always-adapt **0.0003099** (5.2× worse) | same |
| CIFAR-10-C SAR adapt-gap intervals | claimed to exclude zero | **[+0.00101,+0.00158]** (432 cells), **[+0.00003,+0.00298]** (12 clusters), **[+0.00012,+0.00271]** (6 families) — all **against** KGA | pack `item17.cifar10c_sar.cluster_robust` |
| Radius stability (ε c.v. across seeds) | not reported | Tent **0.030**, EATA **0.068**, SAR **0.390** | pack `item23.cifar10c_stress_grid` |
| FA_u ceiling under in-sample rank calibration | not reported | **0.0972** at n=432, **0.0370** at n=27, **0** at n ∈ {9,12,18} | `sec:fa-identity`; verified attained exactly on all 69 shipped per-condition files |
| Comparison-family denominator | 3 candidates | **1,387 recorded `beats_both` determinations, 326 true (23.5%)**; positive rate inside the declared family 3 of 12 | `COMPARISON_FAMILY.md` |
| Cluster-robust beats-both arms | 3 | **exactly 1** (CIFAR-10-C Tent) | `uniform_verdicts.json → migration.cluster_robust_beats_both` |
| Machine-local-path files | 94 (49 `.sh`, 45 `.py`; 191 lines) | **9**, all allowlisted with reasons | `tests/test_reproducibility_hygiene.py` |
| Cowork session-sandbox paths | **0 known** (nobody had looked) | **11 found and fixed** | widened guard; strictly worse than a home directory — valid only inside one ephemeral container |
| Interpolated-radius sites | 21 (verifier's count) | **25 found**, 5 converted, 20 allowlisted and frozen | `tests/test_one_radius_rule.py::INTERPOLATED_RADIUS_ALLOWLIST` |
| `\resizebox` in compiled builds | 3 (verifier's count) | **6 found and removed**, 0 remain in any driver | 3 extra were reached via `manuscript/main.tex → extended_theory_includes.tex` |
| Placeholder artifacts | 142 | **140** measured (74 `.json`, 44 `.py`, 10 `.csv`, 9 `.md`, 3 `.sh`); inventory says 143 | `PLACEHOLDER_INVENTORY.md` + direct scan |
| Test suite | 159 passed / 28 failed | **169 passed / 26 failed** / 11 collection errors / 41 skipped | the 2 removed failures are exactly D7's; **zero new failures** |
| TMLR build | 57 pages | **61 pages**, exit 0, 0 errors, 0 undefined refs/citations, 8 overfull hboxes (worst 22.18 pt) | rebuilt in the final pass with lmodern/microtype and 32 figures stubbed |

---

## Still blocked and why

### 1. iCloud NUL-filled placeholders — 140 files (inventory says 143)
74 `.json`, 44 `.py`, 10 `.csv`, 9 `.md`, 3 `.sh`. Full list with byte sizes and dependency groups
A–H in `docs/research/kbound/PLACEHOLDER_INVENTORY.md`. The ones that actually block something:

* `experiments/kbound/officehome/{run_officehome_kbound.py, oh_analyze.py, oh_data.py,
  oh_candidates.py, train_f0_officehome.py, oh_report.py}` — **Office-Home is a promoted panel row
  and its split definition is unrecoverable without these**, which is why `DATA.md` marks Office-Home
  UNPINNED.
* `experiments/kbound/vendored_from_elara/**` (18 files) — the empirical-Bernstein switching
  certificate the theory appendix cites.
* `experiments/kbound/theory_validation/frontier_decisive/**` (7 files).

**Action:** on the source Mac, `brctl download` the tree (or Finder → right-click → Download Now on
`experiments/kbound/`), then re-run the verification snippet at the end of `PLACEHOLDER_INVENTORY.md`.
**~1 hour** if iCloud still holds them; **unrecoverable if evicted.**

### 2. Camelyon17 — two sealed files, plus a third nobody listed
`audits/integrity_2026-06-20/camelyon_reconciliation/VERDICT_phase1.md` (4,179 B, sha256
`a84c639d…`) and `recon_results.json` (2,719 B, sha256 `0409c221…`) are the only 2 of 72 sealed files
absent; the other 70 hash correctly, re-verified today. A **third** file,
`camelyon_G_reconciliation.py`, is named as evidence by the reconciliation YAML but was never sealed,
so its restoration cannot be verified.

The promoted triple 0.0000 / 0.0000 / 0.1381 at n=18 exists under change control at
`research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29`, but **nothing recomputes it** and the
promoted FA_u = 0 is recorded nowhere. Live Camelyon artifacts give false_adapt 0.0256 (n=54) and
0.0329 (n=324) instead.

**Action:** restore the directory from backup (**hours**, independently verifiable against the two
sealed hashes), or re-run `camelyon_G_reconciliation.py` and re-seal (**~1 day**, and *not*
verifiable because that script itself was never sealed). **Recommendation on record from the cleanup
agent: cut the row rather than ship it.** That is a claim decision and was deliberately left to the
author; the row currently ships with the `NOT_REPRODUCIBLE_FROM_RELEASE` flag and a dashed table row.

### 3. iWildCam — the calibration pool size is undeterminable from the release
**This is the one open item the cleanup round handed over and nobody closed, and it is recorded here
rather than patched, because patching it would require asserting a provenance that cannot be
verified.**

The facts, each checked directly:
* The promoted iWildCam row (N=72, 1 ADAPT, 60 FREEZE, 11 ABSTAIN, regret 0.0041/0.1028/0.0041) is
  one of the **four tracks named in the abstract**.
* Its named raw record file `experiments/kbound/results/iwildcam_full_test/result_e40faf29.json`
  is **absent** (only `PARTIAL_test_57cond.json` is present) — the manifest already says so.
* `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`, the file the manifest names as the source,
  records **no calibration-pool size** at all (keys: `method`, `B`, `alpha`, `wins`).
* Two shipped iWildCam scoring paths do **not** match the declared rule: `analyze_iwildcam_kbound.py`
  (source-CV, source `n < 9`, so under the unclamped rule `k > n` → `+inf` → ABSTAIN everywhere), and
  `run_iwildcam_kga_router.py:438`, which used numpy's interpolated `np.quantile` until it was
  converted in this fix round.
* Therefore **whether the promoted row is reproducible under the rule the paper declares cannot be
  settled from the release.** `kbound_decide.py`'s own docstring says such rows "must be labelled
  accordingly rather than silently re-emitted"; the paper does not label this one.

**Action:** restore `result_e40faf29.json` (**hours**, if it exists in backup) — that settles it
immediately. Otherwise re-run the iWildCam protocol under the declared rule (**~1 day** of GPU) and
either confirm the row or report the track as uncertifiable at α = 0.10. **Do not re-enable the
clamp** — `tests/test_one_radius_rule.py` fails if anyone does. Until then, one honest sentence in
`tab:decision-accounting`'s footnote is the minimum; the larger question — whether iWildCam should
remain one of "the four" in the abstract — is a claim decision for the author.

### 4. The four `bootstrap_win_cis.py` record files
All four inputs at `scripts/bootstrap_win_cis.py:37,43,47` are absent (this is the same absence that
blocks item 3 above). Registered in `STORAGE_MANIFEST.json → absent_required_artifacts`.
**Action:** restore or re-run; **hours**.

### 5. Seed-0 environment heterogeneity — needs a GPU
CIFAR-10-C seed 0 ran on Python 3.12.13 / torch 2.5.1 at a different commit; seeds 1–3 on Python
3.14.3 / torch 2.12.0; seed 4 on a third commit. ImageNet-C seed 0's argv omits `--severities 1 3 5`
and `--max-images 4000`. Disclosed at three levels (REPRODUCE.md §0a, SUBMISSION_LEDGER §10,
CIFAR10C_SAR_QUARANTINE gate 0). **Action to eliminate rather than disclose:** re-run all five seeds
under one pinned stack. Needs GPU/MPS + the raw datasets. **~1 day of compute.**

### 6. Edge split-integrity artifacts — item 8's residue
`experiments/kbound/results/edge_real_phone_v1/` holds only `publication_gate.json`;
`calibration_summary.json` and `split_audit.json` are absent, so
`test_edge_calibration_sessions_disjoint` and `test_edge_split_audit_seals_before_heldout` **skip**.
The split-integrity property is therefore **UNVERIFIED** in this release.
**Action:** commit those two files, then run with `KBOUND_REQUIRE_EDGE_ARTIFACTS=1`. **Minutes**, if
the files exist.

### 7. Figures — this blocks any real PDF
Only `fig_frontier_recovery.png`, `fig_frontier_transition.png` and `fig_frontier_fa_coverage.png`
exist. **32 referenced figures are absent**, including the six the main text needs
(`fig_certificate`, `fig_decision_flow`, `fig_decisive_decisions_cifar10c`,
`fig_decisive_pareto_cifar10c`, `fig_frontier_schematic`, `fig_natural_forest`). Without them
`pdflatex` dies on the first one. The 61-page build was obtained with 1×1 stubs.
Note: **if any figure was generated from the superseded ImageNet-C accuracy 0.427, it must be
regenerated** — the correct value is 0.409.
**Action:** regenerate from `figures/source/` or restore. **Hours.**

### 8. Two tracks whose numbers move if their calibration splits are enlarged
Removing the clamp makes the certificate infeasible at α = 0.10 for exactly two things: Camelyon17
Table VIII (n = 9 cells/seed → LOO pools of 8) and the iWildCam source-CV certificate. Every other
promoted track has a pool of at least 18 (pack §5.2: CIFAR-10-C 432/seed, ImageNet-C 27/seed, D33
130, iWildCam 72, RxRx1 60, CIFAR-10.1 48, Office-Home 35, Camelyon17 pooled 18), so **no promoted
headline changes.** Camelyon17 is already disclosed in the paper; iWildCam is item 3 above.
**Action:** enlarge those calibration splits past `min_calibration_size(alpha)`, or report the tracks
as uncertifiable at α = 0.10. **~1 day each.**

### 9. The BN-statistics-only baseline
Wired into `TTA_METHODS` but never executed. **Do not put a BN-baseline number in the paper until it
has been run.** **Half a day** of compute.

### 10. The official-settings ImageNet-C control
The paper concedes *"we chose the regime in which SAR collapses"* and states the control is unrun.
Running SAR at lr 2.5e-4 with `layer4` frozen is **~1 day** and is the single cheapest way to convert
that concession into evidence.

### 11. Item 32's Lean renames
`forced_abstention_probability` (proved `by linarith`) and `exchangeable_scores_miss_le_alpha` (which
takes its hard step as a hypothesis; no permutation-invariance definition exists in any of the 27
`.lean` files) still carry names that overstate their content.
**Action:** rename to match content (**~1 hour**), or formalise exchangeability properly — the
highest-value addition available and a tractable mathlib exercise (**days**).

### 12. `scripts/migrate_repo_name_to_kbound.sh` — one unreconstructable casualty
An already-executed one-shot directory rename whose every executable line *was* a machine-local path.
The automated sweep collapsed several distinct originals onto one portable token, so the pre-sweep
body cannot be recovered from disk. It is now a documented, deliberately non-runnable record (header
states the five steps it performed; running it explains and exits 2). **Reported, not papered over.**
**Action:** restore from a pre-sweep copy if one exists (**minutes**); otherwise the record is the
honest artifact.

---

## Residual risk

What a referee could still fault, stated without softening.

**1. The paper's one powered result rests on one adapter.** After item 17, exactly **one** arm in the
whole paper is a cluster-robust beats-both: CIFAR-10-C Tent. EATA's adapt-side interval dies as soon
as corruption is the cluster, and SAR's is positive against KGA at every unit. A referee is entitled
to ask whether "CIFAR-10-C Tent, 6 corruption-family clusters" is a benchmark panel or a single
result. The paper's own honesty about this is its best defence, but it does not make the evidence
wider.

**2. The leave-one-corruption-out ablation is a Tent-only result.** It is the strongest robustness
result in the paper, and the pack never ran it for EATA — the candidate whose cluster-robust interval
*fails*. Running it for EATA is the single highest-value cheap experiment remaining. The paper says
so explicitly rather than implying coverage.

**3. Six corruption families is a small cluster count.** The 6-cluster bootstrap intervals are wide
(2.3–3.9× the i.i.d. width) and rest on 6 units. A referee who prefers a cluster-robust sandwich or a
permutation test at that n may not accept the percentile bootstrap. The direction of the finding
(EATA fails, Tent survives) is unlikely to reverse, but the interval endpoints are soft.

**4. Three of the four "one-sided natural tracks" in the abstract are weakly evidenced.** RxRx1 makes
**0** ADAPT decisions, iWildCam **1**, Camelyon17's promoted triple is not reproducible from the
release. Only Office-Home (22 adapts, CP95 0.127) carries a testable guarantee among the four. The
paper daggers the <10-adapt tracks and says the guarantee is untested on them — but the abstract
still names all four, and a referee reading only the abstract gets a stronger impression than the
tables support. **This is the most likely single point of attack on the paper.**

**5. The FA_u = 0 results are mostly arithmetic, and the paper says so — which invites the follow-up
question.** Once you concede that `FA_u <= alpha` is forced under in-sample rank calibration, a
referee will ask why the panel reports it at all on eight tracks. The answer given (it is reported
*against the ceiling*, not against α) is correct but subtle, and the ceiling is 0 on three tracks,
where the statistic carries literally no information.

**6. The ImageNet-C operating point is chosen adversarially and the control is unrun.** lr 4e-3 is
16× SAR's published learning rate. The paper concedes it selected the regime in which SAR collapses.
Until the official-settings control runs, "KGA helps SAR" means "KGA helps SAR at a setting we chose
because SAR fails there."

**7. Provenance is now honest but not clean.** Three promoted rows (Camelyon17, Office-Home,
iWildCam) name record files that are absent, and the manifest says so in each case. That is the right
disclosure, but a referee applying a reproducibility bar strictly can reject on it. `Office-Home`'s
caveat is particularly awkward: the promoted regret 0.0157 comes from the OOF lock while the
protocol artifact on disk gives 0.002198, **7.2× smaller**.

**8. Two aggregates exist for CIFAR-10-C and both are shipped.** 1113 (promoted) and 1109
(superseded, preserved with provenance). This is the honest handling, but a referee who runs the
*old* reproduction command still gets 1109 and will need to read the `superseded_aggregate` block to
understand why. Same for EATA (1244 vs 1243).

**9. The test suite ships 37 failing/erroring tests, none of them K-Bound's.** They point at a
pre-K-Bound monorepo tree (`torch`, `fastapi`, `deploy`, `src.scripts.*`, `orchestration`,
`docker-compose.yml`, `PRODUCTION_RUNBOOK.md`) that this release does not contain. D7 fixed two such
tests; **roughly two dozen more of the same kind remain**. A reviewer who runs `pytest` sees a red
suite and has to be told it is inherited debt. Consider deleting or explicitly quarantining them
before release — this is cheap and materially changes first impressions.

**10. `MONOREPO.md` points at `archive/legacy_elara/`, which does not exist.** Dead link in a
top-level orientation file; a reviewer following it gets nothing. One-line fix, still outstanding.

**11. The clamp removal is safe for every promoted number, and the argument is by enumeration, not by
proof.** Pack §5.2 gives the pool size for all nine tracks; the smallest is 18 against a clamp that
fires only at n ≤ 8. That is airtight for *these* tracks and gives no guarantee for a track added
later. The tripwire test enforces the rule, not the feasibility of any future split.

**12. `kbound_short.tex` and `kbound.tex` have not been compiled since the edits.** `IEEEtran.cls` is
absent from this container and there is no network. `kbound_tmlr.tex` (which shares 100% of the body,
appendix, theory and bibliography with `kbound_short.tex`) compiles clean, and static ref/label
integrity was verified programmatically across **all four** drivers (0 unresolved refs, 0 duplicate
labels, 0 missing `\input`s). **Compile both yourself before submitting.**

**13. The TMLR build was verified with `lmodern` and `microtype` stubbed.** Both are absent here;
`microtype`'s font expansion requires scalable fonts that the stub cannot supply. Line breaking, and
therefore the overfull-hbox profile, will differ slightly under the real style file. The count (8,
worst 22.18 pt) is identical to the pre-edit build, so **none of the new tables overflows** — but
re-check after dropping in the real `tmlr.sty`.

---

## Retained for the record: what the verifier found wrong with the first round

The seven defects that survived the first fix round, and their disposition. Full original text is
preserved below this line in the repository history; the summary is:

| Defect | Disposition |
|---|---|
| **A.** Item 17 never assigned; recomputed numbers flip a headline | **CLOSED** — `tab:cifar-cluster`, 16 intervals + LOCO refit; claim narrowed to Tent only |
| **B.** BLOCKER survives twice in `kbound.tex` (:236, :303) | **CLOSED** — both rewritten; final sweep found and closed 6 more in `.md` |
| **C.** Promoted ImageNet-C number still the leaky in-pool one | **CLOSED** — 0.0289 / 1-of-135 / 13 ADAPT promoted in all six artifacts |
| **D.** Paper declares a clamped radius rule the library no longer implements | **CLOSED** — and a *second* unclamped implementation gap found in the shim every driver calls |
| **E.** `kbound_result_manifest.json` never touched | **CLOSED** — schema v2, all four required edits, 1109-vs-1113 resolved |
| **F.** Shipped decision audit disagrees with the paper's headline count | **CLOSED** — 1113 promoted, 1109 preserved with provenance |
| **G.** 94 files still carry machine-local paths | **CLOSED** — 94 → 9 allowlisted; 11 session-sandbox paths found that nobody had looked for |
| **H.** Item 32's `\resizebox` claim false as stated | **CLOSED** — 6 removed (3 more than enumerated); 0 in any compiled build |
| **I.** "One rule everywhere" has live exceptions the paper denies | **CLOSED** — 25 sites found, 5 converted, 20 allowlisted with reasons and frozen by test |
| **J.** DOCS agent's README rewrite broke two library tests | **CLOSED** — the *test* was stale (it referenced a module that does not exist anywhere in the repo); rewritten around LIVE/RETIRED naming tables |
| **K.** Two of four agent reports never reached disk | **CLOSED** — `report_library.md` written (382 lines). `report_paper.md` was refused twice by the harness guardrail; **its content survives only in the session transcript — capture it before it scrolls.** |
| **L.** Verified sound (LaTeX, Python, seals, routing, `reproduce_submission.sh`) | **RE-VERIFIED** in the final pass; all still sound |


---

## Appendix — the verifier's original findings, verbatim (2026-07-26, first audit)

*Retained unedited. Statuses above supersede it; the text below is the record of what was found.*

Ordered by how much damage each would do if it reached a referee.

### A. Item 17 was never assigned, was never done, and the recomputed numbers flip a headline
No agent owned item 17. The pack **already contains the answer**, and it is not the one the panel
predicted. `NUMBERS_PACK.json → item17.cifar10c_eata.cluster_robust`:

| unit | n | adapt-gap CI | beats both? |
|---|---|---|---|
| cells i.i.d. (as run) | 432 | [-0.0028, -0.0012] | yes |
| twin pairs | 216 | [-0.0031, -0.0010] | yes |
| corruption × severity | 12 | **[-0.0048, +0.00043]** | **no** |
| corruption family | 6 | **[-0.0044, +0.00035]** | **no** |

The pack flags this itself: *"*** DISAGREEMENT WITH THE PANEL *** review_6 item 17 says clustering by
corruption family leaves all CIs excluding zero. That is FALSE for EATA."* EATA also has two families
where KGA is worse than always-adapt (`gaussian_noise` +0.00022, `jpeg_compression` +0.00292). Tent
survives all four designs (widths grow 2.3–3.9×) but reverses sign on `gaussian_noise` (+0.00189).

The paper meanwhile says, at `kbound_short_body.tex:757`: *"For both Tent and EATA, the regret gaps
to **both** fixed policies are positive with 95% bootstrap CIs excluding zero."* — supported only by
the design-based mixing-ratio bootstrap and the 432-cell paired bootstrap. Three lines earlier it
concedes replicate correlation of 0.948–0.999 and *"the effective number of independent conditions is
at most 216, not 432"*, and one paragraph later it reports cluster-robust intervals **for SAR, the arm
that loses**. Reporting cluster-robust intervals only for the losing arm, in a paper about calibrated
honesty, is exactly the pattern the panel flagged as rule-shopping. Fix this before anything else in
this section.

### B. The BLOCKER survives twice in `kbound.tex`
The PAPER agent reported item 1 DONE and said the post-fix sweep was clean — the sweep covered the
*short* paper only. Two live assertions remain:

* `kbound.tex:236`, the caption of `tab:regime-summary`: *"**Headline rows** are held-out natural
  shifts (uniform no-harm)."* — and PACS sits inside that Headline block at KGA 0.0431 vs
  always-adapt 0.0176. This is the original BLOCKER verbatim: a universal claim falsified by the
  rows directly beneath it.
* `kbound.tex:303`: *"K-Bound does not claim universal improvement. **The headline deployment claim
  is uniform no-harm on natural shifts**."* — contradicted by :183 of the same file, 120 lines
  earlier, which says *"It is **not** uniformly no-harm across the panel."*

`kbound_short*.tex`, `kbound_abstract.tex` and `kbound_tmlr.tex` are clean; archived drafts under
`archive/paper_drafts_2026-07-15/` still carry it but are archived.

### C. The paper's promoted ImageNet-C number is still the leaky one
Item 4 mandates leave-one-out-of-pool everywhere. The code obeys. The paper does not: the promoted
triple in `tab:imagenetc-faithful`, `tab:decision-accounting`, `tab:primary-numeric`,
`tab:imagenetc-ci-unit` and the appendix per-seed table is **0.0264 / 0.0529 / 0.0319 at FA_u = 0.000
over 12 ADAPT decisions**, which is the in-pool exact-rank value. The clean LOO value (0.0289,
FA_u 1/135, 13 ADAPT, 107 ABSTAIN) appears in exactly one sentence, framed as *"One further
sensitivity."* The prose is honest about the consequence, but the choice of which radius is primary
is the wrong way round: the paper promotes the radius that saw the test labels and demotes the one
that did not.

Three artifacts now describe this track at two different stages:

| file | triple | FA_u | verdict |
|---|---|---|---|
| `kbound_short_body.tex` (tables) | 0.0264 | 0.000 | beats always-freeze with a condition-level CI |
| `paper/generated/uniform_verdicts.json` | 0.026422 | 0.0 | `"beats always-freeze (CI excludes zero at the condition level)"` |
| `RESULT_MANIFEST.json` / `results_source.json` | 0.0289 | 1/135 | `"DEMOTED … NOT a CI-supported beats-both"`, freeze CI [-0.0085,+0.0038] |

Pick one and propagate it. If you keep 0.0264 you must delete the word "demoted" from the two docs
files; if you move to 0.0289 you must delete "with an interval excluding zero" from the body.

### D. The paper declares a radius rule the shipped library no longer implements
Item 25 removed the `min(n, ·)` clamp from `kga/certificate.py:432`; when `k > n` the library now
warns and returns `+inf`, forcing ABSTAIN. Nobody told the paper. It still declares

> ε = ρ_(k), k = min{n, ⌈(n+1)(1−α)⌉}, α = 0.10 … *This is the only rule used anywhere in the paper*

at `kbound_short_body.tex:561`, and repeats the clamped formula at :348, :579, :601 and :1134. Worse,
the clamp is *load-bearing* in the new §sec:fa-identity argument: *"the clamp k = min{n, ·} makes ε
the maximum residual at n ≤ 9, so FA_u is forced to 0"* (:601) and *"exactly 0 at n ≤ 9"* (:1134).
Under the fixed library that is true only at exactly n = 9; at n ≤ 8 the radius is `+inf` and the
decision is ABSTAIN, so the sentence describes behaviour the code no longer has. The paper also never
mentions the infeasibility branch. Either restate the rule as unclamped-with-ABSTAIN and rework the
n ≤ 9 sentences, or document the clamp as a deliberate replay-only convention and say the released
library differs.

### E. `paper/generated/kbound_result_manifest.json` was never touched
Modified 2026-07-22 — before the fix run. `kbound_short_appendix.tex:298` calls it *"the
authoritative index for every promoted number, seed count, protocol, quantile convention, and known
replay caveat."* It is not. All four changes the DOCS agent specified are missing:

1. `cifar10c_tent.source` / `cifar10c_eata.source` still point at
   `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`, which contains
   0.0016259 / 0.0079757 / 0.1239368 — **not** the 0.0016 / 0.0079 / 0.1241 the paper prints. The
   promoted values live in `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_{tent_primary,eata_secondary}.json`
   (0.001573610885275735 / 0.007923379871580337 / 0.1240979162355264 and
   0.0012675924985497084 / 0.003268287358460603 / 0.13137893428405126).
2. `cifar10_1_K` and `rxrx1_J` still have no `source` field at all.
3. `camelyon17_ood.source` still names a directory that does not exist.
4. `imagenetc_sar` still records 0.026422222 with no LOO caveat.

This is the single defect the panel called "the provenance layer that the appendix calls the
authoritative index", and item 11 is half-applied in the direction that makes the paper assert
something false about an artifact.

### F. The shipped decision audit disagrees with the paper's headline count
`paper/generated/empirical_audit/decision_metrics.json` → CIFAR-10-C TENT records
`"count": 1109`, `"cp95_upper": 0.002698`, sourced to `stress_grid_multiseed_v1` with a
`reproduction_command`. The paper prints **1,113** and **0.0027** in seven places including the
abstract, sourced to `mixed_headtohead_v1`. The bounds agree to printed precision; the counts do not,
and a referee who runs the shipped reproduction command gets 1109. Same defect as E, different file.
Pick one aggregate per row and record it.

### G. Item 30 is reported DONE but 94 tracked files still carry machine-local paths
`grep -rl "AutoML_Flagship_V8|/Volumes/T9|/Users/pratik" --include=*.py --include=*.sh` returns 94
files. Several are real result producers, not scratch: `experiments/kbound/wilds/run_camelyon17_kbound.py`,
`experiments/kbound/poem_aetta/score_official_headtohead.py`,
`experiments/kbound/results/camelyon17_fullscale_B_v1/_locked_B_analysis.py`, and the launch shells
under `experiments/kbound/results/*/`. `EXTERNAL_STORAGE_POLICY.md:18` bans exactly this. The new
guard in `tests/test_reproducibility_hygiene.py` is deliberately scoped to `kga/` and `tests/`, so it
passes while the violation stands. Also note `STORAGE_MANIFEST.json` now *publishes* one such path
(`/Users/pratik_n/imagenetc_local`) — fine for single-blind TMLR, a de-anonymisation hazard anywhere
else.

### H. Item 32's `\resizebox` claim is false as stated
The PAPER agent reported *"grep for `\resizebox{` now returns nothing."* It returns 8 tree-wide, and
**3 of them are inside the compiled builds**: `edge/kbound_camera_main_tables.tex:48` (reached by
`kbound_tmlr.tex`) and `kbound_frontier_appendix.tex:30,67` (reached by `kbound.tex`). The other 5
are in files not currently `\input` by any driver.

### I. "One rule everywhere" has live exceptions the paper denies
`kbound_short_body.tex:564` says *"wherever an earlier draft or an archived artifact reported the
interpolated empirical quantile np.quantile(ρ, 1−α), the number has been regenerated under the rule
above."* An AST scan finds **95 live `quantile()` calls**. Most are benign (bootstrap percentiles,
binning, baseline thresholds) and four are explicitly justified archived-comparison branches
(`g8_exactrank_regen.py:98` behind `--show-archived-interpolated`;
`gate_baseline_comparison.py:200` as the named `in_pool_interp` replay mode;
`frontier_validation.py:119` as the coverage-identity demonstration; `_probe2.py:104` labelled *"the
superseded rule"*). But these are certificate radii computed with the interpolated rule and were not
converted:

* `experiments/kbound/wilds/run_iwildcam_kga_router.py:438` — **iWildCam is a promoted panel track.**
  Uses a genuine DEV/TEST split so there is no leakage, but it is the interpolated rule, not the
  declared one.
* `experiments/kbound/controlled_multimodal_d33.py:56` — the D33 multimodal check the paper cites in
  `app:d33`; in-pool **and** interpolated.
* `experiments/kbound/uniform_rule_generality.py:27`, `officehome_M_bootstrap.py:16`,
  `protocol_f_bootstrap.py:17,22`, `theory_validation/val_thm1_lecam.py:356`.
* `docs/research/kbound/scripts/{cifar_tent_mps.py:167, kbound_full_experiments.py:77,
  kbound_harmful_regime.py:62, knowability_experiment.py:117, mixed_regime_experiment.py:93,
  tta_collapse_experiment.py:104, knowability_frontier_validation.py:73,126,148,
  theory_extensions_validation.py:93}` — v1/exploratory, plus **byte-identical duplicates** of six of
  them under `src/scripts/kbound/`, which is the installed package path.

Either convert these or add one sentence naming them as superseded exploratory code. Note that
`src/scripts/kbound/cifar_tent_mps_v2.py` *is* correctly a 246-byte shim to the canonical script, so
the duplication pattern is already solved — it just was not applied to the other six.

### J. The DOCS agent's README rewrite broke two library-owned tests
`tests/test_research_naming.py::test_elara_full_form_matches_system_purpose` and
`::test_rga_full_form_remains_reliability_gated_attention` now fail because `README.md` no longer
spells out "Evidence-Layered Anomaly Reliability Architecture" / "Reliability-Gated Attention". This
is the only cross-slice breakage I found. Decide whether the test or the README is stale and fix one.

### K. Two of the four reports do not exist on disk
`/home/claude/kb_fixes/` contains `report_docs.md` and `report_scripts.md` only. The PAPER and
LIBRARY agents were both refused `Write` by a harness guardrail and returned their reports as chat
text. Their per-item old/new sentence pairs — the thing you specifically asked to be able to diff —
survive only in the session transcript. Capture them before it scrolls.

### L. Verified sound, for the record
Things I tried to break and could not:

* **LaTeX integrity.** 0 unresolved `\ref`/`\cref`/`\eqref`, 0 duplicate labels, across all three
  drivers (11 files each for the TMLR/IEEE builds, 25 for `kbound.tex`). The one missing `\input`
  (`camera_tables_values.tex`) is `\IfFileExists`-guarded.
* **TMLR build.** With `lmodern.sty` and the six absent figures stubbed, `kbound_tmlr.tex` compiles
  to **57 pages, exit 0, 0 errors, 0 undefined references or citations, 8 overfull hboxes (worst
  22.2 pt)**. Dropping the `\resizebox` wrappers did not blow up the tables.
  `kbound_short.tex` and `kbound.tex` could not be built here — `IEEEtran.cls` is absent from this
  container and there is no network. Compile them yourself before submitting.
* **Python.** 244 files compile clean, 0 syntax errors. 36 `.py` are NUL placeholders (see §Blocked).
* **Test suite.** 230 tests run: **159 passed, 28 failed, 43 skipped**, plus 11 collection errors.
  Every failure except the two in §J traces to a missing module (`torch`, `deploy`, `src.scripts.*`)
  or a missing artifact, none to the applied edits. The slice-relevant selection
  (`test_kga_canonical_rule.py` + `docs/research/kbound/tests/`) is **86 passed, 2 skipped, 0 failed**.
* **Seals.** All 71 `STORAGE_MANIFEST.sealed_evidence_checksums` and all 72 `LOCK_SEAL.json` file
  hashes re-verified against disk **today, after** the SCRIPTS agent's regeneration runs: 0
  mismatches, 2 absent (the known Camelyon pair). No cross-slice checksum drift.
* **Library routing.** `kbound_decide.backend()` returns `kga-library`;
  `selftest_radius_excludes_scored_cell()` runs and returns 0.0; `fa_ceiling(432, 0.1) = 0.09722`
  matches the paper's printed ceiling.
* **`reproduce_submission.sh` runs end to end.** I executed it: all 9 steps, final line
  `=== DONE (PASS) ===`, **exit 0**, writing `RELEASE_MANIFEST.json` and
  `reports/reproducibility_release_report.md`. Core unit tests 119 passed; theory audit
  "PASS (0 issues)"; all TARGET-1 and TARGET-2 validators PASS; step 1b degrades to an honest
  documented SKIP on the missing torch and edge artifacts rather than a silent pass. This was the
  queue's item-8 blocker (`set -euo pipefail` killing everything after step 1) and it is genuinely
  fixed. Caveat: it took roughly 25 minutes in this container, so "one command" is not "one minute" —
  say so wherever you point a reviewer at it.
* **File ownership.** No file appears in two slices' `files_touched`. The only interference is §J.
* **Numbers.** Every headline value I could trace — PACS, ImageNet-R, the SAR quarantine, the ε
  coefficients of variation, the CIFAR triples and adapt counts, the CP95 bounds, all three
  ImageNet-C intervals, and all five §0.4 rule-change disclosures — matches `NUMBERS_PACK.json` to
  the printed precision. The exceptions are C, E and F above.
