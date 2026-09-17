# report_library.md — LIBRARY / TESTS / SCRIPTS / DOCS slice

**Written 2026-07-26.** This is the report the LIBRARY agent was refused permission to write
(defect **K** in `FIXES_APPLIED.md`). Part 1 reconstructs the original library-side fix-queue items
by reading the files as they now stand, not from any agent's account of them. Part 2 covers the
defects assigned to this slice: **D7, D8, D9, D10, D11**.

Every number below was measured in this container today. Where a claim rests on an artifact I could
not read, I say so.

---

## 0. Headline

| | before | after |
|---|---|---|
| `tests/` suite | 129 passed, **28 failed**, 41 skipped, 11 collection errors | 168 passed, **26 failed**, 41 skipped, 11 collection errors |
| the two README-broken tests | FAILED | pass |
| other test paths (`docs/research/kbound/{tests,kbound_pkg/tests,kbound_repro/tests}`) | — | 177 passed, 2 skipped |
| tracked `.py`/`.sh` with machine-local paths | **94** | **9**, each named with a reason |
| files hard-coding a Cowork **session-sandbox** mount | **11** (nobody had looked) | **0** |
| radius rules implemented | **2** (library unclamped, driver shim clamped) | **1** |
| `decide_kga` forks with a body | 0 (but nothing enforced it) | 0, **now enforced by a test** |
| interpolated certificate radii, tree-wide | 25 sites, undocumented | 20 sites, each allowlisted with a reason; 5 converted |

No test was deleted, skipped or `xfail`ed to get there. The 26 remaining failures and 11 collection
errors are all pre-existing and all trace to `torch` / `fastapi` / `src.scripts.*` being absent or to
missing artifacts; the failure set after my work is a strict subset of the failure set before it
(`diff` of the two `FAILED` lists shows only the two removals).

**One finding in here changes a promoted result and needs the paper agent.** See §D9(c).

---

## Part 1 — the original library-side items, re-derived from disk

### Item 25 — exact-rank quantile, no clamp, honest small-*n*
`kga/certificate.py` implements `split_conformal_rank_radius` as `eps = r_(k)`,
`k = ceil((n+1)(1-alpha))`, with `k > n` giving `+inf` and a `UserWarning`, or
`InsufficientCalibrationError` under `on_infeasible='raise'`. `min_calibration_size(alpha)` returns
`ceil(1/alpha) - 1` (9 at 0.10, 19 at 0.05, 49 at 0.02). `kga/routing.py::route_panel` applies the
guard at the Bonferroni level `alpha/K`. **Verified present. Two residual clamp surfaces found and
removed — see D9.**

### Item 26 — `benefit_range` mandatory
`empirical_bernstein` and `hoeffding` both take `benefit_range` keyword-only with no default;
`Certificate.interval_level` distinguishes `1-2*alpha` from `1-alpha`. **Verified present**, pinned by
`tests/test_kga_canonical_rule.py::TestBenefitRangeRequired`.

### Item 29 — the CLI is not a constant-ABSTAIN generator
`kga/cli.py::decide` produces varying decisions; `kga/evidence.py` importance-weight direction
corrected; `AnytimeMulticandidatePanel` advances all K e-processes before testing rejection.
**Verified present**, pinned by `TestCliIsNotAConstantAbstainGenerator`.

### Item 4 — leave-one-out-of-pool
`conformal_radii_loo` computes cell *i*'s radius from the other *n*-1 residuals.
`kga/policy.py::decide_kga` uses it by default. **Verified present.** Note the knock-on that the
original write-up under-stated: LOO turns an *n*-cell track into pools of *n*-1, so feasibility needs
`N >= 10` cells at `alpha = 0.10`, not 9. That is what bites in D9(c).

### Item 15 — one decision path
Six `decide_kga` definitions survive outside the library
(`experiments/kbound/wilds/analysis.py`, `docs/research/kbound/scripts/{cifar_tent_mps_v2,
kga_breadth, frontier_validation, run_decision_baselines, run_wilds_camelyon17}.py`). I parsed each
with `ast` and confirmed all six are docstring + a single delegating `return`. **Verified — and now
enforced**, see D10.

### Item 30 (library half) — reproducibility hygiene
`stable_seed` is a blake2b digest, not the salted builtin `hash()`;
`tests/test_reproducibility_hygiene.py` proves process-stability across `PYTHONHASHSEED` **and**
proves the control (that `hash()` really is salted), so the assertion is not vacuous. **Verified
present.** The machine-local-path half of item 30 was reported DONE and was not; that is D8.

---

## Part 2 — the surviving defects

### D7 — the README rewrite broke two library tests. **CLOSED.**

**What actually broke.** `tests/test_research_naming.py` asserted that
`"Evidence-Layered Anomaly Reliability Architecture"` and `"Reliability-Gated Attention"` appear in
**both** `README.md` and `src/uais/fusion/attention/__init__.py`.

**Which side was wrong: the test.** Three independent reasons, all checkable:

1. `src/uais/` **does not exist** anywhere in this repository. The second path in each loop could not
   be read at all, so the test would have failed on `FileNotFoundError` even if the README had kept
   the strings. It only ever passed against a tree that no longer exists.
2. `MONOREPO.md` states the repository is "**K-Bound / KGA only**". Grepping the whole tree for
   `Evidence-Layered Anomaly Reliability Architecture` returns exactly one hit: the test file itself.
   The expansion is documented nowhere the test could find it.
3. The paths were relative (`Path("README.md")`), so the test also depended on pytest being invoked
   from the repo root.

Re-adding ELARA branding to a K-Bound README to make a test pass would have been the wrong
direction — it would make the README assert something about code the release does not ship.

**What I did instead of deleting or xfailing.** The invariant the file was protecting — *an acronym is
expanded in the document that introduces it and in the module that implements it* — is real, so I
re-pointed it at the naming this repository actually carries and added a tripwire for the retired one:

* `README.md`: `**KGA** is the practical finite-sample wrapper` -> `**KGA** — **Knowability-Guided
  Adaptation** — is the practical finite-sample wrapper`. This is a genuine gap closed: the README
  used the acronym repeatedly and never once expanded it.
* `tests/test_research_naming.py` rewritten around two tables. `LIVE_NAMING` requires the expansion in
  `README.md` **and** in `kga/__init__.py`. `RETIRED_NAMING` keeps ELARA and RGA listed and asserts
  that the module is either absent *and* `src/uais/` is fully gone (so the test cannot pass on a
  half-removed tree), or present *with* its expansion in both the module and the README. If anyone
  restores the package without the documentation, the test fails and the message tells them to move
  the entry into `LIVE_NAMING`.
* Paths now resolve from `Path(__file__).resolve().parents[1]`, not from the CWD.

3 tests, 3 pass. This is the only cross-slice breakage the verifier found and it is gone.

### D8 — 94 files carried machine-local paths. **CLOSED: 94 -> 9.**

**Before.** `grep -rlE "AutoML_Flagship_V8|/Volumes/T9|/Users/pratik" --include=*.py --include=*.sh`
returned **94** files (49 `.sh`, 45 `.py`; 191 matching lines). The existing guard,
`tests/test_reproducibility_hygiene.py::TestNoMachineLocalPaths`, was scoped to `kga/` and `tests/`
and said so in its own docstring — "the two trees this file can speak for" — so it passed green
through the entire violation. `EXTERNAL_STORAGE_POLICY.md:18` bans exactly this.

**The dominant case was easy and nobody had noticed.** 62 of the 94 contained *only* the old absolute
path of the repository root itself (`/Volumes/T9/uav/AutoML_Flagship_V8`,
`$HOME/Documents/AutoML_Flagship_V8`, `/Users/pratik_n/Documents/K-Bound`). Those are pure
repo-relative rewrites with no policy decision to make.

**A class nobody had looked for.** Widening the grep to `/sessions/` found **11 further files**
hard-coding a Cowork **session-sandbox mount**
(`/sessions/<container-name>/mnt/uav/AutoML_Flagship_V8/...`), including
`experiments/kbound/results/camelyon17_fullscale_B_v1/_locked_B_analysis.py` and
`experiments/kbound/results/stress_grid_multiseed_v1/_locked_forest.py`. These are strictly worse than
a home directory: they are valid only inside one ephemeral container that no longer exists. All 11 are
fixed.

**How each class was fixed.**

| class | fix |
|---|---|
| old repo root | `.sh`: a 9-line `_kb_find_root()` preamble that walks up to `pyproject.toml`, overridable with `$KBOUND_REPO_ROOT`. `.py`: an equivalent `_kb_repo_root()` returning `KB_REPO_ROOT`. Nothing depends on the CWD. |
| genuinely external data | **one** documented variable, `KBOUND_EXTERNAL_ROOT`, with a fixed layout declared in `docs/research/kbound/kbound_repro/paths.py::EXTERNAL_LAYOUT` (`datasets/wilds`, `imagenetc_local`, `kbound_rxrx1_{ckpt,data,results}`, `kbound_inr_results`, `tmp`, `torch_cache`). |
| interpreters | `$KBOUND_PYTHON` (default `python3`) and `$KBOUND_VENV` (default `$HOME/.venv_wilds`), replacing hard-coded `/Users/pratik_n/.venv_wilds/bin/python` and `/opt/anaconda3/envs/aetta/bin/python`. |

**The error, as required, is loud and has no home-directory fallback.**
`kbound_repro.paths.external_root()` raises `ExternalRootUnset` when the variable is unset, printing
the whole expected layout and this sentence:

> There is no default: this used to be one author's external SSD, and silently substituting `$HOME`
> would write gigabytes somewhere you did not choose.

Shell scripts use `: "${KBOUND_EXTERNAL_ROOT:?...}"`, which aborts with the same guidance.

**The guard now matches the policy.** `TestNoMachineLocalPaths` scans the **whole repository**, `.sh`
as well as `.py`, and **does not exempt comment lines** — a runbook comment telling a reader to `cd
/Volumes/T9/...` is exactly as unusable as executable code, and exempting comments is how 94 files
stayed invisible. A second test fails if an allowlist entry outlives its violation, so the list cannot
rot into decoration.

**The 9 I deliberately left, and why.** Every one either *detects* the pattern or *documents* it; none
depends on such a path to run. They are in `MACHINE_LOCAL_ALLOWLIST` in the test itself:

| file | reason |
|---|---|
| `tests/test_reproducibility_hygiene.py` | the guard; it names the fragments it bans |
| `tests/test_rxrx1_9plus_launcher.py` | asserts the launcher does **not** hard-code a home-directory checkpoint |
| `docs/research/kbound/kbound_repro/storage.py` | the scanner; the fragments are its regexes |
| `docs/research/kbound/kbound_repro/check_repo.py` | the scanner's CLI; documents what it flags |
| `docs/research/kbound/kbound_repro/paths.py` | prose describing which paths it replaces |
| `docs/research/kbound/kbound_repro/tests/test_storage.py` | builds a synthetic violating file to test the scanner |
| `docs/research/kbound/scrub_submission.py` | the anonymiser; the fragment is a substitution pattern |
| `docs/research/kbound/scripts/code_audit_uav.py` | one prose line recording the volume's historical name |
| `scripts/migrate_repo_name_to_kbound.sh` | record of the completed rename; the old name is its subject |

**One casualty, reported rather than papered over.** `scripts/migrate_repo_name_to_kbound.sh` was a
one-shot directory-rename script that had **already been executed**; every executable line of it was a
literal machine-local path, because that was its subject. The automated sweep collapsed several
distinct originals onto the same portable token, leaving it syntactically broken and
unreconstructable. I did **not** invent a plausible body. It is now a documented, deliberately
non-runnable record: the header states exactly what the script did (the five steps, including which
trees it skipped and why), and running it prints an explanation and exits 2.

**Non-executable files: a deliberate policy line.** 91 `.md`/`.json`/`.yaml` files also mention such
paths. Most are **run manifests under `experiments/kbound/results/`** that record where a run
actually happened — that is provenance, and rewriting it would falsify the record;
`kbound_repro/storage.py::scan_absolute_paths` already encodes this by only scanning executable
suffixes. I fixed the two categories where the path is an *instruction* rather than a *record*:

* 10 runbooks/checklists under `docs/research/kbound/` (`RUN_ON_MAC.md`, `RUNSHEET.md`,
  `RUNSHEET_WAVE7.md`, `gapclose_wave5/RUNSHEET_WAVE6.md`, `RELEASE_CHECKLIST.md`,
  `RELEASE_10X_TRACK.md`, `FOLDIN_KICKOFF.md`, `notebooks/README.md`,
  `formal/LEAN_COVERAGE_UPGRADE_PLAN.md`, `reports/READINESS_85PLUS.md`) — 23 `cd`/`export` lines now
  use `$KBOUND_REPO_ROOT` / `$KBOUND_EXTERNAL_ROOT`.
* `docs/research/kbound/STORAGE_MANIFEST.json` published `/Users/pratik_n/imagenetc_local` in an
  `expected_location` field — the de-anonymisation hazard the verifier flagged. The private path is
  gone; the field now names `$KBOUND_IMAGENETC_ROOT` and points at `DATA.md` for acquisition. **JSON
  re-parsed; all 69 present `sealed_evidence_checksums` still verify.**

**Deliberately left (non-executable):** four dated historical audit reports
(`reports/CODE_AUDIT_UAV.md`, `reports/LOCAL_TODO_CLOSURE_2026-07-08.md`,
`reports/NONTRAINING_CLOSURE_REVIEW_2026-07-21.md`, `reports/T9_LOCAL_EVIDENCE_AUDIT_2026-07-21.md`)
and the per-run `result_manifest.json` files. These are records of what was audited or run, on a named
date, at a named path. If the author wants them scrubbed for anonymity that is a release-time
`scrub_submission.py` pass, not a source edit.

**A false claim corrected while I was there.** `REPRO_HARDENING_REPORT.md` said "**70
legacy/executable scripts still hard-code `/Volumes/T9` or `/Users/pratik_n`** and are listed as a
remediation backlog; several are the *active-training* runners and were deliberately not edited."
Neither half survived: the count was 94, **no such list existed in the repository**, and the
"deliberately not edited" set included result producers nobody could run without editing them first
(`experiments/kbound/wilds/run_camelyon17_kbound.py`,
`experiments/kbound/poem_aetta/score_official_headtohead.py`, the launch shells under
`experiments/kbound/results/*/`). A dated correction now stands in its place.

**Seals re-verified after every edit:** `STORAGE_MANIFEST.sealed_evidence_checksums` 69/69 match (2
absent — the known Camelyon pair); `LOCK_SEAL.json` 70/70 match (2 absent, same pair). **0
mismatches.** None of the files I touched is under seal.

### D9 — one radius rule, stated once and implemented once. **CLOSED, with a consequence.**

**(a) The second implementation, which nobody had reported.** Item 25 removed the clamp from
`kga/certificate.py`. It did **not** remove it from
`docs/research/kbound/scripts/kbound_decide.py` — the shim every driver actually calls — which kept
`clamp="min_n"` as its **default**, and justified it in its own docstring:

> `clamp="min_n"` (default) `k <- min(n, k)`; eps = max residual when k > n. **This is what
> `kga/certificate.split_conformal_rank_radius` does today** and what `NUMBERS_PACK.md` used, so it is
> the default: re-running the fixed code reproduces the pack.

The first sentence was false the moment item 25 landed. So the shipped library and the code that
produces the tables implemented **different rules** for every pool of `n <= 8` at `alpha = 0.10`, and
the paper declared a third thing. `clamp` was a parameter on `conformal_radius`, `radii_in_pool`,
`radii_loo`, `radii_holdout`, `decide_kga` and `decide_from_records`.

**(b) What I removed.**

* `kga/certificate.py`: `on_infeasible` is now `{'inf','raise'}`; passing `'clamp'` raises a
  `ValueError` that names the replacement. The superseded value is still computable, but only from a
  new, separately named `legacy_clamped_radius()`, documented as "SUPERSEDED, under-covering rule
  `r_(min(n,k))`. Never call for a new number," and reachable from **no** decision path. The
  Monte-Carlo under-coverage test (genuine evidence for the paper: the clamped rule covers `n/(n+1)`,
  not `1-alpha`) now measures that function by name instead of a mode of the canonical one.
  `conformal_attained_level` keeps its `min(n, .)` but its docstring now says explicitly that it is a
  *ceiling on attainable coverage*, not a radius clamp.
* `kga/policy.py`: `decide_kga` no longer accepts `on_infeasible` at all. Its docstring used to claim
  "exactly one rule, with no options that change the statistics" while exposing both `calibration` and
  `on_infeasible`; it now states the truth — one radius rule, one switch (`calibration`), and that
  switch replays an archived artifact and is forbidden for a new number.
* `docs/research/kbound/scripts/kbound_decide.py`: `clamp` removed from all six signatures. The local
  fallback (`_rank_radius_local`, used only in a bare checkout) now warns and returns `+inf`, matching
  the library branch for branch.

**(c) THE CONSEQUENCE — this needs the paper agent.** The clamp only ever fired for pools of
`n <= 8` at `alpha = 0.10`. Every promoted panel track has a larger pool (`NUMBERS_PACK.md` 5.2:
CIFAR-10-C 432/seed, ImageNet-C 27/seed, D33 130, iWildCam 72, RxRx1 60, CIFAR-10.1 48, Office-Home
35, Camelyon17 pooled 18), so **no promoted headline number changes**. Two rows do change:

1. **Camelyon17 Table VIII**, `n = 9` cells per seed. Under the default leave-one-out pool the pools
   are size 8, the exact rank `k = 9` exceeds `n`, and no finite radius attains `1-alpha`: **every
   cell ABSTAINs.** The archived per-seed exact-rank column was produced under the clamp.
   `run_wilds_camelyon17.py` previously documented this as "`k = min(9, ceil(10*0.9)) = 9`, so eps is
   the MAXIMUM residual and FA_u is forced to exactly 0" — that describes the *in-pool* computation
   under the clamped rule, and it is now corrected in place.
2. **iWildCam's source-CV certificate** in `experiments/kbound/wilds/analyze_iwildcam_kbound.py`,
   whose source split has `n < 9`. Same conclusion. The archived iWildCam row (N = 72, 1 ADAPT, 60
   FREEZE, 11 ABSTAIN) is **not reproducible under the declared rule**. Both docstrings now say so and
   say what to do instead: enlarge the split past `min_calibration_size(alpha)`, or report the track as
   uncertifiable at `alpha = 0.10`. Neither says "re-enable the clamp".

I did not recompute either track — that needs artifacts and a GPU I do not have here. **Flagged for
the paper agent, not silently absorbed.**

**(d) The rule, written down once for quotation.** `docs/research/kbound/THEORY_TO_CODE_MAP.md` gains
a new **1a "The radius rule — stated once, verbatim, for quotation"**: a block-quoted normative
statement of `k`, `eps`, the LOO pool, the infeasible branch and the strict trichotomy; three things
the rule *is not* (not `np.quantile`, not clamped, not silently feasible at every `n`); the
feasibility thresholds 9 / 19 / 49; a table of where each piece is implemented; where it is enforced;
and the (c) consequence with the per-track pool sizes. The paper agent can quote it verbatim.

**(e) The tripwire.** New file `tests/test_one_radius_rule.py`, 34 tests, all passing. Source-level
(AST, not regex): `min(n, ceil(...))` may appear only in three allowlisted, decision-path-free
functions (`conformal_attained_level`, `legacy_clamped_radius`, `fa_ceiling`); no radius function may
call `quantile`/`percentile`; no radius function may take a `clamp` argument; `decide_kga` may not
grow an infeasibility knob. Behavioural: infeasible pools never return a finite radius and the legacy
clamped value is provably smaller; the radius is always a member of the residual set (an interpolated
quantile almost surely would not be); nine cells under LOO ABSTAIN everywhere; and the library and the
shim agree at `n` in {1,3,5,8,9,10,27,60} — **including the infeasible sizes, which is exactly where
they had drifted apart.**

### D10 — item 15 completion check. **CLOSED.**

**`decide_kga` forks.** Six definitions survive outside `kga/policy.py` and `kbound_decide.py`. I
parsed every `.py` in the tree and confirmed each is docstring + a single delegating `return`. Nothing
enforced that, so a new test does:
`test_every_decide_kga_fork_is_a_bodiless_delegation` fails if any of them grows a body back.

**Interpolated radii.** A tree-wide AST scan (any `quantile`/`percentile` call bound to a
radius-shaped name, plus `return <quantile>` inside a function named `*radius*`/`*eps*`/`*conformal*`)
found **25 sites**, four more than the verifier's section I listed. **Five converted to the canonical
call**, chosen because they are on promoted or paper-cited paths:

| file | why it mattered |
|---|---|
| `experiments/kbound/wilds/run_iwildcam_kga_router.py:438` | **iWildCam is a promoted panel track** — the only promoted track still scored under an undeclared rule. Now `kga.certificate.split_conformal_rank_radius` at the Bonferroni level `delta_K = alpha/K`. Docstring records that `K = 2` at `alpha = 0.10` needs `n >= 19` DEV windows. |
| `experiments/kbound/controlled_multimodal_d33.py:56` | the D33 check the paper cites in `app:d33` |
| `experiments/kbound/ppi_micro_probe.py:98` | the D25 PPI leave-one-category-out radius |
| `experiments/kbound/theory_validation/val_thm1_lecam.py:356` | Theorem-1 numerical validator |
| `experiments/kbound/{uniform_rule_generality,officehome_M_bootstrap,protocol_f_bootstrap}.py` | three re-scoring probes; genuine DEV/TEST splits, so item 4 never applied — only the rule was wrong. Their *bootstrap* percentiles are untouched, correctly. |

**Six byte-identical forks deleted outright.** `src/scripts/kbound/{cifar_tent_mps,
kbound_full_experiments, knowability_experiment, mixed_regime_experiment, tta_collapse_experiment,
kbound_harmful_regime}.py` were copies of the canonical scripts on the **installed package path**,
each carrying its own interpolated radius (`kbound_harmful_regime.py` had additionally drifted:
different figure labels, 130 dpi vs 300). They are now 6-line delegating shims via
`src/scripts/kbound/_canonical.py`, applying the pattern `cifar_tent_mps_v2.py` already used. That is
six fewer places for the rule to fork.

**Eight superseded v1 scripts banner-marked, not silently converted.**
`docs/research/kbound/scripts/{cifar_tent_mps, kbound_full_experiments, kbound_harmful_regime,
knowability_experiment, mixed_regime_experiment, tta_collapse_experiment,
knowability_frontier_validation, theory_extensions_validation}.py` each carry a `SUPERSEDED RULE --
EXPLORATORY v1 CODE` banner stating the declared rule, why this file does not use it, that no promoted
number comes from it, and that converting it in place would make its archived outputs
irreproducible. Deliberate: converting exploratory code silently would break the only thing it is good
for.

**The survivors are enumerated with reasons and frozen.** `INTERPOLATED_RADIUS_ALLOWLIST` in
`tests/test_one_radius_rule.py` lists all 20 remaining sites with a one-line reason each, in four
groups: superseded v1 (8); deliberate replays *of* the superseded rule, whose whole purpose is the
comparison (`eps_recal/_probe2.py` labelled "the superseded rule", `eps_recal_camelyon.py`);
never-promoted exploratory sweeps (`win_hunt_A_universal_gate.py`, `win_hunt_E_universal7.py`,
`verify_realshift_win.py`, `deepgrid_audit.py` — the last is a *parametric-bootstrap* deviation
quantile, not a conformal residual radius at all; `test_3dadam_{bootstrap,namedcond}.py`, a different
benchmark); and immutable archived analysis scripts
(`camelyon17_fullscale_B_v1/_locked_B_analysis.py`, `estimator_dryrun/dryrun.py`) which *are* the
record of how an archived number was made. Two entries are not radii at all:
`analyze_{camelyon,iwildcam}_kbound.py`'s `tau_star` is a **baseline router's** threshold on a source
statistic, which the K-Bound rule does not govern. The test fails on any new entry **and** on any
entry that no longer has a violation.

### D11 — `report_library.md`. **CLOSED.** This file.

---

## What I did not do, and why

* **I did not recompute Camelyon17 Table VIII or iWildCam** under the unclamped rule. Both need
  artifacts and compute this container does not have. D9(c) states precisely what changes and what
  the two honest options are.
* **I did not convert the 20 allowlisted interpolated radii.** Each has a written reason; five of them
  exist *to demonstrate* the superseded rule and converting them would delete the comparison.
* **I did not touch any `.tex` file or anything under `docs/research/kbound/paper/`** — the paper
  agent's slice. The only file outside my listed ownership I edited is `README.md`, which D7
  explicitly authorised.
* **I did not reconstruct `migrate_repo_name_to_kbound.sh`.** Its body was unrecoverable and I will
  not invent one. See D8.

## For the paper agent

1. **Quote `THEORY_TO_CODE_MAP.md` 1a** for the radius rule. The paper's `kbound_short_body.tex:561`
   still declares `k = min{n, ceil((n+1)(1-alpha))}` and leans on the clamp being load-bearing at
   `:601` and `:1134`; the shipped library and the drivers now agree that there is no clamp, and 1a
   is the text they agree on.
2. **The "exactly 0 at n <= 9" sentences need reworking**, and not only for the reason the verifier
   gave. Under the declared **leave-one-out** pool the threshold is `N >= 10` **cells**, not `n >= 9`
   residuals. At `N = 9` the radius is `+inf` and the decision is ABSTAIN, so there is no FA_u to
   discuss.
3. **Two rows are affected**: Camelyon17 Table VIII per-seed, and iWildCam. See D9(c).
4. **iWildCam's radius rule changed** from interpolated to exact-rank (D10). Any iWildCam number
   re-derived from `run_iwildcam_kga_router.py` after today is under a different rule than the
   archived one.

## Reproduction

```bash
python3 -m pytest tests/ --continue-on-collection-errors     # 168 passed, 26 failed, 41 skipped, 11 errors
python3 -m pytest tests/test_one_radius_rule.py              # 34 passed  (D9 + D10 tripwires)
python3 -m pytest tests/test_research_naming.py              # 3 passed   (D7)
python3 -m pytest tests/test_reproducibility_hygiene.py      # 8 passed   (D8)
python3 -m pytest docs/research/kbound/tests docs/research/kbound/kbound_pkg/tests \
                  docs/research/kbound/kbound_repro/tests    # 177 passed, 2 skipped
grep -rlE "AutoML_Flagship_V8|/Volumes/T9|/Users/pratik" --include=*.py --include=*.sh . | wc -l   # 9
```
