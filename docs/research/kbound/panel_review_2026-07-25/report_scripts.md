# Fix-queue report — SCRIPTS slice

Owner: experiment / analysis scripts.
Scope: `docs/research/kbound/scripts/**`, `src/scripts/kbound/**`, `experiments/kbound/**/*.py`,
`docs/research/kbound/experiments/**` (scripts only). No `.tex`, no `.md`, no `kga/`, no `tests/`.

Every Python file touched was `python3 -m py_compile`d; a whole-tree compile sweep over all
non-placeholder `.py` under the three owned roots passes.

---

## Resume audit

| Item | State found | This session |
|---|---|---|
| 3 (`g8_exactrank_ci.py`) | already done | verified by running; no edit |
| 4 (radius, 5 scripts + 7 forks + 2 inlined) | already done at all 14 named sites | verified against NUMBERS_PACK by re-running; extended to 5 unnamed drivers; added an item-4 regression self-check |
| 8 (repro paths) | already done | verified by running `reproduce_submission.sh` end-to-end (PASS) and `ablation_exactrank.py`; extended the same fix to `ablation_sweep.py` (identical defects, unnamed) |
| 14 (`frontier_validation.py` + `frontier_sweep.py`) | already done | verified by running both |
| 15 (route drivers through the library) | already done via the `kbound_decide` shim | verified all 7 forks delegate; audited every call site for the scalar→array `eps` change |
| 16 (BN-only arm) | already done | verified; added eval-mode / eval-chunk disclosure + manifest stamping to the WILDS runner |
| 18 (baseline parity) | already done | verified against NUMBERS_PACK §7.2 on real data |
| 28 (one false-adapt definition) | **partially done** | **finished**: 2 remaining `beats_both` gates, 3 report/print sites, 1 strict-`<` marginal rate |
| 30 (repro hygiene) | **partially done** | **finished the in-scope remainder** (`ppi_micro_probe.py`) + sklearn/PYTHONHASHSEED stamping in both production manifests |

Nothing was applied twice; every edit below was made against the file's current state.

---

### Item 3 — DONE (verified, no edit needed)

`scripts/g8_exactrank_ci.py` (178 lines) already implements the fix: `--unit
{condition,cell_seed,seed,family}` defaulting to `condition`, seed-averaging inside a unit before
resampling (`paired_boot`, `:118-125`), 20 000 replicates, `SEED = 20260720`, paired by
construction (the per-row difference is formed first), and a stdout `[WARNING]` for `cell_seed`
(135 correlated rows) and `family` (only 3 clusters). The docstring `:18-32` carries the "why the
cell-seed unit is wrong" argument.

Verified by running:

```
--calibration in_pool --unit condition:  adapt [-0.0808,+0.0175] includes 0
                                         freeze [-0.0093,-0.0023] EXCLUDES 0
--calibration loo     --unit condition:  adapt [-0.0755,+0.0179] includes 0
                                         freeze [-0.0086,+0.0038] includes 0
```
Pack §3.1/§0.1: in-pool `[-0.0808,+0.0175]`/`[-0.0093,-0.0023]`; LOO `[-0.0755,+0.0181]`/`[-0.0085,+0.0038]`.
4th-decimal differences are bootstrap-stream noise; verdicts identical. The script prints
`CI-supported beats-both at unit=condition: False` in both cases — the demotion the item asked for,
and it matches NUMBERS_PACK §0.1's disagreement with review_6 (under LOO the freeze gap also loses
its interval).

Not mine: Table XV / `tab:primary-numeric` / `uniform_verdicts.json`.

---

### Item 4 — DONE (verified against NUMBERS_PACK; extended beyond the named sites)

All 14 named sites route through `scripts/kbound_decide.py`, the single shim calling
`kga.certificate.split_conformal_rank_radius` / `kga.policy.decide`. Runtime `backend()` reports
`kga-library`.

Reproduction check I ran (`kbound_decide` on the committed dumps):

```
h2h tent x5    2160 cells  loo-vs-in_pool decisions changed = 0  FA_u 0/2160  regret 0.00158518
h2h eata x5    2160 cells                                    0  FA_u 0/2160  regret 0.00127986
stress tent    1728 cells                                    0  FA_u 0/1728  regret 0.00165509
stress eata    1728 cells                                    0  FA_u 0/1728  regret 0.00128530
stress sar     1728 cells                                    0  FA_u 0/1728  regret 0.00161690
TOTAL 9504 cells, 0 changed
```
Matches NUMBERS_PACK §4.1 to all 8 printed digits, including "0 of 9 504".

```
$ python3 scripts/g8_canonical_pooling.py --calibration loo
SAR   KGA=0.028893 adapt=0.052933 freeze=0.031894  {'ADAPT':13,'FREEZE':15,'ABSTAIN':107}  FA_u=1/135
EATA  KGA=0.000670                                                                          FA_u=1/135
TENT  0 ADAPT, KGA identical to always-freeze
```
Matches NUMBERS_PACK §4.2 exactly.

New in this session — sites the queue did not name that had the same *rule* defect. Each uses a
genuine held-out calibration split (so the leakage never applied), but each used numpy's
interpolated quantile instead of the declared exact rank:

- `experiments/kbound/wilds/analyze_iwildcam_kbound.py:113` (`fit_certificate`)
  OLD `eps = float(np.quantile(res, 1 - alpha))`
  NEW `eps = float(_kb.conformal_radius(res, alpha))` + docstring explaining why item 4 does not
  bite here and why the rule still had to change.
- `experiments/kbound/wilds/analyze_camelyon_kbound.py:73` — same change.
- `scripts/pacs_vlcs_runner.py:186` (`decide_transfer`)
  OLD `eps = float(np.quantile(np.abs(loo - Bc), 1 - alpha))` +
      `dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))`
  NEW `eps = float(_kb.conformal_radius(...))`, `dec = _kb.decide(...)`. Docstring records that this
  moves the PACS/VLCS numbers and that the committed PACS artifacts carry no `b_hat`, so the track
  must be re-run to be reported under the declared rule (NUMBERS_PACK §8.2).
- `scripts/mixed_stream_kbound.py:59` (`per_condition`)
  OLD `eps = float(np.quantile(np.abs(_loo - Bc), 1 - af.ALPHA))`; `dec = af.decide_global(...)`
  NEW `_kb.conformal_radius(...)`; `_kb.decide(..., alpha=af.ALPHA)`.
- `scripts/ablation_sweep.py:114` (`eps_q`) — see item 8.

Regression check added (`tests/` is not mine, so it lives in the module that owns the rule):
`kbound_decide.selftest_radius_excludes_scored_cell()`, `kbound_decide.py:375-410`, run by
`python3 kbound_decide.py`. It asserts (a) perturbing residual *i* to a 100x outlier leaves
`eps[i]` bit-identical under LOO, (b) `eps[j]` equals the exact-rank radius of
`np.delete(residuals, j)` for every *j*, and (c) the in-pool control **does** move under the same
perturbation, so the test cannot pass vacuously. Output:

```
item-4 regression  : PASS (scored cell excluded from its own pool; max dev vs explicit
                     np.delete recomputation 0.0e+00)
```

Not mine: `PHASE6_LEAKAGE_AUDIT.md`; the Camelyon17 Table VIII re-score into the paper.

---

### Item 8 — DONE (verified by running the whole thing)

`scripts/reproduce_submission.sh` — `set -euo pipefail` -> `set -uo pipefail` plus a `step`
harness (`:66-88`) recording PASS/FAIL/SKIP and continuing, a `required|optional` declaration per
step, a `require_file` helper (`:90-112`) that detects NUL-filled iCloud placeholders by comparing
byte counts with and without NULs (`grep -q $'\0'` does NOT work — bash collapses it to the empty
pattern, which matches everything), `export PYTHONHASHSEED=0`, and exit 1 iff a REQUIRED step failed.

Verified end-to-end; all 9 steps now execute (previously step 1 killed the run):

```
Overall: PASS (0 required step(s) failed)
1 core unit tests PASS | 1b env-dependent SKIP (torch + edge artifacts absent)
2/2b/2c PASS | 3 gate selftest PASS | 4a/4b PASS | 5 PASS | 6 PASS | 7 PASS | 8 PASS
```

`scripts/ablation_exactrank.py` — ranked `INPUT_CANDIDATES` resolver (`:59-95`) raising with the
full list of paths tried and an explicit note that `stress_grid_multiseed_v1/seed0/` has no
per-condition dump at all. `a_oracle`, absent from every committed 432-cell dump, is derived as
`max(a0, a_adapted)` (its definition), announced on stdout, and recorded under
`config.a_oracle_source` — not substituted silently. Verified:

```
[ablation] tent: 'a_oracle' absent from .../per_condition_cifar10c_tent_seed1.json; derived as max(a0, a_adapted).
ANCHOR tent alpha=0.10 (locked gate: regret 0.0017, FA_u 0): {'regret': 0.0016, 'FA_u': 0.0, ...}
```
This materialised `docs/research/kbound/experiments/kbound/results/ablation_exactrank.json`, which
was a NUL-filled placeholder; it is now real LOO exact-rank content with the provenance block above.

`scripts/gate_baseline_comparison.py` — `--from-percondition` is now the default input
(`rows_from_percondition`, `:107-140`) reading the committed dumps; `--in cifar10c_percell.json`
(the old only path, pointing at a file that exists nowhere and crashing with
`json.load(open(None))`) is retained as legacy. Verified on seed 0.

New this session: `scripts/ablation_sweep.py` had the identical two defects and was never named. Its
`load()` read the same nonexistent path and the same absent `r["a_oracle"]`. It now imports
`ablation_exactrank._resolve` and `kbound_decide.read_json` rather than re-forking them, defaults to
`seed=1`, derives `a_oracle` with an announcement, and `eps_q` uses the exact rank. Verified:
```
[ablation_sweep] tent: 'a_oracle' absent ...; derived as max(a0, a_adapted).
[ablation_sweep] tent: 432 cells (grid nominal 432) <- .../stress_grid_multiseed_v1/seed1/...
WROTE .../ablation_all.json
```

BLOCKED-NEEDS-DATA sub-parts, handled rather than hidden: the 432-cell seed-0 per-condition dumps
and `cifar10c_percell.json` are genuinely absent. Both scripts now name the exact paths tried and
what to do; `kbound_decide.read_json` (`:325-350`) distinguishes "absent" from "NUL-filled
placeholder" with a different actionable message for each.
`tests/test_calibration_split_integrity.py` is in `tests/` (not mine); `reproduce_submission.sh` now
runs it in the OPTIONAL `run_env_tests` group so its failure degrades one row, not the run.

---

### Item 14 (code half) — DONE (verified by running)

`scripts/frontier_validation.py`:
- Docstring rewritten (`:1-51`) under `WHAT THIS SCRIPT IS -- AND WHAT IT IS NOT`.
  OLD title: `frontier_validation.py -- validation of the K-Bound benefit-sign frontier.`
  NEW title: `frontier_validation.py -- ILLUSTRATION (not a test) of the K-Bound benefit-sign frontier.`
  NEW in the docstring: "A real test of the frontier needs (a) Z that is not a noisy copy of M,
  (b) gamma whose 0.9-quantile is not beta, and (c) a held-out calibration set. That experiment is
  scaffolded in `frontier_sweep.py` and has NOT been run; do not cite this file's numbers as
  evidence for the frontier claim in the paper."
- `gen_world` -> `gen_circular_world` (`:76`, aliased); `main` -> `run_illustration` (`:122`, aliased).
- Runtime assertion `_assert_circular` (`:94-108`) fails loudly if a future edit makes Z something
  other than a near-deterministic function of M, "so the label cannot silently drift away from the
  code"; `max_abs_corr_Z_M` is written to the results JSON.
- `_quantile_identity_coverage(n)` (`:111-119`) prints, next to the "empirical coverage" number, the
  coverage an in-pool empirical quantile returns for ANY data at that n.
- Results JSON carries `"status": "ILLUSTRATION -- NOT A TEST OF THE FRONTIER"` plus per-panel
  `why_this_is_not_evidence` / `coverage_caveat`.

Verified by running (~15 min):
```
FRONTIER ILLUSTRATION -- NOT A TEST (fix-queue item 14)
  Z is four noisy copies of M => residual == -gamma => eps -> 0.9*beta by algebra.
  (A) recovery: coverage 90.00% (an in-pool quantile would return 90.00% for ANY data at n=400)
      MAE 0.0536  eps 0.0975 (algebra predicts 0.09)
```

`scripts/frontier_sweep.py` (234 lines) scaffolds the real experiment: source-only ATC margin M,
beta declared from historical dev-to-deployment gaps and NOT fitted, sweep over
`beta in {0,0.02,0.05,0.10,0.20}`, leave-one-corruption-out radius, and a pre-commitment to report a
negative result. It fabricates nothing — without `--beta-source` and `--i-have-run-the-real-thing`
it prints the plan and exits 2:
```
SCAFFOLD ONLY -- no results were computed and no file was written.
```

Not mine: `kbound_short.tex:593-612`.

---

### Item 15 — DONE

Every driver routes through `scripts/kbound_decide.py`. It imports the library defensively
(`:88-104`) — `from kga.certificate import split_conformal_rank_radius`, `from kga.policy import
decide` — and falls back to a byte-identical local implementation in a bare checkout, stamping which
path ran in `BACKEND`. Confirmed at runtime: `backend(): kga-library`. I imported only the stable
public surface (`certificate`, `policy`) and did not touch `kga/`.

The 7 forks and 2 inlined copies have no bodies left; each is a signature-preserving delegation with
a docstring quoting the code it replaced. I audited every call site for the consequence of the
signature change (`eps` is now a per-cell ndarray): `cifar_tent_mps_v2.py:307-310`,
`kga_breadth.py:261-264`, `run_decision_baselines.py:220-224`,
`wilds/run_camelyon17_kbound.py:267-270`, `wilds/analysis.py:216-222`,
`wilds/per_condition_serialize.py:214-252` all serialise per-cell radii and label any scalar as
"mean of the per-cell leave-one-out-of-pool radii". `poem_aetta/run_mixed_headtohead.py:98` discards
`eps`. The two WILDS analyzers keep a genuine scalar source-fit radius.
`multicandidate_decide_kga.py` was never a fork — it already called `kga.routing.route_panel`.

`kbound_decide` documents the one place library and local path can disagree (`clamp="min_n"` vs
`"inf"` at n <= 8 — fix-queue item 25, the library agent's) and defaults to the convention
NUMBERS_PACK used, so re-running the fixed code reproduces the pack.

Not mine: the `kbound_short.tex:549-550` vs `:800-801` config-table contradiction.

---

### Item 16 — DONE

`scripts/cifar_tent_mps_v2.py`:
- `bn_adapt` (`:876-903`): source weights, target BN statistics, zero gradient steps. Clones exactly
  as the gradient arms do (same parameter set, same running-statistics nulling), runs the stream
  forward under `no_grad` so batch statistics are the target's, takes no optimiser step, returns
  `update_norm == 0` by construction. Docstring names the confound it isolates.
- `TTA_METHODS` (`:905-907`) gains `"bn": bn_adapt`; `--methods` accepts `bn` (`:1344-1346`). Wired
  but not run by default, as asked.
- `EVAL_CHUNK = 512` is a named constant (`:111-114`) documented as an operating point, not an
  implementation detail; `acc_on`'s docstring (`:947-956`) states the frozen arm is evaluated in
  `eval()` and the adapted arm in `train()` mode. `EVAL_CHUNK` appears in `policy_metrics` output
  (`:314`), every per-condition dump (`:1541`), and now the run manifest.

New this session, `scripts/run_wilds_camelyon17.py`:
- `evaluate()` docstring (`:452-465`) discloses that on this track BOTH arms are evaluated in
  `eval()` mode with running BN statistics, so the eval batch size (`_EVAL_BS = 64`) does NOT change
  the reported accuracies — the CIFAR chunk-size confound does not transfer. It also says B still
  bundles BN drift with the gradient update, and the isolating control is the `bn` arm, which has no
  WILDS counterpart.
- The run manifest gains a `"kga"` block: `eps_rule`, `eps_calibration`, `backend`,
  `eval_batch_size`, `eval_mode`.

I did NOT invent the "256 WILDS" chunk size from the queue text: the Camelyon17 runner's eval batch
size is 64 (`:518`) and, because evaluation is in `eval()` mode, it does not affect the numbers. The
manifest now records the real value.

---

### Item 18 — DONE (verified against NUMBERS_PACK §7.2 on real data)

`scripts/gate_baseline_comparison.py`:
- The parity claim was replaced. OLD (as the file itself records it): every gate is calibrated
  "LEAVE-ONE-TASK-OUT (task = corruption), exactly like KGA, so the comparison is apples-to-apples".
  NEW (`:20-45`), a `BASELINE PARITY -- READ BEFORE CITING THIS TABLE` block stating the sentence
  "was false on BOTH sides", with each rule's actual budget: gates 1-2 unfitted sign rules with zero
  fitted parameters; gates 3-4 leave-one-corruption-out with one parameter; KGA
  leave-one-CELL-out with 431 GBR fits and NOT leave-one-task-out.
- Two new rows at the gates' own budget: `KGA (no radius, LOCO)`, `KGA (certificate, LOCO)`.
- A `WHAT THE RADIUS BUYS` block (`:46-52`) making the item's argument.
- Every row prints its calibration budget as a column; the harmful-subset block is emitted.

Verified on the committed seed-0 head-to-head dump (n = 432, harmful = 146 — matching the pack's
"146/145/139/144/137 per seed" and contradicting the published caption's 149):

```
confidence gate         0.0082  FA_u 0.255   NONE -- unfitted sign rule
entropy gate            0.0084  FA_u 0.252   NONE -- unfitted sign rule
drift/KL gate           0.1240  FA_u 0.000   leave-one-CORRUPTION-out, 6 folds
ATC-style gate          0.0035  FA_u 0.111   leave-one-CORRUPTION-out, 6 folds
KGA (no radius)         0.0004  FA_u 0.046   leave-one-CELL-out GBR (431 fits)
KGA (certificate)       0.0013  FA_u 0.000   leave-one-CELL-out GBR + exact rank
KGA (no radius, LOCO)   0.0005  FA_u 0.051   gates' own budget
KGA (certificate, LOCO) 0.0055  FA_u 0.000   gates' own budget

- radius-free FA_u = 0.046 (meets the declared budget: True); certificate FA_u = 0.000
- certificate regret is 3.48x the radius-free variant's
- harmful-cell adapt rate: 0.137 without the radius (20 false adapts) -> 0.000 with it (0)
```
Same structure and magnitudes as the pack's 5-seed pooled row; residual differences are seed 0 vs
5-seed mean. At equal budget the ATC gate has lower regret than the certificate AND breaks the
declared budget (FA_u 0.111 > alpha = 0.10) — the honest comparison, stated in the script's own
docstring.

Not mine: `tab:gates` and its caption.

---

### Item 28 — DONE (finished the outstanding half)

Canonical definition: `kbound_decide.false_adapt` (`:294-315`) — a false adapt is an ADAPT decision
on a cell with `B <= 0`; returns `fa_u` (marginal, what `thm:certificate` bounds) and `fa_c`
(conditional) as separate named fields plus `n_false_adapt`, `n_adapt`, `n`. The weak inequality is
deliberate and documented: 500 archived cells have `B` exactly 0.0 and 102 of them ADAPT.

Already fixed before this session: `wilds/analysis.py:87` (the named site — both rates emitted,
`beats_both` gated on `fa_u`, `beats_both_regret_only` split out), `cifar_tent_mps_v2.py:182`,
`run_wilds_camelyon17.py`, `kga_breadth.py`'s metrics dict.

Fixed in this session:

1. `experiments/kbound/wilds/analyze_iwildcam_kbound.py:129-171` (`policy`) — a SECOND `beats_both`
   gate on the wrong quantity, in the file producing the iWildCam track.
   OLD `"beats_both": bool(rk < ra - 1e-9 and rk < rf - 1e-9 and adapt.any() and float(np.mean(B[adapt] < 0)) <= alpha)`
   NEW `... and adapt.any() and float(_fa["fa_u"]) <= alpha`, with `false_adapt_unconditional`,
   `false_adapt_conditional`, `false_adapt_definition`, `beats_both_gate`;
   `false_adapt_rate_among_adapt` retained, marked DEPRECATED, gating nothing.
2. `experiments/kbound/wilds/analyze_camelyon_kbound.py:73-108` (`policy`) — same fix, plus a defect
   the item did not name: the old gate read `(fa is None or fa <= alpha)`, so a run with ZERO ADAPT
   decisions passed the budget gate vacuously. The new gate requires `adapt.any()`.
3. `scripts/run_decision_baselines.py:68-104` (`policy_metrics`) — the baseline-comparison harness,
   where an asymmetric definition is worst: the table compared every rule on the
   conditional-strict rate, and which column you print changes the ranking (a rule that adapts
   rarely looks bad on `fa_c`, good on `fa_u`). Both now emitted for every rule; the `harmful` mask
   changed from `B < 0` to `B <= 0`. Its markdown table (`:257-273`) now has TWO columns, `FA_u`
   and `FA_c`, with a defining footnote, instead of one ambiguous `false-adapt` column.
4. `scripts/kga_breadth.py:322-329`, `:359-364`, `:350-354` — metrics dict was fixed but the console
   print and generated markdown still read the deprecated field.
   OLD print `False-adapt rate (B<0 | ADAPT): {pm['false_adapt_rate_B<0']:.3f}`
   NEW print `FA_u = Pr[ADAPT and B <= 0]: ... (k/n)` then `FA_c = Pr[B <= 0 | ADAPT]: ...`, with
   `n/a (no ADAPT decisions -- guarantee untested)` when nothing adapted; markdown row split in two.
   Also the table's rule description:
   OLD "LOO gradient-boosted Bhat +/- split-conformal eps (alpha=0.10), identical to cifar_tent_mps_v2.py"
   NEW names the exact rank `k = ceil((n+1)(1-alpha))`, the leave-one-out-of-pool pool, and says it
   is "the same function object cifar_tent_mps_v2.py calls, not a copy" — a claim the code can keep.
5. `scripts/pacs_vlcs_runner.py:287-292`
   OLD `fa_u = float(np.mean(adapt & (B < 0))); fa_c = float(np.mean(B[adapt] < 0)) if adapt.any() else 0.0`
   NEW both from `_kb.false_adapt` (weak inequality); `fa_c` is `None`, not a misleading `0.0`, when
   nothing adapted. This changes the input to the PACS `win` verdict expression, which is the point.

Not mine: `kga/policy.py` (the item asks for the definition to live there; `kbound_decide.false_adapt`
is the scripts-side single source every driver I own now calls); `SUBMISSION_LEDGER.md:89`.
Deliberately left: the deprecated `false_adapt_rate_B<0` key is still *written* by several one-shot
demo scripts (`tta_collapse_experiment.py`, `cifar_tent_mps.py`, `kbound_harmful_regime.py`,
`mixed_regime_experiment.py`, `kbound_full_experiments.py` and their `src/scripts/kbound/` twins).
None feeds a published table or gates a verdict, and removing the key would break
`collate_final.py` / `unified_result_audit.py`'s historical key lists; it is kept as a deprecated
alias everywhere it mattered.

---

### Item 30 — DONE (in-scope parts); PARTIAL on out-of-scope scratch files

F2-8 — salted `hash()` seeding. `src/scripts/kbound/cifar10c_suite.py:18-36` defines
`stable_seed(*parts)` on `hashlib.blake2b`; `:93` uses
`np.random.default_rng(stable_seed("cifar10c_suite", corr, sev))`.
OLD `r = np.random.default_rng(sev*131 + hash(corr)%9973)`.
The docstring records why it mattered: three interpreters drew three different 800-image subsamples
for the same cell, which is why `CIFAR10C_SAR_QUARANTINE.md`'s reinstatement gate #2 could not be
met. The sibling `src/scripts/kbound/imagenette_c_suite.py:81-92` had the identical defect
(`hash((corr,sev))%9991`) and the identical fix. `reproduce_submission.sh` also exports
`PYTHONHASHSEED=0`.

F3-17 — machine-local paths. All analysis scripts named or implied are parameterised through
`kbound_decide.repo_path` / `results_root` with `KBOUND_REPO_ROOT` / `KBOUND_RESULTS_ROOT`:
`g8_canonical_pooling.py`, `g8_exactrank_ci.py`, `g8_exactrank_regen.py`, `uniform_scorer.py:28-37`,
`theory_v2/realdata/eps_recal/_probe2.py:18-21`.
New this session: `experiments/kbound/ppi_micro_probe.py:13`
OLD `REPO = "/Volumes/T9/uav/AutoML_Flagship_V8"`
NEW resolves from `__file__` with a `KBOUND_REPO_ROOT` override and a comment citing
`EXTERNAL_STORAGE_POLICY.md:18`.

F2-16 — the tautological assertion. `scripts/test_sar_faithful.py:54-79`.
OLD `res_new = (w.clone() - w).norm().item(); assert res_new < 1e-9 < res_old, "SAM restore must be exact"`
— `||w - w||` is 0 by construction on a local scratch tensor and touched no project code.
NEW calls `H.sar_adapt(base, stream(), steps=5, lr=0.0, num_classes=10)` and asserts the REAL
parameter tensors come back bit-identical (with lr = 0 the SAM ascent is the only thing that can
move them), keeping the old bug's analytic drift as the contrast value. The old code is quoted
in-place so the change is diffable.

Reproducibility stamping (added this session; supports F4-14). `b_hat` comes from
`GradientBoostingRegressor(subsample=0.8)` and 0 of 43 archived run manifests record a scikit-learn
version, so eps and every decision are version-dependent. Both production runners now stamp it:
- `cifar_tent_mps_v2.py:1592-1620`: a `"kga"` block (`eps_rule`, `eps_calibration`, `backend`,
  `eval_chunk`) plus `"scikit_learn"` and `"pythonhashseed"`.
- `run_wilds_camelyon17.py:~715-740`: the same `"kga"` block plus `env.scikit_learn` and
  `env.pythonhashseed`.
This is the script half of fix-queue item 19; item 19's re-run of seed 0 is out of my slice.

PARTIAL / deliberately not done — these still hold `/Volumes/T9/...` paths:
- Shell data-prep drivers, which document one machine's data layout rather than analysis code:
  `quick_check.sh`, `full_run.sh`, `kbtrain.sh`, `kbound_{shot,train}_queue.sh`,
  `prep_internal_camelyon.sh`, `verify_imagenetc_tars.sh`, `supervise_rxrx1_9plus.sh`,
  `run_officehome_protocol_m_replicate.sh`, `run_all_multiseed.sh`. Most already accept `REPO=`;
  none produces a published number.
- `code_audit_uav.py`, whose entire subject is the `/Volumes/T9/uav` tree.
- One-shot scratch probes under `docs/research/kbound/theory_v2/realdata/` (`_inspect*.py`,
  `run_p1.py`, `run_p2.py`, `_consolidate.py`, `make_figs.py`, `realdata_audit.py`,
  `deepgrid_audit/`, `eps_recal/eps_recal_camelyon.py:62`) and under
  `experiments/kbound/results/*/` (`_cmp_extract.py`, `_cmp2.py`, `_validate_f0.py`,
  `bias_variance_diag/diag.py`). `theory_v2/` is outside my declared ownership except the
  `_probe2.py` the queue named, which is fixed. Recommend a release guard grepping tracked
  `.py`/`.sh` for `^/Volumes/` and `~/Documents/` — it catches all of these mechanically.

---

## Files changed in this session

```
docs/research/kbound/scripts/kbound_decide.py           item-4 regression selftest
docs/research/kbound/scripts/cifar_tent_mps_v2.py       manifest: kga block, sklearn, hashseed
docs/research/kbound/scripts/run_wilds_camelyon17.py    eval-mode disclosure, manifest stamping
docs/research/kbound/scripts/run_decision_baselines.py  item 28: policy_metrics + md table
docs/research/kbound/scripts/kga_breadth.py             item 28: print + md table + rule text
docs/research/kbound/scripts/pacs_vlcs_runner.py        items 4/15/28
docs/research/kbound/scripts/mixed_stream_kbound.py     items 4/15
docs/research/kbound/scripts/ablation_sweep.py          items 4/8/15
experiments/kbound/wilds/analyze_iwildcam_kbound.py     items 4/15/28
experiments/kbound/wilds/analyze_camelyon_kbound.py     items 4/15/28
experiments/kbound/ppi_micro_probe.py                   item 30
```

Artifacts regenerated as a side effect of RUNNING the fixed scripts (flagged for the author):
`docs/research/kbound/experiments/kbound/results/ablation_exactrank.json` (was a NUL placeholder,
now real), `.../ablation_all.json` (new), `docs/research/kbound/RELEASE_MANIFEST.json`,
`docs/research/kbound/reports/reproducibility_release_report.md`,
`docs/research/kbound/reports/THEORY_AUDIT_FULL.md`,
`docs/research/kbound/frontier_validation_results.json`,
`docs/research/kbound/figures/fig_frontier_{recovery,transition,fa_coverage}.png`, plus the
theory_v2 validator outputs written by step 2c of `reproduce_submission.sh`.

## Notes for the paper / library agents

1. NUMBERS_PACK §0.1 is now what the code says. `g8_exactrank_ci.py --calibration loo --unit
   condition` prints `CI-supported beats-both at unit=condition: False` for ImageNet-C on BOTH gaps.
   Do not write "with a CI excluding zero" for the freeze gap under the LOO radius.
2. The CIFAR flagship is untouched by item 4 — 0 of 9 504 decisions change, FA_u stays 0, regret
   triples bit-identical. That is a strength; say it as one.
3. PACS/VLCS and the two WILDS analyzers now use the exact-rank rule. Their published numbers were
   produced under the interpolated rule, and the committed PACS artifacts carry no `b_hat`, so PACS
   cannot be re-scored offline — it must be re-run or reported as unrecomputable.
4. `kbound_decide.fa_ceiling(n, alpha)` gives the `(n-k)/n` identity ceiling for item 5's prose.
5. `kbound_decide.selftest_radius_excludes_scored_cell()` is importable — if the library agent wants
   a `tests/` case for item 4, calling that function is a one-liner.
6. The `bn` arm is wired but has never been run. Do not put a BN-baseline number in the paper until
   it has.
