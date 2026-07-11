# K-Bound Result Audit — Reconciliation + Leakage (Phase 4)

Every number below was checked against the actual source files in this checkout
(`results_source.json`, `pacs_result.json`, `gate_comparison.json`, the `experiments/kbound/results/`
artifacts, and `research_lock/`), not against a prior draft. Values I changed in the manuscript are
marked **FIXED**; values with no traceable in-repo artifact are marked **UNSOURCED → flagged** (kept
in the paper with a visible DRAFT TODO, never deleted).

## A. Headline numbers that reconcile cleanly (no change)

- **ImageNet-C SAR table** (`tab:imagenetc-faithful`, `tab:primary-numeric`): regrets
  `0.0108 / 0.0625 / 0.0319`, 27 cells, 14.8% harmful, FA_u=0, 12/27 abstain — **match
  `experiments/kbound/results/win_hunt_v5/imagenetc_aggr/decisive_tta_results.json` byte-for-byte**;
  the gap-freeze CI `[-0.032,-0.011]` and Holm p match `results_source.json`.
- **Decision-gate comparison** (`tab:gates`): all 36 cells match `gate_comparison.json` exactly
  (n=432, 149 harmful; KGA-certificate regret 0.0017, FA_u 0; no-radius 0.0004, FA_u 0.049→0.141 on
  harmful; coverage 0.898).
- **CIFAR-10-C stress regrets** (`tab:decisive`, `tab:uniform-panel`): Tent `0.0016/0.0079/0.1241`,
  EATA `0.0013/0.0033/0.1314`, FA_u=0 — match `results_source.json` locked_analysis.
- **Synthetic frontier** (§Synthetic): 90.0% coverage, 6.5%→96.7% commit, 99.3% sign-correct,
  FA_u≤0.014 — match `frontier_validation_results.json`.
- Camelyon17 OOD (n=18, 0/0/0.1381), RxRx1 (0/0.2531/0), iWildCam (0.0041/0.1028/0.0041),
  Office-Home (0.0157/0.0468/0.0158) — match `results_source.json` / `recon_results.json`.

## B. Mismatches corrected in the manuscript (**FIXED**)

| # | Location | Was | Now | Source of truth |
|---|----------|-----|-----|-----------------|
| 1 | §CIFAR-10-C prose | SAR harmful **16%** | **≈10%** (mean over seeds) | `results_source.json`: SAR `harmful_frac` per seed = 0.116/0.102/0.088/0.074/0.100 → mean 0.096 |
| 2 | §Mixed head-to-head prose | harmful fraction **≈16–18%** | **≈33%** (= always-adapt FA_u of `tab:headtohead-poem-aetta`) | table's own `always-adapt FA_u=0.329`; Tent full-grid harmful 34% |
| 3 | Uniform panel, PACS row | **18** cells each; FA_u=0 | **108** cells each; three safety targets FA_u=0, one null (**photo, FA_u=0.056**) | `pacs_result.json`: `n_test_cells:108` ×4; photo verdict NULL, FA_u 0.0556 |

These are integrity corrections (they *reduce* or qualify claims); none strengthens a result.

## C. Numbers with no in-repo artifact (**UNSOURCED → flagged**, not deleted)

The raw results tree (`experiments/kbound/results/**`) and several `research_lock/*.json` were on the
external T9 drive; only the ImageNet-C run was copied into the repo. As a result the following
manuscript numbers trace only to prose/curated files, not to independent raw artifacts. Each now
carries a visible DRAFT TODO in the draft PDF:

1. **Three-source KGA leg `0.0059`** (`tab:primary-numeric`, verdict-migration fig): only the
   *withdrawn in-sample* `0.0026` exists on disk (`…/camelyon_reconciliation/mixed_stream_recompute.json`);
   `0.0059` appears only in `FOLDIN_KICKOFF.md`. The adapt/freeze legs `0.0632/0.0342` **do** match.
   → row re-labeled "constructed mixture; OOF replay pending".
2. **Streaming demo** (`35,370` images, `45` windows, `75.6%` harmful, gate `0.6995`, §Extensions):
   no result JSON under `experiments/kbound/results/`; only the producing script
   `gapclose_wave5/win_hunt_D_anytime_stream.py` exists. → specific figures removed from the headline;
   qualitative anytime-FA=0 claim retained with a replay TODO.
3. **CIFAR-10.1 regret triple `0.0021/0.0190/0.0017`** (uniform panel, diagnostic row): only
   `FA_u=0.444` is traceable (`result_manifest.json`). Row is already "no claim"; tier changed to
   "diagnostic; replay pending".

## D. Provenance gaps (cited paths absent from the checkout)

The paper `\path{}`-cites artifacts that are not in this repo (were on T9):
`stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`,
`mixed_headtohead_v1/HEADTOHEAD_RESULTS_*.json`,
`research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`, `research_lock/WIN_HUNT_v3_ARM_F_result.json`,
`WIN_HUNT_v2/v3_PROTOCOL.yaml`, `mixed_protocol_oof_v2`. `research_lock/` currently holds only
`WIN_HUNT_v5_PROTOCOL_SHELL.yaml`. The confirmed CIFAR-10-C, head-to-head, and natural-shift regrets
therefore trace to the curated `results_source.json` "single source of truth" rather than to
independent raw records. **Action (TODO):** re-copy these artifacts from backup, or remove the
dangling citations. This is a provenance gap, not a method error.

## E. Leakage audit

| Pattern | Verdict | Evidence |
|---------|---------|----------|
| In-sample residual calibration | **CLEAN** | radius is genuine leave-one-out/out-of-fold: `analyze_F.py:172-179` ("in-sample radius was ~10× too small"), `score_kbound_holdout.py:75-90`, `run_protocol_dev_lock.py:75-80` |
| Target-test labels in selection/fit/radius | **CLEAN** | `B_hat` fit on cal only; adapter picked on dev only; evidence panel pre-declared |
| id_val pooled into OOD (Camelyon-G) | **CONCERN, headline withdrawn** | withdrawn in prose (`kbound_short.tex` Camelyon-G block) + audit `…/camelyon_reconciliation/VERDICT_phase1.md`; residuals: domain-blind seed-split scorer, `bootstrap_win_cis.py` still hard-codes pooled n=54, `WIN_HUNT_v5_PROTOCOL_SHELL.yaml:97` re-lists id_val |
| Best candidate by test regret | **CLEAN** | `run_protocol_dev_lock.pick_adapter` ranks on dev only, STOPs if no dev candidate clears FA≤α |
| Repeated test evaluation | **CLEAN (disclosed)** | score-once enforced; re-scorings are disclosed corrections, prior numbers withdrawn not retained |
| Pooled incompatible units | **CONCERN (disclosed)** | 7-source / v5 "universal gate" mixes iWildCam macro-F1 with top-1 accuracy; disclosed as constructed, per-dataset gates |

Both previously-claimed fixes (Camelyon id_val withdrawal; in-sample→OOF radius) are **confirmed in
code and in `results_source.json`**; the only gap is the missing `_oof` locked JSON (see §D).

## F. Net effect on claims
The three CI-confirmed beats-both results (CIFAR-10-C Tent/EATA stress, ImageNet-C SAR, decision-gate
comparison) survive unchanged and are the most defensible. The natural-shift story remains honest
no-harm. The corrections tighten three overstated figures and downgrade three unsourced ones to
flagged/provisional — moving the paper strictly toward defensibility.
