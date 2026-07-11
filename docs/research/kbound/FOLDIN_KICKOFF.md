# Fold-in scoring pass — kickoff (paste into a fresh session)

**Goal.** Score every K-Bound v5 dataset **once** against the frozen pre-registered bars using **one uniform CI-robust verdict rule**, fold the verdicts into the paper, regenerate both PDFs. No new training. No re-rolling — score once and take what it gives.

## Inputs (result JSONs, already on disk — no raw data needed)
Repo: `/Volumes/T9/uav/AutoML_Flagship_V8`, results under `experiments/kbound/results/`
- CIFAR-10-C — `win_hunt_v5/cifar10c_aggr/seed{0..4}/decisive_tta_results.json`
- CIFAR-10.1 — `win_hunt_v5/cifar101_aggr/seed{0..4}/decisive_tta_results.json`
- RxRx1 — `win_hunt_v5/rxrx1_aggr/result_4a2840ef.json`
- Camelyon17 (aggressive) — `wilds_kbound/result_8d3c0c41.json`
- ImageNet-R — `imagenetr_kbound_debug_mps/result_75ee8322.json`
- Office-Home — `officehome_kbound_run/result_target_test_d2f4bf2c.json`
- iWildCam — `win_hunt_v5_iwildcam/result_0ba633eb.json`
- PACS — `win_hunt_v5/pacs_aggr/pacs_result.json`
- ImageNet-C — `win_hunt_v5/imagenetc_aggr/decisive_tta_results.json` (full-scale, 3 noise corruptions, seed 0, completed 2026-07-09). Point results: **SAR beats-both** (KGA regret 0.0108 vs adapt 0.0625, freeze 0.0319, harmful 0.148); tent no-harm (ties freeze); eata no-harm (helpful-dominated). Score for CI-robustness like the rest.

## Scorer
Extend `docs/research/kbound/scripts/uniform_scorer.py` (point-tier already built) to the **CI tier**:
per-cell `(acc_freeze, acc_adapt, acc_oracle, acc_kga)` extraction into `recs[]`, 95% **paired-bootstrap** CI on the regret gap, **Holm** correction across the wave. Adapters already exist for the `cifar` / `wilds` / `pacs` schema families.

## Frozen verdict rule (α = 0.10 — identical for every dataset)
- **BEATS-BOTH** = `regret_kga < regret_adapt` AND `< regret_freeze` AND the 95% CI on the gap to the **better** fixed policy excludes 0 AND clean held-out.
- **NO-HARM** = ties the better policy (gap-CI includes 0) + beats the worse + `FA_u ≤ α`.
- **FAIL** = `FA_u > α`.  **NULL** = anything else.
Point-beats-both is necessary, not sufficient — the CI is the discriminator.

## Known point-regrets (cross-check only; the CI decides the label)
| dataset | kga | adapt | freeze | expected under CI rule |
|---|--:|--:|--:|---|
| CIFAR-10-C (tent/eata) | — | — | — | BEATS-BOTH (CI-robust) |
| Mixed pooled stream | 0.0059 | 0.0632 | 0.0342 | BEATS-BOTH (large gap) |
| Camelyon17 aggressive | mixed, AUC 0.855 | — | — | test at fold-in (~30–45%) |
| Office-Home | 0.0022 | 0.0468 | 0.0158 | NO-HARM (small-n CI) |
| iWildCam | 0.0037 | 0.1028 | 0.0041 | NO-HARM (gap to freeze ≈ 0) |
| ImageNet-R | 0.0031 | 0.0105 | 0.0037 | NO-HARM (CI to freeze fails) |
| RxRx1 | 0.0000 | 0.2531 | 0.0000 | NO-HARM (ties freeze) |
| Camelyon17 G | — | — | — | WITHDRAWN (contaminated held-out — do NOT reinstate) |

## Also fold in
- WIN_HUNT_v4 arms **D** (per-sample), **E** (seeds 5–9 + pooled CIs), **F** (composite).
- Reviewer analyses: **benign-vs-harsh paired table**, per-dataset FA_u CIs, per-method decomposition, abstention columns.

## Output
Uniform verdict table → regenerate `paper/generated/kbound_numbers` via `scripts/make_tables.py` → recompile `kbound_short.tex` and `kbound.tex` (`pdflatex` ×2 each; `grep -c '??'` must be 0).

## Integrity guardrails
Score once. Report every arm (wins, no-harm, NULL, NOT_RUN). Do **not** reinstate the withdrawn Camelyon-G beats-both (it pooled in-distribution `id_val` into the held-out set). β is declared, not estimated; ε is the conformal radius, not β.

## Already done this session (don't redo)
Lean count 58→43 + softened "first machine-checked"; β operational note; long-paper descriptive captions; synthetic frontier validation (`scripts/frontier_validation.py` + 3 figures + subsection). Backup `research_lock/` + `results/` off the T9 drive before heavy work — it has been unstable.
