# Calibration value: completed retrospective study

Final authority is `output/next_phase/calibration_value_v1/FINAL_RESULTS_SUMMARY.json`. It combines corrected ImageNet grouping from `attempt_04_imagenet_schema_corrected` with corrected scoring for CIFAR and the checkpoint panel in `report_03`. Earlier attempts remain preserved; their ImageNet rows are superseded. This is a new retrospective analysis of already opened records, not new raw-data experiments or prospective validation.

## Main result

Calibration changes the safety–utility operating point, but a general advantage over a tune-selected fixed margin is **not established**. Scaled calibration improves the prespecified harm-weight-5 loss on CIFAR Tent; it loses to simple margins on CIFAR EATA/SAR and to point or margin policies in the other cohorts. Crucially, nominal 90% intervals transfer very poorly to held-out CIFAR corruption types. These results strengthen the limitation, not a 90% new-environment coverage claim.

## Prespecified comparison

All arms share one GBRT predictor within each fold. Whole semantic groups stay out of fitting, tuning and calibration when scored. CIFAR/ImageNet hold out complete corruption types. Checkpoint evaluation holds out the entire source model and rotates held-out condition groups; joint evaluation excludes both scored model and corruption. Fixed margins minimize tune-only loss at harm weights 1/5/20. Calibration uses exact rank at alpha .10, .05 and .20. The conditional-risk arm tests a fixed threshold family with Bonferroni-adjusted one-sided binomial bounds. Its population interpretation would require independent, identically distributed group trials with a common conditional error probability; independence alone is insufficient, and these assumptions are not established here.

The primary loss below is mean positive benefit forgone plus five times mean accepted negative-benefit magnitude. Values are percentage points; lower is better. This loss grid was frozen before new fitting.

| Cohort / outer hold-out | Cells | Point | Tuned margin (harm 5) | Constant 90% | Scaled 90% |
|---|---:|---:|---:|---:|---:|
| cifar10c/eata / environment | 2160 | 0.1077 | 0.0663 | 0.1556 | 0.0914 |
| cifar10c/sar / environment | 2160 | 0.1677 | 0.0465 | 0.1291 | 0.0844 |
| cifar10c/tent / environment | 2160 | 0.2867 | 0.2264 | 0.2150 | 0.1429 |
| imagenetc/eata / environment | 135 | 0.0481 | 0.4443 | 0.6939 | 1.8787 |
| imagenetc/sar / environment | 135 | 0.0000 | 0.2115 | 2.5989 | 1.8754 |
| imagenetc/tent / environment | 135 | 2.1913 | 1.3756 | 1.4465 | 1.4465 |
| mixed_checkpoints/tent / checkpoint | 108 | 0.3236 | 0.4032 | 0.7954 | 5.2708 |
| mixed_checkpoints/tent / joint | 108 | 0.5829 | 0.4611 | 0.9403 | 2.6329 |

For CIFAR Tent, scaled calibration reduces loss from 0.2867 to 0.1429 percentage points versus point prediction, and from 0.2264 versus the tuned margin. This is one retrospective cohort, not an independently replicated deployment effect. At equal accepted exposure, its advantage over point ranking is much smaller: 0.1429 versus 0.1610 percentage points.

## Calibration transfer and insufficient information

| Cohort / outer hold-out | Constant-cell interval inclusion | Scaled-cell inclusion | Cell inclusion under group-max intervals |
|---|---:|---:|---:|
| cifar10c/eata / environment | 54.35% | 51.67% | 61.90% |
| cifar10c/sar / environment | 50.51% | 48.19% | 61.30% |
| cifar10c/tent / environment | 60.69% | 58.24% | 73.19% |
| imagenetc/eata / environment | 91.11% | 92.59% | unavailable / unbounded |
| imagenetc/sar / environment | 97.04% | 93.33% | unavailable / unbounded |
| imagenetc/tent / environment | 93.33% | 96.30% | unavailable / unbounded |
| mixed_checkpoints/tent / checkpoint | 85.19% | 90.74% | unavailable / unbounded |
| mixed_checkpoints/tent / joint | 79.63% | 87.96% | unavailable / unbounded |

All percentages in this table measure cell inclusion, including the group-max interval column; they do not measure simultaneous group coverage. Finite interval inclusion is an observed diagnostic, not a population guarantee. CIFAR constant-cell inclusion is 60.69% Tent, 54.35% EATA and 50.51% SAR, far below nominal 90%; cell inclusion under group-max intervals also remains below nominal. Every independent-environment calibration arm abstains: calibration environments reused in fit/tune are excluded, leaving zero independently held-out calibration environments. The data cannot supply the minimum nine such environments for a finite 90% exact-rank radius.

The 10% conditional group-risk arm also abstains in every cohort. With 13 candidate thresholds and within-fold confidence error .05, even zero observed harmful groups needs 53 adapted calibration groups for its one-sided upper bound to reach .10. The largest calibration partition has only 45 groups. This is a precomputable sample-size limitation, not empirical proof that no safe policy exists. The 20% risk sensitivity accepts some CIFAR updates and remains vacuous for the small ImageNet/checkpoint calibration partitions.

Constant-radius gates select exactly the same top-scored cells as point prediction at equal exposure. All 513 original fold/alpha/constant-radius checks pass, with 81 more checks for the corrected ImageNet rerun. They cannot improve ranking; any benefit concerns an operating point. Locally scaled radii can change ranking, but do not improve it consistently.

## Audit trail and costs

- 32 allowlisted inputs bound by SHA-256. Compact CIFAR/ImageNet hashes also match the existing compact-source manifest; this does not recover missing original execution evidence. No historical raw file or sealed release was changed.
- Main execution: 57 outer folds, 20 arms, 142,020 scored fold-cell/arm decisions. The 108 checkpoint rows appear under two labeled designs; this reuse does not add independent observations. A deterministic ImageNet schema bug fix reran its nine folds under identical statistical choices.
- All fold decisions were generated before the scoring phase. Full outcome perturbation tests leave fitted model hashes, radii, thresholds and decision bytes unchanged for the held-out fold.
- 14 task tests plus 109 relevant existing tests pass (123 total). Ruff formatting and lint pass on the three owned Python source files. Independent scalar recomputation passes 17,297 original/corrected-report comparisons plus 3,204 corrected-ImageNet comparisons.
- Main decision execution took 9.314 seconds on this host: 2.057 seconds predictor fitting, 6.321 seconds grouped OOF/scale fitting, 0.061 seconds tuning/calibration and 0.129 seconds decision generation. Original scoring took 1.143 seconds. These are saved-feature replay costs; neural adaptation, image inference, raw-data I/O, GPU memory and production calibration maintenance were not measured here.
- Bootstrap results use 1,000 paired resamples of entire corruption types (six CIFAR, three ImageNet) or three source checkpoints. These low-N, related historical units support descriptive sensitivity, not strong deployment-level confidence claims.

## Corrections retained rather than hidden

1. Attempt 01 stopped before scoring because a NumPy integer risk count was not strict-JSON serializable. A targeted failing test reproduced it; scalar conversion fixed it without changing the design.
2. Attempt 02 completed all decisions/scoring. Parent review found that the original false-FREEZE counter included ABSTAIN. `report_03` rescored identical decision bytes: committed FREEZE with B >= 0 is now distinct from forgone helpful non-ADAPT, negative-benefit ADAPT and zero-benefit ADAPT. Utility, coverage and decisions did not change. Original executed sources and the hash-matching scoring source are retained.
3. The source schema uses a terminal repeat token for CIFAR, but ImageNet ends in composition. The first loader removed the last token indiscriminately, producing a coarser ImageNet grouping. This did not leak outcomes, but deviated from intended grouping. A regression test now requires removal only of a literal r-plus-digits tag; ImageNet alone was rerun after recording the bug fix, using the unchanged protocol settings. These post-result deterministic corrections do not make the analysis prospective.

## Remaining scientific limitations and deferred comparisons

B6 remains descriptive: this analysis neither repairs clipped historical ranks nor invents missing raw execution evidence. Three model names in the recovered checkpoint panel are not three newly executed independent deployments. New natural-shift validation, source/checkpoint authentication and full deployment costs belong to separate studies. Native TTALine/AETTA reruns, delayed-label ACI, a Mondrian/groupwise sweep, Holm significance claims and independent deployment replication are not implemented in this calibration task. No theoretical innovation is claimed for GBRT, split conformal, local residual scaling, binomial risk testing or these baselines.

## Reproduction

The internal evidence archive also preserves all 32 protocol-allowlisted original compact inputs under `calibration_value_v1/original_inputs/` (7,203,880 bytes), with exact original hashes in `input_copy_receipt.json`. This postrun byte copy enables inspection without the original absolute source directory; it does not recover missing raw producer execution. For a new reproduction, point a copied protocol's `source_root` to that preserved input directory and use a fresh output path. Such a run receives a new protocol/source identity. The archived executed-source snapshots remain the authorities for the original runs; do not relax their code-hash checks to replay them under edited source.

Use `/Users/pratik_n/.cache/kbound-release-verify.MsziU5/venv/bin/python`. The current corrected runner can reproduce the full intended design in a fresh directory:

```sh
python docs/research/kbound/scripts/run_calibration_value.py run --output output/next_phase/calibration_value_reproduction
```

The actual corrected ImageNet-only command is recorded in its runtime receipt. Protocol, per-fold receipts, per-group metrics, all alpha/cost sensitivities, exposure-matched comparisons and explicit unavailable arms are saved beside each authority. The original executed source snapshots are required to reproduce historical coarse-group attempt 02 exactly.

## Held-out semantic-group risk correction

Independent review identified that the initial report displayed calibration-group risk thresholds but omitted the matching held-out semantic-group estimand. The immutable supplement `output/next_phase/calibration_value_v1/group_risk_report_05/heldout_semantic_group_risk.json` now reports every one of the 160 cohort/arm combinations, plus 1,140 per-fold/arm rows. A unit is `(fold_id, semantic_group)`: any ADAPT exposes the group; any accepted cell with B <= 0 makes it an error group. Conditional risk divides errors by adapted groups and is null with zero exposure; unconditional risk divides by all scored groups. Strictly negative and zero-only error groups are distinguished. Utility and harm-weighted loss are averaged within each group and then equally across groups, separately from equal-cell metrics.

For the 20% risk-selection arm on CIFAR, the held-out conditional group error rates are EATA **8/107 = 7.48%**, SAR **19/180 = 10.56%**, and Tent **0/65 = 0%**, versus cell-level rates of **1.08%, 1.59%, and 0%**. Each cohort contains 216 total held-out semantic groups. These descriptive rates do not establish independent-group assumptions or population risk control. The 10% arm has zero exposed groups, so its conditional rate is **undefined**, not zero. No model, threshold, calibration radius, decision, prior metric, or canonical result authority was changed. The supplement receipt binds source/output/code SHA-256 values and preserves its scoring source. Independent recomputation passes 1,600 comparisons; 16 task tests plus 109 existing tests pass, and Ruff lint/format checks pass.
