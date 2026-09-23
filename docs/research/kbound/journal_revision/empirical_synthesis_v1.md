# Empirical synthesis from frozen evidence

This versioned analysis recomputes preserved results without refitting, training, opening targets, or replacing historical authorities. All new uncertainty and influence analyses are retrospective and descriptive.

## Shared-predictor comparisons

Values are **100L5**, percentage points of harm-weight-5 decision loss, lower is better. They are not accuracy or F1.

| Cohort / hold-out | Cells | Clusters | Point | Margin | Constant 90% | Scaled 90% |
|---|---:|---:|---:|---:|---:|---:|
| cifar10c/eata / environment | 2160 | 6 | 0.1077 | 0.0663 | 0.1556 | 0.0914 |
| cifar10c/sar / environment | 2160 | 6 | 0.1677 | 0.0465 | 0.1291 | 0.0844 |
| cifar10c/tent / environment | 2160 | 6 | 0.2867 | 0.2264 | 0.2150 | 0.1429 |
| imagenetc/eata / environment | 135 | 3 | 0.0481 | 0.4443 | 0.6939 | 1.8787 |
| imagenetc/sar / environment | 135 | 3 | 0.0000 | 0.2115 | 2.5989 | 1.8754 |
| imagenetc/tent / environment | 135 | 3 | 2.1913 | 1.3756 | 1.4465 | 1.4465 |
| mixed_checkpoints/tent / checkpoint | 108 | 3 | 0.3236 | 0.4032 | 0.7954 | 5.2708 |
| mixed_checkpoints/tent / joint | 108 | 3 | 0.5829 | 0.4611 | 0.9403 | 2.6329 |

The eight rows are cohort/design summaries, not eight independent deployments. CIFAR has six corruption clusters; ImageNet and the checkpoint designs each have three clusters. The 108 checkpoint records recur in the checkpoint and joint designs.

## Coverage and the unit of inclusion

| CIFAR method | Constant-cell inclusion | Scaled-cell inclusion | Group-max cell inclusion | Group-max simultaneous-group inclusion |
|---|---:|---:|---:|---:|
| eata | 54.35% | 51.67% | 61.90% | 56.94% |
| sar | 50.51% | 48.19% | 61.30% | 45.83% |
| tent | 60.69% | 58.24% | 73.19% | 61.57% |

All intervals above were nominally 90%. These observed rates do not support nominal transfer coverage. The independent-environment arms have no eligible calibration environments. Zero exposure in the 10% risk arm leaves conditional error undefined.

## Tent sensitivity

- Versus point: scaled-minus-comparator loss -0.143843 pp; descriptive 95% bootstrap range [-0.428565, 0.138755] pp. Omitting JPEG for sensitivity only gives -0.004889 pp.
- Versus fixed_margin_harm5: scaled-minus-comparator loss -0.083495 pp; descriptive 95% bootstrap range [-0.416991, 0.188282] pp. Omitting JPEG for sensitivity only gives 0.090889 pp.
- Versus point at matched exposure: scaled-minus-comparator loss -0.018148 pp; descriptive 95% bootstrap range [-0.042500, -0.001065] pp. Omitting JPEG for sensitivity only gives -0.006528 pp.

All-cluster estimates remain primary. These low-N, related historical clusters do not establish population significance. Equal-exposure differences isolate ranking from acceptance rate; the much larger comparison against the unthresholded point arm is not an accuracy improvement.

## Principal CIFAR policies

| Candidate | Frozen accuracy | Adapt accuracy | KGA accuracy | KGA minus adapt | KGA regret |
|---|---:|---:|---:|---:|---:|
| cifar10c/eata | 67.0931% | 79.8992% | 80.0666% | 0.1673 pp | 0.1603 pp |
| cifar10c/sar | 67.0931% | 81.1067% | 80.9741% | -0.1326 pp | 0.1661 pp |
| cifar10c/tent | 67.0931% | 78.6893% | 79.3076% | 0.6183 pp | 0.1793 pp |

These are means over recorded condition/seed cells. Their aggregation weights differ from image-pooled accuracy or deployment frequency. SAR remains unfavorable to KGA versus always-adapt.

## CCT-20 retention and routing

Across 45 checkpoint/location cells (five checkpoints, nine locations), equal-cell set-membership top-1 accuracy is 52.2936% frozen, 34.1034% always-adapt and 52.2936% KGA. KGA makes 0 ADAPT, 44 FREEZE and 1 ABSTAIN decisions. It misses 1/1 helpful opportunities; conditional acceptance error is undefined at zero exposure.

The locked retention endpoint is met; stronger routing success is not. Published cross-classified interval and location-test results are preserved separately in JSON. Archived 16-indicator cell-level macro-F1 is checked from class counts and retained as secondary descriptive evidence. Aggregate macro-F1 and an aggregate inference claim remain null.

## Reproduction and boundaries

Extract the released next_phase_evidence.zip at the repository root first; use the pinned research environment and run:

```sh
python docs/research/kbound/scripts/build_empirical_synthesis.py
python docs/research/kbound/scripts/build_empirical_synthesis.py --check
```

The generated input manifest binds every consulted file and CCT bundle member by SHA-256. Inputs are read only. JSON contains all 160 calibration arm rows, all coverage denominators, matched exposure, paired cluster effects, leave-one-cluster-out diagnostics, nine principal CIFAR policy rows and three CCT policy rows. CSV is a flat summary; JSON is authoritative for nested uncertainty and unavailable-value reasons.

Historical producer-provenance gaps, calibration transfer failures and lack of independent deployment evidence remain. No new protected target was opened and no confirmatory result was manufactured.
