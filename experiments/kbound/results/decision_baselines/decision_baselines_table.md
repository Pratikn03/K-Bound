# Decision baselines vs KGA (same logged conditions, same metrics)

## tent  (oracle acc 0.454, harmful base rate 8%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| always_freeze | 0.410 | 0.0440 | — | 0.00 | 0.095 | 1.00 |
| atc_conf | 0.441 | 0.0129 | 0.11 | 1.00 | 0.095 | 1.00 |
| atc_conf_loo | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| ent_progress | 0.441 | 0.0129 | 0.11 | 1.00 | 0.095 | 1.00 |
| ent_progress_loo | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| eata_filter_loo | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| drift_gate_loo | 0.453 | 0.0005 | 0.09 | 1.00 | 0.150 | 1.00 |
| gbm_committal | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| best_single_hindsight | 0.453 | 0.0003 | 0.08 | 1.00 | 0.150 | 1.00 |
| KGA | 0.449 | 0.0042 | 0.00 | 0.00 | 0.150 | 0.72 |

## eata  (oracle acc 0.444, harmful base rate 3%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.444 | 0.0001 | 0.03 | 1.00 | 0.162 | 1.00 |
| always_freeze | 0.410 | 0.0349 | — | 0.00 | 0.095 | 1.00 |
| atc_conf | 0.413 | 0.0315 | 0.12 | 1.00 | 0.095 | 1.00 |
| atc_conf_loo | 0.444 | 0.0001 | 0.03 | 1.00 | 0.162 | 1.00 |
| ent_progress | 0.413 | 0.0311 | 0.11 | 1.00 | 0.095 | 1.00 |
| ent_progress_loo | 0.444 | 0.0001 | 0.03 | 1.00 | 0.162 | 1.00 |
| eata_filter_loo | 0.444 | 0.0001 | 0.03 | 1.00 | 0.162 | 1.00 |
| drift_gate_loo | 0.442 | 0.0020 | 0.03 | 1.00 | 0.095 | 1.00 |
| gbm_committal | 0.444 | 0.0001 | 0.03 | 1.00 | 0.162 | 1.00 |
| best_single_hindsight | 0.444 | 0.0000 | 0.00 | 0.00 | 0.162 | 1.00 |
| KGA | 0.444 | 0.0003 | 0.00 | 0.00 | 0.162 | 0.94 |

## sar  (oracle acc 0.437, harmful base rate 31%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.377 | 0.0606 | 0.31 | 1.00 | 0.162 | 1.00 |
| always_freeze | 0.410 | 0.0277 | — | 0.00 | 0.095 | 1.00 |
| atc_conf | 0.410 | 0.0274 | 0.00 | 0.00 | 0.095 | 1.00 |
| atc_conf_loo | 0.419 | 0.0178 | 0.14 | 0.27 | 0.095 | 1.00 |
| ent_progress | 0.409 | 0.0285 | 0.50 | 0.09 | 0.095 | 1.00 |
| ent_progress_loo | 0.390 | 0.0470 | 0.75 | 0.27 | 0.095 | 1.00 |
| eata_filter_loo | 0.400 | 0.0370 | 1.00 | 0.09 | 0.095 | 1.00 |
| drift_gate_loo | 0.407 | 0.0306 | 0.50 | 0.18 | 0.095 | 1.00 |
| gbm_committal | 0.437 | 0.0000 | 0.00 | 0.00 | 0.162 | 1.00 |
| best_single_hindsight | 0.437 | 0.0000 | 0.00 | 0.00 | 0.162 | 1.00 |
| KGA | 0.429 | 0.0086 | 0.00 | 0.00 | 0.095 | 0.61 |
