# Decision baselines vs KGA (same logged conditions, same metrics)

## tent  (oracle acc 0.455, harmful base rate 6%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.454 | 0.0007 | 0.06 | 1.00 | 0.164 | 1.00 |
| always_freeze | 0.405 | 0.0494 | — | 0.00 | 0.088 | 1.00 |
| atc_conf | 0.443 | 0.0114 | 0.07 | 1.00 | 0.088 | 1.00 |
| atc_conf_loo | 0.454 | 0.0007 | 0.06 | 1.00 | 0.164 | 1.00 |
| ent_progress | 0.446 | 0.0089 | 0.06 | 1.00 | 0.088 | 1.00 |
| ent_progress_loo | 0.454 | 0.0007 | 0.06 | 1.00 | 0.164 | 1.00 |
| eata_filter_loo | 0.454 | 0.0007 | 0.06 | 1.00 | 0.164 | 1.00 |
| drift_gate_loo | 0.452 | 0.0028 | 0.06 | 1.00 | 0.088 | 1.00 |
| gbm_committal | 0.455 | 0.0000 | 0.00 | 0.00 | 0.164 | 1.00 |
| best_single_hindsight | 0.454 | 0.0007 | 0.06 | 1.00 | 0.164 | 1.00 |
| KGA | 0.449 | 0.0060 | 0.00 | 0.00 | 0.164 | 0.69 |

## eata  (oracle acc 0.444, harmful base rate 3%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.444 | 0.0000 | 0.03 | 1.00 | 0.169 | 1.00 |
| always_freeze | 0.405 | 0.0386 | — | 0.00 | 0.088 | 1.00 |
| atc_conf | 0.411 | 0.0328 | 0.10 | 1.00 | 0.088 | 1.00 |
| atc_conf_loo | 0.444 | 0.0000 | 0.03 | 1.00 | 0.169 | 1.00 |
| ent_progress | 0.412 | 0.0321 | 0.09 | 1.00 | 0.088 | 1.00 |
| ent_progress_loo | 0.444 | 0.0000 | 0.03 | 1.00 | 0.169 | 1.00 |
| eata_filter_loo | 0.444 | 0.0000 | 0.03 | 1.00 | 0.169 | 1.00 |
| drift_gate_loo | 0.441 | 0.0023 | 0.03 | 1.00 | 0.088 | 1.00 |
| gbm_committal | 0.444 | 0.0000 | 0.03 | 1.00 | 0.169 | 1.00 |
| best_single_hindsight | 0.444 | 0.0000 | 0.00 | 0.00 | 0.169 | 1.00 |
| KGA | 0.439 | 0.0047 | 0.00 | 0.00 | 0.169 | 0.75 |

## sar  (oracle acc 0.432, harmful base rate 44%)

| policy | mean acc | regret | false-adapt | harmful adapted | worst case | coverage |
|---|--:|--:|--:|--:|--:|--:|
| always_adapt | 0.319 | 0.1124 | 0.44 | 1.00 | 0.061 | 1.00 |
| always_freeze | 0.405 | 0.0267 | — | 0.00 | 0.088 | 1.00 |
| atc_conf | 0.365 | 0.0673 | 1.00 | 0.25 | 0.088 | 1.00 |
| atc_conf_loo | 0.394 | 0.0375 | 1.00 | 0.06 | 0.088 | 1.00 |
| ent_progress | 0.355 | 0.0765 | 1.00 | 0.31 | 0.088 | 1.00 |
| ent_progress_loo | 0.394 | 0.0375 | 1.00 | 0.06 | 0.088 | 1.00 |
| eata_filter_loo | 0.395 | 0.0369 | 1.00 | 0.06 | 0.088 | 1.00 |
| drift_gate_loo | 0.394 | 0.0382 | 1.00 | 0.12 | 0.088 | 1.00 |
| gbm_committal | 0.431 | 0.0010 | 0.05 | 0.06 | 0.103 | 1.00 |
| best_single_hindsight | 0.414 | 0.0181 | 0.17 | 0.06 | 0.061 | 1.00 |
| KGA | 0.409 | 0.0229 | 0.00 | 0.00 | 0.088 | 0.39 |
