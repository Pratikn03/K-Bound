# ImageNet-R / RxRx1 Hard-Dataset Win Loop

CPU-only audit over existing record artifacts. These rows rerun the locked K-Bound decision evaluator on fixed seed splits; no model training is performed.

- Stability groups with replicated wins: 0
- Stability groups with at least one split/modelseed win: 2
- Interpretation rule: split-only/modelseed-only wins are leads, not paper headline claims, until they replicate across the planned stability checks.

## Stability Summary

| group | rows | wins | win rate | replicated? | margin range |
|---|---:|---:|---:|:---:|---:|
| `imagenetr_light_tent_gbr_cqr` | 4/4 | 1 | 0.25 |  | -0.00875..0.00104167 |
| `rxrx1_tent_gbr_mondrian` | 4/4 | 1 | 0.25 |  | 0..0.000325521 |

## Per-Run Results

| dataset | run | win? | candidate | est | conf | split | margin | KGA | adapt | freeze | FA | cov | n |
|---|---|:---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| imagenet-r | `imagenetr_light_split01_23` | YES | `tent_online` | gbr | cqr | `[0, 1]->[2, 3]` | 0.00104167 | 0.00416667 | 0.00520833 | 0.0180208 | 0.0714 | 0.583 | 24 |
| imagenet-r | `imagenetr_light_split12_03` |  | `tent_online` | gbr | cqr | `[1, 2]->[0, 3]` | -0.0053125 | 0.00989583 | 0.00458333 | 0.0159375 | 0.231 | 0.583 | 24 |
| imagenet-r | `imagenetr_light_split02_13` |  | `tent_online` | gbr | cqr | `[0, 2]->[1, 3]` | -0.00791667 | 0.0111458 | 0.00322917 | 0.0215625 | 0 | 0.375 | 24 |
| imagenet-r | `imagenetr_light_split03_12` |  | `tent_online` | gbr | cqr | `[0, 3]->[1, 2]` | -0.00875 | 0.0107292 | 0.00197917 | 0.0252083 | 0 | 0.625 | 24 |
| rxrx1 | `rxrx1_modelseed0_tent_mondrian` | YES | `tent_online` | gbr | mondrian | `[0, 1, 2, 3, 4]->[5, 6, 7, 8, 9]` | 0.000325521 | 0.000813802 | 0.0521484 | 0.00113932 | 0 | 0.783 | 60 |
| rxrx1 | `rxrx1_modelseed1_tent_mondrian` |  | `tent_online` | gbr | mondrian | `[0, 1, 2, 3, 4]->[5, 6, 7, 8, 9]` | 0 | 0.000195313 | 0.0594727 | 0.000195313 | 0 | 0.967 | 60 |
| rxrx1 | `rxrx1_modelseed2_tent_mondrian` |  | `tent_online` | gbr | mondrian | `[0, 1, 2, 3, 4]->[5, 6, 7, 8, 9]` | 0 | 0.000520833 | 0.0559896 | 0.000520833 | 0 | 0.95 | 60 |
| rxrx1 | `rxrx1_pooled_modelseeds0_2_tent_mondrian` |  | `tent_online` | gbr | mondrian | `[0, 1, 2, 3, 4]->[5, 6, 7, 8, 9]` | 0 | 0.00061849 | 0.0558702 | 0.00061849 | 0 | 0.717 | 180 |
