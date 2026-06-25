# K-Bound Edge -- Physical Camera Validation Report

- **Protocol:** `edge_real_phone_v1`
- **Model Version:** `70fa847746b50885`
- **Config Hash:** `0c43e8aeea06cce8`
- **Epsilon:** `0.0000`
- **Alpha:** `0.10`

## 1. Anti-Leakage Audit Results

| Check | Status | Observed |
|---|---|---|
| Frozen checkpoint unchanged after candidate adaptation? | **PASS** | `70fa847746b50885` |
| Held-out labels inaccessible to the live runtime? | **PASS** | `0 forbidden keys` |
| Feature schema unchanged after protocol lock? | **PASS** | `['entropy_drop', 'frac_highconf', 'marginal_KL', 'mean_js_div', 'pbal_drop', 'post_conf', 'post_entropy', 'post_pbal', 'post_top2_margin', 'pre_conf', 'pre_entropy', 'pre_pbal', 'pred_flip_rate', 'update_norm']` |
| Epsilon calibrated only from the conformal split? | **PASS** | `fit=['S03', 'S04'], conformal=['S05', 'S06']` |
| Held-out sessions excluded from adapter, feature, and threshold tuning? | **PASS** | `0 overlaps: []` |
| Identical held-out stream replayed for every policy? | **PASS** | `observed window counts: [256, 256, 256, 256, 256, 256]` |
| Config hash stored in every log row? | **PASS** | `0 mismatches` |
| Model hash stored in every log row? | **PASS** | `0 mismatches` |

## 2. Held-Out Replay Results (Phone A)

```
policy            mean_regret  false_adapt_uncond  false_adapt_cond  adapt_rate  freeze_rate  abstain_rate
----------------------------------------------------------------------------------------------------------
always_freeze          0.0000              0.0000            0.0000       0.000        1.000         0.000
always_adapt           0.0000              0.0000            0.0000       1.000        0.000         0.000
confidence_gate        0.0000              0.0000            0.0000       0.000        1.000         0.000
entropy_gate           0.0000              0.0000            0.0000       0.102        0.898         0.000
kga_no_radius          0.0000              0.0000            0.0000       0.000        0.000         1.000
kga_full               0.0000              0.0000            0.0000       0.000        0.000         1.000
```

## 3. External-Device Replication Results (Phone B)

```
policy            mean_regret  false_adapt_uncond  false_adapt_cond  adapt_rate  freeze_rate  abstain_rate
----------------------------------------------------------------------------------------------------------
always_freeze          0.0000              0.0000            0.0000       0.000        1.000         0.000
always_adapt           0.0000              0.0000            0.0000       1.000        0.000         0.000
confidence_gate        0.0000              0.0000            0.0000       0.000        1.000         0.000
entropy_gate           0.0000              0.0000            0.0000       0.160        0.840         0.000
kga_no_radius          0.0000              0.0000            0.0000       0.000        0.000         1.000
kga_full               0.0000              0.0000            0.0000       0.000        0.000         1.000
```

## 4. Resource and Live-Runtime Profile

| Component | Mean (ms) | p95 (ms) |
|---|---|---|
| frozen_inference | 319.91 | 360.93 |
| tent_update | 1091.40 | 1103.60 |
| candidate_inference | 286.74 | 287.54 |
| evidence | 0.13 | 0.15 |
| gate | 3.69 | 8.01 |
| end_to_end | 1715.12 | 1761.40 |
| capture_preprocess | 13.25 | 13.57 |

## 5. Conformal Gate Ablation Results

| Variant | Regret | FA_u | Adapt Rate | Abstain Rate | Epsilon |
|---|---|---|---|---|---|
| full_kga | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |
| no_radius | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |
| no_blur_brightness | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |
| no_disagreement | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |
| confidence_only | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |
| entropy_only | 0.0000 | 0.0000 | 0.000 | 1.000 | 0.0000 |

