# Task 4 Final Closeout: Authenticated Matched SAR Control Study

**Date:** 2026-09-17  
**Branch:** `main`  
**Host / Device:** Apple Silicon arm64 (MPS available)  
**Python Environment:** `/opt/anaconda3/envs/automl311/bin/python` (Python 3.11.14, PyTorch 2.10.0, NumPy 2.4.4)  

---

## 1. Panel and Checkpoint Identity (Predeclared Prior to Execution)

- **Source Checkpoint Path:** `/Users/pratik_n/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`
- **Source Checkpoint SHA-256:** `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- **Architecture:** Torchvision ResNet-50 (`torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`)
- **Dataset Root / Version:** `/Users/pratik_n/imagenetc_local` (ImageNet-C extracted group structure, 50,000 validation images per corruption)
- **Corruption Names (3 noise corruptions):** `["gaussian_noise", "shot_noise", "impulse_noise"]`
- **Severity Levels (3 representative severities):** `[1, 3, 5]`
- **Compositions (3 stream shift regimes):** `["iid", "imbalanced", "single_class"]`
- **Batch Regime:** `small` (stream adaptation batch size $b = 16$; stream length $16 	imes 8 = 128$ images)
- **Evaluation Sample Size:** 4,000 images per (corruption, severity) condition, balanced across 1,000 classes using fixed pseudo-random sample selection with seed 0 (`rng = np.random.default_rng(0)`).
- **Candidate Scoring Timing:** Streaming adaptation over 128 images ($8 	imes 16$), candidate evaluated on the 4,000 held-out images with test-time batch statistics enabled (`train_mode=True`), matching the canonical Tent/SAR protocol.
- **Reset Semantics:** Full per-cell model reset from the frozen source checkpoint; no model state or optimizer momentum leaks between cells.
- **Number of Cells per Arm:** 27 cells ($3 \text{ corruptions} \times 3 \text{ severities} \times 3 \text{ compositions}$).
- **Total Evaluations Across 5 Arms:** $27 \times 5 = 135$ condition runs.
- **Seed:** 0.
- **Evaluation Metrics:** Candidate accuracy, frozen accuracy, candidate benefit $B = a_{\text{adapted}} - a_0$, paired oracle regret, $\mathrm{FA}_u$ (unconditional false adaptation rate), $\mathrm{FF}_u$ (unconditional false freeze rate).
- **KGA Protocol Version:** Protocol B (three-way disjoint fit / residual-calibration / score cross-fitting; $\alpha = 0.10$, exact conformal rank rule $k = \lceil(n_{\text{cal}} + 1)(1 - \alpha)\rceil$, fixed zero decision threshold $\tau = 0$).

---

## 2. Predeclared Five-Arm Matched Configuration

| Arm ID | Arm Name | Learning Rate $\eta$ | Adapted Layers | Layer 4 State | Test BN Statistics | Gradient Updates |
|:---:|---|:---:|---|:---:|:---:|:---:|
| **Arm 1** | BN-statistics only | $0.0$ | None | Frozen | Enabled (test batch stats) | None |
| **Arm 2** | Reference SAR (official) | $2.5 \times 10^{-4}$ | Layers 1–3 affine | Frozen | Enabled (upstream contract) | Active (SAM) |
| **Arm 3** | Rate only | $4.0 \times 10^{-3}$ | Layers 1–3 affine | Frozen | Enabled (upstream contract) | Active (SAM) |
| **Arm 4** | Mask only | $2.5 \times 10^{-4}$ | Layers 1–4 affine | Adapted | Enabled (upstream contract) | Active (SAM) |
| **Arm 5** | Historical combined | $4.0 \times 10^{-3}$ | Layers 1–4 affine | Adapted | Enabled (upstream contract) | Active (SAM) |

---

## 3. Pre-Execution Mechanical Verification

Commands executed:
1. `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/test_sar_faithful.py`
   - **Result:** PASSED (A1 defaults, A2 exact SAM restore drift $0.0\times 10^0$, A3 recovery restore exact, A4 finite & bounded).
2. `/opt/anaconda3/envs/automl311/bin/pytest tests/test_sar_final_block_batch_stats.py tests/test_sar_runbook_exit_status.py -v`
   - **Result:** 8 passed in 3.23s.
   - Parity with official SAR repository (`external/sar_official/sar.py`, SHA-256 `0553a395ac2bc087049720f1f162789079fe5ee25aa6a1cf82b2ea6286d66eff`) confirmed for layer-4 affine exclusion and BN statistics.

---

## 4. Two-Cell Execution Pilot Verification

- **Command:** `/opt/anaconda3/envs/automl311/bin/python -u docs/research/kbound/scripts/run_task4_matched_sar.py --spec experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_SPEC.json --output-dir experiments/kbound/results/task4_matched_sar_v1 --pilot`
- **Conditions Executed:**
  1. `gaussian_noise|s1|small|aggressive|iid` (Condition 1, 273.5s)
  2. `gaussian_noise|s1|small|aggressive|imbalanced` (Condition 2, 273.5s)
- **Total Wall-Clock Time:** 550s (9.1 min). Memory peak: 158MB RSS (no paging, MPS stable).
- **Results:** 10 records total (2 per arm $\times$ 5 arms).
- **Sample Selection & Baseline Parity:** Frozen accuracy $a_0 = 0.68325$ matches historical seed-0 baseline exactly.
- **State Isolation:** All 5 candidate model states yielded distinct SHA-256 hashes after adaptation; parameter updates ranged from $0.00$ (Arm 1 BN-only) to $0.14$ (Arm 5 aggressive).
- **Verification Checks:** Automated authentication script (`authenticate_task4_matched_sar.py --expected-cells 2`) verified all 11 checks (Checks A–K) as **PASS**.

---

## 5. Full 27-Cell $\times$ 5-Arm Matrix Execution

- **Command:** `/opt/anaconda3/envs/automl311/bin/python -u docs/research/kbound/scripts/run_task4_matched_sar.py --spec experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_SPEC.json --output-dir experiments/kbound/results/task4_matched_sar_v1 --resume`
- **Total Execution Time:** 4,636.8s (77.3 min) across all 27 panel conditions.
- **Total Evaluations:** $27 \text{ conditions} \times 5 \text{ arms} = 135$ completed evaluations.
- **Source Checkpoint:** ResNet-50 (`ResNet50_Weights.IMAGENET1K_V2`, SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`).
- **Hardware / Device:** Apple Silicon arm64 MPS (Metal Performance Shaders).
- **Raw Outputs:**
  - `arm1_bn_only.jsonl`: SHA-256 `b2238ed9c69a77ae1e8bf35439576376eb15bbdc57abdd2e66bb15c38e588651`
  - `arm2_reference_sar.jsonl`: SHA-256 `71eb43108dd7cad210f78ecc04cb7c897b65cebe3909c39784511b08ea255548`
  - `arm3_rate_only.jsonl`: SHA-256 `998edea99d8229eac902a80bd80b42930ef207ab8cb3209cdce3703a896fe2b4`
  - `arm4_mask_only.jsonl`: SHA-256 `ee4f036136c914a9e04d9c2c254b54c3cfae484de26c5a3a348652bb3433ee16`
  - `arm5_combined.jsonl`: SHA-256 `524857c5859e2101dd8196ae5d5a37b468257c365fffc2b1bbfa544fa46c0172`

---

## 6. Authentication Manifest Verification (Checks A–K)

Executed: `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/authenticate_task4_matched_sar.py --expected-cells 27`

- `check_a_record_counts`: **PASS** (exactly 27 records per arm).
- `check_b_schema`: **PASS** (all required fields present).
- `check_c_numeric_bounds`: **PASS** (no NaN or Inf).
- `check_d_frozen_accuracy_identity`: **PASS** (identical frozen accuracy across all 5 arms per cell).
- `check_e_sample_hash_identity`: **PASS** (identical sample hashes across all 5 arms per cell).
- `check_arm1_configuration`: **PASS** (lr=0, grad=False, upd_norm=0.0).
- `check_arm2_configuration`: **PASS** (lr=2.5e-4, grad=True, freeze_layer4=True).
- `check_arm3_configuration`: **PASS** (lr=4.0e-3, grad=True, freeze_layer4=True).
- `check_arm4_configuration`: **PASS** (lr=2.5e-4, grad=True, freeze_layer4=False).
- `check_arm5_configuration`: **PASS** (lr=4.0e-3, grad=True, freeze_layer4=False).
- `check_k_distinct_updates`: **PASS** (distinct state hashes across distinct configurations).
- **Overall Authentication Status:** **PASSED** (`TASK4_MATCHED_SAR_MANIFEST.json`).

---

## 7. Five-Arm Summary and Protocol B KGA Disjoint Cross-Fitting

| Arm | Configuration | Learning Rate $\eta$ | Adapted Layers | Cand Acc $a_{\mathrm{cand}}$ | Benefit $\Delta$ | Update Norm $\|\Delta\theta\|_2$ | KGA Acc $a_{\mathrm{KGA}}$ | KGA Regret $R_{\mathrm{KGA}}$ | $\mathrm{FA}_u$ | Decisions (A/F/Abs) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm 1** | BN-only (zero grad) | $0.0$ | None | 44.25% | +3.30% | 0.0000 | 44.05% | 0.20% | 0.0% | 23 / 0 / 4 |
| **Arm 2** | Reference SAR (official) | $2.5\times 10^{-4}$ | Layers 1–3 | 44.25% | +3.30% | 0.0011 | 44.11% | 0.14% | 0.0% | 24 / 0 / 3 |
| **Arm 3** | Rate only | $4.0\times 10^{-3}$ | Layers 1–3 | 44.26% | +3.30% | 0.0182 | 44.06% | 0.19% | 0.0% | 23 / 0 / 4 |
| **Arm 4** | Mask only | $2.5\times 10^{-4}$ | Layers 1–4 | 44.25% | +3.30% | 0.0015 | 44.25% | 0.00% | 0.0% | 27 / 0 / 0 |
| **Arm 5** | Historical combined | $4.0\times 10^{-3}$ | Layers 1–4 | 44.26% | +3.31% | 0.0241 | 44.26% | 0.00% | 0.0% | 27 / 0 / 0 |

---

## 8. Definitive Resolution of the Six Core Scientific Questions

1. **Learning-Rate Effect (Arm 2 vs Arm 3):**
   - Under faithful test-time BatchNorm statistics, scaling learning rate 16-fold from reference $2.5\times 10^{-4}$ to $4.0\times 10^{-3}$ (with layer 4 frozen) increases update norm from $0.0011$ to $0.0182$ but yields virtually identical accuracy ($44.26\%$ vs $44.25\%$). Model collapse does **not** occur.
2. **Mask Effect (Arm 2 vs Arm 4):**
   - Unfreezing layer 4 affine parameters at reference learning rate $2.5\times 10^{-4}$ produces identical candidate accuracy ($44.25\%$). While freezing layer 4 is the upstream official SAR convention, it does not alter adaptation accuracy when test batch statistics are enabled.
3. **Combined Effect (Arm 2 vs Arm 5):**
   - Combining aggressive learning rate $4.0\times 10^{-3}$ with full layer 1–4 affine adaptation yields update norm $0.0241$ (highest among all arms) with candidate accuracy $44.26\%$. Crucially, severe collapse does **not** occur under faithful test-time normalization.
4. **BN-Statistics-Only Task Accuracy (Arm 1 vs Arm 2):**
   - Arm 1 (BN statistics only, zero gradient steps) achieves $44.25\%$ accuracy ($\Delta = +3.30\%$).
   - Arm 2 (Reference SAR with SAM gradient optimization) achieves $44.25\%$ accuracy ($\Delta = +3.30\%$).
   - Test-time batch statistics alone account for **100.0%** of the adaptation gain observed on this panel. Incremental gradient updates under SAR contribute no measurable additional benefit beyond test-time batch statistics.
5. **KGA Routing Utility:**
   - Protocol B KGA strictly maintains $\mathrm{FA}_u = 0.0000$ (0 false adaptations across all 135 evaluations). Across all arms, regret is bounded below $0.20\%$.
6. **Nature of KGA Gains:**
   - On this panel, candidate adaptation is universally positive ($\Delta > 0$ for all 27 conditions). KGA's abstentions (3–4 cells in Arms 1–3) reflect finite-sample uncertainty under exact conformal bounds rather than filtering of negative cells.

---

## 9. Final Deliverable Status Block

```
TASK 4 STATUS: COMPLETE

AUTHENTICATED STUDY:
experiments/kbound/results/task4_matched_sar_v1/ (MANIFEST: TASK4_MATCHED_SAR_MANIFEST.json)

SOURCE CHECKPOINT:
/Users/pratik_n/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
SHA-256: 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca

PANEL:
27 cells (3 corruptions: gaussian_noise, shot_noise, impulse_noise x 3 severities: 1, 3, 5 x 3 stream compositions: iid, imbalanced, single_class; small batch regime b=16, seed 0; 135 evaluations total across 5 arms)

ARMS:
1. BN-only: eta=0.0, no gradient updates, test batch stats enabled
2. Reference SAR: eta=2.5e-4, layers 1-3 affine adapted, layer 4 frozen
3. Rate only: eta=4.0e-3, layers 1-3 affine adapted, layer 4 frozen
4. Mask only: eta=2.5e-4, layers 1-4 affine adapted
5. Combined: eta=4.0e-3, layers 1-4 affine adapted

PRIMARY FINDINGS:
[Candidate-Only Results]
- Frozen baseline accuracy: 40.95%
- Arm 1 (BN-only): Cand Acc 44.25% (Delta = +3.30%, Norm = 0.0000)
- Arm 2 (Reference SAR): Cand Acc 44.25% (Delta = +3.30%, Norm = 0.0011)
- Arm 3 (Rate only): Cand Acc 44.26% (Delta = +3.30%, Norm = 0.0182)
- Arm 4 (Mask only): Cand Acc 44.25% (Delta = +3.30%, Norm = 0.0015)
- Arm 5 (Combined): Cand Acc 44.26% (Delta = +3.31%, Norm = 0.0241)

[Routing Results (Protocol B KGA, alpha=0.10, tau=0)]
- Arm 1: KGA Acc 44.05%, Regret 0.20%, Decisions: 23A / 0F / 4Abs, FA_u = 0.00%
- Arm 2: KGA Acc 44.11%, Regret 0.14%, Decisions: 24A / 0F / 3Abs, FA_u = 0.00%
- Arm 3: KGA Acc 44.06%, Regret 0.19%, Decisions: 23A / 0F / 4Abs, FA_u = 0.00%
- Arm 4: KGA Acc 44.25%, Regret 0.00%, Decisions: 27A / 0F / 0Abs, FA_u = 0.00%
- Arm 5: KGA Acc 44.26%, Regret 0.00%, Decisions: 27A / 0F / 0Abs, FA_u = 0.00%
- FA_u = 0.0000 (strictly zero false adaptation) across all 5 arms.

INTERPRETATION:
Under the matched panel, candidate accuracy differences across arms are <= 0.01%. Test-time BatchNorm statistics account for 100.0% of the adaptation gain (+3.30% over frozen baseline). Incremental gradient updates under SAR contribute <= 0.01% additional benefit. Neither aggressive learning rate nor unfreezing layer 4 affine parameters induces model collapse under faithful test-time normalization. The earlier historical collapse was an artifact of improper final-block BatchNorm handling rather than an inherent failure of SAR gradient updates.

MANUSCRIPT FILES UPDATED:
- docs/research/kbound/kbound_submission_supplement.tex (Appendix B.3, Table 23 + narrative)
- docs/research/kbound/paper/generated/tab_task4_matched_sar.tex (Table 23 source)
- docs/research/kbound/TASK1_4_FINAL_CLOSEOUT.md
- docs/research/kbound/FINAL_TASK1_4_CLOSEOUT.md
- docs/research/kbound/TASK4_FINAL_CLOSEOUT.md

VALIDATION:
- docs/research/kbound/scripts/test_sar_faithful.py: PASS (A1-A4)
- tests/test_sar_final_block_batch_stats.py: PASS (4/4)
- tests/test_sar_runbook_exit_status.py: PASS (4/4)
- tests/test_manuscript_calibration_scope.py: PASS (6/6)
- tests/test_cifar_current_arithmetic.py: PASS (13/13)
- src/scripts/validate_manuscript_claims.py: PASS (40 maintained LaTeX sources, 0 errors)
- Full core pytest suite: PASS (50 passed in 6.15s)

PDF BUILD:
compact PASS (docs/research/kbound/kbound_short_final_draft.pdf, 53 pages, 1,247,907 bytes, 0 errors, 0 ??, 0 [?], Table 23 on p. 34)
TMLR PASS (docs/research/kbound/kbound_tmlr.pdf, 60 pages, 1,285,719 bytes, 0 errors, 0 ??, 0 [?], Table 23 on p. 39)

REMAINING LIMITATION:
The matched-control panel evaluates 27 noise conditions around ResNet-50 seed 0. It replaces the un-authenticated historical 60-cell matrix as release-promotable evidence for the learning-rate and parameter-mask questions on ImageNet-C, but does not claim universal multi-architecture invariance outside the evaluated panel.
```
