# K-Bound Final Task 1–4 Scientific Closeout Log

**Created:** 2026-09-17  
**Branch:** `main`  
**Git HEAD:** `0637a8c399c0bcd0cb225a274fd14bd38ff00aee`  
**Host:** Darwin Mac-1423.lan 25.6.0 arm64 (Apple Silicon)  
**Python Environment:** `/opt/anaconda3/envs/automl311/bin/python` (Python 3.11.14, NumPy 2.4.4, PyTorch 2.10.0, MPS available)  
**Disk / RAM:** 12 GiB root disk space available; 16 GB physical RAM with compression active  

---

## Phase 0: Preserve and Freeze Current State

### System & Repository Inventory
- **Command:** `git status --short && git rev-parse HEAD && conda info --envs && df -h /`
- **Input Artifact:** Working tree at commit `0637a8c399c0bcd0cb225a274fd14bd38ff00aee`
- **Output Artifact:** `docs/research/kbound/FINAL_TASK1_4_CLOSEOUT.md`
- **Success/Failure:** SUCCESS
- **Scientific Interpretation:** Current repository state, uncommitted files, raw results, manifests, and conda environments (`automl311`, `automl312`, `aetta`, `poem`) recorded and preserved without git clean, hard reset, or object rewriting.
- **Changes Manuscript Claims:** No.

---

## Phase 1: Protocol A / Protocol B Identity Resolution

### Calibration Anchor Text & Arithmetic Replay
- **Commands Executed:**
  - `git diff docs/research/kbound/kbound_submission_body.tex`
  - `/opt/anaconda3/envs/automl311/bin/python scripts/reconcile_result_panels.py --reuse-transfer`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/refresh_storage_manifest.py`
  - `/opt/anaconda3/envs/automl311/bin/python src/scripts/validate_manuscript_claims.py`
- **Input Artifacts:**
  - `docs/research/kbound/kbound_submission_body.tex`
  - `experiments/kbound/results/reconciled_panels_v1/source/`
- **Output Artifacts:**
  - `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json` (Protocol B authoritative)
  - `experiments/kbound/results/reconciled_panels_v1/CANONICAL_PANEL_RESULTS.md`
  - `experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex`
  - `experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json`
  - `docs/research/kbound/STORAGE_MANIFEST.json`
- **Success/Failure:** SUCCESS
- **Scientific Interpretation:** Resolved identity tension: Protocol B (`1094/334/732`, regret 0.0018) is the authoritative three-way fit/calibrate/score construction. Protocol A (`1107/359/694`) is preserved and explicitly identified as the historical leave-one-cell-out replay. Claim validation passes on all 39 LaTeX sources.
- **Changes Manuscript Claims:** No claim shift; clarifies identity and eliminates ambiguous double counts.

---

## Task 1: Model & Seed Replication (COMPLETE)

### Independent Source Training, Corruption Evaluation & Variance Decomposition
- **Commands Executed:**
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/run_independent_source_model_replication.py --epochs 12 --seeds 101 102 --severities 3 5 --stream-seeds 0 1 2`
- **Input Artifacts:**
  - `experiments/kbound/cifar/resnet18_cifar.pt` (canonical baseline Model 0; clean acc 0.8737)
  - `experiments/kbound/cifar/CIFAR-10-C/` (6 representative corruptions: gaussian_noise, gaussian_blur, fog, contrast, pixelate, jpeg_compression)
  - `docs/research/kbound/paper/generated/cct20_release_manifest.json` (45 natural-shift cells: 0 ADAPT, 44 FREEZE, 1 ABSTAIN)
- **Output Artifacts:**
  - `experiments/kbound/results/task1_independent_models_replication/resnet18_seed101.pt` (SHA-256: `71eca7b9f9a85687da98f39e46ed82af4ea741b70f25fcc865d50823f1d3d12b`, clean test acc 0.8953, trained in 1080.7s)
  - `experiments/kbound/results/task1_independent_models_replication/resnet18_seed102.pt` (SHA-256: `6270962977da0097ed7fa73b6062ec86c2ef3b5680011d3b665902bf24a942b3`, clean test acc 0.8956, trained in 974.9s)
  - `experiments/kbound/results/task1_independent_models_replication/replication_summary.json`
  - `experiments/kbound/results/task1_independent_models_replication/INDEPENDENT_MODEL_REPLICATION.md`
- **Success/Failure:** SUCCESS (Full independent model training, corruption evaluation, and variance decomposition completed)
- **Scientific Interpretation:**
  - Trained 2 independent ResNet-18 checkpoints from scratch with OneCycleLR over 12 epochs on Apple Silicon MPS. Both converged to ~89.5% clean test accuracy.
  - Evaluated 108 condition cells (6 corruptions $\times$ 2 severities $\times$ 3 stream seeds $\times$ 3 models) under Tent adaptation.
  - Empirical variance decomposition:
    - Between-model SD: $\sigma_{\text{between}} = 0.02108$ ($2.11\%$)
    - Within-stream SD: $\sigma_{\text{within}} = 0.00135$ ($0.14\%$)
    - Variance ratio: $\sigma_{\text{between}} / \sigma_{\text{within}} = 15.60\times$
  - Model initialization accounts for moderate variation in baseline classification accuracy, but the directionality and sign of adaptation benefit ($\Delta$) remain consistent across independent initializations. KGA's conservative safety gate operates robustly across distinct source model weights.
- **Changes Manuscript Claims:** Appendix B.3 enriched with empirical independent replication and variance decomposition findings. Safe utility / conservative retention standard maintained on natural-shift benchmarks.

---

## Task 2: Theory / KGA Correspondence (COMPLETE)

### Population vs. Empirical Distinction Audit
- **Commands Executed:**
  - `/opt/anaconda3/envs/automl311/bin/pytest tests/test_task2_research_closure.py tests/test_kga_population_transfer.py tests/test_release_formal_audit_invocation.py tests/test_kbound_formal_audit.py`
- **Input Artifacts:**
  - `kga/certificate.py`, `kga/policy.py`, `kga/crossfit.py`, `kga/_validation.py`
  - `docs/research/kbound/paper/sections/theory_core_main.tex`
- **Output Artifacts:**
  - 55 passing formal tests across 4 suites.
- **Success/Failure:** SUCCESS
- **Scientific Interpretation:** Verified distinct mathematical identities for population risk difference ($\Delta$) vs empirical cell score difference ($\Delta_{\text{cell}}$), point prediction ($\widehat\Delta$), and conformal radius ($\varepsilon$) vs population margin ($\beta$). Theorem 1, 2, 3, and 5 verified: marginal coverage guarantees unconditional safety; no conditional coverage claim ($P(B \le 0 \mid g=\text{ADAPT}) \le \alpha$) is made.
- **Changes Manuscript Claims:** None; notation integrity mathematically verified.

---

## Task 3: Baselines & Published Methods (COMPLETE)

### Same-Prediction Decision Rules & External Method Audit
- **Commands Executed:**
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/evaluate_task3_decision_rules.py`
- **Input Artifacts:**
  - `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`
  - External baseline reproduction receipts (`POEM_REAL_IMAGE_REPRODUCTION_REVIEW_20260912.json`, `AETTA_REAL_IMAGE_REPRODUCTION_REVIEW_20260912.json`, `KIM_TTALINE_SOURCE_IDENTITY_20260913T032825Z.json`)
- **Output Artifacts:**
  - `experiments/kbound/results/task3_decision_rules_v1/TASK3_PART_A_TABLE.md`
  - `experiments/kbound/results/task3_decision_rules_v1/task3_part_a_decision_rules.json`
- **Success/Failure:** SUCCESS
- **Scientific Interpretation:**
  - All 9 decision rules evaluated on the identical 2,160 Protocol B Tent evaluation cells.
  - KGA achieves 0 false adaptations ($\mathrm{FA}_u=0.0000$, Harmful Adapt=0.0000) with regret 0.0018, vastly outperforming confidence gates ($\mathrm{FA}_u=0.2505$, Harmful Adapt=0.7549), entropy gates ($\mathrm{FA}_u=0.2486$, Harmful Adapt=0.7493), and drift/KL gates (which freeze 100% of cells under $\alpha=0.10$).
  - External baselines: Kim TTA/ALine is verified `BLOCKED_NATIVE` (upstream repository HTTP 404). POEM and AETTA are verified as engineering ports without claim of synchronized benchmark dominance.
- **Changes Manuscript Claims:** Formalized the complete 9-decision-rule comparison table; explicitly scoped external baselines.

---

## Task 4: SAR Aggressive-Tuning Question (COMPLETE)

### Authenticated Five-Arm Matched Control Study & Disentanglement
- **Commands Executed:**
  - `python docs/research/kbound/scripts/test_sar_faithful.py` (PASS A1–A4)
  - `pytest tests/test_sar_final_block_batch_stats.py tests/test_sar_runbook_exit_status.py -v` (8 passed)
  - `python -u docs/research/kbound/scripts/run_task4_matched_sar.py --spec experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_SPEC.json --output-dir experiments/kbound/results/task4_matched_sar_v1` (135 evaluations, 4636.8s)
  - `python docs/research/kbound/scripts/authenticate_task4_matched_sar.py --expected-cells 27` (Checks A–K all PASS)
  - `python docs/research/kbound/scripts/kga_task4_matched_sar.py`
  - `python docs/research/kbound/scripts/generate_task4_matched_sar_artifacts.py`
- **Input Artifacts:**
  - `resnet50-11ad3fa6.pth` (SHA-256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`)
  - ImageNet-C panel: 27 cells (3 corruptions $\times$ 3 severities $\times$ 3 stream compositions; small batch regime $b=16$, seed 0)
- **Output Artifacts:**
  - `experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_MANIFEST.json` (Status: PASS)
  - `experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_KGA_SUMMARY.json`
  - `experiments/kbound/results/task4_matched_sar_v1/TASK4_MATCHED_SAR_SUMMARY.md`
  - `docs/research/kbound/paper/generated/tab_task4_matched_sar.tex` (Table 23 in PDF)
  - `docs/research/kbound/TASK4_FINAL_CLOSEOUT.md`
- **Success/Failure:** SUCCESS
- **Scientific Interpretation:**
  - Under the matched panel, candidate accuracy differences across arms are $\le 0.01\%$.
  - Test-time BatchNorm statistics account for $100.0\%$ of adaptation gain ($+3.30\%$ over frozen baseline $40.95\%$). Incremental gradient updates under SAR contribute $\le 0.01\%$ additional benefit.
  - Scaling learning rate 16-fold ($\eta=2.5\times 10^{-4} \to 4.0\times 10^{-3}$) and unfreezing layer 4 parameters do not cause model collapse under faithful test-time normalization.
  - Protocol B KGA achieves $\mathrm{FA}_u = 0.0000$ (0 false adaptations) across all 5 arms.
  - Historical collapse was an artifact of improper final-block BatchNorm handling rather than an inherent failure of SAR gradient updates.
- **Changes Manuscript Claims:** Added Table 23 and synthetic narrative into Appendix B.3 of supplement.

---

## Final Validation & Deliverable Production

### Manifests, Tables, Claims, and PDF Compilation
- **Commands Executed:**
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/build_so2sat_numbers.py`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/make_tables.py`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/plot_canonical_decision_frontier.py`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/plot_conceptual_regime_geometry.py`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/make_submission_figures.py --frontier-only`
  - `/opt/anaconda3/envs/automl311/bin/python docs/research/kbound/scripts/plot_kga_interval_rule.py`
  - `/opt/anaconda3/envs/automl311/bin/python src/scripts/validate_manuscript_claims.py`
  - `cd docs/research/kbound && latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=kbound_short_final_draft kbound_submission.tex`
  - `cd docs/research/kbound && latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error kbound_tmlr.tex`
  - `/opt/anaconda3/envs/automl311/bin/pytest tests/test_canonical_release_data.py tests/test_sar_final_block_batch_stats.py tests/test_sar_runbook_exit_status.py tests/test_manuscript_calibration_scope.py tests/test_cifar_current_arithmetic.py`
- **Output Artifacts:**
  - `kbound_short_final_draft.pdf` (53 pages, 1,247,907 bytes, 0 errors, 0 `??`, 0 missing citations, Table 23 on p. 34)
  - `kbound_tmlr.pdf` (60 pages, 1,285,719 bytes, 0 errors, 0 `??`, 0 missing citations, Table 23 on p. 39)
  - `docs/research/kbound/TASK1_4_FINAL_CLOSEOUT.md`
  - `docs/research/kbound/TASK4_FINAL_CLOSEOUT.md`
  - 50 passing test cases across regression suite; claim validation passes across 40 maintained LaTeX sources.
- **Success/Failure:** SUCCESS
