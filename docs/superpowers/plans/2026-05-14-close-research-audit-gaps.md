# Close Research Audit Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the research-audit blockers that prevent the ELARA/RGA paper from being treated as defensible empirical evidence.

**Architecture:** Fix validity at the data contract first: generated fusion CSVs must carry explicit split metadata, benchmark builders must avoid scorer/test transduction, and the experiment runner must respect predefined splits when requested. Then fix statistical/reporting inconsistencies and add checks that fail before stale claims reach the paper.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, pytest, LaTeX asset scripts.

---

### Task 1: RealFusion-LA Source-Disjoint Fusion Splits

**Files:**
- Modify: `src/scripts/prepare_real_fusion_benchmark.py`
- Test: `tests/test_real_fusion_harder_benchmark.py`

- [ ] **Step 1: Write failing tests** that call a helper to generate composite rows with `fusion_split` and assert no `(domain, source_row, label)` key appears in more than one split.
- [ ] **Step 2: Run the targeted test** with `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_real_fusion_harder_benchmark.py -q` and verify the new test fails against the current global sampler.
- [ ] **Step 3: Implement split-safe source pools** by stratifying every domain frame into train/validation/test source pools before composite sampling.
- [ ] **Step 4: Emit split metadata**: `fusion_split`, source split counts, and a metadata flag declaring source-row-disjoint generation.
- [ ] **Step 5: Run the targeted test** and verify it passes.

### Task 2: Runner Predefined Split Support

**Files:**
- Modify: `src/scripts/run_breakthrough_experiment.py`
- Test: create `tests/test_breakthrough_predefined_split.py`
- Config: `configs/attention_real_fusion.yaml`

- [ ] **Step 1: Write failing tests** for `_split` using `predefined_split` labels and expected train/validation/test indices.
- [ ] **Step 2: Run the targeted test** and verify it fails because `_split` currently ignores predefined splits.
- [ ] **Step 3: Add split metadata loading** from the fusion DataFrame and pass it into `_run_experiment_arrays`.
- [ ] **Step 4: Add config keys** for `training.split_column`, `training.train_split_values`, `training.val_split_values`, and `training.test_split_values`.
- [ ] **Step 5: Run targeted runner tests** and verify they pass.

### Task 3: MVTec 3D Non-Transductive Score Prep

**Files:**
- Modify: `src/scripts/prepare_mvtec3d_fusion_benchmark.py`
- Test: `tests/test_mvtec3d_benchmark.py`
- Config: `configs/attention_mvtec3d_fusion.yaml`

- [ ] **Step 1: Write failing tests** asserting MVTec normal-reference scorers fit only `split=train, defect_type=good` observations and metadata records scorer/evaluation protocol.
- [ ] **Step 2: Run the targeted MVTec test** and verify it fails against full-dataset normalization.
- [ ] **Step 3: Fit score references only on original train-good samples** and normalize distances with train-good statistics.
- [ ] **Step 4: Emit protocol metadata** that distinguishes score-fitting split from the supervised fusion stress-test split.
- [ ] **Step 5: Update MVTec config comments** so they no longer claim the old bagel smoke-run protocol.
- [ ] **Step 6: Run targeted tests** and verify they pass.

### Task 4: Paired DeLong Test

**Files:**
- Modify: `src/uais/utils/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 1: Write failing tests** showing identical paired score vectors return a non-significant p-value and that finite filtering preserves row pairing.
- [ ] **Step 2: Run `tests/test_stats.py`** and verify failure against independent-variance logic where appropriate.
- [ ] **Step 3: Replace the current independent variance sum with paired DeLong covariance for two correlated ROC curves.**
- [ ] **Step 4: Run `tests/test_stats.py`** and verify it passes.

### Task 5: Build Reproducibility and Paper Claim Cleanup

**Files:**
- Modify: `docs/research/data/FUSION_SCHEMA.md`
- Modify: `docs/research/PAPER_DRAFT_v1.tex`
- Modify: `docs/research/THESIS_CHAPTER_v1.tex`
- Modify: `docs/research/PAPER_DRAFT_v1.md`
- Track: `scripts/rebuild_paper.sh`, `src/scripts/emit_mvtec3d_assets.py`, generated `elara_*` and `mvtec3d_*` paper assets needed by the manuscripts.

- [ ] **Step 1: Update schema docs** to require split-safe alignment when `fusion_split` is present.
- [ ] **Step 2: Update manuscripts** so claims match split-safe RealFusion and MVTec protocol metadata.
- [ ] **Step 3: Rebuild assets/PDF** with `./scripts/rebuild_paper.sh`.
- [ ] **Step 4: Run full tests** with `PYTHONPATH=src ./.venv/bin/python -m pytest -q`.
- [ ] **Step 5: Re-run the audit checks** for source-row split overlap and MVTec score protocol metadata.
