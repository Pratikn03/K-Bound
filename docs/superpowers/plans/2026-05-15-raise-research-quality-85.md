# Raise Research Quality Toward 85+

> **For Pratik/Codex:** Execute with validation-safe tuning only. Do not tune on
> held-out test labels. Prefer evidence that makes the benchmark harder over
> changes that only increase headline scores.

**Goal:** Lift the research package from a 72/100 audit state toward an 85+
submission-quality evidence base by improving threshold discipline, benchmark
difficulty, reproducibility, and local verification.

**Architecture:** The experiment runner owns benchmark loading, predefined split
handling, model training, validation-threshold selection, stress evaluation, and
JSON output. Metric utilities own threshold selection. Baselines receive the
same validation-derived threshold policy as RGA/static attention.

**Tech Stack:** Python, PyTorch, NumPy, pandas, scikit-learn, pytest, Ruff,
YAML configs, LaTeX rebuild script.

## Completed Work

- [x] Add a validation-only threshold selector to `uais.utils.metrics`.
- [x] Record the applied `decision_threshold` in classification metric outputs.
- [x] Add tests proving `val_f1` thresholding uses validation labels and rejects
      invalid strategies such as `test_f1`.
- [x] Wire validation-selected thresholds through clean metrics, bootstrap F1
      CIs, stress rows, tau sweep rows, component ablation rows, and failure
      cases.
- [x] Apply the same threshold policy to the standard baseline suite.
- [x] Add `evaluation.decision_threshold: val_f1` to the primary benchmark
      configs.
- [x] Add `configs/attention_real_fusion_hard.yaml` for a hard RealFusion-LA
      rerun using `--scorer-train-fraction 0.05`.
- [x] Add config tests for validation thresholding and hard-mode artifact paths.
- [x] Extend `.gitignore` for test/lint/coverage/document build artifacts.
- [x] Update README with local quality gates and hard-mode reproduction
      commands.
- [x] Make `scripts/rebuild_paper.sh` compile both the manuscript and thesis
      chapter.
- [x] Fix the two undefined-name issues caught by the Ruff correctness gate.

## Verification

- [x] Focused threshold/config tests passed.
- [x] Full pytest suite passed: 224 collected, 3 skipped, 221 passing.
- [x] Ruff correctness gate passed:
      `PYTHONPATH=src ./.venv/bin/python -m ruff check --select E9,F63,F7,F82 .`
- [x] Synthetic end-to-end experiment smoke run completed and wrote threshold
      metadata.
- [x] Paper/thesis rebuild completed:
      `output/pdf/PAPER_DRAFT_v1.pdf` (17 pages) and
      `output/pdf/THESIS_CHAPTER_v1.pdf` (24 pages).

## Remaining 85+ Evidence Step

- [ ] Run the hard RealFusion-LA benchmark on real inputs and update the
      manuscript/result JSONs:

```bash
PYTHONPATH=src python src/scripts/prepare_real_fusion_benchmark.py \
  --scorer-train-fraction 0.05 \
  --output experiments/fusion/real_domain_fusion_hard_inputs.csv \
  --metadata experiments/fusion/real_domain_fusion_hard_metadata.json

PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_real_fusion_hard.yaml \
  --output experiments/fusion/craf_real_results_hard.json
```
