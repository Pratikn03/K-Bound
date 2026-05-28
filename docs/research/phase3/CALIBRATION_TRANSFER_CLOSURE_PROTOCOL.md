# Calibration-Transfer Closure Protocol

Status: preregistered execution package

## Scope

This protocol closes one gap only:

Low clean false-fire plus positive transfer benefit for base RGA on held-out naturally paired multimodal data under pre-specified degradation.

## Primary hypothesis

Under frozen validation-only calibration and threshold selection, base RGA improves over static attention on primary stress endpoints while maintaining clean validation false-fire rate <= 0.010.

## Dataset and pairing

- Dataset: Eyecandies (RGB + depth)
- Pairing: naturally co-observed modalities
- Splits: train normal-only, validation normal-only, test held-out anomalous

## Methods and comparators

- Primary method: base RGA
- Comparator: fixed static attention
- Policy: one frozen gate policy across cells, no per-cell reselection

## Locked endpoints

- D-EYE-1: depth score collapse
- D-EYE-2: rgb score collapse
- D-EYE-3: single-modality missingness (secondary descriptive)

## Statistical policy

- Primary metric: Delta AUC (RGA - static)
- Seeds: target 30, minimum 15
- Inference: DeLong paired test + paired bootstrap CI (10,000 iterations)
- Multiplicity: Holm-Bonferroni, K=2 across D-EYE-1 and D-EYE-2
- Practical threshold: minimum meaningful positive delta = 0.010

## Mandatory provenance disclosure (must be completed in result summary)

1. Exact data used to fit KS/calibration reference.
2. Confirmation that only normal validation data were used for that fit.
3. Confirmation that no test scores, labels, or outcome metrics influenced threshold or calibration selection.
4. Confirmation that calibration repair and threshold policy were frozen before reading final test outcomes.

If any item above is false, classify the run as exploratory post-test repair evidence, not held-out confirmatory transfer evidence.

## Execution

Dry-run (recommended first; no test fold execution):

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase3_calibration_transfer_closure.py
```

One-time held-out run (authorized execution):

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase3_calibration_transfer_closure.py --full-run
```

Custom seed range:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase3_calibration_transfer_closure.py --full-run --seeds 30 --seed-start 42
```

## Outputs

- Manifest JSON with frozen hashes and provenance fields:
  - docs/research/phase3/CALIBRATION_TRANSFER_CLOSURE_MANIFEST.json
- Family-D per-cell outputs and archives under:
  - experiments/phase2/family_d/
- Family-D inference summary:
  - docs/research/phase2/FAMILY_D_V3_INFERENCE_REPORT.md

## Claim ceiling

Allowed positive claim requires all of:

- clean false-fire <= 0.010 on validation,
- positive primary endpoint effect with CI excluding 0 after multiplicity correction,
- practical threshold satisfied.

Otherwise, retain bounded negative or inconclusive transfer finding.