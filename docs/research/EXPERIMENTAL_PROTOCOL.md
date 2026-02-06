# Experimental Protocol

## Splits

- Default split: 70/15/15 train/val/test with fixed seeds.
- Sequential domains use time-based splits.
- Alignment is performed within split boundaries only.

## Metrics

- ROC-AUC, PR-AUC, F1, balanced accuracy.
- Detection rate at FPR=1 percent.
- Calibration: Brier score and ECE.

## Runs

- Run 3 seeds and report mean +/- std.
- Use bootstrap confidence intervals for PR-AUC and F1.
- DeLong test for ROC-AUC comparisons.

## Ablations

- Attention vs no attention.
- Domain embeddings on/off.
- Confidence estimator on/off.
- Head count: 4, 8, 16.

## Robustness

- Domain dropout at train time.
- Remove each domain at test time and report degradation.
