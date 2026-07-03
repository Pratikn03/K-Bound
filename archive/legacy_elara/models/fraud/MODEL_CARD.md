# Model Card — Fraud Supervised Classifier

- **Model name:** `fraud` (registry key `fraud__supervised__fraud_model`)
- **Artifact file:** `models/fraud/supervised/fraud_model.pkl`
- **Version:** 1.0
- **SHA-256:** see `models/MANIFEST.json` (verified by `uais.registry.ModelRegistry`)

## Task
Binary supervised classification of payment transactions as fraudulent vs.
legitimate, producing a calibrated fraud probability.

## Framework
scikit-learn. The artifact is a serialized `sklearn.pipeline.Pipeline`
(`ColumnTransformer` → `SimpleImputer(strategy="median", add_indicator=True)` →
`HistGradientBoostingClassifier`). It is a small CPU pickle loaded via `joblib`,
not a deep-learning checkpoint. Training configuration:
`HistGradientBoostingClassifier` with `max_depth=4`, `learning_rate=0.1`,
`max_iter=200` (see `src/scripts/run_fraud_experiment.py` and
`uais.supervised.train_fraud_supervised.FraudModelConfig`).

## Training data
Kaggle-style **Credit Card Fraud** tabular dataset loaded by
`uais.data.load_fraud_data` (`Time`, `Amount`, anonymized PCA features `V1`–`V28`,
binary target `Class`). Data is highly imbalanced (fraud is a small minority
class). Split 60/20/20 train/val/test with `random_state=42`, stratified on the
target. A synthetic fallback exists for smoke tests but is not used for the
shipped artifact.

## Intended use
Research / demonstration scoring of credit-card-style transaction feature
vectors within this repository (served via `deploy/api/main.py:/predict_fraud`).
It is **not** a production fraud-decisioning system and must not be used to make
automated approve/deny decisions about real customers.

## Metrics (held-out test split, recorded in `experiments/fraud/metrics/metrics.json`)
| Metric | Value |
|---|---|
| ROC-AUC | 0.892 |
| PR-AUC (average precision) | 0.592 |
| F1 | 0.708 |
| Precision | 0.630 |
| Recall | 0.808 |
| Balanced accuracy | 0.904 |
| Brier score | 0.0012 |
| ECE | 0.0008 |

A hybrid score (supervised blended with an Isolation Forest anomaly score,
alpha=0.7) is also computed in the experiment and reaches test ROC-AUC ≈ 0.958,
but the **shipped artifact is the supervised pipeline only** — the hybrid score
is a downstream combination, not a saved model.

## Known limitations
- Severe class imbalance: high overall accuracy is dominated by the majority
  (legitimate) class; PR-AUC (0.592) is the more honest headline metric.
- Trained and evaluated on a single public dataset; no temporal or
  cross-institution generalization has been validated.
- Calibration metrics (Brier/ECE) are computed on the same held-out split used
  for reporting and may be optimistic under distribution shift.
- The anonymized PCA features (`V1`–`V28`) are dataset-specific and not
  reproducible on raw transaction streams without the original transform.
