# Model Card — Cyber Intrusion Supervised Classifier

- **Model name:** `cyber` (registry key `cyber__supervised__cyber_model`)
- **Artifact file:** `models/cyber/supervised/cyber_model.pkl`
- **Version:** 1.0
- **SHA-256:** see `models/MANIFEST.json` (verified by `uais.registry.ModelRegistry`)

## Task
Binary supervised classification of network-flow records as attack vs. normal
traffic, producing an attack probability.

## Framework
scikit-learn. The artifact is a serialized
`sklearn.ensemble.HistGradientBoostingClassifier` (a small CPU pickle loaded via
`joblib`, not a deep-learning checkpoint). Training configuration:
`loss="log_loss"`, `learning_rate=0.1`, `max_iter=200`, `max_depth=6`
(see `src/scripts/run_cyber_experiment.py` and
`uais.supervised.train_cyber_supervised.CyberModelConfig`).

## Training data
**UNSW-NB15** network intrusion dataset loaded by `uais.data.load_cyber_data`
(mixed categorical/numeric flow features such as `proto`, `service`, `state`,
`dur`, `sbytes`, `dbytes`, `sttl`, `dttl`; binary `label` where non-`normal`
traffic is positive). Split 60/20/20 train/val/test with `random_state=42`,
stratified on the target. A synthetic UNSW-like fallback exists for smoke tests
but is not used for the shipped artifact.

## Intended use
Research / demonstration scoring of network-flow feature vectors within this
repository (served via `deploy/api/main.py:/predict_cyber`). It is **not** a
production intrusion-detection system.

## Metrics (held-out test split, recorded in `experiments/cyber/metrics/metrics.json`)
| Metric | Value |
|---|---|
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| F1 / Precision / Recall | 1.000 |
| Balanced accuracy | 1.000 |
| Brier score | ~3.3e-15 |

## Known limitations — read before trusting these numbers
- **The reported test metrics are a perfect 1.000 across the board, which is a
  red flag rather than a success.** Perfect separation on UNSW-NB15 is almost
  always a symptom of (a) label leakage from features that encode the label, (b)
  train/test contamination, or (c) evaluating on a trivially separable split.
  These metrics should be treated as **not validated for generalization** and
  must not be cited as real detection performance.
- For contrast, the unsupervised Isolation Forest baseline computed in the same
  run scores ROC-AUC ≈ 0.357 (worse than chance) on the same data, underscoring
  that the supervised "perfect" result does not reflect genuine separability of
  attacks from normal traffic by honest features.
- Recommended follow-up before any reliance: audit `build_cyber_feature_table`
  and the dropped-columns config for leakage, and re-evaluate with a
  leakage-free, time-ordered split. Until then, treat true generalization
  performance as **not recorded / unknown**.
- Trained and evaluated on a single public dataset; no cross-network or temporal
  generalization has been validated.
