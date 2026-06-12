# Model Card — Behavior Anomaly Detectors (Autoencoder + LOF)

This card covers the two unsupervised behavior-anomaly artifacts that ship
together:

- **`behavior_autoencoder`** — `models/behavior/behavior_autoencoder.pkl`
  (registry key `behavior__behavior_autoencoder`)
- **`behavior_lof`** — `models/behavior/behavior_lof.pkl`
  (registry key `behavior__behavior_lof`)

- **Version:** 1.0
- **SHA-256:** see `models/MANIFEST.json` (verified by `uais.registry.ModelRegistry`)

## Task
Unsupervised anomaly scoring of user-session / behavior records. Each model
produces an anomaly score; higher means more anomalous. There is no supervised
target at training time (these are density / reconstruction estimators).

## Framework
scikit-learn (small CPU pickles loaded via `joblib`, not deep-learning
checkpoints). Each artifact is a dict `{"model": <estimator>, "preprocessor":
<StandardScaler>}`.

- **Autoencoder:** `sklearn.neural_network.MLPRegressor` trained to reconstruct
  its standardized input; anomaly score is the per-row mean squared
  reconstruction error. Configuration: `hidden_layer_sizes=(64, 32, 64)`,
  `max_iter=200`, `learning_rate_init=0.001` (see
  `uais.anomaly.train_autoencoder`).
- **LOF:** `sklearn.neighbors.LocalOutlierFactor(novelty=True)`; anomaly score is
  the negated `score_samples` output (see `uais.anomaly.train_lof`). The shipped
  pickle reports `n_neighbors=20`, `contamination≈0.1`, `metric="minkowski"`
  fit on a standardized feature matrix.

## Training data
Behavior data loaded by `uais.data.load_behavior_data` — by default the
**Online Shoppers Purchasing Intention** tabular dataset (session features such
as `Administrative`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`,
`PageValues`, `Month`, `VisitorType`; the `Revenue` column is used only as an
evaluation label, not for training). If a CERT r4.2 LDAP directory is present it
is loaded instead. Models are fit unsupervised; a stratified split is used only
to compute evaluation metrics (`src/scripts/run_behavior_experiment.py`).

## Intended use
Research / demonstration anomaly scoring of behavior feature vectors within this
repository, and as one input domain to the fusion meta-model. Not a production
insider-threat or fraud-detection system.

## Metrics (evaluation split, recorded in `experiments/behavior/metrics/metrics.json`)
Metrics treat the `Revenue` label as ground truth purely to gauge ranking
quality; the models themselves are unsupervised.

| Metric | Autoencoder | LOF |
|---|---|---|
| ROC-AUC | 0.515 | 0.524 |
| PR-AUC | 0.160 | 0.166 |
| F1 | 0.114 | 0.137 |
| Balanced accuracy | 0.515 | 0.524 |

## Known limitations — read before trusting these numbers
- **Both detectors score essentially at chance (ROC-AUC ≈ 0.51–0.52).** On this
  benchmark, reconstruction error and local density carry almost no signal about
  the `Revenue` label. These should be regarded as **weak / near-random baselines**,
  not validated anomaly detectors.
- Evaluating unsupervised anomaly scores against the purchase-intent `Revenue`
  label is a proxy: "anomalous session" and "purchasing session" are not the
  same concept, which partly explains the low scores.
- The autoencoder is an `MLPRegressor` (a shallow feedforward net), not a tuned
  deep autoencoder; capacity and training budget (`max_iter=200`) are small.
- No distribution-shift, temporal, or cross-population validation has been done.
