# Model Card — Fusion Meta-Model (Score Stacker)

- **Model name:** `fusion` (registry key `fusion__fusion_meta_model`)
- **Artifact file:** `models/fusion/fusion_meta_model.pkl`
- **Version:** 1.0
- **SHA-256:** see `models/MANIFEST.json` (verified by `uais.registry.ModelRegistry`)

## Task
Late-fusion meta-classification: combine per-domain anomaly/risk scores
(fraud, cyber, behavior, vision) into a single overall risk probability.

## Framework
scikit-learn. The artifact is a dict `{"model": LogisticRegression, "scaler":
StandardScaler}` (a small CPU pickle loaded via `joblib`, not a deep-learning
checkpoint). The logistic regressor is trained with `max_iter=200` and
`class_weight="balanced"` on standardized meta-features derived from the
per-domain score vectors (see `uais.fusion.train_fusion_model.train_fusion_meta_model`
and `uais.fusion.build_embeddings.generate_meta_features`).

> Note: the deployment API (`deploy/api/main.py`) references the fusion artifact
> at `experiments/fusion/models/fusion_meta_model.pkl`. The canonical shipped
> artifact tracked here lives at `models/fusion/fusion_meta_model.pkl` (the path
> produced by `uais.utils.paths.domain_paths("fusion")`). See the registry
> upgrade report for the path-alignment recommendation.

## Training data
Aligned per-domain score CSVs from the individual experiments
(`experiments/<domain>/scores.csv` for fraud / cyber / behavior / vision),
joined on `sample_id` (and `timestamp` when available) or truncated to the
shortest common length. Meta-features are generated from these scores; the binary
label is carried through from the per-domain score files. Split with
`test_size=0.2`, `random_state=42`, stratified; a 3-fold stratified
cross-validated ROC-AUC is also reported for stability.

## Intended use
Research / demonstration aggregation of multiple anomaly domains into one risk
score within this repository (served via `deploy/api/main.py:/predict_fusion`).
Not a production risk-decisioning system.

## Metrics (held-out test split, recorded in `experiments/fusion/metrics/metrics.json`)
| Metric | Value |
|---|---|
| ROC-AUC (test) | 0.694 |
| Cross-validated ROC-AUC (3-fold mean) | 0.814 |
| PR-AUC | 0.520 |
| F1 / Precision / Recall | 0.545 |
| Balanced accuracy | 0.772 |
| Brier score | 0.063 |
| ECE | 0.235 |

## Known limitations
- The held-out test split is **very small** (confusion matrix totals: tp=6, fp=5,
  tn=2450, fn=5), so point metrics are high-variance; the CV ROC-AUC (0.814) is a
  more stable estimate than the single-split test ROC-AUC (0.694).
- Calibration is poor (ECE ≈ 0.235): predicted probabilities should not be read
  as well-calibrated risk without recalibration.
- Fusion quality is bounded by the quality of its inputs. The behavior inputs are
  near-random (see `models/behavior/MODEL_CARD.md`) and the cyber input's headline
  metrics are not validated for generalization (see `models/cyber/MODEL_CARD.md`),
  so the fusion score inherits those caveats.
- Score alignment across domains depends on consistent `sample_id`/`timestamp`
  keys; misaligned or truncated inputs change what the model actually fuses.
