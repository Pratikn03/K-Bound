# Model Registry Index

This index lists every serialized model artifact that ships under `models/`,
together with its task, framework, model card, and promotion status.

Integrity for these artifacts is tracked in [`MANIFEST.json`](./MANIFEST.json)
(SHA-256 per artifact) and enforced at load time by
`uais.registry.ModelRegistry` (raises `IntegrityError` on mismatch). Regenerate
real hashes with:

```bash
PYTHONPATH=src python -m uais.registry.build_manifest
```

## Shipped artifacts

| Model | File | Task | Framework | Card | Status |
|---|---|---|---|---|---|
| `fraud` | `models/fraud/supervised/fraud_model.pkl` | Supervised fraud classification | scikit-learn Pipeline (HistGradientBoosting) | [card](./fraud/MODEL_CARD.md) | Research artifact; test ROC-AUC 0.892 / PR-AUC 0.592 |
| `cyber` | `models/cyber/supervised/cyber_model.pkl` | Supervised intrusion classification | scikit-learn HistGradientBoostingClassifier | [card](./cyber/MODEL_CARD.md) | Research artifact; **test metrics 1.000 — NOT validated (likely leakage)** |
| `behavior_autoencoder` | `models/behavior/behavior_autoencoder.pkl` | Unsupervised behavior anomaly (reconstruction) | scikit-learn MLPRegressor | [card](./behavior/MODEL_CARD.md) | Research artifact; near-chance (ROC-AUC 0.515) |
| `behavior_lof` | `models/behavior/behavior_lof.pkl` | Unsupervised behavior anomaly (density) | scikit-learn LocalOutlierFactor | [card](./behavior/MODEL_CARD.md) | Research artifact; near-chance (ROC-AUC 0.524) |
| `fusion` | `models/fusion/fusion_meta_model.pkl` | Late-fusion meta-classification | scikit-learn LogisticRegression | [card](./fusion/MODEL_CARD.md) | Research artifact; CV ROC-AUC 0.814 (small test split) |

The "Status" column reflects the honest, recorded experiment results
(`experiments/<domain>/metrics/metrics.json`), not a production-readiness claim.
None of these artifacts is a production-decisioning model; see each card's
*Intended use* and *Known limitations* sections.

## Relationship to the frozen component registry

`research_lock/model_registry_v1.yaml` is a **separate, frozen** registry that
tracks ELARA / attention-fusion *components* and reliability-gate variants
(e.g. `static_attention`, `base_rga`, `rga_plus`, gate variants `G0`–`G3`,
estimator candidates `R0`–`R5`). Those entries map to source modules under
`src/uais/fusion/attention/` and `src/uais/utils/metrics.py`, **not** to the five
supervised/unsupervised pickles above. The most relevant promotion statuses from
that frozen file:

| Component (frozen registry) | Maps to | Status (verbatim from `model_registry_v1.yaml`) |
|---|---|---|
| `static_attention` | `AttentionFusionModel` (reference comparator) | reference comparator / baseline |
| `base_rga` | `ReliabilityEstimator` | PARTIAL (B1 confirmed; sensitive gates rejected; transfer not confirmed) |
| `rga_plus` | `ReliabilityBoostedFusion` | CONFIRMED vs fixed static attention only (P2 open) |
| `router` | validation-selected model selection | NOT_INDEPENDENTLY_VALIDATED |
| `monitor_certificate` | `bounded_switching_certificate`, `calibration_monitor_report` | PARTIAL (retrospective only) |

The supervised/unsupervised artifacts in the table above are **not tracked** in
`model_registry_v1.yaml`; their status here is sourced from their own recorded
experiment metrics. A trained attention-fusion checkpoint
(`models/fusion/attention/attention_fusion.pt`) is referenced by the deployment
API but is **not present** in this repo snapshot, so it is intentionally omitted
from `MANIFEST.json`.
