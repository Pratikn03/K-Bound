# `kbound` source map

This wrapper package indexes the real K-Bound implementations without relocating
load-bearing code. Each subpackage's `__init__.py` names the concrete source file.

| Subpackage | Real implementation |
|---|---|
| `evidence/`   | `src/uais/drift/*`, `evidence_vector()` in `src/scripts/kbound/cifar_tent_mps_v2.py` |
| `estimators/` | `decide_kga()` (LOO-GBM + conformal) in `cifar_tent_mps_v2.py`; `vendored_from_elara/certification/switching_certificate.py` (empirical-Bernstein) |
| `decision/`   | `knowability_experiment.py`, `mixed_regime_experiment.py`, `policy_metrics()` |
| `metrics/`    | `src/uais/utils/metrics.py` + inline |
| `theory/`     | `vendored_from_elara/theory/*` (T1–T9, GDR) + certification |
| `data/`       | loaders point to `manifests/score_cache_manifest.csv` (no data duplicated) |
| `utils/`      | `src/uais/utils/*` |

The experiment entry points are copied under `../scripts/`. Originals remain in
`src/scripts/kbound/` (nothing deleted).
