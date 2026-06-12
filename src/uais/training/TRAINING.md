# UAIS Unified Training Harness

A thin, reproducible orchestration layer that wraps the project's existing
`train_*.py` scripts behind one consistent interface. It standardises **seeding**,
**experiment tracking**, **artifact persistence**, and **model-card emission**
without re-implementing any model logic — every trainer is a small adapter that
imports and calls the original entrypoint.

> Design rule: *additive only*. The harness never modifies or duplicates the
> existing training scripts; it composes them.

---

## Why this exists

The repo had a dozen-plus independent `train_*.py` modules with divergent
signatures, no shared seeding, no run tracking, and no provenance for produced
artifacts. The harness gives every model:

- deterministic seeding (`random` + NumPy + Torch when present),
- a uniform `build -> fit -> evaluate -> log -> save -> model-card` lifecycle,
- pluggable experiment tracking (MLflow when installed, JSON fallback otherwise),
- a self-describing `model_card.json` capturing config, metrics, and environment.

---

## Module map

| Module | Responsibility |
| --- | --- |
| `trainer.py` | `set_seed`, `TrainConfig` dataclass, abstract `Trainer` base (`run()` orchestrator). |
| `tracking.py` | `get_tracker()` factory; `MLflowTracker` / `JSONTracker` with one shared interface. |
| `registry.py` | `TRAINERS` registry, `@register` decorator, `get_trainer()` factory, and the built-in adapters. |
| `cli.py` / `__main__.py` | `python -m uais.training.cli --config <yaml>` runner. |
| `TRAINING.md` | This document. |

Importing `uais.training` is dependency-light: **no PyTorch, TensorFlow, or
MLflow is required at import time.** Heavy backends are imported lazily inside a
trainer's `fit()` only when that trainer is actually run.

---

## Config schema (`configs/training/*.yaml`)

Each YAML file describes one training run and maps onto `TrainConfig`:

```yaml
name: isolation_forest          # required: unique run name (also default trainer key)
trainer: isolation_forest       # optional: explicit registry key (defaults to `name`)
seed: 42                        # RNG seed applied before any work
data_path: data/processed/...   # input data (relative paths resolve to PROJECT_ROOT); may be null
output_dir: experiments/...     # where artifacts + model_card.json are written
hyperparams:                    # free-form, forwarded to the underlying train_* function
  contamination: 0.01
tracker: json                   # "json" (default) or "mlflow"
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | str | — (required) | Non-empty; also the tracking experiment name. |
| `trainer` | str | `name` | Registry key resolving the adapter. |
| `seed` | int | `42` | Must be a non-negative int. |
| `data_path` | str \| null | `null` | Resolved against `PROJECT_ROOT` if relative. |
| `output_dir` | str | `experiments` | Resolved against `PROJECT_ROOT` if relative. |
| `hyperparams` | mapping | `{}` | Passed through to the underlying script. |
| `tracker` | `"json"` \| `"mlflow"` | `"json"` | Validated; anything else raises. |

Unrecognised top-level keys are preserved in `TrainConfig.extra` for lossless
round-tripping. Validation happens in `__post_init__` (empty name, negative
seed, bad tracker, and non-mapping `hyperparams`/`extra` all raise `ValueError`).

Ships with configs for: `isolation_forest`, `lof`, `ocsvm`, `autoencoder`,
`fusion_meta`, `vae`.

---

## Reproducibility

`set_seed(seed)` seeds, in order:

1. `PYTHONHASHSEED` (env var),
2. the stdlib `random` module,
3. NumPy's global RNG,
4. PyTorch CPU + all CUDA RNGs, **and** sets cuDNN to deterministic mode
   (`cudnn.deterministic = True`, `cudnn.benchmark = False`) — *only when torch
   is importable*; otherwise this step is skipped silently.

`Trainer.run()` calls `set_seed(config.seed)` as its very first step, so every
run is deterministic given a fixed config and environment. The applied seed is
echoed into the model card and the tracker params.

---

## Experiment tracking (MLflow optional)

`get_tracker(kind, name)` returns a backend implementing a single interface:

```python
tracker.start_run()
tracker.log_params({...})
tracker.log_metrics({...})
tracker.end_run()          # also usable as a context manager
```

- **`MLflowTracker`** — selected automatically when `kind="mlflow"` *and* the
  optional `mlflow` package is importable. Logs to the active MLflow store.
- **`JSONTracker`** — the zero-dependency fallback. Appends one JSON record per
  run (name, UTC timestamp, params, metrics, duration) to
  `runs/<name>/metrics.json` under the project root, preserving full history.

If `tracker: mlflow` is requested but MLflow is **not** installed, the factory
transparently falls back to the JSON backend — the harness never hard-fails on a
missing optional dependency.

---

## Model cards

After evaluation, `Trainer.emit_model_card(...)` writes `model_card.json` into
the run's `output_dir`. It always contains at least:

`model_name`, `trainer`, `created_at`, `seed`, `data_path`, `output_dir`,
`hyperparams`, `tracker`, `metrics`, `artifact_path`, and an `environment` block
(Python version, platform, `mlflow_available`, `numpy_available`).

---

## How to run

List registered trainers:

```bash
python -m uais.training.cli --list
```

Validate a config without training (resolves the trainer, echoes the parsed
config):

```bash
python -m uais.training.cli --config configs/training/isolation_forest.yaml --dry-run
```

Run a full training job:

```bash
python -m uais.training.cli --config configs/training/isolation_forest.yaml
```

Override seed or tracker from the command line:

```bash
python -m uais.training.cli -c configs/training/lof.yaml --seed 7 --tracker json
```

`python -m uais.training` is equivalent to `python -m uais.training.cli`.

### Programmatic use

```python
from uais.training import TrainConfig, get_trainer

config = TrainConfig.from_yaml("configs/training/isolation_forest.yaml")
trainer = get_trainer(config.trainer_key, config)
summary = trainer.run()      # seeds, fits, evaluates, logs, saves, emits card
print(summary["metrics"], summary["model_card"]["card_path"])
```

---

## Adding a new trainer

1. Implement an adapter subclassing `Trainer` and decorate it with
   `@register("my_model")`. Inside `fit()`, **import and call** the existing
   `train_*` function — do not copy its logic. Import any heavy backend lazily
   inside the method so package import stays dependency-light.
2. Add a `configs/training/my_model.yaml`.
3. (Optional) extend `tests/test_training_harness.py`.

```python
from uais.training.registry import register
from uais.training.trainer import Trainer

@register("my_model")
class MyTrainer(Trainer):
    def build(self): ...
    def fit(self):
        from uais.my_domain.train_my_model import train_my_model  # lazy import
        ...
    def evaluate(self): ...
```
