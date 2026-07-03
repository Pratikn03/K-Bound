# Engineering & Contributor Guide

This is the single document a new contributor reads before working on this
repository. It covers the project layout, environment setup, the test / lint /
pre-commit / smoke loops, full reproduction of the K-Bound paper, the
orchestration runner, the model registry and manifest, the unified training
harness, the production API (including the KGA `/decide` endpoint), the CI
overview, and the integrity / reproducibility guarantees.

The repository hosts two intertwined things:

1. **A research system — K-Bound / KGA (Knowability-Guided Adaptation).** The
   theory, proofs, experiments, and paper for a label-free decision rule that
   decides whether to **ADAPT**, **FREEZE**, or **ABSTAIN** on a test
   distribution, with a certified bound on the probability of a *harmful*
   adaptation.
2. **A production-grade anomaly-intelligence toolkit — UAIS** (`src/uais`,
   `deploy/`), the engineering substrate the research is instantiated in: data
   loaders, anomaly/fusion models, a FastAPI service, orchestration, a model
   registry, and a training harness.

---

## 1. Project layout

| Path | What lives there |
| --- | --- |
| `kga/` | **Top-level** Knowability-Guided Adaptation package. Pure `numpy`/`scipy`, deterministic, torch-free. The importable, tested implementation of the K-Bound decision rule (`KGA`, `Decision`, `Certificate`, `Evidence`). CLI: `python -m kga decide`. |
| `src/uais/` | The UAIS library (installed package `universal-anomaly-intelligence`): `anomaly/`, `fusion/`, `data/`, `nlp/`, `vision/`, `generative/`, plus the new `registry/` and `training/` subpackages and `utils/`. |
| `src/orchestration/` | Prefect flows (`fraud`, `cyber`, `behavior`, `fusion`, `nlp`, `vision`) with an import shim so they run with or without Prefect. CLI: `python -m src.orchestration.runner`. |
| `src/scripts/` | Standalone runnable scripts, including `src/scripts/kbound/` (the K-Bound experiments and the hermetic smoke). |
| `experiments/` | Experiment inputs/outputs. `experiments/kbound/theory_validation/` holds the numerical theorem validators; `experiments/kbound/results/` and `experiments/elara_u/score_archive/` hold cached artifacts. |
| `deploy/` | Production deployment: `deploy/api/` (FastAPI app, auth, KGA routes), Dockerfile, compose. |
| `models/` | Serialized model artifacts, per-domain `MODEL_CARD.md` files, and `MANIFEST.json` (the integrity manifest). |
| `configs/` | YAML configs, including `configs/training/*.yaml` for the training harness. |
| `notebooks/` | `00_KBound_Reproduction.ipynb` … `09_Conclusions_and_Reproducibility.ipynb` walk through the paper end-to-end. `notebooks/legacy_elara/` are the older domain EDA/modeling notebooks. |
| `docs/research/kbound/` | The **paper**: `kbound.tex` → `K-Bound_paper.pdf`, figures, references, and the asset/code maps. |
| `docs/` | This guide, runbooks, data docs (`docs/research/DATASET_USE_MATRIX.md`), and the proposed CI (`_ci_proposed.yml`). |
| `tests/` | `pytest` suite (theory/certificate unit tests plus the new package tests). |
| `scripts/` | Repo-level shell entrypoints: `rebuild_kbound.sh` (full repro), `smoke_kbound.sh` (hermetic smoke), and the data/training run scripts. |

> Note: macOS AppleDouble sidecar files (`._*`, `.__*`) may appear next to real
> files on some checkouts of the data volume. They are not source and are
> ignored by tooling.

---

## 2. Environment setup

Python **3.11** is the supported CI interpreter (the code targets 3.9+).

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2a. Full toolkit (UAIS library + dev tooling)
pip install -e .                 # installs the `uais` package from src/
pip install -r requirements-dev.txt

# 2b. OR a minimal env for just the kga package / theory work
pip install numpy scipy pytest
```

`PYTHONPATH` conventions (CI sets both):

- **`kga`** is a *top-level* package — put the **repo root** on `PYTHONPATH`
  (or `pip install -e .` once `kga` is added to packaging — see the hardening
  report).
- **`uais.*`** and **`orchestration`** live under `src/` — put **`src`** on
  `PYTHONPATH` (`export PYTHONPATH="$PWD:$PWD/src"`).

The heavy / optional dependencies (PyTorch, TensorFlow, MLflow, Prefect) are
**never required at import time**. Each is imported lazily only when the code
path that needs it actually runs, and degrades gracefully when absent.

---

## 3. The `kga` package (the K-Bound decision rule)

`kga` turns label-free evidence and a finite-sample certificate into a decision:

```
ADAPT    if  delta_hat - epsilon > 0     (adapting is provably beneficial)
FREEZE   if  delta_hat + epsilon < 0     (adapting is provably harmful)
ABSTAIN  otherwise                       (the sign of the benefit is not knowable)
```

with the probability of a *false adapt* bounded by `alpha` (Theorem 3 of the
paper). Public surface:

| Symbol | Role |
| --- | --- |
| `kga.KGA` | Facade gate: evidence → certify → decide → explain. |
| `kga.Decision` | `ADAPT` / `FREEZE` / `ABSTAIN` enum (a `str` enum, JSON-clean). |
| `kga.Certificate` | A `delta_hat ± epsilon` certificate at level `alpha`. |
| `kga.Evidence` | The label-free evidence `Z` container (KS drift, disagreement, entropy/confidence shift, ESS). |

Certificate estimators (`kga.certificate`): `empirical_bernstein` (`ebern`),
`hoeffding`, `conformal_split` (`conformal`), `evalue_anytime` (`evalue`).

**CLI:**

```bash
python -m kga decide --calib calib.npy --test test.npy --alpha 0.1
```

It loads calibration and test detector scores from `.npy` files, computes the
evidence `Z`, and prints the decision and certificate as JSON. Because labels
are unavailable at the CLI, it reports a conservative `delta_hat = 0` with a
split-conformal radius derived from the calibration dispersion, exercising the
full evidence → certificate → decision pipeline deterministically.

---

## 4. Tests

The suite is `pytest`-based and lives in `tests/`.

| Group | Files | Deps |
| --- | --- | --- |
| **KGA package** | `tests/test_kga_package.py` | `numpy`, `scipy` |
| **Model registry** | `tests/test_model_registry.py` | stdlib (+`pyyaml` optionally) |
| **Training harness** | `tests/test_training_harness.py` | `numpy`, `scikit-learn`, `pandas`, `pyyaml`, `joblib` |
| **Data manifest** | `tests/test_data_manifest.py` | `numpy` |
| **Hermetic smoke** | `tests/test_smoke_trichotomy.py` | `numpy`, `scipy` |
| **Theory / certificates** | the broader `tests/test_*` suite (33/33 theorem & certificate unit tests) | `numpy`, `scipy`, `scikit-learn` |

Common invocations:

```bash
# Minimal, fast: the kga package only
PYTHONPATH="$PWD" pytest tests/test_kga_package.py -q

# The new packages, with coverage on kga and src/uais
PYTHONPATH="$PWD:$PWD/src" pytest \
  tests/test_kga_package.py tests/test_model_registry.py \
  tests/test_training_harness.py tests/test_data_manifest.py \
  tests/test_smoke_trichotomy.py \
  --cov=kga --cov=src/uais --cov-report=term-missing

# Everything (the existing full suite installs torch CPU; see CI)
PYTHONPATH="$PWD:$PWD/src" pytest tests -q
```

The `kga` tests are deterministic and fixed-seed: identical distributions →
ABSTAIN; the certificate radius shrinks as `n` grows; the trichotomy boundaries
are hit exactly; a non-identifiability witness (identical `Z`, opposite truth) →
ABSTAIN; and the empirical false-adapt rate stays `<= alpha` over ≥ 2000
synthetic trials.

---

## 5. Lint, format, and typing

Ruff is the single source of truth for both linting and formatting. The
configuration is in `pyproject.toml` under `[tool.ruff]` (line-length **120**,
rule set **E, W, F, I, C, B, UP**; `E501` is delegated to the formatter). The
legacy `[tool.black]` / `[tool.isort]` sections remain compatible (also
line-length 120, isort `profile = "black"`). `mypy` config is also in
`pyproject.toml`.

```bash
ruff check .              # lint
ruff check . --fix        # lint + autofix
ruff format .             # format
ruff format --check .     # CI format gate
mypy kga                  # type-check the pure-numpy kga package
```

---

## 6. Pre-commit

`.pre-commit-config.yaml` wires the same tooling into a git hook:

```bash
pip install pre-commit
pre-commit install            # enable on commit
pre-commit run --all-files    # run over the whole tree once
```

Hooks: **ruff** (lint, `--fix`) and **ruff-format**; the standard hygiene set
(`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`,
`check-added-large-files` at `maxkb=2048`, `check-merge-conflict`,
`debug-statements`); and **mypy** restricted to `kga/` (`--ignore-missing-imports`,
`additional_dependencies: [numpy]`). The hooks inherit the line-length-120 ruff
config from `pyproject.toml`, so local and CI results agree.

---

## 7. Hermetic smoke path

The hermetic smoke is the fastest end-to-end signal that the K-Bound machinery
works: **zero external data, < 60 s**, exercising the `kga` trichotomy on
synthetic inputs.

```bash
bash scripts/smoke_kbound.sh        # -> src/scripts/kbound/smoke_trichotomy.py
```

It validates the canonical behaviours (informative drift → ADAPT/FREEZE,
non-identifiable witness → ABSTAIN) without touching the model artifacts, GPU,
or any dataset. It is covered by `tests/test_smoke_trichotomy.py` and runs as the
`hermetic-smoke` CI job. The companion data manifest at
`src/uais/data/manifest.py` (covered by `tests/test_data_manifest.py`) records
dataset provenance for the non-hermetic experiments.

---

## 8. Full reproduction of the K-Bound paper

The complete, paper-grade rebuild is a single script:

```bash
bash scripts/rebuild_kbound.sh                  # CPU experiments + compile PDF
KBOUND_GPU=1 bash scripts/rebuild_kbound.sh     # also CIFAR-10 + Tent on GPU
KBOUND_SKIP_DEPS=1 bash scripts/rebuild_kbound.sh
PYTHON=.venv/bin/python bash scripts/rebuild_kbound.sh
```

It runs (from repo root): the knowability, harmful-regime, mixed-regime, and
full (`rigor` / `ablation` / `regression` / `witness`) experiments under
`src/scripts/kbound/`; optionally the CIFAR-10 + Tent GPU experiment; and then
compiles `docs/research/kbound/kbound.tex` twice into `K-Bound_paper.pdf`.

Inputs/outputs map as follows (see `docs/research/kbound/PACKAGE_STRUCTURE.md`):

| Kind | Location |
| --- | --- |
| Experiment code | `src/scripts/kbound/` |
| Cached inputs | `experiments/elara_u/score_archive/` (shared with ELARA) |
| Experiment outputs | `experiments/kbound/results/` (+ `experiments/kbound/cifar/`) |
| Figures | `docs/research/kbound/figures/` |
| Paper source / PDF | `docs/research/kbound/kbound.tex` → `K-Bound_paper.pdf` |

### Theorem validators

Each theorem in the paper has a standalone numerical validator (pure
`numpy`/`scipy`, no labels, no GPU) under
`experiments/kbound/theory_validation/`:

| Script | Validates |
| --- | --- |
| `val_thm1_lecam.py` | Theorem 1 — Le Cam two-point minimax lower bound on label-free committal error. |
| `val_thm2_regret.py` | Theorem 2 — regret bound. |
| `val_thm3_evalue.py` | Theorem 3 — anytime-valid e-value / false-adapt certificate. |
| `val_thm5_multiclass.py` | Theorem 5 — multiclass extension. |
| `val_thm9prime_drift.py`, `val_rademacher_router.py`, `val_conj1_caltransfer.py` | additional drift / router / calibration-transfer results. |

Run one directly, e.g. `python experiments/kbound/theory_validation/val_thm1_lecam.py`;
the `theorem-validators` CI job runs every `val_thm*.py` and fails if any exits
non-zero.

---

## 9. Orchestration (the runner)

`src/orchestration/` exposes six Prefect flows. The package imports cleanly
**with or without Prefect**: `src/orchestration/_compat.py` degrades the
`@flow`/`@task` decorators to no-ops when Prefect is absent, and
`src.orchestration.PREFECT` reports whether the real package is active. Flows are
resolved lazily, so listing them never pulls in heavy backends (e.g. TensorFlow
for vision).

```bash
# List the registry (imports nothing heavy)
python -m src.orchestration.runner list

# Run a flow; forward --key value params to the flow signature
python -m src.orchestration.runner fraud
python -m src.orchestration.runner nlp --max_samples 2000 --max_features 4000
python -m src.orchestration.runner vision --epochs 2 --image_size 224
python -m src.orchestration.runner fusion --run_attention_validation true

# The import-shim contract (works with no prefect installed):
python -c "import src.orchestration as o; print(o.PREFECT, sorted(o.FLOWS.keys()))"
```

`FLOWS` is a lazy, dict-like registry (`FLOWS["fraud"]` → the fraud flow
callable; `FLOWS.keys()` → the flow names).

---

## 10. Model registry, model cards, and the manifest

`src/uais/registry/` provides integrity-checked model loading:

- `ModelRegistry` — manifest-backed registry that resolves an artifact's path
  and **fails closed** on a SHA-256 mismatch (`IntegrityError`) rather than
  returning a possibly-tampered file. Per-model digests can be overridden via
  `UAIS_MODEL_SHA256_<NAME>`, mirroring the deployment API.
- `ModelCard` — structured, honest metadata (task, framework, training data,
  intended use, metrics, limitations) that mirrors the on-disk
  `models/<domain>/MODEL_CARD.md` files so the two never drift.

The manifest at `models/MANIFEST.json` records, per artifact, its
repo-relative `path`, `sha256`, `size_bytes`, and `mtime`. Regenerate it (it
hashes every `*.pkl`/`*.pt`/`*.joblib` under `models/`):

```bash
PYTHONPATH=src python -m uais.registry.build_manifest
# or, with explicit paths / sentinel hashes:
PYTHONPATH=src python -m uais.registry.build_manifest --repo-root . --output models/MANIFEST.json
PYTHONPATH=src python -m uais.registry.build_manifest --no-hashes
```

`ModelRegistry.verify_all()` returns a structured per-artifact report
(`ok | mismatch | missing | pending`) for CI gates and runbooks. Covered by
`tests/test_model_registry.py`.

---

## 11. Unified training harness

`src/uais/training/` standardises seeding, experiment tracking, artifact
persistence, and model-card emission *around* the existing `train_*.py`
entrypoints — it never re-implements model logic.

| Piece | Role |
| --- | --- |
| `set_seed(seed)` | Seeds `PYTHONHASHSEED`, `random`, NumPy, and (if importable) PyTorch + cuDNN deterministic mode. |
| `TrainConfig` | Validated, YAML-loadable run config (`from_yaml` / `to_yaml`). |
| `Trainer` | Abstract lifecycle: `set_seed → build → fit → evaluate → log_metrics → save_artifact → emit_model_card`. |
| `registry` | `register` / `get_trainer` / `available_trainers`; adapters for `isolation_forest`, `lof`, `ocsvm`, `autoencoder`, `fusion_meta`, `vae`. |
| `tracking` | `get_tracker` returns an MLflow tracker when available, else a JSON tracker (`mlflow_available()`). |

```bash
python -m uais.training.cli --list
python -m uais.training.cli --config configs/training/isolation_forest.yaml
python -m uais.training.cli -c configs/training/lof.yaml --dry-run
python -m uais.training.cli -c configs/training/vae.yaml --seed 7 --tracker json
```

Configs live in `configs/training/*.yaml`. Heavy backends (TF for `vae`) are
imported only when that trainer actually runs. Covered by
`tests/test_training_harness.py`.

---

## 12. The production API (and the KGA `/decide` endpoint)

The FastAPI app is `deploy/api/main.py` (`UAIS-V Enhanced API`). It ships
rate-limiting, request-timeout, request-logging, and CORS middleware, optional
metrics, API-key auth, and artifact integrity checks. The KGA routes are mounted
into the same app (`app.include_router(kga_router)`):

| Method & path | Auth | Purpose |
| --- | --- | --- |
| `POST /decide` | yes | Given `calib_scores`, `test_scores`, and optional `alpha`, returns the ADAPT/FREEZE/ABSTAIN decision, the certificate (`delta_hat`, `epsilon`, `method`), and the label-free evidence `Z`. |
| `GET /kga/health` | no | Liveness probe for the KGA subsystem (returns the `kga` version). |

The decision math lives entirely in the importable `kga` package; the routes
(`deploy/api/kga_routes.py`) are a thin, validated transport layer (request
sizes are bounded; non-finite inputs are rejected). As at the CLI, labels are
unavailable at request time, so the benefit estimate is reported conservatively
(`delta_hat = 0`) with a split-conformal radius from the calibration dispersion.

Local run and operations details are in `docs/production/PRODUCTION_RUNBOOK.md`
and `docs/PRODUCTION_API_RUNBOOK.md`.

---

## 13. CI overview

Two workflows cover the repo:

- **`.github/workflows/ci.yml` (existing).** The heavyweight gate: a ruff
  syntax/undefined-name lint pass, the full `pytest` suite with coverage
  (installs `requirements-api.txt` plus pandas/torch CPU etc.), a fusion smoke,
  best-effort orchestration smokes, and a `deploy-security` job (API security
  regression tests, Bandit scan, Docker build, and a production-env API smoke).
- **`docs/_ci_proposed.yml` (proposed, lead installs into
  `.github/workflows/`).** A fast, layered, data-free / torch-free matrix on
  `ubuntu-latest` / Python 3.11, triggered on push + PR, with
  `concurrency.cancel-in-progress` to drop superseded runs:

| Job | What it does |
| --- | --- |
| `lint` | `ruff check .` then `ruff format --check .`. |
| `kga-package` | `pip install numpy scipy pytest`; import `kga` + print version; `pytest tests/test_kga_package.py`. |
| `unit-tests` | minimal deps; the five new test modules with `--cov=kga --cov=src/uais`; uploads `coverage.xml`. |
| `hermetic-smoke` | `pip install numpy scipy`; `bash scripts/smoke_kbound.sh`. |
| `theorem-validators` | runs every `experiments/kbound/theory_validation/val_thm*.py`; fails on any non-zero exit. |
| `orchestration-import` | proves `import src.orchestration` and the flow registry work **without** Prefect installed. |

---

## 14. Integrity & reproducibility guarantees

The repo is engineered so results are reproducible and artifacts are
tamper-evident:

- **Deterministic core.** `kga` is pure `numpy`/`scipy` with no global state;
  tests are fixed-seed. `set_seed` seeds every RNG (stdlib, NumPy, torch) and
  requests deterministic cuDNN.
- **Certified decisions.** The trichotomy fires ADAPT only when the certified
  lower bound is positive, bounding the false-adapt probability by `alpha`
  (Theorem 3); the anytime e-value variant holds simultaneously over all sample
  sizes (Theorem 3b). Each theorem has an executable validator.
- **Artifact integrity.** Models are addressed by SHA-256 in
  `models/MANIFEST.json`; loading fails closed on any mismatch, and digests can
  be pinned via environment overrides.
- **Honest metadata.** `ModelCard` and the `MODEL_CARD.md` files record metrics
  as actually measured (or `"not recorded in repo"`), never invented.
- **One-command rebuilds.** `scripts/rebuild_kbound.sh` regenerates every
  experiment, figure, and the PDF from source; `scripts/smoke_kbound.sh` gives a
  < 60 s hermetic check; `notebooks/00_…_09` reproduce the paper interactively.
- **Layered, hermetic CI.** The proposed workflow installs only what each job
  needs and touches no external data, so a green run means the theory, the
  package, the smoke, and the orchestration import all hold on a clean machine.

For data provenance and dataset-use policy see `docs/research/DATASET_USE_MATRIX.md`
and the data docs; for the research narrative see `docs/research/kbound/`.
