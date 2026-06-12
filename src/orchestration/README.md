# Orchestration (`src.orchestration`)

Production-grade pipeline orchestration for the UAIS-V / K-Bound anomaly-intelligence
system, built on [Prefect](https://docs.prefect.io). Each domain pipeline is a real
Prefect **flow** composed of retried **tasks** that wrap the existing experiment
scripts *without changing their computation*.

Prefect is an **optional dependency**: a compatibility shim (`_compat.py`) degrades
`@flow` / `@task` / `get_run_logger` to no-ops when Prefect is not installed, so every
flow stays importable and runnable as a plain function.

## Flows

| Name       | Module             | Callable             | Wraps                                                                 |
|------------|--------------------|----------------------|----------------------------------------------------------------------|
| `fraud`    | `fraud_flow.py`    | `fraud_pipeline`     | `src.scripts.run_fraud_experiment.main`                              |
| `cyber`    | `cyber_flow.py`    | `cyber_pipeline`     | `src.scripts.run_cyber_experiment.main`                             |
| `behavior` | `behavior_flow.py` | `behavior_pipeline`  | `src.scripts.run_behavior_experiment.main`                          |
| `fusion`   | `fusion_flow.py`   | `fusion_pipeline`    | `run_fusion_experiment` + optional attention validation / harness   |
| `nlp`      | `nlp_flow.py`      | `nlp_pipeline`       | `uais.nlp.train_text_classifier.run_text_experiment`               |
| `vision`   | `vision_flow.py`   | `vision_pipeline`    | `uais.vision.train_vision_model.run_vision_experiment`             |

Every flow returns a JSON-serializable result `dict` containing at least
`domain`, `status` (`"completed"` / `"failed"`), and `prefect` (whether the real
Prefect runtime is active). The `nlp` / `vision` flows additionally return the
experiment `metrics`; `fusion` reports whether the optional attention steps ran.

## Running a flow locally

### With the CLI runner

```bash
# List the registered flows
python -m src.orchestration.runner list

# Run a flow (parameters are forwarded as --key value pairs)
python -m src.orchestration.runner fraud
python -m src.orchestration.runner nlp --max_samples 2000 --max_features 4000
python -m src.orchestration.runner vision --epochs 2 --image_size 224
python -m src.orchestration.runner fusion --run_attention_validation true
```

CLI values are coerced automatically: `true`/`false` → `bool`, `none`/`null` →
`None`, integers and floats are parsed, everything else stays a string. A bare
`--flag` (no following value) is treated as `True`. The runner prints the flow's
result dict as indented JSON.

### From Python

```python
from src.orchestration import fraud_pipeline, FLOWS, PREFECT

result = fraud_pipeline()              # direct call
result = FLOWS["nlp"](max_samples=2000)  # via the registry
print("Prefect active:", PREFECT)
```

### Without Prefect installed

Nothing changes for local use. The shim makes `@flow` / `@task` no-ops and
`get_run_logger()` returns a standard `logging.getLogger("orchestration")`, so
the commands above run identically — just without the Prefect UI, retries, or
caching. `PREFECT` will be `False`.

### With Prefect installed

```bash
pip install "prefect>=2.14"
```

The decorators become the *real* Prefect ones. Flow and task runs are tracked,
retries and caching take effect, and `python -m src.orchestration.runner <flow>`
executes a genuine Prefect flow run. To use the Prefect UI / API server:

```bash
prefect server start          # local Orion server + UI at http://127.0.0.1:4200
# (in another shell) then run any flow as above
```

## What the orchestration layer adds

* **Retries** — every task is decorated with `@task(retries=2,
  retry_delay_seconds=5, log_prints=True)`, so a transient failure (I/O, flaky
  download, OOM hiccup) is retried twice with a 5-second backoff before the run
  is marked failed.
* **Structured logging** — tasks log start/finish and metrics through
  `get_run_logger()`. Under real Prefect these become per-run logs surfaced in
  the UI; under the shim they go to the `orchestration` stdlib logger.
* **`log_prints=True`** — `print(...)` statements inside the wrapped experiment
  scripts are captured as Prefect logs rather than lost to stdout.
* **Idempotent failure handling** — heavy / optional imports (TensorFlow for
  vision, scikit-learn for NLP, the attention scripts for fusion) are performed
  *lazily inside the tasks* and guarded, so importing the package never pulls in
  the heavy stack and a missing optional dependency yields a logged
  `status="failed"` rather than a crash.
* **Typed flow parameters** — flows expose typed, defaulted parameters
  (`nlp_pipeline(dataset_path, text_column, label_column, max_samples,
  max_features)`, `vision_pipeline(dataset_dir, epochs, batch_size,
  image_size)`, `fusion_pipeline(run_attention_validation,
  run_attention_harness)`). Defaults reproduce the original behavior exactly,
  including the legacy `RUN_ATTENTION_VALIDATION` / `RUN_ATTENTION_HARNESS`
  environment-variable gating for the fusion attention steps.

### Caching note

`cache_key_fn` is intentionally **not** enabled on these tasks. Each task runs a
full experiment for side effects (it trains models and writes metrics / scores /
plots / model artifacts to disk) and depends on on-disk datasets and configs, so
it is not a pure, deterministic, return-value-cacheable step. Caching the return
value would skip those side effects on a cache hit, changing behavior. The
`_compat` shim and the `@task` signature nonetheless accept `cache_key_fn`, so a
genuinely pure sub-step added later can opt in with
`@task(..., cache_key_fn=task_input_hash)` without further changes.

## Mapping to a real Prefect deployment

Because the flows are already real Prefect flows, deploying them is configuration,
not code changes. A typical setup:

1. **Work pool** — create a process (or Docker / Kubernetes) work pool:

   ```bash
   prefect work-pool create uais-pool --type process
   ```

2. **Deployments** — deploy each flow to the pool. Using the `flow.deploy(...)`
   API (Prefect 2.14+):

   ```python
   from src.orchestration import fraud_pipeline, nlp_pipeline

   fraud_pipeline.deploy(
       name="fraud-nightly",
       work_pool_name="uais-pool",
       cron="0 2 * * *",            # 02:00 daily
   )
   nlp_pipeline.deploy(
       name="nlp-weekly",
       work_pool_name="uais-pool",
       cron="0 3 * * 1",            # 03:00 Mondays
       parameters={"max_samples": 5000},
   )
   ```

   (Equivalently, declare deployments in a `prefect.yaml` and run
   `prefect deploy`.)

3. **Schedules** — set via the `cron=` / `interval=` argument above, or attach an
   `RRule`/`Cron`/`Interval` schedule in the Prefect UI per deployment.

4. **Workers** — start a worker that polls the pool and executes scheduled or
   ad-hoc runs:

   ```bash
   prefect worker start --pool uais-pool
   ```

5. **Parameters & retries** — deployment `parameters=` map directly onto each
   flow's typed signature; the per-task retry / backoff policy defined in code is
   honored by every run. Increase resilience further per deployment with job
   variables (e.g. memory limits) on the work pool.

The registry in `runner.py` (`FLOWS`) is convenient for building these
deployments programmatically — iterate `FLOWS.keys()` and call `.deploy(...)` on
each resolved flow.
