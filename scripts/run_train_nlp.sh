#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import pandas as pd

from uais.nlp.train_text_classifier import NLPConfig, run_text_experiment

processed = Path("data/processed/nlp/enron_emails.csv")
raw_fakenews = Path("data/raw/nlp/fakenews/fake_news_labeled.csv")
if processed.exists():
    dataset_path = processed
    text_column = "text"
elif raw_fakenews.exists():
    dataset_path = raw_fakenews
    text_column = "content"
else:
    raise FileNotFoundError(
        "No standalone NLP dataset found. Expected data/processed/nlp/enron_emails.csv "
        "or data/raw/nlp/fakenews/fake_news_labeled.csv."
    )

cfg = NLPConfig(
    dataset_path=dataset_path,
    text_column=text_column,
    label_column="label",
    max_samples=10000,
)
metrics = run_text_experiment(cfg)

metrics_dir = Path("experiments/nlp/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)
(metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
pd.DataFrame([{"Metric": key, "Value": value} for key, value in metrics.items()]).to_csv(
    metrics_dir / "metrics.csv",
    index=False,
)
print("NLP dataset:", dataset_path)
print("NLP metrics:", metrics)
PY
