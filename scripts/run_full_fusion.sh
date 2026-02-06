#!/usr/bin/env bash
set -euo pipefail

echo "[fraud] running supervised + anomaly experiment"
python src/scripts/run_fraud_experiment.py

echo "[cyber] running supervised + anomaly experiment"
python src/scripts/run_cyber_experiment.py

echo "[behavior] running autoencoder + LOF experiment"
python src/scripts/run_behavior_experiment.py

echo "[fusion] stacking domain scores"
python src/scripts/run_fusion_experiment.py

if [[ "${RUN_ATTENTION_VALIDATION:-0}" == "1" ]]; then
  echo "[fusion] validating attention inputs"
  python src/scripts/run_attention_validation.py
fi

if [[ "${RUN_ATTENTION_HARNESS:-0}" == "1" ]]; then
  echo "[fusion] running attention evaluation harness"
  python src/scripts/run_attention_harness.py
fi
