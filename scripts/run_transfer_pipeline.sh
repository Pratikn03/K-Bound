#!/usr/bin/env bash
# Transfer pipeline v1 — run without pasting shell comments into zsh.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python

case "${1:-run}" in
  dry-run)
    "$PY" src/scripts/scenario_c/run_transfer_development_pipeline.py --dry-run
    ;;
  m2-only)
    "$PY" src/scripts/scenario_c/run_transfer_development_pipeline.py --m2-only
    ;;
  calibrators)
    "$PY" src/scripts/scenario_c/freeze_domain_calibrators.py
    ;;
  eyecandies)
    "$PY" src/scripts/run_breakthrough_experiment.py \
      --config configs/attention_eyecandies_transfer_dev_v1.yaml \
      --output experiments/fusion/eyecandies_transfer_dev_v1_seed42.json \
      --seed 42
    ;;
  experts)
    "$PY" src/scripts/scenario_c/upgrade_mvtec_experts.py
    ;;
  m2)
    "$PY" src/scripts/scenario_c/run_m2_external_confirmatory.py --transfer-v1
    ;;
  run|*)
    "$PY" src/scripts/scenario_c/run_transfer_development_pipeline.py
    ;;
esac
