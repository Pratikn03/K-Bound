#!/usr/bin/env bash
# =============================================================================
# rebuild_kbound.sh  —  Rebuilds the K-BOUND paper (the main paper).
#
# K-Bound now lives in the repo tree:
#   code     -> src/scripts/kbound/
#   outputs  -> experiments/kbound/results/   (+ experiments/kbound/cifar/)
#   figures  -> docs/research/kbound/figures/
#   paper    -> docs/research/kbound/kbound.tex  -> K-Bound_paper.pdf
#   inputs   -> experiments/elara_u/score_archive  (canonical, shared with ELARA)
#
# ELARA's own paper builds remain separate: scripts/rebuild_paper.sh, rebuild_elara_u_paper.sh
#
# Usage:
#   bash scripts/rebuild_kbound.sh                 # CPU experiments + compile
#   KBOUND_GPU=1 bash scripts/rebuild_kbound.sh    # also CIFAR-10 + Tent on GPU
#   KBOUND_SKIP_DEPS=1 bash scripts/rebuild_kbound.sh
#   PYTHON=.venv/bin/python bash scripts/rebuild_kbound.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."                    # -> repo root (AutoML_Flagship_V8/)
PY="${PYTHON:-python3}"
echo "K-Bound rebuild — interpreter: $PY  root: $(pwd)"

if [ "${KBOUND_SKIP_DEPS:-0}" != "1" ]; then
  echo "[1/4] deps"; $PY -m pip install -q -r docs/research/kbound/requirements.txt || \
    echo "  (pip skipped/failed — continuing if deps present)"
else echo "[1/4] skip deps"; fi

echo "[2/4] core experiments (CPU)"
$PY src/scripts/kbound/knowability_experiment.py
$PY src/scripts/kbound/kbound_harmful_regime.py
$PY src/scripts/kbound/mixed_regime_experiment.py
$PY src/scripts/kbound/kbound_full_experiments.py rigor
$PY src/scripts/kbound/kbound_full_experiments.py ablation
$PY src/scripts/kbound/kbound_full_experiments.py regression
$PY src/scripts/kbound/kbound_full_experiments.py witness

if [ "${KBOUND_GPU:-0}" = "1" ]; then
  echo "[3/4] GPU experiment (CIFAR-10 + Tent)"; $PY src/scripts/kbound/cifar_tent_mps.py
else echo "[3/4] skip GPU (set KBOUND_GPU=1)"; fi

echo "[4/4] compile paper"
( cd docs/research/kbound \
  && pdflatex -interaction=nonstopmode kbound.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode kbound.tex >/dev/null 2>&1 \
  && cp -f kbound.pdf K-Bound_paper.pdf )
echo "DONE -> docs/research/kbound/K-Bound_paper.pdf"
