#!/usr/bin/env bash
# run_g5_finalize.sh — finish WIN_HUNT_v4 arm D (official POEM/AETTA head-to-head).
# Runs per-sample stress seeds 2-4, then scores ONCE at seeds 0-4 (the official verdict).
# Guards: interpreter import-check up front; never scores a partial seed set.
# Run from the repository root on the Mac:  bash docs/research/kbound/g5_finalize/run_g5_finalize.sh
set -euo pipefail

REPO="$(pwd)"
# Interpreter: same env as RUNSHEET_WAVE7 PHASE 1 (repo venv). Override with PY_CORE=... if needed.
if [ -z "${PY_CORE:-}" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PY_CORE="$REPO/.venv/bin/python"; else PY_CORE="python3"; fi
fi
RUN_DIR="experiments/kbound/results/stress_persample_v1"
RUNNER="docs/research/kbound/scripts/cifar_tent_mps_v2.py"
SCORER="experiments/kbound/poem_aetta/score_official_headtohead.py"

[ -f "$RUNNER" ] || { echo "ERROR: $RUNNER not found (run from repo root)"; exit 1; }
[ -f "$SCORER" ] || { echo "ERROR: $SCORER not found"; exit 1; }

echo "== Interpreter preflight: $PY_CORE =="
"$PY_CORE" -c "import numpy, torch, torchvision; import sys; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| mps:', torch.backends.mps.is_available())" || {
  echo "ERROR: $PY_CORE cannot import numpy/torch/torchvision."
  echo "Set the interpreter explicitly, e.g.:  PY_CORE=\"$REPO/.venv/bin/python\" bash $0"
  exit 1
}

echo "== G5 finalize: per-sample runs for missing seeds =="
for S in 0 1 2 3 4; do
  PC="$RUN_DIR/per_condition_cifar10c_tent_seed${S}.json"
  if [ -f "$PC" ]; then
    echo "  seed $S: per_condition present — skipping run"
  else
    echo "  seed $S: running (this regenerates any stale partial npz for the seed)…"
    caffeinate -is "$PY_CORE" "$RUNNER" \
      --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
      --methods tent --device mps --seed "$S" --log-samples \
      --out-results "$RUN_DIR"
  fi
done

echo "== Guard: verifying all five per_condition JSONs exist =="
MISSING=0
for S in 0 1 2 3 4; do
  [ -f "$RUN_DIR/per_condition_cifar10c_tent_seed${S}.json" ] || { echo "  MISSING seed $S"; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "ERROR: incomplete seed set — NOT scoring (official verdict is scored once at 0-4)"; exit 2; }

echo "== Scoring ONCE at seeds 0-4 (official arm-D verdict) =="
"$PY_CORE" "$SCORER" \
  --run-dir "$RUN_DIR" \
  --dataset cifar10c --adapter tent --seeds 0 1 2 3 4 --nboot 10000

echo "== Done. Verdict written to research_lock/WIN_HUNT_v4_ARM_D_result.json =="
echo "Next: update WIN_HUNT_v4_ARM_D_STATUS.json, then fold into the paper per the"
echo "pre-committed outcome handling in docs/research/kbound/g5_finalize/G5_STATUS_AND_FINALIZE.md"
