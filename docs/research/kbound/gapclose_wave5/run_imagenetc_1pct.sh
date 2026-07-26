#!/bin/bash
# ImageNet-C v5 — 1% subsample on MPS (GPU). Full 27-cell grid (3 noise x 3 sev x 3 comp x 3 methods),
# but ~50 images/cell instead of the full pool -> cells finish in seconds.
# This is a REDUCED-SCALE / directional run: it proves MPS works and gives a directional signal.
# It is NOT the frozen full-scale ImageNet-C number and must be reported as a 1% subsample if used.
# --- external (git-excluded) data volume: ONE documented variable, no default.
# --- defect D8: portable roots. No machine-local absolute paths in tracked code
# --- (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md). KB_REPO_ROOT is discovered
# --- from this script's own location; override with KBOUND_REPO_ROOT.
_kb_find_root() {
  d=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
  while [ "$d" != "/" ]; do
    [ -f "$d/pyproject.toml" ] && { printf '%s\n' "$d"; return 0; }
    d=$(dirname "$d")
  done
  echo "ERROR: repository root not found above $(dirname "${BASH_SOURCE[0]:-$0}")" >&2
  return 1
}
KB_REPO_ROOT="${KBOUND_REPO_ROOT:-$(_kb_find_root)}" || exit 1

: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets/checkpoints/caches (layout: docs/research/kbound/kbound_repro/paths.py, acquisition: DATA.md)}"
KB_EXTERNAL_ROOT="$KBOUND_EXTERNAL_ROOT"

set -u
cd "$KB_REPO_ROOT" || exit 1
source "${KBOUND_VENV:-$HOME/.venv_wilds}/bin/activate"   # override with KBOUND_VENV
export TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"

echo "== kill any lingering imagenet-c run =="
pkill -9 -f cifar_tent_mps_v2 2>/dev/null || true
sleep 2

OUT=experiments/kbound/results/win_hunt_v5/imagenetc_aggr_1pct
rm -rf "$OUT"

echo "== 1% MPS run: 3 noise x 3 sev x 3 comp x 3 methods, ~50 imgs/cell, caffeinated =="
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$HOME/imagenetc_local" --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --max-images 50 --out-results "$OUT"
