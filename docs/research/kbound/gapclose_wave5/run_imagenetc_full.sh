#!/bin/bash
# ImageNet-C v5 FULL-scale run: 3 noise corruptions (protocol-E scope), full image pool.
# ~16 h on MPS. Hardens against every prior failure:
#   - makes the OUTPUT dir first (prior runs crashed when it was deleted mid-run)
#   - activates the venv (else "python: No such file or directory")
#   - caffeinate (else the Mac/drive sleep-stalls in uninterruptible I/O)
#   - checks the data exists before burning 16 h
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

IC="$HOME/imagenetc_local"
OUT="experiments/kbound/results/win_hunt_v5/imagenetc_aggr"
T9DATA="experiments/kbound/data/imagenet-c"

# 1) data must be present (we deleted copies during cleanup)
if [ ! -d "$IC/gaussian_noise" ]; then
  echo "[data] not at $IC -- trying to restore from T9 ($T9DATA) ..."
  if [ -d "$T9DATA/gaussian_noise" ]; then
    mkdir -p "$IC"
    cp -R "$T9DATA/gaussian_noise" "$T9DATA/shot_noise" "$T9DATA/impulse_noise" "$IC/"
    echo "[data] restored 3 noise corruptions to $IC"
  else
    echo "ERROR: no ImageNet-C noise data at $IC or $T9DATA."
    echo "Re-download the noise set from Zenodo (record 2235448, noise.tar), extract so you have"
    echo "  $IC/gaussian_noise/<sev>/<wnid>/*.JPEG  (also shot_noise, impulse_noise), then re-run."
    exit 1
  fi
fi

# 2) output dir MUST exist (this is what crashed the last run)
mkdir -p "$OUT"

# 3) full-scale run (no --max-images cap), caffeinated. Resumes from checkpoint if present.
echo "== FULL ImageNet-C: 3 noise x 3 sev x 3 comp x 3 methods, full image pool -> $OUT =="
echo "== leave it alone; ~16 h; do NOT delete $OUT while it runs =="
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$IC" --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --out-results "$OUT"
