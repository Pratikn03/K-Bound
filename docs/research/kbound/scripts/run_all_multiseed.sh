#!/usr/bin/env bash
# Reproduce every multi-seed no-harm result from the committed INTERNAL per-condition logs.
# No T9 drive, no GPU, no raw images. Writes locked artifacts to experiments/kbound/results/multiseed/.
#
# Coverage (the only 3 of 9 tracks with multi-seed per-condition logs on disk):
#   CIFAR-10-C   stress_grid_multiseed_v1   5 seeds x 432 conditions   (seeds 1-4: a_kbound/a_oracle
#                                                                        reconstructed, exact-validated)
#   CIFAR-10.1   cifar101_multiseed_v1      5 seeds x  24 conditions
#   Camelyon17   natural_win_v2_camelyon    4 seeds x  36 conditions
# iWildCam / Office-Home / RxRx1 / PACS have only single-run per-condition logs; ImageNet-R's per-seed
# logs are debug-scale (n=3). Those need the seed-0..4 GPU re-run (scripts/run_multiseed.sh).
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

set -euo pipefail
ROOT="${ROOT:-$KB_REPO_ROOT}"
PY="${PY:-$HOME/.venv_wilds/bin/python}"; [ -x "$PY" ] || PY=python3
SC="$ROOT/docs/research/kbound/scripts/multiseed_natural.py"
R="$ROOT/experiments/kbound/results"; OUT="$R/multiseed"; mkdir -p "$OUT"
run(){ "$PY" "$SC" --dataset "$1" --candidate "$2" --dir "$3" --out "$OUT/multiseed_${1}_${2}.json" \
        | grep -E '&' ; }
echo "python=$PY"
for c in tent eata sar; do run cifar10c   "$c" "$R/stress_grid_multiseed_v1"; done
for c in tent eata sar; do run cifar101   "$c" "$R/cifar101_multiseed_v1";    done
for c in tent eata sar; do run camelyon17 "$c" "$R/natural_win_v2_camelyon";  done
echo "multi-seed artifacts written to $OUT"
