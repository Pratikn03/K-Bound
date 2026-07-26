#!/bin/bash
# ImageNet-C v5 aggressive arm. Set IC_ROOT to the folder that holds EITHER
#   (A) the 5 tars:  noise.tar blur.tar weather.tar digital.tar extra.tar   (streamed, no extract)
#   (B) flat corruption dirs:  gaussian_noise/<sev>/<wnid>/*.JPEG , motion_blur/<sev>/... etc.
# A category-nested layout ( <root>/noise/gaussian_noise/... ) will NOT be found — flatten it.
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

set -e
cd "$KB_REPO_ROOT"
source "${KBOUND_VENV:-$HOME/.venv_wilds}/bin/activate"   # override with KBOUND_VENV
export TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"   # pretrained resnet50 f0 cache

# >>> EDIT THIS ONE LINE <<<
IC_ROOT="$KB_REPO_ROOT"/experiments/kbound/data/imagenet-c

echo "IC_ROOT=$IC_ROOT"
echo "contents:"; ls -1 "$IC_ROOT" 2>/dev/null | grep -viE '^\._' | head

# clear any partial/quick-set output so the corrected run starts clean
rm -rf experiments/kbound/results/win_hunt_v5/imagenetc_aggr

# FULL standard-15 ImageNet-C. Deliberate expansion beyond protocol E's 3-noise baseline,
# chosen BEFORE the frozen scoring pass (adds ALL corruptions incl. unfavorable ones -> not
# operating-point shopping). Aggressive operating point (--adapt-lr 0.004, online, batch small);
# arch/severities/compositions per the v5 config. STANDALONE full-benchmark aggressive result
# (NOT a paired benign-vs-E contrast, since E ran only the 3 noise corruptions).
# caffeinate -is keeps the Mac + external drive awake for the whole run (protocol E did this;
# omitting it is what let the T9 sleep mid-run and deadlock the exFAT I/O in uninterruptible wait).
caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc --methods tent eata sar --device mps --seed 0 --imagenetc-root "$IC_ROOT" --corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression --arch resnet50 --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --imagenetc-composition iid imbalanced single_class --out-results experiments/kbound/results/win_hunt_v5/imagenetc_aggr
