#!/usr/bin/env bash
# Official POEM (Bar et al., NeurIPS 2024) on ImageNet-C — item-11 baseline.
# CORRECTION vs original plan: POEM has NO cifar10 path (main.py is ImageNet-only);
# it runs on ImageNet-C with a timm ResNet50-GN (official weights, no training).
# We evaluate the 3 noise corruptions K-Bound's ImageNet-C panel uses
# (gaussian/shot/impulse), so the head-to-head is protocol-matched.
#
# Prereqs already satisfied by prep:
#   - conda env 'poem' has all deps (timm/pycm/loguru/cotta/...); main.py --help OK
#   - external/poem/main.py line 40 import guarded (models.Res only needed by bn_torch)
#   - data at ~/imagenetc_local/<corruption>/<1..5>/<1000 class dirs>
# Runs GPU (MPS). Do NOT launch while the AETTA source training holds the GPU.
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

set -u
R="$KB_REPO_ROOT"
P="$R/external/poem"
OUT="$R/experiments/kbound/results/official_repro_v1/poem_imagenetc"
IC="${IMAGENETC_ROOT:-$HOME/imagenetc_local}"
PYBIN="/opt/anaconda3/envs/poem/bin/python"
SEEDS="${SEEDS:-0}"
SEVERITIES="${SEVERITIES:-5}"            # K-Bound panel uses sev{1,3,5}; default 5 (POEM headline)
CORRUPTIONS="${CORRUPTIONS:-gaussian_noise shot_noise impulse_noise}"
BATCH="${TEST_BATCH_SIZE:-64}"           # POEM headline is bs1 (online); 64 for tractability — DECLARE in paper
mkdir -p "$OUT"
LOG="$OUT/poem_imagenetc.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- preflight (fail-closed) ----
[ -x "$PYBIN" ] || { say "FATAL: poem env python missing ($PYBIN)"; exit 2; }
[ -d "$IC/gaussian_noise/5" ] || { say "FATAL: ImageNet-C not at $IC (need <corruption>/<level>/<classes>)"; exit 2; }
NCLS=$(ls "$IC/gaussian_noise/5" 2>/dev/null | wc -l | tr -d ' ')
[ "$NCLS" = "1000" ] || { say "FATAL: expected 1000 class dirs under $IC/gaussian_noise/5, found $NCLS"; exit 2; }
"$PYBIN" -c "import timm; timm.create_model('resnet50_gn', pretrained=True)" >/dev/null 2>&1 \
  || { say "FATAL: timm resnet50_gn did not load in poem env"; exit 2; }
say "preflight OK — data=$IC seeds=[$SEEDS] sev=[$SEVERITIES] corruptions=[$CORRUPTIONS] bs=$BATCH"

# ---- runs: for each (method in poem,no_adapt) x seed x severity x corruption ----
cd "$P" || { say "FATAL: cannot cd $P"; exit 2; }
for M in poem no_adapt; do
  for S in $SEEDS; do
    for L in $SEVERITIES; do
      for C in $CORRUPTIONS; do
        say "RUN method=$M seed=$S sev=$L corruption=$C"
        caffeinate -is "$PYBIN" -u main.py \
          --method "$M" --model resnet50_gn_timm --exp_type normal \
          --data "$IC" --data_corruption "$IC" \
          --corruption "$C" --level "$L" --seed "$S" \
          --test_batch_size "$BATCH" --workers 4 \
          --output "$OUT/exps_${M}_s${S}_l${L}_${C}" >> "$LOG" 2>&1
        say "  rc=$?"
      done
    done
  done
done
say "POEM runs done -> raw CSV/JSON under $OUT/exps_*/imagenet/<method>/normal/*.json"
say "NEXT (post-run wiring, needs K-Bound ImageNet-C decision file as the condition basis):"
say "  build poem_decisions.json = per-condition {action:ADAPT, a_adapted:poem.top1, a0:no_adapt.top1}"
say "  then: official_baselines_headtohead.py --candidate <kbound> --decisions poem=poem_decisions.json"
