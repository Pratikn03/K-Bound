#!/usr/bin/env bash
# Official POEM algorithm (Bar et al., NeurIPS 2024) on ImageNet-C — item-11 baseline.
# Uses a reviewed native-algorithm protocol adapter for one corruption per run;
# the driver is derived, not unchanged native-driver or benchmark attestation.
# CORRECTION vs original plan: POEM has NO cifar10 path (main.py is ImageNet-only);
# it runs on ImageNet-C with a timm ResNet50-GN (official weights, no training).
# We evaluate the 3 noise corruptions K-Bound's ImageNet-C panel uses
# (gaussian/shot/impulse). This overlap alone does not establish a matched
# head-to-head: cell identities, trajectories and scoring must also be audited.
# The pinned normal-mode driver otherwise iterates all 15 corruptions despite
# --corruption; the authenticated bootstrap derives only its dispatch list and
# records POEM_PROTOCOL_DERIVATION_RECEIPT.json in each fresh run output.
#
# Prereqs for a native run:
#   - the pinned upstream checkout is selected with POEM_SOURCE (default:
#     external/poem_official), and its native dependencies are installed
#   - the pinned environment exposes CUDA; the upstream implementation uses
#     unconditional CUDA calls and cannot be relabelled as native on MPS/CPU
#   - clean ImageNet is available at IMAGENET_ROOT/val (the upstream loader
#     uses it to create the holdout split and to preserve class-index order)
#   - data at ~/imagenetc_local/<corruption>/<1..5>/<1000 class dirs>
# This runbook intentionally fails before model construction on a non-CUDA host.
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
R="$KB_REPO_ROOT"
P="${POEM_SOURCE:-$R/external/poem_official}"
SAR="${SAR_SOURCE:-$R/external/sar_official}"
BOOTSTRAP="$R/docs/research/kbound/scripts/poem_dependency_bootstrap.py"
OUT="${POEM_OUTPUT:-$R/experiments/kbound/results/official_repro_v1/poem_imagenetc_native}"
CLEAN="${IMAGENET_ROOT:-}"
IC="${IMAGENETC_ROOT:-}"
PYBIN="${POEM_PYTHON:?Set POEM_PYTHON to the pinned native CUDA runtime}"
SEEDS="${SEEDS:-0}"
SEVERITIES="${SEVERITIES:-1 3 5}"        # synchronized K-Bound panel
CORRUPTIONS="${CORRUPTIONS:-gaussian_noise shot_noise impulse_noise}"
BATCH="${TEST_BATCH_SIZE:-1}"             # POEM's online reference setting
if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "FATAL: output directory must be fresh: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/poem_imagenetc.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Keep the command portable: caffeinate is useful on this Mac but is absent
# on the Linux/NVIDIA host where native CUDA evidence is expected.
run_native_cmd() {
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -is "$@"
  else
    "$@"
  fi
}

# ---- preflight (fail-closed) ----
[ -x "$PYBIN" ] || { say "FATAL: poem env python missing ($PYBIN)"; exit 2; }
[ -e "$P/.git" ] || { say "FATAL: pinned POEM checkout missing ($P)"; exit 2; }
# The pinned upstream import is incomplete. Supply the exact authenticated SAR
# dependency in memory; an ignored local shim is never loaded or certified.
if ! "$PYBIN" "$BOOTSTRAP" --poem-source "$P" --sar-source "$SAR" -- --help >> "$LOG" 2>&1; then
  say "FATAL: authenticated POEM dependency/entrypoint preflight failed; see $LOG"
  exit 2
fi
[[ -n "$CLEAN" ]] || { say "FATAL: set IMAGENET_ROOT to clean ImageNet root containing val/"; exit 2; }
[[ -d "$CLEAN/val" ]] || { say "FATAL: clean ImageNet root has no val/: $CLEAN"; exit 2; }
CLEAN_CLASSES=$(find "$CLEAN/val" -mindepth 1 -maxdepth 1 -type d -print | wc -l | tr -d ' ')
[ "$CLEAN_CLASSES" = "1000" ] || { say "FATAL: expected 1000 clean ImageNet val class dirs under $CLEAN/val, found $CLEAN_CLASSES"; exit 2; }
[[ -n "$IC" ]] || { say "FATAL: set IMAGENETC_ROOT to ImageNet-C root"; exit 2; }
for C in $CORRUPTIONS; do
  for L in $SEVERITIES; do
    level_dir="$IC/$C/$L"
    [ -d "$level_dir" ] || { say "FATAL: ImageNet-C missing: $level_dir"; exit 2; }
    NCLS=$(find "$level_dir" -mindepth 1 -maxdepth 1 -type d -print | wc -l | tr -d ' ')
    [ "$NCLS" = "1000" ] || { say "FATAL: expected 1000 class dirs under $level_dir, found $NCLS"; exit 2; }
  done
done
BACKEND=$("$PYBIN" -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))")
[ "$BACKEND" = "cuda" ] || { say "FATAL: native POEM requires CUDA; pinned environment reports $BACKEND"; exit 2; }
"$PYBIN" -c "import timm; timm.create_model('resnet50_gn', pretrained=True)" >/dev/null 2>&1 \
  || { say "FATAL: timm resnet50_gn did not load in poem env"; exit 2; }
say "preflight OK — reviewed native-algorithm protocol adapter; source=$P backend=$BACKEND clean=$CLEAN data=$IC seeds=[$SEEDS] sev=[$SEVERITIES] corruptions=[$CORRUPTIONS] bs=$BATCH"

# ---- runs: for each (method in poem,no_adapt) x seed x severity x corruption ----
cd "$P" || { say "FATAL: cannot cd $P"; exit 2; }
for M in poem no_adapt; do
  for S in $SEEDS; do
    for L in $SEVERITIES; do
      for C in $CORRUPTIONS; do
        say "RUN method=$M seed=$S sev=$L corruption=$C protocol_adapter=single-corruption"
        run_native_cmd "$PYBIN" -u "$BOOTSTRAP" --poem-source "$P" --sar-source "$SAR" \
          --single-corruption "$C" -- \
          --method "$M" --model resnet50_gn_timm --exp_type normal \
          --data "$CLEAN" --data_corruption "$IC" \
          --corruption "$C" --level "$L" --seed "$S" \
          --test_batch_size "$BATCH" --workers 4 \
          --output "$OUT/exps_${M}_s${S}_l${L}_${C}" >> "$LOG" 2>&1
        say "  rc=$?"
      done
    done
  done
done
say "POEM protocol-adapter runs done -> raw CSV/JSON under $OUT/exps_*/imagenet/<method>/normal/*.json"
say "NEXT (post-run wiring, needs K-Bound ImageNet-C decision file as the condition basis):"
say "  build poem_decisions.json = per-condition {action:ADAPT, a_adapted:poem.top1, a0:no_adapt.top1}"
say "  then: official_baselines_headtohead.py --candidate <kbound> --decisions poem=poem_decisions.json"
