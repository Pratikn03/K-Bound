#!/usr/bin/env bash
# ============================================================================
# SAR OFFICIAL-SETTINGS CONTROL  (review fix-queue item 7, the optional half)
# ============================================================================
# WHY THIS RUN EXISTS
#   The promoted ImageNet-C result is a "beats-both" against always-adapt and
#   always-freeze, and the always-adapt arm is SAR. But the shipped runs drive
#   SAR at the harness's shared lr 4e-3 -- 16x the 2.5e-4 that Niu et al. use --
#   with layer4 adapted, which SAR's own recipe freezes. A reviewer will ask the
#   obvious question:
#
#       "Does KGA beat always-adapt because adaptation is genuinely harmful
#        here, or because you ran SAR 16x off its own operating point?"
#
#   This control answers it. It re-runs the SAME 27-cell x 5-seed grid, the same
#   protocol, the same everything, changing ONLY SAR's optimizer settings to the
#   official ones (--sar-lr 2.5e-4 --sar-freeze-layer4; both flags already exist
#   in cifar_tent_mps_v2.py:1383,1387).
#
# HOW TO READ THE RESULT
#   * If SAR still collapses at official settings -> the beats-both is about the
#     shift, not the hyperparameters. The claim gets STRONGER and you say so.
#   * If SAR does NOT collapse at official settings -> the ImageNet-C beats-both
#     is an artifact of the aggressive operating point. That is a real negative
#     result: report it, narrow the claim to the aggressive regime, and keep the
#     freeze-side no-harm statement, which does not depend on SAR failing.
#   Either outcome is publishable. Do not re-run until you like the answer.
#
# COST  ~1 method x 27 cells x 5 seeds on MPS. Budget a few hours; it is
#       resumable, so an interrupted run picks up at the next incomplete seed.
#
# USAGE
#   export KBOUND_EXTERNAL_ROOT=/Volumes/T9/uav      # volume holding imagenetc_local
#   export KBOUND_PYTHON="$(command -v python)"      # the conda 'aetta' python
#   bash docs/research/kbound/runbooks/run_sar_official_control.sh --preflight
#   bash docs/research/kbound/runbooks/run_sar_official_control.sh
# ============================================================================
set -u

# --- portable roots (EXTERNAL_STORAGE_POLICY.md: no machine-local paths) -----
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
: "${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT to the volume holding the git-excluded datasets (layout: docs/research/kbound/kbound_repro/paths.py)}"

REPO="$KB_REPO_ROOT"
PY="${KBOUND_PYTHON:-python3}"
SCRIPT="$REPO/docs/research/kbound/scripts/cifar_tent_mps_v2.py"
IC="$KBOUND_EXTERNAL_ROOT/imagenetc_local"
BASE="$REPO/experiments/kbound/results/imagenetc_sar_official_control_v1"
SEEDS="${KBOUND_SEEDS:-0 1 2 3 4}"

export TORCH_HOME="$KBOUND_EXTERNAL_ROOT/torch_cache" TMPDIR="$KBOUND_EXTERNAL_ROOT/tmp"

# --- preflight ---------------------------------------------------------------
preflight() {
  local ok=0
  echo "== SAR official-settings control : preflight =="
  echo "repo root          : $REPO"
  echo "external root      : $KBOUND_EXTERNAL_ROOT"
  echo "python             : $PY"
  [ -f "$SCRIPT" ]  && echo "runner             : OK  $SCRIPT" || { echo "runner             : MISSING $SCRIPT"; ok=1; }
  [ -d "$IC" ]      && echo "ImageNet-C root    : OK  $IC" || { echo "ImageNet-C root    : MISSING $IC"; ok=1; }
  for c in gaussian_noise shot_noise impulse_noise; do
    for s in 1 3 5; do
      [ -d "$IC/$c/$s" ] || { echo "  MISSING corruption dir: $IC/$c/$s"; ok=1; }
    done
  done
  [ "$ok" = 0 ] && echo "corruption dirs    : OK  (3 corruptions x severities 1,3,5)"
  "$PY" - <<'PYEOF' || ok=1
import sys
try:
    import torch
except Exception as e:
    print("torch              : MISSING (%s)" % e); sys.exit(1)
print("torch              : OK  %s" % torch.__version__)
avail = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
print("mps                : %s" % ("OK" if avail else "NOT AVAILABLE -- pass --device cpu (much slower)"))
PYEOF
  echo "output dir         : $BASE"
  [ "$ok" = 0 ] && echo "PREFLIGHT PASS -- rerun without --preflight to start." \
                || echo "PREFLIGHT FAIL -- fix the MISSING lines above first."
  return "$ok"
}

if [ "${1:-}" = "--preflight" ]; then preflight; exit $?; fi
preflight || { echo "Refusing to start."; exit 1; }

mkdir -p "$BASE" "$TMPDIR"
cd "$REPO" || exit 1
echo "[$(date '+%F %T')] SAR OFFICIAL CONTROL START (seeds: $SEEDS)" >> "$BASE/launch.log"

for S in $SEEDS; do
  OUT="$BASE/seed$S"
  mkdir -p "$OUT"
  if [ -f "$OUT/decisive_tta_results.json" ]; then
    echo "[$(date '+%F %T')] SEED $S SKIP (already complete)" | tee -a "$BASE/launch.log"
    continue
  fi
  echo "[$(date '+%F %T')] SEED $S START -> $OUT" | tee -a "$BASE/launch.log"
  caffeinate -is "$PY" "$SCRIPT" \
    --benchmarks imagenetc \
    --imagenetc-root "$IC" \
    --corruptions gaussian_noise shot_noise impulse_noise \
    --arch resnet50 \
    --methods sar \
    --device "${KBOUND_DEVICE:-mps}" \
    --seed "$S" \
    --severities 1 3 5 --max-images 4000 \
    --imagenetc-composition iid imbalanced single_class \
    --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
    --sar-lr 2.5e-4 --sar-freeze-layer4 \
    --out-results "$OUT" \
    >> "$OUT/seed${S}.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] SEED $S DONE rc=$rc" | tee -a "$BASE/launch.log"
done

echo "[$(date '+%F %T')] SAR OFFICIAL CONTROL COMPLETE" | tee -a "$BASE/launch.log"
cat <<EOF

Done. Results under:
  $BASE/seed*/

NEXT: compare against the promoted aggressive-settings run. The comparison that
matters is the always-adapt (SAR) arm's mean regret and harmful-cell fraction:

  official settings collapse as badly  -> beats-both is about the shift; strengthen the claim
  official settings do NOT collapse    -> beats-both is an artifact of lr 4e-3; narrow the
                                          adapt-side claim to the aggressive regime and keep
                                          the freeze-side no-harm result, which is unaffected

Baseline for the comparison:
  experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed{0..4}.json
EOF
