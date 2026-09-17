#!/usr/bin/env bash
# Foolproof K-Bound training launcher. Run from ANY folder, e.g.:
#   bash "$KB_REPO_ROOT"/docs/research/kbound/scripts/kbtrain.sh noise
#
# It always: cd's to the repo, activates ~/.venv_wilds (the one WITH torch+wilds),
# sets TMPDIR/TORCH_HOME to T9, wraps in caffeinate, and verifies the venv first.
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

set -uo pipefail
REPO="$KB_REPO_ROOT"
VENV="$HOME/.venv_wilds"
IC=experiments/kbound/data/imagenet-c
IC_FAST="$HOME/kbound_inc"            # fast internal-SSD copy of ImageNet-C noise (prep_internal_noise.sh)
CAM_FAST="$HOME/kbound_cam/wilds"     # fast internal-SSD copy of Camelyon17 (prep_internal_camelyon.sh)
INR_FAST="$HOME/kbound_inr/imagenet-r" # fast internal-SSD copy of ImageNet-R
RES=experiments/kbound/results
S2=docs/research/kbound/scripts/cifar_tent_mps_v2.py
WILDS=docs/research/kbound/scripts/run_wilds_camelyon17.py
RXRX1_9PLUS=docs/research/kbound/scripts/run_rxrx1_9plus.sh
IMAGENETR=experiments/kbound/wilds/run_imagenetr_kbound.py
WIN_FINDER=docs/research/kbound/scripts/find_kbound_wins.py
HOLDOUT_SCORER=docs/research/kbound/scripts/score_kbound_holdout.py
PROTO_DEV_LOCK=docs/research/kbound/scripts/run_protocol_dev_lock.py
WIN_LOOP=docs/research/kbound/scripts/run_win_loop.py
HARD_WIN_LOOP=docs/research/kbound/scripts/run_hard_dataset_win_loop.py
OFFICEHOME_REPL=docs/research/kbound/scripts/run_officehome_protocol_m_replicate.sh
KGA_ELARA=src/scripts/kbound/run_kga_elara_integration.py

cd "$REPO" || { echo "ERROR: repo not found at $REPO"; exit 1; }
[ -d "$VENV" ] || { echo "ERROR: $VENV missing. Create it once with run_wilds.sh"; exit 1; }
source "$VENV/bin/activate"
export TMPDIR="$KB_EXTERNAL_ROOT/tmp" TORCH_HOME="$KB_EXTERNAL_ROOT/torch_cache"
mkdir -p "$TMPDIR" "$TORCH_HOME"
python -c "import torch, wilds" 2>/dev/null || { echo "ERROR: wrong venv (torch/wilds missing). Expected $VENV"; exit 1; }
echo ">> repo=$REPO  venv=$VENV  job=${1:-<none>}"

case "${1:-}" in
  noise)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" ;;
  noise-full)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise_full" \
      --severities 1 2 3 4 5 --all-batch --max-images 2000 ;;
  noise-quick)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" --quick ;;
  noise-smoke)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_smoke" \
      --quick --max-images 64 ;;
  camelyon)
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$RES/wilds" --seeds 0 1 2 3 --epochs 4 --steps 10 --lr 1e-3 --retrain ;;
  camelyon-smoke)
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$RES/wilds_smoke" --seeds 0 1 2 3 --epochs 1 --steps 5 --lr 1e-3 --frac 0.001 --retrain ;;
  camelyon-fast)
    [ -d "$CAM_FAST/camelyon17_v1.0/patches" ] || { echo "ERROR: internal copy not found at $CAM_FAST -- run prep_internal_camelyon.sh first."; exit 1; }
    caffeinate -is python "$WILDS" --wilds-root "$CAM_FAST" \
      --output-dir "$RES/wilds" --seeds 0 1 2 3 --epochs 4 --steps 10 --lr 1e-3 --retrain ;;
  camelyon-fast-1pct)
    [ -d "$CAM_FAST/camelyon17_v1.0/patches" ] || { echo "ERROR: internal copy not found at $CAM_FAST -- run prep_internal_camelyon.sh first."; exit 1; }
    caffeinate -is python "$WILDS" --wilds-root "$CAM_FAST" \
      --output-dir "$RES/wilds_1pct" --seeds 0 1 2 3 --epochs 1 --steps 5 --lr 1e-3 --frac 0.01 --retrain ;;
  noiseblur)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noiseblur" ;;
  vit)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch vit_b16 --methods tent eata sar --out-results "$RES/imagenetc_noise_vit" ;;
  noise-fast)
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_noise" --cooldown 3 ;;
  noise-fast-1pct)
    # FULL grid (3 corruptions x 3 sev x 3 methods, all 108 conditions) at ~1% images/cell.
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_1pct" \
      --max-images 50 ;;
  noise-fast-01pct)
    # 0.1% SMOKE: full 3x3 grid, ~8 imgs/cell -> proves the pipeline end-to-end in ~1-2 min.
    # Writes to a SEPARATE dir so it NEVER contaminates the full run's resume checkpoint.
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch resnet50 --methods tent eata sar --out-results "$RES/imagenetc_smoke01" \
      --max-images 8 ;;
  vit-fast)
    [ -d "$IC_FAST/gaussian_noise/1" ] || { echo "ERROR: internal copy not found at $IC_FAST -- run prep_internal_noise.sh first."; exit 1; }
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
      --corruptions gaussian_noise shot_noise impulse_noise \
      --arch vit_b16 --methods tent eata sar --out-results "$RES/imagenetc_noise_vit" ;;
  cifar101)
    # CIFAR-10.1 NATURAL distribution shift (reuses experiments/kbound/cifar/resnet18_cifar.pt; auto-downloads ~30MB .npy)
    caffeinate -is python "$S2" --benchmarks cifar101 \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --out-results "$RES/cifar101" ;;
  cifar101-quick)
    caffeinate -is python "$S2" --benchmarks cifar101 \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --out-results "$RES/cifar101_quick" --quick ;;
  rxrx1-9plus)
    bash "$RXRX1_9PLUS" ;;
  rxrx1-9plus-dry-run)
    bash "$RXRX1_9PLUS" --dry-run ;;
  imagenetr-d)
    _INR="${INR_FAST:-}"
    [ -d "$_INR" ] || _INR=experiments/kbound/data/imagenet-r
    [ -d "$_INR" ] || { echo "ERROR: ImageNet-R not found at $INR_FAST or experiments/kbound/data/imagenet-r"; exit 1; }
    caffeinate -is python "$IMAGENETR" --panel diverse_backbones \
      --imagenetr-dir "$_INR" --seeds 0 1 2 3 \
      --compositions iid imbalanced single_class --batch-regimes small tiny \
      --aggressiveness mild aggressive --n-eval 500 --n-batches 4 \
      --frozen-eval-batch 32 \
      --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 \
      --device auto --run-name imagenetr_protocol_d_size_diverse_panel_v2 ;;
  imagenetr-d-dry-run)
    python "$IMAGENETR" --panel diverse_backbones --dry-run \
      --imagenetr-dir "$INR_FAST" --seeds 0 1 2 3 \
      --compositions iid imbalanced single_class --batch-regimes small tiny \
      --aggressiveness mild aggressive --n-eval 500 --n-batches 4 \
      --frozen-eval-batch 32 \
      --tau-star 0.52 --kappa 2.5 --sd-L 0.6 --delta 0.05 \
      --device auto --run-name imagenetr_protocol_d_size_diverse_panel_v2 ;;
  win-finder)
    "$VENV/bin/python" "$WIN_FINDER" --top 40 ;;
  win-loop)
    "$VENV/bin/python" "$WIN_LOOP" --refresh-finder --top 60 --top-per-dataset 4 ;;
  hard-win-loop)
    "$VENV/bin/python" "$HARD_WIN_LOOP" ;;
  officehome-holdout)
    "$VENV/bin/python" "$HOLDOUT_SCORER" \
      --cal-records experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json \
      --test-records experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json \
      --candidate sar_online_aggressive --estimator gbr --conformal global \
      --output-dir experiments/kbound/results/officehome_holdout_sar_aggr_gbr_global_single ;;
  officehome-repl)
    bash "$OFFICEHOME_REPL" ;;
  protocol-h-v2)
    "$VENV/bin/python" "$PROTO_DEV_LOCK" --protocol-yaml research_lock/IWILDCAM_PROTOCOL_H_v2.yaml ;;
  protocol-m-v2)
    "$VENV/bin/python" "$PROTO_DEV_LOCK" --protocol-yaml research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml ;;
  kga-elara-integrated)
    "$VENV/bin/python" "$KGA_ELARA" \
      --protocol research_lock/KGA_ELARA_INTEGRATION_v1.yaml ;;
  kga-elara-integrated-dry-run)
    "$VENV/bin/python" "$KGA_ELARA" \
      --protocol research_lock/KGA_ELARA_INTEGRATION_v1.yaml --dry-run ;;
  theory-v2)
    bash docs/research/kbound/scripts/run_theory_v2_validators.sh ;;
  multicandidate-panel)
    "$VENV/bin/python" docs/research/kbound/scripts/multicandidate_decide_kga.py --selftest ;;
  cifar10c)
    # CIFAR-10-C Protocol A stress grid (matches stress_grid_multiseed_v1 / LOCKED_ANALYSIS).
    S="${2:-0}"
    caffeinate -is python "$S2" --benchmarks cifar10c --quick \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --device "${KB_DEVICE:-mps}" --seed "$S" \
      --out-results "$RES/stress_grid_multiseed_v1/seed$S" ;;
  pacs)
    caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
      --root experiments/kbound/domainbed --device "${KB_DEVICE:-mps}" \
      --out "$RES/pacs_result.json" ;;
  smoke-all)
    # ~0.5% smoke across all 9 datasets (separate output dirs; does NOT overwrite final-all).
    export KB_DEVICE="${KB_DEVICE:-mps}"
    STAMP=$(date +%Y%m%d_%H%M%S)
    SMOKE_ROOT="$RES/smoke05_${STAMP}"
    mkdir -p "$SMOKE_ROOT"
    echo ">> SMOKE-ALL (~0.5%)  device=$KB_DEVICE  out=$SMOKE_ROOT"
    # 1 CIFAR-10-C
    caffeinate -is python "$S2" --benchmarks cifar10c \
      --data-root experiments/kbound/cifar --methods tent \
      --device "$KB_DEVICE" --seed 0 --quick --max-cells 8 \
      --out-results "$SMOKE_ROOT/cifar10c_stress/seed0"
    # 2 ImageNet-C (~20 imgs/cell ≈ 0.5% of 4000)
    caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
      --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 \
      --methods tent --device "$KB_DEVICE" --seed 0 --quick --max-images 20 \
      --out-results "$SMOKE_ROOT/imagenetc_noise/seed0"
    # 3 CIFAR-10.1
    caffeinate -is python "$S2" --benchmarks cifar101 \
      --data-root experiments/kbound/cifar --methods tent \
      --device "$KB_DEVICE" --seed 0 --quick \
      --out-results "$SMOKE_ROOT/cifar101/seed0"
    # 4 Camelyon17 (0.5% of patches)
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$SMOKE_ROOT/wilds" --seeds 0 --epochs 1 --steps 5 --lr 1e-3 \
      --frac 0.005 --retrain
    # 5 RxRx1 (minimal grid)
    if [ -d "${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}/rxrx1_v1.0" ]; then
      RXRX1_MODEL_SEEDS=0 RXRX1_CONDITION_SEEDS=0 RXRX1_N_EVAL=3 RXRX1_N_BATCHES=2 \
        RXRX1_RESULTS_ROOT="$SMOKE_ROOT" RXRX1_RUN_TAG=rxrx1_smoke05 \
        bash "$RXRX1_9PLUS"
    else
      echo ">> SKIP RxRx1 smoke: data not at ${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}/rxrx1_v1.0"
    fi
    # 6 ImageNet-R
    [ -d "$INR_FAST" ] || INR_FAST=experiments/kbound/data/imagenet-r
    caffeinate -is python "$IMAGENETR" --panel diverse_backbones --smoke \
      --imagenetr-dir "$INR_FAST" --results-root "$SMOKE_ROOT" \
      --run-name imagenetr_smoke05
    # 7 PACS
    caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
      --root experiments/kbound/domainbed --device "$KB_DEVICE" --seed 0 --smoke \
      --out "$SMOKE_ROOT/pacs_smoke.json"
    # 8 iWildCam + 9 Office-Home (OOF scoring of locked logs — fast, no retrain)
    "$VENV/bin/python" "$PROTO_DEV_LOCK" \
      --protocol-yaml research_lock/IWILDCAM_PROTOCOL_H_v2.yaml \
      --output-dir "$SMOKE_ROOT/iwildcam_protocol_H_v2"
    "$VENV/bin/python" "$PROTO_DEV_LOCK" \
      --protocol-yaml research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml \
      --output-dir "$SMOKE_ROOT/officehome_protocol_M_v2"
    "$VENV/bin/python" docs/research/kbound/scripts/collate_final.py \
      --results "$SMOKE_ROOT" --stamp "smoke05_${STAMP}"
    echo ">> SMOKE-ALL done -> $SMOKE_ROOT/final_manifest_smoke05_${STAMP}.{json,md}" ;;
  smoke-all-v2)
    # Multiseed smoke (~1%): Protocol-A CIFAR, 3 adapters, theory+routing preflight, separate output dir.
    bash "$0" theory-v2
    bash "$0" multicandidate-panel
    export KB_DEVICE="${KB_DEVICE:-mps}"
    SMOKE_SEEDS="${KB_SMOKE_SEEDS:-0 1}"
    STAMP=$(date +%Y%m%d_%H%M%S)
    SMOKE_ROOT="$RES/smoke_ms_${STAMP}"
    mkdir -p "$SMOKE_ROOT"
    echo ">> SMOKE-ALL-V2 (multiseed)  device=$KB_DEVICE  seeds=[$SMOKE_SEEDS]  out=$SMOKE_ROOT"
    for s in $SMOKE_SEEDS; do
      echo "==== smoke seed $s ===="
      caffeinate -is python "$S2" --benchmarks cifar10c --quick \
        --data-root experiments/kbound/cifar --methods tent eata sar \
        --device "$KB_DEVICE" --seed "$s" \
        --out-results "$SMOKE_ROOT/stress_grid_multiseed_v1/seed$s"
      caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
        --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 \
        --methods tent eata sar --device "$KB_DEVICE" --seed "$s" --quick \
        --max-images "${KB_SMOKE_IC_IMG:-40}" \
        --out-results "$SMOKE_ROOT/imagenetc_noise/seed$s"
      caffeinate -is python "$S2" --benchmarks cifar101 \
        --data-root experiments/kbound/cifar --methods tent eata sar \
        --device "$KB_DEVICE" --seed "$s" --quick \
        --out-results "$SMOKE_ROOT/cifar101/seed$s"
      caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
        --root experiments/kbound/domainbed --device "$KB_DEVICE" --seed "$s" --smoke \
        --out "$SMOKE_ROOT/pacs_seed$s.json"
    done
    # Camelyon: need >=2 seeds for cross-seed KGA certificate
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$SMOKE_ROOT/wilds" --seeds $SMOKE_SEEDS --epochs 1 --steps 8 --lr 1e-3 \
      --frac "${KB_SMOKE_CAM_FRAC:-0.01}" --retrain
    if [ -d "${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}/rxrx1_v1.0" ]; then
      RXRX1_MODEL_SEEDS="$SMOKE_SEEDS" RXRX1_CONDITION_SEEDS=0 RXRX1_N_EVAL=8 RXRX1_N_BATCHES=2 \
        RXRX1_RESULTS_ROOT="$SMOKE_ROOT" RXRX1_RUN_TAG=rxrx1_smoke_ms \
        bash "$RXRX1_9PLUS"
    else
      echo ">> SKIP RxRx1 smoke: data not at ${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}/rxrx1_v1.0"
    fi
    _INR="${INR_FAST:-}"; [ -d "$_INR" ] || _INR=experiments/kbound/data/imagenet-r
    caffeinate -is python "$IMAGENETR" --panel diverse_backbones --smoke \
      --imagenetr-dir "$_INR" --results-root "$SMOKE_ROOT" --run-name imagenetr_smoke_ms
    "$VENV/bin/python" "$PROTO_DEV_LOCK" \
      --protocol-yaml research_lock/IWILDCAM_PROTOCOL_H_v2.yaml \
      --output-dir "$SMOKE_ROOT/iwildcam_protocol_H_v2"
    "$VENV/bin/python" "$PROTO_DEV_LOCK" \
      --protocol-yaml research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml \
      --output-dir "$SMOKE_ROOT/officehome_protocol_M_v2"
    mkdir -p "$SMOKE_ROOT/stress_grid_multiseed_v1"
    cp "$RES/stress_grid_multiseed_v1/_locked_analysis_script.py" \
      "$SMOKE_ROOT/stress_grid_multiseed_v1/_locked_analysis_script.py"
    "$VENV/bin/python" docs/research/kbound/scripts/collate_final.py \
      --results "$SMOKE_ROOT" --stamp "smoke_ms_${STAMP}"
    echo ">> SMOKE-ALL-V2 done -> $SMOKE_ROOT/final_manifest_smoke_ms_${STAMP}.{json,md}"
    echo ">> Analyze: python docs/research/kbound/scripts/smoke_pipeline_report.py --smoke-root $SMOKE_ROOT" ;;
  final-all-v2)
    # Full 9-dataset rerun + Wave 4 theory/routing preflight (does not change headline protocol path).
    bash "$0" theory-v2
    bash "$0" multicandidate-panel
    RXRX1_DATA="${RXRX1_DATA_ROOT:-$HOME/kbound_rxrx1_data}"
    if [ ! -d "$RXRX1_DATA/rxrx1_v1.0" ]; then
      echo ">> WARN: RxRx1 data missing at $RXRX1_DATA/rxrx1_v1.0"
      echo ">>       Download WILDS RxRx1 to that path before step 5, or set RXRX1_DATA_ROOT."
    fi
    bash "$0" final-all ;;
  final-all)
    # ── FINAL multi-seed end-to-end rerun of ALL 9 datasets on the OUT-OF-FOLD code. ──
    # CIFAR-10-C, ImageNet-C, CIFAR-10.1, Camelyon17, RxRx1, ImageNet-R, PACS, iWildCam, Office-Home.
    SEEDS="${KB_SEEDS:-0 1 2 3 4}"; export KB_DEVICE="${KB_DEVICE:-mps}"
    STAMP=$(date +%Y%m%d_%H%M%S)
    # (A) INTEGRITY GUARD: refuse to run if any in-sample-eps pattern is back in a scorer.
    if grep -REn "predict\(Zc\) - Bc|abs\(Bhat_c - Bc\)" \
         docs/research/kbound/scripts/analyze_F.py docs/research/kbound/scripts/score_kbound_holdout.py \
         docs/research/kbound/scripts/mixed_stream_kbound.py "$PROTO_DEV_LOCK" 2>/dev/null \
       | grep -vE "resid_c|_loo|out-of-fold"; then
      echo "ABORT: in-sample-eps pattern detected in a scorer -- fix before the final run."; exit 1; fi
    echo ">> FINAL-ALL  seeds=[$SEEDS]  device=$KB_DEVICE  stamp=$STAMP  (caffeinated; expect many hours)"
    for s in $SEEDS; do
      echo "==== seed $s : per-seed TTA generation ===="
      bash "$0" cifar10c "$s"                                                          # 1 CIFAR-10-C
      caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC" \
        --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 \
        --methods tent eata sar --device "$KB_DEVICE" --seed "$s" \
        --max-images "${KB_IC_MAXIMG:-2000}" --cooldown "${KB_IC_COOLDOWN:-2}" \
        --out-results "$RES/imagenetc_noise/seed$s"                                    # 2 ImageNet-C (T9 path; pre-reg image cap; --cooldown gives the USB drive breathing room; resumable via checkpoint.json)
      caffeinate -is python "$S2" --benchmarks cifar101 --data-root experiments/kbound/cifar \
        --methods tent eata sar --device "$KB_DEVICE" --seed "$s" \
        --out-results "$RES/cifar101/seed$s"                                           # 3 CIFAR-10.1
      caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
        --root experiments/kbound/domainbed --device "$KB_DEVICE" --seed "$s" \
        --out "$RES/pacs_seed$s.json"                                                  # 7 PACS
    done
    caffeinate -is python "$WILDS" --wilds-root experiments/kbound/data/wilds \
      --output-dir "$RES/wilds" --seeds 0 1 2 3 --epochs 4 --steps 10 --lr 1e-3   # 4 Camelyon17 (T9 data; reuse f0_seed*.pt, no --retrain -> resumable)
    bash "$0" rxrx1-9plus        # 5 RxRx1       (9+ pre-set configs)
    bash "$0" imagenetr-d        # 6 ImageNet-R  (runner does 4 training seeds in one call)
    # 8/9 pre-registered LOCKED-protocol scoring -- NOT seed-redrawn (that would be p-hacking);
    #     their multi-seed robustness is the condition-bootstrap (bootstrap_win_cis.py).
    bash "$0" protocol-h-v2      # 8 iWildCam (out-of-fold scoring of locked logs)
    bash "$0" protocol-m-v2      # 9 Office-Home (out-of-fold scoring of locked logs)
    "$VENV/bin/python" docs/research/kbound/scripts/collate_final.py --results "$RES" --stamp "$STAMP"
    echo ">> FINAL-ALL done -> $RES/final_manifest_$STAMP.{json,md}  (re-run for multi-time; each stamps a new manifest)" ;;
  *)
    echo "usage: bash kbtrain.sh [noise|...|smoke-all|smoke-all-v2|final-all|final-all-v2|theory-v2|multicandidate-panel|...]"; exit 1 ;;
esac
