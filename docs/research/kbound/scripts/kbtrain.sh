#!/usr/bin/env bash
# Foolproof K-Bound training launcher. Run from ANY folder, e.g.:
#   bash /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/scripts/kbtrain.sh noise
#
# It always: cd's to the repo, activates ~/.venv_wilds (the one WITH torch+wilds),
# sets TMPDIR/TORCH_HOME to T9, wraps in caffeinate, and verifies the venv first.
set -uo pipefail
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
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
export TMPDIR=/Volumes/T9/uav/tmp TORCH_HOME=/Volumes/T9/uav/torch_cache
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
    [ -d "$INR_FAST" ] || { echo "ERROR: ImageNet-R internal copy not found at $INR_FAST"; exit 1; }
    caffeinate -is python "$IMAGENETR" --panel diverse_backbones \
      --imagenetr-dir "$INR_FAST" --seeds 0 1 2 3 \
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
  cifar10c)
    # CIFAR-10-C stress grid (decide_kga, out-of-fold LOO radius). Seed via 2nd arg: kbtrain.sh cifar10c 0
    S="${2:-0}"
    caffeinate -is python "$S2" --benchmarks cifar10c \
      --data-root experiments/kbound/cifar --methods tent eata sar \
      --device "${KB_DEVICE:-mps}" --seed "$S" \
      --out-results "$RES/cifar10c_stress/seed$S" ;;
  pacs)
    caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
      --root experiments/kbound/domainbed --device "${KB_DEVICE:-mps}" \
      --out "$RES/pacs_result.json" ;;
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
      caffeinate -is python "$S2" --benchmarks imagenetc --imagenetc-root "$IC_FAST" \
        --corruptions gaussian_noise shot_noise impulse_noise --arch resnet50 \
        --methods tent eata sar --device "$KB_DEVICE" --seed "$s" \
        --out-results "$RES/imagenetc_noise/seed$s"                                    # 2 ImageNet-C
      caffeinate -is python "$S2" --benchmarks cifar101 --data-root experiments/kbound/cifar \
        --methods tent eata sar --device "$KB_DEVICE" --seed "$s" \
        --out-results "$RES/cifar101/seed$s"                                           # 3 CIFAR-10.1
      caffeinate -is python docs/research/kbound/scripts/pacs_vlcs_runner.py --dataset PACS \
        --root experiments/kbound/domainbed --device "$KB_DEVICE" --seed "$s" \
        --out "$RES/pacs_seed$s.json"                                                  # 7 PACS
    done
    bash "$0" camelyon-fast      # 4 Camelyon17  (runner does 4 training seeds in one call)
    bash "$0" rxrx1-9plus        # 5 RxRx1       (9+ pre-set configs)
    bash "$0" imagenetr-d        # 6 ImageNet-R  (runner does 4 training seeds in one call)
    # 8/9 pre-registered LOCKED-protocol scoring -- NOT seed-redrawn (that would be p-hacking);
    #     their multi-seed robustness is the condition-bootstrap (bootstrap_win_cis.py).
    bash "$0" protocol-h-v2      # 8 iWildCam (out-of-fold scoring of locked logs)
    bash "$0" protocol-m-v2      # 9 Office-Home (out-of-fold scoring of locked logs)
    "$VENV/bin/python" docs/research/kbound/scripts/collate_final.py --results "$RES" --stamp "$STAMP"
    echo ">> FINAL-ALL done -> $RES/final_manifest_$STAMP.{json,md}  (re-run for multi-time; each stamps a new manifest)" ;;
  *)
    echo "usage: bash kbtrain.sh [noise|...|officehome-holdout|officehome-repl|protocol-h-v2|protocol-m-v2|kga-elara-integrated|kga-elara-integrated-dry-run|noiseblur|vit|vit-fast]"; exit 1 ;;
esac
