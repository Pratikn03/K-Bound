# RUNSHEET — Wave 7 (WIN_HUNT_v5 aggressive-regime natural wave)
# Protocol: research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml  (config_lock FROZEN 2026-07-05).
# One aggressive operating point per dataset:  adapter=ONLINE (no episodic reset),
# adapt-lr=0.004 (= 4x the 1e-3 shared baseline), batch=16, aggressive steps (50; iWildCam/
# Office-Home 30).  Every command below sets ONLY those operating-point flags on top of each
# runner's existing protocol defaults; with the flags removed each runner is byte-identical to
# its prior protocol (A/E/G/H/M/J/K/D, PACS prereg).
# All commands from repo root:  cd /Volumes/T9/uav/AutoML_Flagship_V8
#
# Verification status (2026-07-05, this build session, CPU/sandbox):
#   * all 7 runners py_compile clean; CIFAR + PACS --help expose --adapt-lr/--batch-regimes/
#     --aggressiveness; selection + lr-override logic unit-checked vs the real module constants
#     (small=16, aggressive steps=50/30, lr->0.004); wilds --online-only filter -> exactly the
#     three *_online candidates.  The GPU/MPS execution itself is the user's step (needs torch).

set -euo pipefail
REPO=/Volumes/T9/uav/AutoML_Flagship_V8
cd "$REPO"

# ============================================================================
# PHASE 0 — environment
# ============================================================================
export TMPDIR=/Volumes/T9/uav/tmp TORCH_HOME=/Volumes/T9/uav/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"
# Two interpreters (both have torch + MPS):
#   PY_CORE  = repo venv, used for the CIFAR-family runner + PACS (protocol-E precedent).
#   .venv_wilds = has the `wilds` package; used for the WILDS runners + Office-Home.
PY_CORE="$REPO/.venv/bin/python"

# ============================================================================
# PHASE 1 — CIFAR-family GPU runs (CIFAR runner; PY_CORE).  Data already on disk.
# ============================================================================

## 1a. cifar10c_stress  (Protocol A splits; seeds 0-4; 15 corruptions x sev{1,3,5} x 3 comps)
for S in 0 1 2 3 4; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --data-root experiments/kbound/cifar \
    --methods tent eata sar --device mps --seed "$S" \
    --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
    --out-results experiments/kbound/results/win_hunt_v5_cifar10c/seed$S
done

## 1b. imagenetc  (Protocol E splits; seeds 0-2; resnet50; small-batch aggressive point)
IC="$REPO/experiments/kbound/data/imagenet-c"
for S in 0 1 2; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks imagenetc --imagenetc-root "$IC" --arch resnet50 \
    --methods tent eata sar --device mps --seed "$S" \
    --severities 1 3 5 --max-images 4000 \
    --imagenetc-composition iid imbalanced single_class \
    --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
    --out-results experiments/kbound/results/win_hunt_v5_imagenetc/seed$S
done

## 1c. cifar10_1  (Protocol K; natural shift; reuses the CIFAR-10 f0; seeds 0-4)
for S in 0 1 2 3 4; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar101 --data-root experiments/kbound/cifar \
    --methods tent eata sar --device mps --seed "$S" \
    --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
    --out-results experiments/kbound/results/win_hunt_v5_cifar101/seed$S
done

# ============================================================================
# PHASE 2 — WILDS + Office-Home GPU runs (.venv_wilds).  Data-roots/ckpts default to
#           each runner's protocol paths; override --data-root/--ckpt if your layout differs.
# ============================================================================
source ~/.venv_wilds/bin/activate

## 2a. camelyon17  (Protocol G; seeds 0-3; online-only pool: freeze + {tent,eata,sar}_online)
caffeinate -is python experiments/kbound/wilds/run_camelyon17_kbound.py \
  --seeds 0 1 2 3 --device mps \
  --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only \
  --run-name win_hunt_v5_camelyon17
#   -> experiments/kbound/results/win_hunt_v5_camelyon17/result_<sha8>.json
#      + per_condition_camelyon17_<method>_seed<S>.json

## 2b. iwildcam  (Protocol H; macro-F1; seeds 0-2).  Supply the Protocol-H f0 via --ckpt
##     (or add --retrain --train-seed 0 to train a fresh f0 first).
caffeinate -is python experiments/kbound/wilds/run_iwildcam_kbound.py \
  --split val --seeds 0 1 2 --device mps \
  --ckpt experiments/kbound/results/iwildcam_f0_erm/f0_resnet50_erm_seed0.pt \
  --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 \
  --candidates tent_online eata_online sar_online \
  --run-name win_hunt_v5_iwildcam

## 2c. officehome  (Protocol M v2 dev-lock; roles source -> target_val -> target_test; seeds 0-1)
##     Online + aggressive-step candidates selected by name; lr overridden to 0.004.
for ROLE in source target_val target_test; do
  caffeinate -is python experiments/kbound/officehome/run_officehome_kbound.py \
    --role "$ROLE" --seeds 0 1 --device mps \
    --batch-regimes small --adapt-lr 0.004 \
    --candidates tent_online_aggressive eata_online_aggressive sar_online_aggressive \
    --run-name win_hunt_v5_officehome
done
#   target_test is the HELD-OUT split, scored ONCE (per Protocol M v2).

## 2d. rxrx1  (Protocol J; OOD 'test' = 14 unseen experiments; seeds 0-3; online-only)
caffeinate -is python experiments/kbound/wilds/run_rxrx1_kbound.py \
  --split test --seeds 0 1 2 3 --device mps \
  --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only \
  --run-name win_hunt_v5_rxrx1

## 2e. imagenet_r  (Protocol D shared_tta panel on the 200-class rendition set; seeds 0-3; online-only)
caffeinate -is python experiments/kbound/wilds/run_imagenetr_kbound.py \
  --panel shared_tta --seeds 0 1 2 3 --device mps \
  --batch-regimes small --aggressiveness aggressive --adapt-lr 0.004 --online-only \
  --results-root experiments/kbound/results \
  --run-name win_hunt_v5_imagenetr
#   -> .../win_hunt_v5_imagenetr/result_<sha8>.json + per_condition_imagenet-r_<method>_seed<S>.json

# ============================================================================
# PHASE 3 — PACS (PACS runner; PY_CORE).  4 leave-one-domain-out test domains internal;
#           calibration = last source domain (r0 cal / r1 test).  batch tiny = 16.  seeds 0-2.
# ============================================================================
deactivate 2>/dev/null || true
PACS_ROOT="$REPO/experiments/kbound/domainbed"     # contains PACS/{art_painting,cartoon,photo,sketch}/<class>/*
mkdir -p experiments/kbound/results/win_hunt_v5_pacs
for S in 0 1 2; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/pacs_vlcs_runner.py \
    --dataset PACS --root "$PACS_ROOT" --device mps --seed "$S" \
    --batch-regimes tiny --aggressiveness aggressive --adapt-lr 0.004 \
    --out experiments/kbound/results/win_hunt_v5_pacs/pacs_v5_seed$S.json
done

# ============================================================================
# PHASE 4 — SCORING (CPU, scored ONCE; run after the GPU phases complete).
#   Bars (frozen in the shell): per-dataset WIN iff KGA beats BOTH fixed policies with both
#   95% paired-bootstrap CIs excluding zero AND FA_u <= 0.10 on held-out; NO_HARM = ties the
#   better policy / beats the worse at FA_u <= 0.10; else FAIL.  Report ALL verdicts.
# ----------------------------------------------------------------------------
# Per-dataset regret-gap CIs + FA_u pool over seeds from the per_condition / result JSONs:
#   experiments/kbound/wilds/multiseed_paired_ci.py   (wilds + CIFAR seed pooling)
#   PACS result JSON already carries per-domain FA_u, boot CIs and a WIN/SAFETY/NULL verdict.
# Pooled 9-source harsh-deployment headline under ONE universal gate (Arm-E / base-11 evidence):
#   docs/research/kbound/gapclose_wave5/win_hunt_E_universal7.py
# Exact scorer flags are finalized at scoring time against the produced run dirs (the shell
# scores once after the GPU runs).  Write every verdict JSON to research_lock/ regardless of
# outcome:  WIN_HUNT_v5_<dataset>_result.json  and  WIN_HUNT_v5_POOLED_result.json.

# ============================================================================
# APPENDIX — WIN_HUNT_v4 carryover GPU runs still pending (Wave 6).  Same env as PHASE 1.
# ----------------------------------------------------------------------------
# Arm D data on disk (experiments/kbound/results/stress_persample_v1) is PARTIAL: seed0 +
# gaussian_noise only, and NO per_condition_*.json — so score_official_headtohead.py cannot
# score it yet.  COMPLETE the arm-D run (quick grid, seeds 0-4, --log-samples) so it becomes
# scoreable, then score (Phase 3 of RUNSHEET_WAVE6):
for S in 0 1 2 3 4; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
    --methods tent --device mps --seed "$S" --log-samples \
    --out-results experiments/kbound/results/stress_persample_v1
done
#   then:  "$PY_CORE" experiments/kbound/poem_aetta/score_official_headtohead.py \
#            --run-dir experiments/kbound/results/stress_persample_v1 \
#            --dataset cifar10c --adapter tent --seeds 0 1 2 3 4 --nboot 10000
#
# Arm E seed extension 5-9 (pooled 10-seed CIs; ALL seeds enter the pool):
for S in 5 6 7 8 9; do
  caffeinate -is "$PY_CORE" docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
    --methods tent eata sar --device mps --seed "$S" \
    --out-results experiments/kbound/results/stress_grid_multiseed_v1/seed$S
done
