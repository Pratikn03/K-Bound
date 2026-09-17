# RUNSHEET — Wave 6 (WIN_HUNT_v4), ordered. Run phases in order; STOP if a phase fails.
# Protocol: research_lock/WIN_HUNT_v4_PROTOCOL.yaml (bars frozen 2026-07-04).
# All commands from repo root: cd "$KBOUND_REPO_ROOT"        # set KBOUND_REPO_ROOT to your checkout

## PHASE 0 — CPU validators (Mac, ~10 min total). ALL must print PASS / exit 0 before anything else.
.venv/bin/python docs/research/kbound/gapclose_wave5/radius_jackknife_plus.py
.venv/bin/python docs/research/kbound/gapclose_wave5/val_jackknife_plus.py
.venv/bin/python docs/research/kbound/gapclose_wave5/val_estimator_v2.py
.venv/bin/python docs/research/kbound/gapclose_wave5/val_tau_adaptive.py

## PHASE 1 — CPU re-analysis of LOGGED data (arms A, B, C; minutes each; scored ONCE).
.venv/bin/python docs/research/kbound/gapclose_wave5/rerun_A_jkplus_logged.py --run-dir experiments/kbound/results/natural_win_v2_camelyon --dataset camelyon17 --alpha 0.10
.venv/bin/python docs/research/kbound/gapclose_wave5/rerun_A_jkplus_logged.py --run-dir experiments/kbound/results/natural_win_v1_imagenetr --dataset imagenet-r --panel --alpha 0.10
.venv/bin/python docs/research/kbound/gapclose_wave5/rerun_BC_logged.py --arm BC --dataset camelyon17 --run-dir experiments/kbound/results/natural_win_v2_camelyon

## PHASE 2 — GPU runs (arms D, E). Disk: arm D ~35-85 MB; arm E same size as existing seeds.
# Arm D: per-sample stress run, Tent, seeds 0-4 (official POEM/AETTA head-to-head)
source ~/.venv_wilds/bin/activate
export TMPDIR="${KBOUND_EXTERNAL_ROOT:?set KBOUND_EXTERNAL_ROOT}"/tmp \
       TORCH_HOME="$KBOUND_EXTERNAL_ROOT"/torch_cache
mkdir -p "$TMPDIR" "$TORCH_HOME"
for S in 0 1 2 3 4; do
  caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
    --methods tent --device mps --seed "$S" --log-samples \
    --out-results experiments/kbound/results/stress_persample_v1
done
# Arm E: seed extension 5-9, full method set (pooled 10-seed CIs; ALL seeds enter the pool)
for S in 5 6 7 8 9; do
  caffeinate -is python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
    --benchmarks cifar10c --quick --data-root experiments/kbound/cifar \
    --methods tent eata sar --device mps --seed "$S" \
    --out-results experiments/kbound/results/stress_grid_multiseed_v1/seed$S
done

## PHASE 3 — CPU scoring (scored ONCE each).
export PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds:$PWD/experiments/kbound/poem_aetta"
.venv/bin/python experiments/kbound/poem_aetta/score_official_headtohead.py \
  --run-dir experiments/kbound/results/stress_persample_v1 \
  --dataset cifar10c --adapter tent --seeds 0 1 2 3 4 --nboot 10000
# Arm E pooled 10-seed re-score: paste the seed 5-9 completion note back to the session;
# the pooled bootstrap uses the existing scripts/percondition_bootstrap.py over
# stress_grid_multiseed_v1/seed{0..9} (exact invocation confirmed at fold-in time).

## PHASE 4 — fold-in (session-side)
# Arm F (composite jk+ + estimator-v2 on natural protocols) is wired AFTER arms A and B
# validators + reruns pass, per the yaml. Paste every PASS/verdict output back; results fold
# into the paper per the frozen replacement policy (CI-robust improvement or reported
# alongside as KGA-v2), then tables + both PDFs regenerate.

## Verdict files this wave writes (report-all, no exceptions):
# research_lock/WIN_HUNT_v4_ARM_A_validator.json, _ARM_A_<ds>_result.json
# research_lock/WIN_HUNT_v4_ARM_B_validator.json, _ARM_B_<ds>_result.json
# research_lock/WIN_HUNT_v4_ARM_C_validator.json, _ARM_C_<ds>_result.json
# research_lock/WIN_HUNT_v4_ARM_D_result.json
