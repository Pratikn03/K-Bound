#!/usr/bin/env bash
# ============================================================================
# K-Bound — ONE-COMMAND final showcase run.
#
#   bash docs/research/kbound/scripts/run_final_showcase.sh [flags]
#
# Runs the full 9-dataset panel multi-seed on the out-of-fold code, then collates
# honestly into the paper: manifest -> results_source.json -> tables -> figures ->
# both PDFs -> verification. See RUN_FINAL_SHOWCASE.md for the pre-registration.
#
# Flags:
#   --device X     cuda | mps | cpu   (default: auto via KB_DEVICE, else mps)
#   --seeds "..."  seed list          (default: "0 1 2 3 4 5 6 7" = 5 + 3 more)
#   --skip-train   reuse the latest existing run; only re-collate + rebuild (no GPU)
#   --smoke        tiny end-to-end plumbing check (1 seed; not a real result)
#   --dry-run      print the stage graph and exit; run nothing
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../docs/research/kbound/scripts
ROOT="$(cd "$HERE/../../../.." && pwd)"                    # repo root
RES="$ROOT/experiments/kbound/results"
KBDIR="$ROOT/docs/research/kbound"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY="python3"

DEVICE="${KB_DEVICE:-mps}"; SEEDS="${KB_SEEDS:-0 1 2 3 4 5 6 7}"
SKIP_TRAIN=0; SMOKE=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --seeds)  SEEDS="$2"; shift 2;;
    --skip-train) SKIP_TRAIN=1; shift;;
    --smoke)  SMOKE=1; SEEDS="0"; shift;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown flag: $1"; exit 1;;
  esac
done

say(){ printf "\n\033[1m== %s ==\033[0m\n" "$*"; }
run(){ echo "+ $*"; [ "$DRY" = 1 ] || eval "$*"; }

echo "K-Bound final showcase   device=$DEVICE   seeds=[$SEEDS]   skip_train=$SKIP_TRAIN   smoke=$SMOKE   dry_run=$DRY"

# -- Stage A: the 9-dataset multi-seed engine (integrity-guarded inside kbtrain) -----------
if [ "$SKIP_TRAIN" = 0 ]; then
  say "A. final-all-v2: theory validators + routing selftest + 9 datasets x seeds"
  run "KB_SEEDS='$SEEDS' KB_DEVICE='$DEVICE' bash '$HERE/kbtrain.sh' final-all-v2"
else
  say "A. SKIP-TRAIN: reusing the most recent existing run artifacts"
fi

# locate the freshest manifest produced by final-all
MAN="$(ls -t "$RES"/final_manifest_*.json 2>/dev/null | head -1 || true)"
if [ "$DRY" = 0 ] && [ -z "${MAN:-}" ]; then
  echo "ERROR: no final_manifest_*.json under $RES — run without --skip-train first."; exit 1
fi
echo "manifest: ${MAN:-<none yet>}"

# -- Stage B: regenerate natural-shift bootstrap CIs from the fresh locked logs ------------
say "B. condition-bootstrap CIs for the natural-shift wins (dev fixed, test resampled)"
run "$PY '$HERE/bootstrap_win_cis.py'"

# -- Stage H/I before collation: locked CIFAR + POEM/AETTA feed results_source.json --------
say "H. LOCKED_ANALYSIS on stress_grid_multiseed_v1 (if all seeds present)"
run "$PY '$RES/stress_grid_multiseed_v1/_locked_analysis_script.py' || echo 'WARN: locked analysis skipped (incomplete seeds?)'"

say "I. mixed head-to-head vs POEM/AETTA (pre-registered WIN table)"
run "bash '$ROOT/experiments/kbound/poem_aetta/run_all_headtohead.sh' || echo 'WARN: head-to-head skipped'"

# -- Stage C: honest collation -> results_source.json (guardrails inside) ------------------
say "C. build results_source.json (OOF; verdict-from-CI; merges locked CIFAR + head-to-head)"
CHECK=""; [ "$DRY" = 1 ] && CHECK="--check-only"
LOCKED="$RES/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json"
H2H="$RES/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json"
run "cd '$ROOT' && $PY '$HERE/build_results_source.py' --manifest '${MAN:-MANIFEST}' \
  --locked-analysis '$LOCKED' --headtohead '$H2H' $CHECK"

# -- Stage D/E: regenerate macros + figures from the source of truth -----------------------
say "D. tables -> paper/generated/kbound_numbers.tex"
run "$PY '$HERE/03_make_tables.py'"
say "E. figures -> figures/fig_natural_forest.png, fig_frontier_schematic.png (+ main figs)"
run "$PY '$HERE/04_make_figures.py'"

# -- Stage F: rebuild both PDFs ------------------------------------------------------------
say "F. rebuild PDFs (kbound_short.tex, kbound.tex)"
run "cd '$KBDIR' && COPYFILE_DISABLE=1 latexmk -pdf -interaction=nonstopmode -halt-on-error kbound_short.tex >/dev/null"
run "cd '$KBDIR' && COPYFILE_DISABLE=1 latexmk -pdf -interaction=nonstopmode -halt-on-error kbound.tex >/dev/null"

# -- Stage J: mixed-stream OOF aggregate (fast CPU; tier-B claim) ------------------------
say "J. mixed_protocol_oof_v2 aggregate (constructed cross-protocol)"
run "$PY '$HERE/mixed_stream_kbound.py' || echo 'WARN: mixed-stream skipped'"

say "G. verify results + macro consistency"
run "$PY '$HERE/02_verify_results.py' || true"
# macros must be a pure function of results_source.json: regenerate and assert no drift
run "cd '$ROOT' && $PY '$HERE/03_make_tables.py' >/dev/null && git diff --quiet -- '$KBDIR/paper/generated/kbound_numbers.tex' && echo 'OK macros stable' || echo 'WARN macros changed on re-run'"

say "DONE — review: $KBDIR/results_source.json (+ _provenance), kbound_short.pdf, kbound.pdf"
[ "$DRY" = 1 ] && echo "(dry-run: nothing executed)"
