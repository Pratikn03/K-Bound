#!/usr/bin/env bash
# Item-11 finisher: waits for the multi-seed chain to free the GPU, then re-runs the
# AETTA TTA (now that the checkpoint-path symlink is fixed) -> print_est -> adapter -> harness.
# Checkpoint fix: log/cifar10/Src/tgt_test/reproduce_src_0/cp/cp_last.pth.tar -> trained cp.
# --- interpreter: $KBOUND_PYTHON, default python3 (was a hard-coded venv path).
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

KB_PYTHON="${KBOUND_PYTHON:-python3}"

set -u
R="$KB_REPO_ROOT"; A="$R/AETTA"; K="$R/docs/research/kbound"
OUT="$R/experiments/kbound/results/official_repro_v1"; LOG="$OUT/item11_finish.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "FINISH START — waiting for multi-seed chain to free the GPU"
while pgrep -f "run_multiseed|run_officehome_kbound|run_iwildcam|run_rxrx1_kbound|run_multiseed_chain" >/dev/null 2>&1; do sleep 300; done
say "GPU idle — re-running AETTA TTA with fixed checkpoint"
[ -f "$A/log/cifar10/Src/tgt_test/reproduce_src_0/cp/cp_last.pth.tar" ] || { say "FATAL: checkpoint symlink missing"; exit 2; }
for M in TENT Src; do
  ( cd "$A" && caffeinate -is conda run -n aetta bash "tta_item11_$M.sh" ) >> "$LOG" 2>&1
  say "TTA $M rc=$?"
done
say "print_est dump"
( cd "$A" && conda run -n aetta python print_est.py --dataset cifar10outdist --target aetta ) > "$OUT/aetta_print_est_raw.txt" 2>>"$LOG"
say "print_est bytes: $(wc -c < "$OUT/aetta_print_est_raw.txt")"
conda run -n aetta python3 "$K/scripts/baseline_decisions_adapter.py" --method aetta \
  --input "$OUT/aetta_print_est_raw.txt" --out "$OUT/aetta_decisions.json" >> "$LOG" 2>&1 \
  && { rm -f "$OUT/NEEDS_WIRING_AETTA_CSV"; say "aetta_decisions.json WRITTEN"; } \
  || say "adapter still needs format fix — REAL print_est saved for manual wiring"
if [ -s "$OUT/aetta_decisions.json" ] && [ "$(tr -d '[:space:]' <"$OUT/aetta_decisions.json")" != "{}" ]; then
  cd "$R" && "$KB_PYTHON" "$K/scripts/official_baselines_headtohead.py" \
    --candidate tent --decisions "aetta=$OUT/aetta_decisions.json" >> "$LOG" 2>&1
  say "harness rc=$? -> official_headtohead.json"
else say "no valid decisions yet — harness deferred"; fi
say "FINISH DONE"
