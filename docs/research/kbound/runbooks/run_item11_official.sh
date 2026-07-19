#!/usr/bin/env bash
# Item 11 orchestrator: OFFICIAL AETTA + OFFICIAL POEM -> per-condition decisions ->
# K-Bound head-to-head harness. Phased, restartable, logs everything.
#   Phase A: official AETTA (vendored repo, conda env 'aetta'):
#            data symlinks -> source-model training (if missing) -> TENT+aetta estimator run
#            -> Src (frozen) reference run -> print_est dump -> adapter -> aetta_decisions.json
#   Phase B: official POEM: clone+pin -> conda env 'poem' -> pip install -> dump --help
#            -> STOP with NEEDS_WIRING_POEM marker (exact run flags wired after inspection;
#            no guessed flags — no fabricated protocol).
#   Phase C: if decisions exist -> official_baselines_headtohead.py (Holm, CIs, FA_u).
# Monitor:  tail -f experiments/kbound/results/official_repro_v1/item11.log
set -u
R="$HOME/Documents/AutoML_Flagship_V8"
A="$R/AETTA"
K="$R/docs/research/kbound"
OUT="$R/experiments/kbound/results/official_repro_v1"
LOG="$OUT/item11.log"
mkdir -p "$OUT"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "ITEM11 ORCHESTRATOR START"

# ---------- Phase A: official AETTA ----------
say "A1: dataset symlinks"
mkdir -p "$A/dataset"
[ -e "$A/dataset/CIFAR-10-C" ] || ln -s "$R/experiments/kbound/cifar/CIFAR-10-C" "$A/dataset/CIFAR-10-C"
[ -e "$A/dataset/cifar-10-batches-py" ] || ln -s "$R/experiments/kbound/cifar/cifar-10-batches-py" "$A/dataset/cifar-10-batches-py"

find_ckpt() { find "$A/log" "$A/reproduce_src" \( -name "*.pth" -o -name "cp*.pth.tar" \) 2>/dev/null | head -1; }
CKPT_FOUND=$(find_ckpt)
if [ -z "$CKPT_FOUND" ] && pgrep -f "main.py.*method Src" >/dev/null 2>&1; then
  say "A2: source training already running elsewhere -> waiting for checkpoint (poll 2 min, max 12 h)"
  for i in $(seq 1 360); do
    sleep 120
    CKPT_FOUND=$(find_ckpt)
    [ -n "$CKPT_FOUND" ] && break
    pgrep -f "main.py.*method Src" >/dev/null 2>&1 || { say "A2: external training exited without checkpoint"; break; }
  done
fi
if [ -z "$CKPT_FOUND" ]; then
  say "A2: no source checkpoint -> training source model (single seed 0); this is the long step (~6-7 h)"
  sed 's/for SEED in 0 1 2; do/for SEED in 0; do/' "$A/train_src.sh" > "$A/train_src_item11.sh"
  ( cd "$A" && caffeinate -is conda run -n aetta bash train_src_item11.sh ) >> "$LOG" 2>&1
  say "A2: train_src_item11.sh rc=$?"
  CKPT_FOUND=$(find_ckpt)
fi
say "A2: source checkpoint: ${CKPT_FOUND:-STILL MISSING}"

say "A3: TTA runs — TENT + aetta estimator (seed 0), then Src frozen reference"
for M in TENT Src; do
  sed -e 's/^METHODS=.*/METHODS=("'"$M"'")/' \
      -e 's/^SEEDS=.*/SEEDS=(0)/' \
      "$A/tta.sh" > "$A/tta_item11_$M.sh"
  ( cd "$A" && caffeinate -is conda run -n aetta bash "tta_item11_$M.sh" ) >> "$LOG" 2>&1
  say "A3: method $M rc=$?"
done

say "A4: estimator dump + adapter"
( cd "$A" && conda run -n aetta python print_est.py --dataset cifar10outdist --target aetta ) \
  > "$OUT/aetta_print_est_raw.txt" 2>>"$LOG"
say "A4: print_est raw -> $OUT/aetta_print_est_raw.txt"
# Adapter accepts official CSV/JSON; try raw dump first, else leave marker for csv assembly.
conda run -n aetta python3 "$K/scripts/baseline_decisions_adapter.py" --method aetta \
    --input "$OUT/aetta_print_est_raw.txt" --out "$OUT/aetta_decisions.json" >> "$LOG" 2>&1 \
  && say "A4: aetta_decisions.json WRITTEN" \
  || { say "A4: adapter needs csv assembly -> NEEDS_WIRING_AETTA_CSV (raw dump saved)"; \
       touch "$OUT/NEEDS_WIRING_AETTA_CSV"; }

# ---------- Phase B: official POEM ----------
say "B1: clone + pin POEM"
P="$R/external/poem"
if [ ! -d "$P" ]; then git clone https://github.com/yarinbar/poem "$P" >> "$LOG" 2>&1; fi
( cd "$P" && git rev-parse HEAD > "$OUT/poem_commit.txt" ) 2>>"$LOG"
say "B1: POEM commit $(cat "$OUT/poem_commit.txt" 2>/dev/null)"

say "B2: conda env 'poem' (create if missing) + requirements"
if ! conda env list | grep -q "^poem "; then
  conda create -y -n poem python=3.10 >> "$LOG" 2>&1
fi
( cd "$P" && conda run -n poem pip install -r requirements.txt ) >> "$LOG" 2>&1
say "B2: pip rc=$?"

say "B3: dump POEM CLI for exact wiring (no guessed flags)"
( cd "$P" && conda run -n poem python main.py --help ) > "$OUT/poem_help.txt" 2>&1
grep -n "dataset\|cifar" "$P/main.py" | head -20 > "$OUT/poem_dataset_hints.txt" 2>/dev/null
touch "$OUT/NEEDS_WIRING_POEM"
say "B3: poem_help.txt + hints saved -> NEEDS_WIRING_POEM (next: wire cifar10 run flags)"

# ---------- Phase C: harness (runs with whatever decisions exist) ----------
DEC=""
[ -f "$OUT/aetta_decisions.json" ] && DEC="aetta=$OUT/aetta_decisions.json"
[ -f "$OUT/poem_decisions.json" ] && DEC="$DEC poem=$OUT/poem_decisions.json"
if [ -n "$DEC" ]; then
  say "C: harness with: $DEC"
  cd "$R" && conda run -n aetta python3 "$K/scripts/official_baselines_headtohead.py" --candidate tent \
    --decisions $DEC >> "$LOG" 2>&1
  say "C: harness rc=$? -> experiments/kbound/results/official_headtohead.json"
else
  say "C: no decisions files yet; harness deferred"
fi
say "ITEM11 ORCHESTRATOR DONE (check NEEDS_WIRING_* markers in $OUT)"
