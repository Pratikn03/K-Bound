#!/usr/bin/env bash
# Item 11: official AETTA (vendored) + official POEM (cloned, pinned) on the locked
# CIFAR-10-C stress stream, exporting per-condition decisions for
# scripts/official_baselines_headtohead.py --decisions ...
#
# GPU required. No number produced here becomes evidence until the head-to-head harness
# replays the pre-registered criterion (MIXED_BENCHMARK_PROTOCOL.md).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
OUT="$REPO/experiments/kbound/results/official_repro_v1"
mkdir -p "$OUT"

# --- 1) Official AETTA (vendored at $REPO/AETTA, Lee et al. 2024) --------------------
# Uses the authors' env + entry points; restrict to the locked stream's corruptions/severities.
# See AETTA/README.md: conda env create -f aetta.yml; bash download_cifar10c.sh
if [ ! -d "$REPO/AETTA/eval_results" ]; then
  echo "[item11] Run the vendored AETTA first (their env, their tta.sh) with:"
  echo "         BASE_DATASETS=cifar10outdist METHODS=TENT SEEDS=0..4, dropout estimator ON."
  echo "         Then re-run this script to convert logs -> decisions."
else
  python3 "$K/runbooks/convert_official_logs_to_decisions.py" \
    --method aetta --logs "$REPO/AETTA/eval_results" \
    --stream "$REPO/experiments/kbound/results/per_condition_cifar10c_tent_seed0.json" \
    --out "$OUT/aetta_decisions.json"
fi

# --- 2) Official POEM (Bar, Shaer, Romano, NeurIPS 2024) ------------------------------
# Official repo: https://github.com/yarinbar/poem  (arXiv:2408.07511)
POEM_DIR="$REPO/external/poem"
if [ ! -d "$POEM_DIR" ]; then
  git clone https://github.com/yarinbar/poem "$POEM_DIR"
fi
( cd "$POEM_DIR" && git rev-parse HEAD > "$OUT/poem_commit.txt" )   # pin the commit
echo "[item11] POEM pinned at commit: $(cat "$OUT/poem_commit.txt")"
if [ -f "$POEM_DIR/decisions_per_condition.json" ]; then
  python3 "$K/runbooks/convert_official_logs_to_decisions.py" \
    --method poem --logs "$POEM_DIR/decisions_per_condition.json" \
    --stream "$REPO/experiments/kbound/results/per_condition_cifar10c_tent_seed0.json" \
    --out "$OUT/poem_decisions.json"
else
  echo "[item11] Run POEM's protector over each locked condition's batch sequence"
  echo "         (their cdf.py + protector.py; martingale fires => freeze), dump"
  echo "         decisions_per_condition.json in the repo, then re-run this script."
fi

echo "[item11] When both decision files exist:"
echo "  python3 $K/scripts/official_baselines_headtohead.py \\"
echo "    --decisions poem=$OUT/poem_decisions.json aetta=$OUT/aetta_decisions.json"
