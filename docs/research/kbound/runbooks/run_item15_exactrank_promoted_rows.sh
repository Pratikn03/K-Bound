#!/usr/bin/env bash
# Item 15: rerun every PROMOTED certificate row with the released exact-rank radius
#   eps = r_(k), k = min{n, ceil((n+1)(1-alpha))}
# on the SAME frozen evidence/estimator artifacts, then diff verdict/regret/FA_u against the
# promoted (archived interpolated-quantile) rows. CPU-only for the grid rows (logged evidence);
# natural-track reruns reuse saved records.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
OUT="$REPO/experiments/kbound/results/exactrank_rerun_v1"
PY="${PY:-python3}"
mkdir -p "$OUT"

# --- Grid rows: CIFAR-10-C Tent/EATA (SAR gated, see below) ---------------------------
# ablation_exactrank.py already implements the exact-rank rule over the logged per-condition
# evidence and prints the anchor against the locked gate row.
"$PY" "$K/scripts/ablation_exactrank.py" 2>&1 | tee "$OUT/cifar_exactrank.log"

# --- SAR stress-grid row: stays WITHHELD until the clean five-seed tree is rebuilt ------
if [ "${SAR_REBUILD:-0}" = "1" ]; then
  echo "[item15] SAR rebuild enabled: rerun the five-seed stress grid from the immutable tree"
  echo "         (RUNSHEET_WAVE7.md §1a with --methods sar), then rerun this script."
else
  echo "[item15] SAR row skipped (set SAR_REBUILD=1 only when the clean five-seed tree exists)."
fi

# --- Natural-track promoted rows: Camelyon17, iWildCam, Office-Home, RxRx1 -------------
# score_kbound_holdout.py rescoring against the saved cal/test records; the released
# kbound_pkg rule is exact-rank. Wire --cal-records/--test-records per track from the
# manifest (paper/generated/kbound_result_manifest.json) — paths differ per protocol lock.
"$PY" - <<'EOF'
import json, os
man = "docs/research/kbound/paper/generated/kbound_result_manifest.json"
if os.path.exists(man):
    d = json.load(open(man))
    print("[item15] natural-track record paths from the manifest:")
    for k, v in (d.items() if isinstance(d, dict) else []):
        s = json.dumps(v)[:140]
        if any(t in k.lower() for t in ("camelyon", "iwildcam", "office", "rxrx")):
            print(f"  {k}: {s}")
else:
    print("[item15] manifest not found — check paper/generated/")
EOF
echo "[item15] For each natural track:"
echo "  $PY $K/scripts/score_kbound_holdout.py --cal-records <lock> --test-records <lock> \\"
echo "      --candidate <adapter> --conformal global --output-dir $OUT/<track>_exactrank"
echo "  (verify the scorer's radius code path is the packaged exact-rank rule before promoting)"

# --- Diff report ------------------------------------------------------------------------
echo "[item15] Compare each rerun verdict/regret/FA_u to the promoted rows"
echo "         (tab:decisive, tab:uniform-panel, tab:primary-numeric); write report to"
echo "         $OUT/report.json. Verdicts unchanged => relabel rows 'exact-rank' in the paper"
echo "         and delete the interpolated-quantile caveats; any flip => exact-rank number wins."
