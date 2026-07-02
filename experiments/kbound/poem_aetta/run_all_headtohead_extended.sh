#!/usr/bin/env bash
# run_all_headtohead_extended.sh
# -----------------------------------------------------------------------------
# Extends the pre-registered KGA-vs-POEM-vs-AETTA head-to-head beyond the
# CIFAR-10-C primary set to:
#   (A) the full CACHED CIFAR-10-C suite  -- tent (primary), eata, tent+eata,
#       and SAR (benign coverage row); all run NOW from cached records.
#   (B) any ADDITIONAL dataset whose per-condition records use the SAME 11-dim
#       evidence panel POEM/AETTA require (baselines.Z_NAMES). A set with no
#       compatible records is reported ABSENT with the exact command to enable
#       it -- it is NEVER silently dropped (protocol sec 5).
#
# Pre-registration:
#   docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md       (original)
#   docs/research/kbound/MIXED_BENCHMARK_EXT_PROTOCOL.md   (this extension; sets +
#                                                           win criterion + honest
#                                                           expected verdicts)
#
# Integrity: the cached arm is pure numpy (no torch). KGA's decision is taken from
# the cached `kga_decision` field; POEM/AETTA are faithful ports (baselines.py).
# The run decides WIN/TIE/LOSE; we do not. WIN = beats POEM AND AETTA (Holm) at
# false-adapt <= alpha. A 0%-harmful (benign) set where KGA loses to always-adapt
# is a COVERAGE row, not a headline win -- the summary labels it as such.
#
# Overrides via env: REPO, OUT_DIR, SEEDS, PY.
# -----------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/Volumes/T9/uav/AutoML_Flagship_V8}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/kbound/results/mixed_headtohead_ext_v1}"
SEEDS="${SEEDS:-0 1 2 3 4}"
PY="${PY:-$REPO/.venv/bin/python}"; [[ -x "$PY" ]] || PY=python3

cd "$REPO"
export PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds:$PWD/experiments/kbound/poem_aetta"
H="experiments/kbound/poem_aetta/run_mixed_headtohead.py"
FS=$(echo $SEEDS | awk '{print $1}')   # first seed (for the compatibility probe)
mkdir -p "$OUT_DIR"

echo "############################################################################"
echo "# EXTENDED MIXED HEAD-TO-HEAD  (cached CIFAR suite + auto-detected extras)"
echo "#   out   : $OUT_DIR"
echo "#   seeds : $SEEDS"
echo "#   note  : run decides WIN/TIE/LOSE; absent sets are flagged, never dropped"
echo "############################################################################"

# --- compatibility probe: 0 iff records exist AND Z_names == baselines.Z_NAMES ---
compatible(){ # args: records_dir dataset adapter firstseed
  "$PY" - "$@" <<'PY'
import sys, os, json
rd, ds, ad, s = sys.argv[1:5]
sys.path.insert(0, os.path.join(os.getcwd(), "experiments/kbound/poem_aetta"))
try:
    import baselines as BL; need = list(BL.Z_NAMES)
except Exception as e:
    print("NOBASE", e); sys.exit(2)
for p in (os.path.join(rd, f"seed{s}", f"per_condition_{ds}_{ad}_seed{s}.json"),
          os.path.join(rd, f"per_condition_{ds}_{ad}_seed{s}.json")):
    if os.path.exists(p):
        try:
            r = json.load(open(p))["records"][0]
            sys.exit(0 if r.get("Z_names") == need else 3)
        except Exception:
            sys.exit(4)
sys.exit(1)   # no records found
PY
}

run_set(){ # args: dataset records_dir set_name <adapter flags...>
  local ds="$1" rd="$2" name="$3"; shift 3
  "$PY" "$H" --records-dir "$rd" --dataset "$ds" "$@" \
       --seeds $SEEDS --out-dir "$OUT_DIR" --set-name "$name"
}

echo; echo "### [0] synthetic apparatus check (proves machinery, decides nothing) ###"
"$PY" experiments/kbound/poem_aetta/verify_headtohead.py >/dev/null 2>&1 \
  && echo "apparatus OK" || echo "apparatus check FAILED (investigate before trusting verdicts)"

# ---------------------------------------------------------------------------
# A. CACHED CIFAR-10-C suite (records present -> runs now)
# ---------------------------------------------------------------------------
RD_C="$REPO/experiments/kbound/results/stress_grid_multiseed_v1"
echo; echo "### [A] CIFAR-10-C cached suite ###"
run_set cifar10c "$RD_C" cifar10c_tent_primary     --adapter tent
run_set cifar10c "$RD_C" cifar10c_eata_secondary   --adapter eata
run_set cifar10c "$RD_C" cifar10c_tent_eata_pooled --pool-adapters tent eata
run_set cifar10c "$RD_C" cifar10c_sar_secondary    --adapter sar   # 0% harmful -> COVERAGE row

# ---------------------------------------------------------------------------
# B. EXTENSION datasets -- run iff compatible 11-dim-panel records exist.
#    Add a line "dataset|records_dir|adapters" per dataset. Point records_dir at
#    the per_condition_*_seed<S>.json files once generated (see EXT protocol).
# ---------------------------------------------------------------------------
declare -a EXT=(
  "imagenetc|$REPO/experiments/kbound/results/imagenetc_stress_grid_v1|tent eata sar"
  # "officehome|$REPO/experiments/kbound/results/officehome_h2h_v1|deployed"   # needs 11-dim-panel regen (see EXT protocol)
)
echo; echo "### [B] extension datasets (auto-detected) ###"
for spec in "${EXT[@]}"; do
  IFS='|' read -r ds rd ads <<<"$spec"
  for ad in $ads; do
    if compatible "$rd" "$ds" "$ad" "$FS"; then
      echo; echo "  -> ${ds}/${ad}: compatible records found; running"
      run_set "$ds" "$rd" "${ds}_${ad}" --adapter "$ad"
    else
      rc=$?
      case $rc in
        1) why="no per_condition records under $rd";;
        3) why="records exist but Z_names != the 11-dim POEM/AETTA panel (regen needed)";;
        4) why="records unreadable";;
        *) why="probe error (code $rc)";;
      esac
      echo; echo "  -> ${ds}/${ad}: ABSENT -- not run, not dropped. reason: $why"
      echo "     enable: generate per_condition_${ds}_${ad}_seed<S>.json with the 11-dim"
      echo "     evidence panel Z=[pre_entropy,pre_conf,pre_pbal,post_entropy,post_conf,"
      echo "     post_pbal,pbal_drop,entropy_drop,frac_highconf,marginal_KL,update_norm] + B,"
      echo "     under $rd , then re-run. See MIXED_BENCHMARK_EXT_PROTOCOL.md sec 3."
    fi
  done
done

# ---------------------------------------------------------------------------
# C. combined honest summary
# ---------------------------------------------------------------------------
echo; echo "############ COMBINED VERDICT SUMMARY ############"
"$PY" - "$OUT_DIR" <<'PY'
import sys, glob, os, json
od = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(od, "HEADTOHEAD_RESULTS_*.json"))):
    try: J = json.load(open(f))
    except Exception: continue
    name = os.path.basename(f).replace("HEADTOHEAD_RESULTS_", "").replace(".json", "")
    hh = J.get("headtohead", {}); hh = hh if isinstance(hh, dict) else {}
    verdict = str(hh.get("VERDICT", "?"))
    bt = hh.get("legacy_beats_both_trivials", None)
    hr = J.get("harmful_base_rate_range", None)
    try: hr_s = f"{hr[0]*100:.0f}-{hr[1]*100:.0f}%"
    except Exception: hr_s = str(hr)
    if verdict == "WIN" and bt is True:
        label = "WIN (headline: beats SOTA + both trivials)"
    elif verdict == "WIN" and bt is False:
        label = "COVERAGE (benign; beats SOTA but loses to always-adapt)"
    else:
        label = verdict
    rows.append((name, verdict, str(bt), hr_s, label))
if not rows:
    print("(no HEADTOHEAD_RESULTS_*.json found in", od, ")")
else:
    w = max([len(r[0]) for r in rows] + [16])
    print(f"{'set'.ljust(w)}  verdict  beats_triv  harmful  honest_label")
    for n, v, b, h, lab in rows:
        print(f"{n.ljust(w)}  {v.ljust(7)}  {b.ljust(10)}  {h.ljust(7)}  {lab}")
    print("\nWIN = beats POEM AND AETTA (Holm) at FA<=alpha.  beats_triv = also beats BOTH")
    print("trivial policies.  A WIN with beats_triv=False on a ~0%-harmful panel is benign")
    print("COVERAGE (always-adapt is trivially optimal there), NOT a headline beats-both win.")
PY
echo
echo "Results + per-policy arrays: $OUT_DIR/HEADTOHEAD_RESULTS_*.json"
echo "Done."
