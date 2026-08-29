#!/usr/bin/env bash
# Fail-closed preparation/audit for official-code POEM and AETTA comparisons.
# This script never clones, edits, or silently upgrades an external method.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
OUT="${KBOUND_OFFICIAL_OUT:-$REPO/experiments/kbound/results/official_repro_v1}"
PY="${KBOUND_PYTHON:-python3}"
STREAM="${KBOUND_CIFAR_STREAM:-$REPO/experiments/kbound/results/stress_persample_v1/per_condition_cifar10c_tent_seed0.json}"
AUDIT="$OUT/OFFICIAL_BASELINE_AUDIT.json"
mkdir -p "$OUT"

[[ -f "$STREAM" ]] || { echo "FATAL: locked CIFAR stream missing: $STREAM" >&2; exit 2; }
[[ -d "$REPO/AETTA" ]] || { echo "FATAL: vendored AETTA source missing" >&2; exit 2; }
[[ -d "$REPO/external/poem/.git" ]] || {
  echo "FATAL: pinned POEM checkout missing; clone it explicitly and review its license/commit" >&2
  exit 2
}

echo "[item11] initial provenance audit"
"$PY" "$K/scripts/audit_official_baselines.py" --repo "$REPO" --out-dir "$OUT" --output "$AUDIT"

# Conversion is opt-in because native log formats are method/version specific.
# AETTA_LOG_JSON must be the exported condition -> estimated-accuracy JSON from
# the authors' entry point, not the protocol-matched K-Bound port.
if [[ -n "${AETTA_LOG_JSON:-}" ]]; then
  [[ -f "$AETTA_LOG_JSON" ]] || { echo "FATAL: AETTA_LOG_JSON missing: $AETTA_LOG_JSON" >&2; exit 2; }
  "$PY" "$K/runbooks/convert_official_logs_to_decisions.py" \
    --method aetta --logs "$AETTA_LOG_JSON" --stream "$STREAM" \
    --provenance-audit "$AUDIT" --out "$OUT/aetta_decisions.json"
fi

# POEM's official entry point is ImageNet-C-only in this pinned checkout. It
# must not be mapped onto the CIFAR stream. Run the ImageNet-C driver and its
# dedicated converter when that protocol adapter is complete.
if [[ -n "${POEM_LOG_JSON:-}" ]]; then
  echo "FATAL: POEM_LOG_JSON cannot be scored against the CIFAR stream." >&2
  echo "Use run_poem_imagenetc.sh and a locked ImageNet-C protocol adapter." >&2
  exit 2
fi

echo "[item11] final provenance audit"
"$PY" "$K/scripts/audit_official_baselines.py" --repo "$REPO" --out-dir "$OUT" --output "$AUDIT"

DECISIONS=()
if [[ -s "$OUT/aetta_decisions.json" ]]; then
  DECISIONS+=("aetta=$OUT/aetta_decisions.json")
fi

if (( ${#DECISIONS[@]} )); then
  "$PY" "$K/scripts/official_baselines_headtohead.py" \
    --candidate tent --decisions "${DECISIONS[@]}" \
    --out "$OUT/cifar10c_headtohead.json"
else
  echo "[item11] no complete converted official-code decision file; port labels remain"
fi

echo "[item11] audit: $AUDIT"
echo "[item11] require full promotion with:"
echo "  $PY $K/scripts/audit_official_baselines.py --repo $REPO --out-dir $OUT --require-promotable"
