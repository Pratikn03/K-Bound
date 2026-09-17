#!/usr/bin/env bash
# Consume an existing authenticated AETTA package on the legacy CIFAR/TENT panel.
# This does not execute a native baseline, create an audit/witness, or close the
# ImageNet-C all-method common-panel requirement.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
K="$REPO/docs/research/kbound"
OUT="${KBOUND_OFFICIAL_OUT:-$REPO/experiments/kbound/results/official_repro_v1}"
PY="${KBOUND_PYTHON:-python3}"
STREAM="${KBOUND_CIFAR_STREAM:-$OUT/locked_stream.json}"
AUDIT="${KBOUND_OFFICIAL_AUDIT:-$OUT/OFFICIAL_BASELINE_AUDIT.json}"
ENVIRONMENT="${KBOUND_OFFICIAL_ENVIRONMENT_RECEIPT:-$OUT/environment_receipt.json}"
TOOLCHAIN="${KBOUND_OFFICIAL_TOOLCHAIN_RECEIPT:-$OUT/toolchain_receipt.json}"
VERIFICATION="${KBOUND_OFFICIAL_VERIFICATION:-$OUT/OFFICIAL_BASELINE_VERIFICATION.json}"
DECISIONS="$OUT/aetta_decisions.json"
SCORE="$OUT/cifar10c_headtohead.json"

# Raw-log conversion must be reviewed separately; the legacy converter's
# unverified schema-2 output cannot replace an existing schema-3 evidence package.
if [[ -n "${AETTA_LOG_JSON:-}" || -n "${POEM_LOG_JSON:-}" ]]; then
  echo "FATAL: this consumer requires existing schema-3 AETTA decisions and an independent provenance audit." >&2
  echo "Raw-log conversion and native ImageNet-C POEM execution are separate workflows." >&2
  exit 2
fi
for INPUT in "$AUDIT" "$DECISIONS" "$STREAM" "$ENVIRONMENT" "$TOOLCHAIN"; do
  [[ -f "$INPUT" ]] || { echo "FATAL: required audit artifact missing: $INPUT" >&2; exit 2; }
done
if [[ -e "$SCORE" || -L "$SCORE" ]]; then
  echo "FATAL: score output already exists; preserve the earlier score: $SCORE" >&2
  exit 2
fi

echo "[item11] verify existing AETTA artifacts on the legacy CIFAR/TENT stream"
"$PY" "$K/scripts/audit_official_baselines.py" \
  --repo "$REPO" --out-dir "$OUT" --audit "$AUDIT" --output "$VERIFICATION" \
  --method aetta --stream "$STREAM" --environment-receipt "$ENVIRONMENT" \
  --toolchain-receipt "$TOOLCHAIN" --require-promotable

"$PY" "$K/scripts/official_baselines_headtohead.py" \
  --candidate tent --stream "$STREAM" --decisions "aetta=$DECISIONS" \
  --provenance-audits "aetta=$AUDIT" --out "$SCORE"

echo "[item11] existing audit: $AUDIT"
echo "[item11] verification: $VERIFICATION"
echo "[item11] legacy CIFAR/TENT score: $SCORE; ImageNet-C common-panel completeness remains separate"
