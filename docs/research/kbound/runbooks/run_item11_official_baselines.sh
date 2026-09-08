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
ENVIRONMENT_RECEIPT="${KBOUND_OFFICIAL_ENVIRONMENT_RECEIPT:-$K/audits/python_environment_2026_09_02.json}"
TOOLCHAIN_RECEIPT="${KBOUND_OFFICIAL_TOOLCHAIN_RECEIPT:-$K/audits/release_toolchain_2026_09_05_v2.json}"

# POEM's pinned native entry point is ImageNet-C-only. Its adapter cannot be
# promoted or scored on CIFAR, even when an old decisions file exists.
if [[ -n "${POEM_LOG_JSON:-}" ]]; then
  echo "FATAL: POEM_LOG_JSON cannot be scored against the CIFAR stream." >&2
  echo "Use run_poem_imagenetc.sh and a locked ImageNet-C protocol adapter." >&2
  exit 2
fi

[[ -f "$STREAM" ]] || { echo "FATAL: locked CIFAR stream missing: $STREAM" >&2; exit 2; }
[[ -f "$ENVIRONMENT_RECEIPT" ]] || { echo "FATAL: environment receipt missing: $ENVIRONMENT_RECEIPT" >&2; exit 2; }
[[ -f "$TOOLCHAIN_RECEIPT" ]] || { echo "FATAL: toolchain receipt missing: $TOOLCHAIN_RECEIPT" >&2; exit 2; }
[[ -d "$REPO/AETTA" ]] || { echo "FATAL: vendored AETTA source missing" >&2; exit 2; }
[[ -d "$REPO/external/poem/.git" ]] || {
  echo "FATAL: pinned POEM checkout missing; clone it explicitly and review its license/commit" >&2
  exit 2
}

mkdir -p "$OUT"

# Conversion is opt-in because native log formats are method/version specific.
# AETTA_LOG_JSON must be the exported condition -> estimated-accuracy JSON from
# the authors' entry point, not the protocol-matched K-Bound port.
if [[ -n "${AETTA_LOG_JSON:-}" ]]; then
  [[ -f "$AETTA_LOG_JSON" ]] || { echo "FATAL: AETTA_LOG_JSON missing: $AETTA_LOG_JSON" >&2; exit 2; }
  "$PY" "$K/runbooks/convert_official_logs_to_decisions.py" \
    --method aetta --logs "$AETTA_LOG_JSON" --stream "$STREAM" \
    --stage --out "$OUT/aetta_decisions.staged.json"
fi

echo "[item11] schema-3 provenance audit of staged decisions and native evidence"
"$PY" "$K/scripts/audit_official_baselines.py" --repo "$REPO" --out-dir "$OUT" --output "$AUDIT" \
  --locked-stream "$STREAM" --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --toolchain-receipt "$TOOLCHAIN_RECEIPT"

# A request must pass this run's strict promotion before any scoring call.
# A stale validated artifact alone is never a trigger. Missing independent
# execution attestation fails here without replacing the previous final file.
if [[ -n "${AETTA_LOG_JSON:-}" ]]; then
  "$PY" "$K/runbooks/convert_official_logs_to_decisions.py" \
    --method aetta --logs "$AETTA_LOG_JSON" --stream "$STREAM" \
    --provenance-audit "$AUDIT" --require-official-label \
    --environment-receipt "$ENVIRONMENT_RECEIPT" --toolchain-receipt "$TOOLCHAIN_RECEIPT" \
    --out "$OUT/aetta_decisions.json"
  "$PY" "$K/scripts/official_baselines_headtohead.py" \
    --candidate tent --decisions "aetta=$OUT/aetta_decisions.json" \
    --provenance-audit "$AUDIT" --source-logs "aetta=$AETTA_LOG_JSON" \
    --locked-stream "$STREAM" --environment-receipt "$ENVIRONMENT_RECEIPT" \
    --toolchain-receipt "$TOOLCHAIN_RECEIPT" \
    --out "$OUT/cifar10c_headtohead.json"
else
  echo "[item11] audit only; no native log conversion was requested and no outcomes were scored"
fi

echo "[item11] audit: $AUDIT"
echo "[item11] require full promotion with:"
echo "  $PY $K/scripts/audit_official_baselines.py --repo $REPO --out-dir $OUT --locked-stream $STREAM --environment-receipt $ENVIRONMENT_RECEIPT --toolchain-receipt $TOOLCHAIN_RECEIPT --require-promotable"
