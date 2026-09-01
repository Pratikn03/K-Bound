#!/usr/bin/env bash
# Build the pinned formal sources and inspect the actual kernel dependencies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Builds never delete source, dataset sidecars, or dependency-cache files. Cache
# maintenance and dependency updates are separate explicit operations.
echo "Using pinned lean-toolchain and lake-manifest.json; no lake update or cleanup."
FORMAL_PYTHON="${KBOUND_PYTHON:-${PYTHON:-python3}}"
echo "Building KBound and auditing registered declarations and transitive axioms ..."
"$FORMAL_PYTHON" formal_audit.py --build --strict-core "$@"

echo ""
echo "OK: the declared core and scoped probability capstones passed kernel checks."
echo "The historical full one-bit/H extension remains outside this verified scope."
echo "For the stronger six-layer gate: python3 formal_audit.py --build --full-foundations"
echo "See KBound/TheoremMap.lean for paper-label to Lean-name mapping."
