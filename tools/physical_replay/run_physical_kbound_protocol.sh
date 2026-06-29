#!/usr/bin/env bash
# Wrapper: full physical K-Bound protocol pipeline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh" "$@"
