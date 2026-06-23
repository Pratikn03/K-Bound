#!/usr/bin/env bash
# Wrapper — launches Protocol B v2 GPU run from repo root.
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec "$ROOT/experiments/kbound/results/camelyon17_fullscale_B_v2/protocol_b_v2_launch.sh" "$@"
