#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# smoke_kbound.sh -- zero-dependency CPU smoke for the K-Bound / KGA algorithm.
#
# This runs the hermetic trichotomy smoke (<60s, CPU-only) that needs NO
# external data and NO torch: it generates a tiny deterministic synthetic score
# archive and exercises the real `kga` package (evidence -> certificate ->
# ADAPT/FREEZE/ABSTAIN), asserting that the helpful / harmful / non-identifiable
# tasks map to ADAPT / FREEZE / ABSTAIN respectively. It exits non-zero on any
# failure and prints "SMOKE PASS" with a one-line metrics summary on success.
#
# Usage:
#   bash scripts/smoke_kbound.sh
#   PYTHON=python3.11 bash scripts/smoke_kbound.sh   # override interpreter
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root as the parent of this script's directory, then cd into it so
# the relative module path below is stable regardless of caller cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PYTHON:-python3}"

echo "[smoke_kbound] repo root: ${REPO_ROOT}"
echo "[smoke_kbound] python:    $("${PY}" --version 2>&1)"
echo "[smoke_kbound] running hermetic KGA trichotomy smoke (no external data)..."

"${PY}" src/scripts/kbound/smoke_trichotomy.py

echo "[smoke_kbound] done."
