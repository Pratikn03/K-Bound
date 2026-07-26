#!/usr/bin/env bash
# Run all Wave 4 theory_v2 validators + kga routing contract selftest.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
TV2="${REPO}/docs/research/kbound/theory_v2"
# Interpreter resolution (fix-queue items 8 + 30).  This used to be
#   PY="${TV2}/.venv/bin/python"; [[ -x $PY ]] || PY="${REPO}/.venv/bin/python"
# with no fallback, so on any checkout without a committed .venv every line
# below failed with "No such file or directory" -- and because
# reproduce_submission.sh called this script under `set -e`, that killed the
# whole reproduction.  Honour $PYTHON first, then the two venvs, then python3.
PY="${PYTHON:-}"
if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  if [[ -x "${TV2}/.venv/bin/python" ]]; then
    PY="${TV2}/.venv/bin/python"
  elif [[ -x "${REPO}/.venv/bin/python" ]]; then
    PY="${REPO}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "No Python interpreter found. Set PYTHON or install python3." >&2
    exit 1
  fi
fi
echo "interpreter: ${PY}"
cd "${TV2}"
for s in val_tight_constants.py val_multiclass_multicandidate.py val_anytime_multicandidate.py \
         val_multiclass_capacity.py val_margin_computability.py val_regression_bracketing_closure.py; do
  echo "::group::${s}"
  "${PY}" "${s}" --fast 2>/dev/null || "${PY}" "${s}"
  echo "::endgroup::"
done
cd "${REPO}"
"${PY}" docs/research/kbound/theory_v2/val_minimax_optimality.py --part all
"${PY}" docs/research/kbound/scripts/multicandidate_decide_kga.py --selftest
echo "ALL theory_v2 + routing engineering checks PASS"
