#!/usr/bin/env bash
# Run all Wave 4 theory_v2 validators + kga routing contract selftest.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
TV2="${REPO}/docs/research/kbound/theory_v2"
PY="${TV2}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="${REPO}/.venv/bin/python"
fi
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
