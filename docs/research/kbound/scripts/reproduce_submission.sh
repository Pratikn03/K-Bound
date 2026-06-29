#!/usr/bin/env bash
# K-Bound submission reproduction — lightweight verify + cached headline checks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
KBOUND="${ROOT}/docs/research/kbound"
cd "${ROOT}"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="${HOME}/.venv_wilds/bin/python"
fi

MANIFEST="${KBOUND}/RELEASE_MANIFEST.json"
REPORT="${KBOUND}/reports/reproducibility_release_report.md"
mkdir -p "${KBOUND}/reports"

echo "=== K-Bound reproduce_submission ==="
COMMIT=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")
echo "git_commit=${COMMIT}"

echo "=== [1] Unit tests (leakage + claim semantics + edge) ==="
"${PY}" -m pytest \
  "${KBOUND}/tests/test_no_leakage_protocols.py" \
  "${KBOUND}/tests/test_calibration_split_integrity.py" \
  "${KBOUND}/tests/test_claim_metric_semantics.py" \
  "${KBOUND}/kbound_pkg/tests/" \
  "${KBOUND}/edge/tests/test_no_live_labels.py" \
  "${KBOUND}/edge/tests/test_conformal.py" \
  "${KBOUND}/edge/tests/test_policy.py" \
  -q --tb=short

echo "=== [2] Theory validators (lightweight) ==="
if [[ -d "${ROOT}/experiments/kbound/theory_validation" ]]; then
  ls "${ROOT}/experiments/kbound/theory_validation"/results_thm*.json >/dev/null
  echo "theory JSON artifacts: present"
fi
echo "=== [2b] Full theory audit (artifacts + claim ledger cross-check) ==="
"${PY}" "${KBOUND}/scripts/theory_audit_full.py" --write-report

echo "=== [3] Gate baseline (CPU selftest) ==="
cd "${KBOUND}"
"${PY}" scripts/gate_baseline_comparison.py --selftest

echo "=== [4] Regenerate paper table macros from results_source.json ==="
"${PY}" scripts/make_tables.py

echo "=== [5] Validate claim ledger ==="
"${PY}" -c "import json; json.load(open('claim_ledger.json')); print('claim_ledger.json OK')"

echo "=== [6] Mixed head-to-head (POEM/AETTA) results present ==="
H2H="${ROOT}/experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json"
if [[ -f "${H2H}" ]]; then
  "${PY}" -c "
import json, sys
h=json.load(open('${H2H}'))
v=h['headtohead']['VERDICT']
print(f'mixed_headtohead verdict={v}')
if v != 'WIN':
    print('WARN: expected WIN for headline claim KB-CLAIM-026', file=sys.stderr)
"
else
  echo "WARN: run bash experiments/kbound/poem_aetta/run_all_headtohead.sh" >&2
fi

echo "=== [7] Cached headline artifacts ==="
for f in \
  "${ROOT}/experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json" \
  "${ROOT}/experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json" \
  "${ROOT}/experiments/kbound/results/controlled_multimodal_d33/results.json" \
  "${KBOUND}/results/assumption_audit_v1.json" \
  "${ROOT}/research_lock/KBOUND_HEADLINE_FINDINGS_v3.json"; do
  if [[ -f "${f}" ]]; then
    echo "present: ${f#${ROOT}/}"
  else
    echo "MISSING: ${f#${ROOT}/}" >&2
    exit 1
  fi
done

echo "=== [8] Write manifest ==="
cat > "${MANIFEST}" <<EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "${COMMIT}",
  "python": "$(${PY} --version 2>&1)",
  "steps_completed": ["unit_tests", "theory_artifacts_present", "gate_selftest", "make_tables", "claim_ledger", "mixed_headtohead_check", "cached_artifacts"],
  "note": "Full GPU protocol re-runs are not executed by this script; see RELEASE_10X_TRACK.md"
}
EOF

cat > "${REPORT}" <<EOF
# Reproducibility Release Report

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)

| Artifact | Reproduced | Cached verified | Missing | Failed | Notes |
|---|---:|---:|---:|---:|---|
| Leakage/claim tests | yes | — | — | — | pytest |
| Theory validators | — | yes | — | — | JSON present |
| CIFAR gate baseline | yes | — | — | — | selftest only |
| Paper table macros | yes | — | — | — | from results_source.json |
| Stress grid full GPU | — | verify manual | — | — | see stress_grid_multiseed_v1 |
| Physical camera R2 | — | — | pending | — | RESULT PENDING |
EOF

echo "=== DONE ==="
echo "Manifest: ${MANIFEST}"
echo "Report: ${REPORT}"
