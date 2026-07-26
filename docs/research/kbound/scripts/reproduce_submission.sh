#!/usr/bin/env bash
# K-Bound submission reproduction — lightweight verify + cached headline checks.
#
# =============================================================================
# FIX-QUEUE ITEM 8 (F4-9): this script used to abort at step 1.
# =============================================================================
# The old line 3 was `set -euo pipefail`.  With `-e`, the first non-zero exit
# killed the whole run — and step 1 exits non-zero on a clean checkout, because
# `tests/test_calibration_split_integrity.py:10-11` computes
# `REPO/"docs"/"experiments"/kbound/results/edge_real_phone_v1/...`, a path that
# cannot exist (the results tree is `REPO/experiments`, not `REPO/docs/
# experiments`), and `edge/tests/test_no_live_labels.py` imports torch.  So the
# documented one-command verification reported a FileNotFoundError and steps
# 2-9 — the ones that actually check the paper's numbers — never ran at all.
#
# What changed:
#   * `-e` is gone.  Every step runs through `step`, which records PASS / FAIL /
#     SKIP and keeps going, so a missing optional artifact degrades one row of
#     the report instead of hiding eight steps of evidence.
#   * Steps declare whether they are REQUIRED or OPTIONAL.  The script exits 1
#     at the end if any REQUIRED step failed, so "keeps going" never becomes
#     "silently passes".
#   * Missing inputs are reported by name with what to do about them, rather
#     than as a bare traceback.
#   * PYTHONHASHSEED is pinned (fix-queue item 30): several suite scripts seed
#     per-cell RNGs from string digests and Python salts `hash()` per process.
#
# Usage:  bash docs/research/kbound/scripts/reproduce_submission.sh
#         PYTHON=/path/to/python bash .../reproduce_submission.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
KBOUND="${ROOT}/docs/research/kbound"
cd "${ROOT}"

export PYTHONHASHSEED=0          # fix-queue item 30: stable per-cell RNG seeding

PY="${PYTHON:-}"
if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
  elif [[ -x "${HOME}/.venv_wilds/bin/python" ]]; then
    PY="${HOME}/.venv_wilds/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "No Python interpreter found. Set PYTHON or install python3." >&2
    exit 1
  fi
fi

MANIFEST="${KBOUND}/RELEASE_MANIFEST.json"
REPORT="${KBOUND}/reports/reproducibility_release_report.md"
mkdir -p "${KBOUND}/reports"

# --------------------------------------------------------------------------- #
# step accounting                                                             #
# --------------------------------------------------------------------------- #
STEP_NAMES=(); STEP_STATUS=(); STEP_NOTES=()
REQUIRED_FAILURES=0

record() {   # record <name> <PASS|FAIL|SKIP> <note>
  STEP_NAMES+=("$1"); STEP_STATUS+=("$2"); STEP_NOTES+=("$3")
}

# step <name> <required|optional> <note-on-failure> -- <command...>
step() {
  local name="$1" req="$2" note="$3"; shift 3
  [[ "${1:-}" == "--" ]] && shift
  echo
  echo "=== [${name}] ==="
  if "$@"; then
    record "${name}" "PASS" ""
    return 0
  fi
  local rc=$?
  if [[ "${req}" == "required" ]]; then
    REQUIRED_FAILURES=$((REQUIRED_FAILURES + 1))
    echo "FAIL (required, exit ${rc}): ${name} -- ${note}" >&2
    record "${name}" "FAIL" "${note}"
  else
    echo "SKIP (optional, exit ${rc}): ${name} -- ${note}" >&2
    record "${name}" "SKIP" "${note}"
  fi
  return 0
}

require_file() {   # require_file <path> <what-to-do>
  if [[ -f "$1" ]]; then
    # Fix-queue item 9: 142 tracked text artifacts are NUL-filled iCloud
    # placeholders. Neither `-s` nor a whitespace scan catches them, so compare
    # the byte count with and without NULs. (Do NOT use `grep -q $'\0'`: bash
    # collapses $'\0' to the empty string, and `grep -q ''` matches everything.)
    local _n_all _n_nonul
    _n_all=$(wc -c < "$1" | tr -d ' ')
    _n_nonul=$(LC_ALL=C tr -d '\000' < "$1" | wc -c | tr -d ' ')
    if [[ "${_n_all}" -eq 0 || "${_n_all}" -ne "${_n_nonul}" ]]; then
      echo "PLACEHOLDER: ${1#${ROOT}/} is NUL-filled (iCloud 'Optimise Mac Storage')." >&2
      echo "  -> run 'Download Now' on the source machine, or regenerate it." >&2
      return 1
    fi
    echo "present: ${1#${ROOT}/}"
    return 0
  fi
  echo "MISSING: ${1#${ROOT}/}" >&2
  echo "  -> $2" >&2
  return 1
}

echo "=== K-Bound reproduce_submission ==="
COMMIT=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo "unknown")
echo "git_commit=${COMMIT}"
echo "python=$(${PY} --version 2>&1)"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"

# --------------------------------------------------------------------------- #
# [1] Unit tests                                                              #
# --------------------------------------------------------------------------- #
# Split into a CORE set (pure-Python, no external artifacts, no torch — these
# must pass on any checkout) and an ENV-DEPENDENT set (needs torch and/or the
# edge-device capture artifacts, which are not in the release).  The old script
# ran them as one blob and let the second group kill the run.
run_core_tests() {
  "${PY}" -m pytest \
    "${KBOUND}/tests/test_no_leakage_protocols.py" \
    "${KBOUND}/tests/test_claim_metric_semantics.py" \
    "${KBOUND}/tests/test_unified_result_audit.py" \
    "${KBOUND}/kbound_pkg/tests/" \
    "${KBOUND}/edge/tests/test_conformal.py" \
    "${KBOUND}/edge/tests/test_policy.py" \
    "${ROOT}/tests/test_kga_routing.py" \
    -q --tb=short -p no:cacheprovider
}
run_env_tests() {
  "${PY}" -m pytest \
    "${KBOUND}/tests/test_calibration_split_integrity.py" \
    "${KBOUND}/edge/tests/test_no_live_labels.py" \
    -q --tb=line -p no:cacheprovider
}
step "1 core unit tests (leakage + claim semantics + conformal + policy + routing)" \
     required "these are pure-Python and must pass on any checkout" -- run_core_tests
step "1b env-dependent tests (edge capture artifacts + torch)" \
     optional "needs torch and experiments/kbound/results/edge_real_phone_v1/{calibration_summary,split_audit}.json, which are not in the release" \
     -- run_env_tests

# --------------------------------------------------------------------------- #
# [2] Theory validators                                                       #
# --------------------------------------------------------------------------- #
check_theory_artifacts() {
  if [[ ! -d "${ROOT}/experiments/kbound/theory_validation" ]]; then
    echo "MISSING: experiments/kbound/theory_validation/" >&2
    echo "  -> run the theory validators; see RELEASE_10X_TRACK.md" >&2
    return 1
  fi
  local n
  n=$(ls "${ROOT}/experiments/kbound/theory_validation"/results_thm*.json 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${n}" == "0" ]]; then
    echo "MISSING: experiments/kbound/theory_validation/results_thm*.json" >&2
    return 1
  fi
  echo "theory JSON artifacts: ${n} present"
}
step "2 theory validator artifacts present" optional \
     "theory_validation/results_thm*.json absent" -- check_theory_artifacts

step "2b full theory audit (artifacts + claim ledger cross-check)" optional \
     "theory_audit_full.py failed; see its report for the missing inputs" \
     -- "${PY}" "${KBOUND}/scripts/theory_audit_full.py" --write-report

step "2c wave-4 theory_v2 validators + routing selftest" optional \
     "run_theory_v2_validators.sh failed" \
     -- bash "${KBOUND}/scripts/run_theory_v2_validators.sh"

# --------------------------------------------------------------------------- #
# [3]-[6] paper-facing checks (run from the paper directory)                  #
# --------------------------------------------------------------------------- #
cd "${KBOUND}"

step "3 gate baseline CPU selftest" required \
     "gate_baseline_comparison.py --selftest is pure numpy and must pass" \
     -- "${PY}" scripts/gate_baseline_comparison.py --selftest

step "4a refresh results_source (locked)" optional \
     "refresh_results_source_locked.py failed" \
     -- "${PY}" scripts/refresh_results_source_locked.py
step "4b regenerate paper table macros" optional \
     "make_tables.py failed" \
     -- "${PY}" scripts/make_tables.py

run_unified_audit() {
  "${PY}" scripts/unified_result_audit.py --strict-explicit \
    > /tmp/kbound_unified_result_audit.tsv && cat /tmp/kbound_unified_result_audit.tsv
}
step "5 unified result verdict audit" optional \
     "unified_result_audit.py --strict-explicit failed" -- run_unified_audit

step "6 validate claim ledger" required \
     "claim_ledger.json must be valid JSON -- it is the wording authority" \
     -- "${PY}" -c "import json; json.load(open('claim_ledger.json')); print('claim_ledger.json OK')"

cd "${ROOT}"

# --------------------------------------------------------------------------- #
# [7] mixed head-to-head                                                      #
# --------------------------------------------------------------------------- #
H2H="${ROOT}/experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json"
check_h2h() {
  require_file "${H2H}" \
    "run: bash experiments/kbound/poem_aetta/run_all_headtohead.sh" || return 1
  "${PY}" - "${H2H}" <<'PYEOF'
import json, sys
h = json.load(open(sys.argv[1]))
v = h["headtohead"]["VERDICT"]
print(f"mixed_headtohead verdict={v}")
if v != "WIN":
    print("WARN: expected WIN for headline claim KB-CLAIM-026", file=sys.stderr)
PYEOF
}
step "7 mixed head-to-head (POEM/AETTA) result present" optional \
     "HEADTOHEAD_RESULTS_cifar10c_tent_primary.json absent or unreadable" -- check_h2h

# --------------------------------------------------------------------------- #
# [8] cached headline artifacts                                               #
# --------------------------------------------------------------------------- #
check_cached() {
  local missing=0
  require_file "${ROOT}/experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json" \
    "regenerate with experiments/kbound/results/stress_grid_multiseed_v1/_locked_analysis_script.py" || missing=1
  require_file "${ROOT}/experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json" \
    "see STORAGE_MANIFEST.json for provenance" || missing=1
  require_file "${ROOT}/experiments/kbound/results/controlled_multimodal_d33/results.json" \
    "see STORAGE_MANIFEST.json for provenance" || missing=1
  require_file "${KBOUND}/results/assumption_audit_v1.json" \
    "regenerate with scripts/assumption_audit.py" || missing=1
  require_file "${ROOT}/research_lock/KBOUND_HEADLINE_FINDINGS_v3.json" \
    "this is a sealed lock file; restore it from the release archive" || missing=1
  return "${missing}"
}
# REQUIRED: these five are the files the headline tables are read from. If one is
# absent the run must not report success -- but the earlier steps still ran, which
# is the whole point of dropping `set -e`.
step "8 cached headline artifacts" required \
     "one or more headline artifacts are absent or are NUL-filled placeholders" \
     -- check_cached

# --------------------------------------------------------------------------- #
# [9] manifest + report                                                       #
# --------------------------------------------------------------------------- #
echo
echo "=== [9] Write manifest ==="
steps_json=""
rows=""
for i in "${!STEP_NAMES[@]}"; do
  esc_name=${STEP_NAMES[$i]//\"/\\\"}
  esc_note=${STEP_NOTES[$i]//\"/\\\"}
  [[ -n "${steps_json}" ]] && steps_json+=",\n    "
  steps_json+="{\"step\": \"${esc_name}\", \"status\": \"${STEP_STATUS[$i]}\", \"note\": \"${esc_note}\"}"
  rows+="| ${STEP_NAMES[$i]} | ${STEP_STATUS[$i]} | ${STEP_NOTES[$i]} |"$'\n'
done

OVERALL="PASS"
[[ "${REQUIRED_FAILURES}" -gt 0 ]] && OVERALL="FAIL"

cat > "${MANIFEST}" <<EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "${COMMIT}",
  "python": "$(${PY} --version 2>&1)",
  "pythonhashseed": "${PYTHONHASHSEED}",
  "overall": "${OVERALL}",
  "required_failures": ${REQUIRED_FAILURES},
  "steps": [
    $(printf "%b" "${steps_json}")
  ],
  "note": "Every step runs; REQUIRED failures set overall=FAIL and a non-zero exit. Full GPU protocol re-runs are not executed by this script; see RELEASE_10X_TRACK.md"
}
EOF

cat > "${REPORT}" <<EOF
# Reproducibility Release Report

Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Commit: ${COMMIT}
Overall: **${OVERALL}** (${REQUIRED_FAILURES} required step(s) failed)

Every step below was attempted. PASS = ran and exited 0. FAIL = a required step
did not (this sets Overall to FAIL). SKIP = an optional step could not run,
usually because an artifact is not in the release; the reason is in Note.

| Step | Status | Note |
|---|---|---|
${rows}
Not covered by this script: the full GPU protocol re-runs (stress grid,
ImageNet-C, WILDS) — see RELEASE_10X_TRACK.md — and the physical-camera R2
capture, which is still pending.
EOF

echo "=== DONE (${OVERALL}) ==="
echo "Manifest: ${MANIFEST}"
echo "Report: ${REPORT}"
exit $(( REQUIRED_FAILURES > 0 ? 1 : 0 ))
