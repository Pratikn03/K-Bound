#!/usr/bin/env bash
# =============================================================================
# release_candidate.sh -- clean-checkout reproducibility driver for K-Bound.
#
#   bash docs/research/kbound/runbooks/release_candidate.sh [MODE]
#
# MODE (default: all):
#   preflight         resolve root, validate environment + datasets (read-only)
#   validate-results  validate result schemas, seeds, protocol/config hashes
#   generate          rebuild aggregates -> manifest -> claim matrix/tables/figures
#   test              software tests + forbidden-claim checks + formal audit
#   pdf               build short/long PDFs + compact DOCX and render PDF pages
#   source-seal       bind clean HEAD source/tree and maintained release blobs
#   deep-local-provenance  explicitly dereference/hash machine-local CCT upstreams
#   all               everything above, in order, then emit checksums
#
# Guarantees:
#   * Portable: the repository root is DISCOVERED (no /Users/... or /Volumes/...).
#   * Read-only on data: datasets are validated, never modified.
#   * Fails closed: missing required evidence aborts the release.
#   * NEVER launches training. Training is a separate, explicit command.
# =============================================================================
set -euo pipefail

# --- 1. resolve repository root (portable) ----------------------------------
resolve_root() {
  if git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel
    return
  fi
  # marker walk fallback
  local d; d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "$d" != "/" ]]; do
    if [[ -e "$d/pyproject.toml" || -d "$d/.git" ]]; then echo "$d"; return; fi
    d="$(dirname "$d")"
  done
  echo "ERROR: could not resolve repository root" >&2; exit 3
}

REPO="$(resolve_root)"
cd "$REPO"
KB="docs/research/kbound"
export PYTHONPATH="${REPO}:${REPO}/src:${REPO}/${KB}:${REPO}/${KB}/kbound_pkg:${REPO}/${KB}/edge/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${KBOUND_PYTHON:-}" ]]; then
  PY="$KBOUND_PYTHON"
elif [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  PY="python3.12"
else
  PY="python3"
fi
if [[ "$PY" != /* && "$PY" == */* && -x "$REPO/$PY" ]]; then
  PY="$REPO/$PY"
fi
# Child processes (including relocation tests that invoke this runbook again)
# must not inherit a repository-relative interpreter path after changing cwd.
export KBOUND_PYTHON="$PY"
MODE="${1:-all}"
WARN=()
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { WARN+=("$*"); printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
run_if() { # run_if <script> <desc> ; skip (warn) if script absent
  local s="$1"; shift
  if [[ -f "$s" ]]; then log "$*"; else warn "missing $s ($*) -- skipped"; return 0; fi
}

# --- 2. steps ---------------------------------------------------------------
step_preflight() {
  log "MODE preflight -- repo root: $REPO"
  "$PY" - <<'PYEOF'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"release requires Python 3.12; got {sys.version_info.major}.{sys.version_info.minor} "
        f"from {sys.executable}"
    )
sys.path.insert(0, "docs/research/kbound")
from kbound_repro import runtime
info = runtime.describe_runtime()
print("  runtime:", {k: info[k] for k in ("python", "platform")})
for pkg in ("numpy", "torch", "torchvision", "sklearn"):
    print(f"  {pkg}: {info.get(pkg)}")
# Preflight is intentionally structural and read-only. Numerical release steps
# import their own required dependencies and fail with actionable messages.
print("  OK: portable structural preflight complete.")
PYEOF
  # datasets: validate presence (read-only) via env vars; warn (don't fail) here
  "$PY" - <<'PYEOF' || true
import sys; sys.path.insert(0, "docs/research/kbound")
from kbound_repro import paths
for name, fn, env in [("ImageNet-R", paths.imagenetr_root, "KBOUND_IMAGENETR_ROOT"),
                      ("PACS", paths.pacs_root, "KBOUND_PACS_ROOT")]:
    p = fn()
    print(f"  dataset {name}: {p}  ->", "present" if p.is_dir() else "ABSENT (set %s)" % env)
PYEOF
  log "preflight complete"
}

step_validate_results() {
  log "MODE validate-results -- canonical panel, schema, hashes, and claim authority"
  "$PY" -m pytest -q tests/test_reconciled_panels.py tests/test_manuscript_claim_consistency.py
  "$PY" -m pytest -q \
    tests/test_kga_experiment_contract.py \
    tests/test_kga_frontier_api.py \
    tests/test_pacs_replay_artifact.py \
    tests/test_independent_checkpoint_audit.py \
    tests/test_official_baseline_provenance.py \
    tests/test_natural_target_provenance.py \
    tests/test_exact_confirmation_pipeline.py \
    tests/test_release_checksum_verifier.py \
    tests/test_release_source_seal.py \
    tests/test_so2sat_prospective_protocol.py \
    tests/test_so2sat_numbers_builder.py \
    tests/test_so2sat_target_boundary.py
  "$PY" -m kbound_repro.release_checks --require-manifest
  "$PY" "$KB/scripts/validate_closure_protocol.py"
}

validate_manuscript_claims() {
  case "${KBOUND_DEEP_LOCAL_CCT20_PROVENANCE:-0}" in
    0) "$PY" src/scripts/validate_manuscript_claims.py ;;
    1) "$PY" src/scripts/validate_manuscript_claims.py --deep-local-cct20-provenance ;;
    *)
      warn "KBOUND_DEEP_LOCAL_CCT20_PROVENANCE must be 0 or 1"
      return 1
      ;;
  esac
}

step_generate() {
  log "MODE generate -- rebuild the current source-hashed panel, manifest, tables, and figures"
  "$PY" scripts/reconcile_result_panels.py
  "$PY" "$KB/scripts/analyze_current_policy_cluster_inference.py"
  "$PY" scripts/sync_reconciled_panels.py
  "$PY" "$KB/scripts/build_result_manifest.py"
  "$PY" "$KB/scripts/build_so2sat_numbers.py"
  "$PY" "$KB/scripts/build_results_source_compat.py"
  "$PY" "$KB/scripts/make_tables.py"
  "$PY" "$KB/scripts/plot_canonical_decision_frontier.py"
  "$PY" "$KB/scripts/plot_conceptual_regime_geometry.py"
  "$PY" "$KB/scripts/run_frontier_kga_bridge.py"
  "$PY" "$KB/scripts/audit_natural_target_provenance.py"
  "$PY" "$KB/scripts/audit_official_baselines.py" --repo "$REPO"
  "$PY" "$KB/scripts/audit_empirical_data_quality_2026_08_27.py" --wording-only
  "$PY" "$KB/scripts/build_empirical_data_quality_report_artifact.py"
  "$PY" "$KB/scripts/build_dashboard_snapshot.py"
  "$PY" "$KB/scripts/refresh_storage_manifest.py" --write
  "$PY" "$KB/scripts/validate_canonical_release_data.py"
  validate_manuscript_claims
}

step_test() {
  log "MODE test -- collection, software tests, claim checks, and formal audit"
  "$PY" -m pytest --collect-only -q
  "$PY" -m pytest "$KB/kbound_repro/tests" -q
  "$PY" -m pytest -q \
    "$KB/kbound_pkg/tests" \
    "$KB/tests" \
    "$KB/edge/tests" \
    tests/test_kga_experiment_contract.py \
    tests/test_kga_frontier_api.py \
    tests/test_pacs_replay_artifact.py \
    tests/test_independent_checkpoint_audit.py \
    tests/test_official_baseline_provenance.py \
    tests/test_natural_target_provenance.py \
    tests/test_exact_confirmation_pipeline.py \
    tests/test_so2sat_numbers_builder.py \
    tests/test_cct20_release_builder.py \
    tests/test_cct20_manuscript_claim_validation.py \
    tests/test_build_docx_pipeline.py \
    tests/test_kga_canonical_rule.py \
    tests/test_release_checksum_verifier.py \
    tests/test_release_source_seal.py \
    tests/test_reconciled_panels.py \
    tests/test_manuscript_claim_consistency.py \
    tests/test_so2sat_prospective_protocol.py \
    tests/test_so2sat_target_boundary.py \
    "$KB/edge/tests/test_protocol_inventory_reporting.py"
  "$PY" -m kbound_repro.release_checks --require-manifest
  validate_manuscript_claims
  "$PY" -m kbound_repro.check_repo --staged
  log "Lean/Mathlib build and formal audit"
  bash "$KB/formal/build.sh"
}

step_pdf() {
  log "MODE pdf -- build the maintained compact PDF, long PDF, and compact DOCX; render every PDF page"
  if [[ -f "$KB/scripts/build_pdfs.sh" ]]; then
    BUILD_LONG_TMLR=1 PYTHON="$PY" BUILD_DOCX=1 bash "$KB/scripts/build_pdfs.sh" || {
      warn "compact PDF, long PDF, or compact DOCX build failed"
      return 1
    }
  else
    warn "build_pdfs.sh absent -- cannot build PDFs"; return 1
  fi
  # page-render / page-count check if a helper exists
  if [[ -f "$KB/scripts/render_pdf_pages.py" ]]; then
    "$PY" "$KB/scripts/render_pdf_pages.py" || {
      warn "PDF page render check failed"
      return 1
    }
  else
    warn "render_pdf_pages.py absent -- cannot verify release pages"
    return 1
  fi
}

step_source_seal() {
  local source_commit
  source_commit="${KBOUND_SOURCE_COMMIT:-$(git -c core.preloadindex=false -c core.fsmonitor=false rev-parse HEAD)}"
  log "MODE source-seal -- bind maintained release sources at $source_commit"
  "$PY" "$KB/scripts/build_release_source_seal.py" --source-commit "$source_commit"
  "$PY" "$KB/scripts/build_release_source_seal.py" --check --source-commit "$source_commit"
}

step_deep_local_provenance() {
  log "MODE deep-local-provenance -- dereference all sealed CCT-20 machine-local upstream paths"
  KBOUND_DEEP_LOCAL_CCT20_PROVENANCE=1 validate_manuscript_claims
}

emit_checksums() {
  log "output checksums (authoritative artifacts)"
  local output="$KB/KBOUND_RELEASE_SHA256SUMS.txt"
  local files=(
    "$KB/claim_ledger.json"
    "$KB/RESULT_MANIFEST.json"
    "$KB/STORAGE_MANIFEST.json"
    "$KB/results_source.json"
    "$KB/audits/empirical_data_quality_2026_08_27/artifact.json"
    "$KB/audits/empirical_data_quality_2026_08_27/audit_summary.json"
    "$KB/audits/empirical_data_quality_2026_08_27/reviewer_scorecard.csv"
    "$KB/dashboard/data/snapshot.json"
    "$KB/paper/generated/kbound_numbers.tex"
    "$KB/paper/generated/kbound_result_manifest.json"
    "$KB/paper/generated/current_policy_family_sensitivity.tex"
    "$KB/paper/generated/empirical_audit/decision_metrics.json"
    "$KB/paper/generated/uniform_verdicts.json"
    "$KB/figures/fig_decision_value_frontier.png"
    "$KB/figures/fig_phase_diagram.png"
    "$KB/kbound_short_final_draft.pdf"
    "$KB/kbound_short_final_draft.docx"
    "$KB/kbound_tmlr.pdf"
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
    "experiments/kbound/results/reconciled_panels_v1/CANONICAL_PANEL_RESULTS.md"
    "experiments/kbound/results/reconciled_panels_v1/canonical_panel_table.tex"
    "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"
    "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
    "experiments/kbound/frontier_sweep_v1/decision_value_results.json"
    "research_lock/KBOUND_PROSPECTIVE_CLOSURE_v1.yaml"
    "research_lock/KBOUND_EXACT_CONFIRMATION_UNSEALED_v1.json"
    "experiments/kbound/results/frontier_kga_bridge_v1/bridge_results.json"
    "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json"
    "experiments/kbound/results/official_repro_v1/OFFICIAL_BASELINE_AUDIT.json"
    "experiments/kbound/results/smoke_pacs_replay_v2/PACS_REPLAY_AUDIT.json"
    "experiments/kbound/results/edge_real_phone_v1/publication_gate.json"
    "$KB/audits/phase1_provenance_2026_08_27/provenance_seal.json"
    "$KB/audits/release_source_seal_2026_08_29.json"
  )
  local cct_manifest="$KB/paper/generated/cct20_release_manifest.json"
  local cct_files=(
    "$cct_manifest"
    "$KB/paper/generated/cct20_release_manifest.json.receipt.json"
    "$KB/paper/generated/cct20_numbers.tex"
    "$KB/paper/generated/cct20_primary_table.tex"
    "$KB/paper/generated/cct20_location_effects.tex"
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml"
    "research_lock/KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml.sha256"
  )
  local cct_file
  for cct_file in "${cct_files[@]}"; do
    if [[ ! -f "$cct_file" ]]; then
      warn "completed CCT-20 release is incomplete: missing $cct_file"
      return 1
    fi
  done
  files+=("${cct_files[@]}")
  local so2sat_dir="experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1"
  local so2sat_files=(
    "$so2sat_dir/so2sat_candidate_selection.json"
    "$so2sat_dir/so2sat_candidate_selection.json.receipt.json"
    "$so2sat_dir/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json"
    "$so2sat_dir/so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json"
    "$so2sat_dir/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json"
    "$so2sat_dir/so2sat_sar_sam_bn_affine_probe_transfer_v1.gate_fit.json.receipt.json"
    "$KB/paper/generated/so2sat_numbers.tex"
    "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json"
    "research_lock/KBOUND_SO2SAT_DEVELOPMENT_RUNTIME_AMENDMENT_v1.json.receipt.json"
    "experiments/kbound/so2sat/prospective_protocol_v1.json"
    "experiments/kbound/so2sat/prospective_protocol_v1.json.receipt.json"
  )
  local so2sat_file
  for so2sat_file in "${so2sat_files[@]}"; do
    if [[ ! -f "$so2sat_file" ]]; then
      warn "completed So2Sat development result is incomplete: missing $so2sat_file"
      return 1
    fi
  done
  files+=("${so2sat_files[@]}")
  local required_pdf
  for required_pdf in \
    "$KB/kbound_short_final_draft.pdf" \
    "$KB/kbound_tmlr.pdf"; do
    if [[ ! -f "$required_pdf" ]]; then
      warn "missing maintained release PDF: $required_pdf"
      return 1
    fi
  done
  if [[ ! -f "$KB/kbound_short_final_draft.docx" ]]; then
    warn "missing maintained release DOCX: $KB/kbound_short_final_draft.docx"
    return 1
  fi
  if [[ ! -f "$KB/audits/release_source_seal_2026_08_29.json" ]]; then
    warn "missing final release source seal"
    return 1
  fi
  local f
  for f in "${files[@]}"; do
    if [[ ! -f "$f" ]]; then
      warn "missing checksum input: $f"
      return 1
    fi
  done
  local hash_kind
  if have sha256sum; then
    hash_kind="sha256sum"
  elif have shasum; then
    hash_kind="shasum"
  else
    warn "no SHA-256 utility available"
    return 1
  fi
  local checksum_tmp
  checksum_tmp="$(mktemp "$KB/.KBOUND_RELEASE_SHA256SUMS.XXXXXX")"
  trap 'rm -f "$checksum_tmp"' RETURN
  for f in "${files[@]}"; do
    local digest rel
    if [[ "$hash_kind" == "sha256sum" ]]; then
      digest="$(sha256sum "$f" | awk '{print $1}')"
    else
      digest="$(shasum -a 256 "$f" | awk '{print $1}')"
    fi
    rel="$f"
    [[ "$rel" == "$REPO/"* ]] && rel="${rel#"$REPO/"}"
    printf '%s  %s\n' "$digest" "$rel" | tee -a "$checksum_tmp"
  done
  "$PY" "$KB/scripts/verify_release_checksums.py" "$checksum_tmp" --root "$REPO"
  mv -f "$checksum_tmp" "$output"
  trap - RETURN
  log "wrote ${output#"$REPO/"}"
}

# --- 3. dispatch ------------------------------------------------------------
case "$MODE" in
  preflight)        step_preflight ;;
  validate-results) step_preflight; step_validate_results ;;
  generate)         step_preflight; step_generate ;;
  test)             step_preflight; step_test ;;
  pdf)              step_preflight; step_pdf ;;
  source-seal)      step_source_seal ;;
  deep-local-provenance) step_preflight; step_deep_local_provenance ;;
  # Hashing is intentionally environment-independent: this mode reads bytes
  # only. Scientific validation/build modes retain the strict Python preflight.
  checksums)         emit_checksums ;;
  all)
    step_preflight
    step_validate_results
    step_generate
    step_test
    step_pdf
    step_source_seal
    emit_checksums
    ;;
  *) echo "unknown MODE '$MODE' (preflight|validate-results|generate|test|pdf|source-seal|deep-local-provenance|checksums|all)" >&2; exit 2 ;;
esac

# --- 4. summary -------------------------------------------------------------
if [[ ${#WARN[@]} -gt 0 ]]; then
  echo
  log "completed MODE=$MODE with ${#WARN[@]} warning(s):"
  for w in "${WARN[@]}"; do echo "  - $w"; done
else
  log "completed MODE=$MODE with no warnings"
fi
echo
log "NOTE: training is intentionally NOT run by this script. It remains a separate explicit command."
