#!/usr/bin/env bash
# =============================================================================
# release_candidate.sh -- clean-checkout reproducibility driver for K-Bound.
#
#   bash docs/research/kbound/runbooks/release_candidate.sh [MODE]
#
# MODE (default: all):
#   preflight         resolve root, validate environment + datasets (read-only)
#   deep-local-provenance verify the mounted CCT-20 prospective evidence chain
#   validate-results  validate result schemas, seeds, protocol/config hashes
#   generate          rebuild aggregates -> manifest -> claim matrix/tables/figures
#   test              software tests + forbidden-claim checks + formal audit
#   pdf               build short & long PDFs and render every page
#   all               everything above, in order, then emit checksums
#
# Guarantees:
#   * Portable: the repository root is discovered; no machine-specific path is embedded.
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
export PYTHONPATH="${REPO}/${KB}:${PYTHONPATH:-}"
PY="${KBOUND_PYTHON:-${PYTHON:-python3}}"
MODE="${1:-all}"
WARN=()
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { WARN+=("$*"); printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
run_if() { # run_if <script> <desc> ; skip (warn) if script absent
  local s="$1"; shift
  if [[ -f "$s" ]]; then log "$*"; else warn "missing $s ($*) -- skipped"; return 0; fi
}

# The release toolchain is verified once and its exact executable paths are
# exported to every child phase.  This is opt-in in the compact compatibility
# runbook because older author-machine environments may not have the locked
# Python 3.12 profile installed; enabling KBOUND_VERIFY_TOOLCHAIN=1 makes the
# gate fail closed instead of silently falling back to PATH discovery.
configure_release_tool_path() {
  local resolved_file="$1" override_name override extra seen=":" count=0
  while IFS=$'\t' read -r override_name override extra; do
    [[ -z "$override_name" && -z "$override" && -z "$extra" ]] && continue
    if [[ -n "$extra" || -z "$override_name" || -z "$override" ]]; then
      warn "malformed verified tool-path record"; return 1
    fi
    case "$override_name" in
      KBOUND_TOOL_LATEXMK|KBOUND_TOOL_LATEXPAND|KBOUND_TOOL_PANDOC|KBOUND_TOOL_PERL|\
      KBOUND_TOOL_PDFLATEX|KBOUND_TOOL_PDFINFO|KBOUND_TOOL_PDFTOPPM|\
      KBOUND_TOOL_PDFTOTEXT|KBOUND_TOOL_PDFDETACH|KBOUND_TOOL_SOFFICE) ;;
      *) warn "unexpected verified tool-path variable: $override_name"; return 1 ;;
    esac
    case "$seen" in *":$override_name:"*) warn "duplicate verified tool-path variable: $override_name"; return 1;; esac
    case "$override" in /*) ;; *) warn "$override_name must be absolute"; return 1;; esac
    if [[ ! -f "$override" || ! -x "$override" || -L "$override" ]]; then
      warn "$override_name does not name a resident executable"; return 1
    fi
    export "$override_name=$override"
    seen="${seen}${override_name}:"
    count=$((count + 1))
  done <"$resolved_file"
  [[ "$count" -eq 10 ]] || { warn "verified release tool set is incomplete: expected 10, got $count"; return 1; }
}

verify_release_toolchain_for_phase() {
  [[ "${KBOUND_VERIFY_TOOLCHAIN:-0}" == "1" ]] || return 0
  local resolved_dir resolved_file
  resolved_dir="$(mktemp -d "${TMPDIR:-/tmp}/kbound-release-tools.XXXXXX")"
  resolved_file="$resolved_dir/resolved-tools.tsv"
  "$PY" "$KB/scripts/verify_release_toolchain.py" \
    --profile "$KB/release_toolchain_macos_arm64.json" \
    --output "$KB/audits/release_toolchain_2026_09_02.json" \
    --resolved-tools-output "$resolved_file"
  configure_release_tool_path "$resolved_file"
  rm -f "$resolved_file"; rmdir "$resolved_dir" 2>/dev/null || true
}

verify_release_python_environment() {
  [[ "${KBOUND_VERIFY_TOOLCHAIN:-0}" == "1" ]] || return 0
  "$PY" "$KB/scripts/verify_python_environment.py" \
    --lock requirements-release-macos-arm64.lock.txt \
    --content-profile "$KB/release_python_environment_macos_arm64.json" \
    --output "$KB/audits/python_environment_2026_09_02.json"
}

# --- 2. steps ---------------------------------------------------------------
step_preflight() {
  log "MODE preflight -- repo root: $REPO"
  verify_release_python_environment
  verify_release_toolchain_for_phase
  "$PY" - <<'PYEOF'
import sys
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
  log "preflight complete"
}

step_deep_local_dataset_preflight() {
  # Only explicit local-data verification may inspect mounted datasets.
  "$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "docs/research/kbound")
from kbound_repro import paths
missing = []
for name, fn, env in [("ImageNet-R", paths.imagenetr_root, "KBOUND_IMAGENETR_ROOT"),
                      ("PACS", paths.pacs_root, "KBOUND_PACS_ROOT")]:
    p = fn()
    print(f"  dataset {name}: {p}  ->", "present" if p.is_dir() else "ABSENT (set %s)" % env)
    if not p.is_dir(): missing.append(env)
if missing:
    raise SystemExit("required local datasets missing: " + ", ".join(missing))
PYEOF
}

step_deep_local_cct20_provenance() {
  # Equivalent explicit switch: --deep-local-cct20-provenance.
  [[ "${KBOUND_DEEP_LOCAL_CCT20_PROVENANCE:-0}" == "1" ]] || return 0
  log "MODE deep-local-provenance -- verify mounted CCT-20 prospective evidence"
  "$PY" "$KB/scripts/verify_cct20_prospective_evidence.py" \
    --local-release-manifest "$KB/paper/generated/cct20_release_manifest.json" --check
}

step_validate_results() {
  verify_release_toolchain_for_phase
  log "MODE validate-results -- schema + seed + hash validation"
  if [[ -f "$KB/scripts/02_verify_results.py" ]]; then
    log "verify results (02)"
    "$PY" "$KB/scripts/02_verify_results.py"
  else
    warn "missing $KB/scripts/02_verify_results.py (verify results (02)) -- skipped"
  fi
  "$PY" -m kbound_repro.release_checks || {
    warn "authority/consistency checks reported problems (see above)"; return 1;
  }
}

step_generate() {
  verify_release_toolchain_for_phase
  log "MODE generate -- rebuild aggregates -> manifest -> matrix/tables/figures"
  "$PY" "$KB/scripts/verify_cifar_current_arithmetic.py"
  # Current-policy authority must be refreshed before the release surfaces are
  # synchronized.  The canonical reconciler requires the pinned analysis
  # runtime (numpy 2.4.4 / scikit-learn 1.8.0), which may differ from the
  # author-machine release runtime.  Set KBOUND_REFRESH_CANONICAL=1 and point
  # KBOUND_ANALYSIS_PYTHON at that verified interpreter for a real refresh;
  # the default compact run only validates already-sealed authority files.
  if [[ "${KBOUND_REFRESH_CANONICAL:-0}" == "1" ]]; then
    local analysis_py="${KBOUND_ANALYSIS_PYTHON:-$PY}"
    log "refresh current-policy canonical panel"
    "$analysis_py" scripts/reconcile_result_panels.py --reuse-transfer
    "$analysis_py" docs/research/kbound/scripts/analyze_current_policy_cluster_inference.py
  fi
  "$PY" scripts/sync_reconciled_panels.py --public-only
  "$PY" "$KB/scripts/build_result_manifest.py" --public-only
  "$PY" "$KB/scripts/refresh_storage_manifest.py" --write-public-only
}

step_test() {
  verify_release_toolchain_for_phase
  log "MODE test -- software tests + forbidden-claim checks + formal audit"
  # inexpensive first: canonical toolkit tests (torch-independent)
  "$PY" -m pytest "$KB/kbound_repro/tests" -q
  # forbidden-claim / authority gate
  "$PY" -m kbound_repro.release_checks
  # Focused regression coverage for the current-policy diagnostics, generated
  # display tables, narrative status guards, and isolated PDF build.  These
  # tests are deliberately listed here so the release receipt records the
  # same behavioral checks that protect the maintained manuscript surfaces.
  local regression_tests=(
    tests/test_kbound_interval_diagnostics.py
    tests/test_kbound_metric_display_tables.py
    tests/test_kbound_narrative_revision.py
    tests/test_kbound_pdf_build_isolation.py
    tests/test_cct20_release_builder.py
    tests/test_cct20_manuscript_claim_validation.py
    tests/test_build_docx_pipeline.py
    tests/test_release_checksum_producer.py
    tests/test_ci_action_pins.py
    tests/test_archived_authorities.py
    tests/test_sar_final_block_batch_stats.py
    tests/test_sar_runbook_exit_status.py
    tests/test_official_headtohead_outcome_exclusion.py
    tests/test_kbound_formal_audit.py
    tests/test_kga_masked_inputs.py
    tests/test_kbound_current_policy_bindings.py
  )
  log "focused regression tests"
  "$PY" -m pytest "${regression_tests[@]}" -q
  # Full verification is bound to the captured revision. Protected outcome
  # readers remain excluded; no authorization flag is supplied here.
  PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PY" "$KB/scripts/run_repository_verification.py" \
    --repo "$REPO" --python "$PY" --all-gates \
    --expected-source-commit "${RELEASE_SOURCE_COMMIT:?release source was not captured}"
  # Kernel-checked registered scope, not a claim of universal formalization.
  log "formal audit (strict core and registered declarations)"
  FORMAL_PYTHON="$PY" bash "$KB/formal/build.sh" --json-out "$REPO/$KB/audits/formal_foundations_2026_08_31.json"
}

step_pdf() {
  verify_release_toolchain_for_phase
  log "MODE pdf -- build short & long PDFs, render every page"
  if [[ -f "$KB/scripts/build_pdfs.sh" ]]; then
    PYTHON="$PY" BUILD_LONG_TMLR=1 BUILD_DOCX=1 bash "$KB/scripts/build_pdfs.sh" || { warn "PDF build failed"; return 1; }
  else
    warn "build_pdfs.sh absent -- cannot build PDFs"; return 1
  fi
  # Page rendering is required evidence, not an optional warning.
  if [[ -f "$KB/scripts/render_pdf_pages.py" ]]; then
    "$PY" "$KB/scripts/render_pdf_pages.py" || { warn "PDF page render check failed"; return 1; }
  else
    warn "render_pdf_pages.py absent -- cannot verify PDF pages"; return 1
  fi
}

# Optional packaging phases are kept separate from the portable manuscript
# build. They never invent missing evidence: each command fails closed when its
# authenticated release inputs are absent.
step_public_bundle() {
  log "MODE public-bundle -- build from authenticated public evidence"
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --replace-verified
  step_verify_public_bundle
}

step_verify_public_bundle() {
  log "MODE public-bundle-verify -- verify committed public CCT-20 evidence"
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --check
}

step_anonymous_supplement() {
  verify_release_toolchain_for_phase
  log "MODE anonymous-supplement -- build/check the deterministic reviewer package"
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --root "$REPO" \
    --source-seal "$KB/audits/release_source_seal_2026_08_29.json" \
    --checksums "$KB/KBOUND_RELEASE_SHA256SUMS.txt" \
    --public-bundle "$KB/release/cct20_public_evidence_bundle.zip" \
    --output "$KB/release/kbound_anonymous_supplement.zip" --replace-verified
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$KB/release/kbound_anonymous_supplement.zip" --check
}

emit_post_checksums() {
  log "MODE post-checksums -- hash and verify the anonymous package"
  local artifact="$KB/release/kbound_anonymous_supplement.zip"
  [[ -f "$artifact" ]] || { warn "missing post-checksum input: $artifact"; return 1; }
  KBOUND_VERIFY_TOOLCHAIN=1 verify_release_python_environment
  KBOUND_VERIFY_TOOLCHAIN=1 verify_release_toolchain_for_phase
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$artifact" --portable-check
  local digest
  if have sha256sum; then digest="$(sha256sum "$artifact" | awk '{print $1}')"; \
  elif have shasum; then digest="$(shasum -a 256 "$artifact" | awk '{print $1}')"; \
  else warn "no SHA-256 utility available"; return 1; fi
  printf '%s  %s\n' "$digest" "$artifact" >"$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt"
  "$PY" "$KB/scripts/verify_release_checksums.py" \
    "$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt" --root "$REPO" --post-checksum
}

start_release_run() {
  # Capture the starting source identity for strict callers. The normal
  # compact runbook remains usable in an iCloud checkout whose Git index is
  # unavailable; KBOUND_STRICT_SOURCE_SEAL=1 turns that condition into a hard
  # failure before any generated output is written.
  if ! RELEASE_SOURCE_COMMIT="$(git -c core.preloadindex=false -c core.fsmonitor=false rev-parse --verify HEAD 2>/dev/null)"; then
    if [[ "${KBOUND_STRICT_SOURCE_SEAL:-0}" == "1" ]]; then
      warn "could not resolve a full immutable source commit"; return 1
    fi
    RELEASE_SOURCE_COMMIT=""
  fi
  export RELEASE_SOURCE_COMMIT
}

check_release_source() {
  "$PY" "$KB/scripts/build_release_source_seal.py" \
    --source-commit "$RELEASE_SOURCE_COMMIT" --check-source
}

emit_checksums() {
  log "output checksums (authoritative artifacts)"
  "$PY" "$KB/scripts/verify_release_checksums.py" --list-required || return 1
  # The shared writer requires every canonical file and atomically replaces
  # the receipt only after self-verification. Missing inputs never get skipped.
  "$PY" "$KB/scripts/verify_release_checksums.py" \
    "$KB/KBOUND_RELEASE_SHA256SUMS.txt" --root "$REPO" --write
}

verify_portable_release() {
  # Read-only artifact verification: no author-machine toolchain or datasets.
  "$PY" "$KB/scripts/build_release_source_seal.py" --check-portable
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --portable-check
  "$PY" "$KB/scripts/verify_release_checksums.py" \
    "$KB/KBOUND_RELEASE_SHA256SUMS.txt" --root "$REPO"
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$KB/release/kbound_anonymous_supplement.zip" --portable-check
  "$PY" "$KB/scripts/verify_release_checksums.py" \
    "$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt" --root "$REPO" --post-checksum
}

# --- 3. dispatch ------------------------------------------------------------
case "$MODE" in
  preflight)        step_preflight ;;
  deep-local-preflight) KBOUND_VERIFY_TOOLCHAIN=1 step_preflight; step_deep_local_dataset_preflight ;;
  deep-local-provenance) KBOUND_DEEP_LOCAL_CCT20_PROVENANCE=1; export KBOUND_DEEP_LOCAL_CCT20_PROVENANCE; step_preflight; step_deep_local_cct20_provenance ;;
  validate-results) step_preflight; step_validate_results ;;
  generate)         step_preflight; step_generate ;;
  test)             start_release_run; step_preflight; step_test ;;
  pdf)              step_preflight; step_pdf ;;
  public-bundle-verify) step_preflight; step_verify_public_bundle ;;
  public-bundle) step_public_bundle ;;
  anonymous-supplement) step_preflight; step_anonymous_supplement ;;
  post-checksums) emit_post_checksums ;;
  verify-portable-release) verify_portable_release ;;
  checksums)
    "$PY" "$KB/scripts/build_release_source_seal.py" --check-structure
    emit_checksums
    ;;
  all)
    start_release_run
    "$PY" "$KB/scripts/build_release_source_seal.py" \
      --source-commit "$RELEASE_SOURCE_COMMIT" --preflight
    step_preflight
    check_release_source
    step_validate_results
    check_release_source
    step_generate
    check_release_source
    step_test
    check_release_source
    step_pdf
    check_release_source
    # Packaging is explicit and fail-closed. Set KBOUND_PACKAGE_RELEASE=1
    # only after the authenticated public bundle/source seal are present.
    if [[ "${KBOUND_PACKAGE_RELEASE:-0}" == "1" ]]; then
      PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PY" "$KB/scripts/run_repository_verification.py" --repo "$REPO"
      step_verify_public_bundle
    fi
    check_release_source
    emit_checksums
    if [[ "${KBOUND_PACKAGE_RELEASE:-0}" == "1" ]]; then
      step_anonymous_supplement
      emit_post_checksums
    fi
    ;;
  *) echo "unknown MODE '$MODE' (preflight|deep-local-preflight|deep-local-provenance|validate-results|generate|test|pdf|public-bundle-verify|anonymous-supplement|post-checksums|all)" >&2; exit 2 ;;
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
