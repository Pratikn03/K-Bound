#!/usr/bin/env bash
# =============================================================================
# release_candidate.sh -- clean-checkout reproducibility driver for K-Bound.
#
#   bash docs/research/kbound/runbooks/release_candidate.sh [MODE]
#
# MODE (default: all):
#   preflight         resolve root and validate the portable release environment
#   deep-local-preflight  additionally require machine-local release datasets
#   validate-results  validate result schemas, seeds, protocol/config hashes
#   generate          rebuild aggregates -> manifest -> claim matrix/tables/figures
#   test              software tests + forbidden-claim checks + formal audit
#   pdf               build the four release-role PDFs and render every page
#   source-seal       bind clean HEAD source/tree and maintained release blobs
#   public-bundle     deep-local build and independently verify portable CCT-20 evidence
#   public-bundle-verify  verify committed portable CCT-20 evidence without source dereference
#   anonymous-supplement  build the deterministic source-bound reviewer package
#   post-checksums    hash and independently verify post-checksum artifacts
#   deep-local-provenance  explicitly dereference/hash machine-local CCT upstreams
#   deep-local-cct20-refresh  explicitly refresh CCT release files from machine-local upstreams
#   verify-portable-release  verify committed release bytes/semantics without build preflight
#   checksums         hash bytes after a structural (not checkout-semantic) source-seal check
#   all               portable default; reuse verified public CCT evidence and emit checksums
#   all-deep-local    all plus explicit CCT upstream refresh and deep provenance verification
#
# Set KBOUND_AUTHORIZE_PROTECTED_SO2SAT=1 only for a separately authorized
# execution that may read the sealed So2Sat development authority.
#
# Guarantees:
#   * Portable: the repository root is discovered; no author-machine path is embedded.
#   * Read-only on data: only explicit deep-local modes resolve datasets, and never modify them.
#   * Fails closed: missing required evidence aborts the release.
#   * Pins a completely clean source HEAD before all work; checks it after each phase.
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
KBOUND_AUTHORIZE_PROTECTED_SO2SAT="${KBOUND_AUTHORIZE_PROTECTED_SO2SAT:-0}"
case "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" in
  0|1) ;;
  *)
    echo "ERROR: KBOUND_AUTHORIZE_PROTECTED_SO2SAT must be 0 or 1" >&2
    exit 2
    ;;
esac
WARN=()
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { WARN+=("$*"); printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

configure_release_tool_path() {
  # Import the verifier's private TSV without eval. Every consumer receives an
  # exact, already-hashed realpath; PATH order is not part of the release trust
  # boundary (separate tool directories may contain conflicting command names).
  local resolved_file="$1"
  local override_name override extra seen=":" count=0
  while IFS=$'\t' read -r override_name override extra; do
    if [[ -n "$extra" || -z "$override_name" || -z "$override" ]]; then
      warn "malformed verified tool-path record"
      return 1
    fi
    case "$override_name" in
      KBOUND_TOOL_LATEXMK|KBOUND_TOOL_LATEXPAND|KBOUND_TOOL_PANDOC|\
      KBOUND_TOOL_PERL|\
      KBOUND_TOOL_PDFLATEX|KBOUND_TOOL_PDFINFO|KBOUND_TOOL_PDFTOPPM|\
      KBOUND_TOOL_PDFTOTEXT|KBOUND_TOOL_PDFDETACH|KBOUND_TOOL_SOFFICE) ;;
      *) warn "unexpected verified tool-path variable: $override_name"; return 1 ;;
    esac
    case "$seen" in
      *":$override_name:"*) warn "duplicate verified tool-path variable: $override_name"; return 1 ;;
    esac
    case "$override" in
      /*) ;;
      *) warn "$override_name must be an absolute executable realpath"; return 1 ;;
    esac
    if [[ ! -f "$override" || ! -x "$override" || -L "$override" ]]; then
      warn "$override_name does not name a resident non-symlink executable"
      return 1
    fi
    export "$override_name=$override"
    seen="${seen}${override_name}:"
    count=$((count + 1))
  done <"$resolved_file"
  if [[ "$count" -ne 10 ]]; then
    warn "verified release tool set is incomplete: expected 10, got $count"
    return 1
  fi
}

verify_release_toolchain_for_phase() {
  local resolved_dir resolved_file
  resolved_dir="$(mktemp -d "${TMPDIR:-/tmp}/kbound-release-tools.XXXXXX")"
  resolved_file="$resolved_dir/resolved-tools.tsv"
  if ! "$PY" "$KB/scripts/verify_release_toolchain.py" \
    --profile "$KB/release_toolchain_macos_arm64.json" \
    --output "$KB/audits/release_toolchain_2026_09_02.json" \
    --resolved-tools-output "$resolved_file"; then
    rm -f "$resolved_file"
    rmdir "$resolved_dir" 2>/dev/null || true
    return 1
  fi
  if ! configure_release_tool_path "$resolved_file"; then
    rm -f "$resolved_file"
    rmdir "$resolved_dir" 2>/dev/null || true
    return 1
  fi
  rm -f "$resolved_file"
  rmdir "$resolved_dir"
}

verify_release_python_environment() {
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
  log "preflight complete"
}

step_deep_local_dataset_preflight() {
  log "MODE deep-local-preflight -- require read-only machine-local release datasets"
  "$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "docs/research/kbound")
from kbound_repro import paths
missing = []
for name, fn, env in [("ImageNet-R", paths.imagenetr_root, "KBOUND_IMAGENETR_ROOT"),
                      ("PACS", paths.pacs_root, "KBOUND_PACS_ROOT")]:
    p = fn()
    present = p.is_dir()
    print(f"  dataset {name}: {p}  ->", "present" if present else "ABSENT (set %s)" % env)
    if not present:
        missing.append(env)
if missing:
    raise SystemExit("required release datasets are absent; set " + ", ".join(missing))
PYEOF
  log "deep-local dataset preflight complete"
}

step_validate_results() {
  verify_release_toolchain_for_phase
  log "MODE validate-results -- canonical panel, schema, hashes, and claim authority"
  "$PY" -m pytest -q tests/test_manuscript_claim_consistency.py
  "$PY" -m pytest -q \
    tests/test_kga_experiment_contract.py \
    tests/test_kga_frontier_api.py \
    tests/test_pacs_replay_artifact.py \
    tests/test_independent_checkpoint_audit.py \
    tests/test_official_baseline_provenance.py \
    tests/test_natural_target_provenance.py \
    tests/test_exact_confirmation_pipeline.py \
    tests/test_release_checksum_verifier.py \
    tests/test_release_source_seal.py
  "$PY" -m kbound_repro.release_checks --require-manifest
  "$PY" "$KB/scripts/validate_closure_protocol.py"
  if [[ "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" == "1" ]]; then
    log "explicit authorization: validate protected So2Sat development authority"
    "$PY" -m pytest -q \
      tests/test_reconciled_panels.py \
      tests/test_canonical_release_data.py \
      tests/test_so2sat_numbers_builder.py \
      tests/test_so2sat_prospective_v2.py
    "$PY" "$KB/scripts/validate_canonical_release_data.py"
  fi
}

validate_manuscript_claims() {
  case "${KBOUND_DEEP_LOCAL_CCT20_PROVENANCE:-0}:$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" in
    0:0) "$PY" src/scripts/validate_manuscript_claims.py ;;
    0:1) "$PY" src/scripts/validate_manuscript_claims.py --authorize-protected-so2sat ;;
    1:0) "$PY" src/scripts/validate_manuscript_claims.py --deep-local-cct20-provenance ;;
    1:1) "$PY" src/scripts/validate_manuscript_claims.py \
      --deep-local-cct20-provenance --authorize-protected-so2sat ;;
    *)
      warn "KBOUND_DEEP_LOCAL_CCT20_PROVENANCE must be 0 or 1"
      return 1
      ;;
  esac
}

step_generate() {
  verify_release_toolchain_for_phase
  log "MODE generate -- rebuild the unprotected repository-backed release surface"
  "$PY" scripts/reconcile_result_panels.py
  "$PY" "$KB/scripts/analyze_current_policy_cluster_inference.py"
  # The mixed So2Sat synchronizer/result-manifest/validator family and broad
  # result-tree provenance/storage scanners are not part of the portable
  # default. They may dereference protected gate-calibration or target-result
  # artifacts and therefore belong to a separately authorized, post-source-seal
  # natural-shift workflow.
  "$PY" "$KB/scripts/build_results_source_compat.py"
  "$PY" "$KB/scripts/build_current_policy_interval_diagnostics.py" --refresh-existing
  "$PY" "$KB/scripts/make_tables.py"
  "$PY" "$KB/scripts/plot_canonical_decision_frontier.py"
  "$PY" "$KB/scripts/plot_conceptual_regime_geometry.py"
  "$PY" "$KB/scripts/run_frontier_kga_bridge.py"
  "$PY" "$KB/scripts/audit_official_baselines.py" \
    --repo "$REPO" \
    --locked-stream "$REPO/experiments/kbound/results/stress_persample_v1/per_condition_cifar10c_tent_seed0.json" \
    --environment-receipt "$KB/audits/python_environment_2026_09_02.json" \
    --toolchain-receipt "$KB/audits/release_toolchain_2026_09_02.json"
  "$PY" "$KB/scripts/audit_empirical_data_quality_2026_08_27.py" --wording-only
  "$PY" "$KB/scripts/build_empirical_data_quality_report_artifact.py"
  "$PY" "$KB/scripts/build_dashboard_snapshot.py"
  validate_manuscript_claims
}

step_generate_deep_local_cct20() {
  verify_release_toolchain_for_phase
  log "MODE deep-local-cct20-refresh -- refresh CCT release files from sealed author-machine upstreams"
  local cct_history_dir
  cct_history_dir="$(mktemp -d "${TMPDIR:-/tmp}/kbound-cct20-history.XXXXXX")"
  if [[ -L "$cct_history_dir" || ! -d "$cct_history_dir" ]]; then
    warn "could not create a private CCT release-transaction history directory"
    return 1
  fi
  "$PY" "$KB/scripts/build_cct20_release.py" --refresh-from-existing \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --generated-dir "$KB/paper/generated" \
    --history-dir "$cct_history_dir"
  log "CCT prior-generation transaction archive retained at $cct_history_dir"
}

step_test() {
  verify_release_toolchain_for_phase
  log "MODE test -- complete tracked test inventory, software tests, claim checks, and formal audit"
  local protected_args=()
  if [[ "$KBOUND_AUTHORIZE_PROTECTED_SO2SAT" == "1" ]]; then
    protected_args+=(--authorize-protected-so2sat)
  fi
  "$PY" "$KB/scripts/run_repository_verification.py" \
    --python "$PY" \
    --output "$KB/audits/repository_test_inventory.json" \
    "${protected_args[@]}" \
    --all-gates
  "$PY" -m kbound_repro.release_checks --require-manifest
  validate_manuscript_claims
  "$PY" -m kbound_repro.check_repo --staged
}

step_pdf() {
  verify_release_toolchain_for_phase
  log "MODE pdf -- build the four current PDF roles; render every release page"
  if [[ -f "$KB/scripts/build_pdfs.sh" ]]; then
    BUILD_LONG_TMLR=1 \
    BUILD_SHORT_MAIN=1 \
    BUILD_SHORT_SUPPLEMENT=1 \
    BUILD_FULL_REPORT=1 \
    PYTHON="$PY" BUILD_DOCX=0 bash "$KB/scripts/build_pdfs.sh" || {
      warn "maintained four-role PDF build failed"
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

start_release_run() {
  # Capture HEAD once, before validation/generation can write any outputs.
  # A caller-supplied ref is resolved now and must name that exact clean source.
  RELEASE_SOURCE_COMMIT="$(git -c core.preloadindex=false -c core.fsmonitor=false rev-parse --verify HEAD)"
  readonly RELEASE_SOURCE_COMMIT
  if [[ -n "${KBOUND_SOURCE_COMMIT:-}" ]]; then
    local requested_source
    requested_source="$(git -c core.preloadindex=false -c core.fsmonitor=false rev-parse --verify "${KBOUND_SOURCE_COMMIT}^{commit}")"
    if [[ "$requested_source" != "$RELEASE_SOURCE_COMMIT" ]]; then
      warn "KBOUND_SOURCE_COMMIT must equal the starting HEAD"
      return 1
    fi
  fi
  export KBOUND_SOURCE_COMMIT="$RELEASE_SOURCE_COMMIT"
  log "pin clean release source $RELEASE_SOURCE_COMMIT"
  "$PY" "$KB/scripts/build_release_source_seal.py" --preflight --source-commit "$RELEASE_SOURCE_COMMIT"
}

check_release_source() {
  "$PY" "$KB/scripts/build_release_source_seal.py" --check-source \
    --source-commit "${RELEASE_SOURCE_COMMIT:?release source was not captured}"
}

step_source_seal() {
  local source_commit
  source_commit="${RELEASE_SOURCE_COMMIT:-${KBOUND_SOURCE_COMMIT:-$(git -c core.preloadindex=false -c core.fsmonitor=false rev-parse HEAD)}}"
  log "MODE source-seal -- bind maintained release sources at $source_commit"
  "$PY" "$KB/scripts/build_release_source_seal.py" --source-commit "$source_commit"
  "$PY" "$KB/scripts/build_release_source_seal.py" --check --source-commit "$source_commit"
}

step_public_bundle() {
  log "MODE public-bundle -- source-verify and build portable, content-addressed CCT-20 evidence"
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" \
    --replace-verified
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --check
}

step_verify_public_bundle() {
  log "MODE public-bundle-verify -- verify committed strict-provenance public evidence"
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --check
}

step_anonymous_supplement() {
  verify_release_toolchain_for_phase
  log "MODE anonymous-supplement -- build deterministic reviewer package"
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --root "$REPO" \
    --source-seal "$KB/audits/release_source_seal_2026_08_29.json" \
    --checksums "$KB/KBOUND_RELEASE_SHA256SUMS.txt" \
    --public-bundle "$KB/release/cct20_public_evidence_bundle.zip" \
    --output "$KB/release/kbound_anonymous_supplement.zip" \
    --replace-verified
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$KB/release/kbound_anonymous_supplement.zip" --check
}

step_deep_local_provenance() {
  log "MODE deep-local-provenance -- dereference all sealed CCT-20 machine-local upstream paths"
  KBOUND_DEEP_LOCAL_CCT20_PROVENANCE=1 validate_manuscript_claims
}

step_verify_portable_release() {
  log "MODE verify-portable-release -- verify committed bytes, checkout binding, and archive semantics"
  log "This mode does not claim to reproduce the macOS build or validate machine-local datasets."
  "$PY" "$KB/scripts/build_release_source_seal.py" --check-portable
  "$PY" "$KB/scripts/verify_release_checksums.py" --root "$REPO"
  "$PY" "$KB/scripts/build_cct20_public_bundle.py" \
    --release-manifest "$KB/paper/generated/cct20_release_manifest.json" \
    --output "$KB/release/cct20_public_evidence_bundle.zip" --portable-check
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$KB/release/kbound_anonymous_supplement.zip" --portable-check
  "$PY" "$KB/scripts/verify_release_checksums.py" \
    "$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt" --root "$REPO" --generic \
    --require "$KB/release/kbound_anonymous_supplement.zip"
}

emit_checksums() {
  log "output checksums (authoritative artifacts)"
  local output="$KB/KBOUND_RELEASE_SHA256SUMS.txt"
  # Producer and verifier share one complete authoritative inventory. The
  # default verifier rejects a valid-but-truncated manifest; --generic is opt-in.
  local inventory
  inventory="$("$PY" "$KB/scripts/verify_release_checksums.py" --list-required)"
  local files=()
  local required_path
  while IFS= read -r required_path; do
    if [[ -n "$required_path" ]]; then files+=("$required_path"); fi
  done <<<"$inventory"
  if [[ ${#files[@]} -eq 0 ]]; then
    warn "release checksum inventory is empty"
    return 1
  fi
  # A fresh byte list must not bless an absent, stale, or malformed source seal.
  "$PY" "$KB/scripts/build_release_source_seal.py" --check-structure
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

emit_post_checksums() {
  verify_release_python_environment
  verify_release_toolchain_for_phase
  log "post-checksum receipt for the anonymous package"
  local artifact="$KB/release/kbound_anonymous_supplement.zip"
  local output="$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt"
  if [[ ! -f "$artifact" ]]; then
    warn "missing post-checksum input: $artifact"
    return 1
  fi
  # A digest proves byte identity only. Re-run the complete semantic/privacy
  # verifier under the exact hashed Poppler executables before certifying it.
  "$PY" "$KB/scripts/build_anonymous_supplement.py" \
    --output "$artifact" --portable-check
  local digest
  if have sha256sum; then
    digest="$(sha256sum "$artifact" | awk '{print $1}')"
  elif have shasum; then
    digest="$(shasum -a 256 "$artifact" | awk '{print $1}')"
  else
    warn "no SHA-256 utility available"
    return 1
  fi
  local checksum_tmp
  checksum_tmp="$(mktemp "$KB/.KBOUND_POST_CHECKSUM_SHA256SUMS.XXXXXX")"
  trap 'rm -f "$checksum_tmp"' RETURN
  printf '%s  %s\n' "$digest" "$artifact" >"$checksum_tmp"
  "$PY" "$KB/scripts/verify_release_checksums.py" "$checksum_tmp" \
    --root "$REPO" --generic --require "$artifact"
  mv -f "$checksum_tmp" "$output"
  trap - RETURN
  log "wrote ${output#"$REPO/"}"
}

# --- 3. dispatch ------------------------------------------------------------
case "$MODE" in
  preflight)        step_preflight ;;
  deep-local-preflight) step_preflight; step_deep_local_dataset_preflight ;;
  validate-results) step_preflight; step_validate_results ;;
  generate)         step_preflight; step_generate ;;
  test)             step_preflight; step_test ;;
  pdf)              step_preflight; step_pdf ;;
  source-seal)      step_source_seal ;;
  public-bundle)    step_public_bundle ;;
  public-bundle-verify) step_verify_public_bundle ;;
  anonymous-supplement) step_anonymous_supplement ;;
  post-checksums)    emit_post_checksums ;;
  deep-local-provenance) step_preflight; step_deep_local_provenance ;;
  deep-local-cct20-refresh) step_preflight; step_generate_deep_local_cct20 ;;
  verify-portable-release) step_verify_portable_release ;;
  # Hashing is intentionally environment-independent: this mode reads bytes
  # only. Scientific validation/build modes retain the strict Python preflight.
  checksums)         emit_checksums ;;
  all)
    start_release_run
    step_preflight
    check_release_source
    step_generate
    check_release_source
    step_validate_results
    check_release_source
    step_test
    check_release_source
    step_pdf
    check_release_source
    step_source_seal
    check_release_source
    step_verify_public_bundle
    check_release_source
    emit_checksums
    check_release_source
    step_anonymous_supplement
    check_release_source
    emit_post_checksums
    check_release_source
    "$PY" "$KB/scripts/build_release_source_seal.py" --check --source-commit "$RELEASE_SOURCE_COMMIT"
    "$PY" "$KB/scripts/verify_release_checksums.py" --root "$REPO"
    "$PY" "$KB/scripts/verify_release_checksums.py" \
      "$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt" --root "$REPO" --generic \
      --require "$KB/release/kbound_anonymous_supplement.zip"
    ;;
  all-deep-local)
    start_release_run
    step_preflight
    step_deep_local_dataset_preflight
    check_release_source
    step_generate
    check_release_source
    step_generate_deep_local_cct20
    check_release_source
    step_validate_results
    check_release_source
    step_deep_local_provenance
    check_release_source
    step_test
    check_release_source
    step_pdf
    check_release_source
    step_source_seal
    check_release_source
    step_public_bundle
    check_release_source
    emit_checksums
    check_release_source
    step_anonymous_supplement
    check_release_source
    emit_post_checksums
    check_release_source
    "$PY" "$KB/scripts/build_release_source_seal.py" --check --source-commit "$RELEASE_SOURCE_COMMIT"
    "$PY" "$KB/scripts/verify_release_checksums.py" --root "$REPO"
    "$PY" "$KB/scripts/verify_release_checksums.py" \
      "$KB/KBOUND_POST_CHECKSUM_SHA256SUMS.txt" --root "$REPO" --generic \
      --require "$KB/release/kbound_anonymous_supplement.zip"
    ;;
  *) echo "unknown MODE '$MODE' (preflight|deep-local-preflight|validate-results|generate|test|pdf|source-seal|public-bundle|public-bundle-verify|anonymous-supplement|checksums|post-checksums|deep-local-provenance|deep-local-cct20-refresh|verify-portable-release|all|all-deep-local)" >&2; exit 2 ;;
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
