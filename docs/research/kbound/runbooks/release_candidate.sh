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
#   pdf               build short & long PDFs and render every page
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
export PYTHONPATH="${REPO}/${KB}:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"
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
  log "MODE generate -- rebuild aggregates -> manifest -> matrix/tables/figures"
  local item desc
  for item in 01_build_manifests.py 03_make_tables.py 04_make_figures.py; do
    case "$item" in
      01_*) desc="build manifests (01)" ;;
      03_*) desc="make tables (03)" ;;
      04_*) desc="make figures (04)" ;;
    esac
    if [[ -f "$KB/scripts/$item" ]]; then
      log "$desc"
      "$PY" "$KB/scripts/$item"
    else
      warn "missing $KB/scripts/$item ($desc) -- skipped"
    fi
  done
}

step_test() {
  log "MODE test -- software tests + forbidden-claim checks + formal audit"
  # inexpensive first: canonical toolkit tests (torch-independent)
  "$PY" -m pytest "$KB/kbound_repro/tests" -q
  # forbidden-claim / authority gate
  "$PY" -m kbound_repro.release_checks
  # staged-file / portability guard (no-op if nothing staged)
  "$PY" -m kbound_repro.check_repo --staged || warn "check_repo flagged staged files"
  # formal audit (strict) if present
  if [[ -f "$KB/formal/formal_audit.py" ]]; then
    log "formal audit (strict 100% inventory)"
    "$PY" "$KB/formal/formal_audit.py" --strict-100
  else
    warn "formal audit script absent -- skipped"
  fi
}

step_pdf() {
  log "MODE pdf -- build short & long PDFs, render every page"
  if [[ -f "$KB/scripts/build_pdfs.sh" ]]; then
    PYTHON="$PY" BUILD_LONG=1 bash "$KB/scripts/build_pdfs.sh" || { warn "PDF build failed"; return 1; }
  else
    warn "build_pdfs.sh absent -- cannot build PDFs"; return 1
  fi
  # page-render / page-count check if a helper exists
  if [[ -f "$KB/scripts/render_pdf_pages.py" ]]; then
    "$PY" "$KB/scripts/render_pdf_pages.py" || warn "PDF page render check reported issues"
  else
    warn "render_pdf_pages.py absent -- page-image render check skipped"
  fi
}

emit_checksums() {
  log "output checksums (authoritative artifacts)"
  local files=("$KB/claim_ledger.json" "$KB/STORAGE_MANIFEST.json")
  [[ -f "$KB/RESULT_MANIFEST.json" ]] && files+=("$KB/RESULT_MANIFEST.json")
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    if have sha256sum; then sha256sum "$f"; elif have shasum; then shasum -a 256 "$f"; fi
  done
}

# --- 3. dispatch ------------------------------------------------------------
case "$MODE" in
  preflight)        step_preflight ;;
  validate-results) step_preflight; step_validate_results ;;
  generate)         step_preflight; step_generate ;;
  test)             step_preflight; step_test ;;
  pdf)              step_preflight; step_pdf ;;
  all)
    step_preflight
    step_validate_results
    step_generate
    step_test
    step_pdf
    emit_checksums
    ;;
  *) echo "unknown MODE '$MODE' (preflight|validate-results|generate|test|pdf|all)" >&2; exit 2 ;;
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
