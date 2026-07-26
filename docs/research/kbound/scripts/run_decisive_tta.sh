#!/usr/bin/env bash
# =============================================================================
#  K-Bound — turnkey DECISIVE deep-TTA run (CIFAR-10-C + Tent/EATA/SAR/KGA)
#  ONE command. Run on your Mac (Apple-silicon / MPS) or any CUDA box.
#
#  From the repo root, just run:
#      bash docs/research/kbound/scripts/run_decisive_tta.sh
#
#  It will: make a venv, install torch+deps, verify MPS, auto-download
#  CIFAR-10-C (~2.9 GB, md5-checked), run a quick smoke test, then the full
#  CIFAR-10-C grid for Tent/EATA/SAR with the KGA certificate, and write
#  results + figures into the repo. Safe to re-run (everything is idempotent).
#
#  Options (env vars):
#      METHODS="tent eata sar"   # which adaptations to wrap (default all three)
#      FULL=1                    # also run the full grid after the quick test (default 1)
#      WITH_C100=1               # also fetch + run CIFAR-100-C (default 0)
#      PYTHON=python3            # interpreter to bootstrap the venv
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DATA_ROOT="$REPO_ROOT/experiments/kbound/cifar"
# Keep the venv on the internal APFS disk: `ensurepip` fails when a venv is created
# on an exFAT external drive (see $KBOUND_EXTERNAL_ROOT). Data + results stay in the repo.
VENV="${KB_VENV:-$HOME/.venv_kbound_tta}"
# Pick a torch-friendly interpreter (torch/torchvision wheels lag new Python releases).
pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for c in python3.12 python3.11 python3.10 python3.13 python3; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
  done
  echo python3
}
PY="$(pick_python)"
METHODS="${METHODS:-tent eata sar}"
FULL="${FULL:-1}"
WITH_C100="${WITH_C100:-0}"

echo "=============================================================="
echo " K-Bound decisive deep-TTA runner"
echo "   repo root : $REPO_ROOT"
echo "   data root : $DATA_ROOT"
echo "   methods   : $METHODS"
echo "=============================================================="

# 1) ---- environment -----------------------------------------------------------
PYVER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "[env] interpreter: $PY (Python $PYVER)"
case "$PYVER" in
  3.10|3.11|3.12) : ;;
  *) echo "[env] NOTE: Python $PYVER may have no prebuilt torch/torchvision wheels (they lag new Pythons)."
     echo "      If the torch install fails below, install Python 3.12 and re-run:"
     echo "        brew install python@3.12      # or:  uv python install 3.12"
     echo "        rm -rf \"$VENV\" && PYTHON=python3.12 bash \"$0\"" ;;
esac
# Rebuild the venv if it is missing or was built with a different Python version.
if [ -d "$VENV" ]; then
  EXIST_VER="$("$VENV/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo none)"
  [ "$EXIST_VER" = "$PYVER" ] || { echo "[env] existing venv is Python $EXIST_VER; rebuilding with $PYVER"; rm -rf "$VENV"; }
fi
# Isolate from any polluted user site-packages (the stray "-pip" warnings earlier).
export PYTHONNOUSERSITE=1; unset PYTHONPATH 2>/dev/null || true
make_venv() {
  rm -rf "$VENV"
  if "$PY" -m venv --copies "$VENV" 2>/tmp/kb_venv_err; then return 0; fi
  echo "[env] standard venv creation failed; bootstrapping pip manually:"; cat /tmp/kb_venv_err || true
  "$PY" -m venv --without-pip --copies "$VENV"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$VENV/bin/python3" /tmp/get-pip.py
}
if [ ! -d "$VENV" ]; then
  echo "[env] creating venv at $VENV (internal disk)"
  make_venv
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[env] installing deps (torch torchvision numpy scikit-learn matplotlib) ..."
python -m pip install --quiet --upgrade pip
if ! python -m pip install --quiet torch torchvision numpy scikit-learn matplotlib; then
  echo "[env] ERROR: dependency install failed (often torch/torchvision on a too-new Python)."
  echo "      Fix: use Python 3.12 ->  rm -rf \"$VENV\" && PYTHON=python3.12 bash \"$0\""
  exit 1
fi

echo "[env] backend check:"
python - <<'PY'
import sys
try:
    import torch, torchvision  # noqa: F401
except Exception as e:
    print("   ERROR importing torch/torchvision:", e)
    print("   -> Most likely no wheel for this Python version. Use Python 3.12 (see note above).")
    sys.exit(1)
mps = getattr(torch.backends, "mps", None)
print("   torch", torch.__version__,
      "| MPS", bool(mps and mps.is_available()),
      "| CUDA", torch.cuda.is_available())
if not ((mps and mps.is_available()) or torch.cuda.is_available()):
    print("   WARNING: no GPU/MPS detected — this will run on CPU and be slow.")
PY

# 2) ---- data ------------------------------------------------------------------
echo "[data] ensuring CIFAR-10-C is present ..."
python "$SCRIPT_DIR/fetch_cifar_c.py" --which 10 --data-root "$DATA_ROOT"
if [ "$WITH_C100" = "1" ]; then
  echo "[data] ensuring CIFAR-100-C is present ..."
  python "$SCRIPT_DIR/fetch_cifar_c.py" --which 100 --data-root "$DATA_ROOT"
fi

# 3) ---- quick smoke test ------------------------------------------------------
echo "[run] QUICK smoke test (subset of corruptions, severities 1 & 5) ..."
python "$SCRIPT_DIR/cifar_tent_mps_v2.py" \
    --benchmarks cifar10c --data-root "$DATA_ROOT" --methods $METHODS --quick

# 4) ---- full grid -------------------------------------------------------------
if [ "$FULL" = "1" ]; then
  BENCH="cifar10c"; [ "$WITH_C100" = "1" ] && BENCH="cifar10c cifar100c"
  echo "[run] FULL grid: $BENCH x {$METHODS} (this is the ~1–3 h run on M5) ..."
  python "$SCRIPT_DIR/cifar_tent_mps_v2.py" \
      --benchmarks $BENCH --data-root "$DATA_ROOT" --methods $METHODS
fi

echo "=============================================================="
echo " DONE. Outputs:"
echo "   results : $REPO_ROOT/experiments/kbound/results/decisive_tta_results.json"
echo "   per-cond: $REPO_ROOT/experiments/kbound/results/per_condition_*_seed*.json"
echo "   manifest: $REPO_ROOT/experiments/kbound/results/result_manifest.json"
echo "   table   : $REPO_ROOT/experiments/kbound/results/decisive_tta_table.md"
echo "   figures : $SCRIPT_DIR/../figures/fig_decisive_*_cifar10c.png"
echo ""
echo " Next: fold decisive_tta_table.md + fig_decisive_pareto_cifar10c.png into"
echo " kbound.tex (replaces the §Limitations CIFAR-100-C/ImageNet-C 'future work'"
echo " line), then recompile:  pdflatex kbound.tex (x2)."
echo "=============================================================="
