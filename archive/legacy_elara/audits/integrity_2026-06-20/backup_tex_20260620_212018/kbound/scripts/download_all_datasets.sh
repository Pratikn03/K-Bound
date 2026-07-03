#!/usr/bin/env bash
# =============================================================================
#  download_all_datasets.sh — fetch the data for the K-Bound deep-TTA experiments
#  RUN THIS ON YOUR MAC (downloads land on T9). Resumable + idempotent.
#
#    bash docs/research/kbound/scripts/download_all_datasets.sh
#
#  Gets:  (1) Camelyon17 (WILDS)         ~10 GB
#         (2) ImageNet-C (Zenodo 2235448) ~47 GB (15 corruptions; +15 GB with extra)
#         (3) ResNet-50 + ViT-B/16 pretrained weights (small, cached by torchvision)
#
#  Env options:
#    DATA_ROOT=/path        where data goes (default <repo>/experiments/kbound/data)
#    WITH_EXTRA=1           also fetch ImageNet-C extra.tar (speckle/spatter/gaussian_blur/saturate)
#    SKIP_CAMELYON=1 | SKIP_IMAGENETC=1 | SKIP_BACKBONES=1
#    PYTHON=python3.12      interpreter to bootstrap the venv
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/experiments/kbound/data}"
VENV="${KB_VENV:-$HOME/.venv_wilds}"          # on internal disk (APFS), not exFAT
PY="${PYTHON:-python3}"
mkdir -p "$DATA_ROOT"

echo "=============================================================="
echo " K-Bound dataset downloader"
echo "   repo      : $REPO_ROOT"
echo "   data root : $DATA_ROOT"
echo "   free space:"; df -h "$DATA_ROOT" | tail -1
echo "=============================================================="
# Sanity: need ~60 GB free for the full pull.
FREE_GB=$(df -Pk "$DATA_ROOT" | tail -1 | awk '{print int($4/1024/1024)}')
echo "   ~${FREE_GB} GB free; need ~60 GB for Camelyon17 + ImageNet-C(15)."
[ "${FREE_GB:-0}" -lt 60 ] && echo "   WARNING: low free space."

dl() {  # dl URL OUTFILE  (resumable)
  local url="$1" out="$2"
  if command -v wget >/dev/null 2>&1; then wget -c -O "$out" "$url"
  else curl -fL -C - -o "$out" "$url"; fi
}

# ── venv with torch/torchvision/wilds ────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  command -v uv >/dev/null 2>&1 || { echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  uv venv "$VENV" --python 3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# uv-created venvs ship no `pip`; install into the active venv via `uv pip`.
PIP="uv pip"; command -v uv >/dev/null 2>&1 || PIP="python -m pip"
python -c "import torch" 2>/dev/null || $PIP install -q torch==2.5.1 torchvision==0.20.1
python -c "import wilds" 2>/dev/null || $PIP install -q wilds

# ── 1. Camelyon17 (WILDS) ────────────────────────────────────────────────────
if [ "${SKIP_CAMELYON:-0}" != 1 ]; then
  echo "==> [1/3] Camelyon17 (WILDS) -> $DATA_ROOT/wilds  (~10 GB) ..."
  python - "$DATA_ROOT/wilds" <<'PY'
import sys
from wilds import get_dataset
get_dataset(dataset="camelyon17", download=True, root_dir=sys.argv[1])
print("    camelyon17 ready")
PY
fi

# ── 2. ImageNet-C (Zenodo record 2235448, grouped tars) ──────────────────────
if [ "${SKIP_IMAGENETC:-0}" != 1 ]; then
  IC="$DATA_ROOT/imagenet-c"; mkdir -p "$IC"
  TARS="blur noise weather digital"            # standard 15 corruptions
  [ "${WITH_EXTRA:-0}" = 1 ] && TARS="$TARS extra"
  echo "==> [2/3] ImageNet-C -> $IC  (tars: $TARS) ..."
  for t in $TARS; do
    if [ -f "$IC/.done_$t" ]; then echo "    $t already extracted, skip"; continue; fi
    echo "    downloading $t.tar ..."
    dl "https://zenodo.org/records/2235448/files/$t.tar?download=1" "$IC/$t.tar"
    echo "    extracting $t.tar ..."
    tar xf "$IC/$t.tar" -C "$IC" && touch "$IC/.done_$t" && rm -f "$IC/$t.tar"
  done
  echo "    ImageNet-C layout: $IC/<corruption>/<severity 1-5>/<class>/*.JPEG"
fi

# ── 3. Backbone weights (small; cached in torch hub) ─────────────────────────
if [ "${SKIP_BACKBONES:-0}" != 1 ]; then
  echo "==> [3/3] Caching ResNet-50 + ViT-B/16 pretrained weights ..."
  python - <<'PY'
import torchvision.models as tvm
tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
print("    backbones cached in torch hub")
PY
fi

echo "=============================================================="
echo " DONE. Next:"
echo "   WILDS Camelyon17 :  WILDS_ROOT=$DATA_ROOT/wilds bash docs/research/kbound/scripts/run_wilds.sh"
echo "   ImageNet-C       :  python docs/research/kbound/scripts/cifar_tent_mps_v2.py \\"
echo "                          --benchmarks imagenetc --imagenetc-root $DATA_ROOT/imagenet-c \\"
echo "                          --arch resnet50 --quick"
echo "=============================================================="
