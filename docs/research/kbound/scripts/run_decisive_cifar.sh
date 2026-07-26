#!/usr/bin/env bash
# One-touch decisive deep-TTA run for K-Bound. Run on your M5 (MPS) or any CUDA box.
# Produces:  experiments/kbound/results/decisive_tta_results.json + decisive_tta_table.md
#            docs/research/kbound/figures/fig_decisive_*.png
set -euo pipefail

# --- 0. go to repo root ---
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
echo "repo root: $(pwd)"

# --- 1. environment (first time only) ---
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q torch torchvision numpy scikit-learn matplotlib
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# --- 2. data (first time only) — CIFAR-10-C & CIFAR-100-C. VERIFY these Zenodo URLs. ---
D=experiments/kbound/cifar; mkdir -p "$D"
[ -d "$D/CIFAR-10-C" ]  || { curl -L -o "$D/CIFAR-10-C.tar"  https://zenodo.org/records/2535967/files/CIFAR-10-C.tar  && tar -xf "$D/CIFAR-10-C.tar"  -C "$D"; }
[ -d "$D/CIFAR-100-C" ] || { curl -L -o "$D/CIFAR-100-C.tar" https://zenodo.org/records/3555552/files/CIFAR-100-C.tar && tar -xf "$D/CIFAR-100-C.tar" -C "$D"; }

# --- 3. smoke test FIRST (a few minutes) ---
python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks cifar10c --quick

# --- 4. full run (CIFAR-10-C + CIFAR-100-C; Tent/EATA/SAR) ---
python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks cifar10c cifar100c --methods tent eata sar

# --- 5. (optional, heavy ~100+GB) ImageNet-C: download separately, then ---
# python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks imagenetc \
#     --imagenetc-root /path/to/ImageNet-C

echo "DONE -> experiments/kbound/results/decisive_tta_results.json + decisive_tta_table.md"
