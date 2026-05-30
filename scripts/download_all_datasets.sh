#!/usr/bin/env bash
# Acquire all raw datasets required to reproduce ELARA Phase-2 + Master-C.
#
# Wraps the existing per-dataset acquisition scripts in a single entry point
# and verifies downloaded archives against the frozen SHA256 anchors where
# they exist. The script is idempotent: per-dataset scripts skip files that
# are already on disk.
#
# Total raw data once complete: ~88 GB across data/raw/{eyecandies, mvtec3d,
# real3d, visa, mvtec_loco, unsw_nb15, fraud, behavior, nlp, vision, healthcare}.
#
# Usage:
#   bash scripts/download_all_datasets.sh                 # all datasets
#   bash scripts/download_all_datasets.sh --only eyecandies,mvtec3d
#   bash scripts/download_all_datasets.sh --skip kaggle   # skip everything that needs the Kaggle API
#   bash scripts/download_all_datasets.sh --verify-only   # check hashes, do not download

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export PYTHONPATH=src

ALL=("eyecandies" "mvtec3d" "real3d" "mvtec_loco" "visa" "kaggle_set" "nlp_vision" "healthcare")
ONLY=""
SKIP=""
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)        ONLY="$2"; shift 2 ;;
    --skip)        SKIP="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    *) echo "unknown arg: $1" && exit 2 ;;
  esac
done

want() {
  local name="$1"
  [[ -n "$ONLY"  && ",${ONLY}," != *",${name},"* ]] && return 1
  [[ -n "$SKIP"  && ",${SKIP}," == *",${name},"* ]] && return 1
  return 0
}

verify_hashes() {
  local kind="$1" anchor="$2" dir="$3"
  if [[ ! -f "$anchor" ]]; then
    echo "  [verify $kind] no anchor file at $anchor — skipping"
    return 0
  fi
  echo "  [verify $kind] checking SHA256 anchors from $anchor"
  $PY - <<EOF
import hashlib, sys
from pathlib import Path
anchor = Path("$anchor")
data_dir = Path("$dir")
expected = {}
for line in anchor.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    # Accept either "<sha>  <name>" (two cols) or
    # "<sha>  <bytes>  <name>" (three cols; Eyecandies format).
    if len(parts) == 2:
        sha, name = parts
    elif len(parts) >= 3:
        sha = parts[0]
        name = parts[-1]
    else:
        continue
    # Auto-append .tar if the anchor lists bare category names
    if not (data_dir / name).exists() and (data_dir / f"{name}.tar").exists():
        name = f"{name}.tar"
    expected[name] = sha
ok = miss = bad = 0
for name, sha in expected.items():
    p = data_dir / name
    if not p.exists():
        miss += 1
        print(f"    MISSING: {name}")
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual == sha:
        ok += 1
    else:
        bad += 1
        print(f"    HASH MISMATCH: {name}\n      expected={sha}\n      actual  ={actual}")
print(f"  [verify $kind] ok={ok}/{len(expected)} missing={miss} mismatched={bad}")
sys.exit(0 if bad == 0 and miss == 0 else 1)
EOF
}

echo "==================================================="
echo " ELARA dataset acquisition (verify_only=$VERIFY_ONLY)"
echo "==================================================="

# --- Eyecandies (~27 GB, gdown via Google Drive; required for Family-D)
if want "eyecandies"; then
  echo "[eyecandies] target: data/raw/eyecandies/_archives/*.tar"
  if [[ $VERIFY_ONLY -eq 0 ]]; then
    $PY src/scripts/family_d_v2_download_eyecandies.py || \
      echo "  WARN: eyecandies download script failed (network? gdown?). Continuing."
  fi
  verify_hashes eyecandies \
    "experiments/phase2/family_d/eyecandies_archive_sha256.txt" \
    "data/raw/eyecandies/_archives" || true
fi

# --- MVTec 3D-AD (~27 GB; full archive)
if want "mvtec3d"; then
  echo "[mvtec3d] target: data/raw/mvtec3d/<category>/{train,validation,test}/..."
  if [[ $VERIFY_ONLY -eq 0 ]]; then
    if [[ ! -d data/raw/mvtec3d/bagel ]]; then
      echo "  data/raw/mvtec3d/ appears empty — see download_mvtec3d_kaggle_subset.py for the Kaggle subset path."
      echo "  Full MVTec 3D-AD (~25GB) must be acquired from https://www.mvtec.com/company/research/datasets/mvtec-3d-ad"
    else
      echo "  data/raw/mvtec3d/ already populated; skipping."
    fi
  fi
fi

# --- Real3D-AD (~11 GB)
if want "real3d"; then
  echo "[real3d] target: data/raw/real3d/"
  if [[ ! -d data/raw/real3d ]] || [[ -z "$(ls -A data/raw/real3d 2>/dev/null)" ]]; then
    echo "  Real3D-AD must be acquired manually from https://github.com/M-3LAB/Real3D-AD"
  else
    echo "  data/raw/real3d/ already populated; skipping."
  fi
fi

# --- MVTec LOCO-AD (~12 GB)
if want "mvtec_loco"; then
  echo "[mvtec_loco] target: data/raw/mvtec_loco/"
  if [[ ! -d data/raw/mvtec_loco ]] || [[ -z "$(ls -A data/raw/mvtec_loco 2>/dev/null)" ]]; then
    echo "  MVTec LOCO-AD must be acquired manually from https://www.mvtec.com/company/research/datasets/mvtec-loco"
  else
    echo "  data/raw/mvtec_loco/ already populated; skipping."
  fi
fi

# --- VisA (~4 GB)
if want "visa"; then
  echo "[visa] target: data/raw/visa/"
  if [[ ! -d data/raw/visa ]] || [[ -z "$(ls -A data/raw/visa 2>/dev/null)" ]]; then
    echo "  VisA must be acquired manually from https://github.com/amazon-science/spot-diff"
  else
    echo "  data/raw/visa/ already populated; skipping."
  fi
fi

# --- Kaggle-bundled set: fraud (credit card), cyber (UNSW-NB15), behavior (online shoppers)
if want "kaggle_set"; then
  echo "[kaggle_set] target: data/raw/{fraud,cyber,behavior}/"
  if [[ $VERIFY_ONLY -eq 0 ]]; then
    if ! $PY -c "import kagglehub" 2>/dev/null; then
      echo "  WARN: kagglehub not installed (pip install kagglehub) — skipping kaggle_set."
    else
      $PY src/scripts/download_datasets.py || \
        echo "  WARN: kaggle download script failed (auth? quota?). Continuing."
    fi
  fi
fi

# --- NLP (Enron) + Vision (CIFAR-10)
if want "nlp_vision"; then
  echo "[nlp_vision] target: data/raw/{nlp,vision}/"
  if [[ $VERIFY_ONLY -eq 0 ]]; then
    $PY scripts/download_data.py --all || \
      echo "  WARN: nlp/vision download script failed. Continuing."
  fi
fi

# --- Healthcare GridPulse (M3 development dataset)
if want "healthcare"; then
  echo "[healthcare] target: data/raw/healthcare/"
  if [[ ! -d data/raw/healthcare ]] || [[ -z "$(ls -A data/raw/healthcare 2>/dev/null)" ]]; then
    echo "  Healthcare data acquisition is custom; see research_lock/dataset_registry_v2.yaml for the M3 candidate seal."
  else
    echo "  data/raw/healthcare/ already populated; skipping."
  fi
fi

echo
echo "==================================================="
echo " Dataset acquisition pass complete."
echo " Run 'bash $0 --verify-only' at any time to re-check hash anchors."
echo "==================================================="
