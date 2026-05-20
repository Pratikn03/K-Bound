# Dataset download runbook for Tracks A1-A3

This file gives you the exact download commands and expected layouts for
MVTec LOCO-AD (A1), Real3D-AD (A2), and VisA (A3). All three are
public; none requires credentialed access.

Run these in parallel (different connections). Total wall-clock at
30 MB/s: ~25-40 minutes. After each completes, the existing
`prepare_*_fusion_benchmark.py` scripts and runner configs are ready
to consume the data with one command.

---

## A1 - MVTec LOCO-AD (~10 GB)

**Source:** MVTec Software GmbH (no login required).

```bash
# 1. Create target directory
mkdir -p data/raw/mvtec_loco

# 2. Download (single tar.xz, ~10 GB)
cd data/raw/mvtec_loco
curl -L -O https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz

# 3. Extract
tar -xJf mvtec_loco_anomaly_detection.tar.xz
rm mvtec_loco_anomaly_detection.tar.xz
cd ../../..

# 4. Expected layout
# data/raw/mvtec_loco/
#   breakfast_box/
#     train/good/{rgb,xyz}/...
#     test/{good,logical_anomalies,structural_anomalies}/{rgb,xyz}/...
#   juice_bottle/
#   pushpins/
#   screw_bag/
#   splicing_connectors/
```

**Once downloaded, build the fusion CSV with:**

```bash
PYTHONPATH=src python src/scripts/prepare_mvtec_loco_fusion_benchmark.py \
  --dataset-root data/raw/mvtec_loco \
  --feature-mode patchcore \
  --embedding-dim 16 \
  --output experiments/fusion/mvtec_loco_patchcore_inputs.csv \
  --metadata experiments/fusion/mvtec_loco_patchcore_metadata.json
```

**Then run the 5-seed benchmark:**

```bash
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_mvtec_loco_patchcore.yaml \
  --output experiments/fusion/mvtec_loco_patchcore_results.json
```

---

## A2 - Real3D-AD (~5 GB)

**Source:** Liu et al. NeurIPS 2023; mirrored on Google Drive.

```bash
# 1. Create target directory
mkdir -p data/raw/real3d

# 2. Download
# Real3D-AD is distributed as a single Google Drive folder; the official
# URL is https://github.com/M-3LAB/Real3D-AD. The dataset is split across
# 12 category zip files (airplane, candybar, chicken, diamond, duck, ...).
#
# Use gdown:
pip install gdown
cd data/raw/real3d

# Each category is a separate Google Drive file id; fetch each:
# (the exact ids are listed at https://github.com/M-3LAB/Real3D-AD#dataset)
# Example for one category:
gdown --folder https://drive.google.com/drive/folders/<FOLDER_ID>

# 3. Extract each category zip
for f in *.zip; do unzip "$f" && rm "$f"; done
cd ../../..

# 4. Expected layout
# data/raw/real3d/
#   airplane/
#     train/{*.pcd}        # point cloud files
#     test/{good,*}/{*.pcd}
#   candybar/
#   ...
```

**Note:** Real3D-AD uses point clouds, not RGB+depth. The prep script
`prepare_real3d_fusion_benchmark.py` builds two co-observed domains:
the raw point cloud (FPFH features) and a depth-image projection. Both
are derived from the same .pcd file, so the pairing is natural.

```bash
PYTHONPATH=src python src/scripts/prepare_real3d_fusion_benchmark.py \
  --dataset-root data/raw/real3d \
  --output experiments/fusion/real3d_fusion_inputs.csv \
  --metadata experiments/fusion/real3d_fusion_metadata.json
```

**Then:**

```bash
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_real3d_fusion.yaml \
  --output experiments/fusion/real3d_fusion_results.json
```

---

## A3 - VisA (~10 GB)

**Source:** Amazon Visual Anomaly dataset (Zou et al. ECCV 2022).

```bash
# 1. Create target directory
mkdir -p data/raw/visa

# 2. Download (single zip, ~10 GB)
cd data/raw/visa
curl -L -O https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
tar -xf VisA_20220922.tar
rm VisA_20220922.tar
cd ../../..

# 3. Expected layout
# data/raw/visa/
#   candle/
#     Data/
#       Images/
#         Normal/...
#         Anomaly/...
#       Masks/
#         Anomaly/...
#     image_anno.csv
#     split_csv/1cls.csv
#   capsules/
#   cashew/
#   ... (12 categories)
```

**VisA is RGB-only, so the prep script augments it with synthetic
"shape-prior" depth derived from edge / gradient features. This is
documented in the script as a "depth-proxy" rather than true paired
depth, but it gives the fusion pipeline two domains to operate on while
keeping a real visual benchmark in the cross-benchmark contrast.**

```bash
PYTHONPATH=src python src/scripts/prepare_visa_fusion_benchmark.py \
  --dataset-root data/raw/visa \
  --output experiments/fusion/visa_fusion_inputs.csv \
  --metadata experiments/fusion/visa_fusion_metadata.json
```

**Then:**

```bash
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_visa_fusion.yaml \
  --output experiments/fusion/visa_fusion_results.json
```

---

## One-shot orchestration after downloads complete

The union runner is wired with all three new prep + train pairs:

```bash
PYTHONPATH=src python src/scripts/run_union_research_system.py \
  --mode paper \
  --with-tests \
  --continue-on-error
```

This will skip any step whose `required_paths:` are missing, so partial
downloads work too.

---

## Disk space summary

| Dataset | Compressed | Extracted | Total |
|---|---|---|---|
| MVTec LOCO-AD | ~3 GB | ~10 GB | 10 GB |
| Real3D-AD | ~2 GB | ~5 GB | 5 GB |
| VisA | ~4 GB | ~10 GB | 10 GB |
| **Sum** | **~9 GB** | **~25 GB** | **~25 GB** |

Together with the existing MVTec 3D-AD (~26 GB) and the GridPulse
parquet, this brings `data/raw/` to ~55 GB. Make sure the volume hosting
the repo has room.
