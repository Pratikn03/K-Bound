# `data/` — raw datasets for VERA/RGA

This directory holds the public datasets the VERA/RGA pipeline trains on.
The contents of `data/raw/` are **gitignored** (large files plus license
restrictions); each user must download the datasets themselves and place
the files at the paths listed below.

Quick setup (creates empty folders only):
```bash
mkdir -p data/raw/fraud data/raw/cyber data/raw/behavior data/raw/nlp/fakenews data/raw/mvtec3d data/raw/vision data/interim data/processed
```

`interim/` and `processed/` are temporary scratch directories used by some
preprocessing scripts; they are also gitignored and don't need to be
populated manually.

---

## RealFusion-LA benchmark — 4 tabular/text domains

Used by [`configs/attention_real_fusion.yaml`](../configs/attention_real_fusion.yaml)
via [`src/scripts/prepare_real_fusion_benchmark.py`](../src/scripts/prepare_real_fusion_benchmark.py).

| Domain | File path | Source | Notes |
|---|---|---|---|
| fraud | `data/raw/fraud/creditcard.csv` | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | ~144 MB; 284 807 rows |
| cyber | `data/raw/cyber/UNSW_NB15_training-set.csv` | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Pre-split training partition |
| cyber | `data/raw/cyber/UNSW_NB15_testing-set.csv` | UNSW-NB15 | Pre-split testing partition |
| behavior | `data/raw/behavior/online_shoppers_intention.csv` | [UCI — Online Shoppers Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | ~1 MB; 12 330 sessions |
| nlp | `data/raw/nlp/fakenews/fake_news_labeled.csv` | [Kaggle — Fake News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | Must have `text` + binary label columns |

After uploading the four files, build the composite benchmark:
```bash
python -m src.scripts.prepare_real_fusion_benchmark \
    --output experiments/fusion/real_domain_fusion_inputs.csv \
    --metadata experiments/fusion/real_domain_fusion_metadata.json
```
This emits the long-format CSV that `configs/attention_real_fusion.yaml`
consumes.

---

## MVTec 3D-AD — naturally paired RGB + 3D anomaly detection

Used by [`configs/attention_mvtec3d_fusion.yaml`](../configs/attention_mvtec3d_fusion.yaml)
via [`src/scripts/prepare_mvtec3d_fusion_benchmark.py`](../src/scripts/prepare_mvtec3d_fusion_benchmark.py).

| Asset | File path | Source |
|---|---|---|
| bagel category (smallest, ~700 MB) | `data/raw/mvtec3d/bagel/{train,validation,test}/...` | [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) |

Expected layout within each category:
```
data/raw/mvtec3d/bagel/
  train/good/{rgb,xyz}/*.{png,tiff}
  validation/good/{rgb,xyz}/*.{png,tiff}
  test/good/{rgb,xyz}/*.{png,tiff}
  test/<defect_class>/{rgb,xyz,gt}/*.{png,tiff}
```

Build the fusion CSV:
```bash
python -m src.scripts.prepare_mvtec3d_fusion_benchmark \
    --dataset-root data/raw/mvtec3d \
    --output experiments/fusion/mvtec3d_fusion_inputs.csv \
    --metadata experiments/fusion/mvtec3d_fusion_metadata.json
```

---

## End-to-end training

Once the prep step has produced the input CSVs, run a full multi-seed
experiment with all reviewer stress tests (τ-sweep, component ablation,
drift, adversarial, missing-domain, calibration, CDA):

```bash
# Real-fusion 4-domain benchmark
python -m src.scripts.run_breakthrough_experiment \
    --config configs/attention_real_fusion.yaml

# MVTec 3D-AD smoke benchmark
python -m src.scripts.run_breakthrough_experiment \
    --config configs/attention_mvtec3d_fusion.yaml
```

CPU is fine — both configs are sized for laptop runs (≤ 1 h per seed on a
4-domain real-fusion run; minutes for MVTec bagel).

To compare batch-level vs per-sample RGA gating (paper r_{i,d} formalism),
set `reliability.per_sample_gating: true` in the config and re-run.

---

## Where outputs land

| Artifact | Path |
|---|---|
| Per-seed metrics JSON | `experiments/fusion/*_results.json` |
| Aggregated tables (LaTeX) | `docs/research/tables/*.tex` |
| Paper figures (PNG) | `docs/research/figures/*.png` |
| Trained model checkpoint | `models/fusion/attention_{real,mvtec3d}/...pt` |
| Fitted reliability estimator | `experiments/{benchmark}/reliability_estimator.joblib` |

---

## Synthetic-only smoke run (no real data required)

Useful for CI / sanity checks without uploading the raw datasets:
```bash
python -m src.scripts.run_breakthrough_experiment --synthetic
```
This uses internally generated data and exercises every code path (τ-sweep,
ablation, drift, adversarial, missing, CDA) in ~2 minutes.

---

## Optional secondary datasets

These are not required for the VERA/RGA paper but referenced by other
research notebooks in the repo:

| Use case | Path | Source |
|---|---|---|
| Vision side-experiments | `data/raw/vision/cifar-10-python/` | Pull via `python src/scripts/download_nlp_vision.py --cifar10` |
| Email NLP (Enron) | `data/raw/nlp/enron_emails.csv` | Pull via `python src/scripts/download_nlp_vision.py --enron` |
