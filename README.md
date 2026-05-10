# Universal Anomaly Intelligence System (UAIS‑V)

UAIS-V is a multimodal anomaly-intelligence research codebase for training and
evaluating domain experts, score-level fusion models, and dashboard/API
surfaces. Some domains are fully scripted with local data; others are optional
or connector-style modules that require the corresponding datasets and model
dependencies.

---

## 🌟 Highlights
- **Domain coverage:** LightGBM/CatBoost tabular fraud + cyber, behavior
  sequence modeling, transformer-compatible NLP hooks, vision hooks, optional
  VAE/GAN synthesis, and score-level fusion.
- **MLOps tooling:** Prefect flows, MLflow tracking, reproducible configs, and scripted runners.
- **Explainability:** SHAP summaries, Grad-CAM heatmaps, saliency scores, and drift checks saved under `experiments/`.
- **Deployment surfaces:** FastAPI endpoints (`deploy/api`) and Streamlit dashboard (`dashboard/`) wired to produced artifacts.

---

## 📁 Repository Map (trimmed)
```
config/                  # YAML configs per domain
data/                    # raw / processed datasets
notebooks/               # 00–100 analysis notebooks
src/
  ├── uais/              # primary package
  │   ├── data/, features/, supervised/, anomaly/, sequence/, nlp/, vision/, generative/, fusion/, explainability/
  ├── orchestration/     # Prefect flows
  └── scripts/           # CLI helpers
experiments/             # metrics, plots, saved scores/models
reports/                 # CSV summaries + docs
dashboard/               # Streamlit UI
deploy/                  # FastAPI app
```

---

## ⚙️ Setup

```bash
# clone repo first, then:
python -m venv .venv-macos           # any name works
source .venv-macos/bin/activate      # Windows: .\.venv-macos\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Kaggle credentials (for optional data helpers)
Download `kaggle.json` from https://www.kaggle.com/settings/account and run:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📦 Data

```bash
# Fetch optional public datasets, depending on configured helpers
python scripts/download_data.py --all

# No Kaggle? place the needed CSV/image folders manually and re-run:
python scripts/download_data.py --all --no-kaggle
```

Optional preprocessing:
```bash
bash scripts/run_ingest.sh
bash scripts/run_build_features.sh
```

---

## 🏋️ Training Flows

All scripts assume the virtualenv is active and `PYTHONPATH=src`.

```bash
# Domain trainers (run the ones you need)
bash scripts/run_train_fraud.sh        # LightGBM
bash scripts/run_train_cyber.sh        # CatBoost
bash scripts/run_train_behavior.sh     # LSTM autoencoder
bash scripts/run_train_nlp.sh          # transformer text classifier, if data/deps exist
bash scripts/run_train_vision.sh       # vision classifier, if data/deps exist
python src/uais/generative/train_vae.py --config config/base_config.yaml   # optional VAE/GAN

# Fusion stacker (after domains finish)
bash scripts/run_fusion.sh

# End-to-end (ingest → features → every domain → fusion; ~4 hrs on M-series GPU)
bash scripts/run_full_fusion.sh
```

Outputs:
- `experiments/<domain>/` → models, plots, Grad-CAM, saliency, etc.
- `reports/metrics_<domain>.csv` → scoreboard for dashboard/API.
- `src/mlruns/` → MLflow artifacts.

---

## 📓 Notebooks

Use notebooks for EDA or report figures after scripted training:

| Notebook | Purpose |
|----------|---------|
| `00_data_overview.ipynb` | sanity check & join data sources |
| `10_supervised_fraud.ipynb`, `20_unsupervised_fraud.ipynb` | fraud modeling |
| `30_sequence_models.ipynb` | behavior sequence modeling |
| `70_nlp_email_anomalies.ipynb` | email/news text anomaly experiments |
| `80_vision_forgery_detection.ipynb` | vision classification experiments |
| `90_generative_synthesis.ipynb` | VAE/GAN data augmentation |
| `100_fusion_and_dashboard.ipynb` | combine scores + preview dashboard feeds |

---

## 🖥️ Serving & Dashboard

```
streamlit run dashboard/app_streamlit.py --server.port 8501
uvicorn deploy.api.main:app --reload --port 8000
```
- Streamlit reads from `experiments/<domain>/` & `reports/metrics_*.csv`.  
- FastAPI exposes `/predict_fraud`, `/predict_cyber`, `/predict_behavior`, `/predict_nlp`, `/predict_vision`, `/predict_fusion` if model artifacts exist.

For Dockerized stack (API + Streamlit + MLflow):
```bash
docker-compose up --build
```

---

## 📊 Testing & Reports

- `pytest` covers core helpers (`tests/`).
- Reports + deliverables live under `reports/`, including metrics CSVs consumed by the dashboard and exported plots (e.g., `notebooks/figures/ablations/*.png`).

---

## 🤝 Contributing
1. Fork & branch from `main`.
2. Keep configs/data paths env-agnostic.
3. Run relevant scripts or tests before pushing.
4. PR with a concise summary + screenshots if you touched dashboard/API.

UAIS-V is maintained by **Pratik Niroula**. The repository is intended for reproducible research and engineering review; preserve provenance when adapting it.
