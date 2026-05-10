# UAIS-V (Universal Anomaly Intelligence System – Vision & Language Edition)

**Comprehensive Project Summary & Implementation Guide**

Author: Pratik Niroula

---

## 1. Overview

The UAIS-V Project (Universal Anomaly Intelligence System – Vision & Language Edition) is a multimodal AI research and engineering prototype that brings several anomaly-detection domains into one repository. It combines Machine Learning (ML), Natural Language Processing (NLP), Computer Vision (CV), Generative AI, and Behavioral Analytics modules for finance, cybersecurity, user behavior, text, and image-analysis experiments.

UAIS-V is best read as a reproducible prototype and research scaffold, not a production-validated anomaly platform. This document outlines the technical blueprint, datasets, tools, implementation roadmap, and current evidence boundary of the project.

---

## 2. Project Objectives
1. Build a unified multimodal AI pipeline that ingests, preprocesses, and analyzes structured, textual, and visual data.
2. Detect anomalies in financial transactions, user activity logs, network traffic, and document/media authenticity.
3. Leverage boosting frameworks (LightGBM, XGBoost, CatBoost) to ensure high accuracy and interpretability.
4. Integrate advanced deep learning models (LSTM, Transformer, ViT, GAN) for sequential, textual, and visual domains.
5. Develop a Streamlit dashboard for real-time anomaly visualization and interpretability using SHAP and GradCAM.
6. Implement MLOps practices (Prefect orchestration + MLflow tracking) to enable automated, reproducible experiments.
7. Produce a 4000+ word IEEE-style research report summarizing methods, results, and future enhancements.

---

## 3. Domains and Use Cases

| Domain              | Objective                                 | Real-life Application           |
|---------------------|-------------------------------------------|---------------------------------|
| Fraud Detection     | Identify credit card and financial fraud   | Banking, eCommerce security     |
| Cybersecurity       | Detect network intrusions and threats      | Network operations centers      |
| Behavioral Analytics| Analyze behavior logs or enterprise-style sequences | Corporate IT monitoring         |
| NLP (Text Intelligence) | Detect suspicious or anomalous text patterns | Email/news text triage |
| Computer Vision     | Detect document forgeries and fake media   | KYC verification, forensics     |
| Generative Modeling | Create synthetic datasets for low-data domains | Data augmentation, simulation   |

---

## 4. Frameworks and Libraries

| Category         | Frameworks                                  | Purpose                                 |
|------------------|---------------------------------------------|-----------------------------------------|
| Machine Learning | scikit-learn, XGBoost, LightGBM, CatBoost   | Tabular modeling and baseline performance|
| Deep Learning    | TensorFlow, PyTorch                         | LSTM, CNN, Transformer, and GAN architectures|
| NLP              | Hugging Face Transformers                   | Transformer-compatible text classification|
| Computer Vision  | OpenCV, torchvision, ViT                    | Image preprocessing and forgery detection|
| Generative AI    | TensorFlow-GAN, PyTorch-VAE                 | Synthetic data generation               |
| MLOps            | Prefect, MLflow                             | Workflow orchestration and experiment tracking|
| Visualization    | Streamlit, SHAP, GradCAM                    | Dashboards and explainability tools      |

---

## 5. Datasets Used

| Domain      | Dataset                | Source      | Description                        | Size   |
|-------------|------------------------|------------|------------------------------------|--------|
| Fraud       | Credit Card Fraud      | Kaggle     | 284,807 transactions, labeled      | 150 MB |
| Cybersecurity| UNSW-NB15             | UNSW       | Network traffic logs with attacks  | 2 GB   |
| Behavior    | Online behavior or enterprise-style logs | Local/manual | Sequential behavior records | Var. |
| NLP         | Labeled email/news text | Local/Kaggle | Text classification inputs | Var. |
| Vision      | Document/image datasets | Local/Kaggle | Genuine/forged or anomaly imagery | Var. |
| Generative  | Synthetic fraud/behavior | Custom | Generated using VAE/GAN scripts | Var. |

---

## 6. System Architecture

**Data Flow:**
Raw Data → Ingestion → Cleaning → Feature Engineering → Modeling (ML/DL) → Fusion → Explainability → Dashboard → Reports

**Main Components:**
1. Data Ingestion and Validation (via src/pipeline/ingest.py)
2. Feature Engineering per domain (src/uais/features/)
3. Model Training and Evaluation (src/uais/supervised/, src/uais/anomaly/)
4. Sequence and NLP Modules (src/uais/sequence/, src/uais/nlp/)
5. Vision and Generative Modules (src/uais/vision/, src/uais/generative/)
6. Fusion Layer for unified anomaly scoring (src/uais/fusion/)
7. Dashboard visualization (dashboard/app_streamlit.py)

---

## 7. File Structure (Final)

```
universal-anomaly-intelligence/
│
├─ data/
│   ├─ raw/
│   │   ├─ fraud/creditcard.csv
│   │   ├─ cyber/UNSW-NB15.csv
│   │   ├─ behavior/<behavior_logs>.csv
│   │   ├─ nlp/<email_or_news>.csv
│   │   └─ vision/document_forgery/
│
├─ notebooks/
│   ├─ 00_data_overview.ipynb
│   ├─ 01_eda_fraud.ipynb
│   ├─ 10_supervised_fraud.ipynb
│   ├─ 20_unsupervised_fraud.ipynb
│   ├─ 30_sequence_models.ipynb
│   ├─ 70_nlp_email_anomalies.ipynb
│   ├─ 80_vision_forgery_detection.ipynb
│   ├─ 90_generative_synthesis.ipynb
│   └─ 100_fusion_and_dashboard.ipynb
│
├─ src/
│   ├─ uais/
│   │   ├─ data/, features/, supervised/, anomaly/, nlp/, vision/, generative/, fusion/, explainability/
│   └─ scripts/
│       ├─ run_fraud_experiment.py
│       ├─ run_cyber_experiment.py
│       ├─ run_behavior_experiment.py
│       ├─ run_fusion_experiment.py
│       └─ generate_reports.py
│
├─ dashboard/app_streamlit.py
├─ configs/*.yaml
├─ requirements.txt
└─ README.md
```

---

## 8. Installation & Setup

**Step 1: Create Virtual Environment**

```sh
python3 -m venv venv
source venv/bin/activate
```

**Step 2: Install Dependencies**

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 3: Optional Kaggle credentials for data helpers**

Download `kaggle.json` from your Kaggle account settings, then:

```sh
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 3: Verify GPU (Apple Metal Acceleration)**

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

✅ Output should include /device:GPU:0 for GPU acceleration.

---

## 9. How to Run the Project

### A. Stage Data

```sh
# Download optional NLP/Vision data via helper (requires ~/.kaggle/kaggle.json)
python scripts/download_data.py --all

# Optional: ingest + engineered feature tables
bash scripts/run_ingest.sh
bash scripts/run_build_features.sh
```

If you cannot use Kaggle, place the required local CSV or image folders manually and re-run with `--no-kaggle`.

### B. Train Each Domain (Prefect-powered scripts)

```sh
# run what you need; artifacts saved under experiments/<domain>/
bash scripts/run_train_fraud.sh
bash scripts/run_train_cyber.sh
bash scripts/run_train_behavior.sh
bash scripts/run_train_nlp.sh
bash scripts/run_train_vision.sh
python src/uais/generative/train_vae.py --config config/base_config.yaml  # optional

# fusion stacker after individual models finish
bash scripts/run_fusion.sh

# or trigger everything end-to-end
bash scripts/run_full_fusion.sh
```

Each script wires into `src/orchestration/<domain>_flow.py`, which logs metrics to MLflow and exports models/plots into `experiments/` and `reports/metrics_*.csv`.

### C. Step-by-Step Notebook Execution (for analysis)

Use the notebooks to inspect data, reproduce plots, or validate outputs after training:
1. 00_data_overview.ipynb – Data loading and validation
2. 01_eda_fraud.ipynb – Fraud dataset analysis
3. 10_supervised_fraud.ipynb – Fraud model training walkthrough
4. 20_unsupervised_fraud.ipynb – Isolation Forest anomaly detection
5. 30_sequence_models.ipynb – Behavior sequence modeling
6. 70_nlp_email_anomalies.ipynb – Email/news text anomaly experiments
7. 80_vision_forgery_detection.ipynb – Vision classification experiments
8. 90_generative_synthesis.ipynb – Data generation using GAN/VAE
9. 100_fusion_and_dashboard.ipynb – Combined score and Streamlit integration

### D. Launch Dashboard/API

```sh
streamlit run dashboard/app_streamlit.py
```

---

## 10. Training Time (Approximate)

| Module    | Model         | Est. Runtime (Mac M5 GPU) | Notes                    |
|-----------|--------------|---------------------------|--------------------------|
| Fraud     | LightGBM     | 15–25 min                 | Fast on tabular data     |
| Cyber     | CatBoost     | 40–60 min                 | Requires memory tuning   |
| Behavior  | LSTM         | 1–1.5 hr                  | Sequential data processing|
| NLP       | Transformer text classifier | 40 min | GPU recommended when using large models |
| Vision    | ViT          | 1–2 hr                    | Train on subset first    |
| Generative| GAN/VAE      | 1 hr                      | Optional augmentation    |
| Fusion    | Meta LightGBM| 20 min                    | Final combination layer  |

Times assume Apple Silicon or similar with GPU acceleration. Running `bash scripts/run_full_fusion.sh` executes these sequentially (≈4 hrs).

---

## 11. Example Code Snippets

**LightGBM Model (Fraud)**

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

# Load dataset
df = pd.read_csv('data/raw/fraud/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:,1]
print('ROC-AUC:', roc_auc_score(y_test, preds))
```

**LSTM Autoencoder (Behavior)**

```python
from tensorflow.keras import layers, models

input_shape = (10, 2)  # sequence length, features
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.LSTM(16, activation='relu', return_sequences=False),
    layers.RepeatVector(10),
    layers.LSTM(16, activation='relu', return_sequences=True),
    layers.TimeDistributed(layers.Dense(2))
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=10, batch_size=64)
```

**Transformer Text Classifier (optional example)**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "your-text-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

train_encodings = tokenizer(list(texts), truncation=True, padding=True)
```

---

## 12. Real-World Applications

| Sector      | Use Case                  | Benefit                    |
|-------------|---------------------------|----------------------------|
| Banking     | Fraud prevention          | Reduced financial losses   |
| Cybersecurity| Insider threat detection  | Early intervention         |
| Government  | Document verification     | Identity protection        |
| Enterprise  | Email phishing monitoring | Reduced attack surface     |
| Research    | Data synthesis            | Enables reproducible studies|

---

## 13. Future Enhancements
1. Integrate cross-modal transformers (e.g., CLIP, FLAVA) for multimodal learning.
2. Apply reinforcement learning for adaptive anomaly thresholding.
3. Introduce federated learning for privacy-preserving AI.
4. Extend generative models with diffusion architectures.
5. Deploy complete system via Docker + FastAPI microservices.

---

## 14. Research Status and Evidence Boundary
- Applied scope: Demonstrates integrated ML, DL, NLP, CV, and MLOps components.
- Research scope: Provides a reproducible multimodal anomaly-fusion prototype with benchmark scripts, generated metrics, and known limitations.
- Evidence boundary: Current results should be treated as draft research evidence until naturally paired benchmarks, external baselines, and independent replication are complete.

---

## 15. Conclusion

UAIS-V represents a multimodal AI research prototype capable of fusing structured, textual, and visual data into a unified anomaly-analysis pipeline. Through modular ML, NLP, and CV subsystems, explainability layers, and orchestration scripts, it provides a reproducible foundation for further evaluation.

The system’s adaptability is supported by boosting models, neural networks, and fusion experiments across multiple domains. Its current value is strongest as an auditable research and engineering artifact; deployment or publication claims require the additional validation described in the research manuscript.

---

End of UAISV_Final_Project_Summary.md
