# UAIS-V Research Quality Audit

**Repo:** `AutoML_Flagship_V8` (Universal Anomaly Intelligence System)
**Audit date:** 2026-05-07
**Scope:** ML research methodology — experiment design, splits, metrics, baselines, statistical rigor, reproducibility.
**Verdict:** **The published metrics are not defensible as research results.** Several issues are severe enough that current numbers in `reports/metrics_*.csv` and the IEEE-style summary should not be cited until fixed. Most issues are mechanical and fixable.

---

## TL;DR — what to fix before claiming results

| # | Severity | Domain | Issue | Action |
|---|---|---|---|---|
| 1 | 🔴 Critical | Cyber | Loader concatenates official train + test splits with raw split files → train/test leakage; 17.28% duplicates in audit | Use `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` only; honor the official split |
| 2 | 🔴 Critical | Fusion | `_train_fraud_scores` / `_train_cyber_scores` score the **full** X (train + val) with a model fit on train → leakage into meta-model | Score only held-out folds (out-of-fold predictions) |
| 3 | 🔴 Critical | Fusion | Domain rows are concatenated by index (`fraud[i]`, `cyber[i]`, `behavior[i]`); these are unrelated entities, not aligned samples | Either build a dataset with shared sample IDs, or drop sample-level fusion and document the framework as conceptual only |
| 4 | 🔴 Critical | Behavior | Documented as CERT r4.2 insider-threat dataset; actual data is `online_shoppers_intention.csv` (e-commerce conversion). Task and dataset disagree | Either re-source CERT r4.2 or rewrite docs as e-commerce intent classification |
| 5 | 🔴 Critical | NLP | Documented as Enron phishing detection; only working pipeline uses `fakenews/fake_news_labeled.csv`. Transformer trainer is a stub returning `accuracy: 0.0` | Pick one: real fake-news task with DistilBERT fine-tune, or actual Enron phishing labels |
| 6 | 🔴 Critical | All | `reports/metrics_*.csv` values are suspiciously round (0.92, 0.82…), `std` column is empty, no `cv_metrics.csv` is produced. They appear to be hand-written placeholders, not run outputs | Regenerate via `make_tables.py` after real runs; do not cite current numbers |
| 7 | 🟠 High | Fraud | Random stratified split on `creditcard.csv` despite `Time` column. Random splitting with temporal data masks distribution drift | Use temporal split (oldest 70% train, next 10% val, newest 20% test) |
| 8 | 🟠 High | Vision | `image_dataset_from_directory` is pointed at the dataset root containing pre-existing `Train/`, `Test/`, `Validation/` folders, then `validation_split=0.2` re-shuffles them | Use the predefined splits; do not let TF re-split |
| 9 | 🟠 High | All | Single seed (42) everywhere. No multi-seed runs, no confidence intervals, no statistical tests | Run ≥3 seeds; report mean ± std; bootstrap CIs for AUC |
| 10 | 🟠 High | All | No real baselines reported. README claims "baseline performance" but no baseline metric is logged alongside the boosting model | Log a logistic-regression / majority-class baseline per domain |
| 11 | 🟠 High | Repro | `mlflow.enabled: false` in every domain config despite README claiming MLflow tracking; no per-run env capture | Either enable MLflow or remove the claim. Pin Python (`.python-version`) and lock dependencies |
| 12 | 🟡 Medium | Configs | `cyber_baseline.yaml` and `behavior_baseline.yaml` contain the same YAML document twice (concatenated) — last block wins silently | Deduplicate |
| 13 | 🟡 Medium | NLP / Vision | `train_transformer_text.py` is a TODO stub (`accuracy: 0.0`); Vision config says `resnet18` but code only supports `simple_cnn`/`resnet50` | Implement or remove from documented pipeline |
| 14 | 🟡 Medium | Fraud | Isolation Forest fit on `X_train` is reasonable, but **anomaly + supervised blend weights `alpha=0.7, beta=0.3` are hard-coded** and never tuned on a held-out set | Tune on val; report sensitivity |
| 15 | 🟡 Medium | All | Synthetic-data fallbacks silently activate when files are missing (`load_fraud_data`, `load_cyber_data`). A run on synthetic data can produce metrics indistinguishable from real | Make synthetic mode opt-in; log a banner warning when used |

---

## 1. What this project actually is

UAIS-V is a multi-domain anomaly-intelligence system: tabular fraud/cyber, sequence behavior, NLP, vision, and a fusion meta-model. The MLOps surface (Prefect flows, FastAPI, Streamlit, pre-commit, CI) is in good shape. The **research** layer underneath is where the issues are.

Layout summary (post-audit):

- `data/raw/{fraud,cyber,behavior,nlp,vision}/` — datasets present
- `src/uais/{supervised,anomaly,sequence,nlp,vision,fusion,…}` — model code
- `src/scripts/run_*_experiment.py` — per-domain runners
- `configs/*.yaml` and `config/*.yaml` — **two parallel config dirs, both used** (configs/baseline read by `run_*_experiment.py`, config/ read by `train_vae` and others)
- `reports/metrics_*.csv` — final summary tables (currently unreliable)
- `experiments/<domain>/metrics/metrics.{json,csv}` — per-run outputs
- `tests/` — 16 test files; mostly unit-level

---

## 2. Domain-by-domain findings

### 2.1 Fraud (`creditcard.csv`)

**Pipeline:** `src/scripts/run_fraud_experiment.py` → `train_fraud_supervised.py` (`hist_gb` default) + Isolation Forest blend.

What works:
- Stratified 60/20/20 split with `random_state=42`.
- Reasonable metric set (`roc_auc`, `pr_auc`, `f1`, `precision`, `recall`, ECE, Brier, TPR@FPR=1%) in `utils/metrics.py:69`.
- Pipeline auto-encodes categoricals (`build_tabular_pipeline`).

Problems:
- **Temporal leakage.** `creditcard.csv` has a `Time` column (seconds since first transaction). `train_test_split(stratify=y, random_state=42)` ignores time, so future transactions appear in the train fold. ROC-AUC under random splits is known to overstate performance vs. forward-chained splits.
- **Hybrid weights `alpha=0.7, beta=0.3` are hard-coded** at `src/scripts/run_fraud_experiment.py:95`; never tuned, never ablated.
- **Synthetic fallback** (`_synthetic_fraud`) silently activates if the file is missing. A run that produces "great metrics" on synthetic data is indistinguishable from a real run unless you read the warning print.
- The threshold for F1 is fixed at 0.5; with class imbalance ~0.17%, this is suboptimal — `best_f1_threshold` exists in `metrics.py` but is not invoked here.

### 2.2 Cyber (`UNSW-NB15`)

**Pipeline:** `src/scripts/run_cyber_experiment.py` → `train_cyber_supervised.py`.

The most damaging issue:
- `load_cyber_data` does `raw_dir.rglob("*.csv")` then `pd.concat` (line 58, 113). The directory contains both:
  - The four split CSVs `UNSW-NB15_{1,2,3,4}.csv` (the original full dump), and
  - The official `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
- Those overlap by construction. `data_audit.md` already reports **17.28% duplicates** and missing rates of 79.30% / 40.13% in unnamed columns — both red flags consistent with concatenating heterogeneous files with mismatched schemas.
- After concat, `train_test_split` reshuffles everything → records that are part of UNSW's official **test** set appear in the model's **train** fold. Reported AUCs are inflated.

Other:
- Same hard-coded `random_state=42`, single split, no out-of-fold averaging in the runner (CV helper exists in `train_cyber_supervised.py:116` but isn't called by the experiment script).
- `cyber_baseline.yaml` is duplicated (the file is two YAMLs concatenated). PyYAML returns the last document — silent.

### 2.3 Behavior (claimed CERT r4.2, actually `online_shoppers_intention.csv`)

- Documentation everywhere (`README.md`, `UAISV_Final_Project_Summary.md`) describes CERT r4.2 insider-threat logon logs.
- `configs/data_behavior.yaml` and the behavior runner load `data/raw/behavior/online_shoppers_intention.csv` with target `Revenue` — that's the UCI Online Shoppers Purchasing Intention dataset (12,330 rows of e-commerce sessions, label = whether the visitor bought something).
- **The task as documented (insider threat) and the task as run (purchase prediction) are different problems.** Any "behavior anomaly" claims based on this run are unsupported.
- The autoencoder + LOF approach assumes anomalous = rare, but `Revenue` is just imbalanced binary — not the same as rare-event anomaly. The reconstruction-error metric reported for "behavior anomaly" is therefore measuring something other than what the docs describe.

### 2.4 NLP (claimed Enron phishing, actually fake news)

- README says: "DistilBERT NLP" + "Detect phishing and insider communication anomalies" + "Enron Emails (Kaggle, 400MB)."
- Real artifacts on disk: `data/raw/nlp/enron_emails.csv` exists (text + binary label) **and** `data/raw/nlp/fakenews/fake_news_labeled.csv`.
- `src/uais/nlp/train_text_classifier.py` is a TF-IDF + logistic regression, not DistilBERT. The transformer file `train_transformer_text.py` is a placeholder that returns `{"model": ..., "accuracy": 0.0, "f1": 0.0}` — i.e., it has not been implemented.
- Either way, no actual phishing detection is happening. The docs and the code are about different tasks.

### 2.5 Vision (document forgery)

- Folder layout: `data/raw/vision/document_forgery/{Test,Train,Validation}` — the dataset *already provides splits*.
- `run_vision_experiment` in `src/uais/vision/train_vision_model.py` uses `tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset=...)`. Pointed at the parent folder, TF reads `Train/`, `Test/`, `Validation/` as **classes**, then re-splits 80/20. The pre-existing test set is not held out.
- Config says `resnet18` but the code only branches for `resnet18 | resnet50 | resnet` and instantiates **ResNet50** for all three (`train_vision_model.py:42`). The "ResNet18" claim in docs is incorrect.
- ViT mentioned in docs is not implemented anywhere in `src/uais/vision/`.

### 2.6 Fusion (the most fragile link)

`run_fusion_experiment.py`:

1. `_train_fraud_scores` fits a fraud model on a 80/20 split, then immediately calls `model.predict_proba(X)[:, 1]` on the **entire X** (line 58). The score for every training row is generated by a model that already saw it. Same pattern in `_train_cyber_scores` (line 81). Behavior reuses the autoencoder on its own training data (line 99).
2. `_save_attention_fusion_inputs` aligns by `min_len` truncation (line 162). Fraud, cyber, and behavior are different datasets with no shared entity. Row 17 in fraud is not the same as row 17 in cyber. The "multimodal fusion" is effectively training a meta-model on synthetic alignment.
3. `train_fusion_meta_model` then `train_test_split`s the meta-features and reports test AUC — but its features were generated via the leakage from step 1. The 0.97 ROC-AUC in `metrics_fusion.csv` is consistent with this.

Until samples are joinable (shared `sample_id`), fusion AUC numbers should not be reported as a measurement of fused detection quality.

---

## 3. Statistical rigor

- **One seed.** Every `random_state=42` is hard-coded in configs and scripts. Re-running gives identical numbers; nothing about the variance of the estimator is measured.
- **No confidence intervals.** `metrics.py` produces point estimates; no bootstrap CI for AUC, no DeLong test, no McNemar between hybrid vs. supervised.
- **Empty `std` column.** `reports/metrics_fraud.csv` literally has `std,` empty. `make_tables.py` *would* compute std if multiple metric files existed — but there's only one. So the aggregator is well-meaning, but no upstream variance feeds it.
- **No baselines reported.** README mentions LightGBM/CatBoost/XGBoost/HistGB. The runner uses `hist_gb` only (`run_fraud_experiment.py:62`); no head-to-head with the alternatives is logged. No naive baseline (logistic regression, prevalence-only) is shown.
- **No ablations end-to-end.** Folder `notebooks/figures/ablations/` is referenced in the README but not produced by any scripted run.

---

## 4. Reproducibility

| Item | State |
|---|---|
| Python version pin | `.python-version` exists ✅ |
| Dependency lock | `requirements.txt` uses lower bounds (`>=`), no lockfile |
| Determinism utilities | `uais_v/utils/seed.set_global_seed` exists but is **only called by `train_30seq` / `train_nlp` / `train_vision` in the `uais_v` package** — the actual experiment scripts in `src/scripts/` do not call it |
| MLflow | `mlflow_config.yaml` exists; every domain config has `mlflow.enabled: false` |
| Data hash / version | None |
| Run manifest | `experiments/<domain>/metrics/metrics.json` exists per domain but does not record git SHA, env, seed, dataset hash |
| CI | CI runs `pytest` and `ruff`, but does not run any experiment-as-test |
| Config drift | Two config trees (`config/` and `configs/`) with overlapping fields used by different code paths |

---

## 5. Tests

16 test files. Coverage skews toward feature-engineering helpers and attention-fusion plumbing. Notable gaps:
- No test asserts that train/test splits are disjoint after `load_cyber_data`.
- No test asserts metric reproducibility across seeds (which would have caught the empty `std`).
- No integration test that checks the documented dataset is what the loader returns (would have caught the CERT/online-shoppers swap).

---

## 6. Concrete fix order (week-by-week)

**Week 1 — credibility floor.**
1. Cyber loader: switch to official `training-set.csv` / `testing-set.csv` and add a test asserting train ∩ test = ∅.
2. Fusion: replace `model.predict_proba(X)` with `cross_val_predict(method="predict_proba")` so meta-features are out-of-fold.
3. Either re-source CERT r4.2 or rename the behavior task in all docs to "Online Shoppers Intent."
4. Either implement DistilBERT fine-tuning or drop the DistilBERT claims from docs.
5. Delete `reports/metrics_*.csv` and add a CI check that fails if those files contain hand-edited values (e.g., require a hash from a real run).

**Week 2 — rigor.**
6. Switch fraud to a temporal split.
7. Vision: respect the predefined `Train/Test/Validation` folders.
8. Run ≥3 seeds (`{0, 1, 42}`) for each domain; report mean ± std and a bootstrap 95% CI for ROC-AUC.
9. Add a logistic-regression baseline per tabular domain to give the boosted models something to beat.
10. Tune the hybrid `alpha` on val; report sensitivity curve.

**Week 3 — repro & honesty.**
11. Enable MLflow in every config (`mlflow.enabled: true`) and log: git SHA, dataset hash, seed, full hyperparam dict, env (`pip freeze`).
12. Generate a `lockfile` (`pip-compile` or uv) and add it to CI.
13. Make synthetic fallbacks opt-in; raise loud warnings when active.
14. Deduplicate `cyber_baseline.yaml` / `behavior_baseline.yaml`; collapse `config/` and `configs/` into one tree.
15. Implement fusion only across rows that share a `sample_id` — or document fusion as a *framework demo* with no headline AUC.

**Week 4 — write-up.**
16. Regenerate `reports/metrics_*.csv` via `make_tables.py` from the multi-seed runs.
17. Update `UAISV_Final_Project_Summary.md` and `README.md` so dataset, model, and metric claims match what the code does.
18. Add an honest "limitations" section: single-machine eval, no production traffic, fusion alignment caveat.

---

## 7. What to keep claiming and what to drop

**Keep:** modular code organization, explainability hooks (SHAP / Grad-CAM), Streamlit + FastAPI surface, decent test scaffold, Prefect orchestration.

**Drop until fixed:** the specific ROC-AUC / F1 numbers in `reports/metrics_*.csv` and in `UAISV_Final_Project_Summary.md`; the "DistilBERT on Enron phishing" framing; the "CERT insider threat" framing; the "ResNet18/ViT" framing; the "stacked multimodal fusion" framing in the strong sense.

**Honest replacement framing:**
> "UAIS-V is a multimodal anomaly-detection scaffold spanning tabular (fraud, cyber), sequence (behavior intent), text (fake-news classification), and vision (document forgery). Each domain is trained independently with a gradient-boosting or anomaly baseline; a logistic-regression meta-model demonstrates how scores would be combined if a shared sample identifier were available. Headline metrics are reported as preliminary and produced under a single seed; multi-seed CIs and a CERT/Enron-based instantiation are future work."

That sentence is publishable. The current claims are not.
