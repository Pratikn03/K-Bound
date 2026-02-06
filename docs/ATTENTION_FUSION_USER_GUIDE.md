# Attention Fusion User Guide

This guide covers training, evaluation, and visualization for Phase 2 attention fusion.

## 1) Data requirements

Use the long-format fusion schema in `docs/research/data/FUSION_SCHEMA.md`.

Minimum columns:
- `sample_id`
- `domain`
- `score`

Optional columns:
- `confidence`
- `embedding_*`
- `label`
- `timestamp`

## 2) Generate fusion inputs

Option A: Generate from existing domain experiments:
```bash
python3 -m scripts.run_fusion_experiment
```
This writes `experiments/fusion/attention_fusion/fusion_inputs.csv`.

Option B: Supply your own CSV/Parquet and set `data.path` in
`src/uais/fusion/attention/attention_config.yaml`.

## 3) Train attention fusion

```bash
python3 -m uais.fusion.attention.train_attention_fusion
```

The checkpoint defaults to `models/fusion/attention/attention_fusion.pt`.

## 4) Run evaluation harness

```bash
python3 -m uais.fusion.attention.evaluate_attention_harness
```

Outputs `experiments/fusion/attention_fusion/harness_metrics.json` and optional
attention weights under `experiments/fusion/attention_fusion/attention_weights/`.

## 5) Generate plots

```bash
python3 -m scripts.generate_attention_reports
```

Plots are saved to `experiments/fusion/attention_fusion/plots/`.

## 6) Streamlit dashboard

```bash
streamlit run dashboard/app_streamlit.py
```

Look under the Fusion tab for attention plots and harness summaries.

## 7) Configuration highlights

Edit `src/uais/fusion/attention/attention_config.yaml`:
- `model.use_input_confidence`: include confidence in inputs
- `model.use_positional_embeddings`: positional/domain embedding toggles
- `training.domain_dropout`: robustness to missing domains
- `evaluation.seeds`, `evaluation.n_bootstrap`: evaluation settings
