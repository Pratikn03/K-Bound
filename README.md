# ELARA — Evidence-Layered Reliability for Anomaly Fusion

ELARA is a research prototype for **score-level multimodal anomaly fusion**
under domain-reliability stress. The core mechanism is **RGA**
(Reliability-Gated Attention), a masked cross-domain attention module
extended with a conservative reliability gate that decides at inference
time whether to keep the static attention path or inject reliability
weights derived from validation calibration, score-distribution drift,
and prediction sharpness.

> **Paper status:** preprint-track. The conference manuscript
> [docs/research/PAPER_DRAFT_v1.tex](docs/research/PAPER_DRAFT_v1.tex)
> and the companion thesis chapter
> [docs/research/THESIS_CHAPTER_v1.tex](docs/research/THESIS_CHAPTER_v1.tex)
> share the same evidence base and asset pipeline. The headline finding is
> a contrastive negative result: validation-derived KS-drift gates
> *help* on label-aligned coherent score-collapse attacks but *hurt* on
> naturally paired MVTec 3D-AD data because the drift signal mis-fires on
> legitimate inter-category variation. See the paper's §IX.D.5
> Cross-Benchmark Contrast for the full argument.

---

## What's in this repo

| Top-level | Purpose |
|---|---|
| `src/uais/` | Primary research package — `fusion/attention/`, `supervised/`, `anomaly/`, `sequence/`, `nlp/`, `vision/`, `explainability/`, `utils/` |
| `src/uais_v/` | 30-sequence behavior dataset builder + model definitions used by `tests/test_multi_sequence_30_*.py` |
| `src/scripts/` | Experiment runners (`run_breakthrough_experiment.py`, prep scripts, asset generators) |
| `configs/` | YAML configs for each fusion benchmark and baseline |
| `data/raw/` | Datasets (Credit Card Fraud, UNSW-NB15, Online Shoppers, news text, MVTec 3D-AD) |
| `experiments/fusion/` | Result JSONs and benchmark metadata that feed the paper |
| `docs/research/` | Manuscripts, table sources, figures, and the audit / review folder |
| `tests/` | 221 passing tests (3 skipped) |
| `deploy/api/` + `dashboard/` | FastAPI service and Streamlit dashboard for demo use |
| `scripts/rebuild_paper.sh` | One-command paper-and-thesis rebuild from current JSON artifacts |

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the regression suite
PYTHONPATH=src pytest tests/ -q

# 3. Smoke-test the fusion pipeline
PYTHONPATH=src python scripts/ci_smoke.py

# 4. Rebuild the paper + thesis from the current JSON artifacts
./scripts/rebuild_paper.sh
# → output/pdf/PAPER_DRAFT_v1.pdf
# → output/pdf/THESIS_CHAPTER_v1.pdf
```

## Local quality gates

Use the same commands before treating a research run or manuscript rebuild as
current:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -q
PYTHONPATH=src ./.venv/bin/python -m ruff check --select E9,F63,F7,F82 .
./scripts/rebuild_paper.sh
```

The benchmark harness selects classification thresholds from the validation
split (`evaluation.decision_threshold: val_f1`) and applies those fixed
thresholds to the held-out test split. Do not tune thresholds on test labels.

## Reproducing the headline experiments

```bash
# A. Build the naturally paired MVTec 3D-AD fusion inputs (requires
#    data/raw/mvtec3d/<category>/{train,validation,test}/...)
PYTHONPATH=src python src/scripts/prepare_mvtec3d_fusion_benchmark.py

# B. Build the label-aligned secondary benchmark (RealFusion)
PYTHONPATH=src python src/scripts/prepare_realfusion_la_benchmark.py

# C. Run the 5-seed breakthrough experiment on each benchmark
PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_mvtec3d_fusion.yaml \
  --output experiments/fusion/mvtec3d_results.json

PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_real_fusion.yaml \
  --output experiments/fusion/craf_real_results.json

# Optional hard-mode RealFusion rerun: first build inputs with
# --scorer-train-fraction 0.05, then run the hard config.
PYTHONPATH=src python src/scripts/prepare_real_fusion_benchmark.py \
  --scorer-train-fraction 0.05 \
  --output experiments/fusion/real_domain_fusion_hard_inputs.csv \
  --metadata experiments/fusion/real_domain_fusion_hard_metadata.json

PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_real_fusion_hard.yaml \
  --output experiments/fusion/craf_real_results_hard.json

# D. Regenerate the paper assets and recompile the PDFs
./scripts/rebuild_paper.sh
```

Both benchmark configs honor the underlying dataset's predefined train /
validation / test split via the `split_column` setting (MVTec uses
`split`; RealFusion uses `fusion_split`). The fusion train fold is
therefore disjoint from the rows the per-domain scorers fit on.

## Headline numbers (current, under disciplined splits)

**Naturally paired MVTec 3D-AD (3,226 paired samples, 8 categories, 22.4% positive):**

| Method | Clean ROC-AUC |
|---|---|
| Random forest | 0.959 ± 0.008 |
| Early fusion MLP | 0.828 ± 0.021 |
| Tent score adapter | 0.670 ± 0.019 |
| Static attention | 0.650 ± 0.038 |
| **RGA attention** | **0.610 ± 0.031** |
| TTT pseudo-label | 0.569 ± 0.039 |
| Conf.-weighted mean | 0.509 ± 0.019 |

RGA gate *hurts* by $-0.040$ clean and by $-0.117$ to $-0.124$ under
all-domain coherent score-collapse attacks. Component ablation shows the
KS-drift signal is the only component that controls whether the gate
fires; removing it returns the model to static-attention performance.

**Label-aligned secondary benchmark (RealFusion, 8,000 composite samples):**

The clean split is near-saturated. Under all-domain coherent attacks,
the same gate *improves* ROC-AUC by $+0.0506$ (zero attack) and
$+0.0319$ (max attack). Removing the ECE term further improves the
attack gain.

The asymmetry between the two benchmarks is the central scientific
finding of the paper.

## Layout caveats

- `mlflow_config.yaml`, `prefect` in `requirements.txt`, and the FastAPI /
  Docker / Streamlit surface are **demo-grade scaffolding**, not
  production tooling. MLflow is opt-in per config (`mlflow.enabled: true`)
  and is off by default. Prefect flow files under `src/orchestration/`
  are thin function wrappers, not real `@flow`-decorated tasks.
- `reports/metrics_*.csv` files are legacy dashboard placeholders, not
  research metrics — the research metrics live in `experiments/fusion/`
  and the rendered tables under `docs/research/tables/`.

## Status and roadmap

The research roadmap, gap analysis, and submission plan are tracked in
[docs/research/manuscript_review/](docs/research/manuscript_review/):

- [SENIOR_ENGINEER_AUDIT_2026-05-15.md](docs/research/manuscript_review/SENIOR_ENGINEER_AUDIT_2026-05-15.md) — full-repo audit
- [PUBLICATION_ROADMAP.md](docs/research/manuscript_review/PUBLICATION_ROADMAP.md) — arXiv → workshop → conference plan
- [REVIEWER_RATING_AND_PHASE_PLAN.md](docs/research/manuscript_review/REVIEWER_RATING_AND_PHASE_PLAN.md) — reviewer-style rating and remediation
- [FULL_RESEARCH_AUDIT_2026-05-14.md](docs/research/manuscript_review/FULL_RESEARCH_AUDIT_2026-05-14.md) — split-discipline and benchmark-construction audit

## License

MIT. See [LICENSE](LICENSE).

## Author

Pratik Niroula — independent researcher.
