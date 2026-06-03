# ELARA-U — Cross-Domain Anomaly Meta-Routing Benchmark

**ELARA-U** studies anomaly detection as a *meta-routing* problem: given a
heterogeneous detector zoo and a labelled validation slice, decide when to
**select, fuse, stack, or fall back** — without using test labels. It is a
benchmark + an honest result, not a new detector.

### The precise claim (verified)
> **Stacking beats selection; reliability routing helps *iff* modalities can fail independently.**

- On **123 tasks / 5 families** (tabular, image-OOD, text, cyber, fraud), a
  rank-normalized logistic **stack** beats validation auto-selection
  (**+0.036 AUROC**, CI [0.023, 0.050]; 86/123 wins) and the best fixed
  detector (**+0.075**, CI [0.052, 0.101]) — leakage-free, all **Holm-robust**.
- **Reliability/drift routing adds no value** on single-input data across
  4 deployment regimes + a sealed natural temporal-shift benchmark (D22), and a
  reliability gate significantly *hurts* the stack (Holm p < 10⁻⁸).
- **But it wins where it structurally should:** on real **Real-IAD-D3**
  multimodal data with independent modality failure (**D23**), drift-gated
  fusion recovers performance (0.517 → 0.715; H1 +0.221, H2 +0.334, H3 +0.198,
  all CIs exclude zero).

We do **not** claim universal/per-dataset SOTA. The contribution is a
cross-domain benchmark, a strong stacking baseline, a Holm-robust negative
result for reliability routing on single-input data, and a positive result that
pins its exact operating boundary.

> **Canonical paper + thesis:**
> [docs/research/ELARA_U_PAPER_v0.tex](docs/research/ELARA_U_PAPER_v0.tex) and
> [docs/research/ELARA_U_THESIS_CHAPTER.tex](docs/research/ELARA_U_THESIS_CHAPTER.tex).
> Rebuild both (regenerating all tables/figures from verified data) with
> `bash scripts/rebuild_paper.sh` → `output/pdf/`.

### Reproduce the evidence
```bash
PYTHONPATH=src python src/scripts/elara_u/honest_benchmark.py             # main 123-task result
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py  # D23 (Real-IAD-D3)
PYTHONPATH=src python src/scripts/elara_u/natural_shift_benchmark.py      # D22 sealed natural shift
PYTHONPATH=src python src/scripts/elara_u/statistical_audit.py            # Holm correction
PYTHONPATH=src python -m pytest tests/elara_u/                            # no-leakage + schema + smoke
bash scripts/rebuild_paper.sh                                            # paper + thesis PDFs
```
Curated results live in `experiments/elara_u/` (`honest_benchmark.json`,
`statistical_audit.json`, `multimodal_reliability_results.json`,
`natural_shift_results.json`, `calibration_results.json`, `manifest.json`,
`sha256sums.txt`); the Real-IAD-D3 score cache is in
`experiments/fusion/realiad_d3_score_cache/`.

---

### Legacy: ELARA / RGA source system
ELARA-U reuses the earlier **ELARA / RGA** (Reliability-Gated Attention) program
as *foundation and honest-negative evidence*, not as universal proof: the
in-domain MVTec-3D RGA+ win and the dead-modality stress mechanism are strong,
while the clean-transfer ties/failures (and Theorem T9) are exactly what motivate
the cross-domain routing framing. The RGA source code remains under
`src/uais/fusion/attention/`; its evidence is summarised in the paper's
source-system table. The legacy RGA manuscripts have been superseded by the
ELARA-U paper/thesis above.

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
| `tests/` | 240 passing tests (2 skipped) |
| `deploy/api/` + `dashboard/` | FastAPI service and Streamlit dashboard for demo use |
| `scripts/rebuild_paper.sh` | One-command paper-and-thesis rebuild from current JSON artifacts |

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For exact reproduction of headline numbers, prefer the pinned lockfile:
# pip install -r requirements.lock.txt

# 2. Run the regression suite
PYTHONPATH=src pytest tests/ -q

# 3. Smoke-test the fusion pipeline
PYTHONPATH=src python scripts/ci_smoke.py

# 4. Rebuild the paper + thesis from the current JSON artifacts
bash scripts/rebuild_paper.sh
# → output/pdf/PAPER_DRAFT_v1.pdf
# → output/pdf/THESIS_CHAPTER_v1.pdf
```

> **Note on exFAT volumes**: If the repo lives on an exFAT-formatted external
> drive (as on the author's setup), unix execute bits are pinned by the
> filesystem and `./scripts/X.sh` may fail. Invoke shell scripts via
> `bash scripts/X.sh` instead — this works on every filesystem.

## Data acquisition (~88 GB total)

```bash
# Acquire all raw datasets in one pass (verifies SHA256 anchors where available)
bash scripts/download_all_datasets.sh
bash scripts/download_all_datasets.sh --verify-only   # re-check anchors anytime
bash scripts/download_all_datasets.sh --only eyecandies,mvtec3d
```

## One union research system

The older standalone domain pipelines and the newer paper-grade research
pipeline are now joined by one orchestrator:

```bash
# Preview every step without running training.
PYTHONPATH=src .venv/bin/python src/scripts/run_union_research_system.py \
  --mode full \
  --with-tests \
  --dry-run

# Run the full union system: legacy fraud/cyber/behavior, paper fusion,
# healthcare audits, tables, figures, PDFs, and focused verification.
PYTHONPATH=src .venv/bin/python src/scripts/run_union_research_system.py \
  --mode full \
  --with-tests \
  --continue-on-error
```

The runner writes per-step logs and a machine-readable summary under
`experiments/union_research_system/`. By default it runs the legacy standalone
fraud/cyber/behavior experiments plus the current research evidence pipeline,
but keeps deprecated dashboard fusion and optional standalone NLP/vision
training opt-in:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_union_research_system.py \
  --mode full \
  --include-deprecated-dashboard \
  --include-optional-nlp-vision
```

This does not blur the evidence boundary: legacy-domain outputs are retained for
system continuity, while paper claims still come from the disciplined fusion
benchmarks, healthcare replay audits, and regenerated manuscript assets.

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

# D. Build and run the public MVTec PatchCore supervised-paired protocol
PYTHONPATH=src python src/scripts/prepare_mvtec3d_fusion_benchmark.py \
  --dataset-root data/raw/mvtec3d \
  --feature-mode patchcore \
  --embedding-dim 16 \
  --supervised-paired \
  --output experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv \
  --metadata experiments/fusion/mvtec3d_patchcore_supervised_paired_metadata.json

PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py \
  --config configs/attention_mvtec3d_patchcore_supervised_paired.yaml \
  --output experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json

# E. Regenerate the paper assets and recompile the PDFs
./scripts/rebuild_paper.sh
```

The benchmark configs honor disciplined train / validation / test boundaries via
the `split_column` setting: MVTec variants use `split`, and RealFusion uses
`fusion_split`. The fusion train fold is therefore disjoint from held-out
benchmark rows.

## Headline numbers (current, under disciplined splits)

**Naturally paired MVTec 3D-AD (3,226 paired samples, 8 categories, 22.4% positive):**

| Method | Clean ROC-AUC |
|---|---|
| **RGA attention** | **0.561 ± 0.017** |
| Early fusion MLP | 0.545 ± 0.025 |
| Static attention | 0.542 ± 0.025 |
| Random forest | 0.500 ± 0.000 |
| Late fusion / Tent / TTT | 0.500 ± 0.000 |
| Conf.-weighted mean | 0.446 ± 0.000 |

These numbers follow MVTec's canonical one-class protocol: train and
validation are normal-only, while test is mixed. That makes the supervised
fusion table a protocol diagnostic rather than a normal two-class leaderboard.
The repo also includes held-out-category, M3DM-style, PatchCore, and
PatchCore supervised-paired variants to probe this boundary. In the public
PatchCore supervised-paired protocol, RGA+ boosted fusion reaches ROC-AUC
`0.738`, the auxiliary RGA+ router reaches `0.740`, Tent reaches `0.735`, TTT
`0.724`, random forest `0.702`, static attention `0.632`, and base RGA
`0.628`. That result is included with the negative controls: canonical
PatchCore is still led by confidence-weighted mean, and held-out-category ROC
is only a marginal RGA+ win (`0.517` vs TTT `0.516`) with PR-AUC still favoring
TTT.

**Label-aligned stress-only secondary benchmark (RealFusion, 8,000 composite samples):**

The clean split is near-saturated. Under all-domain coherent attacks,
the same gate *improves* ROC-AUC by $+0.0506$ (zero attack) and
$+0.0319$ (max attack). Removing the ECE term further improves the
attack gain.

The paper's central claim is now sharper: reliability gating is diagnostically
useful for coherent score-collapse stress when two-class fusion training exists,
but this does not transfer into a general SOTA claim on naturally paired
one-class anomaly protocols.

## Engineering gap-closure utilities

The repo now includes local engineering prerequisites for the four enterprise
readiness gaps, without claiming the external evidence is complete:

- `validate_incident_protocol` checks shared incident IDs, temporal split order,
  incident-level split isolation, and multi-domain coverage before a dataset is
  treated as naturally co-observed.
- `CategoryAwareReliabilityEstimator` uses category-conditional drift references
  so legitimate category-mix changes do not automatically trigger the global KS
  gate.
- The long-format fusion schema remains the plug-in boundary for stronger
  domain experts, including future RGB-3D anomaly scorers.
- `calibration_monitor_report` and `bounded_switching_certificate` provide
  deployment calibration alerts and the finite-sample switching condition for
  preferring the reliability path.

After copying the local GridPulse healthcare data, the clinical audit can be
run with:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_healthcare_gap_closure.py \
  --data-root data/raw/healthcare/gridpulse
```

The generated `experiments/fusion/healthcare_gap_validation.json` is a
Retrospective local healthcare replay, not clinical deployment evidence. It
projects 146,688 co-observed vital-sign incidents into four fusion domains
(heart rate, oxygenation, respiration, shock index) with hashed patient and
incident identifiers. The structural checks pass: four domains per incident, no
incident split leakage, no patient overlap, temporal ordering preserved, and no
score/confidence range violations. The provided validation and test windows are
single-class positive, so this time-forward surface remains a temporal-reference
audit rather than supervised clinical-performance evidence.

For Gap 1 specifically, the stricter time-forward replay cannot close the
supervised detection claim because its validation and test windows are
single-class positive. A second audit therefore uses:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_healthcare_gap_closure.py \
  --data-root data/raw/healthcare/gridpulse \
  --split-strategy patient_stratified \
  --report experiments/fusion/healthcare_gap1_patient_stratified_validation.json \
  --fusion-output experiments/fusion/healthcare_gap1_patient_stratified_fusion_inputs.csv
```

This patient-disjoint stratified replay closes Gap 1 locally: every split has
both labels, patient overlap is zero, each incident has all four domains, and
the held-out multimodal score reaches ROC-AUC 0.806 versus 0.770 for the best
single domain. The tradeoff is explicit: `temporal_order_valid is false`, so
this is local two-class incident-detection evidence, not time-forward clinical
deployment evidence.

The same replay now closes Gap 2 locally through a reliability stress audit:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_healthcare_gap_closure.py \
  --data-root data/raw/healthcare/gridpulse \
  --split-strategy patient_stratified \
  --report experiments/fusion/healthcare_gap2_reliability_stress_validation.json \
  --fusion-output experiments/fusion/healthcare_gap2_reliability_stress_fusion_inputs.csv
```

The audit calibrates a conservative category-aware gate threshold from the
validation natural replay, then evaluates held-out natural replay and injected
score-collapse episodes. Gap 2 locally closes because the held-out natural fire
rate is 0.0 while the mean collapse fire rate is 1.0 across the four domains.

Gap 3 and Gap 4 are now closed locally on the same patient-disjoint replay:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_healthcare_gap_closure.py \
  --data-root data/raw/healthcare/gridpulse \
  --split-strategy patient_stratified \
  --report experiments/fusion/healthcare_gap4_deployment_audit_validation.json \
  --no-fusion-output

PYTHONPATH=src .venv/bin/python src/scripts/generate_healthcare_gap_assets.py \
  --report experiments/fusion/healthcare_gap4_deployment_audit_validation.json
```

Gap 3 closes through the schema-integration report: 586,752 fusion rows,
four complete domains per incident, a `146688 x 4 x 4` tensor, no raw
`patient_id` column, hashed patient keys, and no missing schema columns. Gap 4
closes as a local deployment-replay audit: the provided split remains the
time-forward temporal reference, manifest review records PhysioNet access and
citation requirements, calibration monitoring is active, leave-one-domain-out
CDA-style attribution is emitted, and the diagnostic switching certificate is
true (`static_loss=0.800`, `policy_loss=0.000`). This is still local replay
readiness, not prospective clinical deployment or regulated clinical use.

## Layout caveats

- `mlflow_config.yaml`, `prefect` in `requirements.txt`, and the FastAPI /
  Docker / Streamlit surface are **demo-grade scaffolding**, not
  production tooling. MLflow is opt-in per config (`mlflow.enabled: true`)
  and is off by default. Prefect flow files under `src/orchestration/`
  are thin function wrappers, not real `@flow`-decorated tasks.
- `reports/metrics_*.csv` files are legacy dashboard placeholders, not
  research metrics — the research metrics live in `experiments/fusion/`
  and the rendered tables under `docs/research/tables/`.

## Status and roadmap (Scenario C / Flagship)

Governance and gates:

- [research_lock/SCENARIO_C_CLAIM_CONTRACT.md](research_lock/SCENARIO_C_CLAIM_CONTRACT.md) — claim contract
- [elara_master_c/audits/FINAL_CHECKLIST_VERDICT.md](elara_master_c/audits/FINAL_CHECKLIST_VERDICT.md) — checklist verdict
- [docs/research/scenario_c/WIN_VS_SAR_VALIDATION.md](docs/research/scenario_c/WIN_VS_SAR_VALIDATION.md) — Flagship / ELARA deploy validation
- [docs/research/scenario_c/MASTER_C_TRAINING_EXECUTION.md](docs/research/scenario_c/MASTER_C_TRAINING_EXECUTION.md) — Master C runbook

## License

MIT. See [LICENSE](LICENSE).

## Author

Pratik Niroula — independent researcher.
