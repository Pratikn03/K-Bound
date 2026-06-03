# Multimodal Reliability-Gating Protocol v1 (pre-registered)

The one honest shot at a **novel-method** claim for ELARA-U. The 123-task
single-modality benchmark closed the reliability-routing premise *there*
(`honest_benchmark.json` + three ablations); the synthetic PoC
(`synthetic_multimodal_poc.json`) showed **why**: reliability gating helps **iff a
channel can fail independently while others stay clean**. RGB and 3D fail
independently — single-modality detectors do not. This protocol tests that on real
multimodal data. Runs on a GPU box (not in the dev session).

## Pre-registered hypothesis (frozen before looking at real results)

Under **independent per-modality** deployment degradation (one modality degrades at
test; the other stays clean; validation stays clean), reliability-gated fusion will:

- **H1** beat equal-weight fusion (`mean`) — paired-bootstrap CI > 0
- **H2** beat stale validation-AUROC selection (`val_select`) — CI > 0
- **H3** beat its own no-test-time-reliability ablation (`rel_gate_abl`) — CI > 0
  (i.e. the *test-time* reliability signal, not validation quality, carries the gain)

**Decision rule.** All of H1–H3 significant → POSITIVE, a real scoped novel claim
(reliability gating recovers anomaly detection under per-modality deployment failure).
Any of H1–H3 fails → NEGATIVE, the reliability premise is closed even where it
structurally should hold, and we ship the honest single-modality measurement paper
without a novel-method claim. **No metric, dataset, or degradation will be changed
after seeing results to flip a verdict.**

## Data

- **MVTec-3D-AD** (10 categories, RGB + 3D point cloud / depth): https://www.mvtec.com/company/research/datasets/mvtec-3d-ad
- **Real-IAD D3** (RGB + pseudo-3D + depth; 30+ categories — preferred for power): https://huggingface.co/datasets/Real-IAD-D3/Real-IAD
- Put under `data/raw/mvtec3d_ad/<category>/...` or `data/raw/real_iad_d3/...`.

## Environment

```bash
# GPU box, Python 3.10+:
pip install torch torchvision pyod scikit-learn scipy pillow numpy
# (a CUDA-enabled torch; the harness checks torch.cuda.is_available())
```

## Detector design (important for power)

Use **≥2 detectors per modality** so a modality failure removes a *block* of channels
while the other modality's block stays clean (this is what the smoke positive control
uses, and what gives reliability gating something to recover):

- RGB block: PatchCore on a vision backbone (WideResNet50 / DINO) + a second cheap
  RGB detector (e.g. color-histogram KNN).
- 3D block: PatchCore on depth-rendered geometry images + point-cloud covariance
  (`score_one_class_point_cloud`).

Reuse the existing implementations:
`uais.fusion.attention.realiad_3d_detector.{score_one_class_patchcore,
score_one_class_point_cloud, load_modality_image, pcd_to_geometry_image}` and
`uais.fusion.attention.cross_modal_patchcore`.

## Protocol (per category)

1. Build each detector's memory bank on the **normal-only train** split.
2. Score a **labelled validation** split (clean) → `Sval[n_val, M]`, `val_auc`.
3. Score the **test** split; then apply **independent per-modality degradation to the
   test inputs of ONE modality only** (validation stays clean), using
   `degrade_image()` (RGB: `blur`/`noise`/`bright`) or `degrade_depth()`
   (3D: `dropout`/`noise`/`quant`). Re-score the degraded modality. → `Stest[n_test, M]`.
4. `evaluate_fusion(Sval, yval, Stest, ytest)` → per-category fusion AUROCs.
5. Repeat over **≥5 degradation seeds × all categories**; the statistical unit for the
   contract bootstrap is **(category × seed)** (10 categories alone are too few —
   the smoke control demonstrates this).
6. `run_contract(per_unit)` → H1–H3 with paired-bootstrap CIs → verdict.

**No test labels** are used by any fusion rule (`mean`, `val_select`, `rel_gate`,
`rel_gate_abl`). `rel_gate` uses validation quality × *unlabeled* test-time
consensus-agreement; the val→test drift / disagreement are deployment-monitoring
signals, not labels.

## Running

```bash
# 1. logic check (CPU, no data) — must PASS (positive control):
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_experiment.py --smoke

# 2. real run (GPU): wire score_real() to your data loader (the only NotImplemented
#    piece — feature extraction + memory banks; fusion/contract are already done), then:
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_experiment.py \
    --data-root data/raw/mvtec3d_ad --categories bagel cable_gland carrot cookie dowel \
    foam peach potato rope tire --degrade-modality depth --degrade-kind dropout --degrade-severity 0.5
```

Output: `experiments/elara_u/multimodal_reliability_result.json` with the H1–H3
contrasts and the PASS/FAIL verdict.

## What each outcome means for the paper

- **PASS** → the honest paper gains a real, scoped novel-method section: *"reliability
  gating recovers multimodal anomaly detection under independent per-modality
  deployment failure (H1–H3, CI>0), the regime single-modality detection cannot
  exhibit."* That is the flagship-track contribution.
- **FAIL** → report it honestly in the paper as the final boundary: reliability gating
  does not help even on multimodal data; the contribution is the strong stacking
  baseline + the rigorous negative + the mechanistic characterization. No novel-method
  claim. **Do not retune to chase a pass.**
