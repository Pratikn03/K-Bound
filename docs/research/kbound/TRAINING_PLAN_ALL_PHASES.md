# K-Bound — Dataset Analysis & Phase 1–4 Training Plan
_generated 2026-06-07 · run all GPU jobs on the Mac (MPS), one at a time · Claude runs CPU/theory + aggregation_

## A. Dataset inventory (what's on disk)

| Dataset | Status | Usable now? | Notes |
|---|---|---|---|
| **CIFAR-10-C** | ✅ complete | yes | 19 corruptions + labels (20 `.npy`), `resnet18_cifar.pt` present |
| **CIFAR-10 clean** | ✅ complete | yes | `cifar-10-batches-py` (for online/non-stationary stream) |
| **CIFAR-100-C** | ❌ absent | no | optional breadth; not downloaded |
| **ImageNet-C — noise** | ✅ complete | yes | gaussian/shot/impulse, all 5 severities × 1000 classes, `.done_noise` |
| **ImageNet-C — blur** | ⏳ extracting | partial | defocus in progress; glass/motion/zoom pending |
| **ImageNet-C — weather/digital/extra** | ❌ absent | no | optional; not downloaded |
| **Camelyon17 (WILDS)** | ⚠️ ~90% | yes (robust loader) | 46 node-folders; missing/corrupt patches auto-skipped; archive 9.1 G kept |
| **Anomaly / pipeline data** (`data/`) | ✅ used already | n/a | feeds knowability/harmful/mixed experiments — results already computed |

## B. Results that already exist (no re-run needed)
15 result JSONs are already on disk: `decisive_tta_results`, `cifar10c_suite_results`, `cifar_tent_online_results`, `cifar_tent_results`, `knowability_results`, `kbound_harmful_results`, `mixed_regime_results`, `regression_covariate`, `rigor_multiseed`, `ablations`, `breadth_existing_datasets`, `witness_clean`, `tta_collapse_results`, `kboundopt_results`, `decisive_tta_cis`.

➡️ **The only genuinely NEW training is the deep benchmarks: Camelyon17 + ImageNet-C (noise, then blur).** Everything else is either done or re-runnable for freshness.

## C. The four phases

### Phase 1 — Theory validation (CPU; Claude runs this in-sandbox)
Seven validators under `experiments/kbound/theory_validation/`: `val_thm1_lecam`, `val_thm2_regret`, `val_thm3_evalue`, `val_thm5_multiclass`, `val_thm9prime_drift`, `val_conj1_caltransfer`, `val_rademacher_router`.
Status: Thm 1 (Le Cam), Thm 2 (regret identity), Thm 5 (multiclass), Conj 1 (calibration transfer) **confirmed passing in-sandbox**; the other three validated previously. **No Mac time needed.**

### Phase 2 — CIFAR-10-C core empirics (MPS, Mac)
Already have results from Jun 5; re-run only if you want them fresh on this machine.
```
cd /Volumes/T9/uav/AutoML_Flagship_V8
source ~/.venv_wilds/bin/activate
export TMPDIR=/Volumes/T9/uav/tmp; mkdir -p "$TMPDIR"
python docs/research/kbound/scripts/cifar_tent_mps_v2.py --benchmarks cifar10c --methods tent eata sar
```

### Phase 3 — New deep benchmarks (MPS, Mac — ONE AT A TIME)
**3a. Camelyon17** (natural hospital shift — running now via robust loader):
```
export TMPDIR=/Volumes/T9/uav/tmp;            mkdir -p "$TMPDIR"
export TORCH_HOME=/Volumes/T9/uav/torch_cache; mkdir -p "$TORCH_HOME"
python docs/research/kbound/scripts/run_wilds_camelyon17.py \
  --wilds-root experiments/kbound/data/wilds \
  --output-dir experiments/kbound/results/wilds \
  --seeds 0 --epochs 2 --steps 10 --lr 1e-3
```
**3b. ImageNet-C noise** (ResNet-50, data complete — run after 3a finishes):
```
python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
  --benchmarks imagenetc --imagenetc-root experiments/kbound/data/imagenet-c \
  --corruptions gaussian_noise shot_noise impulse_noise \
  --arch resnet50 --methods tent eata sar \
  --out-results experiments/kbound/results/imagenetc_noise
```
**3c. ImageNet-C noise+blur** (after blur extraction finishes — `.done_blur` appears):
```
python docs/research/kbound/scripts/cifar_tent_mps_v2.py \
  --benchmarks imagenetc --imagenetc-root experiments/kbound/data/imagenet-c \
  --corruptions gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur \
  --arch resnet50 --methods tent eata sar \
  --out-results experiments/kbound/results/imagenetc_noiseblur
```

### Phase 4 — Aggregate + paper (Claude, after results land)
Collect every JSON → tables + figures → fold into `kbound.tex` → rebuild PDF:
`02_verify_results.py`, `03_make_tables.py`, `04_make_figures.py`, then the LaTeX build. Camelyon's ~90%-subset footnote included.

## D. Rules that keep it from breaking
1. **One MPS job at a time** — never run two GPU scripts together (they fight for Metal + internal-SSD temp).
2. Always `export TMPDIR=/Volumes/T9/uav/tmp` (and `TORCH_HOME=/Volumes/T9/uav/torch_cache`) so the full internal SSD isn't the bottleneck.
3. Data loaders run `num_workers=0` (exFAT can't do worker shared-memory).
4. Let `blur` keep extracting in its own terminal — that's CPU/disk, fine alongside one MPS job.

## E. Recommended order tonight
1. (now) Camelyon17 finishing → paste result.
2. ImageNet-C **noise** run → paste result.
3. Claude runs Phase 1 theory + Phase 4 aggregation, folds Camelyon + noise into the PDF.
4. Tomorrow, once blur is extracted: ImageNet-C **noise+blur** run → fold in.
