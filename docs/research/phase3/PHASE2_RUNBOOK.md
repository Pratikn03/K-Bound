# Phase 2 Runbook — ELARA Three Ceiling Experiments (L3.1 / L3.2 / OpenOOD+MVTec-AD-2)

**Status:** scripts written and dry-run-validated; **all heavy runs are queued for after the
RxRx1 K-Bound job finishes.** Nothing here has been run at scale.

**Hard rules while RxRx1 is live:** do **not** run any GPU/MPS step (Item 1 build, Item 3
feature extraction). Item 2 is CPU-only and may run anytime. Every script is **additive**
(new files only); none modifies an existing result JSON, the score archive, the manifest,
or the paper. The natural-degradation builder derives **real** image-space artifacts and
**fails loudly** (`exit 2`, `DATA NEEDED`) if raw data is absent — it never fabricates.

Scripts (all under `src/scripts/elara_u/`):
`build_natdeg_cache_multimodal.py` · `oneclass_fusion_eval.py` · `ingest_openood.py` ·
`ingest_mvtec_ad_2.py`. Run everything with `PYTHONPATH=src` from the repo root.

| Step | Needs GPU free? | Needs download? | Raw present today? |
|---|---|---|---|
| Item 2 (one-class) | no (CPU) | no | yes (`experiments/fusion/mvtec3d_score_cache`) |
| Item 1 (natural degradation) | **yes** (PatchCore rescore) | no | yes (`data/raw/{mvtec3d,3d_adam_anomalib}`) |
| Item 3 (OpenOOD / MVTec AD 2) | **yes** (feature extraction) | **yes** (staged) | **no** — stage first |

**Recommended order:** Item 2 first (CPU, immediate), then — once RxRx1 is done — Item 1,
then Item 3.

---

## Item 2 — MVTec 3D-AD one-class fusion (L3.1) · CPU · ~minutes

Leaderboard-comparable one-class image-AUROC from the existing per-category caches.

```bash
PYTHONPATH=src python src/scripts/elara_u/oneclass_fusion_eval.py --dry-run    # one category, sanity
PYTHONPATH=src python src/scripts/elara_u/oneclass_fusion_eval.py              # full ~8–10 categories
```
Produces (both new): `experiments/elara_u/oneclass_fusion_results.json` and
`docs/research/tables/mvtec3d_oneclass_demarcation.tex` (ELARA gated-CW / CW / RGB-only /
depth-only rows next to M3DM 0.945 / AST 0.937 / PatchCore-3D 0.901, marked
protocol-comparable, different method). Honest expectation: gated-CW lands **below**
M3DM/AST (no cross-modal patch interaction). Dry-run on `bagel` already gives
gate=CW=0.837 (gate correctly defaults to CW when no differential drift). Numbers are on
each category's **capped cache pool, single seed** — average over several `--seed` values
before quoting; it is an approximate leaderboard position, not the full canonical split.

---

## Item 1 — Natural modality degradation (L3.2) · GPU · ~1–3 h/dataset

Builds **real**-artifact degradation caches and runs them through the `--natural` D23
runner (no synthetic score injection). Raw present at `data/raw/mvtec3d` (8 cats) and
`data/raw/3d_adam_anomalib`. **Run only after RxRx1 frees the GPU.**

Artifacts are **deterministic and derived from the real data** (no PRNG, no score-space
noise): `missing_returns` removes the most dropout-prone real returns — grazing-angle /
steep geometry (high local gradient) and valid pixels bordering the sensor's own existing
no-return holes; `quantization` = coarser bit-depth (information loss); `low_illumination`
= exposure scale+gamma roll-off. Each is applied to the modality **images** of the TEST
split only and re-scored through the same one-class PatchCore detector, so the score drop
is a genuine consequence of degraded inputs.

Validate first (CPU, no scoring):
```bash
PYTHONPATH=src python src/scripts/elara_u/build_natdeg_cache_multimodal.py \
    --dataset mvtec3d --degradation missing_returns --dry-run
```
Build + evaluate (per dataset × artifact; `geom`=depth/point-cloud, `rgb`=colour):
```bash
# 1) build the real-degradation cache (GPU: PatchCore rescore of the degraded modality on TEST only)
PYTHONPATH=src python src/scripts/elara_u/build_natdeg_cache_multimodal.py \
    --dataset mvtec3d --degradation missing_returns \
    --cache experiments/fusion/mvtec3d_natdeg_missing_returns_score_cache
# 2) score the gate under genuine degradation (CPU; --natural = test as-is, no injection)
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py --natural \
    --cache experiments/fusion/mvtec3d_natdeg_missing_returns_score_cache --glob '*.npz' \
    --tag MVTec-3D-NatDeg-missing_returns \
    --out experiments/elara_u/multimodal_reliability_results_mvtec3d_natdeg_missing.json
```
Repeat with `--degradation {quantization,low_illumination}` and `--dataset 3d_adam`
(low_illumination targets `rgb`; the others target `geom`; tune `--severity` in `[0,1]`,
default 0.5). Each `multimodal_reliability_results_*natdeg*.json` is the natural-degradation
analogue of the Real-IAD-D3 D29 negative — report ties/losses honestly. Outputs are new
files; nothing existing is overwritten.

---

## Item 3 — OpenOOD + MVTec AD 2 ingestion (Gate U families) · download + GPU

Converts each dataset into ADBench-style `{X,y}` feature tasks (ResNet-18 512-d) that the
Gate U / `honest_benchmark` pipeline consumes, then (optionally) emits **additive**
`score_archive` npz so `honest_benchmark.py` (which globs `*.npz`) picks them up.
Note: `gate_u_seed_eval.py` scans a **fixed** set of raw subdirs and will not auto-discover
the new `data/raw/{openood,mvtec_ad_2}_tasks` dirs — the wired route into the pipeline is
`--to-archive` → `honest_benchmark`; to feed `gate_u_seed_eval` instead, point it at the
new dir or copy the npz into a scanned dir.

### 3a. Stage downloads (do after freeing the drive)
Drive is ~93% full (**≈136–145 GB free** measured) with the RxRx1 job writing. The CIFAR
OpenOOD suite (~12 GB) and MVTec AD 2 (~20 GB) each **do** pass the scripts' preflight
(`free ≥ size×1.5`), but the scripts **stage rather than auto-download by design**, and the
downloads were **intentionally deferred** here: do not add tens of GB of disk I/O while the
training job is writing checkpoints to this tight drive. Run them yourself when the job is
idle. Sizes:

- **OpenOOD** — CIFAR OOD suite ≈ 5–12 GB (fits); full ImageNet OOD ≈ **150+ GB (will NOT
  fit the current drive — do not pull it)**. Use the CIFAR/default suite only.
  ```bash
  git clone https://github.com/Jingkang50/OpenOOD && cd OpenOOD
  python ./scripts/download/download.py --contents images --datasets default \
      --save_dir <repo>/data/raw/openood        # expects <raw>/id/test/* and <raw>/ood/<set>/*
  ```
- **MVTec AD 2** — ≈ **20 GB**, license-gated. Register + download from
  <https://www.mvtec.com/company/research/datasets/mvtec-ad-2>, unpack to
  `data/raw/mvtec_ad_2/<category>/{train/good, test*/{good,<defects>}}/`.

### 3b. Preflight + ingest (GPU recommended after RxRx1)
```bash
PYTHONPATH=src python src/scripts/elara_u/ingest_openood.py --dry-run       # disk + presence only
PYTHONPATH=src python src/scripts/elara_u/ingest_openood.py   --device mps --to-archive
PYTHONPATH=src python src/scripts/elara_u/ingest_mvtec_ad_2.py --device mps --to-archive
```
Feature extraction is GPU-heavy (lazy torch; pass `--device mps`/`cuda` only once RxRx1 is
done — `cpu` works but is slow). `--to-archive` writes new `score_archive/openood_*.npz` /
`mvtecad2_*.npz` only (skips any that exist; never edits the manifest).

### 3c. Refresh the benchmark (deliberate, after RxRx1)
```bash
PYTHONPATH=src python src/scripts/elara_u/honest_benchmark.py   # globs the archive -> 123 + N tasks
```
**Caution:** `honest_benchmark.py` **rewrites** `experiments/elara_u/honest_benchmark.json`
(currently hash-locked in `manifest.json`). Commit/lock the current archive first, then run
this to fold the new families in and refresh `manifest.json` with `emit_manifest.py`.

---

## Checklist
- [ ] RxRx1 finished / GPU free (Items 1 and 3b only).
- [ ] **Item 2** dry-run → full → `oneclass_fusion_results.json` + demarcation `.tex`. (CPU, do first.)
- [ ] **Item 1** dry-run per dataset → build (GPU) → `--natural` score, all artifacts × {mvtec3d, 3d_adam}.
- [ ] **Item 3** free disk → stage OpenOOD (CIFAR suite) + MVTec AD 2 → ingest (GPU) `--to-archive`.
- [ ] (Optional) lock current archive, then `honest_benchmark.py` + `emit_manifest.py` to refresh the 123+N benchmark.
- [ ] Report every win/tie/loss honestly; natural-degradation results are expected to be weaker than injected.
