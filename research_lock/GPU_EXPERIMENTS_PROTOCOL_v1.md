# GPU Experiments Protocol v1 (frozen)

Three evidence-expansion tracks to run on a CUDA box. Each reuses proven PatchCore
primitives (`src/uais/fusion/attention/`); only data loading is new. **Frozen rules:**
no test labels enter any routing/scoring step; multimodal hypotheses H1–H3 are
pre-registered (below); no retuning to flip a verdict. Results flow back into the
canonical build (`bash scripts/rebuild_paper.sh`) with no manual number entry.

Data already on the box (per user): Real-IAD-D3, MVTec-3D, MVTec AD, VisA.

---

## Track 1 — Expand D23 multimodal (Real-IAD-D3) — *highest value, lowest risk*

The D23 cache currently has 9–11 categories; **20** Real-IAD-D3 category zips are
available. This just runs the **already-proven** builder on all of them.

```bash
CATS=$(ls data/raw/realiad_d3/realiad_d3_raw/*.zip | grep -v '/\._' | xargs -n1 basename | sed 's/.zip//')
PYTHONPATH=src python src/scripts/scenario_c/run_realiad_d3_fusion_test_a_v2.py --categories $CATS   # builds cache (GPU)
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py             # D23 over the full cache
```
→ `experiments/elara_u/multimodal_reliability_results.json` (now ~20 categories,
tighter CIs). The canonical build + `statistical_audit.py` pick it up automatically.

**Pre-registered hypotheses (failure regime):** reliability gate beats (H1) equal-weight
fusion, (H2) stale auto-select, (H3) validation-only fusion — each paired-bootstrap CI
> 0. Decision rule unchanged from the 9-category result.

## Track 2 — Second multimodal dataset (MVTec-3D)

Corroborate D23 on an independent multimodal dataset (RGB + XYZ).

```bash
PYTHONPATH=src python src/scripts/elara_u/gpu_build_mvtec3d_cache.py                  # build cache (GPU)
PYTHONPATH=src python src/scripts/elara_u/multimodal_reliability_test.py \
    --cache experiments/fusion/mvtec3d_score_cache --glob '*.npz' --tag MVTec-3D \
    --out experiments/elara_u/multimodal_reliability_results_mvtec3d.json
```
A second PASS here makes the "reliability helps under independent modality failure"
claim dataset-independent.

## Track 3 — Industrial-vision single-input families (MVTec AD, VisA)

Add native industrial families to the 123-task benchmark via ResNet-50 embeddings
(same recipe as the existing image-OOD / ADBench-CV family).

```bash
mkdir -p data/raw/adbench_industrial
# MVTec AD (repeat per category):
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  PYTHONPATH=src python src/scripts/elara_u/gpu_build_image_embeddings.py \
    --normal "data/raw/mvtec_ad/$c/train/good/*.png" "data/raw/mvtec_ad/$c/test/good/*.png" \
    --anomaly "data/raw/mvtec_ad/$c/test/*/*.png" --exclude-anomaly-glob '*/good/*' \
    --out "data/raw/adbench_industrial/mvtecad_$c.npz"
done
# VisA (repeat per category):
for c in candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
  PYTHONPATH=src python src/scripts/elara_u/gpu_build_image_embeddings.py \
    --normal "data/raw/visa/$c/Data/Images/Normal/*.JPG" \
    --anomaly "data/raw/visa/$c/Data/Images/Anomaly/*.JPG" \
    --out "data/raw/adbench_industrial/visa_$c.npz"
done
```
Then register the new family in `gate_u_seed_eval.load_tasks` (add an
`("adbench_industrial", "industrial", "ind_")` entry to the feature-vector loop),
rebuild the score archive, and re-run the benchmark:
```bash
PYTHONPATH=src python src/scripts/elara_u/build_score_archive.py   # re-extract zoo scores (GPU not needed)
PYTHONPATH=src python src/scripts/elara_u/honest_benchmark.py      # new family enters rank/regret/CI
bash scripts/rebuild_paper.sh                                      # regenerate tables/figures
```

## Track 4 (optional) — OpenOOD (download required)

OpenOOD data is **not** on the box. Download CIFAR/ImageNet near/far-OOD, then use the
same `gpu_build_image_embeddings.py` (ID = normal, OOD = anomaly) to make
`data/raw/adbench_industrial/openood_*.npz`, and register as an `ood_native` family.

---

## Integration & honesty checklist (after any track)
- [ ] `PYTHONPATH=src python -m pytest tests/elara_u/` — no-leakage + schema + smoke pass.
- [ ] `statistical_audit.py` re-run; primary claims still Holm-robust; multimodal verdict updated.
- [ ] `emit_manifest.py` re-run (refresh sha256sums).
- [ ] No verdict changed by post-hoc tuning; new families reported even if they weaken a family-level CI.
- [ ] Commit the new caches/embeddings (small) so results are reproducible.
