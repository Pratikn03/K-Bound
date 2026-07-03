# Running the two remaining K-Bound GPU experiments on your Mac

These two experiments are the only open **NEEDS-GPU** gaps. Everything downstream of
model inference (per-condition serialization, the single-candidate KGA decision rule, and
the multi-seed paired-bootstrap CIs) is torch-free and has already been **verified on CPU**
with a synthetic harness (see *Pre-flight* below). What is left genuinely needs your Apple
GPU + local datasets.

| Gap | What | Output run dir |
|-----|------|----------------|
| **B4** | ImageNet-R multi-seed (>=3 seeds), 48-condition diverse-backbone grid, OOM-hardened, per-condition arrays serialized | `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/` |
| **B1** | Camelyon17 SAR completion (candidate set already includes `{tent,eata,sar} x {online,episodic}`); finishes the `B_v2` sweep + serializes per-condition arrays incl. SAR | `experiments/kbound/results/camelyon17_fullscale_B_v2/` |

No committed/canonical artifact is overwritten. B4 writes a **new** dir; B1 **resumes** the
existing non-canonical `camelyon17_fullscale_B_v2/` checkpoint (currently 27/90 cells).

---

## 0. Pre-flight (already done; reproduce in ~5 s if you like)

Proves the serialization + SAR plumbing + paired-CI logic is correct *before* spending GPU
time. Torch-free, sklearn-free, synthetic scores (every field stamped `_synthetic_smoke`):

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds" \
  python3 experiments/kbound/theory_validation/verify_runner_pipeline.py --smoke
```

Expect `ALL_ASSERTIONS_PASSED = True` and, for both datasets, `cells=9/9 seeds=[0,1,2]`,
`SAR column present : True`, `paired CIs computable for 6 comparisons`. Artifacts land in
`experiments/kbound/results/_pipeline_smoke_verify/` (synthetic; safe to delete).

---

## 1. One command (recommended)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
bash scripts/run_remaining_gpu_experiments.sh
```

Defaults: `DEVICE=auto` (-> MPS on Apple silicon), `SEEDS="0 1 2"`, `NBOOT=10000`. It runs
**both** experiments and then runs the multi-seed paired-CI analysis on each.

Common overrides:

```bash
# stronger statistics (5 seeds), explicit device
SEEDS="0 1 2 3 4" DEVICE=mps bash scripts/run_remaining_gpu_experiments.sh

# only one of the two
ONLY=imagenetr bash scripts/run_remaining_gpu_experiments.sh
ONLY=camelyon  bash scripts/run_remaining_gpu_experiments.sh

# if MPS runs out of memory on ImageNet-R, shrink the frozen-eval batch
INR_BATCH=12 ONLY=imagenetr bash scripts/run_remaining_gpu_experiments.sh

# point at your own dataset copies
IMAGENETR_DIR="$HOME/kbound_inr/imagenet-r" \
WILDS_DATA_ROOT="$HOME/datasets/wilds" \
  bash scripts/run_remaining_gpu_experiments.sh
```

**Environment the script expects**
- `PY` (default `<repo>/.venv/bin/python`): a python with `torch`. For **Camelyon17** it
  also needs `wilds`; the script auto-uses `~/.venv_wilds/bin/python` if present, else falls
  back to `PY` and skips Camelyon17 with a clear message if `wilds` is missing.
- ImageNet-R images at `IMAGENETR_DIR` (default `experiments/kbound/data/imagenet-r`, which
  is present in this repo). Class index at `experiments/kbound/data/imagenet_class_index.json`
  (present).
- Camelyon17 WILDS data under `WILDS_DATA_ROOT` and the f0 checkpoints at
  `experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{0..4}.pt` (present).

---

## 2. Exact equivalent manual commands

If you prefer to run each step yourself (the script runs exactly these):

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
export PYTHONPATH="$PWD:$PWD/src:$PWD/experiments/kbound/wilds"
PY="$PWD/.venv/bin/python"          # torch
WPY="$HOME/.venv_wilds/bin/python"  # torch + wilds (Camelyon17)
```

### B4 — ImageNet-R multi-seed (MPS)
```bash
OUT="experiments/kbound/results/imagenetr_protocol_d_multiseed_v1"
caffeinate -is "$PY" -u experiments/kbound/wilds/run_imagenetr_kbound.py \
  --panel diverse_backbones \
  --imagenetr-dir experiments/kbound/data/imagenet-r \
  --seeds 0 1 2 \
  --compositions iid imbalanced single_class \
  --batch-regimes small tiny \
  --aggressiveness mild aggressive \
  --n-eval 500 --n-batches 4 \
  --frozen-eval-batch 24 \
  --device mps --resume --serialize-per-condition \
  --run-name imagenetr_protocol_d_multiseed_v1

# multi-seed paired-CI analysis (each frozen backbone is one column)
"$PY" experiments/kbound/wilds/multiseed_paired_ci.py \
  --run-dir "$OUT" --dataset imagenet-r \
  --methods resnet101 resnet152 resnext101_32x8d efficientnet_b0 efficientnet_b3 \
            convnext_tiny convnext_base vit_b_16 swin_t swin_b \
  --seeds 0 1 2 --nboot 10000
```

### B1 — Camelyon17 SAR completion (MPS, resumes B_v2)
```bash
OUT="experiments/kbound/results/camelyon17_fullscale_B_v2"
caffeinate -is "$WPY" -u experiments/kbound/wilds/run_camelyon17_kbound.py \
  --data-root "$HOME/datasets/wilds" \
  --f0-template experiments/kbound/results/camelyon17_fullscale_B_v1/f0_seed{seed}.pt \
  --seeds 0 1 2 \
  --domains test val id_val \
  --compositions iid imbalanced single_class \
  --batch-regimes small \
  --aggressiveness mild aggressive \
  --n-eval 1024 --n-batches 4 \
  --tau-star 0.52 --kappa 2.5 --delta 0.05 --sd-L 0.6 \
  --evidence-panel base \
  --device mps --resume --serialize-per-condition \
  --run-name camelyon17_fullscale_B_v2

"$WPY" experiments/kbound/wilds/multiseed_paired_ci.py \
  --run-dir "$OUT" --dataset camelyon17 \
  --methods tent eata sar --seeds 0 1 2 --nboot 10000
```

> Use the same `--seeds` in the runner and the analysis. `--resume` means a crash (OOM,
> sleep, Ctrl-C) can be restarted with the identical command; completed cells are skipped
> and the per-seed RNG stays in lock-step so unfinished cells are byte-identical.

---

## 3. Expected outputs

Per run dir you will find:
- `result_<sha8>.json` — the full manifest (`records[]`, `conditions[]`, routing/kbound summaries).
- `per_condition_<dataset>_<method>_seed<S>.json` — one file per (method/backbone, seed),
  matching the `stress_grid_multiseed_v1` schema (each record has `B, a0, a_adapted, regime,
  oracle_action, Z, Z_names, b_hat, eps_conformal, kga_decision, ...`). On the Mac these
  carry `"kga_backend": "sklearn_gradient_boost"` (production certificate).
- `MULTISEED_ANALYSIS_RESULTS.json` — pooled-across-seeds regret, **paired** bootstrap 95%
  CIs + Holm over the (method x {always-adapt, always-freeze}) comparisons, plus the p*
  regime-law separability check. Same schema as
  `experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json`.

**Approx runtime** (Apple-silicon MPS, very rough; scales linearly with seeds):
- B4 ImageNet-R, 3 seeds x 12 cells x (1 f0 + 10 frozen backbones): ~2–5 h
  (heaviest backbones — `convnext_base`, `swin_b`, `vit_b_16` — dominate; lazy-loaded).
- B1 Camelyon17, completing 90 cells x (1 f0 + 6 candidates) at n_eval=1024: ~3–6 h.
  Already 27/90 done in the existing checkpoint, so the remainder is less.

Run overnight with `caffeinate` (the script adds it automatically when available).

---

## 4. Verify success (schema check)

A one-liner that confirms every (method, seed) cell was serialized with the required fields,
all seeds are present, the **SAR** column exists (Camelyon17), and the multi-seed CIs are
finite:

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
python3 - <<'PY'
import json, os, glob, re
def check(run_dir, dataset, want_seeds, require_sar):
    pcs = glob.glob(os.path.join(run_dir, f"per_condition_{dataset}_*_seed*.json"))
    assert pcs, f"no per-condition files in {run_dir}"
    methods, seeds = set(), set()
    REQ = {"B","a0","a_adapted","regime","oracle_action","Z","Z_names",
           "b_hat","eps_conformal","kga_decision"}
    for p in pcs:
        m = re.match(rf"per_condition_{re.escape(dataset)}_(.+)_seed(\d+)\.json$", os.path.basename(p))
        methods.add(m.group(1)); seeds.add(int(m.group(2)))
        d = json.load(open(p)); r0 = d["records"][0]
        assert REQ <= set(r0), f"{p} missing {REQ - set(r0)}"
        assert len(r0["Z"]) == len(r0["Z_names"]) == 11
    assert set(want_seeds) <= seeds, f"{dataset}: seeds {seeds} missing {set(want_seeds)-seeds}"
    if require_sar:
        assert "sar" in methods, f"{dataset}: SAR column absent (methods={sorted(methods)})"
    ana = os.path.join(run_dir, "MULTISEED_ANALYSIS_RESULTS.json")
    if os.path.exists(ana):
        a = json.load(open(ana))
        for c in a["comparisons"]:
            assert all(map(lambda x: x==x, [c["ci95_lo"], c["ci95_hi"]])), "non-finite CI"
    print(f"OK  {dataset}: methods={sorted(methods)} seeds={sorted(seeds)} cells={len(pcs)}"
          + ("  [SAR present]" if require_sar else ""))

R = "experiments/kbound/results"
check(f"{R}/imagenetr_protocol_d_multiseed_v1", "imagenet-r", [0,1,2], require_sar=False)
check(f"{R}/camelyon17_fullscale_B_v2",          "camelyon17", [0,1,2], require_sar=True)
print("SCHEMA CHECK PASSED")
PY
```

(Run only the `check(...)` line for whichever experiment(s) you actually ran. For
ImageNet-R the "methods" are the 10 frozen backbones, so `require_sar=False`.)

---

## 5. Integrity notes

- The runners **append** per-condition serialization; the existing `result_<sha8>.json`
  manifest is unchanged in shape. Toggle with `--no-serialize-per-condition` if ever needed.
- The CPU pre-flight uses a documented numpy fallback estimator
  (`kga_backend="numpy_knn_fallback"`) because the sandbox lacks sklearn; the **decision
  rule, conformal radius, serialization layout, and paired-CI/Holm machinery are identical**
  to production. On the Mac (sklearn present) the real gradient-boosted certificate runs and
  the files are stamped `kga_backend="sklearn_gradient_boost"`.
- Canonical files left untouched: `experiments/kbound/results/wilds/wilds_camelyon17_kga.json`
  and `experiments/kbound/results/stress_grid_multiseed_v1/*`.
