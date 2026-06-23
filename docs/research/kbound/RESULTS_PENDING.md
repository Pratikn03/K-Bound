# Pending GPU Runs → Paper Fold-In

Two natural-shift GPU jobs are in flight. This file states **exactly** what each
produces, which paper table/claim it fills, and the **one-line command** to fold
the result into paper-ready artifacts the moment it finishes.

The fold-in integrator is already proven end-to-end on the synthetic smoke report
(see `CLAIMS_CALIBRATION.md` §C). It reads only real fields and **errors loudly**
(exit 3) rather than emit a placeholder if any required field is missing.

Git commit at authoring: `eeb04ca6a85a7bb7a023dce146f0b926d49346a5`

---

## Run 1 — ImageNet-R, Protocol D, multi-seed

- **Run directory:** `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/`
- **Per-condition inputs it writes** (one per method×seed):
  `per_condition_imagenet-r_<method>_seed<S>.json` for `method ∈ {tent,eata,sar}`,
  `S ∈ {0,1,2}` — i.e. 9 cells, 24 conditions each (per the smoke manifest).
- **Aggregated artifact it produces:** `MULTISEED_ANALYSIS_RESULTS.json`
  (LOCKED schema from `experiments/kbound/wilds/multiseed_paired_ci.py`), containing
  `comparisons[]` with `mean_diff_kga_minus_trivial`, `ci95_lo`, `ci95_hi`,
  `p_raw`, `p_holm`, `survives_holm` for each of {tent,eata,sar} × {always-adapt,
  always-freeze}, plus `candidates{}`, `beats_both_by_candidate`, and `pstar_law`.
- **Paper slot it fills:** the natural-shift results table, **ImageNet-R row block**
  — closes the "ImageNet-R multi-seed" row in `CLAIMS_CALIBRATION.md` §C
  (currently PENDING-GPU). Also contributes ImageNet-R points to the Conjecture-1
  p\* separability check (does not prove it).

### Step 0 (only if the aggregate isn't auto-written) — produce MULTISEED_ANALYSIS_RESULTS.json
```bash
python3 experiments/kbound/wilds/multiseed_paired_ci.py \
  --run-dir experiments/kbound/results/imagenetr_protocol_d_multiseed_v1 \
  --dataset imagenet-r --methods tent eata sar --seeds 0 1 2 --nboot 10000
```

### Fold-in (one line) — emits LaTeX rows + Markdown summary, writes files into the run dir
```bash
python3 scripts/foldin_multiseed_results.py \
  --in experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json \
  --dataset imagenet-r --emit both \
  --out-dir experiments/kbound/results/imagenetr_protocol_d_multiseed_v1
```
Outputs: `foldin_imagenet-r.tex` (paste into the natural-shift table) and
`foldin_imagenet-r.md`. Header stamps the source file + git commit. If the run is
real, there is **no** `SYNTHETIC` tag.

---

## Run 2 — Camelyon17-WILDS, full-scale B (includes SAR)

- **Run directory:** `experiments/kbound/results/camelyon17_fullscale_B_v2/`
- **Per-condition inputs it writes:** `per_condition_camelyon17_<method>_seed<S>.json`
  for `method ∈ {tent,eata,sar}`, `S ∈ {0,1,2}` — 9 cells, 36 conditions each
  (per the smoke manifest). The **SAR** column is the headline addition here.
- **Aggregated artifact it produces:** `MULTISEED_ANALYSIS_RESULTS.json` (same
  LOCKED schema as Run 1).
- **Paper slot it fills:** the natural-shift results table, **Camelyon17 row block**
  — closes the "Camelyon17 SAR" row in `CLAIMS_CALIBRATION.md` §C (currently
  PENDING-GPU). Also adds Camelyon17 points to the p\* separability check.

### Step 0 (only if needed)
```bash
python3 experiments/kbound/wilds/multiseed_paired_ci.py \
  --run-dir experiments/kbound/results/camelyon17_fullscale_B_v2 \
  --dataset camelyon17 --methods tent eata sar --seeds 0 1 2 --nboot 10000
```

### Fold-in (one line)
```bash
python3 scripts/foldin_multiseed_results.py \
  --in experiments/kbound/results/camelyon17_fullscale_B_v2/MULTISEED_ANALYSIS_RESULTS.json \
  --dataset camelyon17 --emit both \
  --out-dir experiments/kbound/results/camelyon17_fullscale_B_v2
```
Outputs: `foldin_camelyon17.tex` and `foldin_camelyon17.md`.

---

## After both runs: update the ledger

1. Run the two fold-in commands above.
2. In `CLAIMS_CALIBRATION.md` §C, flip the two PENDING-GPU rows to **CLOSED** and
   paste the real `mean_diff` / CI / Holm-p / survive values from each
   `foldin_<dataset>.md` into the Note column.
3. Paste both `foldin_<dataset>.tex` blocks into the natural-shift results table in
   the K-Bound paper source.

### Notes / guardrails
- The integrator auto-detects schema. The **production** runs write the top-level
  `comparisons[]` schema; the synthetic smoke file uses the nested
  `datasets.<ds>.comparisons[]` schema. Both are handled.
- If a required field is missing, the integrator prints `SCHEMA ERROR: ...` and
  exits **3** — by design it will **not** invent a number. Treat any such error as
  "the run is incomplete / the aggregate wasn't written," not as a formatting bug.
- `--dataset` must match the `dataset` field stamped inside the production JSON, or
  the integrator aborts (exit 3) rather than mislabel a results block.
- Keep using `--seeds 0 1 2`; CIs need ≥2 seeds. If a seed crashed, re-run that one
  cell before aggregating — do **not** aggregate over a partial seed set silently.
