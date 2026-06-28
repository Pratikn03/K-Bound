# K-Bound — Final Multi-Seed Showcase Run (pre-registration)

**Purpose.** One pre-registered, leak-free end-to-end run of the full K-Bound panel on
real GPU, producing the camera-ready numbers, tables, figures, and PDFs from a single
command. Pre-registering the seeds, splits, tuning, and decision rules *before* the run
is what separates an honest robustness pass from a `win_finder`-style search over configs.

> **One command:** `bash docs/research/kbound/scripts/run_final_showcase.sh`
> (see flags at the bottom). It wraps the existing `kbtrain.sh final-all` engine and adds
> the missing tail: out-of-fold collation → `results_source.json` → tables → figures →
> both PDFs → verification.

---

## 1. Datasets (the panel)

The runnable engine (`kbtrain.sh final-all`) executes these nine:

| # | Dataset | Type | How seeds enter | Scored |
|---|---------|------|-----------------|--------|
| 1 | CIFAR-10-C | corruption grid | per-seed loop (`--seed`) | leave-one-condition-out (OOF) |
| 2 | ImageNet-C (noise, SAR) | corruption grid | per-seed loop (`--seed`) | leave-one-condition-out (OOF) |
| 3 | CIFAR-10.1 | natural null | per-seed loop (`--seed`) | grid |
| 4 | Camelyon17 | natural shift (WILDS) | runner trains 4 model seeds | dev{0,1}/test{2,3,4} locked |
| 5 | RxRx1 | natural shift (WILDS) | model-seed loop × condition seeds | locked grid |
| 6 | ImageNet-R | rendition shift | runner trains 4 seeds | locked grid |
| 7 | PACS | domain shift | per-seed loop (`--seed`) | grid |
| 8 | iWildCam | natural shift (WILDS) | YAML dev/test seed split | dev-locked OOF, scored once |
| 9 | Office-Home | domain shift | YAML dev/test seed split | dev-locked OOF, scored once |

### ⚠ OPEN ITEM — FMoW vs PACS (must decide before the camera-ready)

The **paper text** (`kbound_short.tex`, `kbound.tex`) lists the 9th dataset as **FMoW**
(a geo-shift null), but **`kbtrain.sh final-all` runs PACS** as the 9th — there is **no
FMoW runner** in `scripts/` (only `download_wilds_fmow_poverty.sh`). Pick one and make
text + runner agree:

- **Option A (less work):** treat **PACS** as the 9th in both the run and the paper; change
  the paper's "FMoW" null to "PACS". PACS is fully wired and runs today.
- **Option B (matches current text):** wire an FMoW runner (`Protocol L`) so the paper's
  stated panel is reproducible end-to-end, then swap PACS→FMoW in `final-all`.

Until this is resolved, the run uses PACS and the paper still says FMoW — a reviewer-visible
inconsistency. **This is the single most important thing to fix before submission.**

---

## 2. Seeds

`final-all` default is `KB_SEEDS="0 1 2 3 4"` (5 seeds). The showcase adds **3 more →
`0 1 2 3 4 5 6 7`** (8 seeds) for every dataset whose seed is a *training/init* seed:
CIFAR-10-C, ImageNet-C, CIFAR-10.1, PACS, and the model-seed axes of Camelyon17 / RxRx1 /
ImageNet-R. More seeds here = tighter, honest confidence intervals on the grid regrets.

**Not seed-redrawn (deliberately):** the three dev-locked natural-shift protocols
(**Office-Home, iWildCam, Camelyon17**). Their adapter is selected once on a fixed dev
split and the test split is scored exactly once. Re-drawing their test seeds to chase a
better number would be p-hacking; their uncertainty comes from the **condition bootstrap**
(B = 3000 resamples of held-out test conditions, dev held constant), not from seed reruns.

---

## 3. Tuning (leak-free — this is the crux)

"More tuning" means **a larger search on the dev split only**, never on test:

- For the dev-locked protocols, the adapter (and any TTA hyperparameters: lr, steps,
  episodic/online) is chosen on the **dev split** as the candidate with the **largest
  benefit margin subject to false-adapt ≤ α**. The chosen adapter is then frozen and the
  **test split is evaluated exactly once**.
- α = 0.10 and τ\* are **fixed constants — never tuned** (asserted in `analyze_F.py`,
  the protocol YAMLs, and the integrity guard).
- The corruption grids have **no tuning**: all of Tent/EATA/SAR are reported across the
  full severity/batch grid; KGA's radius is leave-one-condition-out.
- **Forbidden (excluded by design):** selecting the best config *on the test set*, or
  scanning many configs and reporting only the wins. That is the `win_finder` /
  `run_win_loop` anti-pattern; those scripts are retained only as a record of what *not*
  to trust and are never read by the pipeline.

---

## 4. Out-of-fold radius

The conformal radius is leave-one-out / cross-fit (`analyze_F.py::run_split`, GBR point
estimator, residual quantile at 1−α). The orchestrator **refuses to run** if any
in-sample-eps pattern (`predict(Zc) - Bc`, `abs(Bhat_c - Bc)` without `resid_c`/`_loo`/
`out-of-fold`) is present in a scorer — the same guard `final-all` already enforces.

`build_results_source.py` additionally **refuses** to copy natural-shift numbers from
`research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json` (that file's Office-Home CI is an in-sample
radius and overstates it as beats-both). Natural-shift numbers are recomputed from the
fresh out-of-fold protocol scorer outputs only.

---

## 5. Decision rule (pre-registered, reported verbatim)

For each dataset, KGA's verdict is computed, not chosen:

- **beats-both** *only if* the bootstrap 95% CI of (KGA − always-adapt) **and**
  (KGA − always-freeze) both exclude 0.
- **damage-prevention / no-harm** if KGA beats always-adapt (CI excludes 0) but ties
  always-freeze (CI includes 0).
- **abstain/null** on the nulls (CIFAR-10.1, FMoW/PACS): expect mostly freeze/abstain with
  false-adapt ≤ α and no spurious win.

**Honest prior for the outcome** (so the run can't be spun): grids → beats-both, CI-robust
(CIFAR-10-C, ImageNet-C). Natural shifts → **no-harm / damage-prevention, not beats-both**
(this is the structural story: on real shifts KGA buys safety, not extra accuracy). Whatever
the run actually computes is what goes in the paper.

---

## 6. Outputs & verification

```
kbtrain.sh final-all (8 seeds, device)            # all 9 datasets, OOF code, integrity-guarded
   └─► final_manifest_<stamp>.{json,md}           # per-dataset mean±std (diagnostic)
build_results_source.py                           # OOF recompute → results_source.json + _provenance + diff
   └─► make_tables.py        → paper/generated/kbound_numbers.tex
   └─► make_submission_figures.py → figures/fig_natural_forest.png, fig_frontier_schematic.png
latexmk kbound_short.tex, kbound.tex              # both PDFs
02_verify_results.py + consistency assert         # paper numbers == results_source; no verdict flipped to beats-both without CI
```

Every number written to `results_source.json` carries a `_provenance` entry (source file,
field, run stamp, git SHA). The verify stage aborts the rebuild if any natural-shift verdict
flips to *beats-both* without its bootstrap CI excluding zero.

---

## 7. Hardware

Device auto-detects (CUDA → MPS → CPU); override with `--device`. CIFAR-C is light;
**ImageNet-C is the slow stage** (tar-streamed, capped images on Apple MPS) — budget several
hours, and it is resumable (re-run to continue). RxRx1 needs pre-placed checkpoints at
`$RXRX1_CKPT_ROOT` (default `$HOME/kbound_rxrx1_ckpt`); everything else self-bootstraps or
downloads via `scripts/download_all_datasets.sh`.

---

## 8. Command

```bash
# full run (8 seeds, auto device):
bash docs/research/kbound/scripts/run_final_showcase.sh

# choose device / seeds:
KB_DEVICE=cuda KB_SEEDS="0 1 2 3 4 5 6 7" bash .../run_final_showcase.sh

# rebuild paper from existing run artifacts only (no GPU): 
bash .../run_final_showcase.sh --skip-train

# preview the command graph without running:
bash .../run_final_showcase.sh --dry-run

# tiny end-to-end check (caps images/steps):
bash .../run_final_showcase.sh --smoke
```
