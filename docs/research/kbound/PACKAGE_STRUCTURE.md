# K-Bound paper package — structure & provenance

This folder is now organized as a self-contained, auditable paper package. **Nothing
was deleted.** Result JSONs, figures, and the manuscript were **copied** into organized
subfolders; the originals remain where the scripts write them (e.g.
`experiments/kbound/results/`, `src/scripts/kbound/`), so existing builds keep working.

```
docs/research/kbound/
├── README.md  README_DECISIVE.md  EXECUTION_STATUS.md  USABLE_CODE_AUDIT.md
├── THEOREM_CODE_STATUS.md  CODE_CONTRIBUTION_MAP.md  ASSET_INVENTORY.md  requirements.txt
├── PACKAGE_STRUCTURE.md            <- this file
│
├── paper/        K-Bound_paper.pdf, PAPER_K-Bound_draft.md, kbound.tex (copy w/ graphicspath),
│                 sections/  appendix/  references/refs_to_verify.md
├── figures/      final/ (14 PNGs)  +  source/ (figure_data_links.md, make_figures.py)
│                 (flat originals also kept alongside for the existing build)
├── results/      main/ ablations/ tta/(+logs) regression/ witness/ multimodal/  + result_manifest.json
├── experiments/  experiment_registry.csv  +  one folder per experiment (README + run_config.yaml)
├── src/kbound/   wrapper package indexing the real code (evidence/estimators/decision/
│                 metrics/theory/data/utils) — see src/kbound/README.md
├── scripts/      00_verify_environment, 01_build_manifests, 02_verify_results, 03_make_tables,
│                 04_make_figures, 99_reproduce_kbound  +  copies of the real experiment scripts
├── manifests/    data_inventory, score_cache_manifest (pointers only — no data duplicated),
│                 result_source_map, figure_source_map, claim_result_map,
│                 used_in_paper, available_not_used
├── vendored_from_elara/   certification/ drift/ theory/  (T1–T9, GDR, switching certificate)
└── archive/      old_drafts/  exploratory_results/  deprecated_figures/
```

## Where the big data lives (NOT duplicated)
`manifests/score_cache_manifest.csv` points to the real locations:
`experiments/elara_u/score_archive` (123-task archive) and
`experiments/fusion/*_score_cache` (multimodal caches). Raw datasets stay under `data/raw/`.

## Reproduce
```
python scripts/99_reproduce_kbound.py          # show the pipeline
python scripts/99_reproduce_kbound.py --run    # run the CPU/numpy stages
python scripts/02_verify_results.py            # audit every result file's headline numbers
```
CIFAR-10/100-C + ImageNet-C (the decisive deep-TTA experiment) run on your M5/GPU via
`scripts/cifar_tent_mps_v2.py` — see `README_DECISIVE.md`.

## Claim status (from manifests/claim_result_map.csv)
- **used** (backed by real runs): clean suite, harmful fusion, mixed regime, 8-seed rigor,
  non-identifiability witness, regression covariate shift, ablations.
- **verify_before_claim**: `cifar_tent`, `cifar_tent_online`, `tta_collapse` — results now
  exist; confirm they're final before updating the paper's "future work" wording.
- **corroborating**: multimodal instantiation (MVTec-3D +0.21 under failure).

## On "elara" / "uav"
Per your structure, `vendored_from_elara/` keeps its name (it's the load-bearing theory +
certificate code the paper depends on). Manifests reference `experiments/elara_u/…` and
`experiments/fusion/…` only as **pointers** to where the real score caches live — no code
or data was renamed or deleted, so nothing breaks.
