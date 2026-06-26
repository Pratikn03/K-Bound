# K-Bound Repo Cleanup & Finalization Plan

*Generated 2026-06-26. **PLAN ONLY — nothing has been deleted.** Execution is gated on your explicit confirmation per tier (see §8).*

---

## 0. TL;DR

- The repo is **~800 GB**; the drive (1.8 TB) is **91% full, 180 GB free**.
- **~99% of the bloat is raw datasets and image caches**, not code. Code + paper + results JSONs are tiny (~1.2 GB total).
- **You can safely recover ~700+ GB** because every paper number is already distilled into small cached result JSONs (kept), and the raw images are re-downloadable via the `_dl_*.py` scripts already in the repo.
- Target after cleanup: a **clean, fixed ~20–30 GB repo** that still fully reproduces the paper from cached artifacts.
- **Key safety point:** "ELARA is integrated into KGA" is a *conceptual* merge. Some ELARA artifacts (the 62-task anomaly breadth, the score archive, the multimodal D33 demo, the sealed Family-D negative) are **load-bearing for the paper** and must be **kept**. Only *unreferenced legacy* ELARA/AD data is removable.

---

## 1. Where the space actually goes (measured)

| Path | Size | Nature |
|---|---:|---|
| `data/raw/realiad_d3` | **259 GB** | Legacy ELARA industrial-AD (RealIAD D3). **Not used** by current KGA paper. |
| `experiments/kbound/data/wilds` | **216 GB** | Raw WILDS images (Camelyon/iWildCam/RxRx1/fMoW/Poverty). Results already cached. |
| `experiments/kbound/data/imagenet-c` | **152 GB** | Raw ImageNet-C images. Results already cached. |
| `data/raw/mvtec3d` | 26 GB | Legacy 3D-AD. |
| `data/raw/eyecandies` | 25 GB | Family-D **sealed FAILED** negative — *result is in `family_d_failure_record.md`*, raw re-downloadable. |
| `data/raw/mulsen_ad` | 19 GB | M2 external (registry). Result distilled. |
| `experiments/kbound/data/acdc_zips` (+`acdc`) | 18 GB | ACDC — **code-only, never run** (per DATA.md). |
| `data/raw/mvtec_loco` | 12 GB | Legacy AD. |
| `data/raw/real3d` | 10 GB | Legacy AD. |
| `data/raw/tsb_ad` | 9.1 GB | Time-series AD — not in current paper breadth. |
| `data/raw/3d_adam_anomalib` | 6.5 GB | M2 external. |
| `data/raw/mvtec_ad` / `visa` / `realiad` | 5.4 / 4.3 / 4.0 GB | Legacy AD. |
| `experiments/kbound/data/imagenet-r` | 7.9 GB | Raw; results cached. |
| `experiments/kbound/data/office_home` | 5.2 GB | Raw; results cached. |
| `docs/research/kbound` | **8.2 GB** | Paper dir — far too big; likely duplicate PDFs + heavy figures (review). |
| `.git` | 3.2 GB | Git history; `git gc` will shrink. |
| `imagenette` / `cifar` / `domainbed` | 3.6 / 3.1 / 2.4 GB | Mixed (CIFAR-10-C backs a win; others legacy). |
| tabular anomaly (`adbench*`,`fraud`,`cyber`,`nlp`,`baf`,`smd`,`nab`,`har`,`healthcare`) | ~3 GB total | Backs the anomaly breadth + BAF panel (mostly **keep**). |
| `src/` | 192 MB | Code (kga + elara + legacy uais). |
| `experiments/kbound/results` | 953 MB | **Cached result JSONs — the paper's numbers. KEEP.** |
| `research_lock` | 15 MB | **Pre-registration / integrity. KEEP.** |

---

## 2. KEEP set — never delete (this *is* the paper, ~1.2 GB)

- **`kga/`** — the KGA certificate core (the productized jewel).
- **`src/elara/`** — theorem implementations (T1–T9) used by the paper.
- **`src/scripts/kbound/`** — experiment drivers (`knowability_experiment.py`, `mixed_regime_experiment.py`, `make_synth_archive.py`, `smoke_trichotomy.py`).
- **`experiments/elara_u/score_archive/`** — the 123-task `.npz` archive backing the anomaly breadth + knowability + mixed-regime results.
- **`experiments/kbound/results/`** (953 MB) — every cached `(Z, a0, aa, B)` JSON behind the headline table and honest nulls.
- **`research_lock/`** — sealed protocols + decision log (the integrity backbone; do not touch).
- **`docs/research/kbound/`** paper *sources* + the two current PDFs (`kbound.tex/.pdf`, `kbound_short.tex/.pdf`, `kbound_frontier_appendix.tex`, `kbound_short_appendix.tex`, `paper/`, `figures/`, `kbound_pkg/`).
- **`DATA.md`, `audits/`, `README.md`**, and the **`_dl_*.py` download scripts** (these are what make raw-data deletion safe — they regenerate it).

> Load-bearing "ELARA" artifacts (keep): score archive, `breadth_existing_datasets.json`, the D33 multimodal demo artifacts (`controlled_multimodal_d33/`), and the sealed Family-D record. The *raw* datasets behind them are re-downloadable (see §4).

---

## 3. Tier 1 — safe deletes (zero paper risk, do first)

Pure junk / duplicates / dead files. No effect on any result or claim.

- `wget-log`, `wget-log.1/.2/.3` (~79 MB stray download logs)
- macOS `._*` sidecars, `.DS_Store`, `__pycache__/`, `*.pyc` (repo-wide)
- `src/mlruns/` (51 committed MLflow files — should never have been committed)
- `src/uais_v/` (empty dead directory)
- Stale paper backups: `kbound_full58_backup_2026-06-10.tex`, `kbound_pre6trim_*.bak.tex`, `kbound_short_pre6edit_*.bak.tex`, `kbound_short_preIEEE_*.bak.tex`
- Duplicate/old PDFs in `docs/research/kbound/`: `K-Bound_paper.pdf`, `K-Bound_paper_CLEAN.pdf`, `K-Bound_paper_officehome_*.pdf`, `K-Bound_paper_with_frontier*.pdf`, `K-Bound_short.pdf`, `kbound_results-integrated_*.pdf`, `kbound_submission.pdf`+`.tex` (stale)
- Scratch logs/dirs in `experiments/kbound/results/`: `_aetta_preview.log`, `_full.log`, `_inr_smoke.out`, `_pipe_smoke.log`, `_preview.log`, `_train_queue.log`, `_pipeline_smoke_verify/`, `_simcheck/`, the stalled `imagenetc_protocol_E_full_v1/` (1/456 cells)
- `git gc --aggressive` (3.2 GB `.git` → typically <1 GB)

**Tier 1 recovers ~0.5–2.5 GB and removes all clutter.**

---

## 4. Tier 2 — raw-data reclaim (huge win, reversible via re-download)

These are raw datasets/images whose **results are already cached** in the KEEP set, or which are **legacy/unused**. Deleting them does **not** change any paper number; it only means you'd re-download to re-run from scratch. The `_dl_*.py` scripts + `DATA.md` provenance make this reversible.

**Delete-safe (re-downloadable, results distilled):**

- `experiments/kbound/data/wilds` — **216 GB**
- `experiments/kbound/data/imagenet-c` — **152 GB**
- `experiments/kbound/data/imagenet-r` — 7.9 GB
- `experiments/kbound/data/office_home` — 5.2 GB
- `experiments/kbound/data/acdc_zips` + `acdc` — 18 GB (ACDC never run)

**Legacy ELARA / industrial-AD not used by the current KGA paper:**

- `data/raw/realiad_d3` — **259 GB** (single biggest reclaim)
- `data/raw/mvtec3d` (26) · `real3d` (10) · `mvtec_loco` (12) · `mvtec_ad` (5.4) · `visa` (4.3) · `realiad` (4) · `tsb_ad` (9.1)

**Review-then-delete (tied to a sealed/registry claim — keep the *result*, raw is re-downloadable):**

- `data/raw/eyecandies` (25) — Family-D sealed negative; result preserved in `family_d_failure_record.md`.
- `data/raw/mulsen_ad` (19) · `data/raw/3d_adam_anomalib` (6.5) — M2 external registry; confirm not cited in the thesis before removing.

**Tier 2 recovers ~700+ GB.** Recommended: move to an external archive *or* delete after a final `git`-committed snapshot of all cached JSONs.

---

## 5. Tier 3 — code archive (tidy, reversible)

Move (not delete) legacy code into `archive/legacy/` or a `legacy` git tag so imports never break:

- `src/uais/` legacy domain-ML (~149 files, ~40k LOC) **except** `src/uais/data/` loaders referenced by the kbound drivers.
- `infer_rga/` (RGA deployment surface).
- The ~160 one-off scripts in `src/scripts/` (esp. the 3,016-line `run_breakthrough_experiment.py`), keeping `src/scripts/kbound/`.
- Superseded result dirs: `win_finder_*`, `win_loop_*`, `hard_dataset_win_loop_v1`, `*_internal`, `*__STALE33_backup`, `*_partial`, `tier2_runs`, the many `imagenetr_kbound_*_mps*` scratch dirs.

---

## 6. Fix, don't delete

- **Broken dependency pins** (`requirements-api.txt`, `ci.yml`): fictional versions (`httpx2`, `pandas==3.0.3`, `certifi==2026.5.20`, `scikit-learn==1.8.0`, …) break the Docker/CI install path. Repair to real, installable versions before submission — reviewers/artifact evaluators will run these.
- `docs/research/kbound` is 8.2 GB — sub-scan before pruning (likely duplicate figure assets); keep `figures/` actually referenced by the two papers.

---

## 7. Proposed clean structure (the "fixed amount", ~20–30 GB)

```
AutoML_Flagship_V8/
├── kga/                      # KGA certificate core (pip-installable)
├── src/
│   ├── elara/                # theorem implementations T1–T9
│   └── scripts/kbound/       # experiment drivers + hermetic smoke
├── experiments/
│   ├── kbound/
│   │   ├── results/          # cached result JSONs (paper numbers)
│   │   ├── *.py              # analysis scripts (analyze_F, bootstraps, …)
│   │   └── data/_dl_*.py     # download scripts (raw data regenerated on demand)
│   └── elara_u/score_archive/  # 123-task anomaly archive
├── research_lock/            # sealed protocols + decision log
├── docs/research/kbound/     # paper sources + 2 current PDFs + figures
├── audits/                   # integrity audit trail
├── DATA.md  README.md  requirements*.txt (fixed)  pyproject.toml
└── archive/legacy/           # moved legacy code (reversible)
```

Raw datasets live **outside** the repo (external drive / re-download), referenced by path in `DATA.md`.

---

## 8. Execution protocol & confirmation gate

**Nothing runs without your go-ahead.** When ready, confirm by tier, e.g. *"do Tier 1"*. Each tier:

1. `git add -A && git commit -m "pre-cleanup snapshot"` first (so everything is recoverable).
2. For Tier 2/3 I will **list exact paths + a final size total**, then **move to `archive/` or an external location** rather than hard `rm`, unless you say delete outright.
3. Re-run the hermetic smoke (`bash scripts/smoke_kbound.sh`) + the test suite after each tier to prove nothing broke.
4. Re-verify both PDFs still build.

**Recommended order:** Tier 1 (safe, now) → Tier 3 (archive code) → Tier 2 (the 700 GB reclaim, after a snapshot + confirming the cached JSONs reproduce the tables).

> Reply with which tier(s) to execute. I will not delete or move anything until you do.
