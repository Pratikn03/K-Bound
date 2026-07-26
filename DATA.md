# DATA.md — dataset provenance, versions, licences, and acquisition

**Written 2026-07-26** to close fix-queue item 20 (F4-13). Three entries in
`docs/research/kbound/STORAGE_MANIFEST.json` pointed at this file and it did not exist; one of the
acquisition commands it gave (`bash scripts/download_data.py --dataset imagenet-r`) is not a real
invocation of that script, and another referenced an `AETTA/` directory that is not in the release.

**Ground rule for this document.** Every version string, split definition, corruption list and grid
shape below was read out of a committed artifact or a committed script in this repository, and the
file it was read from is named. Where a fact is *not* recoverable from the release, the row says
**UNPINNED** and states what the author must supply. Nothing here is inferred from what a dataset
"usually" is. Checksums are recorded as **NOT RECORDED** rather than guessed; they must be filled
at release time by the procedure in `docs/research/kbound/PLACEHOLDER_INVENTORY.md`, Guard 2.

---

## 0. Summary: which table depends on which dataset

| dataset | promoted artifact it supports | evidence tier today |
|---|---|---|
| CIFAR-10 + CIFAR-10-C | `tab:primary-numeric`, uniform panel rows CIFAR-10-C Tent / EATA, `tab:gates`, all ablations | locked; the paper's strongest track |
| ImageNet-C | uniform panel row ImageNet-C SAR, `tab:imagenetc-perseed` | locked; point-estimate claim only after the item-4 radius fix |
| CIFAR-10.1 v6 | uniform panel row CIFAR-10.1 K (declared negative result) | diagnostic fail; no claim |
| WILDS Camelyon17 | uniform panel row Camelyon17 OOD; `tab:multiseed` Camelyon rows | **sealed but not recomputable from release** (see §4b) |
| WILDS iWildCam | uniform panel row iWildCam H v2 | locked row, but its source record file is absent (§4) |
| WILDS RxRx1 | uniform panel row RxRx1 J | locked |
| Office-Home | uniform panel row Office-Home M v2 | locked row, but both source record files are absent and the runner source is an unreadable placeholder (§4) |
| PACS | uniform panel row PACS (null diagnostic) | locked diagnostic; cannot be re-scored (no `b_hat`/ε in released per-cell dumps) |
| ImageNet-R | uniform panel row ImageNet-R D (null diagnostic) | locked diagnostic |

`ImageNet-1k` (the source-domain training set for the ImageNet-C / ImageNet-R backbones) is
**never downloaded by this project**: all ImageNet backbones are torchvision pretrained weights.
See §9.

---

## 1. CIFAR-10 (source domain)

| field | value | read from |
|---|---|---|
| version | canonical CIFAR-10 python batches, obtained via `torchvision.datasets.CIFAR10(download=True)` | `docs/research/kbound/scripts/cifar_tent_mps_v2.py` |
| split used | the 10 000-image test split only; the source model `f0` is a pretrained CIFAR-10 classifier | `cifar_tent_mps_v2.py` |
| licence | MIT-style research use (Krizhevsky 2009); no redistribution in this repository | upstream |
| DOI / URL | `https://www.cs.toronto.edu/~kriz/cifar.html` | upstream |
| archive checksum | **NOT RECORDED** — torchvision verifies its own md5 on download | |

## 2. CIFAR-10-C

| field | value | read from |
|---|---|---|
| **canonical URL** | `https://zenodo.org/records/2535967/files/CIFAR-10-C.tar` | `docs/research/kbound/scripts/run_decisive_cifar.sh:19` |
| Zenodo record | **2535967** (Hendrycks & Dietterich, ICLR 2019) | same |
| licence | CC BY 4.0 (Zenodo record) | upstream |
| local layout expected | `$KBOUND_DATA_ROOT/CIFAR-10-C/<corruption>.npy` + `labels.npy` | `run_decisive_cifar.sh` |
| archive checksum | **NOT RECORDED** — Zenodo publishes an md5 per file; record it at release | |

**The operating point actually used is a subset, and it must be stated wherever the track is
reported.** The promoted 432-cell-per-seed grid is the driver's `--quick` mode:

- **6 of the 15 standard corruptions**, exactly:
  `gaussian_noise, defocus_blur, fog, contrast, pixelate, jpeg_compression`
  (`cifar_tent_mps_v2.py`, `CIFAR_C_QUICK`).
- **severities {1, 5}** only — not 1-5.
- crossed with batch regime `{large_iid, small, tiny}` × label regime
  `{iid, imbalanced, single_class}` × aggressiveness `{mild, aggressive}` × replicate `{r0, r1}`.
- 6 × 2 × 3 × 3 × 2 × 2 = **432 cells per candidate per seed**, which is the published n.

Verified by decomposing the `condition` strings in
`experiments/kbound/results/mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed0.json`
(864 rows = 432 Tent + 432 EATA). Any sentence describing this track as "the official CIFAR-10-C
corruptions" is false as written.

## 3. ImageNet-C

| field | value | read from |
|---|---|---|
| **canonical URL** | `https://zenodo.org/records/2235448/files/<group>.tar?download=1`, groups `blur noise weather digital` (+ `extra`) | `docs/research/kbound/scripts/download_all_datasets.sh:74` |
| Zenodo record | **2235448** (Hendrycks & Dietterich, ICLR 2019) | same |
| licence | CC BY 4.0 (Zenodo record) | upstream |
| local layout | `<root>/<corruption>/<severity 1-5>/<class>/*.JPEG` | `download_all_datasets.sh:78` |
| checksum tooling | `docs/research/kbound/scripts/verify_imagenetc_tars.sh` md5-verifies each tar against a `_zenodo_md5sums.txt` reference file that the user must place next to the tars | that script |
| archive checksum | **NOT RECORDED in the repository** — the verifier exists, the reference sums are not committed | |
| acquisition path in run manifests | `/Users/pratik_n/imagenetc_local` — a private path; replicators must use `--imagenetc-root` | run manifests |

**Operating point of the promoted 27-cell-per-seed grid**, read by decomposing the `condition`
strings in `experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed1.json`:

- **3 corruptions only**: `gaussian_noise, shot_noise, impulse_noise` (all three are the *noise*
  family — the corruption-family clustering in `NUMBERS_PACK.md §0.1` therefore has only 3
  clusters and must not be used as a primary interval).
- severities **{1, 3, 5}**.
- batch regime `small` only, aggressiveness `aggressive` only.
- label regime `{iid, imbalanced, single_class}`.
- 3 × 3 × 1 × 1 × 3 = **27 cells per seed**, 5 seeds = 135 rows.

Note that `IMAGENET_C_QUICK` in `cifar_tent_mps_v2.py` names a *different* 6-corruption subset
(`gaussian_noise, defocus_blur, snow, contrast, elastic_transform, jpeg_compression`). The
promoted run did not use it. Do not cite `IMAGENET_C_QUICK` as the grid.

## 4. WILDS (Camelyon17, iWildCam, RxRx1)

| field | value | read from |
|---|---|---|
| **library version** | `wilds 2.0.0` | `experiments/kbound/wilds/READINESS.md:5`, `experiments/kbound/results/iwildcam_streaming_pilot/PREREG.md:125`, `experiments/kbound/wilds/run_rxrx1_kbound.py:56` |
| torch version alongside it | 2.5.1 (`~/.venv_wilds`, MPS) | `READINESS.md:5` |
| **dataset versions** | `camelyon17_v1.0`, `iwildcam_v2.0`, `rxrx1_v1.0` | `research_lock/KBOUND_6_DATASET_PANEL_v1.yaml:43,112`; `experiments/kbound/wilds/READINESS.md:18`; `run_camelyon17_kbound.py:441` |
| acquisition | `from wilds import get_dataset; get_dataset(dataset="camelyon17", download=True, root_dir=...)` | `download_all_datasets.sh:57-62` |
| licence | Camelyon17 CC0 1.0; iWildCam CC BY 4.0; RxRx1 CC BY-NC-SA 4.0 (per the WILDS dataset cards) | upstream |
| DOI | WILDS: Koh et al., ICML 2021 | upstream |

**Pin this.** `download_all_datasets.sh:52` currently installs WILDS unpinned
(`$PIP install -q wilds`). A replicator on a later WILDS release may get different official splits.
Change that line to `$PIP install -q wilds==2.0.0` before release; that is the version every
committed manifest was produced under.

### 4a. Camelyon17 — data completeness disclosure

`experiments/kbound/wilds/READINESS.md:18-20` records that the runs used an **internal copy that
was 90.9% complete**: 414 389 of 455 954 patches. Center 2 (the `test` = hardest OOD hospital) is
recorded as 100% present; the ~41.5k missing patches are in other centers and the loader's
disk-filter drops them, logged. `T9_AUDIT.md:17` confirms "active run uses internal `~/kbound_cam`,
NOT this [complete T9 copy]".

Consequence: **a third party downloading the complete `camelyon17_v1.0` will not reproduce the
non-test-center conditions cell-for-cell.** This has to be stated in the paper's data section. It
is separate from, and additional to, the fact below.

### 4b. Camelyon17 — the promoted row is sealed but not recomputable

The promoted panel row is `0.0000 / 0.0000 / 0.1381 (n = 18, FA_u = 0)`. Its status, stated
precisely:

- **The regret triple IS recorded on disk**, in exactly one place:
  `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29` —
  `OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381,
  beats_both: false}`. That file is sealed in `LOCK_SEAL.json` and its hash verifies byte-for-byte.
  A grep restricted to `*.json` (as the review panel ran) misses it; it is a `.yaml`.
- **The promoted `FA_u = 0` is not recorded anywhere.** That YAML's only false-adapt figure for
  Camelyon is `idval_only: {false_adapt: 0.80}`. The `OOD_test_only` entry has no false-adapt field.
- **Nothing recomputes it.** The YAML is a hand-transcribed summary of a rerun, not a per-cell
  artifact. The three files it names as its own evidence —
  `audits/integrity_2026-06-20/camelyon_reconciliation/{camelyon_G_reconciliation.py,
  recon_results.json, VERDICT_phase1.md}` — are all absent; two of them are the only 2 of 72 files
  sealed in `LOCK_SEAL.json` that are missing from disk, and the third (the script) was never
  sealed at all.
- **The nearest live artifacts disagree on different slices**, as they should, since they score
  different subsets: `camelyon17_protocol_G_v1` gives `false_adapt` 0.0256 at n = 54 (the
  contaminated pooled split) and `camelyon17_richZ_F_v1` gives 0.0329 at n = 324.

So the correct label is **"sealed but not recomputable from release"**, and it is applied
throughout this repository in place of "locked". A reader can verify that the number was written
down before the paper cited it; a reader cannot verify that it is correct. Restoration procedure:
`docs/research/kbound/SUBMISSION_LEDGER.md §8`.

### 4c. Other absent record files

`docs/research/kbound/scripts/bootstrap_win_cis.py` loads four record files to produce the
promoted Office-Home / iWildCam / Camelyon17 bootstrap intervals. All four are absent:

```
experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json
experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json
experiments/kbound/results/iwildcam_full_test/result_e40faf29.json
experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json
```

These are small per-condition JSONs — exactly the class `EXTERNAL_STORAGE_POLICY.md` declares
"tracked in Git". They must be committed, or registered in `STORAGE_MANIFEST.json` with checksums
and an acquisition procedure. Until then the Office-Home and iWildCam promoted values are
locked-by-lock-file only.

## 5. Office-Home

| field | value | read from |
|---|---|---|
| version / revision | **UNPINNED** | — |
| domains | Art, Clipart, Product, Real-World (4 domains, 65 classes) | upstream convention; not asserted anywhere in the release |
| **split definition** | **UNPINNED.** The protocol names roles (`target_val`, `target_test`, cal seeds {0,1}, test seeds {0,1}) but the code that materializes them is unreadable | `research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml:33-41` |
| acquisition | **NO PATH IN THE RELEASE** | — |
| licence | research use only, by request from the dataset authors (Venkateswara et al., CVPR 2017) | upstream |

**This is the least reproducible dataset in the panel.** The runner
`experiments/kbound/officehome/run_officehome_kbound.py` (17 202 bytes) and the analysis
`oh_analyze.py` (18 989 bytes) are **iCloud placeholders — zero readable bytes** (see
`docs/research/kbound/PLACEHOLDER_INVENTORY.md`, group B), together with nine more files in that
directory. Both source record files are also absent (§4c). Nothing about the Office-Home split can
be recovered from the release.

**Author action required, in this order:** (1) materialize the placeholders (`brctl download`),
(2) read the split out of `oh_data.py` and write it here, (3) commit the two record files, (4)
state the acquisition URL and licence terms.

## 6. PACS

| field | value | read from |
|---|---|---|
| **acquisition** | HuggingFace dataset `flwrlabs/pacs`, exported to the DomainBed ImageFolder layout by `docs/research/kbound/scripts/export_pacs_hf.py` | that script's docstring |
| why not DomainBed directly | the DomainBed Google Drive link is quota-blocked | `export_pacs_hf.py:4` |
| local layout | `<root>/PACS/{art_painting,cartoon,photo,sketch}/<class>/*.jpg` | `scripts/pacs_vlcs_runner.py:21` |
| protocol | leave-one-domain-out; 4 targets × 3 seeds × 18 cells | `PACS_MULTISEED_RESULTS.json` |
| revision pin | **UNPINNED** — `load_dataset("flwrlabs/pacs")` takes no revision; pin the HF commit sha | `export_pacs_hf.py:28` |
| licence | PACS is research-use (Li et al., ICCV 2017); the HF mirror inherits it | upstream |

**Re-scoring limitation.** The released per-cell dumps (`results/per_cell/pacs_*_percell.json`)
carry `Z, a0, aa, B` but **no `b_hat`, no `eps_conformal` and no decision**, and seed 0 has no
per-cell dump at all. PACS therefore cannot be re-scored under a corrected radius from the release;
only the rates in `PACS_MULTISEED_RESULTS.json` are available.

## 7. ImageNet-R

| field | value | read from |
|---|---|---|
| **acquisition** | **NO PATH IN THE RELEASE.** The runner takes `--imagenetr-dir` (default `<data>/imagenet-r`) and assumes the data is already there | `experiments/kbound/wilds/run_imagenetr_kbound.py:478` |
| canonical upstream | Hendrycks et al., "The Many Faces of Robustness", ICCV 2021 — repository `github.com/hendrycks/imagenet-r`. **VERIFY the tarball URL against that repository before release; it is not recorded anywhere in this tree.** | upstream |
| revision | **UNPINNED** | — |
| licence | MIT (per the upstream repository) — **verify before release** | upstream |
| panel shape | 10 backbones × 4 seeds × 12 conditions | `imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` |

`STORAGE_MANIFEST.json` previously gave `bash scripts/download_data.py --dataset imagenet-r` as the
acquisition command. **That is not a real invocation.** `scripts/download_data.py` is the parent
monorepo's Enron/CIFAR-10 downloader and accepts only `--enron`, `--cifar10`, `--all`,
`--no-kaggle`. The line has been corrected in the manifest to point here.

## 8. CIFAR-10.1

| field | value | read from |
|---|---|---|
| **version** | **v6** (the ~2 000-image release) | `cifar_tent_mps_v2.py:471,1186` |
| **data URL** | `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy` | `cifar_tent_mps_v2.py:467` |
| **labels URL** | `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy` | `cifar_tent_mps_v2.py:468` |
| acquisition | automatic — the runner downloads to `<root>/CIFAR-10.1/` if absent | `cifar_tent_mps_v2.py:475-483` |
| split used | the v6 set is split ~half eval / ~half adapt-stream by the runner | `cifar_tent_mps_v2.py:1181` |
| source model | the **CIFAR-10** classifier is reused unchanged | `cifar_tent_mps_v2.py:465` |
| licence | MIT (Recht et al. 2019 repository) | upstream |
| archive checksum | **NOT RECORDED** | |

This is the one dataset in the panel whose acquisition is fully automatic and fully pinned.

## 9. Pretrained backbones

No ImageNet-1k training data is used or required. All ImageNet-side backbones are torchvision
pretrained weights, cached by `download_all_datasets.sh:84-89`:

- `resnet50` — `ResNet50_Weights.IMAGENET1K_V2`
- `vit_b_16` — `ViT_B_16_Weights.IMAGENET1K_V1`
- the ImageNet-R panel additionally uses `convnext_base`, `convnext_tiny`, `efficientnet_b0`,
  `efficientnet_b3`, `resnet101`, `resnet152`, `resnext101_32x8d`, `swin_b`, `swin_t`
  (10 backbones total, named in `MULTISEED_ANALYSIS_RESULTS.json`).

WILDS-side source models `f0` are trained in-repo: 4 DenseNet-121 seeds,
`results/wilds/f0_seed{0..3}.pt`, 28 MB each (`experiments/kbound/wilds/READINESS.md:21`).

---

## 10. What a third party can actually obtain today

| dataset | obtainable from this release alone? |
|---|---|
| CIFAR-10, CIFAR-10-C | **yes** — Zenodo 2535967, command in `run_decisive_cifar.sh` |
| CIFAR-10.1 v6 | **yes** — automatic, URLs pinned in the runner |
| ImageNet-C | **yes** — Zenodo 2235448, command in `download_all_datasets.sh`; md5 reference file not committed |
| WILDS Camelyon17 / iWildCam / RxRx1 | **yes for the data** (pin `wilds==2.0.0` first); **no for the exact Camelyon patch set** (90.9% copy, §4a) |
| PACS | **yes** — HF `flwrlabs/pacs` via `export_pacs_hf.py`; revision unpinned |
| ImageNet-R | **no** — no URL anywhere in the release |
| Office-Home | **no** — no URL, no split definition, unreadable runner |

Two of nine are unobtainable, one is partially reproducible. That is the honest state, and it is an
improvement on the pre-2026-07-26 documentation only in that it is now written down.

## 11. Release checklist for this file

Before the release is cut, each of these must be closed:

1. Pin `wilds==2.0.0` in `docs/research/kbound/scripts/download_all_datasets.sh:52`.
2. Add the ImageNet-R acquisition URL and licence (§7) after verifying it upstream.
3. Materialize the Office-Home placeholders and write the real split definition (§5).
4. Commit `_zenodo_md5sums.txt` for ImageNet-C so `verify_imagenetc_tars.sh` runs unaided (§3).
5. Fill every **NOT RECORDED** checksum, and register each dataset archive in
   `docs/research/kbound/STORAGE_MANIFEST.json` with `sha256` + `size_bytes`.
6. Commit or register the four absent record files (§4c) and restore the Camelyon reconciliation
   directory (§4b).
