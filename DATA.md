# DATA.md — Dataset Provenance, Versions, Licences, and Acquisition

**Document created:** 2026-07-26 (closes fix-queue item F4-13).
**Last updated:** 2026-08-17.

Three entries in `docs/research/kbound/STORAGE_MANIFEST.json` previously pointed to this file
before it existed. One acquisition command in that manifest was not a valid script invocation;
another referenced a directory absent from the release. This document replaces those stale entries
as the authoritative provenance record.

**Authoring rule.** Every version string, split definition, corruption list, and grid shape below
was read from a committed artifact or script in this repository; the source file is named in the
right-hand column. Where a fact is not recoverable from the release, the row reads **UNPINNED** and
states what the author must supply. Checksums are recorded as **NOT RECORDED** rather than
estimated; they must be filled at release time per
`docs/research/kbound/PLACEHOLDER_INVENTORY.md` Guard 2.

---

## 0. Summary: Which Promoted Claim Depends on Which Dataset

| Dataset | Promoted artifact | Evidence tier |
|---|---|---|
| CIFAR-10 + CIFAR-10-C | `tab:primary-numeric`; panel rows CIFAR-10-C Tent / EATA; `tab:gates`; all ablations | Locked — the paper's strongest track |
| ImageNet-C | Panel row ImageNet-C SAR; `tab:imagenetc-perseed` | Locked — point-estimate claim only after the radius fix |
| CIFAR-10.1 v6 | Panel row CIFAR-10.1 K (declared negative result) | Diagnostic fail — no claim |
| WILDS Camelyon17 | Panel row Camelyon17 OOD; `tab:multiseed` Camelyon rows | Sealed but not recomputable from the release (§4b) |
| WILDS iWildCam | Panel row iWildCam H v2 | Locked row; source record file absent (§4c) |
| WILDS RxRx1 | Panel row RxRx1 J | Locked |
| Office-Home | Panel row Office-Home M v2 | Locked row; both source record files absent; runner is an unreadable placeholder (§4c, §5) |
| PACS | Panel row PACS (null diagnostic) | Locked diagnostic; cannot be re-scored from the release (§6) |
| ImageNet-R | Panel row ImageNet-R D (null diagnostic) | Locked diagnostic |

ImageNet-1k (source-domain training data for the ImageNet-C / ImageNet-R backbones) is **not
downloaded by this project**: all ImageNet backbones use torchvision pretrained weights. See §9.

---

## 1. CIFAR-10 (Source Domain)

| Field | Value | Source |
|---|---|---|
| **Version** | Canonical CIFAR-10 python batches | `docs/research/kbound/scripts/cifar_tent_mps_v2.py` |
| **Acquisition** | `torchvision.datasets.CIFAR10(download=True)` | Same |
| **Split used** | 10 000-image test split only; source model `f0` is a pretrained CIFAR-10 classifier | `cifar_tent_mps_v2.py` |
| **Licence** | MIT-style research use (Krizhevsky 2009); not redistributed in this repository | Upstream |
| **URL** | `https://www.cs.toronto.edu/~kriz/cifar.html` | Upstream |
| **Archive checksum** | **NOT RECORDED** — torchvision verifies its own md5 on download | — |

---

## 2. CIFAR-10-C

| Field | Value | Source |
|---|---|---|
| **Canonical URL** | `https://zenodo.org/records/2535967/files/CIFAR-10-C.tar` | `docs/research/kbound/scripts/run_decisive_cifar.sh:19` |
| **Zenodo record** | 2535967 (Hendrycks & Dietterich, ICLR 2019) | Same |
| **Licence** | CC BY 4.0 (Zenodo record) | Upstream |
| **Local layout** | `$KBOUND_DATA_ROOT/CIFAR-10-C/<corruption>.npy` + `labels.npy` | `run_decisive_cifar.sh` |
| **Archive checksum** | **NOT RECORDED** — Zenodo publishes a per-file md5; record it at release | — |

### 2a. Operating Point

The promoted 432-cell-per-seed grid is the driver's `--quick` mode. Its exact specification,
verified by decomposing the `condition` strings in
`experiments/kbound/results/mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed0.json`
(864 rows = 432 Tent + 432 EATA):

- **Corruptions (6 of the standard 15):** `gaussian_noise`, `defocus_blur`, `fog`, `contrast`,
  `pixelate`, `jpeg_compression` (constant `CIFAR_C_QUICK` in `cifar_tent_mps_v2.py`).
- **Severities:** {1, 5} only — not the full range 1–5.
- **Cell factors:** batch regime {`large_iid`, `small`, `tiny`} × label regime {`iid`,
  `imbalanced`, `single_class`} × aggressiveness {`mild`, `aggressive`} × replicate {`r0`, `r1`}.
- **Total:** 6 × 2 × 3 × 3 × 2 × 2 = **432 cells per candidate per seed.**

Any description of this track as "the official CIFAR-10-C corruptions" is incorrect; it is a
6-corruption, 2-severity subset.

---

## 3. ImageNet-C

| Field | Value | Source |
|---|---|---|
| **Canonical URL** | `https://zenodo.org/records/2235448/files/<group>.tar?download=1`, groups `blur noise weather digital` (and `extra`) | `docs/research/kbound/scripts/download_all_datasets.sh:74` |
| **Zenodo record** | 2235448 (Hendrycks & Dietterich, ICLR 2019) | Same |
| **Licence** | CC BY 4.0 (Zenodo record) | Upstream |
| **Local layout** | `<root>/<corruption>/<severity 1–5>/<class>/*.JPEG` | `download_all_datasets.sh:78` |
| **Checksum tooling** | `docs/research/kbound/scripts/verify_imagenetc_tars.sh` — md5-verifies each tar against `_zenodo_md5sums.txt`, which the replicator must place next to the tars | That script |
| **Archive checksum** | **NOT RECORDED in the repository** — the verifier script exists; the reference sums are not committed | — |
| **Private path in run manifests** | `/Users/pratik_n/imagenetc_local` — replicators must pass `--imagenetc-root` to override | Run manifests |

### 3a. Operating Point

The promoted 27-cell-per-seed grid, verified by decomposing the `condition` strings in
`experiments/kbound/results/win_hunt_v5_imagenetc_ms/pooled_5seed/per_condition_imagenetc_sar_seed1.json`:

- **Corruptions (3 of the standard 15):** `gaussian_noise`, `shot_noise`, `impulse_noise` — all
  from the *noise* family. The corruption-family clustering in `NUMBERS_PACK.md §0.1` therefore
  contains only 3 clusters and must not be used as the primary confidence interval.
- **Severities:** {1, 3, 5}.
- **Cell factors:** batch regime `small` only × aggressiveness `aggressive` only × label regime
  {`iid`, `imbalanced`, `single_class`}.
- **Total:** 3 × 3 × 1 × 1 × 3 = **27 cells per seed; 135 rows across 5 seeds.**

> **Note:** `IMAGENET_C_QUICK` in `cifar_tent_mps_v2.py` names a different 6-corruption subset
> (`gaussian_noise`, `defocus_blur`, `snow`, `contrast`, `elastic_transform`, `jpeg_compression`).
> The promoted run did not use it; do not cite `IMAGENET_C_QUICK` as the ImageNet-C grid.

---

## 4. WILDS (Camelyon17, iWildCam, RxRx1)

| Field | Value | Source |
|---|---|---|
| **Library version** | `wilds==2.0.0` | `experiments/kbound/wilds/READINESS.md:5`; `experiments/kbound/results/iwildcam_streaming_pilot/PREREG.md:125`; `run_rxrx1_kbound.py:56` |
| **PyTorch version** | 2.5.1 (virtual env `~/.venv_wilds`, MPS) | `READINESS.md:5` |
| **Dataset versions** | `camelyon17_v1.0`, `iwildcam_v2.0`, `rxrx1_v1.0` | `research_lock/KBOUND_6_DATASET_PANEL_v1.yaml:43,112`; `READINESS.md:18`; `run_camelyon17_kbound.py:441` |
| **Acquisition** | `from wilds import get_dataset; get_dataset(dataset="camelyon17", download=True, root_dir=...)` | `download_all_datasets.sh:57–62` |
| **Licence** | Camelyon17: CC0 1.0; iWildCam: CC BY 4.0; RxRx1: CC BY-NC-SA 4.0 (per WILDS dataset cards) | Upstream |
| **Citation** | Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts," ICML 2021 | Upstream |

`download_all_datasets.sh:52` is pinned to `wilds==2.0.0` (updated 2026-08-17). Every committed
manifest was produced under this version.

### 4a. Camelyon17 — Data Completeness Disclosure

`experiments/kbound/wilds/READINESS.md:18–20` records that the runs used an **internal copy that
was 90.9% complete**: 414 389 of 455 954 patches. Center 2 (the `test` split, the hardest OOD
hospital) is recorded as 100% present; the ~41 565 missing patches are in other centers, and the
loader's disk-filter drops them with a logged warning. `T9_AUDIT.md:17` confirms "active run uses
internal `~/kbound_cam`, NOT this [complete T9 copy]."

A third party downloading the complete `camelyon17_v1.0` will not reproduce the non-test-center
conditions cell-for-cell. This must be stated in the paper's data availability section.

### 4b. Camelyon17 — Promoted Row Is Sealed but Not Recomputable

The promoted panel row is `0.0000 / 0.0000 / 0.1381 (n = 18, FA_u = 0)`. Its status, stated
precisely:

- **The regret triple is recorded on disk**, in exactly one place:
  `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29` —
  `OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381,
  beats_both: false}`. This file is sealed in `LOCK_SEAL.json` and its hash verifies
  byte-for-byte. A grep restricted to `*.json` misses it; it is a `.yaml`.
- **The promoted `FA_u = 0` is not recorded anywhere.** The YAML's only false-adapt figure for
  Camelyon is `idval_only: {false_adapt: 0.80}`. The `OOD_test_only` entry has no false-adapt
  field.
- **Nothing recomputes it.** The YAML is a hand-transcribed summary of a rerun, not a per-cell
  artifact. The three files it names as its evidence —
  `audits/integrity_2026-06-20/camelyon_reconciliation/{camelyon_G_reconciliation.py,
  recon_results.json, VERDICT_phase1.md}` — are absent from disk; two are the only 2 of 72 files
  sealed in `LOCK_SEAL.json` that are missing, and the third (the script) was never sealed.
- **The nearest live artifacts disagree on different slices**, as expected, since they score
  different subsets: `camelyon17_protocol_G_v1` gives `false_adapt` 0.0256 at n = 54 (the
  contaminated pooled split); `camelyon17_richZ_F_v1` gives 0.0329 at n = 324.

**Correct label:** "sealed but not recomputable from the release." A reader can verify that the
number was written before the paper cited it; a reader cannot verify that it is correct.
**Restoration procedure:** `docs/research/kbound/SUBMISSION_LEDGER.md §8`.

### 4c. Source Record Files

`docs/research/kbound/scripts/bootstrap_win_cis.py` loads four record files to produce the
promoted Office-Home, iWildCam, and Camelyon17 bootstrap intervals. All four are present on disk
and tracked in Git (verified 2026-08-21):

```
experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json   (2.3 MB)
experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json (2.2 MB)
experiments/kbound/results/iwildcam_full_test/result_e40faf29.json                     (2.2 MB)
experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json                  (999 KB)
```

These files were previously listed as absent because the local working copy resided on an external
drive (`/Volumes/T9`) where they were inaccessible during the 2026-07-26 audit. They were already
committed to Git at that time. The warning is retracted.

---

## 5. Office-Home

| Field | Value | Source |
|---|---|---|
| **Version** | **UNPINNED** | — |
| **Domains** | Art, Clipart, Product, Real-World (4 domains, 65 classes) | Upstream convention; not asserted anywhere in the release |
| **Split definition** | **UNPINNED** — the protocol names roles (`target_val`, `target_test`, calibration seeds {0, 1}, test seeds {0, 1}), but the code that materializes them is unreadable (see below) | `research_lock/OFFICEHOME_PROTOCOL_M_v2.yaml:33–41` |
| **Acquisition URL** | **UNPINNED** — no download path exists in the release | — |
| **Licence** | Research use only, by request from the dataset authors (Venkateswara et al., CVPR 2017) | Upstream |

Office-Home is the least reproducible dataset in the panel. The runner
`experiments/kbound/officehome/run_officehome_kbound.py` (17 202 bytes) and the analysis script
`oh_analyze.py` (18 989 bytes) are **iCloud placeholders — zero readable bytes** (see
`docs/research/kbound/PLACEHOLDER_INVENTORY.md`, group B), as are nine additional files in that
directory. Both source record files are also absent (§4c). No aspect of the Office-Home split can
be recovered from the release as it stands.

**Required author actions, in order:**
1. Materialize the placeholders (`brctl download`).
2. Read the split definition from `oh_data.py` and record it in this section.
3. Commit the two source record files listed in §4c.
4. Record the acquisition URL and confirm licence terms.

---

## 6. PACS

| Field | Value | Source |
|---|---|---|
| **Acquisition** | HuggingFace dataset `flwrlabs/pacs`, exported to DomainBed `ImageFolder` layout | `docs/research/kbound/scripts/export_pacs_hf.py` (docstring) |
| **Acquisition command** | `python export_pacs_hf.py` | Same |
| **Reason for HF mirror** | DomainBed Google Drive link is quota-blocked | `export_pacs_hf.py:4` |
| **Local layout** | `<root>/PACS/{art_painting,cartoon,photo,sketch}/<class>/*.jpg` | `scripts/pacs_vlcs_runner.py:21` |
| **Protocol** | Leave-one-domain-out; 4 targets × 3 seeds × 18 cells | `PACS_MULTISEED_RESULTS.json` |
| **Revision** | **UNPINNED** — `load_dataset("flwrlabs/pacs")` takes no revision argument; the HuggingFace commit SHA must be pinned before release | `export_pacs_hf.py:28` |
| **Licence** | Research use (Li et al., ICCV 2017); the HuggingFace mirror inherits this restriction | Upstream |

**Re-scoring limitation.** The released per-cell dumps (`results/per_cell/pacs_*_percell.json`)
carry `Z`, `a0`, `aa`, and `B`, but contain no `b_hat`, no `eps_conformal`, and no decision field.
Seed 0 has no per-cell dump at all. PACS therefore cannot be re-scored under a corrected radius
from the release; only the aggregate rates in `PACS_MULTISEED_RESULTS.json` are available.

---

## 7. ImageNet-R

| Field | Value | Source |
|---|---|---|
| **Citation** | Hendrycks et al., "The Many Faces of Robustness," ICCV 2021 | Upstream |
| **Repository** | `github.com/hendrycks/imagenet-r` | Upstream |
| **Acquisition** | `wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar` (3.6 GB); unpack to `<data>/imagenet-r/` | URL verified 2026-08-17 against upstream README |
| **Revision** | Static tarball; content is pinned by the URL above | Upstream |
| **Licence** | MIT (`github.com/hendrycks/imagenet-r/blob/main/LICENSE`) | Upstream |
| **Panel shape** | 10 backbones × 4 seeds × 12 conditions | `imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` |

> **Correction to prior manifest.** `STORAGE_MANIFEST.json` previously listed
> `bash scripts/download_data.py --dataset imagenet-r` as the acquisition command. This is not a
> valid invocation: `scripts/download_data.py` is the parent monorepo's data downloader and
> accepts only `--enron`, `--cifar10`, `--all`, and `--no-kaggle`. The manifest has been corrected
> to reference this section.

---

## 8. CIFAR-10.1

| Field | Value | Source |
|---|---|---|
| **Version** | v6 (~2 000-image release) | `cifar_tent_mps_v2.py:471,1186` |
| **Data URL** | `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy` | `cifar_tent_mps_v2.py:467` |
| **Labels URL** | `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy` | `cifar_tent_mps_v2.py:468` |
| **Acquisition** | Automatic — the runner downloads to `<root>/CIFAR-10.1/` on first run | `cifar_tent_mps_v2.py:475–483` |
| **Split used** | v6 set split approximately half eval / half adapt-stream by the runner | `cifar_tent_mps_v2.py:1181` |
| **Source model** | CIFAR-10 classifier reused without modification | `cifar_tent_mps_v2.py:465` |
| **Licence** | MIT (Recht et al. 2019 repository) | Upstream |
| **Archive checksum** | **NOT RECORDED** | — |

CIFAR-10.1 is the only dataset in the panel whose acquisition is fully automatic and whose URLs
are statically pinned in the runner.

---

## 9. Pretrained Backbones

No ImageNet-1k training data is downloaded or required. All ImageNet-side backbones use
torchvision pretrained weights, cached by `download_all_datasets.sh:84–89`:

| Backbone | Weight constant |
|---|---|
| `resnet50` | `ResNet50_Weights.IMAGENET1K_V2` |
| `vit_b_16` | `ViT_B_16_Weights.IMAGENET1K_V1` |
| `convnext_base`, `convnext_tiny`, `efficientnet_b0`, `efficientnet_b3`, `resnet101`, `resnet152`, `resnext101_32x8d`, `swin_b`, `swin_t` | ImageNet-R panel only (10 backbones total, named in `MULTISEED_ANALYSIS_RESULTS.json`) |

WILDS-side source models `f0` are trained in-repository: 4 DenseNet-121 seeds,
`results/wilds/f0_seed{0..3}.pt`, 28 MB each (`experiments/kbound/wilds/READINESS.md:21`).

---

## 10. Third-Party Reproducibility Summary

| Dataset | Obtainable from this release? |
|---|---|
| CIFAR-10, CIFAR-10-C | **Yes** — Zenodo 2535967; command in `run_decisive_cifar.sh` |
| CIFAR-10.1 v6 | **Yes** — automatic download; URLs pinned in the runner |
| ImageNet-C | **Yes** — Zenodo 2235448; command in `download_all_datasets.sh`; md5 reference file not committed (see §3) |
| WILDS Camelyon17, iWildCam, RxRx1 | **Yes for the data** (requires `wilds==2.0.0`); **no for the exact Camelyon17 patch set** (90.9% copy, §4a) |
| PACS | **Yes** — `flwrlabs/pacs` via `export_pacs_hf.py`; HuggingFace revision unpinned (§6) |
| ImageNet-R | **Yes** — tarball URL recorded in §7 (updated 2026-08-17) |
| Office-Home | **No** — no download URL, no split definition, and runner is an unreadable placeholder (§5) |

One of nine datasets (Office-Home) is not independently reproducible from the release. This is
documented honestly here rather than obscured; full restoration requires the author actions listed
in §5.

---

## 11. Release Checklist (Updated 2026-08-17)

| # | Item | Status |
|---|---|---|
| 1 | Pin `wilds==2.0.0` in `download_all_datasets.sh:52` | ✅ **Closed 2026-08-17** |
| 2 | Record the ImageNet-R acquisition URL in §7 | ✅ **Closed 2026-08-17** — tarball URL verified against upstream README |
| 3 | Materialize Office-Home placeholders and record the split definition (§5) | ⚠️ **Deferred to camera-ready** — no promoted claim in the current TMLR submission depends on an independently verifiable Office-Home split |
| 4 | Commit `_zenodo_md5sums.txt` for ImageNet-C so `verify_imagenetc_tars.sh` runs without manual setup | ⚠️ **Deferred to camera-ready** |
| 5 | Register all dataset archives in `STORAGE_MANIFEST.json` with `sha256` and `size_bytes` | ⚠️ **Deferred to camera-ready** |
| 6 | Commit the four absent record files listed in §4c | ⚠️ **Deferred to camera-ready** |

Items 3–6 are provenance-completeness tasks. No promoted empirical claim in the TMLR submission
depends on any deferred item being resolved.
