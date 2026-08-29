# DATA.md — Dataset Provenance, Versions, Licences, and Acquisition

**Document created:** 2026-07-26 (closes fix-queue item F4-13).
**Last updated:** 2026-08-28.

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

Current retrospective hash coverage is recorded in
`docs/research/kbound/KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md` and its machine-readable
`audits/phase1_provenance_2026_08_27/provenance_seal.json`. A digest computed there after an
experiment is explicitly a post-hoc snapshot, not proof of the bytes used at execution time.

---

## 0. Summary: Dataset Roles and Evidence Status

| Dataset | Maintained artifact or role | Evidence status |
|---|---|---|
| CIFAR-10 + CIFAR-10-C | `tab:primary-numeric`; panel rows CIFAR-10-C Tent / EATA; `tab:gates`; all ablations | Locked — the paper's strongest track |
| ImageNet-C | Panel row ImageNet-C SAR; `tab:imagenetc-perseed` | Locked — point-estimate claim only after the radius fix |
| CIFAR-10.1 v6 | Panel row CIFAR-10.1 K (declared negative result) | Diagnostic fail — no claim |
| WILDS Camelyon17 | Archived Camelyon diagnostics | Opened/archived diagnostic evidence; the historical OOD row is sealed but not recomputable (§4b) |
| WILDS iWildCam | iWildCam H v2 diagnostic | Source record is tracked, but numerical promotion is withheld because the archived scorer does not implement the official WILDS macro-F1 contract (§4c) |
| WILDS RxRx1 | Panel row RxRx1 J | Locked |
| Office-Home | Canonical primary row, 54-cell replication, and later five-checkpoint candidate audit | Primary and replication evidence remain descriptive; only the later five-checkpoint route is invalid (§4c, §5) |
| PACS | Panel row PACS (null diagnostic) | Locked diagnostic; cannot be re-scored from the release (§6) |
| ImageNet-R | Panel row ImageNet-R D (null diagnostic) | Locked diagnostic |
| CCT-20 | Locked natural-location stress test on the official `trans_test` cameras | Execution was outcome-unopened before model execution; archive, population, protocol, and source-training inputs were sealed (§8a) |
| So2Sat LCZ42 v4.2 | Acquisition, label-free population manifest, structural protocol, and source-data preflight | Official repository revision and archive bytes are pinned; this document states no empirical outcome (§8b) |

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

The Phase-1 audit does not claim a complete current CIFAR-10-C content seal: the promoted-subset
`defocus_blur.npy` and `contrast.npy` files are macOS dataless placeholders in this working copy.
The SAR labels and source-checkpoint digests match their predeclared protocol values. Other
currently readable corruption-array hashes are post-hoc snapshots only.

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

### 4b. Camelyon17 — Historical OOD reconciliation provenance

The historical panel row is `0.0000 / 0.0000 / 0.1381 (n = 18, FA_u = 0)`. Its status, stated
precisely:

- **The regret triple is recorded on disk**, in exactly one place:
  `research_lock/CAMELYON17_PROTOCOL_G_RECONCILED_v2.yaml:29` —
  `OOD_test_only: {n_test: 18, regret_kga: 0.0, regret_adapt: 0.0, regret_freeze: 0.1381,
  beats_both: false}`. This file is sealed in `LOCK_SEAL.json` and its hash verifies
  byte-for-byte. A grep restricted to `*.json` misses it; it is a `.yaml`.
- **The historical `FA_u = 0` is not recorded anywhere.** The YAML's only false-adapt figure for
  Camelyon is `idval_only: {false_adapt: 0.80}`. The `OOD_test_only` entry has no false-adapt
  field.
- **The reconciliation files have been restored.** `recon_results.json` and `VERDICT_phase1.md`
  are present and match their historical lock hashes. `camelyon_G_reconciliation.py` is also
  present, but no historical seal authenticates its current bytes; its current digest is therefore
  recorded only as an unsealed post-hoc snapshot.
- **The nearest live artifacts disagree on different slices**, as expected, since they score
  different subsets: `camelyon17_protocol_G_v1` gives `false_adapt` 0.0256 at n = 54 (the
  contaminated pooled split); `camelyon17_richZ_F_v1` gives 0.0329 at n = 324.

**Correct label:** opened/archived diagnostic with restored reconciliation outputs, not untouched
prospective evidence. Restoration improves auditability but does not strengthen the scientific
claim.

### 4c. Source Record Files

`docs/research/kbound/scripts/bootstrap_win_cis.py` loads four record files used by the historical
Office-Home, iWildCam, and Camelyon17 bootstrap intervals. All four are present on disk
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
| **Version** | Hugging Face mirror `flwrlabs/office-home`; repository revision **UNPINNED** | `experiments/kbound/officehome/materialize_officehome.py` |
| **Domains** | Art, Clipart, Product, Real_World (4 domains, 65 classes) | `experiments/kbound/officehome/oh_data.py` |
| **Split definition** | Deterministic per-class split, seed 20260615: Real_World 70% train / 30% validation; each target domain (Art, Clipart, Product) 50% validation / 50% test | `experiments/kbound/officehome/oh_data.py:49–78` |
| **Acquisition** | `python experiments/kbound/officehome/materialize_officehome.py --out <root>`; downloads the `flwrlabs/office-home` mirror and writes `<root>/<Domain>/<Class>/*.jpg` | `experiments/kbound/officehome/materialize_officehome.py` |
| **Licence** | Research use only, by request from the dataset authors (Venkateswara et al., CVPR 2017) | Upstream |

The active materializer, split code, runner, and two source records are readable and tracked.
The former exploratory analysis was moved to
`archive/legacy_kbound/officehome/oh_analyze.py`; it is not an active publication path. A clean
third-party materialization is possible, but exact archive-level reproducibility remains incomplete
until the Hugging Face dataset revision and materialized-image checksum are pinned. The maintained
manuscript retains the canonical primary row and the 54-cell replication as descriptive evidence.
The later five-checkpoint candidate route remains invalid and is reported only as an archived
checkpoint-opportunity audit.

The current 15,588-image materialization has post-hoc content-tree SHA-256
`b995bd0f1ece7b589344c05d03fa61c200fb46fd24df15850d62d668998f8b66`. The Hugging Face revision
was not pinned before the historical run, so this digest is a rerun input receipt, not a historical
execution identity.

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

The current complete 30,000-image materialization has post-hoc content-tree SHA-256
`3f1bbfb98fe6fcaea3f2cf4ac22071330d5bee3ce3ba7ff2e86e84860ed62409`. The historical Protocol-D
run did not record that digest, so it must not be presented as a pre-run population seal.

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

The current v6 files have post-hoc SHA-256 values
`2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6` (data) and
`ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448` (labels). The historical
Protocol-K artifact did not bind these digests.

CIFAR-10.1 is the only dataset in the panel whose acquisition is fully automatic and whose URLs
are statically pinned in the runner.

---

## 8a. CCT-20 (Caltech Camera Traps Benchmark)

| Field | Value | Source |
|---|---|---|
| **Benchmark** | CCT-20 subset introduced by Beery, Van Horn, and Perona, *Recognition in Terra Incognita* (ECCV 2018) | `research_lock/KBOUND_CCT20_TARGET_SELECTION_v1.yaml`; upstream LILA page |
| **Official page** | `https://lila.science/datasets/caltech-camera-traps` | LILA |
| **Image archive** | `https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz` | Target-selection lock |
| **Annotation archive** | `https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz` | Target-selection lock |
| **Licence** | Community Data License Agreement — Permissive, Version 1.0 (`CDLA-Permissive-1.0`) | LILA dataset page; `https://cdla.dev/permissive-1-0/` |
| **Image archive identity** | 6,492,615,601 bytes; md5 `8143c17aa2a12872b66f284ff211531f`; SHA-256 `50d0e46d4f42c4891d99a13cda80b6c062d3586d79296edc9d1406a5e7cc4b20` | `research_lock/KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json` |
| **Annotation archive identity** | 2,997,071 bytes; md5 `66a1f481b44aa1edadf75c9cfbd27aba`; SHA-256 `e31d0162d411fb031ba4741758a54fa15cc7257df6f344581f5fda612b2cc974` | Same |
| **Official archive population** | 57,864 unique image records across the five supplied split files | `experiments/kbound/cct20/prospective_protocol_v1.yaml` |
| **Source training** | `train_annotations.json`, 13,553 images; deterministic sequence-hash partition into 12,083 fit and 1,470 source-monitor images | Source preflight and protocol |
| **Development gate** | FIT: `trans_val` location 125 and `cis_test` location 33; CAL: `cis_test` locations 38, 43, 51, 61, 88, 90, 108, 115, and 120 | Prospective protocol |
| **Locked natural target** | Complete `trans_test`: 23,275 images from previously unseen camera locations 0, 7, 28, 40, 46, 78, 100, 105, and 130 | Prospective protocol and label-free target manifest |
| **Annotation split SHA-256 values** | `train`: `439f8030d8e1200a4ebd9620cd79ad544c5fcdedafe9f3b5c478ba5463a79b6e`; `cis_val`: `d1191d4510307e2d7458b2e2fcb6d363facc754f2980080efd44cea8408c6e7f`; `cis_test`: `d13ef0b4d34c4a072b1bfbb147a7d67c9f513cd53609a01a0ac5f0d8d1cc95e3`; `trans_val`: `4f292434ddd2a727f6c8ab62d2193095d5148831c95359cc64a3239c6d7dc95b`; `trans_test`: `49b7bee90fee877e8c100f561cdce14dc12ed768204af586ff6de31d48cc8cdf` | Source-training seal |
| **Selected-image member-list SHA-256** | `82afcab526bdba692165fd66d819419e3568e1c7acaf1aeb04bbb5f5d6552a96` | Source-training seal |
| **Sealed protocol SHA-256** | `dc6f5da269b7e12523c036030f60b504fa46ca7170f15f9506004aa6e49041a5` | `prospective_protocol_v1.yaml` |

The ECCV paper reports 57,868 images, while the official CCT companion page and the five annotation
files in the downloaded archive contain 57,864 unique image records in total. The experiment uses
all 57,864 records supplied by those files and discloses the four-image difference; no row was
added, removed, or substituted. Archive-wide membership checks passed. Full decode-and-hash checks
passed for all 13,553 source-training images and all 23,275 locked target images.

The target is **outcome-unopened before model execution, not literally label-unopened**. During
candidate ranking, aggregate target annotation metadata was inspected to establish feasibility;
that inspection is recorded in
`research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_ADDENDUM.yaml`. At candidate-ranking and
target-selection time, no CCT-20 pixels, predictions, per-location outcomes, gate actions, or
K-Bound results had been inspected. The target runner is label-free and can read only the sealed
image metadata and pixels; the scorer receives labels once, after all checkpoint-by-location action
and prediction artifacts are immutable.

The archive's sparse category IDs define 16 evaluation classes: 14 animal categories, `car`, and
`empty`. Repeated same-category annotations collapse; distinct categories on one image are retained
as a set-valued target. These rules were fixed in
`research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_LABEL_CONTRACT_ADDENDUM.yaml` before source
training.

---

## 8b. So2Sat LCZ42 v4.2

| Field | Value | Source |
|---|---|---|
| **Benchmark** | So2Sat LCZ42, Culture-10 scenario; 17 local-climate-zone classes; 32 × 32 patches | `experiments/kbound/so2sat/prospective_protocol_v1.json` |
| **Official project** | `https://github.com/zhu-xlab/So2Sat-LCZ42` | Same |
| **Acquisition snapshot** | Hugging Face dataset repository `zhu-xlab/So2Sat-LCZ42`, revision `b5c817486899935e864832b93086cc87f3eee473`, directory `v4/` | `research_lock/KBOUND_SO2SAT_ACQUISITION_MANIFEST_v1.json` |
| **Licence** | CC BY 4.0 | Acquisition manifest |
| **Official split counts** | Training 352,366; validation 24,119; testing 24,188 | Structural protocol and label-free population manifest |
| **Sealed modality** | Sentinel-2 `sen2`, 10 bands | Structural protocol |
| **Training archive** | 16,002,423,246 bytes; SHA-256 `a72f8e834198312360c217b23fc3b9e1af3acd916b046d9bd0f91fa93d402b27` | Acquisition manifest |
| **Validation archive** | 1,100,208,338 bytes; SHA-256 `d8b0ad2030d4b873e0d41bcb69d73f7a9ecd1586d212a869cf7bb1d6f10e59a2` | Acquisition manifest |
| **Testing archive** | 1,100,520,239 bytes; SHA-256 `df7c3498bd265d21dbec03f5f618b0a901848536f96f0ff4f4e80be782d17f97` | Acquisition manifest |
| **Extracted training container** | 52,006,404,720 bytes; SHA-256 `06df6c9b8875e37f172ba548f466640293d77ad3be335c2d5dfcba3d35942daf` | `research_lock/KBOUND_SO2SAT_SOURCE_PREFLIGHT_v1.json` |
| **Geographic population contract** | Complete metadata-only manifest: 42 training cities partitioned into 14 source-fit, 9 gate-fit, and 19 gate-calibration cities; the 10 official Culture-10 cities remain the target population | `research_lock/KBOUND_SO2SAT_POPULATION_MANIFEST_v1.json` |

The acquisition receipt records the three compressed image archives as opaque bytes at the pinned
repository revision. The label-free population manifest opens only the geographic companions'
`city`, `epsg`, and `tfw` datasets. The source preflight opens only `training.h5` and
`training_geo.h5`; it reports 352,366 valid one-hot training labels and no critical finding. Its
status is `SOURCE_DATA_PREFLIGHT_PASSED_WITH_WARNINGS` because the source-monitor role lacks class
IDs 0 and 6 and the gate-fit probe role lacks class ID 6. These are source-data and provenance
facts, not an empirical K-Bound result. The committed acquisition and preflight receipts record no
target image-container or target-outcome-array access.

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
| Office-Home | **Yes, with an unpinned mirror revision** — materializer and deterministic split are in the release; pin the Hugging Face revision and materialized-image checksum for exact byte-level reproduction (§5) |
| CCT-20 | **Yes** — official LILA benchmark archives, exact URLs, md5 values, SHA-256 values, split hashes, and population contract are recorded in §8a |
| So2Sat LCZ42 | **Yes** — the official Hugging Face repository revision, archive byte counts, SHA-256 values, split counts, and source-container receipt are recorded in §8b |

All eleven benchmark entries now have an acquisition path; they name twelve datasets when
CIFAR-10 and CIFAR-10-C are counted separately. Exact byte-level reproduction is still incomplete
where archive revisions or checksums are explicitly marked unpinned, and the internal Camelyon17
copy cannot be reconstructed exactly (§4a).

---

## 11. Release Checklist (Updated 2026-08-28)

| # | Item | Status |
|---|---|---|
| 1 | Pin `wilds==2.0.0` in `download_all_datasets.sh:52` | ✅ **Closed 2026-08-17** |
| 2 | Record the ImageNet-R acquisition URL in §7 | ✅ **Closed 2026-08-17** — tarball URL verified against upstream README |
| 3 | Restore the Office-Home source, materializer, and split definition (§5) | ✅ **Closed 2026-08-27** — active code is readable and the split is documented |
| 4 | Commit `_zenodo_md5sums.txt` for ImageNet-C so `verify_imagenetc_tars.sh` runs without manual setup | ⚠️ **Deferred to camera-ready** |
| 5 | Register dataset/archive identities with `sha256` and `size_bytes` | ⚠️ **Partially closed** — current ImageNet-R, Office-Home, and CIFAR-10.1 bytes are post-hoc hashed; historical binding and the remaining datasets still require pinned archives or sealed reruns |
| 6 | Verify the four source record files listed in §4c | ✅ **Closed 2026-08-27** — all four are present, tracked, and SHA-256 checked |
| 7 | Pin the `flwrlabs/office-home` Hugging Face revision and record a materialized-tree checksum | ⚠️ **Deferred to camera-ready** |
| 8 | Record CCT-20 archive URLs, licence, byte counts, provider md5 values, SHA-256 values, split identities, and the locked target population | ✅ **Closed 2026-08-28** — recorded in §8a and bound by the pre-training seal |
| 9 | Pin the So2Sat LCZ42 repository revision, archive identities, split counts, label-free population, and source-container preflight | ✅ **Closed 2026-08-28** — recorded in §8b and bound by committed receipts |

Items 4, 5, and 7 remain provenance-completeness tasks. No maintained empirical claim in the TMLR
submission depends on treating a post-hoc digest, unpinned dataset, or invalid historical route as
publication-grade.
