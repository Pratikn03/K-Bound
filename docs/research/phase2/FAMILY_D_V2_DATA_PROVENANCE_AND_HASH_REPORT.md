# Family-D v2 — Data Provenance and Hash Report

**Phase:** 2.2C / Step 2
**Status (post Phase 2.2D):** **PROVENANCE + ARCHIVE SHA256 + SCHEMA VERIFICATION COMPLETE.** All 10 Eyecandies category archives downloaded, hashed, and tar-member-scanned without inspecting anomaly mask file contents.

This report is the partition-manifest companion document. It records every provenance fact about the Eyecandies dataset that does not require local download, and explicitly defers the archive SHA256 entries to a future hash-only download pass.

## 1. Dataset

- **Name:** Eyecandies
- **Official source:** https://eyecan-ai.github.io/eyecandies/
- **Official repository:** https://github.com/eyecan-ai/eyecandies
- **Official paper:** Bonfiglioli et al., ACCV 2022, *"The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization"*. Bibitem key already in `docs/research/PAPER_DRAFT_v1.tex`: `bonfiglioli2022eyecandies`.

## 2. Release / version identifier

- **Tag:** `eyecandies-1.0.3` (release date 2023-01-23, per GitHub Releases).
- **Package installed in this audit:** `eyecandies==1.0.3` (verified via `pip show eyecandies`).
- **Source code commit at install time:** captured from `pip install git+https://github.com/eyecan-ai/eyecandies` — the installed package's `__version__` is `1.0.3`.

## 3. Per-category official identifiers

Recorded from `eyecandies/commands/download.py`:`EyecandiesDatasetInfo.DATA_IDS` (immutable for release 1.0.3):

| Category | Drive file ID | Local archive path (after download) |
|---|---|---|
| candycane | `1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY` | `data/raw/eyecandies/CandyCane.zip` |
| chocolatecookie | `1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN` | `data/raw/eyecandies/ChocolateCookie.zip` |
| chocolatepraline | `1dRlDAS31QJSwROgA6yFcXo85mL0EBh25` | `data/raw/eyecandies/ChocolatePraline.zip` |
| confetto | `10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq` | `data/raw/eyecandies/Confetto.zip` |
| gummybear | `1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W` | `data/raw/eyecandies/GummyBear.zip` |
| hazelnuttruffle | `1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm` | `data/raw/eyecandies/HazelnutTruffle.zip` |
| licoricesandwich | `1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC` | `data/raw/eyecandies/LicoriceSandwich.zip` |
| lollipop | `1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz` | `data/raw/eyecandies/Lollipop.zip` |
| marshmallow | `1pebIU3AegEFilqqoROaVzOZqkSgX-JTo` | `data/raw/eyecandies/Marshmallow.zip` |
| peppermintcandy | `1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF` | `data/raw/eyecandies/PeppermintCandy.zip` |

## 4. Archive SHA256 (RECORDED — Phase 2.2D)

Recorded in [experiments/phase2/family_d/eyecandies_archive_sha256.txt](../../../experiments/phase2/family_d/eyecandies_archive_sha256.txt) by `src/scripts/family_d_v2_download_eyecandies.py`. Mirrored in `FAMILY_D_PARTITION_MANIFEST_v2.json`.

| Category | Archive SHA256 | Size (bytes) |
|---|---|---:|
| CandyCane | `66ac74c9a3e60648395594d55058c625fb8dc720016983b60cf4e68415476f1c` | 2 401 843 200 |
| ChocolateCookie | `8070c912a45083be79407fd3f241ebab0f7f933b757a96f837bfd87c5de4d26c` | 2 803 148 800 |
| ChocolatePraline | `dfda3b6bed41ccbcd7b3593f7cab893c7c3a49a3e81a79a01cde2c148955150f` | 2 748 620 800 |
| Confetto | `f4b30ab23bab5a9cc36df20c38b2b174040469d3ca0341b00fd60bbcb2848392` | 2 566 348 800 |
| GummyBear | `d0ba4d61c8da2b12bb49a2e6e6cd1ca3d04661b6da0bce41cfcddc80140fe68d` | 2 886 205 440 |
| HazelnutTruffle | `dac3afb0090fe7b011b3b7fa7b85a5c2d63bcf954c63c427f65a632f671d3550` | 3 562 772 480 |
| LicoriceSandwich | `d06c1cb5cbbf28d12276cf75fe306541c4f2e83b5b0ba7484fb01fc7ee025958` | 2 821 570 560 |
| Lollipop | `2596ffb70882406b78e372f0e3dd7f19e7686a83833e833565e8211379bd40a8` | 2 396 805 120 |
| Marshmallow | `8363bb917e91c399ea79da7e53551197a4b1c93a32a17a5fc520889886969ab9` | 2 518 374 400 |
| PeppermintCandy | `499523d5800c5add427a8efe84e94b59462766fa861c7bea00226faf28bde940` | 2 480 097 280 |

Total: ≈ 27.2 GB across 10 category archives, downloaded via `gdown` from the official per-category Google Drive file IDs encoded in `eyecandies==1.0.3`. The bundled `eyec ec-get` CLI fails on these large files due to outdated Google Drive confirm-token handling; `gdown` 6.0.0 succeeds against the same file IDs.

## 5. Modality availability (DEFERRED to local verification)

Per official source documentation:

| Modality | Documented as available? | Local verification |
|---|:---:|---|
| RGB (6 images per sample under different lighting) | YES | **DEFERRED** |
| Depth map | YES | **DEFERRED** |
| Normal map | YES | **DEFERRED** (not in primary endpoint per D3) |
| Anomaly masks (test split only) | YES (4 channels: bumps, dents, colors, normals) | **MUST NOT be inspected pre-execution** |
| Metadata (pose, depth normalisation, object params) | YES (public test); NO (private test) | n/a (not used in primary) |

## 6. Split semantics (per official source)

- **Train:** anomaly-free.
- **Validation:** anomaly-free.
- **Test:** anomalous samples with ground-truth anomaly masks (public test) or hidden (private test).

This matches the prerequisite for a one-class multimodal protocol (D4 of Phase 2.2C). The primary stress endpoints (D-EYE-1, D-EYE-2) use **only** the normal-only train and validation splits for fitting and calibration; the official anomalous test split is **only** read at the one-time held-out execution after independent review.

## 7. Sample counts (DEFERRED)

Counts per (category, split) are documented in the official paper but must be locally verified by counting files after download. **Not recorded in this report.**

## 8. Eligibility re-check

- Eyecandies is **untouched at outcome level** in this repository (per [FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md](./FAMILY_D_V2_DATASET_AND_PROTOCOL_DECISION.md) §2).
- Eyecandies is **not in Family-A registry** as any executed cell.
- No prediction archive references Eyecandies.

## 9. What this report does NOT contain

- Any anomaly-detection ROC-AUC, PR-AUC, F1, ECE, Brier, delta, p-value, or method-comparison ranking.
- Any model run output.
- Any read of the official test anomaly labels.
