# Family-D v2 — Data Provenance and Hash Report

**Phase:** 2.2C / Step 2
**Status:** **PROVENANCE RECORDED; ARCHIVE SHA256 NOT YET RECORDED (BLOCKED branch).**

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

## 4. Archive SHA256 (DEFERRED)

| Category | Archive SHA256 |
|---|---|
| candycane | **NOT RECORDED — download pass required** |
| chocolatecookie | **NOT RECORDED — download pass required** |
| chocolatepraline | **NOT RECORDED — download pass required** |
| confetto | **NOT RECORDED — download pass required** |
| gummybear | **NOT RECORDED — download pass required** |
| hazelnuttruffle | **NOT RECORDED — download pass required** |
| licoricesandwich | **NOT RECORDED — download pass required** |
| lollipop | **NOT RECORDED — download pass required** |
| marshmallow | **NOT RECORDED — download pass required** |
| peppermintcandy | **NOT RECORDED — download pass required** |

The Eyecandies maintainers do not publish per-archive SHA256 hashes. Recording these requires a local download and `shasum -a 256` pass; see [FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md](./FAMILY_D_V2_RAW_DATA_ACCESS_LOG.md) §4 for the exact verbatim future-task command.

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
