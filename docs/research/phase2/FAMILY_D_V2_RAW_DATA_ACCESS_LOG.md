# Family-D v2 — Raw Data Access Log

**Phase:** 2.2C / Step 2
**Status:** No download performed in this task.

## 1. Mandatory invariants

> **No model evaluation executed.**
> **No performance metric computed.**
> **No result-based protocol changes made.**
> **Official test outcomes remain unused for method comparison.**

The above four statements describe the state of Phase 2.2C with respect to Eyecandies.

## 2. What was done

| Action | Performed? | Notes |
|---|:---:|---|
| Read official project page (`https://eyecan-ai.github.io/eyecandies/`) | YES | Metadata only |
| Read official GitHub README | YES | Metadata only |
| Install `pipelime-python==1.9.1` | YES | Tooling install (no data) |
| Install `eyecandies==1.0.3` from GitHub | YES | Tooling install (no data) |
| Read `eyecandies/commands/download.py` to extract per-category Google Drive file IDs | YES | Source-code inspection only |
| Record per-category file IDs in partition manifest | YES | See `FAMILY_D_PARTITION_MANIFEST_v2.json` `category_sources` |
| Download any Eyecandies archive (.zip) | **NO** | not performed in this task |
| Compute SHA256 of any downloaded archive | **NO** | no download performed |
| Verify RGB / depth alignment locally | **NO** | requires download |
| Count samples per (category, split) | **NO** | requires download |
| Read any Eyecandies anomaly label | **NO** | forbidden by held-out invariant |
| Compute any test-set ROC-AUC / PR-AUC / F1 / ECE / Brier / delta / p-value | **NO** | forbidden by held-out invariant |

## 3. Per-category official source identifiers (recorded from package source code)

The official `eyec ec-get` CLI uses these immutable Google Drive file IDs in `eyecandies==1.0.3`:

| Category | Drive file ID |
|---|---|
| candycane | `1OI0Jh5tUj98j3ihFXCXf7EW2qSpeaTSY` |
| chocolatecookie | `1PEvIXZOcxuDMBo4iuCsUVDN63jisg0QN` |
| chocolatepraline | `1dRlDAS31QJSwROgA6yFcXo85mL0EBh25` |
| confetto | `10GNPUIQTUheT-qd6EzO76fsUgAwsHfaq` |
| gummybear | `1OCAKXPmpNrD9s3oUcQ--mhRZTt4HGJ-W` |
| hazelnuttruffle | `1PsKc4hXxsuIjqwyHh7ciPAeS-IxsPikm` |
| licoricesandwich | `1dtU_l9gD1zoCN7fIYRksd_9KeyZklaHC` |
| lollipop | `1DbL91Zjm2I9-AfJewU3M354pW4vnuaNz` |
| marshmallow | `1pebIU3AegEFilqqoROaVzOZqkSgX-JTo` |
| peppermintcandy | `1tF_1fPJYaUVaf1AwjlEi-fsGWzgCx6UF` |

These file IDs are content-addressed by Google Drive's internal system but are NOT SHA256 hashes. The Phase 2.2C Step 5 partition manifest field "archive or file-manifest SHA256" requires real SHA256 of the downloaded zip archives; that step requires the download pass described in §4.

## 4. Required download command for the future Phase 2.2D-or-later task

The future task that performs the hash-only download must run **exactly**:

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
.venv/bin/eyec ec-get +o data/raw/eyecandies
# then per category, immediately after each archive lands:
for f in data/raw/eyecandies/*.zip; do
    shasum -a 256 "$f"
done > experiments/phase2/family_d/eyecandies_archive_sha256.txt
```

The future task may:
- compute archive SHA256 from the downloaded archives;
- count files per category per split;
- verify RGB / depth alignment by file-name pairing;
- verify normal-map availability per sample;
- write `FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md` with the recorded hashes.

The future task may NOT:
- read any anomaly label or anomaly mask;
- compute any test-set performance metric;
- train a Family-D model;
- change the operator spec or hypotheses CSV after seeing dataset contents.

## 5. Phase 2.2C closure on this log

This log is final. No further data access is performed in Phase 2.2C.
