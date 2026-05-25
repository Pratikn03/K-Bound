# Family-D v2 — Dataset and Protocol Decision

**Phase:** 2.2C / Step 1
**Status:** **DECISIONS LOCKED.** Eyecandies selected; freeze gated on archive hashing (see Step 2 of Phase 2.2C and the Step 11 BLOCKED report).
**Supersedes:** the Phase 2.2B.2 version of this file (which had `V2_FREEZE_BLOCKED_PENDING_USER_DECISIONS`).

## 1. Decision summary

| Item | Locked value |
|---|---|
| Selected dataset | **Eyecandies** |
| Official release/version | **`eyecandies 1.0.3`** (GitHub release at https://github.com/eyecan-ai/eyecandies, latest as of 2023-01-23) |
| Selected modalities | **RGB + depth (primary)**; normal maps available, excluded from primary |
| Primary method | **base RGA** (reliability-aware gating under normal-only calibration) |
| Comparator | **fixed `static_attention`** |
| Protocol | **validation-only degradation-calibrated one-class multimodal** |
| Confirmation claim ceiling | "Held-out confirmatory evidence under the frozen Eyecandies RGB+depth one-class degradation-stress protocol." |
| Family-D family size | 2 primary hypotheses (D-EYE-1, D-EYE-2) + optional descriptive secondary (D-EYE-3) |
| `test_evaluation_executed` | **`false`** (must remain false until independent review + execution authorisation) |

## 2. Untouched eligibility (verified this audit)

`grep -rli "eyecandies"` produces only:

- `docs/research/PAPER_DRAFT_v1.tex` — related-work citation only (`\cite{bonfiglioli2022eyecandies}` reference + one related-work sentence); **no outcome-level use**.
- Phase-2 Family-D contract / eligibility-review / registry / claim-matrix documents — design references only, **no outcome-level use**.

**No experiment CSV, no prediction archive, no inference table, no model run output references Eyecandies.** Eyecandies has never been evaluated for outcome on this repository. ✓ Untouched.

`data/raw/` contains: fraud, mvtec3d, cyber, visa, behavior, healthcare, nlp, vision, mvtec_loco, real3d. **No `eyecandies` subdirectory.** ✓ Not previously downloaded.

## 3. Official modality evidence (verified this audit)

Sources consulted via WebFetch + the installed `eyecandies==1.0.3` package source code (file `eyecandies/commands/download.py`):

- Official project page: https://eyecan-ai.github.io/eyecandies/
- Official paper: Bonfiglioli et al. ACCV 2022, *"The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization"*.
- Official repository: https://github.com/eyecan-ai/eyecandies (release 1.0.3).
- Official distribution: per-category Google Drive archives accessed by the official `eyec ec-get` CLI; per-category file IDs are embedded in the package source and are immutable per the 1.0.3 release.

Modalities per sample (per official page):
- 6 RGB images under different lighting conditions
- depth map
- normal map (surface normals)
- 4-channel anomaly-mask annotations on the public test set
- metadata: camera pose, depth normalisation, object parameters

Splits (per official page): "anomaly-free samples for model training and validation, while anomalous instances with precise ground-truth annotations are provided only in the test set."

Modality alignment under official structure: each sample carries a fixed naming key; RGB and depth share the same key per category per split. Verification of byte-level alignment requires local download (see §5).

## 4. Excluded candidates (carry-over from Phase 2.1 eligibility review)

| Candidate | Status |
|---|---|
| VisA | `INELIGIBLE_FOR_FAMILY_D` — registry-locked in Family A (A-POWERED-4) |
| MPDD | `INELIGIBLE_FOR_INDEPENDENT_MULTIMODAL_CONFIRMATION` — official modality manifest not verified |
| Additional RGB+depth/normal/point-cloud candidate | Deferred per spec D8; **not required for first v2 freeze** |

## 5. Why **base RGA**, not supervised RGA+

Eyecandies' official split is anomaly-free train + anomaly-free validation + anomalous test. **No anomalous validation labels exist** under the official structure. Supervised head-selection (router vs boost) requires labelled anomalous validation data; that is **not admissible** under the held-out invariant of this protocol. Therefore:

- Primary head: **base RGA** (reliability-aware gating; no anomalous-label-dependent selection).
- Selection rule: validation-only, using normal-only validation data and pre-specified evidence-degradation injections (per §6 of the operator spec).
- Comparator: **fixed `static_attention`** (no per-cell selection).

## 6. Confirmation limitations (preserved verbatim)

- Eyecandies is **synthetic** but naturally multimodal; held-out confirmation here is on synthetic-but-naturally-paired data, not on physical-sensor data.
- This v2 is held-out confirmation for **one protocol and one dataset** only; it is not universality or deployment validation.
- Single-dataset confirmation does **not** establish universality.
- A successful Family-D v2 result may **not** retroactively convert Family-A into confirmatory evidence.

## 7. Current freeze status (Phase 2.2C)

All design decisions D1–D8 are locked. The remaining technical blocker for an actual no-placeholder freeze is the **archive SHA256 manifest**: the official Eyecandies maintainers do not publish per-archive SHA256 hashes, so a real partition-manifest hash field requires a local download of all 10 category archives (~30 GB via `eyec ec-get`).

A clean freeze therefore requires a download pass under the authorisation already granted by the Phase 2.2C spec ("authorised to download Eyecandies only for: archive/hash recording; file-structure verification..."). That pass produces:

1. SHA256 of each per-category zip archive.
2. Sample counts per (category, split).
3. Modality-file presence per sample.
4. RGB / depth alignment verification.

See [FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md](./FAMILY_D_V2_DATA_PROVENANCE_AND_HASH_REPORT.md) for the recorded download artifacts (currently a no-download access log per Step 2 of this phase), and [PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md](./PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md) for the honest BLOCKED-branch verdict at the end of Phase 2.2C.

All other freeze-required fields (protocol YAML, degradation operator spec, hypotheses CSV, selection policy, execution commands NOT_RUN, hostile review) are produced in this phase with **no placeholders**; only the partition manifest's archive SHA256 requires the download pass.
