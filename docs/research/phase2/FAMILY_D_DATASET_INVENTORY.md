# Family D — Dataset Inventory (frozen pre-execution)

**Status:** FROZEN — these datasets MUST be untouched until the Family-D execution window opens.

Intent: each candidate dataset is **held out** from any Family A / B / C decision making. None has been used to tune any part of the RGA+ pipeline (no gate threshold, no head selection, no comparator pick, no calibration).

## D1. Primary candidate — MPDD (Metal Parts Defect Detection)

- Modalities: RGB, 3D (depth via structured light).
- Status w.r.t. RGA+: **never touched**. No code path in `src/elara/` has ever read MPDD; verified by `grep -ri "mpdd"` returning no functional hits in `src/`.
- Why it is a fair held-out: the categories (metal parts) are disjoint from the MVTec 3D-AD object set (industrial textures + cookies + bagels + carrots) and from the Real3D object set (toys + figurines).
- Required artefacts before execution: dataset download script, license review, hash manifest.

## D2. Secondary candidate — Eyecandies

- Modalities: RGB, normal map, depth, multi-view.
- Status w.r.t. RGA+: **never touched** outside of citation-level reading. No `src/` code path reads Eyecandies.
- Why it is a fair held-out: synthetic but with different defect physics (transparency, sub-surface scattering) from MVTec 3D-AD.

## D3. Tertiary candidate — VisA (RGB-only)

- Modalities: RGB only.
- Status w.r.t. RGA+: **never touched**.
- Why it is appropriate as a *degenerate* held-out: only the RGA+ head's RGB stream is exercised. A negative result on VisA would not invalidate the multimodal mechanism; a positive result would be a useful upper-bound check.

## Selection ordering

D1 is the **primary**. D2 is executed iff D1 is data-available within the Family-D compute window. D3 is **only** executed as a sensitivity check after D1 has reported.

## Hash / version commitments

The dataset DOIs / release tags MUST be recorded in `FAMILY_D_PARTITION_MANIFEST.json` before any download script is run. Re-downloading at a later date produces a different `partition_manifest_sha256` and invalidates the confirmation.

## Out-of-scope candidates

The following datasets are **explicitly excluded** from Family D:

- MVTec 3D-AD — used in Family A and Family B; not held out.
- Real3D — used in Family A; not held out.
- Any clinical / medical imaging dataset — the forbidden claim "ELARA is validated for clinical deployment" precludes Family D from including a clinical dataset in this contract version.
