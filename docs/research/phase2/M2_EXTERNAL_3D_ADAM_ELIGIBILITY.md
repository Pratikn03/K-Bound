# M2 external dataset eligibility — 3D-ADAM anomalib (D6)

**Ratified:** 2026-05-29  
**Seal:** `research_lock/M2_EXTERNAL_SEALED_v1.yaml`

## Selected dataset

| Field | Value |
|---|---|
| Name | 3D-ADAM anomalib subset (MechMind-Nano) |
| Source | [pmchard/3D-ADAM_anomalib](https://huggingface.co/datasets/pmchard/3D-ADAM_anomalib) |
| License | CC BY-NC-SA 4.0 |
| Modalities | Naturally paired `rgb/*.png` + `xyz/*` per scan (1:1 mapping per authors) |
| Layout | `<category>/{train,validation,test}/<defect>/rgb|xyz` (MVTec-3D-compatible) |

## Prior ELARA use check (must be zero before seal)

| Dataset | Prior use | Blocked for external M2? |
|---|---|---|
| Eyecandies | Family D failed transfer | Yes (development only) |
| MVTec 3D-AD | M1 + inverted M2 proxy | Yes |
| Real3D-AD | Family C exploratory | Yes |
| VisA | Family A development | Yes |
| **3D-ADAM** | **None before 2026-05-29** | **No — selected** |

Repository grep on 2026-05-29: no `3D-ADAM`, `3d_adam`, or `pmchard` paths in
`experiments/phase2/`, prediction archives, or Family A/B registries.

## Protocol

- **12 train categories** fit PatchCore-style scores and fusion calibration.
- **11 held-out test categories** — never used for selection; one-shot evaluation only.
- Validation: 15% of in-category official `test` rows (seed `20260529`), both classes.

## Commands

```bash
# 1) Download (per-category snapshot; or --zip for adam3d_cropped.zip)
.venv/bin/python src/scripts/scenario_c/download_m2_external_3d_adam.py

# 2) Prepare fusion CSV + split hashes (download + prepare)
.venv/bin/python src/scripts/scenario_c/seal_m2_external_3d_adam.py
```

## Artifacts (after seal)

- `data/raw/3d_adam_anomalib/`
- `experiments/fusion/m2_external_3d_adam_acquisition.json`
- `experiments/fusion/m2_external_3d_adam_sealed_inputs.csv`
- `experiments/fusion/m2_external_3d_adam_sealed_metadata.json`
- `elara_master_c/data/splits/split_hashes/m2_external_3d_adam_sealed.json`
