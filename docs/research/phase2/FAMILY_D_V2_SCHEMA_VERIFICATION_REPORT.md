# Family-D v2 — Schema Verification Report

**Phase:** 2.2C / Step 2
**Status:** **Documentary schema recorded; on-disk schema verification DEFERRED to the future hash-only download pass.**

## 1. Documentary schema (from official Eyecandies sources)

Per the official project page + repository README:

```
data/raw/eyecandies/<Category>/
    train/                              # anomaly-free
        data/
            <sample_id>/
                rgb_<view>.png          # 6 views per sample under different lighting
                depth.png               # depth map
                normal.png              # surface normals
                ... metadata files
    val/                                # anomaly-free, official validation
        ... same structure as train ...
    test_public/                        # anomalous with ground-truth masks
        ... same structure + mask annotations (4 channels) ...
    test_private/                       # held-out, no public ground truth
        ... rgb/depth/normal only ...
```

## 2. Required verification at download time

| Item | Method | Performed? |
|---|---|:---:|
| Each category directory exists post-unzip | `ls data/raw/eyecandies/<Category>` | **DEFERRED** |
| `train/`, `val/`, `test_public/` splits present | directory listing | **DEFERRED** |
| Anomaly-free invariant for train+val: no mask files / non-zero anomaly channel | shape-only check (NO label inspection) | **DEFERRED** |
| RGB ↔ depth pairing: every sample dir contains both `rgb_*.png` and `depth.png` | file-name pairing | **DEFERRED** |
| Normal map availability | `ls **/normal.png` count | **DEFERRED** |
| Sample-ID uniqueness within split | hash of filenames | **DEFERRED** |
| Sample count per split per category | file count | **DEFERRED** |

All seven verifications must be added to the future Phase 2.2D download pass.

## 3. Schema invariants the v2 protocol relies on

These invariants are stated explicitly so that the future download pass can verify them without inspecting anomaly labels:

1. **Modality alignment** — for every `(sample_id, category, split)` triple, both an RGB file and a depth file exist.
2. **No test-anomaly leakage into train/val** — no anomaly-mask file under `train/` or `val/`.
3. **Sample-ID uniqueness within split** — no duplicate sample IDs within a (category, split).
4. **Modality stationarity** — RGB and depth images share the same camera-view alignment per sample (camera pose metadata identical).

These invariants are required by the one-class validation-only protocol (see [configs/phase2/family_d_v2_eyecandies_protocol.yaml](../../../configs/phase2/family_d_v2_eyecandies_protocol.yaml) §B).

## 4. Failure path

If the future download pass detects any invariant violation:
- STOP. Do not freeze.
- Document the violation in `FAMILY_D_V2_PRE_TEST_HOSTILE_REVIEW_REPORT.md` (the future hostile-review update).
- The v2 design must be revised or the dataset must be excluded.

## 5. Phase 2.2C status

- Documentary schema: **COMPLETE**.
- On-disk schema verification: **DEFERRED to future download pass**.
- This deferral is one of the contributing reasons for the BLOCKED verdict in [PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md](./PHASE_2_FAMILY_D_V2_BLOCKED_REPORT.md).
