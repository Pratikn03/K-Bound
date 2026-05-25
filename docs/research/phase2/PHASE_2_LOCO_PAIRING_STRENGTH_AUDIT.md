# Phase 2 — A-POWERED-3 (MVTec LOCO-AD) Pairing-Strength Audit

**Phase:** 2.2B.2 / Step 1
**Status:** REPAIR REQUIRED — registry label corrected from `independent_modalities` to `derived_view_proxy`.

## 1. Finding

The Phase-2 v2 experiment registry, the Phase-1 (legacy) registry, and the Family-A v2 static-reference audit report all currently label A-POWERED-3 (MVTec LOCO-AD, PatchCore supervised-paired) with `pairing_strength = independent_modalities`.

Inspection of the dataset-construction code [src/scripts/prepare_mvtec_loco_fusion_benchmark.py](../../../src/scripts/prepare_mvtec_loco_fusion_benchmark.py) lines 1–17 shows verbatim:

> "MVTec LOCO-AD is RGB-only (no depth channel). We construct two co-observed domains from each image:
>   - rgb         — ResNet-50 penultimate features of the colour image
>   - edge_proxy  — ResNet-50 penultimate features of the Sobel-gradient magnitude image. Acts as a hand-crafted structural companion to the semantic RGB stream.
> Both domains are derived from the same single observation, so the pairing is natural."

The config [configs/attention_mvtec_loco_patchcore_supervised_paired.yaml](../../../configs/attention_mvtec_loco_patchcore_supervised_paired.yaml) line 23 confirms `domain_order: [rgb, edge_proxy]`.

Both domains are derived views of the **same** RGB observation. Per Phase-2 contract terminology, this is `derived_view_proxy` (same pairing class as A-POWERED-4 VisA RGB+edge), **not** `independent_modalities`.

## 2. Decision

- `pairing_strength = derived_view_proxy` for A-POWERED-3.
- Numerical result (Δ AUC = +0.1038, CI [+0.058, +0.150], Holm K=5 p = 4.06e-05) is **unchanged** — this audit corrects an interpretive label only.
- A-POWERED-3 may **not** support independent-modality generalization claims.

## 3. Files updated by this audit

- [PHASE_2_EXPERIMENT_REGISTRY_v2.csv](./PHASE_2_EXPERIMENT_REGISTRY_v2.csv) — A-POWERED-3 row `pairing_strength` field changed.
- [PHASE_2_EXPERIMENT_REGISTRY.csv](./PHASE_2_EXPERIMENT_REGISTRY.csv) — preserved as historical record (NOT modified).
- [FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md](./FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md) — pairing-strength column in the cell roster + cell-by-cell caveats §5 updated.
- [src/scripts/emit_phase2_registries_v2.py](../../../src/scripts/emit_phase2_registries_v2.py) — A-POWERED-3 row updated so re-emission stays consistent.

Numerical CSVs (`family_a_v2_primary_cell_level_raw.csv`, `..._holm_k5.csv`) are **not** touched — the bug was interpretive, not numerical.

## 4. Test added

[tests/test_phase2_loco_pairing_strength_verified.py](../../../tests/test_phase2_loco_pairing_strength_verified.py) asserts that:
- the v2 registry row for A-POWERED-3 has `pairing_strength = derived_view_proxy`;
- the v2 Family-A report classifies A-POWERED-3 under `derived_view_proxy` in its caveats section.

## 5. Implication for permitted claims

Both A-POWERED-3 (MVTec LOCO-AD RGB + edge_proxy) and A-POWERED-4 (VisA RGB + edge) now share `derived_view_proxy` status. Neither cell can be cited in support of an independent-modality generalization statement. The Family-A primary claim remains:

> "Family A provides powered audited static-reference evidence across five previously inspected benchmark cells. It evaluates whether validation-frozen RGA+ improves on a fixed static-attention reference; it is not confirmatory replication and is not a strongest-baseline superiority evaluation."

Three of the five cells (A-POWERED-1 MVTec 3D-AD SP, A-POWERED-2 MVTec 3D-AD held-out, A-POWERED-5 UNSW-NB15) carry pairing strengths other than `derived_view_proxy`.
