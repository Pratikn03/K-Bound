# ELARA Phase 2 Compute Plan

**Locked at contract time. Updated as runs land.**

---

## 1. Datasets available locally

All Phase 2 evaluations operate on already-prepared inputs under `experiments/fusion/*_inputs.csv` and the matching metadata JSONs. No new dataset preparation is required.

| dataset | inputs file | metadata | size (samples) |
|---|---|---|---|
| MVTec 3D-AD PatchCore SP | `experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv` | `..._metadata.json` | ~3 226 paired (positive_fraction 0.224) |
| MVTec 3D-AD PatchCore canonical | `experiments/fusion/mvtec3d_patchcore_inputs.csv` | — | 3 226 |
| MVTec 3D-AD PatchCore held-out | `experiments/fusion/mvtec3d_patchcore_heldout_inputs.csv` | — | — |
| MVTec LOCO-AD PatchCore SP | `experiments/fusion/mvtec_loco_patchcore_supervised_paired_inputs.csv` | — | — |
| VisA RGB+edge SP | `experiments/fusion/visa_supervised_paired_inputs.csv` | — | — |
| UNSW-NB15 paired | `experiments/fusion/unsw_paired_inputs.csv` | — | 55 491 |
| ELARA-Bench-LA | `experiments/fusion/real_domain_fusion_inputs.csv` | `real_domain_fusion_metadata.json` | 8 000 |
| Real3D-AD | `experiments/fusion/real3d_supervised_paired_inputs.csv` | — | 1 254 |

## 2. Estimated per-cell training time (this hardware: M-series Mac, MPS)

| cell | seeds | est. minutes / seed | est. total minutes |
|---|---|---|---|
| MVTec 3D-AD PatchCore SP | 30 | 2–4 | 60–120 |
| MVTec 3D-AD PatchCore held-out | 30 | 2–4 | 60–120 |
| MVTec LOCO-AD PatchCore SP | 30 | 3–5 | 90–150 |
| VisA RGB+edge SP | 30 | 2–4 | 60–120 |
| UNSW-NB15 | 30 | 2–4 (large test fold) | 60–120 |
| ELARA-Bench-LA k-of-D (B1+B2) | 30 | 4–6 (sweeps over k) | 120–180 |
| RGA-v2 partial-failure sweep | 30 × 5 gates | 4–6 each | 600–900 |
| Mixture-shift + KS power | 5+ | <5 (synthetic) | <60 |

Estimated total compute for **all** Phase 2 experimental runs: 1100–1770 minutes ≈ **18–30 hours wall-clock**.

## 3. This-session scope (user-confirmed)

Per the scope question at the top of Phase 2:

- Layer 1 (this task): contracts + prediction-archive infrastructure + Family-D contract freeze + **one 30-seed pilot** on MVTec 3D-AD PatchCore SP (~60–120 minutes).
- Layer 2 (future sessions): remaining Family A (4 cells × 30 seeds), Family B mechanism replication (30 seeds), RGA-v2 partial-failure sweep, mixture-shift + KS power, risk-dominance + switching certificates across all cells.

## 4. Pilot cell choice — MVTec 3D-AD PatchCore SP

Rationale:
- Already has a 30-seed Phase-1 result for direct comparison.
- Supervised-paired protocol = inferential cell (not protocol-diagnostic).
- Pairing strength `independent_modalities` = strongest pairing tier available.
- One of the five A-POWERED-* cells, the most-cited in the abstract.

The pilot will validate:
1. The new prediction-archive contract end-to-end.
2. The seed-averaged ensemble-DeLong + paired-sample-bootstrap inference path.
3. Direction / sign agreement vs the Phase-1 audited reanalysis (target: router 0.7389 vs SAR 0.7354, Δ +0.0035, p_Holm 0.919 n.s.).
4. The Holm framework's compatibility with the existing Phase-1 result structure.

If the pilot diverges from Phase 1 by more than the policy-allowed numerical drift, the pilot will be flagged and we will pause for a hostile-review check before generalising to the remaining 4 cells.

## 5. Fallback rule

If the pilot exceeds 2 hours wall-clock, it will be reduced to 5 seeds and explicitly **downgraded to pilot status** in the Phase-2 interim report. No claim other than "pilot reproduction" is then permitted for that cell.

## 6. Status field semantics in the experiment registry

- `pilot_complete_at_30_seeds_for_this_cell_only` — fully run in this session under the pilot scope.
- `pending_compute` — registered, contract-frozen, NOT yet run. Cannot be cited as evidence.
- `contract_frozen_no_execution` — Family D status; cannot be activated without independent review.

## 7. Honest deviation

This compute plan acknowledges that Phase 2 cannot be completed in a single interactive session. The deviation is declared up front, the contract layer is frozen so future runs cannot drift, and the prediction-archive infrastructure is built before any pilot run so future runs can be added without protocol changes.
