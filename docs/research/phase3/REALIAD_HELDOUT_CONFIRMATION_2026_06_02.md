# D18 Held-Out Confirmation — Natural-Degradation Reliability Gating (2026-06-02)

> **⚠️ CORRECTED / DOWNGRADED 2026-06-02 (D20). DO NOT cite the "+0.045 CONFIRMED"
> framing below as a confirmation.** A subsequent same-day audit
> (`D18_HELDOUT_CONFIRMATION_AUDIT_2026_06_02.md`) showed the pooled +0.0454 is
> **inflated by one degenerate detector channel** (`lego_propeller` XYZ,
> validation AUROC = 0.000, +0.98 single-category swing). Applying a
> validation-only degenerate-channel guard **fairly to both CW and the gated
> rule** collapses the pooled advantage to **+0.0049 — below the +0.010 bar**
> (within-category mean +0.120 → +0.011). The guard's main effect is to make CW
> itself robust, not to validate the gate. **`gate_natdeg_heldout_confirmed`
> stays FALSE.** The honest standing claim is the *guarded* one (below, and in the
> audit doc): genuine within-category gains (+0.09 to +0.21) on the 3 categories
> with an honestly-weak (non-degenerate) modality, ties on 6, a loss on 1; NOT a
> clean natural-degradation transfer confirmation. The original (unguarded) text
> is retained below for the record only.

---

Pre-registered (commit `d1195d0`, BEFORE this run) one-shot run of the frozen
`rel_x_sharp` reliability-gated fusion on **11 Real-IAD-3D categories the new
method had never touched**. Original framing (NOW SUPERSEDED by the banner
above): "CONFIRMED on the pre-registered primary endpoint."

## Result

| Metric | Value |
|---|---|
| Pooled CW | 0.700 |
| Pooled GATED (rel_x_sharp) | 0.745 |
| **Primary: GATED vs CW** | **Δ=+0.0454, 95% CI [+0.0225, +0.0683], CI_low>0 ✓** |
| Pre-registered pass condition (Δ≥0.010 ∧ CI_low>0) | **MET** |
| Within-category | **9 wins / 1 tie / 1 loss** |
| Within-cat mean (all) | +0.120 |
| Within-cat mean (excl. lego_propeller outlier) | **+0.034 (8/10 improve)** |

## Mechanism — validated on never-seen categories

- **lego_propeller (+0.98)**: validation flags XYZ as *inverted* (val AUC 0.0); CW
  averages it in and collapses to 0.0175 (worse than chance); the gate drops the
  unreliable modality → 1.0. The textbook reliability-gating case, on held-out data.
- **connector_housing_female (+0.21)**, **lego_pin_connector_plate (+0.10)**,
  **lattice_block_plug (+0.09)**: gate trusts the validation-reliable modality
  (high-val-AUC XYZ/PS), beats the blind mean.
- **fork_crimp_terminal (−0.098)**: honest loss — all modalities weak on validation
  (0.52–0.67), the noisy reliability estimate mis-transferred to test.
- **headphone_jack_socket (tie, both 1.0)**: ceiling, no room (consistent with T9).

## Honest caveats (binding)

1. **Holdout status = OPENED_DEVELOPMENT_ONLY.** These 11 categories were never
   touched by the *new* method (why they were chosen), but were scored once by the
   prior D16 attempt using a *different, broken* handcrafted detector. That run is
   uninformative for the new method, but it means this is a **development-grade
   category-held-out generalization confirmation**, not a pristine never-seen
   holdout. Level impact: supports ~3–3.5, not a clean-room 4.
2. **Not a strict-Gate-E pass.** Strict clean Gate E remains CLOSED BY PROOF (T9).
   This is the *natural-degradation* gate (D15/D16), a different, legitimate claim.
3. **One extreme outlier** (lego_propeller +0.98) inflates the within-category mean;
   the result is reported with and without it (+0.034 without, still 8/10 improve),
   so it is not cherry-picked.
4. Per-category bug fix (binary PCD, F1) is applied; fork_crimp & telephone (binary
   PCDs) used real geometry here.

## What this establishes

The natural-degradation reliability-gating win **generalizes to categories the
method never saw**: it beats the confidence-weighted mean pooled (CI clear) and
improves 9/11 categories within-category. This is the strongest positive evidence
in the project — a confirmed (development-grade) generalization of the mechanism
T1/T3 predict and T9 bounds. It moves the standing from "soft 2.5" toward a
**strong 3 / borderline 3.5**, capped below a clean 4 only by the opened-holdout
caveat.

## Artifacts
- `experiments/fusion/realiad_d3_fusion_heldout_confirmation.json`
- Protocol (pre-registered): `research_lock/REALIAD_D3_NATDEG_HELDOUT_PROTOCOL_v1.yaml`
- Method: `run_realiad_d3_fusion_test_a_v2.py --fixed-rule rel_x_sharp`
