# D18 Real-IAD D3 Held-Out Confirmation — Audit (2026-06-02)

## Verdict

The D18 one-shot held-out run **literally meets** its pooled pass condition but
**does not survive artifact audit**. It must **not** be recorded as
`gate_natdeg_heldout_confirmed = true`. The pooled CI-significant result is
inflated by a single degenerate point-cloud channel (`lego_propeller`, XYZ
validation AUROC = exactly 0.000). The genuine, defensible finding is a
within-category reliability-gating win on 3 of 11 held-out categories where a
modality is honestly (non-degenerately) weak.

Authority: `research_lock/REALIAD_D3_NATDEG_HELDOUT_PROTOCOL_v1.yaml` (D18),
`research_lock/DECISIONS_v1.md` (D17), `research_lock/SCENARIO_C_CLAIM_CONTRACT.md`.

## Run under audit

- Script: `src/scripts/scenario_c/run_realiad_d3_fusion_test_a_v2.py`
- Rule: `rel_x_sharp` (frozen, selected on 8 development categories, validation
  AUROC only).
- Output: `experiments/fusion/realiad_d3_fusion_heldout_confirmation.json`
  (self-labeled `holdout_status: OPENED_DEVELOPMENT_ONLY`,
  `protocol: FUSION_TEST_A_v2_development` — so no gate was auto-flipped).
- Detectors: deep PatchCore (ResNet-50 layer2+layer3) for rgb/ps; PCA
  surface-relief geometry from the PCD → deep PatchCore for xyz.

## Pooled (literal pass condition)

| Metric | Value |
|---|---|
| pooled AUROC CW | 0.7001 |
| pooled AUROC gated (`rel_x_sharp`) | 0.7455 |
| pooled Δ vs CW | **+0.0454** |
| 95% CI | [0.0225, 0.0683] |
| CI low > 0 | true |
| D18 pass condition (Δ ≥ 0.010 ∧ CI low > 0) | **met on paper** |

## Per-category decomposition (the real story)

| Category | weak modality (val AUROC) | CW | gated | Δ | read |
|---|---|---:|---:|---:|---|
| connector_housing_female | rgb 0.414 | 0.758 | 0.967 | +0.209 | genuine gating win |
| lego_pin_connector_plate | xyz 0.375 | 0.843 | 0.945 | +0.102 | genuine gating win |
| lattice_block_plug | xyz 0.611 | 0.741 | 0.831 | +0.090 | genuine gating win |
| purple_clay_pot | — | 0.745 | 0.761 | +0.016 | tie |
| telephone_spring_switch | — | 0.943 | 0.954 | +0.011 | tie |
| crimp_st_cable_mount_box | — | 0.669 | 0.674 | +0.005 | tie |
| limit_switch | — | 0.994 | 0.995 | +0.002 | tie |
| miniature_lifting_motor | — | 0.714 | 0.715 | +0.002 | tie |
| headphone_jack_socket | all 1.000 | 1.000 | 1.000 | 0.000 | degenerate (trivial) |
| fork_crimp_terminal | all ~chance | 0.835 | 0.738 | −0.098 | loss |
| **lego_propeller** | **xyz 0.000** | **0.0175** | 1.000 | **+0.9825** | **artifact** |

Within-category mean Δ:

- all 11: **+0.120** (dominated by lego_propeller)
- excluding lego_propeller: **+0.034** (n=10)
- excluding the 2 degenerate categories: **+0.038** (n=9)

## The artifact: lego_propeller

- XYZ **validation** AUROC = exactly **0.000**; test CW = **0.0175** — a near-perfect
  sign inversion. A real detector essentially never reaches exactly 0.0 on 70
  validation samples; this is the signature of a degenerate / label-correlated
  constant (or polarity-inverted) channel.
- This is the **same failure class as F1** in the D18 protocol amendment (binary
  PCD → degenerate XYZ fallback), which already caused the first held-out run to
  be killed. The `_v2_binpcd` cache fix **was applied** to lego_propeller
  (cache present) but **did not** rescue it — the XYZ point-cloud path still
  produces a degenerate channel for this category.
- A +0.98 within-category swing on 115 test samples dominates the pooled metric;
  the pooled +0.0454 / CI>0 is therefore not a clean confirmation.

## What is genuinely real

On the 3 categories where one modality is honestly weak but not degenerate —
connector_housing_female (rgb 0.41), lego_pin_connector_plate (xyz 0.375),
lattice_block_plug (xyz 0.61) — reliability gating delivers **+0.09 to +0.21**
within-category. This is the T1/T3 stress-regime mechanism behaving as
theorized, on a pre-registered held-out set. Defensible claim:

> On a pre-registered 11-category Real-IAD-D3 held-out set with deep-PatchCore
> detectors, reliability-gated fusion shows genuine within-category gains
> (+0.09 to +0.21) on categories with an honestly weak modality, ties on 6, and
> loses on 1; the pooled CI-significant result is inflated by one degenerate XYZ
> channel and is **not** a clean natural-degradation confirmation.

## Integrity constraints honored

- `gate_natdeg_heldout_confirmed` is **left false**.
- lego_propeller is **not** dropped-and-recomputed into a clean "+0.034 pass":
  post-hoc category removal after seeing test scores would violate D18 ONE_SHOT
  and is itself selection.
- Strict clean Gate E remains `CLOSED_BY_PROOF_T9`; this audit does not touch it.

## Guarded development re-analysis (the real punchline)

A degenerate-channel guard was implemented
(`src/elara/evaluation/degenerate_channel_guard.py`, validation-only, tested in
`tests/test_degenerate_channel_guard.py`) and applied **fairly to both the CW
comparator and the gated rule** on the opened cache (development only,
`src/scripts/scenario_c/guarded_channel_dev_analysis.py` →
`experiments/fusion/guarded_channel_dev_analysis.json`):

| | pooled CW | pooled gated | Δ | within-cat mean Δ |
|---|---:|---:|---:|---:|
| Unguarded (D18 as-run) | 0.7001 | 0.7455 | **+0.0454** | +0.120 |
| Guarded (both sides) | 0.7298 | 0.7347 | **+0.0049** | +0.011 |

Findings:

- The unguarded +0.045 "win" was **almost entirely the degenerate-channel
  artifact**. Guarding both sides shrinks the pooled gating advantage to +0.005
  — **below the +0.010 pass bar** — and the within-category median Δ is 0.000.
- The guard's dominant effect is to make **CW itself robust** (pooled
  0.700 → 0.730) by refusing to trust inverted/saturated channels. That is a
  reusable robustness contribution in its own right.
- Reliability gating's *residual* advantage over a guarded CW survives only in
  the **3–4 categories with ≥2 genuinely reliable channels** (lattice +0.095).
  When a category has only one non-degenerate channel, CW and gated coincide —
  there is nothing to gate.

Implication: the headroom for the gating mechanism is bounded by **detector
quality**. The reason gating has little to do is that most categories currently
expose only one non-degenerate modality. Better detectors (more genuinely
reliable channels per category) are what give the gate something to arbitrate —
this is the real target of Lever 1, not just artifact removal.

## Required next step (legitimate path to a clean confirmation)

1. **Improve detector quality so categories expose ≥2 non-degenerate channels**
   (Lever 1) — this is where the gating mechanism has provable headroom (T1/T3).
   The degenerate-channel guard is now the floor: fold it into the frozen method
   so neither CW nor the gate can be fooled by inverted/saturated channels.
2. **Re-seal a fresh held-out set** (new prelock) with the guarded, improved
   detector frozen beforehand, and run ONE_SHOT. The current 11 categories are
   opened by this run and cannot be reused as a fresh official confirmation.
3. **Set expectations honestly:** on current detectors the guarded development
   delta is ~+0.005, so a fresh holdout would likely *not* clear +0.010 until
   detector quality improves. Do not re-seal until step 1 moves the development
   estimate clear of the bar.
