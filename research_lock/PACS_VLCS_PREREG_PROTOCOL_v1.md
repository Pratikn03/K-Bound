# Pre-registered protocol — PACS & VLCS domain-shift confirmations (item 2, v1)

*Locked 2026-06-22, BEFORE any result. Two additional independent natural/domain-shift datasets to
test whether KGA's adapt/freeze/abstain certificate beats both fixed policies (or correctly
abstains/prevents damage), in the SAME domain-shift family as Office-Home (where KGA already wins
CI-robustly). Pre-registration makes the eventual result credible regardless of outcome: a null is
reported as a null, with no re-tuning.*

## Why these two
PACS and VLCS are the standard DomainBed domain-generalization benchmarks — the closest public
analogues to Office-Home (art/clipart/product/real), where KGA is already a CI-robust beats-both.
If the certificate generalizes, this is where it should; if it only prevents damage, that is an
honest no-harm result. Either way it is an independent real-domain-shift data point.

## Datasets (locked)
- **PACS**: domains {Photo, Art, Sketch, Cartoon}; 7 classes. Leave-one-domain-out: ERM-train on 3
  source domains, deploy on the held-out domain. 4 deployment domains.
- **VLCS**: domains {VOC2007, LabelMe, Caltech101, SUN09}; 5 classes. Leave-one-domain-out. 4
  deployment domains.

## Model & adapters (locked)
- Backbone: ResNet-18 (and ResNet-50 robustness check), ImageNet-pretrained, ERM-trained on the
  source domains with the standard DomainBed recipe.
- Candidate adapter pool = the SAME as Office-Home Protocol M v2: {tent, eata, sar} online,
  BN/LN-affine params only, shared (steps, lr) budget. **Dev-lock exactly ONE adapter per dataset**
  on the source-validation split before any test contact (as in Protocol M v2).

## Decision protocol (canonical KGA only — no diagnostics, no multicandidate)
- Evidence Z: the same 11-dim label-free vector as the locked runner
  (pre/post entropy, pre/post confidence, pre/post prediction-balance, balance-drop, entropy-drop,
  frac-highconf, marginal-KL, update-norm).
- Benefit model: GBR Δ̂(Z) calibrated **leave-one-domain-out** (task = domain). Conformal radius
  ε = (1−α) quantile of |Δ̂−Δ| residuals on the calibration domains. α = 0.10.
- Decision: ADAPT if Δ̂−ε>0, FREEZE if Δ̂+ε<0, else ABSTAIN. No held-out-domain labels touch the
  adapter, ε, the evidence map, or the rule.

## Splits (locked)
- For each dataset and each held-out deployment domain: calibration = the other 3 domains
  (leave-one-domain-out); test = the held-out domain, scored ONCE. Seeds {0,1,2}.

## Pre-registered endpoints & decision rule
- Primary metric: regret-to-oracle of KGA vs always-adapt and vs always-freeze on the held-out domain.
- Secondary: FA_u (unconditional false-adapt P[adapt ∧ Δ≤0]), FA_c, coverage, adapt-rate.
- **WIN (beats-both, CI-robust)** iff BOTH regret-gap 95% bootstrap CIs (B=3000, dev calibration
  held fixed) exclude zero AND FA_u ≤ α.
- **SAFETY (damage-prevention / no-harm)** iff KGA ties the better fixed policy (gap CI includes 0)
  at FA_u ≤ α — reported as no-harm, NOT a win.
- **NULL** iff FA_u > α or KGA loses to a fixed policy — reported as a null.

## Integrity rules (locked — sealed)
- No recalibration on the held-out domain; no adapter re-selection after seeing test; no ε re-tuning.
- A null is published as a null. The decision rule, ε, and evidence map are frozen at calibration.
- Per-domain (Z, Δ, frozen-acc, adapted-acc) are dumped so the gate-baseline comparison
  (`scripts/gate_baseline_comparison.py`) runs on the same cells.

## Compute
- Any CUDA box (Kaggle T4 / lab / NAIRR). PACS (~10k images) and VLCS (~10k) are small; a full
  8-domain × 3-seed sweep is a few GPU-hours. MPS will also work here (small backbones, unlike
  ImageNet-C), so this can run on the Mac if no CUDA is available — but CUDA is cleaner.

## Reporting (locked template)
For each dataset, one row in the natural-shift table: held-out domain, locked adapter, regret
(KGA / adapt / freeze), both gap CIs, FA_u, and the verdict (WIN / SAFETY / NULL). No aggregation
across datasets is claimed as a single deployment (per the cross-protocol-aggregate framing).
