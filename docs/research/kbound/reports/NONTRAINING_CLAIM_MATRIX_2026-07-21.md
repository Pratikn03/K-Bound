# K-Bound Reviewer Claim Matrix (non-training closure)

Generated: 2026-07-21T16:54:35.164744+00:00
Source: `docs/research/kbound/claim_ledger.json` (schema `kbound-claim-ledger-v1`)

Scope note: PACS and ImageNet-R Protocol-D multi-seed remain **pending** the active run and are not asserted here. This matrix is a reviewer-facing projection of the authoritative claim ledger; wording columns are verbatim from the ledger.

**Status totals:** {'supported': 20, 'withdrawn': 5, 'no-harm': 2, 'pending': 1} (28 claims).

| Claim | Status | Tier | Dataset | Protocol | Calibration | Allowed wording | Forbidden wording | Key artifact |
|---|---|---|---|---|---|---|---|---|
| KB-CLAIM-001 | supported | A | synthetic validators | theory_validation | n/a | iff identifiability on declared drift class | assumption-free ; universal | experiments/kbound/theory_validation/results_thm2_regret.json |
| KB-CLAIM-002 | supported | A | witness | theory_validation | n/a | information-theoretically necessary abstention | engineering gap only | docs/research/kbound/results/witness/witness_clean.json |
| KB-CLAIM-003 | supported | A | n/a | certificate | split_conformal_quantile | FA_u <= alpha under stated assumptions | FA_c <= alpha ; guaranteed safe in the wild ; assumption-free | docs/research/kbound/kbound_pkg/kbound/certificate.py |
| KB-CLAIM-004 | withdrawn | A | n/a | n/a | n/a | report FA_c descriptively only | FA_c <= alpha guaranteed | — |
| KB-CLAIM-010 | supported | B | CIFAR-10-C stress grid | STRESS_GRID_MULTISEED_PROTOCOL_A_v1 | loo_gbr + conformal_on_loo_residuals | beats both under Protocol A v1 stress grid | universal TTA improvement ; natural-shift win | experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json |
| KB-CLAIM-011 | supported | B | ImageNet-C | imagenetc_protocol_E_v1 | loo_gbr + conformal | beats both on harmful SAR operating point | all ImageNet-C candidates | experiments/kbound/results/imagenetc_official_sar_E_v1/ |
| KB-CLAIM-012 | withdrawn | B | CIFAR-10-C | STRESS_GRID_MULTISEED_PROTOCOL_A_v1 | loo_gbr | split/LOO conformal empirical certificate; assumptions in guarantee box | jackknife+ ; distribution-free without assumptions | docs/research/kbound/scripts/cifar_tent_mps_v2.py |
| KB-CLAIM-020 | no-harm | B | Office-Home | OFFICEHOME_PROTOCOL_M_v2 | oof_conformal | uniformly no-harm under stated held-out protocol | beats both ; natural-shift win | experiments/kbound/results/officehome_protocol_M_v2/ |
| KB-CLAIM-021 | no-harm | B | iWildCam | IWILDCAM_PROTOCOL_H_v2 | oof_conformal | no-harm; damage-prevention | beats both | experiments/kbound/results/iwildcam_protocol_H_v2/ |
| KB-CLAIM-022 | withdrawn | B | Camelyon17 | CAMELYON17_PROTOCOL_G_v1 | in_sample_radius | withdrawn pooling artifact; genuine OOD no-harm | beats both Camelyon17 | archive/audit_only/camelyon17_protocol_G_pooled_beats_both |
| KB-CLAIM-023 | withdrawn | B | Office-Home+iWildCam+Camelyon | KBOUND_MIXED_STREAM_v1 | in_sample_radius (invalid) | withdrawn in-sample-radius artifact; superseded by mixed_protocol_oof_v2 | 13x ; 24x ; beats both mixed | — |
| KB-CLAIM-024 | supported | B | Office-Home+iWildCam+Camelyon OOD | mixed_protocol_oof_v2 | per_dataset_dev_lock_loo_conformal | beats both on constructed cross-protocol aggregate only | universal mixed deployment ; natural-shift win ; 13x ; 24x | research_lock/KBOUND_MIXED_STREAM_v2.json |
| KB-CLAIM-025 | supported | A | synthetic validators | theory_validation | n/a | resolved negatively; impossibility not open problem | conj:gen open ; universal label-free bracketing exists | docs/research/kbound/paper/sections/main_theory_5.tex |
| KB-CLAIM-026 | supported | B | CIFAR-10-C stress grid (Tent) | MIXED_HEADTOHEAD_PROTOCOL_v1 | loo_gbr + conformal (KGA); faithful POEM/AETTA ports | beats POEM and AETTA on mixed regret (pre-registered WIN) | beats POEM on natural shifts ; official-repo arm without labeling | experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json |
| KB-CLAIM-030 | pending | C | physical packages P01-P10 | edge_real_phone_v1 | session_split_conformal | pre-registered protocol only; feasibility/no-harm if helpful-dominated | real-world win ; deployment success ; 25% balanced acc as headline | docs/experiments/kbound/results/edge_real_phone_v1/ |
| KB-CLAIM-027 | supported | B | MNIST two-view controlled | CONTROLLED_MULTIMODAL_PROTOCOL_D33_v1 | loo_conformal | mechanism confirmation on controlled multimodal harm | natural multimodal SOTA ; universal fusion win | research_lock/CONTROLLED_MULTIMODAL_PROTOCOL_D33_v1.yaml |
| KB-CLAIM-040 | supported | C | stress + deployment slices | assumption_audit_v1 | n/a | audit triggered warnings under designed shifts; does not verify exchangeability | verified exchangeability ; proved risk alignment | docs/research/kbound/results/assumption_audit_v1.json |
| KB-CLAIM-028 | supported | A | synthetic | theory_validation | n/a | weakest one-bit class under GP | unique weakest class unconditionally | experiments/kbound/theory_validation/results_conj1_genpos.json |
| KB-CLAIM-029 | supported | A | synthetic enumeration | theory_validation | n/a | explicit finite family W*; GP as collapsing face | unique weakest class | docs/research/kbound/theory_v2/unconditional_weakest_results.json |
| KB-CLAIM-031 | supported | A | synthetic stream | theory_v2 | e-process | time-uniform FA_u extension; multiclass anytime FWER closed (Wave 4) | multiclass anytime open | docs/research/kbound/theory_v2/val_sequential_anytime_results.json |
| KB-CLAIM-032 | supported | A | synthetic K candidates | theory_v2 | Bonferroni per-candidate | family-wise FA <= alpha under Bonferroni selection-proof | naive argmax L_k has FA <= alpha ; multiclass routing open | docs/research/kbound/theory_v2/val_multicandidate_results.json |
| KB-CLAIM-033 | supported | A | analytic Gaussian validators | theory_v2 | n/a | kappa(alpha) is exact; 4 n_opt is pairwise floor only | tight constants open ; kappa/4 still open | docs/research/kbound/theory_v2/tight_constants_closure.tex |
| KB-CLAIM-034 | supported | A | synthetic multiclass | theory_v2 | Bonferroni per-candidate LCB | FWER <= alpha for multiclass routing on D | multiclass certs not attempted | docs/research/kbound/theory_v2/multiclass_multicandidate_theorem.tex |
| KB-CLAIM-035 | supported | A | synthetic K>=3 | theory_v2 | n/a | closed negatively; per-component orientation minimal supplement | general multiclass capacity exists ; conj:gen-capacity open | docs/research/kbound/theory_v2/multiclass_capacity_impossibility.tex |
| KB-CLAIM-036 | supported | A | structural validators | theory_v2 | n/a | dichotomy not unconditional computability | conj:dich-compute open ; margin always computable | docs/research/kbound/theory_v2/margin_computability_closure.tex |
| KB-CLAIM-037 | supported | A | regression synthetic | theory_v2 | n/a | complete iff under bounded drift; no universal bracket | regression bracketing open ; universal drift bracket | docs/research/kbound/theory_v2/regression_bracketing_closure.tex |
| KB-CLAIM-038 | supported | A | n/a | formal_audit | n/a | closure plan complete within stated scopes | 100% full Mathlib measure theory ; all open problems closed | docs/research/kbound/THEORY_100_PERCENT_CLOSURE_PLAN.md |
| KB-CLAIM-050 | withdrawn | A | all | n/a | n/a | conditional insurance against detectable harm | universal improvement ; always beats adapt | kbound_short.tex sec:limits |

## Withdrawn / boundary claims (must NOT be restored or promoted)

- **KB-CLAIM-004** — Conditional false-adapt FA_c is bounded by alpha.
  Forbidden: _FA_c <= alpha guaranteed_
- **KB-CLAIM-012** — Finite-sample distribution-free jackknife+ guarantee for stress-grid calibration.
  Forbidden: _jackknife+ ; distribution-free without assumptions_
- **KB-CLAIM-022** — Camelyon17 Protocol G pooled beats-both headline.
  Forbidden: _beats both Camelyon17_
- **KB-CLAIM-023** — Cross-protocol mixed aggregate beats both always-adapt and always-freeze.
  Forbidden: _13x ; 24x ; beats both mixed_
- **KB-CLAIM-050** — Universal accuracy improvement from KGA.
  Forbidden: _universal improvement ; always beats adapt_

## No-harm claims (must use no-harm wording, never beats-both)

- **KB-CLAIM-020** (Office-Home) — Office-Home Protocol M v2: KGA is no-harm OOF (beats always-adapt, ties always-freeze).
- **KB-CLAIM-021** (iWildCam) — iWildCam Protocol H v2: KGA is no-harm OOF.
