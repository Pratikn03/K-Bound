# Claims Calibration Ledger — ELARA-U / K-Bound

Maps every headline claim to its evidence status and the exact artifact that
closes (or would close) it. **No number in this table is invented** — every
"CLOSED"/"STRENGTHENED" value is copied from a result JSON in this repo (path in
the Evidence column). Numbers I could not cheaply re-read at authoring time are
marked `verify`.

Status legend:
- **CLOSED** — claim is supported by a committed, real result artifact.
- **STRENGTHENED** — a stronger version than the original paper claim is now supported.
- **PENDING-GPU** — waiting on one of the two in-flight GPU runs (see `RESULTS_PENDING.md`).
- **OPEN** — stated as a conjecture / not yet established.

Git commit at authoring: `eeb04ca6a85a7bb7a023dce146f0b926d49346a5`
Last updated: 2026-06-19

---

## A. ELARA-U stacking (label-free unsupervised model selection)

| Claim | Status | Evidence (file / run) | Note |
|---|---|---|---|
| Stack > auto-select by **+0.036** AUROC (dev, 123 tasks) | CLOSED | `experiments/elara_u/statistical_audit.json` → `family_A_primary_positive[0]` | Real Δ=**0.03551**, 95% CI **[0.02333, 0.04952]**, Holm-p **2.16e-07**, reject=true. Rounds to +0.036. |
| Stack > best-fixed by **+0.075** AUROC (dev, 123 tasks) | CLOSED | `experiments/elara_u/statistical_audit.json` → `family_A_primary_positive[1]` | Real Δ=**0.07501**, 95% CI **[0.05201, 0.10109]**, Holm-p **6.24e-09**, reject=true. Best fixed = `fixed/KNN`. |
| Auto-select > best-fixed (dev) | CLOSED | `experiments/elara_u/statistical_audit.json` → `family_A_primary_positive[2]` | Δ=**0.03951**, 95% CI [0.02330, 0.06056], Holm-p 3.24e-05, reject=true (supporting, not headline). |
| Stack mean AUROC / mean regret (dev, 123 tasks) | CLOSED | `experiments/elara_u/honest_benchmark.json` → `mean_auroc.stack`, `mean_regret.stack` | mean AUROC **0.78356**, mean regret **−0.02743** (beats oracle-relative baseline). Avg rank 2.533 (best of 11). |
| **Sealed** external hold-out: Stack > auto-select **+0.016** | CLOSED | `experiments/elara_u/sealed_external_results.json` → `stack_vs_auto_select` | Real mean **0.0161**, 95% CI **[0.0039, 0.0319]**, pass=true. 74 tasks, frozen ViT/RoBERTa extractors, one-shot. |
| Sealed external: Stack > best-fixed (+0.043) | CLOSED | `experiments/elara_u/sealed_external_results.json` → `stack_vs_best_fixed` | mean **0.0426**, 95% CI [0.022, 0.0729], pass=true. Stack AUROC 0.811 vs KNN 0.7684. |
| **Independent** external (sklearn+HAR): Stack > auto-select **+0.111** | CLOSED | `experiments/elara_u/indep_external_results.json` → `stack_vs_auto_select` | Real mean **0.1106**, 95% CI **[0.0519, 0.1755]**, pass=true. 20 tasks, sources verifiably absent from dev archive, one-shot. |
| Independent external: Stack > best-fixed (+0.209) | CLOSED | `experiments/elara_u/indep_external_results.json` → `stack_vs_best_fixed` | mean **0.2087**, 95% CI [0.1196, 0.3034], pass=true. Stack AUROC 0.7992 vs COPOD 0.5904. |
| Reliability gate **hurts** on i.i.d. (honest negative) | CLOSED | `experiments/elara_u/statistical_audit.json` → `family_B_reliability_ablation` ("stack reliability gate") | Δ=−0.03741, Holm-p 4.96e-09, `reliability_hurts=true`. Reported as a limitation, not buried. |
| Reliability gate **helps** under multimodal val-shift (D23) | CLOSED | `experiments/elara_u/statistical_audit.json` → `family_B_reliability_ablation` (4 multimodal rows) | All 4 reject after Holm (e.g. MVTec-3D Δ=0.2112, Holm-p 1.15e-37). `any_regime_reliability_helps_after_holm=true`. |

> Headline summary string the paper uses — **+0.036 / +0.075 / sealed +0.016 / independent +0.111** — is fully backed by the four JSONs above (rounded from 0.03551 / 0.07501 / 0.0161 / 0.1106).

---

## B. K-Bound theory (theorems + the new finite-n proposition)

| Claim | Status | Evidence (file / run) | Note |
|---|---|---|---|
| **Thm 1** (Non-identifiability / Le Cam regret floor) | CLOSED | `experiments/kbound/theory_validation/results_thm1_lecam.json` | Le Cam TV floor tracks empirical inf_M (e.g. n=64, floor 0.0455 vs inf_M 0.0452). |
| **NEW `prop:lecam-finite`** — finite-n two-point Le Cam LOWER bound on any label-free gate | STRENGTHENED | `experiments/kbound/theory_validation/results_thm2_lecam_finite_n.json` | Floor = Λ·Φ(−c), **constant in n** (`floor_constant_in_n=true`); `bh_certifies_all=true`, `bayes_meets_floor_all=true`, `no_rule_beats_floor_all=true`, `all_ok=true`. 16 cells (c∈{.25,.5,1,1.5} × n∈{10,50,200,1000}), 60k batches each. Upgrades Thm 1 from asymptotic to finite-sample. |
| **Thm 2** (Adapt/freeze/abstain certificate / anytime e-process) | CLOSED | `experiments/kbound/theory_validation/results_thm3_evalue_alpha005.json` → `verdict` | Real `worst_case_false_adapt_h0`=**0.0316**, α=0.05, `controlled=true`. e-process supermartingale check `all_leq_one_within_3sem=true` across 3 streams. Power→1 as δ grows. |
| **Thm 4** (One-bit dichotomy / multiclass/regression swap) | CLOSED | `experiments/kbound/theory_validation/results_thm5_multiclass.json` | `multiclass.equality_ok=true` (max_abs_err 1.11e-16); regression identity max err 5.6e-13; `all_ok=true`. Honest caveat: `regression_shift.con` sign-cert is `cert_correct=false`. |
| **Thm 2 (regret decomposition)** (regret = Σ over abstained-correct, base decomposition) | CLOSED | `experiments/kbound/theory_validation/results_thm2_regret.json` | Underpins the gate regret; same artifact, `exact_identity_holds=true`. Exact identity max gap **2.35e-17**; minimax ratio-to-floor **1.0004**. |
| **Thm 2 (e-value)** (anytime-valid wealth process validity) | CLOSED | `experiments/kbound/theory_validation/results_thm3_evalue_alpha005.json` → `supermartingale_check` | E[wealth] ≤ 1 within 3·SEM at all horizons for twopoint/beta/clipgauss. |
| **5 K-Bound theorems all machine-checked** | CLOSED | the four `results_thm*.json` above | Thm 1 (results_thm1_lecam.json, results_thm2_lecam_finite_n.json); Thm 2 (results_thm2_regret.json, results_thm3_evalue_alpha005.json); Thm 3 (results_frontier.json); Thm 4 (results_knowability_dichotomy.json, results_thm5_multiclass.json); Thm 5 (results_knowability_capacity.json). |

---

## C. Natural-shift empirical items (real TTA backbones)

| Claim | Status | Evidence (file / run) | Note |
|---|---|---|---|
| CIFAR-10-C stress grid: KGA gate beats both trivial policies (multi-seed, Holm) | CLOSED | `experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` | Holm-corrected paired bootstrap (5 seeds, 432 conditions) shows KGA beats both always-adapt and always-freeze for tent and eata (both survive Holm: tent vs adapt/freeze regret diffs -0.0063 / -0.1225; eata vs adapt/freeze -0.0020 / -0.1301). KGA beats freeze but not adapt for sar (+0.0012, does not survive). |
| **ImageNet-R** multi-seed: gate vs always-adapt / always-freeze across 10 backbones | CLOSED | `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` | 10 backbones, 3 seeds. KGA matches or beats the better trivial policy in all cases, surviving Holm in 9/20 comparisons (e.g. efficientnet_b0 vs always-adapt -0.0501 CI [-0.0569, -0.0438]; convnext_base vs always-freeze -0.0683 CI [-0.0733, -0.0628]). No beats-both win. |
| **Camelyon17** full-scale (incl. **SAR**): gate vs trivial policies | CLOSED | `experiments/kbound/results/camelyon17_fullscale_B_v2/MULTISEED_ANALYSIS_RESULTS.json` | 3 seeds, includes SAR. KGA cuts freeze-regret massively (tent: -0.0524, eata: -0.0827, sar: -0.0995, all Holm-significant), but does not beat both (ties always-adapt for eata and sar, slightly trails always-adapt for tent +0.0199). No beats-both win. |
| Pipeline (per-condition serialize → KGA decision → paired bootstrap+Holm) is torch-free and correct | CLOSED | `experiments/kbound/results/_pipeline_smoke_verify/VERIFY_RUNNER_PIPELINE_REPORT.SYNTHETIC.json` (`ALL_ASSERTIONS_PASSED=true`) | **Synthetic** scores only — verifies plumbing, not effect sizes. Re-confirmed end-to-end through `scripts/foldin_multiseed_results.py`. |

---

## D. Conjecture (empirical p\*-law)

| Claim | Status | Evidence (file / run) | Note |
|---|---|---|---|
| **p\*-law conjecture** (p\* regime law: a single harmful-fraction threshold separates "gate beats both" from not) | OPEN | partial signal via `pstar_law.monotone_separable_by_single_threshold` field emitted by `wilds/multiseed_paired_ci.py`; CIFAR grid only | Stated as a conjecture. The two pending runs add ImageNet-R + Camelyon17 points to the separability check but do **not** prove it. Remains OPEN. |

> **Naming disambiguation (integrity pass 2026-06-20).** This empirical **p\*-law conjecture** (regime-separability: a single harmful-fraction threshold separates "gate beats both" from not) was previously labelled "Conjecture 1" here, which **collided** with the paper's distinct **Conjecture 1** — the *label-free benefit-sign bracketing* problem (`\label{conj:gen}` in `paper/sections/main_theory_5.tex`). They are two different open problems; the empirical one is renamed to **p\*-law conjecture** everywhere it appears in this ledger to remove the collision. Naming only — no statement, threshold, or math was changed.

