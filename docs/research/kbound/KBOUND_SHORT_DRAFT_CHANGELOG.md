# K-Bound Short — Draft Revision Changelog

Baseline: pre-revision snapshot `~/kbound_phase1_backup_20260711_014958`.
Unified diff: `kbound_short_draft.diff` (5 files, 15 hunks, +91/−45).
Build: `latexmk` exit 0, **0 undefined references, 0 fatal errors, 20 pages**
(`kbound_short_draft.pdf`). DRAFT TODOs are **visible** in this build (one-line toggle at
`kbound_short.tex:31` hides them for a clean build).

Files touched: `kbound_short.tex`, `paper/sections/theory_setup.tex`,
`paper/sections/theory_core_main.tex`, `paper/sections/main_theory_5.tex`,
`paper/references_kbound_expanded.tex`. No raw result file or original evidence was deleted.

## Phase 1 — Theory corrections (2 genuine bugs)
1. **`η_a` redefinition** (`theory_setup.tex`, Assumption `ass:deploy`). Was the class-conditional
   `Pr_T(Y=1 | f_a=1)` — the exact form flagged in the brief §7.2 — under which the reduction
   `Δ=2μ(D)(ā−½)=sign(M+γ)` does not follow. Now the **pointwise correctness**
   `η_a(x):=Pr_T(f_a(X)=Y | X=x)`, with `s` a calibrated estimate of it, so `E[η_a|D]=ā` and the
   identity holds. Propagates to both papers.
2. **`|M|=β` boundary** (`theory_core_main.tex` `lem:nonid` + `thm:headline`(iii), statements and
   proofs; mirrored in `main_theory_5.tex` `thm:frontier`(ii)). The opposite-*nonzero*-sign /
   minimax-½ witness was asserted on the closed `|M|≤β`, but at `M=β` the sign is `+` or `0`, never
   `−`. Restricted the strict witness to `|M|<β`; documented that at `|M|=β`, `γ=∓β` forces `Δ=0`, so
   abstention on the closed `|M|≤β` remains maximal-sound. Headline IFF (`|M|>β`) unchanged.

   Full detail: `KBOUND_THEORY_AUDIT.md`.

## Phase 2 — Method reproducibility (from the actual code)
3. **Evidence map rewritten** (`§Method`, `Evidence Map and Benefit Estimator`). Replaced the abstract
   7-family sketch (which implied one panel and listed non-existent features) with the **two real
   panels**: the 11-dim base panel for stress grids (frozen/adapted entropy, confidence, marginal
   entropy, entropy/balance drops, frac-conf>0.9, marginal-KL, parameter-update ℓ2) and the 16-dim
   natural-shift panel (adds entropy/conf quantiles, disagreement rate, energy shift, BN-stat KL,
   source-thresholded ATC, confidence drop). Tied the theory score `s` to the implemented
   adapted-confidence/ATC features.
4. **Benefit estimator specified**: scikit-learn `GradientBoostingRegressor`, squared error, 250
   trees, depth 2, lr 0.05, subsample 0.8, seed 0, **refit per dataset and per candidate adapter**
   (no cross-dataset transfer implied). Fills the old `DRAFT TODO` at former line 328.
5. **Evidence table (`tab:evidence-map`) corrected** to list only statistics that are actually
   computed; removed fabricated entries (logit movement, predictive JS, gradient norm, accepted
   samples, reset signal, batch size).
6. **Config table added (`tab:config-table`)**: per-track calibration unit, radius rule
   (LOO-jackknife on grids; split-conformal on natural shift), and test unit; α=0.10 throughout;
   backbones noted (ResNet-18 CIFAR, ResNet-50 ImageNet-C). Replaces the old config `DRAFT TODO`;
   a residual TODO remains for per-adapter LR/steps.

## Phase 4 — Results reconciliation (verified against source files)
7. CIFAR-10-C SAR harmful base rate **16% → ≈10%** (per-seed 0.116/0.102/0.088/0.074/0.100).
8. Head-to-head harmful fraction **16–18% → ≈33%**, made explicit as the always-adapt FA_u of
   `tab:headtohead-poem-aetta` (internal-consistency fix).
9. PACS uniform-panel row **18 → 108 cells/target**; FA_u split into three safety targets (0) and
   the `photo` **null (FA_u=0.056)** — per `pacs_result.json`.
10. Three-source `0.0059` row → "constructed mixture; **OOF replay pending**" + caption DRAFT TODO
    (raw artifact offline; only withdrawn in-sample 0.0026 on disk).
11. Streaming demo → specific figures (35,370 / 45 windows / 75.6% / gate 0.6995) **removed from the
    headline**; qualitative anytime-FA=0 claim kept with a replay DRAFT TODO.
12. CIFAR-10.1 diagnostic row tier → "diagnostic; **replay pending**" (only FA_u=0.444 traceable).

   Full detail incl. clean matches and leakage audit: `KBOUND_RESULT_AUDIT.md`.

## Phase 5 — References + build
13. Removed the one true duplicate bib entry `vovk2005algorithmic` (identical to `vovk2005`; both
    uncited). No cited key was touched; 0 missing citations. Remaining reference cleanups are listed
    in `KBOUND_REMAINING_TODOS.md`.
14. Re-enabled **visible DRAFT TODOs** for the draft build (`kbound_short.tex:31` commented out).

## Phase 3 — Structure
The main text already realizes the required 12-section layout (Introduction, Related Work, Problem
Formulation & Validity, Theory, Method, Experimental Setup, Results, Ablations, Discussion,
Limitations, Reproducibility, Conclusion) plus appendix; no risky wholesale renumber was performed.
Structural gaps were filled in place (method spec, config table) rather than by reorganizing.

## Phase 6 — Ablations + cost + deployment semantics (no T9; from logged data)
Run entirely off the in-repo logged per-condition evidence (`per_condition_cifar10c_*_seed0.json`)
and the `resnet18_cifar.pt` checkpoint — no target labels, no model re-runs, no T9.
- **§Ablations rewritten** from "planned, not yet run" to real results (`scripts/ablation_sweep.py`;
  8-fold cross-fit reproducing the deployed leave-one-cell-out rule — anchor: Tent α=0.10 → regret
  0.0017, FA_u 0, matching the locked gate table). Four ablations + tables:
  (i) α-sweep `tab:abl-alpha` — FA_u=0 at every certified α; radius-free breaks it (0.051/0.032/0.007).
  (ii) estimator `tab:abl-estimator` — GBR 0.0017 / RF 0.0015 / ridge 0.0041 / MLP 0.0143, all FA_u≤0.002.
  (iii) feature-family dropout — regret 0.0013–0.0019, FA_u=0 (no single family load-bearing).
  (iv) cross-adapter transfer `tab:abl-transfer` — breaks FA_u (SAR→Tent 0.26), justifying per-adapter refit.
  (v) β-sweep = the synthetic frontier of §Synthetic.
- **Cost table `tab:cost`** (`scripts/cost_profile.py`): controller adds 0.20 ms/decision + a 44.8 MB
  rollback copy; benefit model 195 KB. Fills the cost DRAFT TODO.
- **Deployment semantics** filled from `kbound_pkg` (`KGA.decide_from_batch`; `abstain_scale=0` →
  skip update / serve frozen; failure-safe defaults keyed to `tab:failure-modes`). Fills the deployment TODO.
- Build: latexmk exit 0, 0 undefined refs, **21 pages**. New artifacts: `scripts/ablation_sweep.py`,
  `scripts/cost_profile.py`, `experiments/kbound/results/ablation_*.json`, `cost_profile.json`.

## Phase 7 — Closed 2 of 4 DRAFT TODOs (no T9)
Extracted the exact feature formulas and adapter configs from code and wrote them into the appendix.
- **New appendix §`app:evidence-schema`** with three tables: `tab:schema-base` (11-dim base panel,
  per-feature formulas from `kbound_pkg/kbound/evidence.py`), `tab:schema-rich` (16-dim natural-shift
  panel from `run_wilds_camelyon17.py`), and `tab:adapter-hparams` (per-track optimizer/steps/lr:
  mild 10/1e-3, aggressive 50/2.5e-3; EATA filter 0.4·ln2; SAR SAM+reset; backbones ResNet-18 CIFAR,
  ResNet-50 ImageNet-C, ResNet-18 PACS, DenseNet-121 Camelyon17 — all verified from code).
- The two §Method / §Experimental-Setup DRAFT TODOs are replaced with pointers to the appendix.
- **Remaining DRAFT TODOs: 2** (both T9-blocked) — streaming window/gate figures and the three-source
  `0.0059` OOF leg. Build: latexmk exit 0, 0 undefined refs, 21 pages.

## Phase 8 — T9 restore + three-source lock
- Restored non-raw artifacts from T9 (~2 GB, 63k files): result JSONs, `research_lock/` (104 files),
  `audits/`, `mixed_headtohead_v1/`, `mixed_protocol_oof_v2/`, and the long `kbound.tex`/`kbound.pdf`.
  Raw data, logs, caches, and model checkpoints (`*.pt`) excluded.
- **Three-source beats-both LOCKED**: `research_lock/KBOUND_MIXED_STREAM_v2.json` confirms
  regret 0.0059/0.0632/0.0342, FA_u 0, both CIs exclude zero, `beats_both_robust=true`. Removed its DRAFT TODO.
- **Office-Home + iWildCam upgraded provisional → reconciled** (backed by `KBOUND_WIN_BOOTSTRAP_CIS_oof.json`).
- **Streaming**: no saved artifact on T9, logs, or the long paper — DRAFT TODO kept (re-run required).
  This is the **only** remaining DRAFT TODO.
- Long companion `kbound.tex` reviewed; not folded into the short paper (already 21pp; the two are a short/long pair).
- Build: latexmk exit 0, 0 undefined refs, 21 pages. Committed to `main` and pushed to origin.

## Phase 9 — External-review response (math + framing fixes)
Addressed the reviewer's concrete issues (outside the DRAFT-TODO markers):
- **Lemma 1 proof corrected** — the line `ā−½ = E[η_a−s|D] = M+γ` was wrong (that expectation is γ);
  now `M+γ = (E[s|D]−½) + E[η_a−s|D] = E[η_a|D]−½ = ā−½`.
- **Invalid one-bit witness removed** — the weakest-class appendix used `a=(+1,±2)` while `a=2η−1∈[−1,1]`
  (impossible). Guarded the one-bit / dominance-polytope material out of the short paper (deferred to
  `kbound.tex`) and fixed the dangling references.
- **Multiclass Proposition added** (`prop:multiclass`), stating exactly what KGA estimates.
- **Softened overclaims**: stress-grid radius no longer called an exact finite-sample guarantee (empirical
  coverage vs split-conformal/jackknife+); β=0 "face" → "β=0 plug-in under an implicit negligible-drift
  assumption"; "limitations are the frontier" now separates theorem-imposed from empirical; Lean novelty
  softened; "safety is a property of the radius" localized to the tested grid.
- **Theory↔KGA bridge** clarified (intro + Fig 2 caption now population-only `|M|>β`; deployed rule uses
  `Δ̂±ε`, Fig 3); **risk alignment** stated as a class assumption with falsification-only diagnostics.
- **Structure**: removed the duplicated §III setup paragraph; fixed the config-table caption.
- Build: exit 0, 0 undefined refs, **20 pages** (down from 21). Only remaining DRAFT TODO: streaming.

## Phase 10 — Second external-review response (edge-cases, tables, figures)
- **Theorem 1** now requires β>0 (|M|<β is vacuous at β=0), with an explicit β=0 note.
- **Theorem 2** reframed as *strict-commitment* identifiability (adapt/freeze uniformly sound iff |M|>β),
  fixing the M=β=0 edge where Δ≡0 is known exactly.
- **Corollary 1** separates risk alignment from Theorem 3's coverage (interval implication holds under
  coverage alone; risk alignment only supports the population-frontier interpretation).
- **|M|=β boundary** given its own zero-vs-strict-sign abstention argument.
- **Calibration wording**: split-conformal now states the order-statistic radius ε=r_(k),
  k=⌈(n+1)(1−α)⌉ (code uses the interpolated quantile — a close approximation); the "0.898 ≥ 0.90 by
  construction" contradiction fixed to "approximately nominal, empirical not exact"; TV drift is
  "conceptually related to but distinct from" β.
- **Tables**: `tab:decisive` coverage column clarified (both fixed policies commit → always-freeze
  0.00→1.00); `tab:abl-alpha` footnote reconciles 8-fold cross-fit vs the locked LOCO gate; ImageNet-C
  27 cells defined as 3 corruptions × 3 severities × 3 compositions (iid/imbalanced/single-class, verified).
- **Figures**: Figure 7 (`fig_alpha_coverage`) regenerated from `ablation_alpha.json` to match Table XVI
  (coverage 0.49/0.64/0.69/0.74, FA_u=0); Figure 4 relabeled a conceptual schematic (axes illustrative).
- **Wording**: real-data KGA "does not numerically use β" (stated in §Synthetic); "KGA does not estimate
  M, learns Δ̂ directly"; ImageNet-R no longer claimed to *prove* unknowability; POEM/AETTA causal claim
  softened; "independent recomputation"→"separate recomputation pipeline"; jackknife+ "tighter radius"
  reworded; two empty subsection headings filled; editing-instruction sentence rewritten as prose.
- Build: exit 0, 0 undefined refs, 20 pages. (The numbering-typo and appendix β=0-face items the review
  cited live in the long `kbound.tex`, not the compiled short paper.)
