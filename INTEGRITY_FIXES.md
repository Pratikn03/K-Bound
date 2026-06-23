# Integrity Fixes Log — 2026-06-14

Resolution of the research-integrity items raised in `GAP_AUDIT.md`. Every change below is backed by a real run this session; no numbers were invented.

## Resolved

**1. Theorem 3 e-process false-adapt number (was A1 / MAJOR).**
The paper cited "false-adapt 0.028 ≤ 0.05", but the only committed artifact had been run at α=0.1 (worst-case 0.0626). We re-ran `val_thm3_evalue.py --alpha 0.05` (seed 0, 20 000 runs, horizon 2 000) → **worst-case false-adapt = 0.0316 ≤ 0.05**, persisted to `experiments/kbound/theory_validation/results_thm3_evalue_alpha005.json`.
**Correction:** the value `0.028` should read **`0.0316`** in the paper. The structure of the claim (anytime false-adapt ≤ α=0.05) is validated; only the digit changes. *Action remaining:* update the LaTeX source string `0.028 ≤ 0.05` → `0.0316 ≤ 0.05` before the next paper rebuild (the merged PDF addendum already records the correction).

**2. Theorem 5 multiclass artifact (was A2 / MAJOR).**
`val_thm5_multiclass.py` was simply never executed in this checkout (it *does* write an artifact). Ran it → **`results_thm5_multiclass.json`** now present: multiclass identity `max_abs_err = 1.11e-16` over 4 000 trials, sign agreement 4000/4000. The paper's "to 10⁻¹⁶" now traces to a committed file. (The artifact also honestly records one concept-shift case where the label-free certificate is wrong — `regression_shift.con.cert_correct=false` — consistent with the paper's stated residual.)

**3. Theorem 2 regret artifact (was A3 / MINOR).**
Ran `val_thm2_regret.py` → **`results_thm2_regret.json`** now present: exact identity `max gap = 2.35e-17`, minimax ratio-to-floor 1.0004, all checks passed. The paper's "to 10⁻¹⁷" now traces to a committed file.

**4. Theorem-numbering ambiguity (was A5 / MINOR).**
Added `docs/research/kbound/THEOREM_NUMBERING_CROSSWALK.md` mapping published Thm 1–5 ↔ validator scripts ↔ internal `THEOREM_CODE_STATUS.md` numbering, and flagging that `test_theorem_registry.py` / `test_novel_theorem_bounds.py` test the *ELARA* theorem set, not the K-Bound five.

**5. Camelyon17 natural-shift win (was B1/B2/B3).**
Recalibrated τ\* on the **stored debug-scale** composition grid (`wilds_kbound_debug_mps`, 432 cells incl. SAR): recalibrated K-Bound reaches **57% coverage at false-adapt 0.097 ≤ α**, beating both trivial policies overall (regret 0.0104 vs always-adapt 0.0130, always-freeze 0.0517) and 32× better on the harmful subset. See `experiments/kbound/theory_validation/frontier_decisive/camelyon_recal/`. This also confirms SAR *is* present in this grid. *Remaining for a publishable headline:* full-scale (`n_eval=1024`) re-run with **per-condition arrays serialized** and a **pre-registered held-out τ\***.

## Correction to GAP_AUDIT.md (false positive)

**B5 ("stale README") was incorrect.** It was based on a stale read of the README. The current `README.md` status block already says "all experiments completed" and lists ImageNet-C / Camelyon17 / ImageNet-R / RxRx1 / CIFAR-10.1 — it is accurate. No change to the experiment claims was needed; we only appended pointers to the new frontier-validation assets. `GAP_AUDIT.md` has been annotated accordingly.

## New validation assets added this session

- `experiments/kbound/theory_validation/frontier_decisive/` — synthetic frontier (`FRONTIER_DECISIVE.md`), real-ImageNet-C frontier (`realdata/`), Camelyon17 recalibration (`camelyon_recal/`), and the re-centered manuscript front matter (`MANUSCRIPT_RECENTER.md`).
- `docs/research/kbound/K-Bound_paper_with_frontier.pdf` — the paper with a 6-page figure addendum.
- `docs/research/kbound/THEOREM_NUMBERING_CROSSWALK.md`.

---

# 2026-06-19 — multi-agent loop closure

Second pass by a four-agent gap-closure loop (cartographer → theory → experiment → integrity-verifier). Every claim below is backed by a run executed this session (validators were *run*, not just inspected) or by a direct file read. No numbers were invented; where a prior instruction would have caused a fabrication, it was declined and reclassified.

## Independent re-execution (integrity-verifier ran these; all exit 0)
- `val_thm1_lecam.py` — floor `1−TV` tracks empirical `inf_M`. PASS.
- `val_thm2_regret.py` — exact identity max gap **2.35e-17**; all checks passed.
- `val_thm2_lecam_finite_n.py` **(new)** — finite-n Le Cam regret floor constant in n; Bayes rule meets floor; no panel rule beats it; Bretagnolle–Huber `(Λ/4)e^{-2c²} ≤ floor`. ALL CHECKS PASS; re-run bit-identical (seed 20260619 → deterministic).
- `val_thm3_evalue.py --alpha 0.05` — worst-case anytime false-adapt **0.0316 ≤ 0.05**.
- `val_thm5_multiclass.py` — multiclass max err **1.11e-16** (sign 4000/4000); regression max err 5.6e-13.
- `val_conj1_genpos.py` **(new)** — general-position obstruction confirmed over 200,000 trials; one bit certifies sign(Δ) on `C_dom` (err 0.0), fails without domination (err 0.2509). Validator's own verdict: unconditional weakest class **OPEN**.
- `val_conj1_caltransfer.py` — PASS.
- `verify_runner_pipeline.py --smoke` — `ALL_ASSERTIONS_PASSED`; all metrics labelled SYNTHETIC; SAR column present; Holm-corrected paired CIs computable.
- Torch-free certificate `pytest` (test_kga_package + holm + phase2_certificate_boundary + gate_p_and_scope_guard + holm_family_size) — **54 passed, 2 skipped** (missing optional .tex). torch-dependent gate-decision e2e tests skipped (no torch in sandbox).

## Anti-fabrication audit of the NEW code
- `val_thm2_lecam_finite_n.py`, `val_conj1_genpos.py`, `per_condition_serialize.py`, `multiseed_paired_ci.py`: all use `np.random.default_rng(<fixed seed>)` (master seed `20260619` with deterministic per-cell derivation `seed + 1000*ci + ni`), **no hardcoded "computed" result floats** in any `json.dump`, hard `assert`s that fail loudly. Grep for `hardcod|fudge|fake|placeholder|magic|stub` over the new files returned **empty**. `per_condition_serialize.py` reshapes already-measured records only and stamps `kga_backend` (`sklearn_gradient_boost` vs `numpy_knn_fallback`) so the CPU fallback can never be mistaken for the production certificate.

## Files created / edited this loop
**Theory (verified present):**
- `docs/research/kbound/paper/sections/main_theory_5.tex` — added Proposition `prop:lecam-finite` (+ `cor:lecam-regret-floor`) and proof (finite-n two-point Le Cam regret lower bound for Thm 2).
- `docs/research/kbound/paper/sections/weakest_class.tex` — sharpened `rmk:genpos`/`thm:cmono-weakest` with a two-region counterexample; **explicitly keeps Conjecture 1 general position OPEN** ("We leave this open and do not claim it").
- `experiments/kbound/theory_validation/val_thm2_lecam_finite_n.py` + `results_thm2_lecam_finite_n.json` (new).
- `experiments/kbound/theory_validation/val_conj1_genpos.py` + `results_conj1_genpos.json` (new).
- A4 loose-gate clarifying comments added to `val_thm2_regret.py` and `val_thm5_multiclass.py` (asserts unchanged).

**Experiment (all under gitignored scratch `experiments/kbound/wilds/`):**
- `experiments/kbound/wilds/per_condition_serialize.py`, `multiseed_paired_ci.py` (new, torch-free).
- Edited `run_camelyon17_kbound.py` (+ serialization) and `run_imagenetr_kbound.py` (+ serialization, `--resume`).
- `scripts/run_remaining_gpu_experiments.sh`, `docs/research/kbound/RUN_ON_MAC.md`, `experiments/kbound/theory_validation/verify_runner_pipeline.py` (new). No canonical result JSON altered (mtimes preserved).

**Documentation fixes applied this loop (integrity-verifier, data-backed only):**
- `kbound.tex` CIFAR-10.1 caption — replaced the stale "single-seed quick protocol; multi-seed re-run is future work" with the real **5-seed** pooled numbers transcribed from `cifar101_multiseed_v1/pooled_summary.json` (seeds [0–4]). Table body unchanged.
- `kbound.tex` contributions bullet (L160) — "ImageNet-scale corruptions" → **"ImageNet-C noise corruptions (ResNet-50, 36 cells)"** (matches existing table captions; B6).
- `kbound.tex` companion-stack appendix — added the verified e-value result **"worst-case anytime false-adapt 0.0316 ≤ α=0.05"** (from `results_thm3_evalue_alpha005.json`) where the anytime e-process certificate was previously only qualitative (A1 done correctly — see next).

## Correction recorded (declined a would-be fabrication)
**A1 was a MISREAD in the 2026-06-14 log.** That log said paper "`0.028` → `0.0316`". Direct inspection shows the `0.028`s in `kbound.tex` are **Office-Home regret** (L1155, L1181) and **anomaly-routing CI** (L1407/L1417) values — *not* the e-process false-adapt rate — and `0.0316` appeared nowhere in the .tex. Changing them would corrupt correct numbers. The integrity-verifier therefore **did not change any `0.028`** and instead added the `0.0316` e-value result to the (previously qualitative) anytime discussion. The Thm-3 mechanism/guarantee were never in question.

## Honest caveats
- `docs/research/kbound/THEOREM_NUMBERING_CROSSWALK.md` is a **tracked deletion** (` D`) in the working tree (part of a broader `docs/research/kbound/*.md` cleanup). Its content survives in the in-paper consolidation table; restore from git if a standalone file is desired.
- `run_remaining_gpu_experiments.sh` lives at repo-root `scripts/` (not `docs/research/kbound/scripts/`).
- One e2e test (`test_audit_gate_decision_rule_e2e.py`) and `test_gate_decision_rule.py` could not be executed here (transitive `import torch`); they are GPU/Mac-env tests, skipped per torch-free policy, not regressions.

## What remains (not closeable in this sandbox)
1. Mac GPU runs: **ImageNet-R multi-seed**; **Camelyon17 `B_v2`** full-scale completion (a `_partial.json` exists) — see `RUN_ON_MAC.md` / `scripts/run_remaining_gpu_experiments.sh`.
2. **Conjecture 1 general position** — a genuinely open research sub-case, honestly labelled open in the paper.


---

# 2026-06-20 — production-readiness integrity cleanup pass

Host run (desktop-commander). Full audit trail + per-file backups in
`audits/integrity_2026-06-20/`. Every changed verdict was re-derived from the underlying
numbers; no fabrication; files backed up before every edit. Git HEAD at start `a17d744`.

## 1. `beats_both` now enforces `false_adapt ≤ α` (α = 0.1)
The stored `beats_both` flag was regret-only in some writers and did not enforce the
pre-registered false-adapt budget. Audited all result JSONs (110 files, 327 stored-True
`beats_both*` nodes). 159 nodes are regret-only-True while FA>α; **154 are correct-by-design**
(a sibling `candidate_win`/`verdict_win`/`fa_ok` already gates them — win-finder scans and the
dev-locked H_v2/M_v2 protocol files) and were left intact; **5 were genuine ungated stored
verdicts** and were corrected (preserve `beats_both_raw`, add `beats_both_corrected` + note,
flip live flag): `iwildcam_full_val` route-b (FA 0.5; the router 0.0307 vs freeze 0.0310
near-tie), `iwildcam_full_idval` route-a sar_online (FA 0.5) & tent_online (FA 0.154),
`imagenetc_1pct` eata (FA 0.111), `imagenetr_kbound_light` sar_online (FA 0.5). No prior
`beats_both_corrected/_raw` markers existed (the iWildCam session left none), so no double-edit.
Per-dataset corrected verdicts: **CIFAR-10-C WIN, ImageNet-C WIN, Camelyon17 (Protocol G) WIN**
(all re-verified, FA≤α and both regret bars); CIFAR-10.1 False (beats both in only 1/5 seeds),
ImageNet-R False (0/10 candidates), RxRx1 False (ties freeze; KGA freezes), iWildCam False
(FA 0.5), Office-Home(val) False (KGA freezes), fMoW False (FA 0.375), PovertyMap N/A
(dev-screen stop), ACDC N/A (code-only/unrun). Code fix: added the `FA ≤ ALPHA` gate to
`policy_metrics` in `src/scripts/kbound/cifar_tent_mps_v2.py` + its docs copy (old comparison
preserved as `beats_both_regret_only`); `wilds/analysis.py` and packaged `_analysis.py` were
already gated (the stale JSONs were written by older code); annotated the `cifar_tent_online.py`
accuracy-dominance flag. Verifier (`verify_after_patch.py`): 0 ungated bugs remain, 0
gate-consistency violations. Summary table: `audits/integrity_2026-06-20/benchmark_verdicts.json`.

## 2. K-Bound self-contained from `src/elara` + provenance
Exactly one live cross-import found (runtime import-trace; `rg` missed it — the file is
git-ignored): `experiments/kbound/vendored_from_elara/theory/__init__.py` did
`from elara.theory.theorem_registry import …`. Repointed to the byte-identical local sibling
`.theorem_registry`. Standalone proof: with `import elara*` blocked, the `kga` package + the
vendored `theory`/`certification`/`drift` trees import and run (registry=10,
`empirical_bernstein` OK) → zero `src/elara` dependency. Added a paper-ready provenance note to
`kga/certificate.py` (empirical-Bernstein / Maurer-Pontil 2009 certificate is shared with the
ELARA companion work, which delegates to this `kga` function). **No ELARA file modified.**

## 3. Manifest completeness + Conjecture-1 disambiguation
`DATA.md` §3b added: the full vision/WILDS benchmark suite (CIFAR-10-C, ImageNet-C, CIFAR-10.1,
ImageNet-R, Camelyon17, RxRx1, iWildCam, Office-Home, fMoW, PovertyMap, ACDC) with corrected
verdicts and the gate definition — §3 had omitted them. `CLAIMS_CALIBRATION.md`: the empirical
**p\*-law** (previously mislabeled "Conjecture 1") renamed to **"p\*-law conjecture"** + a
disambiguation note; the paper's **Conjecture 1** (label-free benefit-sign bracketing,
`\label{conj:gen}` in `paper/sections/main_theory_5.tex`) left untouched. Naming only — no math.
