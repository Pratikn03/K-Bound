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
