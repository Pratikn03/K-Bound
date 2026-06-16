# Theorem Numbering Crosswalk (paper ↔ validators ↔ internal docs)

**Why this exists.** The published paper (`K-Bound_paper.pdf` / `primary.txt`) consolidates to "exactly five theorems," while the internal `THEOREM_CODE_STATUS.md` uses an older scheme (there, *Thm 4* = covariate, *Thm 5* = binary sign-of-difference). The two numberings do not line up, which is a traceability hazard for reviewers and auditors. This table is the single source of truth. All validator artifacts are now generated (2026-06-14).

| Published (paper) | Statement (one line) | Validator script(s) | Artifact (status) | Internal-doc name |
|---|---|---|---|---|
| **Lemma 1** | Reduction to disagreement region: `sign Δ = sign(M + γ)` (binary; K-class & regression extend) | `val_frontier.py`, `val_thm5_multiclass.py` | `results_frontier.json`; `results_thm5_multiclass.json` ✅ | — |
| **Lemma 2** | Plug-in regret identity; committal regret ≤ ε in the low-margin band | `val_thm2_regret.py` | `results_thm2_regret.json` ✅ (max identity gap 2.4e-17) | — |
| **Thm 1** | Unknowable regime is real/tight (Le Cam two-point); abstention mandatory `≥1−2α` | `val_thm1_lecam.py` | `results_thm1_lecam.json` ✅ | "Thm 1 non-identifiability + witness" |
| **Thm 2** | Finite-sample + anytime-valid adapt/freeze/abstain certificate, false-adapt/false-freeze ≤ α | `switching_certificate.py`; `val_thm3_evalue.py` | `val_thm3_evalue_results.json` (α=0.1); `results_thm3_evalue_alpha005.json` ✅ (α=0.05 → 0.0316) | "Thm 3 finite-sample certificate" / repo "T5 switching_certificate" |
| **Thm 3** | **Exact benefit-sign frontier**: identifiable over `{\|γ\|≤β}` iff `\|M\|>β`; heuristics = β=0 face | `val_frontier.py`, `val_agl.py` | `results_frontier.json`, `results_agl.json` ✅ | "frontier law" |
| **Thm 4** | One-bit dichotomy (resolves Conjecture 1's identifiable part): evidence fixes all up to one global sign bit | `theory_v2/` validators | `theory_v2/validation_results.json`, `conj1_closure_results.json` ✅ | repo `theory_v2` |
| **Thm 5** | Label-free rate: `\|b̂_j − b_j\| ≲ √(log(K/δ)/m)`; `m^{-1/2}` minimax | `val_thm5_multiclass.py`, `val_agl.py` | `results_thm5_multiclass.json` ✅ (multiclass max-err 1.1e-16) | "Thm 5 binary→multiclass" |
| **Cor. 1** | Covariate-shift identifiability (importance weighting) | regression track | `results/regression_covariate.json` ✅ | repo "T4 risk_dominance" |

## Important caveat for auditors

`tests/test_theorem_registry.py` and `tests/test_novel_theorem_bounds.py` operate on a **different, ELARA theorem set** (T1–T9, GDR — e.g. stochastic-dilution, KS-window, Bayes-τ), **not** the K-Bound paper's five theorems. They should not be read as tests of the theorems above. The K-Bound theorems' validators are the `val_*.py` scripts in `experiments/kbound/theory_validation/` and the `theory_v2/` validators.

## Open (acknowledged, not a defect)

**Conjecture 1** — the weakest falsifiable structural class under which the one-bit supplement suffices — is left open in §VIII; Thm 4 resolves its identifiable part, not the full conjecture.
