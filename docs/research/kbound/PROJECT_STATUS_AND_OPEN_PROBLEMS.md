# K-Bound / KGA — Canonical Status & Open-Problem Ledger

**This is the single source of truth.** It supersedes all dated `*_2026-06-*` status/report
MD files (see "Deprecated files" at the end). Last reconciled: 2026-06-26, after the
out-of-fold conformal audit.

---

## 1. Theory ledger (what is proven vs open)

### CLOSED / PROVEN (machine-checked where noted)
| Result | Statement | Status |
|---|---|---|
| `thm:frontier` / `thm:headline` | Benefit sign is label-free identifiable **iff** observable margin `|M| > β` (drift budget). | **Proven** (spine) |
| `thm:conj1-dichotomy` | One-bit dichotomy / impossibility: no evidence-definable assumption identifies `sign Δ`; minimal supplement = exactly one bit. | **Proven** |
| `thm:imp` | Two-point impossibility: matched evidence, opposite benefit → minimax committal error ½. | **Proven** |
| `thm:cmono-weakest` (i),(ii) | One bit certifies `sign Δ` on the margin-monotone class **unconditionally**. | **Proven** |
| `thm:cmono-weakest` (iii) | `C_mono` is *a* weakest one-bit class **under** General Position. | **Proven (conditional)** |
| `thm:uncond-weakest` | **Unconditional** weakest one-bit classes = explicit finite family of *dominance polytopes* `W*={T(r)≥G(r)}`; no unique weakest class; GP recovered as the collapsing face. | **Proven + machine-verified** (`val_unconditional_weakest.py`; 2.8e5 box + 3.2e3 polytope fibres, 0 mismatches; independently reproduced) |
| `conj:gen` (**label-free bracketing**) | Universal label-free benefit bracketing **does not exist** (negative resolution = the impossibility); minimal supplement one bit; weakest class characterized by `thm:uncond-weakest`. | **RESOLVED (negatively).** *Not* an open problem. |

> **Note on "C1 / label-free bracketing":** any ledger marking this "Open" is **stale**.
> It is resolved negatively — universal label-free bounds without structure provably cannot
> exist; proving they *do* would contradict `thm:conj1-dichotomy`.

### GENUINELY OPEN (honest frontier — none are claimed as solved)
| Conjecture | What is open | File |
|---|---|---|
| `conj:gen-capacity` | General knowability–capacity: scalar capacity `K>1 ⟺ identifiable` is proven only under regularity R1 (unique flip locus) + R2 (monotone-in-nuisance). Removing R1/R2 (non-monotone flip loci; multiclass `K≥3`) is open. | `knowability_capacity_general.tex` |
| `conj:dich-compute` | "Computability is free" for integral-functional families: whether the frontier margin `m(O)` is always a *computable* functional of `Q_X`. Constructive-measurability question; no probabilistic content. | `knowability_dichotomy.tex` |
| Tight finite-sample rate | A tight (non-conservative) radius / exact evidence-channel rate constants; only conservative bounds proven. | `onebit_audit_rate.tex`, `knowability_rates.tex` |
| Fully-general-drift bracketing | General-drift / regression extension characterization (the unknowable regime). | `regression_conjecture.tex`, `benefit_sign_frontier.tex` |

---

## 2. Empirical ledger (verified with a VALID out-of-fold radius)

### Beats-both (the only verified wins) — synthetic stress grids
| Setting | Result | Radius |
|---|---|---|
| CIFAR-10-C stress grid, Tent | beats both, CI excludes 0, FA_u=0 | LOO (`decide_kga`) ✓ |
| CIFAR-10-C stress grid, EATA | beats both, CI excludes 0, FA_u=0 | LOO ✓ |
| ImageNet-C SAR (mechanism-faithful) | beats both (regret 0.023 vs 0.112/0.027) | LOO ✓ |
| Gate-baseline comparison (Table III) | only the certificate keeps FA_u=0 across 432 cells | LOO (`_kga_bhat`) ✓ |

### Uniformly no-harm — every natural shift (verified out-of-fold)
| Dataset | Result | Note |
|---|---|---|
| Office-Home M v2 | no-harm: beats adapt (+0.031 [0.004,0.062]), ties freeze (+0.0001) | was an in-sample-ε "win"; corrected |
| iWildCam H v2 | no-harm: beats adapt (+0.099 [0.080,0.118]), ties freeze exactly | corrected |
| Camelyon17 | no-harm: beats freeze (+0.072 [0.053,0.093]), ties adapt | helpful regime |
| RxRx1 | no-harm (harmful-dominated; matches freeze-oracle) | — |
| PACS (4 domains, DomainBed) | 3 no-harm + 1 null; 0 beats-both; FA_u ≤ α (photo 0.056) | clean (out-of-fold) ✓ |

### Pending / non-headline
- **Mixed-stream cross-protocol aggregate:** scorer fixed to out-of-fold; **needs re-run** (`scripts/mixed_stream_kbound.py`). Earlier 13–24× was in-sample-ε; withdrawn pending re-run.
- **ELARA-U integration:** retrospective (uses labels), no-harm null, **non-headline**. Correctly labeled.
- **Edge / real-camera deployment:** pipeline built + integrity-clean (proper 3-way split-conformal in `edge/conformal.py`, anti-leakage tests). **Result tables are placeholders** — no camera win yet; present as a feasibility / no-harm study unless a real held-out run clears the bar.

### In-sample-ε audit (root-caused 2026-06-26)
- **Had the bug → fixed:** `analyze_F.run_split` (shared core), `score_kbound_holdout`, `mixed_stream_kbound`. (`run_protocol_dev_lock` had a reporting-only copy.)
- **Always valid (out-of-fold / proper conformal):** `decide_kga`, `gate_baseline_comparison._kga_bhat`, `pacs_vlcs_runner.decide_transfer`, `verify_realshift_win` (k-fold), `edge/conformal.py` (split-conformal), `kbound_pkg/router`.

---

## 3. Freeze gate (must all be green before production freeze)
1. ✅ All scorers compute ε out-of-fold (audit complete, 2026-06-26).
2. ⬜ Mixed-stream re-run folded in (`mixed_stream_kbound.py`).
3. ⬜ Long paper `kbound.tex` reclassified (natural shifts → no-harm; mixed-stream pending) — short paper already done.
4. ⬜ Edge section framed as feasibility/no-harm (or real camera run populated).
5. ⬜ Recompile both PDFs clean (`kbound_short.tex`, `kbound.tex`).
6. ⬜ External sign-off: theory/stats reviewer on `thm:uncond-weakest`+Lemma 1 and the coverage theorem; one independent reproducer (`REVIEWER_REPRO_PACKET.md`).

**Honest headline at freeze:** an impossibility/frontier theorem + a certificate that provably
controls false-adapt (gate table) + beats-both on synthetic stress grids + uniform no-harm on
five real benchmarks. *No* real-shift or camera beats-both is claimed.

---

## 4. Deprecated files (safe to `git rm` — superseded by this doc or the live papers)
Stale dated process/status notes and editor backups. Reversible via git history.

- Backups: `kbound_full58_backup_2026-06-10.tex`, `kbound_pre6trim_20260619_1430.bak.tex`,
  `kbound_short_pre6edit_2229.bak.tex`, `kbound_short_preIEEE_2301.bak.tex`
- Superseded status/process MD: `COMPLETION_STATUS_2026-06-19.md`,
  `LAYOUT_TRIM_REPORT_2026-06-20.md`, `LAYOUT_VERIFY_REPORT_2026-06-21.md`,
  `NEW_MATH_ROADMAP_2026-06-19.md`, `PAPER_BLUEPRINT_80.md`,
  `WINNING_PAPER_ANATOMY_AND_RESTRUCTURE.md`, `WINNING_PAPER_RUBRIC.md`,
  `RESULTS_PENDING.md`, `ELARA_KGA_MERGE_PLAN.md`, `HEADTOHEAD_VERIFICATION.md`,
  `RUN_ON_MAC_POEM_AETTA.md`, `PUBLICATION_POLISH_OPTIONS.md`

**Review before removing (not obviously stale):** `kbound_submission.tex` (old frozen snapshot),
`manuscript/` (parallel book-style document; still contains a stale "open" `conj:gen`).

**Keep (current):** this file, `REVIEWER_REPRO_PACKET.md`, `THEORY_AUDIT_senior_review.md`,
`gate_comparison.md`, `MIXED_BENCHMARK_PROTOCOL.md`, `theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md`,
`realshift_win/PROTOCOL_realshift_win.md`, `edge/` docs, all `paper/sections/*`, the live
`kbound.tex` / `kbound_short.tex`.
