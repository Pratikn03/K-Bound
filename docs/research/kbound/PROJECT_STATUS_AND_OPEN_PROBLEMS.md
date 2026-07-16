# K-Bound / KGA — Canonical Status & Open-Problem Ledger

**This is the single source of truth.** Last reconciled: 2026-07-02 (Wave 4 strict-core closure).

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
| `thm:anytime` + `thm:multicand` (certificate **extensions**) | (a) Anytime-valid streaming: false-adapt ≤ α **time-uniformly** under optional stopping (Ville + one-sided betting supermartingale). (b) Multicandidate routing: **family-wise** false-adapt ≤ α for an arbitrary/adversarial selector (Bonferroni selection-proof containment). | **Proven (core guarantees) + machine-validated + line-by-line proof-audited (2026-06-29).** No fatal flaw; audit found only fixable expository gaps, now fixed (T=1 "specializes to," not "reproduces"; Šidák needs disjoint per-candidate calibration — Bonferroni is the default; CS centering made explicit). Folded into `kbound.tex` App.\ *Theory extensions*. Validators `val_sequential_anytime.py`, `val_multicandidate.py`. |
| `thm:minimax-opt` + `thm:t1c-exact` | On the identifiable side, minimax order-optimality + exact 3-world constant `κ(α)n_opt`. | **Proven + validated** (`val_minimax_optimality.py`, `val_tight_constants.py`). |
| `thm:multiclass-multicand` + `thm:anytime-multicand` | Multiclass routing + anytime multicandidate Bonferroni FWER on `D`. | **Proven + validated** (`val_multiclass_multicandidate.py`, `val_anytime_multicandidate.py`). |
| `thm:mc-cap-impossibility` | No single scalar multiclass capacity in the general vector-concept regime. | **Closed (impossibility)** + `val_multiclass_capacity.py` Block D. |
| `thm:margin-compute-dichotomy` | Frontier margin computability dichotomy (`conj:dich-compute`). | **Closed** + `val_margin_computability.py`. |
| `thm:reg-bracket-dichotomy` | Regression/general drift: bounded-drift iff + general impossibility. | **Closed** + `val_regression_bracketing_closure.py`. |

> **Note:** Section B of `THEORY_100_PERCENT_CLOSURE_PLAN.md` is **fully closed** as of Wave 4 (2026-07-01).
> Wave 6 (2026-07-15) closed the paper-faithful Lean foundation gaps: exchangeable-score
> conformal reduction, discrete Ville / e-process step, two-point Le Cam packaging,
> Hoeffding-radius commit bridge, and evidence-preserving swap involution.
> `python3 formal_audit.py --build --full-foundations` exits 0 (53 theorem checks).
> This is **not** a claim that all of Mathlib probability was rebuilt from axioms.

### GENUINELY OPEN (outside closure-plan scope)

| Item | Note |
|---|---|
| External reviewer sign-off | `REVIEWER_REPRO_PACKET.md` — process, not theory |
| Real-camera held-out R2 tables | Empirical, not theory |

---

## 2. Empirical ledger (verified with a VALID out-of-fold radius)

### Beats-both (the only verified wins) — synthetic stress grids
| Setting | Result | Radius |
|---|---|---|
| CIFAR-10-C stress grid, Tent | beats both, CI excludes 0, FA_u=0 | LOO (`decide_kga`) ✓ |
| CIFAR-10-C stress grid, EATA | beats both, CI excludes 0, FA_u=0 | LOO ✓ |
| CIFAR-10-C mixed head-to-head | **beats POEM and AETTA** (pre-registered WIN), FA_u=0 | cached stress-grid records ✓ |
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
- **Mixed-stream cross-protocol aggregate:** **re-run complete** (`mixed_protocol_oof_v2`, OOF LOO conformal). Beats-both on constructed aggregate ($n{=}143$); not a natural-shift headline. Earlier 13–24× in-sample figures withdrawn.
- **ELARA-U integration:** retrospective (uses labels), no-harm null, **non-headline**. Correctly labeled.
- **Edge / real-camera deployment:** pipeline built + integrity-clean (proper 3-way split-conformal in `edge/conformal.py`, anti-leakage tests). **Result tables are placeholders** — no camera win yet; present as a feasibility / no-harm study unless a real held-out run clears the bar.

### In-sample-ε audit (root-caused 2026-06-26)
- **Had the bug → fixed:** `analyze_F.run_split` (shared core), `score_kbound_holdout`, `mixed_stream_kbound`. (`run_protocol_dev_lock` had a reporting-only copy.)
- **Always valid (out-of-fold / proper conformal):** `decide_kga`, `gate_baseline_comparison._kga_bhat`, `pacs_vlcs_runner.decide_transfer`, `verify_realshift_win` (k-fold), `edge/conformal.py` (split-conformal), `kbound_pkg/router`.

---

## 3. Freeze gate (must all be green before production freeze)
1. ✅ All scorers compute ε out-of-fold (audit complete, 2026-06-26).
2. ✅ Mixed-stream re-run folded in (`mixed_protocol_oof_v2`; `scripts/mixed_stream_kbound.py`).
3. ✅ Long paper `kbound.tex` reclassified (natural shifts → no-harm; mixed-stream OOF results in §`sec:mixedstream`).
4. ✅ Edge section framed as feasibility/no-harm (camera tables RESULT PENDING; real R2 still open).
5. ✅ Recompile both PDFs clean (`kbound_short.tex`, `kbound.tex`) — 2026-06-25.
6. ⬜ External sign-off: theory/stats reviewer on `thm:uncond-weakest`+Lemma 1 and the coverage theorem; one independent reproducer (`REVIEWER_REPRO_PACKET.md`).
7. ⬜ **85+ path:** real camera R2 (`run_edge_source_gate.sh` → S03–S10 → `run_edge_publication_pipeline.sh`); full 5-seed panel with RxRx1; `run_85plus_readiness.sh` score ≥ 85.

**85+ readiness command:** `bash docs/research/kbound/scripts/run_85plus_readiness.sh`

**Honest headline at freeze:** an impossibility/frontier theorem + a certificate that provably
controls false-adapt (gate table) + beats-both on synthetic stress grids + uniform no-harm on
five real benchmarks. *No* real-shift or camera beats-both is claimed.

---

## 4. Documentation hygiene (2026-07-01)

**Canonical index:** [`DOCS_INDEX.md`](DOCS_INDEX.md)

Stale dated process/status MDs listed below were **removed** (recoverable from git history).
Do not recreate them; update `PROJECT_STATUS_AND_OPEN_PROBLEMS.md` and `claim_ledger.json` instead.

**Removed (superseded):** `COMPLETION_STATUS_2026-06-19.md`, `LAYOUT_TRIM_REPORT_2026-06-20.md`,
`LAYOUT_VERIFY_REPORT_2026-06-21.md`, `NEW_MATH_ROADMAP_2026-06-19.md`, `PAPER_BLUEPRINT_80.md`,
`WINNING_PAPER_ANATOMY_AND_RESTRUCTURE.md`, `WINNING_PAPER_RUBRIC.md`, `RESULTS_PENDING.md`,
`ELARA_KGA_MERGE_PLAN.md`, `HEADTOHEAD_VERIFICATION.md`, `RUN_ON_MAC_POEM_AETTA.md`,
`PUBLICATION_POLISH_OPTIONS.md`, `FREEZE_COMPLETION_PLAN.md`, `AI_SLOP_RISK_CLEANUP_PLAN.md`,
`CLAIMS_CALIBRATION.md`, `DICHOTOMY_VERIFICATION_2026-06-19.md`.

**Review before removing (not done):** `kbound_submission.tex` (old frozen snapshot),
`manuscript/` (parallel book-style document; still contains a stale "open" `conj:gen`).

**Keep (current):** this file, `DOCS_INDEX.md`, `REVIEWER_REPRO_PACKET.md`, `THEORY_AUDIT_senior_review.md`,
`gate_comparison.md`, `MIXED_BENCHMARK_PROTOCOL.md`, `theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md`,
`realshift_win/PROTOCOL_realshift_win.md`, `edge/` docs, all `paper/sections/*`, the live
`kbound.tex` / `kbound_short.tex`.

## 2026-07-15: Lean Wave 6 — paper-faithful foundations closed
- New modules: `Exchangeable.lean`, `Ville.lean`, `LeCamMeasure.lean`; upgrades to
  `Dichotomy.lean`, `Rates.lean`.
- `FOUNDATIONAL_PROBABILITY_LIMITS` cleared; `--full-foundations` PASS (53 checks).
- Scope: paper-faithful cores, not a Mathlib probability textbook.


## 2026-07-15: repo cleanup (confusion risks closed)
- Dual packages clarified: edit **`kga/`**; `kbound_pkg/` is frozen (README + `REPO_LAYOUT.md`).
- Dual experiment trees: canonical = `experiments/kbound/`; nested
  `docs/research/kbound/experiments/` documented as stubs; nested CIFAR data blob removed.
- ELARA Family/phase2 tests moved to `archive/legacy_elara/tests/` (not in default pytest).
- Paper `.bak` / Word / 2col drafts archived under `archive/paper_drafts_2026-07-15/`.
- `formal/.lake` wiped locally (~3.6G); remains gitignored.
- Root scratch JSON/coverage/logs removed; tracked root multiseed Camelyon copies dropped.
- `AETTA/` converted from broken submodule gitlink to vendored plain files (`VENDOR.md`).
- Docker/API env rebranded to `KGA_*` (legacy `UAIS_*` still accepted).
