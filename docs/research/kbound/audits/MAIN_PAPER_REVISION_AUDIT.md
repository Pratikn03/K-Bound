# MAIN_PAPER_REVISION_AUDIT — kbound_short.tex → submission-ready main manuscript

> **SUPERSEDED 2026-07-26 on one point.** This dated record asserts *uniform* natural-shift
> no-harm. That claim was withdrawn: no-harm holds on the four one-sided locked tracks
> (Camelyon17, iWildCam, Office-Home, RxRx1) only. PACS loses to always-adapt by 2.45x,
> ImageNet-R loses on 7 of 10 backbones, CIFAR-10.1 fails the transfer bar (FA_u = 0.167).
> The record is left unedited below; the live claim is in `kbound_short_body.tex`.


Date: 2026-07-03. Auditor: Claude session (lead technical editor role), with Pratik.
Scope: revision of the SHORT paper only. The long manuscript (`kbound.tex`) remains the full
technical record and is not edited by this revision except where noted (none).

## 1. Sources and build

| Item | Path |
|---|---|
| Short paper (main-paper candidate) | `docs/research/kbound/kbound_short.tex` (939 lines pre-revision) |
| Short-paper appendix | `docs/research/kbound/kbound_short_appendix.tex` |
| Long paper (technical record) | `docs/research/kbound/kbound.tex` (~2190 lines) |
| Generated result numbers | `docs/research/kbound/paper/generated/kbound_numbers.tex` (from `scripts/make_tables.py`) |
| Bibliography | `docs/research/kbound/paper/references_kbound_expanded.tex` (inline, `\input`) |
| Figures | `docs/research/kbound/figures/` |
| Camera protocol tables | `docs/research/kbound/edge/kbound_camera_main_tables.tex` |
| Lean formalization | `docs/research/kbound/lean/` — 18 files, 52 theorems, 0 `sorry`/`admit`/project axioms (verified by source scan 2026-07-02; build artifacts present; `lake exe cache get` fixes ProofWidgets JS build failures) |
| POEM/AETTA implementation notes | `MIXED_BENCHMARK_PROTOCOL.md` + `experiments/kbound/results/mixed_headtohead_v1/` (documented simplifications: batch-summary entropy, no per-sample streams, no dropout-AETTA) |

Build command (2 passes for cross-refs); sandbox cannot mount the drive, so compile is run by
the author on the host Mac:

```
cd docs/research/kbound
pdflatex -interaction=nonstopmode kbound_short.tex && pdflatex -interaction=nonstopmode kbound_short.tex
```

Current page count: **TBD — to be recorded from the first post-revision compile** (pre-revision
PDF exists but page count could not be read in this environment; IEEEtran conference two-column,
main text ≈ sections I–IX + appendix input).

## 2. Main theorems (pre-revision structure → post-revision structure)

Pre-revision main text: `lem:nonid` (Lemma, non-identifiability), `thm:headline` (ONE consolidated
theorem: certificate + frontier), `cor:abstain-valid`. All other theory (one-bit dichotomy, weakest
one-bit dominance polytopes `thm:uncond-weakest`, minimax rate, 3-world constant, multiclass
capacity, regression bracketing, anytime e-process, family-wise routing) lives in the appendix and
long paper — correct placement per revision spec §H, unchanged.

Post-revision main text (this revision):

| Block | Label | Content | Source of proof |
|---|---|---|---|
| Theorem 1 | `lem:nonid` (label kept; env promoted Lemma→Theorem) | Matched evidence / abstention necessity | Appendix app:theory-full; long paper; Lean `SwapInvolution`/two-point construction |
| Theorem 2 | `thm:headline` | Exact benefit-sign frontier: sign Δ = sign(M+γ), identifiable over C_β iff \|M\|>β | Appendix app:theory-full; Lean frontier files; validators val_* |
| Theorem 3 | `thm:certificate` (new label) | Finite-sample adapt/freeze/abstain certificate, FA_u ≤ α | Appendix; Lean certificate algebra; val_thm3 validators |

Reference retargeting performed: certificate-meaning uses of `\ref{thm:headline}` (method §, metrics
§) → `\ref{thm:certificate}`; frontier-meaning uses kept; textual "Lemma" → "Theorem" at all
`\ref{lem:nonid}` sites (main text; appendix has none); appendix lines 38–39 "consolidated
theorem" wording updated to the three-theorem structure.

## 3. Headline empirical claims → artifacts (verified this session or in the 2026-07-02/03 primary-source verification)

| Claim | Strength wording allowed | Artifact |
|---|---|---|
| CIFAR-10-C stress grid, Tent/EATA: beats-both; SAR ties adapt | beats-both, CI-robust, Holm; identifiable mixed regime only | `experiments/kbound/results/stress_grid_multiseed_v1/seed{0..4}/`, `scripts/percondition_bootstrap.py` |
| Mixed head-to-head vs POEM/AETTA: WIN, replicated 3/3 configs | beats **protocol-matched POEM-style/AETTA-style baselines**; NOT "official reproductions" | `experiments/kbound/results/mixed_headtohead_v1/`, `research_lock/WIN_HUNT_v3_ARM_F_result.json` |
| ImageNet-C SAR: beats-both at mechanism-faithful operating point | operating-point-qualified beats-both | ImageNet-C logged cells (36), appendix table |
| Natural shifts (Office-Home M v2 n=35, iWildCam H v2 n=72, Camelyon17 G n=18 OOD, RxRx1 J, PACS 4 domains): uniformly no-harm | **no-harm only** (ties better policy, beats worse); NO single-dataset natural beats-both | `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json`, per-protocol result JSONs, `docs/research/kbound/pacs_*_percell.json` |
| Camelyon17 earlier n=54 beats-both | **withdrawn** (id_val pooling artifact) — disclosure must remain | `audits/integrity_2026-06-20/camelyon_reconciliation/` (repo-root audits dir pending git restore — see §6) |
| Cross-protocol aggregate n=143 (per-dataset gates) | beats-both on a constructed aggregate; not universal-gate | `mixed_protocol_oof_v2` re-run JSON |
| Universal single gate, 3 sources n=143 | pre-registered CI_ROBUST_WIN | `research_lock/WIN_HUNT_v2_PROTOCOL.yaml` + Arm A verdict JSON |
| Universal single gate, SEVEN sources n=359 | pre-registered CI_ROBUST_WIN; researcher-pooled stream; not a single-dataset natural win | `research_lock/WIN_HUNT_v3_ARM_E_result.json` |
| Real-data anytime e-process demo (iWildCam 35,370 imgs, FREEZE @ window 6, anytime FA 0, ties oracle) | DEMO (freeze is oracle; beats-both structurally impossible) | `research_lock/WIN_HUNT_v3_ARM_D_result.json` |
| Priced certificate validity (λ sweep, FA ≤ α ∀λ, 6912 records) | PASS_priced_validity; λ* medians/q90 only (means are cost→0 artifacts) | `research_lock/WIN_HUNT_v3_ARM_G_result.json` |
| ImageNet-R τ′ re-analysis (83% co-adapted panel rejection vs 1/54 independent; Arm C near-oracle 0.0005 under independent panels; v1 fixed-τ* FAILs FA 3.7–4.9%) | refinement of abstention verdict; v1 FAIL disclosed | Wave-5 `gapclose_wave5/` results + WIN_HUNT_v2 Arm C verdict |
| Decision-baseline comparison (gates table) | only certificate has finite-sample FA_u guarantee | `scripts/gate_baseline_comparison.py` on locked per-cell dump |

## 4. Baseline faithfulness status (spec §B.11–12)

POEM and AETTA are **protocol-matched ports** consuming the same logged per-condition signals,
with documented simplifications (batch-summary entropy rather than per-sample streams; no
dropout-AETTA). They are **not** proven official-repo parity reproductions. Required wording:
"protocol-matched POEM-style and AETTA-style baselines" + explicit sentence that official
per-sample POEM and dropout-based AETTA reproduction remains a camera-ready faithfulness check.
Pre-revision text said "faithful ports of their published protectors" — **changed** in this revision.

## 5. Calibration guarantee status (spec §B.13)

- Natural-shift protocols: dev/test **split-conformal** — exact finite-sample marginal coverage
  under exchangeability of calibration and test cells.
- Per-cell grids: **leave-one-out (jackknife) cross-conformal** radius — calibration-set coverage
  ≥ 1−α by construction (realized 0.898 at nominal 0.90); distribution-free only asymptotically
  under estimator stability. Formal finite-sample alternative jackknife+ (Barber et al. 2021,
  level 1−2α) is **noted but not implemented**. This caveat must survive revision — it does
  (Method §, unchanged in substance).
- ε is an operational radius calibrated from held-out residuals; it is **not** an estimate of β.
  M/γ/β/ε separation enforced (four-quantities table added).

## 6. Caveats that MUST remain in the paper (all verified present post-revision)

1. FA_u ≤ α is the theorem; FA_c is NOT α-bounded (reported empirically) — + new warning box.
2. No universal improvement claim; "safety layer, not accuracy booster."
3. No single-dataset natural beats-both; natural shifts are no-harm only.
4. Universal-gate pooled result: pre-registered, researcher-pooled heterogeneous stream, one
   benefit model + one out-of-fold radius; not a natural single-dataset win.
5. Camelyon17 n=54 withdrawal + id_val pooling artifact disclosure.
6. In-sample-radius beats-both (Office-Home/iWildCam earlier figures) withdrawn; out-of-fold
   re-run is the reported one.
7. ImageNet-C SAR claim qualified to the mechanism-faithful operating point.
8. Jackknife/LOO radius finite-sample status caveat + jackknife+ not implemented.
9. Lean claim wording: kernel-checked formalization of the **algebraic and finite-testing core**;
   deployment assumptions (exchangeability, calibration transfer, risk alignment) remain explicit
   assumptions outside Lean. 0 sorry/admit/project-axioms is claimed because the strict source scan
   confirmed it (2026-07-02).
10. ImageNet-R diagnostic status + τ′ refinement wording (partly threshold artifact; v1 FAIL).

Known repo issue (outside this revision, tracked): repo-root `audits/integrity_2026-06-20/` was
deleted during the ELARA scrub; both papers cite it. Restore via
`git log --diff-filter=D --oneline -- audits` + `git restore --source=HASH^ -- audits` before
sharing the repo.

## 7. Revision outputs

- `docs/research/kbound/audits/MAIN_PAPER_REVISION_AUDIT.md` (this file)
- `docs/research/kbound/audits/MAIN_PAPER_REVISION_CHANGELOG.md`
- `docs/research/kbound/audits/MAIN_PAPER_CLAIM_MATRIX.md`
- Revised `kbound_short.tex` (+ 2-line appendix wording fix)
