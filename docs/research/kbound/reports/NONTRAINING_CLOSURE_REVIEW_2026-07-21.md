# K-Bound Non-Training Closure — Scientific Review Report

**Date:** 2026-07-21 · **Branch:** `main` · **Pre-task HEAD:** `4775850` · **Working tree at start:** clean (tracked), IDE files (`.idea/`, `.virtual_documents/`) held/locked and untouched.

**Session scope:** close remaining non-training scientific / manuscript / evidence gaps for the K-Bound paper **without** touching the active PACS (seeds 1–2) / ImageNet-R (seed 3) run, without launching training, without engineering-hardening, and without committing. This report is an honest account: it separates *verified-already-correct*, *defects fixed this session*, and *remaining / deferred* items, and it is explicit about what the sandbox could and could not execute.

---

## 1. Active run — untouched (safety)

The closure run is in progress. Verified from `experiments/kbound/results/closure_logs/pacs_imagenetr_closure.log`: `run_name=imagenetr_protocol_d_seed3_v1`, "PACS missing seeds 1 and 2 (locked seed-0 operating point)". Off-limits paths confirmed absent/partial and **not read as evidence, not modified**:

- `experiments/kbound/results/pacs_seed1.json`, `pacs_seed2.json` — do not exist yet.
- `experiments/kbound/results/pacs_multiseed_v1/` — does not exist yet.
- `experiments/kbound/results/imagenetr_protocol_d_seed3_v1/` — exists, empty (being written).
- `experiments/kbound/results/per_cell/pacs_*_percell.json` — partial (written during session); not touched.
- `experiments/kbound/results/closure_logs/pacs_imagenetr_closure.log` — read-only (path identification only).

**No active training process or output was modified. No commit or push was performed.** PACS and ImageNet-R Protocol-D multi-seed remain explicitly **pending** in the ledger, manuscripts (`kbound_short.tex` l.1099 "PACS and ImageNet-R remain incomplete diagnostics"), and the claim matrix.

---

## 2. Defects found and fixed this session

### 2.1 Broken references in `kbound.pdf` (real, PDF-visible) — FIXED
`kbound.log` (build of 2026-07-21 10:53) contained **three undefined citations**, rendering as bold `[?]`:
- `tempora2026` (p.3, input line 313), `corradaemmanuel2024` (p.4), `gupta2021toplabel` (p.36).

Root cause: the active bibliography `paper/references_kbound_expanded.tex` (regenerated 2026-07-18) dropped these three `\bibitem`s while the long manuscript still `\cite`s them (the entries survived only in the *unused* `references_kbound_expanded_full.tex` and `references/refs.bib`).

Fix: added three **corrected, web-verified** `\bibitem`s to the active bibliography:
- **tempora2026** → S. Sreeram, Y. D. Kwon, C. Mascolo, *Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation*, **ICML 2026**, arXiv:2602.06136. (The `_full`/`refs.bib` entries had **fabricated authors** "Huh / Wong"; corrected. The paper is genuinely 2026, so the "forward-dated" concern is resolved, not by guessing but by verification.)
- **corradaemmanuel2024** → A. Corrada-Emmanuel, *The logic of NTQR evaluations of noisy AI agents…*, **arXiv:2312.05392, 2023**. The "NeurIPS 2024" venue could **not** be verified, so per the no-fabrication rule it is cited as a preprint; the three **merged arXiv IDs** (2312.05392 + 2409.11052 + 2412.16238) were split down to the single primary record (the "combined records" defect).
- **gupta2021toplabel** → C. Gupta, A. Ramdas, *Top-label calibration and multiclass-to-binary reductions*, **ICLR 2022**, arXiv:2107.08353 (key-year/venue mismatch resolved).

Also de-fabricated the same entries in the unused source files (`references_kbound_expanded_full.tex`, `references/refs.bib`) so a future regeneration cannot reintroduce the fabricated authors; the unverifiable NTQR venue is marked for review rather than asserted.

**Validation:** a minimal `article` document that `\input`s the active bibliography and cites all three keys builds cleanly (`latexmk` exit 0, PDF produced) with **zero undefined citations and no duplicate-`\bibitem`**. The full IEEEtran render must be re-run on the host (see §6).

### 2.2 D33 pre-registration not referenced at manuscript level — FIXED
D33's "pre-registered" wording is substantiated by `research_lock/CONTROLLED_MULTIMODAL_PROTOCOL_D33_v1.yaml` (`date_locked: 2026-06-15`, `SEALED_BEFORE_ANY_RESULT`; results.json written later that day), but that artifact was referenced only in an internal FINDINGS.md. Added an explicit reference in `kbound_short_appendix.tex` (App. D33) and added the prereg path to `KB-CLAIM-027.supporting_artifacts` in `claim_ledger.json`. Neither change alters a number or strengthens the claim.

### 2.3 Stale T9 evidence-audit finding — FIXED
`reports/T9_LOCAL_EVIDENCE_AUDIT_2026-07-21.md` still asserted `KBOUND_REMAINING_TODOS.md` "incorrectly says no saved streaming artifact exists" and listed the D33/streaming actions as pending; the TODO has since been reconciled. Annotated the finding as **reconciled** and marked recommended actions 1–2 **done** (kept as a dated record; did not rewrite history).

### 2.4 iWildCam streaming — concrete collapse added to short paper (optional, permitted)
Short paper already carried the correct disclaimer ("label-informed offline stress diagnostic; not a label-free KGA deployment result; no streaming empirical claim is made"). Added **one compact parenthetical** with the observed magnitude — native 35,360-image stream, continual TENT cumulative macro-F1 **0.02 vs 0.26** frozen, predictions degenerating to a handful of classes — faithful to the authoritative source `pilot_test_native_bs16.json`. The strong disclaimer is preserved adjacent.

---

## 3. Verified already-correct (no change needed)

- **Protocol D33 (Phase 4).** Numbers agree exactly across source → `paper/generated/kbound_result_manifest.json` → `kbound.tex` → `kbound_short_appendix.tex`: KGA 0.8568, always-fuse 0.5832, always-single-A 0.8536, oracle 0.8573; 9/119/2 adapt/freeze/abstain; observed false-adapt 0/130; paired-bootstrap superiority 1.0/1.0; 130 conditions. Wording is controlled-only ("controlled mechanism confirmation, not a natural multimodal benchmark or evidence of universal accuracy improvement"); small gain (+0.0032 over single-A) explicitly acknowledged; uncertainty method stated; reproduction script `experiments/kbound/controlled_multimodal_d33.py` exists; **not** in the natural-shift core panel.
- **iWildCam numbers (Phase 5).** `pilot_test_native_bs16.json`: N_full 35,370 / used 35,360 / 2,210 batches (bs16); frozen 0.2554 vs TENT 0.0219 macro-F1; Δ −0.2335, bootstrap CI [−0.2537, −0.2212], excludes zero, p=1.0; source ckpt `iwildcam_f0_erm/f0_resnet50_erm_seed0.pt`; native order; label-informed. Manuscripts make no streaming claim.
- **Claim ledger (Phase 3).** 28 claims, internally consistent with all boundaries: withdrawn = 004 (FA_c), 012 (jackknife+/distribution-free), 022 (Camelyon pooled beats-both), 023 (in-sample-radius mixed → superseded by 024 OOF), 050 (universal accuracy); no-harm = 020 (Office-Home), 021 (iWildCam); Camelyon retained as genuine-OOD no-harm; D33 = 027 controlled mechanism confirmation. Totals: 20 supported / 5 withdrawn / 2 no-harm / 1 pending.
- **Negatives (Phase 6).** FMoW Protocol L and Poverty Protocol L are **absent from both manuscripts** (honest nulls, not promoted; documented in the T9 audit). CIFAR-10.1 is framed as "locked diagnostic fail / no claim" (FA_u 0.167, FA_c 0.444, cross-seed transfer fails); ImageNet-R and PACS are "incomplete diagnostics"; RxRx1 single-model-seed and 3D-ADAM exploratory results are not promoted.
- **Overclaim scan (Phase 8).** No bare "first-to" novelty claim in either manuscript. "universal" appears only in disclaimers or as the *defined* pre-registered "universal-gate" pooled construct (KB-CLAIM-024, caveated "researcher-pooled; no single-dataset natural beats-both is claimed"). "distribution-free" is narrowed ("only asymptotically … jackknife+ is not claimed"). "maximal" is the defined strict-decision theorem. "finite-sample" is always conditioned on coverage/exchangeability. Abstract footnote: "no conditional-error guarantee is claimed."
- **Empirical language (Phase 7).** Manuscripts use "observed false-adapt", "empirical coverage", "no-harm", and scope "beats both" to the locked comparisons; abstention = retain/serve frozen fallback.
- **Figures (Phase 9).** All 31 `\includegraphics` targets across the main manuscripts and their inputs resolve on disk (no missing-figure defect). No figures were regenerated (constraint honored; no active PACS/ImageNet-R outputs used).

---

## 4. Theorem / Lean coverage (Phases 1–2)

- `formal/formal_audit.py --strict-core`: **PASS** — 53 kernel-checked theorem declarations present, forbidden proof-hole scan (no `sorry`/`admit`/`axiom`) PASS, 0 documented foundational-probability limits. (Source-level strict audit re-run in this session; the kernel `lake build` itself was performed previously on the host — `formal_audit_report.json`: `build_ok=true`, Wave 6 — and could not be re-run in the sandbox: no `lake`/`lean`.)
- `scripts/theory_audit_full.py`: **PASS (0 issues)** — label→proof→validator→artifact→claim wiring intact; no dangling TheoremMap labels.
- **Coverage language is clause-accurate.** `theory_v2/lean_formalization_appendix.tex` states the Lean layer certifies the "algebraic and finite-probability cores" while "continuous measure-theoretic hypotheses … remain stated assumptions bridged by validators (`val_*.py`)". `kbound.tex` l.427 explicitly notes kernel-checked conformal coverage does not exist "to our knowledge." The manuscript does **not** claim helper compilation verifies the complete probabilistic theorems.
- Not attempted: a new large formalization (would risk destabilizing the paper; out of scope per instructions).

---

## 5. Tests run (in Linux sandbox)

| Check | Result |
|---|---|
| `formal_audit.py --strict-core` | PASS (53 kernel-checked; no proof holes) |
| `theory_audit_full.py` | PASS (0 issues) |
| `test_claim_metric_semantics.py` | PASS |
| `test_unified_result_audit.py` | PASS |
| `test_no_leakage_protocols.py` | PASS except `results_source.json` absent (generated artifact — see §7) |
| `test_no_retroactive_confirmatory_language.py` | PASS |
| `test_no_test_selected_comparator.py`, `test_no_test_selected_rga_plus.py` | PASS |
| `test_no_fisher_seed_combination.py`, `test_positive_transfer_protocol_lock.py` | PASS |
| minimal bibliography build (3 fixed cites) | PASS — 0 undefined citations, no duplicate bibitem |
| `git diff --check` (my edits) | clean (no whitespace/conflict errors) |

Failures triaged as **out-of-K-Bound-scope (Elara sub-project, monorepo)** and therefore not addressed: `test_manuscript_claim_consistency` (missing `src/scripts/validate_manuscript_claims.py`, an intentional Elara "Phase 1.G" TODO), `test_primary_metrics_do_not_apply_polarity_flip` (`src/scripts/run_breakthrough_experiment.py`), `test_metrics_manifest_integrity::test_macros_file_exists` (`docs/research/generated/elara_verified_metrics_macros.tex`), `test_ensemble_inference_label` (`docs/research/PAPER_DRAFT_v1.tex`). **Sandbox/harness (must not run):** `test_imagenetr_protocol_d` (subprocess dry-run needs host venv and touches the active ImageNet-R run). `test_kga_decision_rule` needs `sklearn` (not installed).

---

## 6. PDF build & visual inspection

- Both deliverable PDFs (`kbound_short.pdf`, `kbound.pdf`) use `\documentclass{IEEEtran}`. The sandbox TeX Live lacks `IEEEtran.cls`; CTAN is not reachable (network allowlist) and there is no root/`sudo`, so a **full render could not be produced in the sandbox**. This is an environment limitation, not a manuscript defect.
- What *was* validated in-sandbox: the bibliography fix compiles and resolves all three previously-undefined citations; all figure targets resolve; `git diff --check` is clean.
- **Full render + page-level visual inspection (abstract, contributions, theory, tables, D33, negatives, claim map, limitations, conclusion, references) must be run on the host** as part of the post-training rebuild (§8). Expected effect of this session's edits on the PDFs: the three `[?]` citations become resolved numbered references; App. D33 gains one prereg sentence; short-paper §"additional results" gains one parenthetical clause.

---

## 7. Remaining / deferred items (honest)

1. **`docs/research/kbound/results_source.json` absent.** A *generated* convenience file (built by `build_results_source.py` / `refresh_results_source_locked.py`) consumed by `make_tables.py` etc.; one K-Bound leakage test reads it without a skip guard and fails when absent. **Deliberately not regenerated now** — regenerating mid-run would ingest partial PACS/ImageNet-R data. It is produced as part of the post-training regeneration (§8). Authoritative claim numbers do not depend on it (they trace to the sealed lock artifacts).
2. **Host-side full PDF render + visual page sweep** (§6) — deferred to host.
3. **Kernel `lake build`** — not re-run (no Lean toolchain in sandbox); source-level strict audit passed and the prior host build is recorded PASS.
4. **Elara sub-project test failures** (§5) — out of scope for this K-Bound task; flagged for the Elara owners.
5. **Stray validation stub:** `docs/research/kbound/paper/_bibcheck.tex` (237 B, untracked, not referenced by any build). The workspace blocks bash deletes and file-delete permission was declined, so please remove manually: `rm docs/research/kbound/paper/_bibcheck.tex`.

No unsupported claim was strengthened. No withdrawn claim was restored. Camelyon17 remains "reconciled no-harm."

---

## 8. Exact post-training command (run on host, after the active run completes)

> Confirm the exact per-seed filenames/flags against your closure runner; the script names below are the repository's real entrypoints. Run from repo root. Do **not** run until PACS seeds 1–2 and ImageNet-R seed-3 have finished writing.

```bash
cd /Users/pratik_n/Documents/AutoML_Flagship_V8
K=docs/research/kbound

# 1. Validate each completed closure seed (fails closed on partial/mislabeled output)
python3 $K/scripts/validate_closure_seed.py --file experiments/kbound/results/pacs_seed1.json --seed 1
python3 $K/scripts/validate_closure_seed.py --file experiments/kbound/results/pacs_seed2.json --seed 2
python3 $K/scripts/validate_closure_seed.py --run-dir experiments/kbound/results/imagenetr_protocol_d_seed3_v1 --seed 3

# 2. Aggregate PACS seeds 0–2 (and the ImageNet-R Protocol-D multiseed)
python3 $K/scripts/aggregate_pacs_multiseed.py \
    --seed0 experiments/kbound/results/pacs_seed0.json \
    --seed1 experiments/kbound/results/pacs_seed1.json \
    --seed2 experiments/kbound/results/pacs_seed2.json \
    --out   experiments/kbound/results/pacs_multiseed_v1/pacs_multiseed.json
python3 $K/scripts/multiseed_aggregate.py --track ImageNet-R \
    --glob "experiments/kbound/results/imagenetr_protocol_d_*seed*/*.json"

# 3. Regenerate authoritative results source, manifests, tables, figures
python3 $K/scripts/refresh_results_source_locked.py          # results_source.json (fixes §7.1)
python3 $K/scripts/01_build_manifests.py                      # result manifest(s)
python3 $K/scripts/make_tables.py                            # -> paper/generated/kbound_numbers.tex
python3 $K/scripts/seal_nine_track_lock.py                   # re-seal locked evidence (PACS/ImageNet-R now complete)

# 4. Rebuild both PDFs
cd $K && latexmk -pdf kbound_short.tex && latexmk -pdf kbound.tex && cd -

# 5. Re-run the final audit gate
python3 $K/formal/formal_audit.py --strict-core
python3 $K/scripts/theory_audit_full.py
python3 $K/scripts/unified_result_audit.py --strict-explicit
python3 -m pytest -q docs/research/kbound/tests tests/test_no_retroactive_confirmatory_language.py
```

After step 3, flip the PACS and ImageNet-R Protocol-D entries in `claim_ledger.json` / the manuscripts from **pending** to their earned verdict **only if** the aggregated result and its uncertainty pass the locked criterion (no-harm vs beats-both), using the established no-harm/beats-both wording rules.

---

## 9. Files changed this session

**Edited (tracked):**
- `docs/research/kbound/kbound_short.tex` — iWildCam collapse parenthetical (Phase 5).
- `docs/research/kbound/kbound_short_appendix.tex` — D33 pre-registration reference (Phase 4).
- `docs/research/kbound/claim_ledger.json` — KB-CLAIM-027 `supporting_artifacts` += prereg yaml (Phase 4).
- `docs/research/kbound/paper/references_kbound_expanded.tex` — **+3 corrected `\bibitem`s** (fixes 3 undefined citations) (Phase 10).
- `docs/research/kbound/paper/references_kbound_expanded_full.tex` — de-fabricated tempora/corrada entries (Phase 10).
- `docs/research/kbound/paper/references/refs.bib` — de-fabricated tempora/corrada entries (Phase 10).
- `docs/research/kbound/reports/T9_LOCAL_EVIDENCE_AUDIT_2026-07-21.md` — reconciled stale streaming finding + marked actions done (Phase 5/11).
- `docs/research/kbound/reports/THEORY_AUDIT_FULL.md` — regenerated by running the audit (timestamp only; verdict unchanged **PASS**).

**Created:**
- `docs/research/kbound/reports/NONTRAINING_CLAIM_MATRIX_2026-07-21.md` — reviewer claim matrix (deliverable).
- `docs/research/kbound/reports/NONTRAINING_CLOSURE_REVIEW_2026-07-21.md` — this report.

**To remove manually (§7.5):** `docs/research/kbound/paper/_bibcheck.tex`.

**Untouched:** all active PACS/ImageNet-R paths (§1); `.idea/`, `.virtual_documents/`; every unrelated file. No `git add`, commit, or push performed.

---

## 10. Addendum — nine-track lock completed (2026-07-22)

The PACS (seeds 1–2) and ImageNet-R (seed 3) run finished after §1–§9 were written; the user requested the lock. Completed and verified:

- **Fail-closed validation** (`validate_closure_seed.py`): PACS seed 1 and seed 2 VALID (4 domains × 18 cells each, exact protocol override); ImageNet-R seed 3 VALID (10 backbones × 12 conditions, correct seeds). The `_partial.json` files are stale pre-completion checkpoints (predate the final `result_*.json`).
- **Aggregates confirmed** (already produced by the closure run): `pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json` (3 seeds); `imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` (4 seeds, seeds [0,1,2,3]).
- **Verdicts — no promotion.** PACS = "completed three-seed null action-safety diagnostic; no beats-both claim." ImageNet-R = "0/10 CI-supported beats-both" (`beats_both_by_candidate` False for all 10 backbones; Holm α=0.05, nboot=10⁴). Both lock as honest diagnostics; neither is promoted to a win.
- **Numbers verified against authoritative aggregates (exact):** PACS mean regret KGA/adapt/freeze **0.0431 / 0.0176 / 0.0446**, FA_u **2/216**; ImageNet-R mean-across-backbone regret **0.0112 / 0.0064 / 0.0325**, FA_u **1/480**, beats-both **0/10**. All match `kbound.tex`/`kbound_short.tex` (already updated to 4-seed/3-seed completed framing) and the ledger's new **KB-CLAIM-041** (PACS) / **KB-CLAIM-042** (ImageNet-R), both `supported` with beats-both/win explicitly forbidden.
- **Sealed** (`seal_nine_track_lock.py`): `LOCK_SEAL.json` + `.sha256` + `research_lock/NINE_TRACK_LOCK_SEAL_v1.yaml` rewritten. `--verify` OK across **10 tracks**. `pacs_multiseed` (4 files) and `imagenet_r_D` (41 files) now `locked_diagnostic_null`; **`cifar10c_sar` remains `not_locked` (withheld)**; Camelyon unchanged `no-harm`; no withdrawn claim restored.
- **Manifest reconciled:** `paper/generated/kbound_result_manifest.json` `pacs`/`imagenet_r_D` blocks updated from "incomplete/not-in-seal" to the sealed complete state with the verified numbers and claim-id links.
- **Gates re-run post-lock:** formal audit PASS, theory audit PASS (0 issues), `unified_result_audit --strict-explicit` exit 0, seal `--verify` OK.

**Still host-only:** rebuild `kbound_short.pdf` + `kbound.pdf` (IEEEtran unavailable in sandbox) and regenerate `results_source.json`/`kbound_numbers.tex`; both are in the §8 command. **No commit/push performed** — the lock artifacts and edits are on disk, staged for your review.
