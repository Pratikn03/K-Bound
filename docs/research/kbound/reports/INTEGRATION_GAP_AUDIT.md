# K-Bound Integration Gap Audit

**Created:** 2026-06-25  
**Scope:** Artifacts under `experiments/`, `docs/research/kbound/`, `research_lock/` vs live papers `kbound_short.tex` + `kbound.tex`  
**Repro baseline:** `scripts/reproduce_submission.sh` (lightweight; no GPU reruns)

Use this when asking: *“What wins/scripts/figures exist on disk that we forgot to put in the paper?”*

---

**Status:** P0–P4 executed 2026-06-25. Stale locks deprecated; papers synced; repro hardened.

---

## Executive summary (post P0–P4)

| Bucket | Integrated | Partial | Missing / stale |
|--------|:----------:|:-------:|:-------------:|
| Headline WIN results | 6 | 4 | 3 stale manifests |
| Scripts in repro pipeline | 4 | — | 25+ offline |
| Figures (30 PNG on disk) | 19 in TeX | 2 appendix-only | 9+ orphan |
| `claim_ledger` claims | 12 | 2 | 2 pending |
| Protocol locks with results | 8 | 3 | 4 unrun |

**Top risks:** stale `KBOUND_HEADLINE_FINDINGS.json` still lists Camelyon beats-both; **long paper** lacks POEM/AETTA head-to-head; D33 controlled multimodal WIN not cited; assumption_audit + physical R2 unrun.

---

## A. Wins / results ON DISK but not fully in papers

### A1 — Integrated (keep)

| Artifact | Verdict / content | Paper |
|----------|-------------------|-------|
| `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` | Tent/EATA beats-both | `tab:decisive` |
| `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json` | WIN vs POEM/AETTA | `kbound_short` `tab:headtohead-poem-aetta` |
| `mixed_protocol_oof_v2/` + `KBOUND_MIXED_STREAM_v2.json` | OOF beats-both aggregate | §`sec:mixedstream` |
| `results_source.json` → `kbound_numbers.tex` | OOF natural-shift numbers | Office-Home / iWildCam tables |
| `KBOUND_WIN_BOOTSTRAP_CIS_oof.json` | Honest OOF CIs | natural-shift paragraph |
| `witness/witness_clean.json` | Impossibility witness | long paper fig |

### A2 — Partial (exists, under-cited)

| Artifact | Content | Gap | Action |
|----------|---------|-----|--------|
| `HEADTOHEAD_RESULTS_cifar10c_eata_secondary.json` | WIN | Not in TeX | One robustness sentence in short paper appendix |
| `HEADTOHEAD_RESULTS_cifar10c_tent_eata_pooled.json` | WIN | Not in TeX | Optional appendix row |
| `controlled_multimodal_d33/FINDINGS.md` | **STRONG beats-both** (130 cond.) | Only in `manuscript/`, not live papers | Add appendix § or one paragraph (mechanism confirmation) |
| `cifar101_multiseed_v1/` | Low-margin CIFAR-10.1 | Long paper only | OK |
| `imagenetr_protocol_d_multiseed_v1/` | Weak / split-specific | Long paper limits | OK |
| `camelyon17_diagnostics_resolved_v1/` | Diagnostic ladder | Short appendix partial | OK |
| `kga_elara_integrated_v1/` | ELARA retrospective null | Appendix `\IfFileExists` | Non-headline |

### A3 — Not integrated (exploratory; do NOT promote)

| Artifact | Why skip |
|----------|----------|
| `win_loop_v1/`, `win_finder_v1/`, `win_finder_imagenetr_rxrx1_v*` | Dev screening; superseded by Protocol M/H v2 |
| `officehome_full_FINAL/VERDICT_test.json` | Pre-M exploratory |
| `namedcond_3dadam/FINDINGS.md` | Point beats-both, CI fails |
| `fmow_protocol_L_v1/` | `not-cleared`, high FA |
| `poverty_protocol_L_dev/` | dev-screen-stop only |

### A4 — Stale / dangerous (contradicts live paper)

| Artifact | Problem | Action |
|----------|---------|--------|
| `research_lock/KBOUND_HEADLINE_FINDINGS.json` | Lists Camelyon G as `headline_natural_wins` beats-both | **DEPRECATE** or rewrite to no-harm |
| `research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json` | In-sample-radius CIs | Banner: superseded by `_oof.json` |
| `research_lock/KBOUND_MIXED_STREAM_v1.json` | In-sample 13–24× multipliers | Already withdrawn in paper |
| `research_lock/KBOUND_6_DATASET_PANEL_v2.yaml` | Camelyon `robust_beats_both` | Update to v3 no-harm framing |

### A5 — Registered but NOT RUN

| Protocol | Output path | Claim | Paper |
|----------|-------------|-------|-------|
| `STRESS_GRID_STRICT_PROTOCOL_A_v2` | `stress_grid_strict_v2/` missing | — | Not mentioned |
| `assumption_audit_v1` | `results/assumption_audit_v1.json` missing | KB-CLAIM-040 pending | Not in TeX |
| `edge_real_phone_v1` | R2 RESULT PENDING | KB-CLAIM-030 pending | Tables placeholder OK |
| `spotlight_pilot` | not run | — | Out of scope |

---

## B. Scripts that produce numbers but are NOT in `reproduce_submission.sh`

### In repro today
- pytest (leakage, claims, edge)
- `gate_baseline_comparison.py --selftest`
- `make_tables.py`
- HEADTOHEAD primary JSON presence check

### Should add to repro (cached verify only)
| Script | Output |
|--------|--------|
| `mixed_stream_kbound.py` | `mixed_protocol_oof_v2_result.json` |
| `build_results_source.py --check-only` | validates `results_source.json` |
| Check `LOCKED_ANALYSIS_RESULTS.json` exists | stress grid |

### Headline offline (document, don't auto-run)
| Script | Produces |
|--------|----------|
| `experiments/kbound/poem_aetta/run_all_headtohead.sh` | 3 HEADTOHEAD JSONs |
| `cifar_tent_mps_v2.py` | stress grid records (GPU) |
| `run_protocol_dev_lock.py` | M/H v2 protocol results |
| `make_submission_figures.py` / `04_make_figures.py` | short-paper figures |
| `edge/scripts/run_edge_publication_pipeline.sh` | physical R2 |
| `controlled_multimodal_d33.py` | D33 WIN |

### Exploration only (never paper headline)
`run_win_loop.py`, `find_kbound_wins.py`, `run_hard_dataset_win_loop.py`, `bootstrap_win_cis.py` (in-sample)

---

## C. Figures: disk vs `\includegraphics`

### `kbound_short.tex` (5 figures) — all integrated
`fig_decision_flow`, `fig_frontier_schematic`, `fig_decisive_pareto_cifar10c`, `fig_alpha_coverage`, `fig_natural_forest`

### `kbound.tex` + appendix (19+ figures) — integrated
Witness, mixed, certificate, CIFAR collapse, decisive Pareto ImageNet-C, theory_v2 v1–v4, conj1_closure, app_frontier (separate appendix tex)

### On disk, NOT in live `kbound_short.tex` or `kbound.tex`

| Figure | Suggested use |
|--------|----------------|
| `fig_gamma_meter.png` | Cut or theory appendix |
| `fig_kfrontier.png` | Theory supplement |
| `fig_phase_diagram.png` | Cut |
| `fig_regret_summary.png` | Only in draft `paper/kbound.tex` |
| `fig_feasibility.png` | Multicandidate appendix |
| `fig_lecam_bound.png` | Rates appendix |
| `fig_subsumption.png` | Cut |
| `fig_reach_table.png` | Cut |
| `fig_labelshift_boundary.png` | Regression extension |
| `fig_regression_boundary.png` | Regression extension |
| `fig_regret_decomposition.png` | Appendix |
| `fig_krates.png` | Rates appendix |
| `fig_architecture.{png,svg}` | Consider short-paper method fig |
| `theory_v2/fig_minimax_optimality.png` | If minimax section kept |
| `theory_v2/realdata/fig_p*.png` | Optional real-data probes |

**Note:** No figure for POEM/AETTA head-to-head — table only (OK).

---

## D. Long paper (`kbound.tex`) gaps vs short paper

| Item | `kbound_short.tex` | `kbound.tex` |
|------|:------------------:|:------------:|
| POEM/AETTA head-to-head WIN | yes | **no** |
| Mixed-stream OOF §`sec:mixedstream` | yes | yes |
| Guarantee box (FA_u vs FA_c) | yes | partial |
| `tab:headtohead-poem-aetta` | yes | **missing** |

**Action:** Sync head-to-head subsection + table into long paper.

---

## E. Claim ledger vs paper narrative

| ID | Status | In paper? |
|----|--------|-----------|
| KB-CLAIM-001–003, 025 | supported | yes |
| KB-CLAIM-004, 012, 022, 023, 050 | withdrawn | yes (as limitations) |
| KB-CLAIM-010, 011 | supported | yes |
| KB-CLAIM-020, 021 | no-harm | yes |
| KB-CLAIM-024 | mixed OOF | yes |
| KB-CLAIM-026 | HEADTOHEAD WIN | short only |
| KB-CLAIM-030 | pending R2 | placeholder OK |
| KB-CLAIM-040 | pending audit | **not in TeX** |

**Missing claim:** D33 controlled multimodal (no ledger entry yet).

---

## F. Priority action queue

### P0 — Integrity (do first)
1. Deprecate `KBOUND_HEADLINE_FINDINGS.json` (stale Camelyon win).
2. Banner `KBOUND_WIN_BOOTSTRAP_CIS.json` and `KBOUND_MIXED_STREAM_v1.json`.
3. Never cite `docs/experiments/.../edge_real_phone_v1` dev replay as R2.

### P1 — Paper completeness
4. Sync POEM/AETTA head-to-head into `kbound.tex`.
5. Add D33 one-paragraph + optional appendix table (`CONTROLLED_MULTIMODAL_PROTOCOL_D33`).
6. Mention HEADTOHEAD EATA secondary as robustness (one sentence).

### P2 — Repro hardening
7. Extend `reproduce_submission.sh`: mixed OOF JSON check, `LOCKED_ANALYSIS` check, `build_results_source --check-only`.
8. Document `run_all_headtohead.sh` in `REVIEWER_REPRO_PACKET.md`.

### P3 — Open science (human/GPU)
9. Physical R2 capture OR keep RESULT PENDING.
10. Run `assumption_audit_v1` OR remove KB-CLAIM-040 from ledger.
11. Optional: `stress_grid_strict_v2`, official-repo POEM/AETTA arm.

### P4 — Cleanup (low)
12. Archive orphan figures / win_finder dirs.
13. Fix `kbound.tex` POEM citation TODO (author initials).

---

## G. Quick command map

```bash
# Head-to-head (all 3 sets, ~7s CPU)
PY=.venv/bin/python bash experiments/kbound/poem_aetta/run_all_headtohead.sh

# Mixed OOF aggregate
.venv/bin/python docs/research/kbound/scripts/mixed_stream_kbound.py

# Lightweight repro
bash docs/research/kbound/scripts/reproduce_submission.sh

# Full figure regen (manual)
cd docs/research/kbound && .venv/bin/python scripts/make_submission_figures.py
```

---

*Re-run this audit after any new experiment directory appears under `experiments/kbound/results/`.*
