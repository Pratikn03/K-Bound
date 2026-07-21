# K-Bound Phase 6 — Data-Leakage & Timing/Ordering Audit
Read-only audit. Scope: split disjointness, in-sample-radius defect, certificate/oracle timing, multi-seed pooling.
Auditor replay: `numpy` on the canonical raw JSONs (results deterministic; matches `/opt/anaconda3/bin/python3` path). Every finding is quoted `file:line`.

## VERDICT: PASS (clean). No live promoted track computes ε in-sample on the cells it scores. No promoted number uses target labels to choose ε or the decision threshold. KB-CLAIM-022 confirmed quarantined.

---

## (a) Per-track split / ε-source table

ε is *always* fit on the calibration/dev partition (or out-of-fold via leave-one-cell-out); the true test benefit `B` never enters the ε computation or the decision, only the post-hoc score.

| Track (promoted) | Calibration / dev split | Test split scored | Where ε comes from | Scorer (file:line) | Disjoint? |
|---|---|---|---|---|---|
| CIFAR-10-C stress | leave-one-cell-out (per cond., within seed) | the held-out cell (jackknife) | q0.9 of **LOO** residuals `\|Δ̂₋ᵢ(Zᵢ)−Δᵢ\|` | `scripts/cifar_tent_mps_v2.py:143-156` | PASS (cross-fit; no cell's estimator saw it) |
| ImageNet-C SAR (5-seed) | per-seed leave-one-cell-out (27 cells) | same seed's 27 cells; **decisions pooled** across seeds | exact rank ε **per seed** on that seed's OOF residuals | `scripts/g8_canonical_pooling.py:6-17`; `kbound_pkg/kbound/certificate.py:29-66` | PASS (per-seed cross-fit) |
| PACS (LODO, 4 targets) | one source domain (`calibration_domain`) | target domain, 18 cells | ε from source-domain residuals | `results/win_hunt_v5/pacs_aggr/pacs_result.json` (per_domain) | PASS (source≠target) |
| Camelyon17 OOD | dev seeds {0,1} | OOD test-only seeds {2,3,4}, n=18 | **LOO OOF** ε on dev only | `scripts/analyze_F.py:186-199`; `audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json` | PASS (disjoint by seed) |
| iWildCam H v2 | calib seed {0} | held-out seed {1}, n=72 | LOO OOF ε on cal (`eps_global=0.02937`) | `scripts/score_kbound_holdout.py:75-91` / `analyze_F.py` | PASS (disjoint by seed) |
| Office-Home M v2 | calib target-**val** {0,1} | target-**test** {0,1}, n=35 | LOO OOF ε on cal (`eps_global=0.00102`) | `scripts/score_kbound_holdout.py:64-106` | PASS (different files: val vs test) |
| RxRx1 J | dev seeds {0..4} | test seeds {5..9}, n=60 | LOO OOF ε on dev | `scripts/analyze_F.py:186-199`; `results/rxrx1_protocol_J_v1/analyze_F_results.json` | PASS (disjoint by seed) |

Manuscript §method documents this honestly: "we fit the benefit estimator and its radius **out of fold**, so the data used to fit Δ̂ and the data used to calibrate its residual are disjoint for every point" (`kbound_short.tex:305-306`); natural-shift protocols "fit once on a dev split and the untouched held-out target-domain conditions are scored a single time" (`:312-315`); "Labels enter only through the calibration split; the … held-out test scorer never use[s] target labels for the adapter, ε, evidence, or rule selection" (`:316-317`). Exact rank `k=min{n_cal, ⌈(n_cal+1)(1−α)⌉}` (`:318-319`), matching `certificate.py:65`.

## (b) Leakage found: NONE

The two scoring scripts carry an explicit in-fold guard (the fix for the June-2026 defect):
- `analyze_F.py:186-193` — "Out-of-fold (leave-one-out) residuals for the conformal radius -> no in-sample leakage. (The in-sample radius was ~10x too small on small dev sets; see audit 2026-06.)" ε is `conformal_rank_radius(resid_c)` where `resid_c` = LOO residuals on the **calibration** rows only (`:189-193, :198`); decisions are `decide_global(Bhat_t, eps)` on test (`:199`).
- `score_kbound_holdout.py:75-85` — same LOO guard on the calibration file; ε applied to a *separate* test file (`:89-91`).

Decision never sees the test label:
- `certificate.py:113-147` `decide(Bhat, eps)` takes only `(Bhat, ε)`.
- `analyze_F.py:118-135` `metrics()` and `run_wilds_camelyon17.py:62-93` `policy_metrics()` compute `false_adapt = mean(ADAPT & B≤0)` — true `B` used **for scoring only**; `dec` is produced independently by `decide_kga`/`decide_global`. `run_wilds_camelyon17.py:76`: `"FA_u_marginal": float(np.mean(adapt & (B <= 0)))`.

Replay evidence (ImageNet-C SAR, canonical raw JSONs under `results/win_hunt_v5_imagenetc_ms/pooled_5seed`):
- Per-seed exact-rank ε = [0.084, 0.108, 0.046, 0.084, 0.072]; per-seed pooled → KGA/adapt/freeze = **0.0264/0.0529/0.0319, FA_u=0.0000, beats-both=True**. Matches the promoted panel (`paper/generated/kbound_result_manifest.json` `imagenetc_sar` regret [0.02642, 0.05293, 0.03189], FA_u 0.0) and appendix Table `tab:imagenetc-perseed` (`kbound_short_appendix.tex:269-275`, per-seed 0.0108/0.0091/0.0128/0.0056/0.0154, pooled 0.0107).
- ε(seed0)=0.084 is substantial (not ~0), consistent with genuine leave-one-cell-out residuals rather than an over-fit in-sample radius.
- Decisions recomputed from `(b_hat, ε)` alone are identical; corrupting true `B` changes only the post-hoc FA_u score, never a decision.

## (c) KB-CLAIM-022 quarantine — CONFIRMED

- Ledger: `claim_ledger.json:159-173` — `status="withdrawn"`, `calibration_method="in_sample_radius"`, `test_split="pooled id_val (invalid)"`.
- Root cause is documented and reproduced: `camelyon_reconciliation/recon_results.json` re-runs the exact `analyze_F.run_split(dev{0,1}, test{2,3,4})` on domain slices:
  - `POOLED_test_val_idval` (the withdrawn artifact): `beats_both=true, preregistered_win=true, regret_kga=3.6e-05, n_test=54`. The "win" exists **only** because `id_val` (frac_harm `B<0` = **0.767**, mean_B = −0.0075) is pooled into the genuinely-helpful OOD test/val domains to manufacture a mixed regime.
  - `OOD_test_only` (the **promoted** reconciled result): `regret_kga=0.0, regret_adapt=0.0, regret_freeze=0.1381, false_adapt=0.0, beats_both=FALSE, n_test=18` → the manuscript's `0.0000/0.0000/0.1381; FA_u=0` no-harm (`kbound_short.tex:899`, `kbound_result_manifest.json` `camelyon17_ood`, source `audits/.../camelyon_reconciliation/`).
- No live promoted track shares the in-sample-radius defect. The withdrawn artifact path `archive/audit_only/camelyon17_protocol_G_pooled_beats_both` is **not materialized** on disk; no result JSON under `experiments/` asserts the pooled beats-both; the only surviving references (`uniform_scorer.py:14,108`, `claim_ledger.json`, `THEORY_TO_CODE_MAP.md`, `research_lock/mixed_protocol_oof_v2.yaml`) are documentation of the withdrawal. `uniform_scorer.py:108` still hard-fails any contaminated split → `WITHDRAWN`.

## (d) Certificate timing / multi-seed pooling

- Timing: α is fixed at 0.10 everywhere (`certificate.py:29`, `analyze_F.py:40`, `run_wilds_camelyon17.py:42`, all protocol JSONs); the threshold is the fixed constant 0 in `Bhat±ε`. Neither α nor the threshold is selected on test. Adapter/estimator/evidence selection is done on dev only (`iwildcam_protocol_H_v2/protocol_result.json` `dev_screen` cal_seeds[0]/eval_seeds[1]; locked adapter chosen before `heldout` seed{1} scored once).
- Multi-seed pooling (task 4): `g8_canonical_pooling.py:9-13` computes `eps = cexact(rho)` **inside** the per-file (per-seed) loop on that seed's own 27-cell OOF residuals, forms per-seed decisions from `b_hat`, then pools `allDec`/`allB` for the aggregate. It does **not** fit one ε across all 135 pooled cells. Replay contrast: a single pooled ε gives different numbers (SAR: KGA 0.0109, ε 0.0431) than the promoted per-seed pooling (KGA 0.0264), confirming the promoted path is per-seed.

## Prioritized fix list (all non-blocking; leakage list is empty)

1. (traceability, low) Ledger `claim_ledger.json:168` points KB-CLAIM-022 at `archive/audit_only/camelyon17_protocol_G_pooled_beats_both`, which does not exist on disk. Either restore the quarantined artifact there or repoint to the live rationale (`audits/integrity_2026-06-20/camelyon_reconciliation/recon_results.json`). Quarantine is otherwise intact.
2. (traceability, low) RxRx1 J promoted regret_adapt is `0.2587` (`kbound_short.tex:902,940`; manifest) but `results/rxrx1_protocol_J_v1/analyze_F_results.json` records `0.2531`; the promoted value is the 5-seed real-ckpt rerun. Not a leakage issue (adapt_rate=0, all-freeze, FA_u=0 either way) — just confirm the manifest cites the 5-seed artifact, not the older J file.
3. (documentation, low) `run_wilds_camelyon17.py:45-59` `decide_kga` computes ε in-pool (LOO over all seeds) for the **raw** run artifact `wilds_camelyon17_kga.json`; that raw file is an *input* to `analyze_F` re-scoring, never a promoted number. Worth a one-line note in the reproduction packet so a reader does not mistake the raw pooled-LOO ε for the headline.
