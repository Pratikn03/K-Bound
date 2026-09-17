# K-Bound — figure-by-figure guide

What every figure in `kbound.tex` shows, the claim it supports, and the script that makes it.
Status as of 2026-06-21. The paper uses **20 figures** (15 in `figures/`, 5 in `theory_v2/`).
The 23 unused PNGs were moved to `figures/_archive/` (nothing deleted).

Legend: **WHAT** = what's drawn · **ROLE** = the claim it carries · **GEN** = generating script.

---

## A. Method & controlled-regime figures (body, §Experiments)

### 1. `fig_witness_clean.png` — Fig.~\ref{fig:witness}
- **WHAT:** Two synthetic worlds with an *identical* evidence law `Z` (KS p>0.05 on every feature) but *exactly opposite* true adaptation benefit; KGA abstains on 100%.
- **ROLE:** The visual proof of the **impossibility theorem** (Thm `imp`) — when observable evidence is identical but the benefit sign flips, abstention is the only correct action. This is the paper's keystone "you *cannot* know" picture.
- **GEN:** `scripts/kbound_full_experiments.py` (savefig L246).

### 2. `fig_mixed_policies.png` + `fig_mixed_decisions.png` — Fig.~\ref{fig:mix}
- **WHAT:** Left — KGA's accuracy beats always-freeze and ties always-adapt on a mixed stream. Right — KGA's decisions broken out by *true* regime (adapt on detectable failure; abstain/freeze on clean).
- **ROLE:** Shows KGA captures **best-of-both** when helpful and harmful conditions are interleaved — the deployment story in miniature.
- **GEN:** `scripts/mixed_regime_experiment.py` (L164 / L154).

### 3. `fig_kbound_harmful.png` + `fig_certificate.png` — Fig.~\ref{fig:harm}
- **WHAT:** Left — in a harmful regime (the reliability-fusion adapter hurts ~80% of tasks), KGA matches always-freeze ≈ oracle and avoids the harmful adapt path. Right — the certificate's `Δ̂ ± ε` interval vs. ground-truth benefit on the clean suite.
- **ROLE:** **Harmful-regime safety** + evidence the certificate interval is calibrated (brackets the truth).
- **GEN:** `scripts/kbound_harmful_regime.py` (L103) · `scripts/knowability_experiment.py` (L206).
- **NOTE:** ✅ *Just cleaned* — left panel relabeled (`elara_fuse`/`auto_select` → "reliability-fusion"/"best-val model") and regenerated at 300 DPI; numbers verified byte-identical to the committed `kbound_harmful_results.json`.

### 4. `fig_ablation_evidence.png` + `fig_alpha_coverage.png` + `fig_regression.png` — Fig.~\ref{fig:extra}
- **WHAT:** Left — drop-one evidence ablation (detector *disagreement* dominates: removing it raises regret 0.037→0.094). Middle — α sweep tracing the safety/coverage tradeoff. Right — synthetic regression under covariate shift: KGA matches the oracle by *refusing* high-variance importance weighting.
- **ROLE:** (a) which evidence feature matters (supports Thm `disagree`), (b) the α knob behaves as the theory predicts, (c) the certificate's safety **generalizes beyond anomaly detection** to regression.
- **GEN:** `scripts/kbound_full_experiments.py` (L145 / L150 / L197).

### 5. `fig_decision_flow.png` — Fig.~\ref{fig:flow}
- **WHAT:** The KGA schematic: unlabeled batch → evidence `Z=φ(·)` → estimate `Δ̂,ε` → certificate `Δ̂±ε vs 0` → ADAPT / FREEZE / ABSTAIN.
- **ROLE:** The **method diagram** — the first thing a reviewer reads; the clearest one-look statement of the certificate logic. Now the standalone figure.
- **NOTE:** ✅ The former right panel (`fig_regret_summary`) was **dropped** this pass: it had no committed generator, internal labels, no baseline bars, and its bars didn't reconcile with the paper's own CIFAR table. The regret-to-oracle numbers are already presented, properly sourced, in the reviewer-safe summary table (`tab:regime-summary`). `fig_regret_summary.png` moved to `figures/_archive/`.

---

## B. CIFAR experiment figures (body, §Experiments)

### 6. `fig_cifar_tent_online.png` — Fig.~\ref{fig:online}
- **WHAT:** Per-window accuracy on a harm-dominated non-stationary stream (pink = catastrophic "trap" windows). Continual Tent (always-adapt) accumulates damage and **collapses**; KGA freezes/abstains on traps and never collapses.
- **ROLE:** The headline **"catastrophic collapse avoided"** visual — the strongest single picture of why a gate matters.
- **GEN:** `scripts/cifar_tent_online.py` (L149).

### 7. `fig_cifar10c_collapse.png` + `fig_cifar10c_crossover.png` — Fig.~\ref{fig:c10c}
- **WHAT:** Left — severity-5 accuracy by corruption; both Tent and KGA recover the frozen baseline (no per-corruption collapse). Right — knowability crossover map: adaptation headroom (oracle gain over freeze) by corruption × severity.
- **ROLE:** The **honest control** — per-corruption CIFAR-10-C is *helpful-dominated*, so collapse is a continual/mixed-stream phenomenon, not a single-corruption one. Shows where Δ is large.
- **GEN:** `scripts/cifar_tent_mps_v2.py` (the CIFAR-10-C decisive runner).

### 8. `fig_decisive_pareto_cifar10c.png` — Fig.~\ref{fig:decisive} (stress grid)
- **WHAT:** CIFAR-10-C stress grid, mixing-ratio Pareto — mean regret-to-oracle vs. harmful fraction; KGA Pareto-dominates both fixed policies across the grid.
- **ROLE:** The **decisive harmful-regime win** (Tent/EATA beat-both; Holm-significant; 0 false-adapt) in one frontier picture.
- **GEN:** `scripts/cifar_tent_mps_v2.py` (L281).

---

## C. ImageNet-scale figure (appendix)

### 9. `fig_decisive_pareto_imagenetc.png`
- **WHAT:** The same mixing-ratio Pareto at ImageNet scale (ImageNet-C, ResNet-50): KGA Pareto-optimal on the harmful (SAR) arm.
- **ROLE:** **Scale corroboration** of the decisive result — part of the seven-dataset core panel's ImageNet-C entry.
- **GEN:** `scripts/cifar_tent_mps_v2.py` (L281, `imagenetc` tag).

---

## D. Theory-validation figures (appendix, `theory_v2/`)

These are synthetic *validators*: each checks that a theorem's predicted quantity matches the empirical one.

### 10. `theory_v2/fig_v1_flip_witness.png`
- **WHAT/ROLE:** One-bit identification witness for the dichotomy (Thm `conj1-dichotomy`(ii)) — the label-complement flip the residual orientation must resolve.

### 11. `theory_v2/fig_v2_identification_rate.png`
- **WHAT/ROLE:** Empirical identification rate vs. sample size — the gate identifies the benefit sign at the predicted rate.

### 12. `theory_v2/fig_v3_audit.png`
- **WHAT/ROLE:** The budget audit (Thm `conj1-dichotomy`(iv)) — the certificate is *sound at level α* on the audited conditions.

### 13. `theory_v2/fig_v4_rate.png`
- **WHAT/ROLE:** Evidence-channel rate (Thm `ev-rate`) — empirical minimax error tracks the theoretical bound.

### 14. `theory_v2/fig_conj1_closure.png`
- **WHAT/ROLE:** Closing Conjecture 1 (Thm `conj1-dichotomy`(iii)) — the general-position swap construction.

**GEN (group D):** the theory validation scripts in `scripts/` (`theory_extensions_validation.py`, `knowability_*_validation.py`).

---

## Resolution / DPI status

Figures were saved at **dpi=130** by the experiment scripts (≈650–910 px wide for most). Because most are placed at sub-column width (0.32–0.72 `\linewidth`), their *effective* print DPI is mostly acceptable — this is a polish item, not a defect.

- ✅ `fig_kbound_harmful` regenerated at **300 DPI** (2280×1260) this pass.
- The other 13 experiment figures need their scripts re-run on your Mac (torch + data) to refresh at 300 DPI — change `dpi=130`→`dpi=300` in the relevant `savefig` call and re-run. Low priority given sub-column placement.

## Honest punch list (none are "make it 80" — all are polish)

1. **`fig_regret_summary`** — ✅ dropped this pass (weak, untraceable, redundant with the summary table); decision-flow panel now stands alone.
2. **DPI bump** on the remaining experiment figures — cosmetic; do at camera-ready if at all.
3. **Orphans** — already moved to `figures/_archive/` ✓.

The 20 figures are clear, on-message, and honest. They support the claims without over-reaching — exactly what an 80-level paper needs. No figure work changes the level; it only changes polish.
