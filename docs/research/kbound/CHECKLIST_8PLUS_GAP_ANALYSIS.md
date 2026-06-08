# K-Bound — "8+ / breakthrough" checklist: gap analysis & closure status

**Date:** 2026-06-05  ·  **Verified hands-on** (clean Linux env, numpy/scipy/sklearn; all
theory validators re-run; all result JSONs audited; paper recompiled).

Legend: ✅ done & verified · 🟡 partial · 🔧 needs the Mac/GPU run · ⬜ external step.

> **Bottom line.** Every *theory* item (A1–A6), every *metric* (C), the *ablations* (E),
> the *statistics* (F), and *claim discipline* (G) are **done and re-verified**. The one
> substantive item still open for the empirical headline is the **deep-TTA run on real
> CIFAR-10-C with Tent/EATA/SAR (B1)** — which is now a **one-command Mac runner**
> (`scripts/run_decisive_tta.sh`); its torch-free analysis core was smoke-tested here and
> passes (`beats_both=True`). The honest open *theory* residual is the label-free
> *bracketing* conjecture (Conj. 1) — correctly left open, not force-closed.

---

## A. Theory

| # | Item | Status | Evidence / where |
|---|------|--------|------------------|
| A1 | Le Cam two-point lower bound | ✅ | Thm 2 `thm:imp-quant` in `kbound.tex`; `val_thm1_lecam.py` — empirical `inf_g M` tracks closed-form `1−TV` for all n, shift magnitudes |
| A2 | Forced-abstention theorem `P(abstain) ≥ 1−2α` | ✅ **added this pass** | **Cor. 1 `cor:forced-abstain`** (new); complemented by anytime e-process (Thm 5) + minimax floor (Prop 1) |
| A3 | Regret decomposition identity | ✅ | Thm 3 `thm:gate`; `val_thm2_regret.py` — identity gap ~2×10⁻¹⁷, minimax floor ratio 1.0004 |
| A4 | Multiclass extension of sign-of-difference | ✅ | Thm 8 (multiclass) + Thm 9 (regression); `val_thm5_multiclass.py` — exact to 10⁻¹⁶, **100% sign over 4000 trials**, both-wrong-on-D in every trial |
| A5 | Sample-complexity theorem `n=O(σ²/Δ² log 1/α)` | ✅ **added this pass** | **Cor. 2 `cor:samplecomp`** (new); ties to repo T4 risk-dominance |
| A6 | Shift hierarchy | ✅ **added this pass** | **Table `tab:shift`** (new): covariate identifiable / label-shift / disagreement-local / concept non-identifiable / mixed→abstain |
| — | Label-free *bracketing* (Conj. 1) | 🟡 **honest open** | Genuinely the hard residual (= unsupervised accuracy estimation); left open by design, not faked |

## B. Empirical

| # | Item | Status | Notes |
|---|------|--------|-------|
| B1 | CIFAR-10-C, 15 corruptions × sev 1–5 × batches × seeds, **Tent/SAR/EATA** head-to-head | 🔧 **prepared, needs Mac** | `cifar_tent_mps_v2.py` + **`run_decisive_tta.sh`** (auto-downloads 2.9 GB CIFAR-10-C, md5-checked). Analysis core verified here. Online CIFAR-10 TTA already in paper (Table `tab:online`) |
| B2 | CIFAR-100-C / ImageNet-C | 🔧 future / optional | runner supports `WITH_C100=1`; ImageNet-C via `--imagenetc-root` |
| B3 | Harmful suite (collapse, label-shift, tiny batch …) | 🟡→🔧 | done on AD + online stream; deep CIFAR-C harmful cells come from the B1 run |
| B4 | Helpful suite | ✅ | mixed-regime + online helpful stream |
| B5 | Unknowable / ambiguous suite | ✅ | clean witness `witness_clean.json` (100% abstain, KS p>0.05); covert-failure boundary |

## C. Metrics — ✅ all reported & verified
accuracy, **regret-to-oracle**, false-adapt, false-freeze, abstain, coverage, adapt-precision,
benefit-interval coverage, decision confusion-by-regime. (`policy_metrics()` computes all;
`02_verify_results.py` audits them.)

## D. Plots
✅ present: witness, regret-summary, α/coverage, evidence ablation, decision-flow, online-TTA.
🔧 produced by the B1 run: knowability phase diagram, regret-vs-coverage, false-adapt-vs-coverage,
Pareto-over-mix (`make_figures()` → `fig_decisive_{regret,pareto,decisions}_cifar10c.png`).

## E. Ablations — ✅ done & verified
`ablations.json`: evidence-drop (disagreement dominates, regret 0.037→0.094), α-sweep, batch-size sweep.

## F. Statistics — ✅ done & verified
8 seeds, mean±std, paired t-tests (KGA vs freeze `p=7.3e-11`; vs adapt `p=5.1e-5`), CIs.
`rigor_multiseed.json`.

## G. Claims — ✅ disciplined
**Claim-lock table added** to `kbound.tex` §Limitations (allowed vs out-of-scope). Honest
scope box on p.1; every number traces to `scripts/`.

## H–J. Acceptance / execution order
Theory block (A1–A6) ✅ · Metrics/Ablations/Stats ✅ · Reproducibility harness ✅
(`00→04`, `99_reproduce`). Remaining: **run B1 on the Mac**, fold its table+figure into the
paper, then the external steps (⬜ public release, ⬜ venue submission).

---

## What changed this pass (all real, reproducible)
1. **Verified** every theory validator from a clean environment (Le Cam, regret, multiclass/regression) — all pass.
2. **Added & compiled** into `kbound.tex`: Cor. 1 (forced abstention, A2), Cor. 2 (sample complexity, A5), shift-hierarchy table (A6), claim-lock table (G); rebuilt **`K-Bound_paper.pdf`** (14 pp, no undefined refs).
3. **Built the turnkey deep-TTA runner**: `run_decisive_tta.sh` + `fetch_cifar_c.py` (auto-download + md5 + extract); smoke-tested the torch-free analysis core (`beats_both=True`, false-adapt 0, Pareto p*=0.2).

## The single command to close B1 (run on your M5)
```bash
cd AutoML_Flagship_V8
bash docs/research/kbound/scripts/run_decisive_tta.sh
```
Then fold `experiments/kbound/results/decisive_tta_table.md` + `figures/fig_decisive_pareto_cifar10c.png`
into `kbound.tex` (replacing the §Limitations "CIFAR-100-C/ImageNet-C … future work" line) and
recompile `pdflatex kbound.tex` ×2.
