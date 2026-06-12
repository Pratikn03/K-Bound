# K-Bound Theory V2 — First Real-Data Validation (Agent B)

CPU-only · repo venv (py3.14.3, numpy2.4.4, scipy) · numbers in `realdata_audit_results.json`;
figs `fig_p1_sign_vs_tau.png`, `fig_p2_ci_forest.png`. Labels scored ground truth only, never
fit; failures reported as-is.

## P1 — 123-task anomaly bank (6 detectors; f0=best-val, f_a=2nd-best; D={f_a != f0})
46–52 of 62 tabular tasks usable (rest skipped: |D|<40). On TEST-region D we computed
pairwise agreements c, the 4-minor spread tau, product-ratio b_hat with majority anchor,
and scored against true b from labels.

- **H is rejected on ~83–85% of tasks** (tau >> CEI-null q95). This is the diagnostic
  WORKING, not a failure: the 6 ADBench detectors share features, so CEI/per-class-symmetry
  (H) is genuinely violated — exactly the rank-2 departure Thm 0.2 / Cor 0.3 predicts tau
  detects. The theory's own premise is that H is *falsifiable, never verifiable* — and here
  it is loudly falsified on real correlated detectors.
- **Theory's positive prediction is SUPPORTED (median thresholds):** relative-sign
  recovery is **0.78 accurate on H-pass tasks (n=9) vs 0.49 (~chance) on H-reject tasks
  (n=43)** — sign(b_a−b0) is recoverable exactly where H holds and uninformative where the
  diagnostic says don't trust it. This is the headline P1 win.
- **Honest counter-evidence:** under *val-optimal* thresholds the ordering inverts (H-pass
  0.29 n=7 vs H-reject 0.49) — those H-pass tasks sit at the tau=q95 boundary with extreme
  prevalence where val-opt thresholds make D degenerate. Median split is trustworthy; the
  inversion is a real fragility caveat (small H-pass n).
- **|b| magnitude recovery is POOR throughout** (median abs-err ≈ 0.44–0.79), even on H-pass
  tasks, because AD has extreme imbalance (piD≈0.00–0.06) → minority class too thin for stable
  product ratios. Sign (the decision bit) transfers to real panels; magnitude (Thm T-I(a)
  norm) does not under this imbalance. A domain mismatch, not a theory bug.

## P2 — stats hardening of the CIFAR-10-C TTA grid (replaces flagged bootstrap)
Real per-condition data: `cifar10c_65cells.csv` (65 corruption×severity cells, per-cell
accuracy for frozen/tent/eata/sar/kga/oracle). Per-condition paired bootstrap (N=10k) +
paired-t + Holm over the 6-comparison family, replacing the flagged
`pareto_bootstrap_curve` (which resampled an 11-point SYNTHETIC harmful-mix curve).

- **KGA vs always-FREEZE: survives Holm for all 3 methods** — Δregret ≈ **−0.214**,
  95%CI ≈ [−0.245, −0.183], p ≈ 2e-20, t ≈ −13.5. Decisive and real.
- **KGA vs always-ADAPT: does NOT survive Holm for any method** (tent Δ=+0.0001 ns; eata
  Δ≈0 ns; sar Δ=+0.0011, p=0.029 > Holm thr, KGA slightly worse). On the *clean* corruption
  grid adapting is almost always helpful, so KGA's gating has little harm to avoid.
- **This corrects the flagged table.** The old synthetic-stream CIs reported KGA beating
  always-adapt at p≈0.0013, d≈−1.4; that advantage was an artifact of injected harmful
  fractions absent from the real grid. Honest headline: **KGA robustly beats always-freeze;
  it only matches always-adapt absent harmful conditions.** The "beats BOTH baselines" claim
  needs the harmful-condition regime (the 432-stress grid) and is not supported on clean C.

## Caveats
Per-condition accuracy arrays for the 432-condition stress grid were NOT serialized in the
JSONs (only aggregate scalars + the synthetic Pareto curve), so P2 could only harden the
65-cell clean grid that does store per-cell data. P1 thresholds and the small H-pass count
make those sub-rates noisy. No detector retraining was done; scores taken as logged.
