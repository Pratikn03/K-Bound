# Pre-registered protocol: a CI-robust *pure-label-free* real-shift beats-both

Goal (the "85+" lever): land **one clean, CI-robust, pure-label-free beats-both on a real
distribution shift** — honestly, with no target/test tuning. This doc is frozen *before* the run.

Software in this folder (all torch-free, run with `~/.venv_wilds/bin/python`):
- `verify_realshift_win.py` — the **locked verdict**. Fit on calibration only, score held-out test
  once, emit regret + paired-bootstrap 95% CIs → `beats_both_CI_robust`. Self-test passes (rewards a
  genuine win, rejects helpful-dominated / all-harm / anti-transfer).
- `power_goldilocks.py` — the **feasibility phase diagram** (what the data must achieve).
- `feasibility_screen.py` — the **GO/NO-GO screen** to run on a labeled dev/preview split.

## 1. The win, locked (do not change after seeing results)
Held-out test scored **once**. Pre-registered bar:
- false-adapt rate ≤ α = 0.10, **and**
- `regret_KGA < regret_always-adapt` **and** `regret_KGA < regret_always-freeze` (point), **and**
- **CI-robust**: both regret-gap 95% condition-bootstrap CIs exclude 0.

The adapter, the evidence features, the conformal ε, and α are all **dev-locked on source/in-dist**
and frozen. τ is source-calibrated, never tuned on target. If null → report null.

## 2. What the data must achieve (from `power_goldilocks.py`)
P(CI-robust win) as a function of source→OOD detector **transfer** and **n** (conditions/seeds),
at two-sided mixedness p_harm≈0.35, good in-source detector:

| transfer \ n | 60 | 120 | 240 | 480 |
|---|---|---|---|---|
| 0.3 | 0% | 0% | 0% | 0% |
| 0.5 | 0% | 0% | 9% | 29% |
| 0.7 | 0% | 1% | 58% | 99% |
| 0.9 | 0% | 16% | 99% | 100% |
| 1.0 | 0% | 39% | 100% | 100% |

**Three requirements, all needed in ONE domain:**
1. **Two-sided mixedness** for the *single deployed* adapter: p_harm ≈ 0.25–0.60 (so neither
   always-adapt nor always-freeze is already the oracle).
2. **OOD transfer AUC ≥ 0.70** — the binding constraint. More data cannot rescue a non-transferring
   signal (transfer 0.3 → 0% even at n=480).
3. **n ≥ 240** held-out conditions/seeds (for the bootstrap CIs to exclude 0).

## 3. Why every prior attempt failed (your committed numbers, via `feasibility_screen.py`)
- **iWildCam**: mixedness fine (p_harm 0.589) but OOD transfer AUC **0.43** (AETTA 0.375) — below
  chance. NO-GO on transfer.
- **Office-Home**: "detectability ⊥ mixedness" — the mixed domain (Product, p_harm 0.389) transfers
  at **0.53** ≈ chance; the domains that transfer (Art 0.71, Clipart 0.85) are helpful-dominated
  (p_harm 0.09, 0.03). The two requirements never co-occur in one domain.
- **Camelyon17 / RxRx1 / ImageNet-R**: one-sided (helpful-dominated / all-harm) or undetectable.

The lever is **transfer**, and prior detectors anti-transferred because they were *source-calibrated
accuracy/entropy estimators* (entropy, AETTA Δacc, source-fit τ) whose harm↔signal mapping is
dataset-specific and dies on OOD.

## 4. The design that can clear transfer ≥ 0.70 (theory-grounded)
The signals that DID transfer (the CIFAR-10-C / ImageNet-C wins) detect **collapse**, not calibration
drift: when TTA collapses (single-class / tiny-batch / aggressive), predictions *degenerate* the same
way in any domain. So use **scale-invariant collapse features** instead of calibrated estimators:
- entropy of the **marginal predicted-class histogram** over the batch (collapse → low),
- adapted-vs-frozen marginal-prediction **KL**,
- self-normalized **BN-affine / update-norm** magnitude,
- **prediction-flip rate** vs the frozen model,
- (all standardized on source; ratios/entropies, not absolute thresholds).

These mean "the adaptation is degenerating" regardless of domain → they should transfer. The catch:
they only catch **collapse-driven** harm, so the target regime must make harm collapse-driven —
i.e. an **online / continual** stream with **two-sided composition** (benign windows where adapt
helps, interleaved with trap windows — single-class / tiny-batch / aggressive — that collapse). This
is exactly the mechanism of your existing CIFAR-10-C online-stream result, ported to a real shift.

## 5. Recommended candidate (ranked)
1. **iWildCam, ONLINE/continual stream** (top pick — you already have the data + f0, and mixedness is
   already proven at 0.589). Re-run NOT as the episodic grid that failed, but as a per-camera/temporal
   online stream with an aggressive collapse-prone update and scale-invariant collapse features. The
   mixedness is there; the only thing that failed was the source-calibrated detector.
2. **Wild-Time (FMoW-time / Yearbook) temporal stream** — natural non-stationary composition; periods
   that help vs periods that collapse.
3. **CLEAR / a real industrial or medical online stream** with bursty composition.

## 6. Frozen run procedure
1. Pick the domain; split source/in-dist (calibration) vs held-out OOD test by seed/time.
2. Dev-lock on calibration only: deployed adapter (source-best), the scale-invariant collapse
   evidence vector Z, the benefit estimator B̂(Z), conformal ε at α=0.10, τ\*. **Freeze.**
3. **GO/NO-GO**: on a labeled dev/preview slice run `feasibility_screen.py` → require p_harm∈[0.25,0.60]
   and OOD_transfer_AUC ≥ 0.70. If NO-GO, stop (early-stop wisdom) and report the null — do not retune.
4. If GO: run the full held-out test (≥240 conditions/seeds), score **once**, feed the logged
   per-condition `(Z, a0, aa)` to `verify_realshift_win.py`. Its `beats_both_CI_robust` is the verdict.
5. Whatever it says, that is the result. A null is a legitimate, publishable outcome.

## 7. Integrity guardrails (non-negotiable — this is the paper's whole credibility)
- No threshold/τ/ε tuned on target or test. Operating point frozen on calibration.
- Held-out test scored once. The CI/Holm test *is* the claim, not a point estimate.
- Report null if null. A manufactured win (test tuning) is exactly what got Camelyon withdrawn.
