# K-Bound / KGA — External Audit & Reproduction Packet

> **STATUS 2026-08-29 — HISTORICAL REVIEW PACKET, PARTIALLY SUPERSEDED.**
> Use `KBOUND_SHORT_RESULT_AUDIT.md`, `KBOUND_SHORT_CLAIM_MANIFEST.md`, `claim_ledger.json`, and
> `runbooks/release_candidate.sh` for the current review. The dated checks below are retained for
> provenance and reviewer context. They predate CCT-20 (`SAFE_UTILITY_ONLY`), the So2Sat negative
> development stop, the current exact-rank multiplicity reconciliation, and the maintained shared
> manuscript body. Where this packet disagrees with those authorities, this packet loses.

**Purpose.** This packet exists so two independent people can sign off on the paper before it is
frozen for submission:

1. **A theory / statistics reviewer** checks the theorem assumptions (Part A).
2. **An independent reproducer** re-runs one key experiment on a clean machine and confirms the
   numbers in the paper (Part B).

Both are the *user's* action items — this document just makes them fast. Nothing here should be
taken as a claim that the audit has already happened. The sign-off form is at the end (Part D).

Paper under review: `docs/research/kbound/kbound_submission.tex` with
`kbound_submission_body.tex` → `kbound_short_final_draft.pdf`; the synchronized long companion is
`kbound_tmlr.pdf`.

---

## Part A — Theory review checklist

The paper's spine is one identifiability theorem plus a decision certificate built on a conformal
radius. A reviewer should pressure-test the following, in order of how load-bearing they are.

### A1. The conformal coverage assumption (most mechanical)
- **Claim (paper §Theory):** the interval `[Δ̂ − ε, Δ̂ + ε]` has marginal coverage
  `Pr[|Δ̂ − Δ| ≤ ε] ≥ 1 − α`, and this is what makes the unconditional false-adapt
  `FA_u = Pr[adapt, Δ ≤ 0] ≤ α`.
- **What to check.**
  - The radius is calibrated **out of fold**. On the per-cell stress grids (CIFAR-10-C, synthetic),
    `decide_kga` (`scripts/cifar_tent_mps_v2.py`, L129–142) fits `Δ̂₋ᵢ` on the other `N−1` cells and
    evaluates cell `i` out-of-fold; `ε` is the `(1−α)` quantile of the leave-one-out residuals. So
    estimator-fitting and residual-calibration data are disjoint for every point.
  - **Honest caveat to scrutinize:** leave-one-out (jackknife) calibration does **not** give the exact
    `1−α` finite-sample guarantee that a *single clean split* gives. The finite-sample jackknife+
    guarantee is `1 − 2α` (Barber, Candès, Ramdas & Tibshirani, *Ann. Statist.* 2021). The paper now
    states this explicitly in §Method. A reviewer should confirm the paper does **not** silently
    claim exact `1−α` for the LOO grids.
  - **Exchangeability.** Coverage requires the calibration cells and the deployment/test cells to be
    exchangeable. For the per-cell grids this is the pre-registered condition grid (Protocol A). For
    the natural-shift protocols (Camelyon17, iWildCam, Office-Home) `Δ̂` and `ε` are fit once on a dev
    split and the held-out test domain is scored once — ordinary split-conformal **across domains**.
    Check whether cross-domain exchangeability is a reasonable idealization for each dataset (it is the
    standard conformal-under-shift caveat; the paper flags it in Limitations).
  - **Algorithm 2 (`alg:calib-eval`) must match the code.** It was repaired to compute `ε` from
    **out-of-fold** (leave-one-out) residuals and to fit the deployment `Δ̂` separately on all
    calibration pairs — no in-sample residuals. Confirm Algorithm 2, `decide_kga`, and the §Method
    text agree, and that §Method states the guarantee **split-dependently**: split-conformal (exact
    `1−α`) on the natural-shift protocols; jackknife radius with calibration-set coverage `≥1−α` by
    construction on the per-cell grids (realized `0.898` at nominal `0.90`); jackknife+ (`1−2α`) noted
    but not implemented.

### A2. The risk-alignment assumption (the genuinely load-bearing one)
- **Claim:** the decision rule's false-adapt is controlled because the evidence `Z` and the realized
  benefit `Δ` are *risk-aligned* — committing `adapt` only when `Δ̂ − ε > 0` keeps `FA_u ≤ α`.
- **What to check.** This is **not** purely a conformal statement; it assumes the sign of the
  estimated margin transfers to the sign of the true benefit on the deployment distribution. A
  reviewer should look for circularity: does any step assume what it is trying to prove (that `Δ̂`'s
  sign is correct)? The paper separates this into (i) identifiability and (ii) coverage; confirm the
  separation holds and that the guarantee is stated as **conditional on the assumption**, not as an
  unconditional empirical fact about the datasets. (This was the focus of the internal senior-review
  audit, `THEORY_AUDIT_senior_review.md`; an external reader should re-derive it independently.)

### A3. The frontier identifiability theorem
- **Claim:** over the declared class `C_β = {P_T : |γ| ≤ β}`, the sign of the adaptation benefit is
  label-free identifiable **iff** `|M| > β`, where `M = E[s | D] − ½` is the observable margin on the
  disagreement region `D`, `γ` is the unobserved calibration drift, `β` is the supplied drift budget,
  and `ε` is the empirical (finite-sample) radius — **not** a direct estimate of the worst-case `β`.
- **What to check.**
  - The four quantities `M`, `γ`, `β`, `ε` are kept distinct (the paper was revised specifically so
    that `ε` is described as an operational uncertainty radius, not a surrogate for the worst-case
    drift budget). Confirm no sentence collapses `ε` and `β`.
  - The "iff" direction that matters operationally is: if `|M| ≤ β` the sign is **not** identifiable
    label-free → the rule must abstain. Confirm the abstention branch is what the algorithm actually
    does when `|Δ̂| ≤ ε`.
  - The standalone 1-D and general-chain capacity results are in the appendix
    (`paper/sections/knowability_capacity_general.tex`); these are self-contained and can be checked
    independently of the experiments.

### A4. The two false-adapt metrics
- `FA_u = Pr[adapt ∧ Δ ≤ 0]` is the **unconditional** quantity the theorem bounds by `α`.
- `FA_c = Pr[Δ ≤ 0 | adapt]` is the **conditional** rate among committed adapts; it is **not**
  `α`-bounded and is reported empirically.
- **What to check.** They coincide whenever no harmful adapt is committed (so `FA_u = FA_c = 0` on
  every headline dataset, where KGA commits none) and diverge only for rules that commit some — e.g.
  the leaky gates in Table III (`tab:gates`), which reports both plus adapt-rate and coverage. Confirm
  the paper never reports `FA_c` in a place that reads as if it were the `α`-bounded quantity.

### A5. Scope / honesty checks (quick)
- Camelyon17 is reported as a **no-harm / helpful-dominated** result (KGA ties always-adapt on the
  genuine OOD test, `n=18`), **not** a beats-both win. Confirm the withdrawn beats-both claim does not
  reappear in abstract, tables, or mixed-stream.
- The mixed stream is described as a **cross-protocol aggregate** (`n=143`, OOD cells only), not a
  single heterogeneous deployment.
- iWildCam is a **point-estimate** result (no CI claim). **CORRECTED 2026-07-26 (F4-17):** an earlier
  version of this line said "Office-Home and the CIFAR stress grid carry the CI-backed beats-both."
  That was wrong about Office-Home and it was the version handed to external reviewers. Office-Home
  is promoted as **OOF no-harm only**; its LOO beats-both is explicitly **not** promoted, and its own
  artifact `research_lock/KBOUND_WIN_BOOTSTRAP_CIS_oof.json` records `"beats_both_robust": false`
  and `kga_vs_freeze.ci_excludes_zero: false`. The CI-backed beats-both tracks are the **CIFAR-10-C
  stress grid (Tent and EATA)** and the constructed three-source mixture. ImageNet-C SAR's
  freeze-gap CI does not survive the 2026-07-26 radius fix — see `SUBMISSION_LEDGER.md §9`.
- Camelyon17's promoted row is **sealed but not recomputable from this release**; do not treat it as
  a reproduced number (`SUBMISSION_LEDGER.md §8`).

### A6. The unconditional weakest-class theorem (`thm:uncond-weakest`, "Theorem 8") + Appendix `app:weakest`
- **Claim.** With General Position removed, one declared bit certifies `sign(Δ)` on a falsifiable
  class **iff** the dominance margin `inf_W(T−G) ≥ 0` (or `sup_W(T+G) ≤ 0`); the weakest one-bit
  classes are the **dominance polytopes** `W* = {T(r) ≥ G(r)}`, an explicit **finite family** indexed
  by orientation pattern — **not** a unique class. General Position is the face collapsing the family
  to `C_mono`.
- **What to check (priority order).**
  1. **Lemma `lem:canonform` (canonical form) is the one load-bearing input** — it claims *every*
     falsifiable class is `C(P,s,W)`. Check its `(⇒)` direction against the paper's exact definition of
     a falsifiable/evidence-definable class; it should rest only on the swap involution
     (`thm:conj1-dichotomy`(iii)), established earlier. This is where the generality lives.
  2. **Criterion (i)** is "tautological once the interval form is granted"; the content is Lemma 1 +
     the linear-range argument. Note the `inf/sup` must be **exact** (the result is false with
     *sampled* `inf/sup` — a numerical, not mathematical, caveat).
  3. **Scope is a finite family, not uniqueness.** Confirm the paper says "no unique weakest class."
- **Independent verification already done (reproducible).** The criterion matched exact brute-force
  vertex truth on `2.8×10⁵` box fibres and `3.2×10³` non-box polytopes (exact-LP) with `0` mismatches;
  a from-scratch reimplementation reproduced this and the `C_dom(ρ)` lattice flipping at exactly
  `ρ=1`. Run `theory_v2/val_unconditional_weakest.py` (fixed seeds; fails loudly). Full proof:
  `theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md`.

---

## Part B — One-command reproduction of the central decision claim

The central empirical claim is **Table III** (`tab:gates`): on the CIFAR-10-C stress grid, only the
conformal certificate keeps `FA_u = 0` while staying near-oracle, and simple label-free gates leak.
This is the cheapest high-value thing to reproduce: it is pure `numpy` + `scikit-learn` over a
committed per-cell evidence dump (no GPU, seconds to run).

### B1. Environment (clean machine)
```bash
python3 -m venv .venv_audit
source .venv_audit/bin/activate          # Windows: .venv_audit\Scripts\activate
pip install numpy scikit-learn
```

### B2. Run
```bash
cd docs/research/kbound
python scripts/gate_baseline_comparison.py --selftest      # sanity: synthetic check must pass
python scripts/gate_baseline_comparison.py                 # reads the committed per-cell dump
```
The script reads the locked per-cell evidence dump (`cifar10c_percell_*.json`) and writes
`gate_comparison.json` / `gate_comparison.md`.

### B3. Expected output (pass criteria)
`n = 432` cells, `n_harmful = 149`, `α = 0.1`. The six decision rules must reproduce (±0.001):

| Decision rule        | regret | FA_u  | FA_c  | adapt | cov  | FA_u(harm) |
|----------------------|:------:|:-----:|:-----:|:-----:|:----:|:----------:|
| confidence gate      | 0.0084 | 0.257 | 0.301 | 0.85  | 1.00 | 0.745 |
| entropy gate         | 0.0086 | 0.255 | 0.304 | 0.84  | 1.00 | 0.738 |
| drift/KL gate        | 0.1232 | 0.000 | 0.000 | 0.00  | 1.00 | 0.000 (degenerate: never adapts) |
| ATC-style gate       | 0.0045 | 0.116 | 0.172 | 0.67  | 1.00 | 0.336 |
| KGA (no radius)      | 0.0004 | 0.049 | 0.071 | 0.68  | 1.00 | 0.141 |
| **KGA (certificate)**| 0.0017 | **0.000** | **0.000** | 0.51 | 0.68 | **0.000** |

**A reproduction counts as successful iff:**
- `KGA (certificate)` has `FA_u = 0.000` on both the full grid and the harmful subset, and
  `regret ≈ 0.0017`;
- the confidence/entropy gates show `FA_u(harm) ≈ 0.74` (they false-adapt on ~3/4 of harmful cells);
- `KGA (no radius)` has lower regret (≈0.0004) but **nonzero** `FA_u` (≈0.049) — i.e. removing the
  radius trades the guarantee for slightly lower regret.

### B4. Data provenance
The per-cell dump is the locked output of the full CIFAR-10-C stress grid (ResNet-18, Tent candidate,
432 conditions, 5 seeds, pre-registered Protocol A). Each cell records the 11-dim evidence vector `Z`,
the frozen accuracy `a0`, the adapted accuracy `aa`, and the label regime. The gate script makes **no**
model calls — it only re-derives decisions from the logged evidence, so the reproduction isolates the
decision rule from the (expensive) TTA forward passes.

### B5. Second independent check — per-condition bootstrap + empirical coverage
Also pure `numpy` over saved records, no GPU:
```bash
python scripts/percondition_bootstrap.py --root ../../../experiments/kbound/results
```
This recomputes the CIFAR-10-C beats-both with the **conventional paired per-condition bootstrap**
(432 seed-averaged conditions, resampled with replacement), complementing the design-based
mixing-ratio CI. Expected:
- `regret KGA` ≈ 0.0016 / 0.0013 / 0.0015 (tent / eata / sar);
- **Tent and EATA**: both regret-gap CIs exclude zero (beats-both at the realized grid composition);
- **SAR**: gap-vs-always-adapt small and positive (ties / slightly behind always-adapt), still beats
  always-freeze;
- `FA_u = 0.000` for all three;
- **empirical conformal coverage ≈ 0.898**, matching nominal `1 − α = 0.90` — this is the number that
  backs the out-of-fold calibration claim in Part A1.

---

## Part C — Mixed head-to-head vs POEM and AETTA (CPU, ~10 s)

Pre-registered: `docs/research/kbound/MIXED_BENCHMARK_PROTOCOL.md`.

```bash
cd "$KBOUND_REPO_ROOT"        # set KBOUND_REPO_ROOT to your checkout
PY=.venv/bin/python bash experiments/kbound/poem_aetta/run_all_headtohead.sh
```

**Pass criteria (PRIMARY = Tent, 5 seeds):**
- `HEADTOHEAD_RESULTS_cifar10c_tent_primary.json` → `headtohead.VERDICT == "WIN"`
- KGA regret ≈ 0.0016; POEM ≈ 0.0088; AETTA ≈ 0.0073
- `kga_false_adapt_rate == 0.0`
- Both `KGA vs poem` and `KGA vs aetta` have `kga_beats: true` and Holm `p_holm < 0.05`

Secondary arms (`eata_secondary`, `tent_eata_pooled`) should also report WIN.

Also verify cached artifacts:
```bash
bash docs/research/kbound/scripts/reproduce_submission.sh
```

---

## Part D — Deeper reproduction (optional, GPU/MPS)

To regenerate the per-cell dump itself (and the headline regret numbers in Table II), re-run the full
grid. This needs CIFAR-10-C and a few hours on MPS/CUDA.
```bash
cd docs/research/kbound
python scripts/cifar_tent_mps_v2.py --benchmarks cifar10c --device mps \
  --data-root <path-to>/experiments/kbound/cifar --methods tent,eata,sar
```
Expected headline (from `experiments/kbound/results/decisive_tta_cis.json`):
- Tent: regret KGA 0.0016 vs always-adapt 0.0086 vs always-freeze 0.1232 (beats both, CI excludes 0).
- EATA: regret KGA 0.0015 vs 0.0037 vs 0.1311 (beats both).
- SAR: regret KGA 0.0018 vs 0.0018 vs 0.1372 (ties always-adapt — no-harm).

Note: SAR's harmful base rate is sensitive to the model checkpoint and operating point; a fresh train
can shift it. The locked paper numbers come from the committed checkpoint/seeds. Reproducers should use
the committed checkpoint if matching the paper exactly; an independent train is expected to reproduce
the *qualitative* result (certificate ties/▸beats, `FA_u = 0`) but not necessarily the exact harmful
fraction.

---

## Part F — 85+ strong-accept path (physical R2 + full panel)

Scorecard (run anytime):
```bash
bash docs/research/kbound/scripts/run_85plus_readiness.sh
```

### F1. Multiseed smoke preflight (~2–3 h MPS)
```bash
KB_SMOKE_SEEDS="0 1" KB_DEVICE=mps \
  bash docs/research/kbound/scripts/run_smoke_showcase.sh
```
Exit `0` ⇒ all 9 datasets collated (RxRx1 may skip if data missing).

### F2. Physical camera R2 (human recording required)
```bash
# Phase 1 — source gate only (stop if balanced-acc < 0.80)
bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh

# Phase 2 — after gate passes: calibration + held-out + export
bash docs/research/kbound/edge/scripts/run_edge_heldout_capture.sh
```
See `edge/STAGING_GUIDE.md` and `edge/EDGE_COMPLETION_CHECKLIST.md`.

### F3. Full 9-dataset 5-seed refresh
```bash
bash docs/research/kbound/scripts/prepare_rxrx1_data.sh   # once
KB_SEEDS="0 1 2 3 4" KB_DEVICE=mps KB_IC_MAXIMG=2000 \
  caffeinate -is bash docs/research/kbound/scripts/run_final_showcase.sh \
    --device mps --seeds "0 1 2 3 4"
```

### F4. Macro / table drift check
```bash
python docs/research/kbound/scripts/refresh_results_source_locked.py
python docs/research/kbound/scripts/make_tables.py
cd docs/research/kbound && latexmk -pdf kbound_short.tex
```
Tables `tab:decisive` and `tab:headtohead-poem-aetta` read from `paper/generated/kbound_numbers.tex`.

---

## Part E — Sign-off form

### Theory reviewer
- Name / affiliation: ____________________
- Date: ____________________
- A1 conformal coverage (incl. jackknife+ caveat): ☐ sound ☐ needs change — notes: __________
- A2 risk-alignment (no circularity): ☐ sound ☐ needs change — notes: __________
- A3 frontier identifiability (`M`,`γ`,`β`,`ε` distinct; iff scoped): ☐ sound ☐ needs change — notes: __________
- A4 FA_u vs FA_c usage: ☐ sound ☐ needs change — notes: __________
- A5 scope/honesty: ☐ clean ☐ issue — notes: __________
- Overall: ☐ assumptions are correctly stated and the theorem follows ☐ revise

### Independent reproducer
- Name / affiliation: ____________________
- Date / machine / OS: ____________________
- B2 ran clean: ☐ yes ☐ no
- B3 pass criteria met: ☐ yes ☐ no — discrepancies: __________
- Overall: ☐ Table III reproduces ☐ does not reproduce

**Only when both boxes are checked and the repo reproduces the exact PDF numbers from a clean machine
is the paper a frozen, submit-ready final.** Until then the honest label is *near-final, high-quality
submission draft*.
