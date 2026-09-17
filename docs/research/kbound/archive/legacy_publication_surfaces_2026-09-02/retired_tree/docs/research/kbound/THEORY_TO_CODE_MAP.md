# K-Bound: Theory → Proof → Code → Results

**Purpose:** A newcomer (reviewer, collaborator, future you) can follow this document from
theorem labels in the PDF to the proof location, numeric validator, implementation module,
and locked experiment JSON — without guessing from 10 scattered notebooks.

**Canonical status:** [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md)  
**Live tour:** `bash docs/research/kbound/scripts/kbound_tour.sh` (or `python3 docs/research/kbound/scripts/kbound_tour.py`)

---

## 1. Decision pipeline (how theory becomes a button)

```mermaid
flowchart LR
  X[Test batch X] --> Z[Evidence Z = phi(X)]
  Z --> Bhat[Benefit estimate B_hat]
  Bhat --> Eps[OOF conformal radius epsilon]
  Eps --> Rule{ADAPT / FREEZE / ABSTAIN}
  Rule --> Policy[Regret + false-adapt metrics]
  Policy --> JSON[Locked results JSON]
  JSON --> TeX[Paper tables via make_tables.py]
```

| Step | Theory | Code | Notes |
|------|--------|------|-------|
| Evidence `Z` | `thm:frontier` margin `M(O)` | `analyze_F.py`, dataset runners | Label-free features only |
| Radius `ε` | Conformal / LOO split | `decide_kga()`, `score_kbound_holdout.py` | **Must be OOF** (audit 2026-06-26) |
| Decision | Certificate + impossibility | `kbound_pkg/kbound/certificate.py` | FA_u ≤ α under stated assumptions |
| Evaluation | Regret-to-oracle | `multiseed_paired_ci.py`, head-to-head harness | Pre-registered WIN/TIE/LOSE |

---

## 1a. The radius rule — stated once, verbatim, for quotation

**Added 2026-07-26 closing defect D9.**  This is the single normative statement of the
K-Bound certificate radius.  The paper, the library and the drivers must all quote *this*
paragraph; if any of them says something else, that one is wrong.

> **The rule.**  Let `r_1, …, r_n` be the absolute calibration residuals
> `r_i = |Δ̂_i − Δ_i|`, and let `r_(1) ≤ … ≤ r_(n)` be their order statistics.  For a
> miscoverage level `α ∈ (0, 1)`, the radius is the **exact split-conformal rank
> quantile**
>
> ```
> k   = ceil((n + 1) * (1 - α))
> ε   = r_(k)                                     if k ≤ n
> ε   = +∞    (⇒ the decision is ABSTAIN)         if k > n
> ```
>
> The pool is **leave-one-out-of-pool**: the radius used to score cell `i` is computed
> from the other `n − 1` residuals only, so `ε_i` is never a function of the label that
> the `FA_u ≤ α` guarantee attaches to.  Under leave-one-out, a track of `N` cells is
> calibrated from pools of size `N − 1`.
>
> The decision is the strict trichotomy: ADAPT iff `Δ̂_i − ε_i > 0`; FREEZE iff
> `Δ̂_i + ε_i < 0`; ABSTAIN otherwise.  A lower bound of exactly zero ABSTAINs.

**Three things the rule is not.**

1. **Not `np.quantile(r, 1 − α)`.**  NumPy's default quantile linearly interpolates
   between order statistics.  The interpolated value is not an observed residual, so the
   finite-sample rank argument does not apply to it.
2. **Not clamped.**  `k = min(n, ceil((n + 1)(1 − α)))` — the "return the maximum residual
   when `k > n`" convention that shipped before fix-queue item 25 — attains only
   `n / (n + 1) < 1 − α`.  It is removed from `kga/certificate.py`, from
   `kga/policy.py::decide_kga` and, as of D9, from
   `docs/research/kbound/scripts/kbound_decide.py`, which had gone on defaulting to it.
   The superseded value survives only under the explicit name
   `kga.certificate.legacy_clamped_radius`, which no decision path calls.
3. **Not silently feasible at every `n`.**  A finite radius exists iff
   `n ≥ ceil(1/α) − 1` (`kga.certificate.min_calibration_size`): **`n ≥ 9` at `α = 0.10`**,
   `n ≥ 19` at `α = 0.05`, `n ≥ 49` at `α = 0.02` (the Bonferroni level of a five-candidate
   panel at `α = 0.10`).  Below that the library warns and returns `+∞`, or raises
   `InsufficientCalibrationError` on request.  Because the pool is leave-one-out, a track
   needs `N ≥ 10` **cells** at `α = 0.10`.

**Where it is implemented.**  Once, in the library; everything else delegates.

| Layer | Symbol | File |
|---|---|---|
| radius | `split_conformal_rank_radius` | `kga/certificate.py` |
| leave-one-out pool | `conformal_radii_loo` | `kga/certificate.py` |
| feasibility threshold | `min_calibration_size` | `kga/certificate.py` |
| superseded value (not a decision path) | `legacy_clamped_radius` | `kga/certificate.py` |
| trichotomy | `decide`, `decide_batch` | `kga/policy.py` |
| end-to-end rule | `decide_kga` | `kga/policy.py` |
| driver-side single entry point | `conformal_radius`, `decide_kga` | `docs/research/kbound/scripts/kbound_decide.py` |

**Where it is enforced.**  `tests/test_one_radius_rule.py` fails if a clamp or an
interpolated quantile is reintroduced into any radius function, if `decide_kga` grows an
infeasibility knob, or if the library and the driver shim disagree at any pool size —
including the infeasible sizes, which is exactly where they had drifted apart.
`tests/test_kga_canonical_rule.py` pins the behavioural side.

**What removing the clamp costs, stated rather than hidden.**  The clamp only ever fired
for pools of `n ≤ 8` at `α = 0.10`.  Every promoted panel track has a larger pool
(`NUMBERS_PACK.md` §5.2: CIFAR-10-C 432/seed, ImageNet-C 27/seed, D33 130, iWildCam 72,
RxRx1 60, CIFAR-10.1 48, Office-Home 35, Camelyon17 pooled 18), so **no promoted headline
number changes**.  Two rows do change and must be relabelled rather than re-emitted:

* **Camelyon17 Table VIII**, which is `n = 9` cells per seed.  Under leave-one-out the
  pools are size 8, so the exact rank `k = 9` exceeds `n` and the panel is **not
  certifiable at `α = 0.10`** — every cell ABSTAINs.  The archived per-seed row was
  produced under the clamp.
* **iWildCam's source-CV certificate** in
  `experiments/kbound/wilds/analyze_iwildcam_kbound.py`, whose source split has `n < 9`.
  Same conclusion: `+∞` ⇒ ABSTAIN.  The archived row (N = 72, 1 ADAPT, 60 FREEZE,
  11 ABSTAIN) is not reproducible under the declared rule.

Either enlarge those calibration splits past `min_calibration_size(α)`, or report both
tracks as uncertifiable at `α = 0.10`.  Do not re-enable the clamp.

---

## 2. Theory spine (closed for the paper)

### 2.1 Identifiability frontier

| | |
|--|--|
| **Label** | `thm:frontier`, `thm:headline` |
| **Claim** | Benefit sign identifiable iff observable margin exceeds drift budget β |
| **Proof** | `paper/sections/main_theory_5.tex` |
| **Validators** | `experiments/kbound/theory_validation/val_frontier.py`, `val_benefit_frontier.py` |
| **Artifacts** | `experiments/kbound/theory_validation/results_frontier.json` |
| **Ledger** | KB-CLAIM-001 |
| **Empirical hook** | Stress grid: when margin large → KGA commits; when matched → abstains |

### 2.2 Impossibility / one-bit dichotomy

| | |
|--|--|
| **Label** | `thm:imp`, `thm:conj1-dichotomy` |
| **Claim** | Matched evidence + opposite benefit → minimax error ½; one bit is minimal supplement |
| **Proof** | `paper/sections/main_theory_5.tex` |
| **Validators** | `val_thm1_lecam.py`, `val_knowability_dichotomy.py`, `scripts/theory_extensions_validation.py` |
| **Artifacts** | `results_thm1_lecam.json`, `results/witness/witness_clean.json` |
| **Ledger** | KB-CLAIM-002, KB-CLAIM-025 (`conj:gen` **resolved negatively**) |
| **Notebook** | `notebooks/01_Problem_and_Theory.ipynb`, `04_Regression_and_Witness.ipynb` |

### 2.3 Certificate (false-adapt control)

| | |
|--|--|
| **Label** | Guarantee box; `thm:anytime` (extension) |
| **Claim** | FA_u ≤ α under split conformal + assumptions; anytime extension under optional stopping |
| **Proof** | Main text + App. theory extensions in `kbound.tex` |
| **Validators** | `val_thm3_evalue.py`, `theory_v2/val_sequential_anytime.py`, `theory_v2/val_multicandidate.py` |
| **Implementation** | `kbound_pkg/kbound/certificate.py` |
| **Ledger** | KB-CLAIM-003 (**supported**); KB-CLAIM-004 FA_c (**withdrawn**) |
| **Notebook** | `notebooks/07_Certificate_and_Calibration.ipynb` |
| **Empirical hook** | Gate table: only KGA keeps FA_u=0 across 432 cells |

### 2.4 Unconditional weakest one-bit class

| | |
|--|--|
| **Label** | `thm:uncond-weakest` |
| **Claim** | Weakest classes = dominance polytopes; GP is a collapsing face |
| **Proof** | `theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md` |
| **Validator** | `theory_v2/val_unconditional_weakest.py` (machine-checked, 0 mismatches) |
| **Artifact** | `theory_v2/unconditional_weakest_results.json` |
| **Paper** | `kbound.tex` appendix |

---

## 3. Empirical claims (what results prove which story)

### Tier A — Headline wins (beats-both)

| Claim ID | Result | Artifact | Rerun |
|----------|--------|----------|-------|
| KB-CLAIM-010 | CIFAR stress grid Tent/EATA | `stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` | `kbtrain.sh cifar10c` |
| KB-CLAIM-026 | vs POEM/AETTA mixed WIN | `mixed_headtohead_v1/HEADTOHEAD_RESULTS_*.json` | `run_all_headtohead.sh` |
| KB-CLAIM-011 | ImageNet-C SAR harmful | `results_source.json` | locked SAR run |
| KB-CLAIM-024 | Mixed OOF aggregate | `mixed_protocol_oof_v2/` | `mixed_stream_kbound.py` |
| KB-CLAIM-027 | D33 controlled multimodal | `controlled_multimodal_d33/results.json` | D33 runner |

### Tier B — Natural shifts (no-harm only)

| Claim ID | Dataset | Artifact | Rerun |
|----------|---------|----------|-------|
| KB-CLAIM-020 | Office-Home M v2 | `results_source.json` | `kbtrain.sh protocol-m-v2` |
| KB-CLAIM-021 | iWildCam H v2 | `results_source.json` | `kbtrain.sh protocol-h-v2` |
| (panel) | Camelyon, RxRx1, PACS | `results_source.json` | `kbtrain.sh final-all` |

**Withdrawn:** KB-CLAIM-022 (Camelyon pooled beats-both), KB-CLAIM-023 (in-sample mixed aggregate).

### Tier C — Pending / non-headline

| Claim ID | Status | Notes |
|----------|--------|-------|
| KB-CLAIM-030 | pending | Physical camera R2 — protocol only |
| KB-CLAIM-040 | supported | Assumption audit (falsification-oriented) |

---

## 4. How proof connects to results (concrete example)

**Story:** Certificate prevents harmful adaptation on mixed harmful+helpful CIFAR conditions.

1. **Theory** says: if you cannot identify sign Δ, abstain; if you commit ADAPT with certificate, FA_u ≤ α.
2. **Code** `decide_kga()` computes LOO `B_hat` and conformal ε per condition → ADAPT/FREEZE/ABSTAIN.
3. **Stress grid** logs per-condition decisions for Tent adapter (432 conditions × 5 seeds).
4. **Analysis** `LOCKED_ANALYSIS_RESULTS.json`: KGA regret 0.0016 vs adapt 0.0079, FA_u=0.
5. **Head-to-head** same records + POEM/AETTA ports → `VERDICT: WIN`.
6. **Paper** `tab:headtohead-poem-aetta` via `make_tables.py` / inline TeX.

A reviewer can trace: `kbound_short.tex` table → JSON → `run_mixed_headtohead.py` → `decide_kga` → theorem guarantee box.

---

## 5. Validator inventory

Run all lightweight checks:

```bash
bash docs/research/kbound/scripts/reproduce_submission.sh
```

| Script | Checks |
|--------|--------|
| `val_thm1_lecam.py` | Le Cam lower bound + witness worlds |
| `val_thm2_regret.py` | Regret decomposition identity |
| `val_thm3_evalue.py` | E-value / false-adapt certificate |
| `val_thm5_multiclass.py` | Multiclass sign Δ reduction |
| `val_frontier.py` | Frontier margin thresholding |
| `val_knowability_dichotomy.py` | Dichotomy numerics |
| `theory_v2/val_unconditional_weakest.py` | Polytope weakest-class enumeration |
| `theory_v2/val_sequential_anytime.py` | Anytime FA_u |
| `theory_v2/val_multicandidate.py` | Family-wise FA_u (binary K) |
| `theory_v2/val_multiclass_multicandidate.py` | Multiclass multicandidate FWER |
| `theory_v2/val_anytime_multicandidate.py` | Anytime multicandidate FWER |
| `theory_v2/val_tight_constants.py` | Exact 3-world κ(α)n_opt |
| `theory_v2/val_multiclass_capacity.py` | Multiclass capacity + impossibility |
| `theory_v2/val_margin_computability.py` | Margin computability dichotomy |
| `theory_v2/val_regression_bracketing_closure.py` | Regression bracketing |
| `scripts/run_theory_v2_validators.sh` | All Wave 4 validators + routing selftest |
| `scripts/multicandidate_decide_kga.py` | LOO GBR + Bonferroni panel (training hook) |
| `scripts/theory_extensions_validation.py` | Le Cam + forced abstention + multiclass Δ |

CI: `.github/workflows/kbound-ci.yml` runs `val_thm*.py` and `theory_v2` Wave 4 validators + `lean-formal`.

---

## 5b. Wave 4 theory → code map

| Label | Implementation | Training / repro hook |
|-------|----------------|----------------------|
| `thm:multicand` | `kga/routing.py` → `route_panel` | `multicandidate_decide_kga.py` |
| `thm:multiclass-multicand` | `kga/routing.py` → `multiclass_benefit`, `route_panel` | same |
| `thm:anytime-multicand` | `kga/routing.py` → `AnytimeMulticandidatePanel` | `kga.certificate.evalue_anytime` + panel |
| `thm:t1c-exact` | theory only (sample complexity) | `val_tight_constants.py` |
| `thm:mc-cap-impossibility` | theory only (negative) | `val_multiclass_capacity.py` Block D |
| `thm:margin-compute-dichotomy` | theory only | `val_margin_computability.py` |
| `thm:reg-bracket-dichotomy` | `scripts/regression_conjecture_validation.py` | wrapper validator |
| strict-core Lean | `formal/KBound/*.lean` | `formal_audit.py --strict-core` (`--strict-100` is legacy alias) |

Canonical package: **`kga/`**. Frozen paper mirror: **`kbound_pkg/kbound/`** (including `routing.py`).

---

## 6. What the 10 notebooks do / do not cover

| Covered well | Partially stale (pre–POEM/AETTA) |
|--------------|-----------------------------------|
| Theorem validator JSON loading (01) | `00_KBound_Reproduction.ipynb` (123-task ELARA archive) |
| Trichotomy demos (02) | Artifact lists in 09 (missing new HEADTOHEAD paths) |
| Harmful/mixed regimes (03) | Some figures predate OOF mixed v2 |
| Certificate intuition (07) | |
| **Master guide (00 new)** | **Fills the gap — use this first** |

---

## 7. Closure status (Wave 4, 2026-07-01)

Section B of `THEORY_100_PERCENT_CLOSURE_PLAN.md` is **closed** (dichotomies + impossibility).
`formal_audit.py --strict-core` passes locally.

The Lean package is a strict-core mechanization: algebraic theorem spine plus finite-sample bridge
lemmas. It is not a full foundational Mathlib development of measure-theoretic exchangeability,
optional stopping, product KL/TV, martingale rates, or the complete swap-involution construction.

**Still outside theory scope:** physical camera R2 (KB-CLAIM-030), external reviewer sign-off.

**Documentation:** see [`DOCS_INDEX.md`](DOCS_INDEX.md) — stale June 2026 status MDs removed 2026-07-01.

**Engineering note:** Wave 4 **characterization** theorems (impossibility, dichotomies, tight
constants) intentionally do not change the default `decide_kga` binary spine; they bound what
claims are allowed. **Routing** theorems (`multicand`, `multiclass-multicand`, `anytime-multicand`)
are implemented in `kga/routing.py` and exercised via `multicandidate_decide_kga.py` /
`kbtrain.sh theory-v2`.
