# K-Bound — theory audit (senior reviewer pass, item 3)

*Written as a senior theory/stats reviewer would, on the corrected short paper. This is an internal
red-team, not a substitute for the human professor sign-off you still want before a main-track
submission — but it surfaces the load-bearing assumptions and the exact places a referee will push,
with concrete fixes. Verdict first, then the four points you asked me to check, then the attack
surface.*

---

## Verdict

The corrected theory is **sound within its stated scope**. After the recent edits (γ vs ε split,
FA_u vs FA_c, "provisional candidate before commit," Camelyon reclassified), the three errors a
referee would have caught on a first pass are gone. What remains is **statement-level precision, not
broken math** — three assumptions are load-bearing and must be named as such, and one definition
(risk-alignment) is currently near-circular and should be split. The false-adapt guarantee is exactly
as strong as the weakest of: **(R1)** interval coverage, **(R2)** evidence identifiability
(risk-alignment), **(R3)** calibration⇄deployment exchangeability. None is exotic; each is where the
result fails if it fails.

---

## 1. The frontier assumptions — `sign Δ = sign(M + γ)`, identifiable iff `|M| > |γ|`

**What is actually assumed.** A fixed, computable disagreement region `D = {f0 ≠ fa}`; a *scalar*
reduction of the benefit on `D` to an observable margin `M` (source-calibrated score vs ½) plus an
unobservable calibration drift `γ`; and that the sign of the benefit is governed by that single
scalar `M + γ`.

**Where a referee pushes:**
- **`γ` is unobservable**, so the frontier is a *characterization*, not an estimator. The paper now
  says this in the method; the **theorem statement itself should carry the caveat in-line**, so a
  reader can't mistake `|M| > |γ|` for something you can evaluate at deploy.
- **The scalar reduction is a regularity assumption, not a free lunch.** That `D`'s benefit collapses
  to one margin is exactly the aligned/MLR/log-concave-location condition proved in
  `knowability_capacity.tex`. For a general `D` it can fail (non-monotone flip locus). **Cite that
  condition from the theorem**; don't let the frontier read as fully general.
- **Boundary `|M| = |γ|`** is measure-zero — say so explicitly rather than leaving it implicit.

**Fix:** one sentence in the theorem: "under the location/alignment regularity of App. X, and noting
`γ` is a population quantity not observed at deploy."

---

## 2. Risk-alignment — the definition is near-circular as written

Current: *"`Z` is risk-aligned when its induced benefit estimate admits a calibrated interval that
validly brackets `Δ`."* A sharp referee will object: this says **"the method works when the method's
interval is valid,"** which bundles the *property of the evidence* with the *property of the
calibrator* and edges toward assuming the conclusion.

**Fix (this is the single most important remaining theory edit — wording, not new math):** split it.
- **Identifiability (a property of `Z` and the class):** `sign Δ` is a measurable function of the
  population law of `Z` on the deployment class — equivalently, no two admissible worlds with opposite
  benefit sign share a `Z`-law. *This* is risk-alignment.
- **Coverage (a property of the calibrator):** the conformal interval has marginal coverage `≥ 1−α`.

Then the certificate theorem reads cleanly: **identifiability + coverage ⇒ FA_u ≤ α**, and the abstain
action is precisely "identifiability cannot be certified here" (which Corollary 1 now states
correctly). De-circularizing this also makes the impossibility lemma and the abstain rule line up:
non-identifiable ⇔ not risk-aligned ⇔ must abstain.

---

## 3. Is the false-adapt theorem stated at the right probability level?

**Yes — now — but the theorem must say which level.** The rule commits adapt iff `Δ̂ − ε > 0`. On the
coverage event `{|Δ̂ − Δ| ≤ ε}` (prob `≥ 1−α`), adapt ⇒ `Δ > 0`. Therefore:

- **`P(commit adapt AND Δ ≤ 0) ≤ P(coverage fails) ≤ α`.** This is the *unconditional* `FA_u`, and it
  is the correct, defensible claim. The edit now reports `FA_u ≤ α`; keep it **in the theorem**, not
  only in the metrics paragraph.
- **`FA_c = P(Δ ≤ 0 | adapt)` is NOT bounded by α** in general (it scales with the adapt base rate).
  The tables list it empirically — fine — but the theorem must **not** claim α-control of `FA_c`.
- **"Marginal," not "conditional," coverage.** Split-conformal gives *marginal* coverage over the
  calibration draw, so the honest claim is **"marginal unconditional false-adapt ≤ α,"** not
  "for every deployment cell." A measure-theory referee will insist on the word *marginal*. Add it.

---

## 4. Population drift `γ` vs empirical radius `ε`

The corrected text separates them — necessary and correct. Remaining honest points:

- `ε` is the `(1−α)` quantile of `|Δ̂ − Δ|` calibration residuals: a **finite-sample,
  exchangeability-dependent surrogate** for the population budget the frontier expresses through `γ`.
  They coincide only asymptotically and only if the calibration residual law equals the deployment
  residual law **(R3)**.
- **The conformal guarantee covers `ε`, not `Δ̂`.** Split-conformal is valid for *any fixed predictor*
  under exchangeability — so the α-bound survives a biased GBR `Δ̂` — but it does **not** certify that
  `Δ̂` is a good estimator. State plainly that the guarantee is on the *radius/decision*, not on the
  benefit model's accuracy.
- **The exchangeability is at the granularity you calibrate.** If `Δ̂`/`ε` are fit leave-one-task-out
  (task = corruption), the assumption is **task-level exchangeability**, which is *stronger* than
  i.i.d. cells and is violated under genuine task shift (a held-out corruption *family*). Be explicit:
  the bound is conditional on calibration⇄deployment exchangeability at the locked-protocol
  granularity, and degrades under task shift. (This is also why the natural-shift wins are framed
  per-protocol, not as one universal gate — consistent with the cross-protocol-aggregate wording.)

---

## The attack surface — what a referee will actually try

1. **"Your impossibility is the easy half."** Correct, and the paper now treats it as the hook with
   the certificate as the contribution. Good — but make sure the page budget reflects that (proof
   effort on the constructive side).
2. **"Is the non-identifiability witness non-vacuous?"** The 1-D Gaussian two-point construction is
   explicit and validated (40k draws, 0 mismatches in `results_knowability_capacity.json`). Solid —
   cite it from the lemma so the referee sees it's a real witness, not a hypothetical.
3. **"Is the benefit model doing the work, or the conformal radius?"** This is now *answered* by the
   gate-baseline table (`gate_baseline_comparison.py`): `KGA (no radius)` vs `KGA (certificate)`
   isolates the radius — in the self-test the radius drops `FA_u` from 0.073 to 0.007. Put that row
   in the paper; it directly rebuts the objection.
4. **"Risk-alignment is unfalsifiable."** Addressed by the §2 split + the abstain action.
5. **"`FA_u ≤ α` is marginal, not conditional."** Concede explicitly (point 3); do not over-claim.

---

## Bottom line + what to hand the human professor

The theory is **honest and defensible at workshop / short-paper level** once you (a) add the
*marginal* qualifier to the false-adapt theorem, (b) split risk-alignment into identifiability +
coverage (§2), and (c) state the task-level exchangeability assumption for `ε`. **None of these is a
hole in the math — all are precision of statement.**

For a main-track submission, ask a stats/theory professor to specifically sign off on the two places
an expert is most likely to find the assumption stronger than written:

1. **The split-conformal marginal-coverage claim under your leave-one-task-out calibration** — is
   task-level exchangeability defensible for your corruption splits, or should the guarantee be stated
   over i.i.d. cells within a corruption only?
2. **The risk-alignment ⇔ identifiability reformulation** — confirm the impossibility lemma, the
   abstain rule, and this definition are the same condition viewed three ways.

Those two are the load-bearing checks. Everything else in the corrected paper is, in my read,
honest and ready.

---
*Audit prepared as an internal red-team. The gate-baseline evidence referenced in point 3 is
reproducible via `scripts/gate_baseline_comparison.py` (`--selftest` for the synthetic check; `--in`
the runner's per-cell dump for the real table).*
