# The β-declaration impossibility: memo

**Deliverable:** `/home/claude/kb_fixes/theory_beta_impossible.tex` (LaTeX, `\input`-able, compiles
clean — 8 pp, 0 undefined refs, under the paper's amsthm environments with stub labels).
**Numerical artifact:** `/home/claude/kb_fixes/beta_impossible/check_anchor_collapse.py` and
`anchor_collapse_check.json` (computed in this run, reproducible, no network).

---

## 0. One-paragraph verdict

`thm:short-audA` is true and its proof is sound, but it is stated at the weakest point of a much
sharper result. The right statement is not *"label-free audits of β are vacuous over the unrestricted
class"* — it is:

> **For every declared class, the exact minimax label-free drift budget is the fibre radius
> `Γ_z(C) = sup{|γ(P)| : P ∈ C, P matches the observables}`, attained by a constant that reads none
> of its data. For the declared deployment class `C_beta` the fibre radius equals `beta` exactly.
> So the best possible label-free audit returns its own input, and the number it returns is
> simultaneously the radius of the band on which `thm:headline` must abstain.**

Three consequences, all proved:

1. **Audits are decision-inert.** No label-free audit ever converts an abstention into a commitment
   relative to declaring `beta = Γ_z`. (Thm 1(c).)
2. **Yield ≤ δ.** Any δ-valid label-free audited rule commits with probability ≤ δ whenever
   `|M| ≤ Γ_z`. Over the unrestricted class `Γ_z = 1/2 + |M| > |M|` always, so **decision yield is
   bounded by the false-commitment budget, unconditionally, at every batch size.** (Thm 1(d).)
   This is a one-line derivation of the yield/safety frontier the paper measures.
3. **Labels at the wrong domain buy nothing.** A *fully labeled* calibration sample of any size
   leaves `Γ_z` unchanged, absent a declared coupling. (Thm 2.) This is strictly stronger than
   Aud-A and it is the theorem that actually bears on the β-sweep, which used labeled dev cells.

The escape is a matched pair — a labeled anchor **and** a declared invariance/transfer constraint —
with an unimprovable `n_lab^{-1/2}` floor (Thm 5, genuine Le Cam; matched from above by Aud-B).

---

## 1. What is in the .tex, item by item

| result | status | what it is |
|---|---|---|
| `lem:fibre` — kernel freedom | **PROVED** | The one construction. Binary labels on `D` ⇒ `{f_0,f_a}={0,1}` ⇒ any `η: D→[0,1]` is a valid kernel; all label-free data is invariant; `γ` sweeps `[-1/2-M, 1/2-M]` exactly. |
| `thm:beta-minimax` (a)–(d) | **PROVED** | Floor `Pr(β̂ ≥ Γ_z) ≥ 1−δ`; attainment by the constant; decision inertness; **yield ≤ δ**. |
| `cor:audA` | **PROVED** | Aud-A recovered as the case `C = all`, `Γ_z = 1/2+|M|`, plus the new yield bound. |
| `cor:beta-is-beta` | **PROVED** | `Γ_z(C_beta) = beta` for `beta ≤ 1/2`; and `|M| > Γ_z ⟺ thm:headline (iv)`. **The equivalence the brief asked for, as an equality of numbers.** |
| `thm:anchor` | **PROVED** | Labeled calibration data of any size leaves `Γ_z = 1/2+|M|` if no coupling is declared. |
| `prop:threeterm` | **PROVED** | `\|γ_T\| ≤ \|γ_cal\| + Δ_F(μ_cal, μ_T) + ρ`. Exactly one of the three terms is label-free computable. Every one of Aud-B/C/F/G/H is an instance. |
| `thm:lecam` | **PROVED-MODULO-STANDARD-RESULT** | `κ ≥ min{τ(U), 1/4, 1/(2√(2n))}`. Uses Pinsker and `KL ≤ χ²` for Bernoulli, both conditions checked in the proof. |
| `thm:dichotomy` | **PROVED** | (N) no rate / (P) `Θ(n_lab^{-1/2})` above an irreducible declared `ρ` / (D) degenerate declaration. Exhaustive and exclusive. |

### The equivalence claim — it holds, and here is the exact form

The brief hoped that *"the same construction that forces abstention also forbids estimating the
budget that decides abstention."* It does, and it is more than a shared construction:

- `lem:nonid` is `lem:fibre` restricted to the two-point subfamily `η ≡ 1/2 ± δ`.
- `thm:aud-A` is `lem:fibre` at the two extreme points `η ≡ 1` and `η ≡ 0`.
- They are the same one-parameter family `{η ≡ c}_{c∈[0,1]}` read at two different radii.
- **And by `cor:beta-is-beta` the two radii coincide when the class is `C_beta`:** the abstention
  radius and the least label-free-auditable budget are the *same number*, `beta`.

One correction to the brief's framing, which matters for the paper's honesty: **this is not a Le Cam
two-point argument and should not be presented as one.** Within a fibre the observable laws coincide
*exactly* (TV = 0), which is the degenerate limit; there is no distance/error trade-off to state.
`theory_core_main.tex` already says this for `lem:nonid` and it is right to. Le Cam appears
genuinely only once, in `thm:lecam`, where labels make the two worlds distinguishable and the
`√(2n)·t` bound is doing real work.

---

## 2. Is the impossibility vacuous? — the self-check the brief demanded

Three ways it could be vacuous, checked:

**(a) "It forbids only estimators nobody would use."** *Not vacuous, and the tightness is
demonstrable inside this paper.* Every positive audit in `auditable_budgets.tex` — Aud-B, Aud-C,
Aud-F, Aud-G, Aud-H — purchases labels: Aud-B is explicitly labeled; Aud-C, Aud-F, Aud-H all carry
`|γ̂_cal| + t(n_lab, ·)`; Aud-G's (G2) requires "each calibration domain carries a labeled sample."
Aud-D/E is a composition, not a source. So the negative and positive results **meet exactly on the
labels/no-labels axis with no gap.** That is the non-vacuity certificate.

**(b) "The class is empty."** *No.* `lem:fibre` constructs the laws explicitly and checks
admissibility (Bernoulli validity on `D`, `μ_T(D) > 0`, measurability). The extreme worlds
`η ≡ 0, 1` are genuine target laws satisfying `ass:deploy`.

**(c) "It's a tautology."** *Partly yes, and the .tex says so in `rem:honest-scope`.* The mechanism
is "you cannot estimate an unidentified parameter." What is not tautological: the identification of
the exact minimax **value** `Γ_z` for every class (not an impossibility for one class); the
numerical coincidence `Γ_z(C_beta) = beta`; the yield ≤ δ exchange rate; and `thm:anchor`, which
is the statement about labeled-elsewhere data and is what the β-sweep actually violated.

---

## 3. Correspondence with the empirical record

The β-sweep is **not** a label-free audit — it derives `β̂ = q₉₀(|γ|)` from *labeled* source-like dev
cells. `cor:audA` therefore does **not** apply to it, and I do not claim it as a confirmation.
What applies is `thm:anchor` + `prop:threeterm`: the procedure supplies term 1 and sets terms 2 and 3
to zero via an *undeclared* dev↔deployment exchangeability assumption.

### 3.1 The 0.90 null — handled, not read into

`β̂` is a 0.90 quantile on dev cells, so target coverage has a **by-construction null of 0.90** when
the pools are exchangeable (recorded: `coverage_on_dev_cells_BY_CONSTRUCTION` = 0.900 CIFAR,
0.8963 ImageNet).

| pool | measured target coverage | reading |
|---|---|---|
| CIFAR-10-C (6 rows) | **0.274 – 0.756** | far below null → **a measurement**; exchangeability falsified |
| ImageNet-C `loco` (3 rows) | 0.904 – 0.956 | at/above null → **no information**; and `dev_and_target_disjoint = false` |
| ImageNet-C `srclike` M_doc/M_atc4 | 0.956, 0.985 | at/above null → **no information** |
| ImageNet-C `srclike` M_gbm | **0.470** | below null → **a measurement**, and it is the row with commit-error **0.532** on 62 cells (sign match 0.468, CI [0.340, 0.599]) |

The same null contaminates a second column I built: my dev-max vs target-max `|γ|` ratios are exactly
1.00 on ImageNet `loco`/`loso` **because dev ⊆ target there**, so those entries are forced and carry
nothing. Only the CIFAR rows (2.4×–28.4×) and the disjoint ImageNet `srclike` rows are measurements.

### 3.2 The prediction I derived from the theorem and then computed

`thm:anchor` says an uninformative anchor leaves only a class-scale constant. Under the *global*
label permutation in `adversarial_ablations.py`, any estimator fit on the permuted benefit column
converges to `E[Δ]`, so the declaration procedure must return ≈ `q₉₀(|Δ − E Δ|)` — computable from
the raw artifacts with **no model fitting at all**. I computed that number today, then compared:

| pool | **predicted** `q₉₀(\|Δ−EΔ\|)` | measured shuffled `β̂` (6 channels) | ratio | real `β̂` | inflation |
|---|---|---|---|---|---|
| CIFAR-10-C (n=6480) | **0.2376** | 0.2468 – 0.2606 | **1.04 – 1.10** | 0.0178 – 0.0482 | 5.2× – 14.1× |
| ImageNet-C (n=405) | **0.0689** | 0.0742 – 0.1171 | 1.08 – 1.70 | — | see below |

CIFAR-10-C matches to within 4–10% across all six channels, with **zero commitments in 5/5
replicates** for `M_doc` and `M_atc4`. This is a real out-of-sample prediction: the ablation was run
before this theorem existed, and nothing in the prediction was tuned.

### 3.3 A prediction that fails — reported, not buried

On ImageNet-C the *anchored* budget for `M_doc` / `M_atc4` is **larger** than the anchor-free
constant: real 0.199–0.239 vs shuffled 0.074–0.090 (ratio 0.31–0.42). Destroying the labels made the
budget *smaller*.

Nothing in the theory forbids this — `thm:beta-minimax` lower-bounds **valid** audits and is silent
on how far an *invalid* heuristic may overshoot — but I will not pretend it is predicted. The
mechanism is visible in the same artifact: difference-of-confidences is anti-predictive on ImageNet-C
(AUC 0.351), so its residuals exceed the marginal spread of the benefit. The operational reading:
**on that benchmark the anchored declaration is beaten by the trivial constant**, and those are
exactly the rows with decision yield 0.000.

### 3.4 What the theorem does *not* explain

The CIFAR-10-C rows with yield 0.47–0.75 at commit error 3–17% do **not** violate `thm:beta-minimax`
(d) (yield ≤ δ = 0.10), because they are not label-free audits and are not valid. The theorem's yield
bound applies to the label-free regime only. Saying otherwise would be a fake confirmation.

---

## 4. What a referee will attack

**A1. "Theorem 1 is a tautology: you can't estimate an unidentified parameter."**
Conceded on mechanism; `rem:honest-scope` in the .tex says so in the paper's own voice. The defence
is the four items in §2(c). If a referee still rejects it, the fallback claim that survives is
`cor:beta-is-beta` + `thm:beta-minimax`(d), which are statements about the *K-Bound decision rule*,
not about estimation, and are not standard.

**A2. "The unrestricted class is a straw man."**
Fully conceded, and it is why Theorem 1 is stated for an *arbitrary* declared class with the value
`Γ_z(C)`; Aud-A is one endpoint. The interesting content is at the other end: `Γ_z(C_beta) = beta`.

**A3. "You assume `s` is not target-calibrated."** — **The single most load-bearing hypothesis.**
`ass:deploy` declares `s` *source*-calibrated, which places no constraint on `P_T`. If a deployment
could declare target-calibration of `s` on `D`, then `γ ≡ 0`, `beta = 0`, and the frontier collapses
to the plug-in `sign(M)` rule the paper declines to call a guarantee. Preempted in the .tex: that is
the strongest possible declaration and is *exactly as unverifiable* label-free, by `lem:fibre`
applied to `u`. A referee who presses here is pressing on `ass:deploy`, not on the new theorems.

**A4. "Adaptive / sequential / randomized procedures escape."**
No. `def:audit-data` covers any measurable, possibly randomized `g(W, U)` with `U` independent, and
`Law(W)` is fibre-constant including under adaptive collection of *unlabeled* data, because the
entire unlabeled generating law is untouched by the construction. Stated in the .tex.

**A5. "Episodic deployment / delayed feedback escapes."**
That is deployment labels arriving late; it lands in case (P) of the dichotomy at rate
`n_lab^{-1/2}` with `n_lab` = accumulated feedback, and the false-adapt cost is paid before the
feedback arrives. Not an escape from the theorem, a purchase within it.

**A6. "Check the Le Cam constants."**
Explicit, non-asymptotic: `χ²(Ber(1/2+t) ‖ Ber(1/2)) = 4t²`, `KL ≤ χ²`, Pinsker,
`TV ≤ t√(2n)`, so `TV < 1/2` for `t < 1/(2√(2n))`. Mutual absolute continuity holds for `t < 1/2`.
Validity level `δ ≤ 1/6` and usefulness level `2/3` are the only free choices and are stated.

**A7. "Your dichotomy leaves a rate gap."**
Conceded explicitly in `rem:notsharp`: the `O(√(v/n))` transfer-complexity term of Aud-H has **no**
matching lower bound here. That is `conj:aud-maximal`, still open. The dichotomy closes the
labels/no-labels axis and the `n_lab` rate; it does not close the transfer axis.

**A8. "The numerical check is post-hoc."**
The ablation predates the theorem; the predicted quantity `q₉₀(|Δ−EΔ|)` was derived from `thm:anchor`
and computed in this run with a saved script. Ratios 1.04–1.10 on the 6480-cell benchmark. The
weaker ImageNet-C agreement (1.08–1.70, n=405) and the §3.3 sign reversal are both reported.

**A9. "γ is defined as a residual, so this is all bookkeeping."**
Yes — and `lem:fibre` is precisely the statement that *the residual is completely unconstrained by
the observables*, which is a fact about binary labels on the disagreement region, not bookkeeping.
That fact is what makes `beta` undeclarable-from-data; the bookkeeping is what makes it the only
undeclarable quantity.

**A10. "Uniform validity is too strong a requirement."**
Stated in `rem:notsharp`: nothing here forbids an audit valid *at the realised `P_T` only*. But
validity at the realised law is unverifiable and un-declarable, so it is not a specification. This
is the same objection as A3 in different clothing.

---

## 5. Recommended use in the restructure

1. **Promote `lem:fibre` to the main text.** One lemma, four lines, and both `lem:nonid` and the
   budget impossibility are corollaries. This is the structural saving the reframing was looking for.
2. **Lead with `cor:beta-is-beta`.** "The best label-free audit of the drift budget returns the
   budget you declared, and that number is the radius of the band on which the frontier abstains."
3. **Make `thm:beta-minimax`(d) the bridge to the experiments.** *Yield ≤ δ* is the theoretical
   statement of exactly the frontier `DECISION_VALUE_FINDINGS.md` measures.
4. **Replace, in the experiment discussion, "the frontier does not operationalize" with
   `thm:anchor`.** The β-sweep failed because it supplied term 1 of `prop:threeterm` and declared
   terms 2 and 3 to be zero. The coverage column measures that declaration failing on CIFAR-10-C and
   carries no information on ImageNet-C. That is the theorem explaining the experiment.
5. **Do not claim the β-sweep confirms Aud-A.** It used labels. Claiming otherwise would be the
   `frontier_validation.py` failure mode again.
