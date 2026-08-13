# FRONTIER EXPERIMENT VERDICT

**Adversarial review of `beta_sweep.py` / `decision_value.py` and the new paper sections.**
Reviewer premise: assume this is `frontier_validation.py` again — a result true by algebra, dressed
as a measurement — until proven otherwise.

Artifacts produced by this review, all in
`/home/claude/kb/experiments/kbound/frontier_sweep_v1/`:

| file | what it is |
|---|---|
| `adversarial_ablations.py` | end-to-end re-run under label shuffle / Z-noise / Z-shuffle / M-noise, plus circularity, overlap, near-duplicate and coverage-null probes |
| `adversarial_ablations_results.json` | its output (seed 777, 5 reps per perturbation) |
| `adversarial_stability_probes.py` | GBM random-state sensitivity; share-of-ceiling null |
| `adversarial_gbm_seed_stability.json`, `adversarial_share_of_ceiling_probe.json` | their outputs |

---

## Is it circular?

**No.** I tried four ways to break it and it did not break. The dataflow is clean and the headline
separations are far from every null I could construct.

### Dataflow trace (label firewall)

`fs_common.py` splits each record at load: `LABEL_KEYS = ("B", "a_oracle", "oracle_action",
"regime")` go into `Cell.labels`, a dict that is *not* a field any feature builder reads. The
design matrices are built only by `phi_doc` / `phi_atc4` / `phi_full`, all of which take `Z` and
nothing else (`fs_common.py:154-178`). I grepped every use of the four label names and of `a0`,
`a_adapted`, `b_hat`:

* `B` reaches an estimator **only as the regression target `y`** in `crossfit_M`
  (`fs_common.py:225`) and `holdout_M` (`fs_common.py:257`), and only on training folds. A cell
  never contributes its own label to its own `M`. This is the standard "the deployer has labelled
  dev data" assumption, and it is the same assumption the shipped `b_hat` already makes. It is
  disclosed in the paper, and the limitations paragraph tells a reader who rejects it to read the
  AUCs as upper bounds. That is the right call.
* `beta_hat` is `q_{0.90}(|Delta - M|)` on the *source-like dev* cells only (`beta_sweep.py:143`).
  Labels are used, on dev cells, which is the point of the exercise: the paper is testing whether a
  deployer *can* declare beta from data they have.
* The threshold `beta` is never chosen by a label-dependent objective. `matched_yield`
  (`beta_sweep.py:88`) picks beta to match the empirical rule's **yield**, and yield =
  `mean(|M| > beta)` — no labels. `beta_sound_min` *is* label-dependent, and the paper explicitly
  brands it "an oracle diagnostic" in the table caption, the body, and the claim ledger.
* In `decision_value.py`, every policy input passes `_assert_label_free`, a run-time tripwire that
  compares by identity, by base buffer, and by exact value against a registry of every label array
  loaded in the process. `--self-test` demonstrates it firing on `B`, a copy of `B`, a view of `B`,
  `a_oracle`, and `a_adapted`. The script itself documents the guard's limit (it does not catch
  arbitrary functions of a label). I re-read the EVAL-ONLY annotations by hand: no decision rule
  touches `B`, `a0`, `aa` or `oracle`.

**No leak found.** The one place a reviewer could object — `y = Delta` as the regression target —
is disclosed, is cross-fit, and is not the failure mode that killed `frontier_validation.py`.

### Ablations actually run (not asserted)

Primary CIFAR-10-C configuration, `loco` / `M_ATC4`, and the corresponding ImageNet-C rows:

| perturbation | AUC | beta_hat | yield | commit-err | coverage |
|---|---|---|---|---|---|
| **real** | **0.875** | 0.0360 | 0.472 | **0.031** | 0.756 |
| labels shuffled (n=5) | 0.505 | 0.2475 | 0.000 | — (no commits) | 0.901 |
| Z -> Gaussian noise (n=5) | 0.426 | 0.1366 | 0.194 | 0.256 | 0.569 |
| Z -> row-shuffled (n=5) | 0.426 | 0.1366 | 0.191 | 0.238 | 0.570 |
| M -> noise at real M's scale (n=5) | — | — | — | 0.278 | — |

Across all 12 (dataset x split x estimator) configurations, shuffling the labels drives the
out-of-fold AUC to **0.478–0.527** (mean over 5 draws each). `leakage_check.py`'s independent test
gives 0.474–0.501 over 10 permutations. The real AUCs are 0.75–0.97 on CIFAR-10-C. The result does
**not** survive label shuffling — which is exactly what a real result should do.

### The circularity probe the panel would run

`A5` asks the `frontier_validation.py` question directly: is `Z` a noisy copy of the target?

* **ImageNet-C: no.** No `Z` column exceeds |r| = 0.35 with `B`; in-sample R²(B ~ full Z) = 0.358.
* **CIFAR-10-C: not by construction, but very strong.** `pbal_drop` r = −0.930, `pre_pbal` −0.925,
  `marginal_KL` +0.907, `pre_entropy` +0.871; in-sample R²(B ~ full Z) = **0.986**. The mechanism is
  legitimate and label-free — pre-adaptation entropy/confidence is nearly a deterministic function
  of `a0` (r = −0.936), and on this grid `Delta` is largely a function of `a0`. But it means the
  CIFAR-10-C AUCs of 0.97 are a property of *this benchmark*, not a general claim about evidence
  channels. The paper already says this, in the strongest available form (DoC is anti-predictive on
  ImageNet-C at AUC 0.351).

**Verdict: not circular.** The `frontier_validation.py` defect was Z = M + tiny noise so that
eps -> 0.9*beta by algebra. Nothing of that kind is present here.

---

## What the experiment actually establishes

Stated conservatively, and only what the numbers carry:

1. **On these two grids, `beta` declared as `q_{0.90}(|Delta - M|)` on source-like dev cells does
   not bound the realized drift at deployment.** On CIFAR-10-C the declared budget is 1.4x–50x
   smaller than the smallest budget that would make the committed actions sound, in all ten
   configurations, and 24–73% of cells fall outside the declared class. This is a real measurement:
   I re-derived every cell of Table 3 from `beta_sweep_results.json` and all twenty rows match.
2. **Under this declaration procedure, the population rule does not reach the shipped conformal
   rule's operating point.** No beta > 0 matches it on both yield and regret in 15 of 18 held-out
   configurations. I recomputed this from the `sweep` field independently: 15/18 with beta > 0,
   13/18 including the degenerate beta = 0. The paper states 15/18 with the "beta > 0" qualifier and
   separately identifies the two beta = 0 matches. Accurate.
3. **The margin M carries genuine but benchmark-specific sign information.** AUC 0.75–0.97 on
   CIFAR-10-C, 0.35–0.94 on ImageNet-C, against a shuffled-label null of ~0.50 and a noise-Z null
   of 0.43 (CIFAR, `loco`) / 0.49–0.54 (ImageNet-C). The single most standard label-free statistic
   (difference of confidences) is genuinely anti-predictive on ImageNet-C — I confirmed this is not
   a fold artifact, because the ImageNet-C noise-Z control sits at 0.49–0.54, not at 0.35.
4. **The theorem's abstention band is not the coin flip the theorem describes.** In-band AUC of M
   against sign(Delta) is 0.728–0.927 across the reported configurations, and 36.5% of in-band cells
   have |Delta| > 0.02. Sign *balance* in-band does approach 50/50 on CIFAR-10-C (0.506–0.570),
   which the paper correctly reports as a point in the theorem's favour. Both directions are stated.
5. **Abstention is not free, and the certificate's commitments are not random.** The kappa=1 replay
   reproduces the shipped `kga_decision` on 7,365/7,365 cells. Always-abstain is numerically
   always-freeze. On CIFAR-10-C a committed decision is worth +0.0094 accuracy against a permutation
   null of −0.0185 (p < 0.0002 at 5,000 draws), and commitments land on cells whose true effect is
   24.5x that of the declined cells. All of these numbers are in the JSON exactly as printed.
6. **Reproducibility is real, not claimed.** I re-ran both scripts from scratch. Both result JSONs
   came back **byte-identical**. All three figures regenerate byte-identical from the JSONs, and the
   figure script contains no hard-coded results.

---

## What it does not establish

* **It does not test Theorem `thm:headline`.** `gamma := Delta - M` is a definition, so sufficiency
  is interval arithmetic and no experiment can refute it. The paper says this in the first paragraph
  of the section. What is being tested is whether the *antecedent* is declarable. That is a
  narrower, and honest, framing.
* **It does not show the frontier fails "in two directions."** The ImageNet-C zero-yield rows have
  the *same signature the null produces*: under a label shuffle, M collapses toward a constant,
  beta_hat -> `q_{0.90}(|Delta|)`, and yield goes to 0 in 8 of 12 configurations. Those rows are
  evidence that a **weak M** makes beta_hat useless — a statement about the estimator, not an
  independent second failure mode of the rule form. I had the paper narrowed accordingly.
* **The coverage column `Pr(|gamma| <= beta_hat)` has a null of 0.90, not 1.0.** beta_hat is a 0.90
  quantile, so coverage on dev cells is *exactly* 0.900 by `np.quantile`'s definition, and the
  label-shuffled re-run returns 0.890–0.902 on target cells. The ImageNet-C entries (0.896–0.985)
  sit **at or above** that null and carry essentially no information; only the CIFAR-10-C entries
  (0.274–0.756), which are far below it, are measurements. This is the same species of defect as
  `frontier_validation.py` — a quantile's definition mistaken for evidence — in a much milder form,
  and it was undisclosed. It is now disclosed in the table caption.
* **It does not establish anything at the corruption-family level.** Six families on CIFAR-10-C,
  three on ImageNet-C — and all three ImageNet-C "families" are noise variants (gaussian, impulse,
  shot), so leave-one-corruption-family-out there is a weak generalization test. The paper flags the
  family count; it does not flag that the ImageNet-C families are near-siblings.
* **The unit counts overstate independence.** Delta correlates at **0.913** between cells sharing a
  condition and method across seeds (CIFAR-10-C; 0.831 on ImageNet-C) and at **0.995** between the
  r0/r1 design replicates. Two consequences: (a) leave-one-*seed*-out puts near-duplicate rows in
  train and test, so `loso` is an optimistic split and should not be read as held-out; (b) the
  k = 432 condition-clustered interval used for the "one positive result" is not 432 independent
  units. The paper already discloses the r0/r1 correlation elsewhere and already reports that the
  positive result dies at k = 6 family clusters, so it does not lean on the inflated unit.
* **It does not establish that the certificate commits on the right cells via the "share of
  ceiling" statistic.** That ratio is ~1 for *any* estimator with the right marginal spread: I
  replaced `Delta_hat` by `Delta` + a reshuffled residual — pairing destroyed entirely — and the
  share is 1.05 (CIFAR-10-C) / 1.02 (ImageNet-C), *above* the observed 0.99 / 0.89. The statistic
  only discriminates in the extreme, which is where the paper uses it (ImageNet-C Tent, 0.04). The
  claim it was supporting is carried by the `eps / mean|Delta|` column and the permutation test
  instead. Now stated in the paper.
* **The GBM rows are not exact.** Over five estimator random-states, CIFAR-10-C `loco`/GBM gives
  beta_hat in [0.0172, 0.0217] and coverage in [0.579, 0.629]; ImageNet-C `loco`/GBM commit-error is
  0 on four seeds and 0.008 on the fifth. So the headline "3 configurations deliver zero commit
  error at nonzero yield" is really 2–3.

---

## Defects found

Severity: **H** invalidates a claim, **M** requires a stated caveat, **L** cosmetic.

| # | Severity | Defect | Status |
|---|---|---|---|
| 1 | **M** | Coverage column's null is 0.90 by `np.quantile`'s definition (label-shuffled control: 0.890–0.902). ImageNet-C entries sit at the null; presented alongside informative CIFAR entries without distinction. | **Fixed** — table caption now states the null, the shuffled-label control value, and which entries are informative. |
| 2 | **M** | Fit/score overlap: for `loco`/`loso`/`shipped`, the source-like dev cells that declare beta_hat (1,620/6,480 and 135/405) are also inside the scored target set. Direction is *conservative* — excluding them moves CIFAR `loco` coverage 0.452/0.756/0.607 -> 0.303/0.708/0.509, i.e. the published table understates the failure — but it was undisclosed. `srclike` is already disjoint. | **Fixed** — disclosed with numbers in the section's limitations. |
| 3 | **M** | ImageNet-C zero-yield rows are the label-shuffle null's own signature (weak M -> large beta_hat -> zero yield in 8/12 shuffled configs). Framed in the text as an independent second failure direction. | **Fixed** — narrowed to "a weak M makes beta_hat useless," not a second failure mode. |
| 4 | **M** | "Share of ceiling" ~1 is the null for any estimator with matching marginals (destroyed-pairing control: 1.05 / 1.02, above the observed values). Used to support "abstention is the radius, not conservatism." | **Fixed** — null stated, claim reattributed to `eps/mean|Delta|` and the permutation test. |
| 5 | **M** | Intro bullet: ImageNet-R commitments "statistically indistinguishable from a random subset" — true per backbone (p in [0.054, 1.000], 9/10 backbones have value-per-decision exactly 0), but the *pooled* JSON gives p = 0.0. Unqualified in the intro. | **Fixed** — qualified to per-backbone, with the pooling artifact named. |
| 6 | **M** | `kbound.tex` (long form): "Sweeping beta in [0,0.30] moves regret by 26x to 192x" reads as a global range; the true across-config range is 1.0x–192x (ImageNet-C configs span as little as 1.0x). Same for "yield varies by 21 to 52 points." The short form already attributes both correctly. | **Fixed** — both attributed to their specific configurations in `kbound.tex`. |
| 7 | **L** | Figure caption (c): "with a genuinely held-out margin (blue, green) the population rule is dominated." The green curve (`loco`/GBM) is *not* dominated at beta = 0 — it matches on both axes there. Body text handles this correctly; the caption did not. | **Fixed** — "dominated at every beta > 0," with the beta -> 0 endpoint named. |
| 8 | **L** | Table 3, CIFAR `loso`/GBM commit-error printed as 0.023; JSON gives 0.02247. | **Fixed** — 0.022. |
| 9 | **L** | GBM rows move with the estimator random-state (beta_hat +/-13%, coverage +/-2.5 pts, one "zero commit-error" row flips to 0.008). Undisclosed. | **Fixed** — stated, with the "3 configurations" count softened to 2–3. New probe script + JSON added. |
| 10 | **L** | LOCO chance level for AUC is 0.426 on CIFAR-10-C, not 0.50, because a per-fold constant predictor anti-correlates with the held-out fold mean. Does not affect any claim (CIFAR AUCs are 0.75–0.97), and the ImageNet-C control at 0.49–0.54 confirms the DoC anti-predictivity claim is real. | **Fixed** — stated as a caveat, with the ImageNet-C control that rescues the DoC claim. |
| 11 | **L** | ImageNet-C's three "corruption families" are all noise variants, so `loco` there is a weak generalization test. | **Not fixed** — flagged here; the family *count* is already disclosed in the paper. A one-clause addition would close it. |
| 12 | **L** | `loso` is materially leaky (r = 0.913 between same-condition cells across seeds) yet appears in Table 3 next to `loco` and `srclike` without a leakage marker. It is not the primary split and the paper does not lean on it. | **Not fixed** — flagged here. |

Nothing rated **H**. No finding invalidates the negative result; findings 1–4 all cut in the
direction of the paper's own conclusion or shift emphasis rather than reversing it.

---

## What this is worth

**My read: this moves the project from ~6 to ~6.8, and it is the most valuable single unit of work
in the repository's recent history — but it is worth less than the page count suggests, and for an
instructive reason.**

Justification, against the panel's earlier 4.2 and the ~6 post-cleanup estimate:

* **The 4.2 -> 6 move was hygiene.** Fixed queue items, corrected provenance, retired a fabricated
  claim. It removed reasons to reject; it added no reasons to accept.
* **This work adds a reason to accept, and it is a rare one.** The panel's two live objections were
  (a) the theory never touches the experiments and (b) the one experiment that did was circular by
  construction. Both are now closed with real artifacts on 6,885 real cells, and closed *against*
  the authors' interest. A paper that runs the decisive test of its own headline theorem, gets a
  negative answer, and withdraws the operational reading in the abstract, the contributions list,
  the claim ledger and the limitations section is doing something most submissions do not do. That
  is worth roughly +0.6 on its own.
* **The engineering is genuinely above the bar.** Bit-identical re-runs of two multi-hundred-KB
  result JSONs and three PNGs. A run-time label tripwire that self-tests, and that documents its own
  limitation rather than overselling itself. Four estimator variants, four splits, cluster
  bootstraps at three granularities. I went looking for a soft spot for two hours and found ten
  disclosure-level defects and zero invalidating ones. That is a good ratio.
* **Why not more than ~6.8.** Three reasons, and the third is the real one.
  1. *Scope.* Two benchmarks, 6 and 3 corruption families, one of which is three shades of noise.
     The effective independent-unit count is far below 6,885 (r = 0.995 between design replicates).
     A referee will price this as two datasets, not as a survey.
  2. *The measurement is partly about the estimator, not the rule.* Half the reported failure —
     every ImageNet-C zero-yield row — is what a weak M produces, and I showed the label-shuffle
     null produces the same signature. That halves the ImageNet-C evidence.
  3. *A negative result about your own parameter raises the question of what is left.* The paper
     now says: the proofs stand, but sufficiency is arithmetic on a definition, necessity rests on
     the matched-evidence construction, and the operational rule is withdrawn. What remains as a
     *contribution* is Theorem `lem:nonid` and its audit corollary — genuinely non-trivial — plus a
     conformal wrapper the paper itself declines to claim as novel machinery, plus a careful
     nine-track empirical accounting with the negatives kept. That is a solid, honest workshop-to-
     mid-tier paper. It is not a field-shaping one, and this experiment, by being honest, has made
     that clearer rather than less clear. The score goes up because the work is now *trustworthy*;
     it does not go up further because trustworthiness revealed a smaller contribution than the
     earlier framing implied.
* **What would move it further.** Not more benchmarks of the same kind. The single highest-value
  next step is to make the *necessity* construction bite empirically: exhibit, on real data, two
  deployment cells with statistically indistinguishable label-free evidence and opposite benefit
  signs. That converts `lem:nonid` from a theoretical construction into a measured phenomenon and
  would be the first result in this project that is both novel and positively demonstrated. It is
  also runnable on the artifacts already in this tree — the ingredients are the 11-dim Z, the 6,885
  cells, and a matched-evidence nearest-neighbour search. I would rate that at +0.5 to +1.0 if it
  succeeds, and it is publishable if it fails.
